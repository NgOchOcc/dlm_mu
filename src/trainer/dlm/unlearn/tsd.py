"""Trajectory-Selective Distillation (TSD) for masked diffusion LM unlearning.

Implements ``docs/tsd_iclr.tex``. TSD is retain-set-free: it edits the frozen
teacher distribution only at answer-bearing masked positions (demote the gold
token and renormalise) and self-distils the frozen teacher at every remaining
masked position, over answer-stratified partially-masked answer states. One
forward KL objective fits the student to this selectively edited teacher.

The atomic unit is a denoising decision -- a token prediction under a particular
mask configuration -- so the forget signal is applied across partial answer
states, not only the fully masked answer.

Mapping to TOFU. TOFU provides no sub-answer span annotation, so the
answer-bearing set ``A`` defaults to the whole supervised response
(``labels != -100``). With ``A`` = response there are no non-answer response
positions, so the self-distillation term (Eq. retain-target) is vacuous and the
objective reduces to selective *pathwise* demotion with per-token distributional
preservation (only the gold token is removed at each position; the rest of the
teacher's distribution is the target). The self-distillation and balanced-weight
machinery is still implemented in full so the ablation carries over unchanged to
any setting that does supply ``A`` as a strict subset.

Ablation switches (docs/tsd_iclr.tex, Sec. Minimal Experimental Ablation):
  * ``mask_states``  -- ``"pathwise"`` (answer-stratified k~U{1..|A|}) vs
    ``"single"`` (only the fully masked answer).
  * ``demotion``     -- ``"minimal"`` (B = {y_i}) vs ``"belief"`` (B = teacher
    top-K union {y_i}).
  * ``self_distill`` -- add KL(p0 || p_theta) on non-answer masked positions.
"""

import torch

from trainer.dlm.unlearn.grad_diff import DiffusionGradDiff


class DiffusionTSD(DiffusionGradDiff):
    def __init__(
        self,
        mask_states="pathwise",
        demotion="minimal",
        belief_top_k=5,
        self_distill=True,
        balanced_weight=True,
        alpha=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(alpha=alpha, *args, **kwargs)
        if mask_states not in ("pathwise", "single"):
            raise ValueError(f"Unknown mask_states: {mask_states}")
        if demotion not in ("minimal", "belief"):
            raise ValueError(f"Unknown demotion: {demotion}")
        if demotion == "belief" and belief_top_k < 1:
            raise ValueError("belief_top_k must be >= 1 for belief demotion")
        self.mask_states = mask_states
        self.demotion = demotion
        self.belief_top_k = belief_top_k
        self.self_distill = self_distill
        self.balanced_weight = balanced_weight
        self.mask_id = self.processing_class.mask_token_id
        self.vocab_size = len(self.processing_class)
        # Belief demotion must leave at least one surviving token, else the edited
        # target renormalises over an empty support (log-sum-exp of -inf = -inf).
        if demotion == "belief" and belief_top_k + 1 >= self.vocab_size:
            raise ValueError(
                f"belief_top_k={belief_top_k} leaves no surviving vocabulary "
                f"(vocab_size={self.vocab_size})"
            )
        self.ref_model = self._prepare_ref_model(self.model)
        # The teacher is a frozen snapshot: forwarded under no_grad below, but also
        # detach its parameters so an accidental grad path cannot train it.
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self._pending_metrics = {}

    # -- answer-stratified mask sampling (docs/tsd_iclr.tex Sec. 3.1) ----------
    def _answer_stratified_mask(self, answer_mask):
        """A mask set ``M`` with ``M cap A != empty`` (Q_A(.|A)).

        ``pathwise`` draws ``k ~ Uniform{1..|A|}`` and masks ``k`` uniformly
        chosen answer positions, covering every level of partial answer
        visibility (``k = |A|`` is the fully masked answer). ``single`` always
masks the entire answer span. With ``A`` = response there are no
        non-answer response positions to corrupt, so ``M`` lives entirely in
        ``A``.
        """
        if self.mask_states == "single":
            return answer_mask.clone()

        b, l = answer_mask.shape
        count = answer_mask.sum(-1)  # |A| per example
        # Rank answer positions by a random score; keep the top-k as masked.
        scores = torch.rand(b, l, device=answer_mask.device).masked_fill(
            ~answer_mask, -1.0
        )
        order = scores.argsort(dim=-1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(
            -1, order, torch.arange(l, device=answer_mask.device).expand(b, l)
        )
        u = torch.rand(b, device=answer_mask.device)
        k = (u * count.clamp_min(1).float()).floor().long() + 1  # in [1, |A|]
        k = torch.minimum(k, count.clamp_min(1))
        return (rank < k.unsqueeze(-1)) & answer_mask

    def _non_answer_mask(self, response_mask, answer_mask):
        """Ordinary forward-diffusion corruption on the non-answer response
        positions ``R \\ A`` (docs/tsd_iclr.tex Sec. 3.1) -- the states the
        self-distillation term consumes.

        Returns all-False when ``A == R`` (the TOFU mapping), so the
        self-distillation branch is vacuous there *by construction* -- the
        intended reduction -- rather than by a masking bug that confined every
        sampled position to ``A``. Uses the same schedule as the base trainer's
        forward corruption (base.py ``_sample_and_denoise``) so non-answer states
        are drawn from the model's own noising process.
        """
        non_answer = response_mask & ~answer_mask
        if not bool(non_answer.any()):
            return torch.zeros_like(response_mask)
        b, l = response_mask.shape
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=response_mask.device
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)
        keep = torch.rand((b, l), device=response_mask.device) < p_mask
        return keep & non_answer

    def _resolve_answer_mask(self, forget_inputs, response_mask):
        """``A subset R``. Uses an explicit ``answer_mask`` span when the collator
        supplies one; otherwise ``A`` defaults to the whole response ``R`` -- the
        TOFU mapping, since TOFU carries no sub-answer annotation.

        A provided span must already be aligned with the *preprocessed* sequence:
        ``right_shift_logits`` prepends BOS and shifts every position by one, so an
        unshifted span would silently misalign with ``input_ids``. Rather than
        guess the alignment, refuse a shape mismatch loudly.
        """
        provided = forget_inputs.get("answer_mask", None)
        if provided is None:
            return response_mask
        provided = provided.to(response_mask.device).bool()
        if provided.shape != response_mask.shape:
            raise ValueError(
                "answer_mask must be pre-aligned with the preprocessed inputs "
                f"(got {tuple(provided.shape)}, expected {tuple(response_mask.shape)}); "
                "align it with right_shift_logits before passing it in."
            )
        return provided & response_mask

    # -- selective teacher editing (docs/tsd_iclr.tex Sec. 3.2) ---------------
    def _demote_mask(self, log_p0, gold):
        """Boolean ``B_{M,i}`` of vocabulary tokens to demote at each position.

        ``minimal`` demotes only the gold token ``y_i``; ``belief`` additionally
        demotes the teacher's top-K, testing whether broader belief suppression
        curbs paraphrase leakage.
        """
        demote = torch.zeros_like(log_p0, dtype=torch.bool)
        demote.scatter_(-1, gold.unsqueeze(-1), True)
        if self.demotion == "belief":
            topk = log_p0.topk(min(self.belief_top_k, log_p0.shape[-1]), dim=-1).indices
            demote.scatter_(-1, topk, True)
        return demote

    def _tsd_loss(self, model, forget_inputs):
        prep = self._preprocess_inputs(
            {
                "input_ids": forget_inputs["input_ids"],
                "labels": forget_inputs["labels"],
                "attention_mask": forget_inputs.get("attention_mask", None),
            }
        )
        input_ids, labels = prep["input_ids"], prep["labels"]
        attention_mask = prep.get("attention_mask", None)

        # R = supervised response positions; A subset R = answer-bearing positions.
        # On TOFU there is no sub-answer span so A defaults to R, but R and A are
        # kept distinct so a dataset supplying a strict A drives the self-distill
        # branch through non-answer states (M_non), which was previously unreachable.
        response_mask = labels != -100
        answer_mask = self._resolve_answer_mask(forget_inputs, response_mask)

        # M = M_ans union M_non: a nonempty answer-stratified draw over A, plus
        # ordinary forward-diffusion corruption on R\A when self-distilling. With
        # A == R (TOFU) M_non is empty and M == M_ans, reproducing the reduced form.
        m_ans = self._answer_stratified_mask(answer_mask)
        m_non = (
            self._non_answer_mask(response_mask, answer_mask)
            if self.self_distill
            else torch.zeros_like(response_mask)
        )
        masked = m_ans | m_non
        y_m = torch.where(masked, self.mask_id, input_ids)  # y_M

        outputs = self._postprocess_outputs(
            model(input_ids=y_m, attention_mask=attention_mask)
        )
        log_p = self._log_probs(outputs.logits)  # student log p_theta(.|x,y_M,M)
        with torch.no_grad():
            teacher = self.ref_model(input_ids=y_m, attention_mask=attention_mask)
            log_p0 = self._log_probs(self._postprocess_outputs(teacher).logits)

        # Gold ids index the vocabulary axis of the teacher distribution. Surface an
        # out-of-range id rather than silently clamping it onto the last token (which
        # would demote the wrong token and quietly corrupt the objective).
        if bool(response_mask.any()):
            max_gold = int(input_ids[response_mask].max())
            if max_gold >= self.vocab_size:
                raise ValueError(
                    f"gold token id {max_gold} >= vocab_size {self.vocab_size}; "
                    "check the tokenizer / collator alignment."
                )
        gold = input_ids
        demote = self._demote_mask(log_p0, gold)  # B_{M,i}

        # Edited teacher target q (Eq. edited-target): drop the demoted mass and
        # renormalise EXACTLY over the surviving vocabulary, in log space. Setting the
        # demoted logits to -inf and subtracting the survivors' log-sum-exp guarantees
        # sum_v q(v) == 1 for any teacher confidence -- unlike the naive 1/(1-removed)
        # rescale, which needs an arbitrary clamp and under-normalises when the teacher
        # is near-certain of a demoted token (surviving mass below the clamp floor).
        log_q = log_p0.masked_fill(demote, float("-inf"))
        log_q = log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)
        q = log_q.exp()  # exactly 0 on B, sums to 1 over the survivors

        # KL(q || p_theta): q log q - q log p_theta, with q = 0 on the demoted set
        # (0 * -inf is masked away before it can produce a NaN).
        q_log_q = torch.where(demote, torch.zeros_like(log_q), q * log_q)
        kl_demote = q_log_q.sum(-1) - (q * log_p).sum(-1)  # answer positions
        p0 = log_p0.exp()
        removed = (p0 * demote).sum(-1)  # sum_{u in B} p0(u), for diagnostics only
        kl_self = (p0 * (log_p0 - log_p)).sum(-1)  # KL(p0 || p_theta), non-answer
        kl = torch.where(answer_mask, kl_demote, kl_self)  # [b, l]

        # Balanced weighting (Eq. balanced-weight): half the mass to answer-bearing
        # masked decisions, half to non-answer masked decisions; if one group is
        # empty its mass goes to the other. With A = response the non-answer group
        # is empty, so this is a uniform mean over masked answer positions.
        ans_pos = masked & answer_mask
        non_pos = masked & ~answer_mask  # empty unless self_distill and A is a strict subset of R
        ans_cnt = ans_pos.sum(-1).float()
        non_cnt = non_pos.sum(-1).float()
        if self.balanced_weight:
            both = (ans_cnt > 0) & (non_cnt > 0)
            w_ans = torch.where(both, 0.5, 1.0) / ans_cnt.clamp_min(1)
            w_non = torch.where(both, 0.5, 1.0) / non_cnt.clamp_min(1)
        else:
            total = (ans_cnt + non_cnt).clamp_min(1)
            w_ans = w_non = 1.0 / total
        weights = ans_pos * w_ans.unsqueeze(-1) + non_pos * w_non.unsqueeze(-1)

        seq_loss = (weights * kl).sum(-1)  # per-example, weights already sum to 1
        active = (ans_cnt + non_cnt) > 0
        loss = seq_loss[active].mean() if bool(active.any()) else log_p.sum() * 0.0

        metrics = {}
        if model.training:
            with torch.no_grad():
                gold_prob = log_p.gather(-1, gold.unsqueeze(-1)).squeeze(-1).exp()
                has_ans = bool(ans_pos.any())
                metrics = {
                    "tsd/forget_kl": float(loss.detach()),
                    "tsd/mask_rate": float(
                        masked.sum().float() / response_mask.sum().clamp_min(1)
                    ),
                    "tsd/removed_mass": float(removed[ans_pos].mean()) if has_ans else 0.0,
                    "tsd/gold_prob_student": float(gold_prob[ans_pos].mean())
                    if has_ans
                    else 0.0,
                    "tsd/self_distill_frac": float(
                        non_cnt.sum() / masked.sum().clamp_min(1)
                    ),
                }
        return loss, outputs, metrics

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_loss, outputs, metrics = self._tsd_loss(model, inputs["forget"])
        retain_loss = self.optional_retain_loss(model, inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss

        if model.training:
            if not isinstance(retain_loss, float):
                metrics["tsd/retain_loss"] = float(retain_loss.detach())
            self._pending_metrics.update(metrics)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if model.training and self.accelerator.sync_gradients and self._pending_metrics:
            self.log(self._pending_metrics)
            self._pending_metrics = {}
        return loss
