"""Simple Trajectory-Selective Distillation (SimpleTSD) for masked-diffusion LM unlearning.

Retain-set-free; runs on the forget set only. Two differences from ``DiffusionTSD``:

1. PCSS (Alg. 1): score each answer token by prompt-dependence under *partial* masking,
   ``s_i = mean_M [log p0(y_i | x, y_M) - log p0(y_i | x_empty, y_M)]`` (i masked), and
   keep ``A_hat`` by an adaptive threshold (Otsu). Answer tokens that copy a question
   word are dropped. Partial (not full) masking matters: a fully masked answer makes the
   promptless teacher near-uniform everywhere, so ``A_hat`` swallows the whole answer.

2. Self-distillation on the non-selected answer tokens ``R = M \\ A_hat`` (``DiffusionTSD``
   only distils non-answer positions, vacuous on TOFU), so fluency survives without a
   retain set.

Demoted set per selected position: ``{y_i}`` (``demotion="minimal"``) or ``{y_i}`` plus
the teacher's ``belief_top_k`` highest-probability tokens (``demotion="belief"``, as in
``DiffusionTSD``). Special/control tokens are excluded from selection and from both sides
of the forget KL, while the self-distillation path retains the full vocabulary.
Per-example loss ``L = alpha*L_f + (1-alpha)*L_d`` when ``R`` is nonempty, else ``L_f``;
``alpha`` is the forget weight, not a retain-NLL weight.
"""

import re
import warnings
import torch
from trainer.dlm.unlearn.grad_diff import DiffusionGradDiff

_SPECIAL_RE = re.compile(r"<\|?[^<>]*\|?>")


def _group_word_indices(surfaces):
    """Group token positions into whitespace-delimited words, so a name split across
    subword tokens ("Hs"+"iao" -> "Hsiao") is matched as one word."""
    groups, cur = [], []
    for i, s in enumerate(surfaces):
        if s[:1].isspace() and cur:
            groups.append(cur)
            cur = [i]
        else:
            cur.append(i)
    if cur:
        groups.append(cur)
    return groups


def _content_frags(surface, min_len=2):
    surface = _SPECIAL_RE.sub(" ", surface).lower()
    return [f for f in re.findall(r"[a-z0-9]+", surface) if len(f) >= min_len]


class DiffusionSimpleTSD(DiffusionGradDiff):
    def __init__(
        self,
        alpha=0.5,
        tau=0.0,
        gamma=1.0,
        select_method="otsu",
        sel_mc=8,
        loss_mc=1,
        mask_mode="stratified",
        retain_frac=0.5,
        select_temp=1.0,
        drop_question_overlap=True,
        topk_frac=0.5,
        otsu_relax=0.0,
        demotion="minimal",
        belief_top_k=5,
        mask_weight_power=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if select_method not in ("otsu", "tau", "topk"):
            raise ValueError(f"select_method must be otsu|tau|topk, got {select_method!r}")
        if not (0.0 < topk_frac <= 1.0):
            raise ValueError(f"topk_frac must be in (0, 1], got {topk_frac}")
        if int(sel_mc) < 1:
            raise ValueError(f"sel_mc must be >= 1, got {sel_mc}")
        if int(loss_mc) < 1:
            raise ValueError(f"loss_mc must be >= 1, got {loss_mc}")
        if mask_mode not in ("stratified", "cover"):
            raise ValueError(f"mask_mode must be stratified|cover, got {mask_mode!r}")
        if not (0.0 <= retain_frac < 1.0):
            raise ValueError(f"retain_frac must be in [0, 1), got {retain_frac}")
        if not (float(select_temp) > 0.0):
            raise ValueError(f"select_temp must be > 0, got {select_temp}")
        if not (0.0 <= otsu_relax < 1.0):
            raise ValueError(f"otsu_relax must be in [0, 1), got {otsu_relax}")
        if demotion not in ("minimal", "belief"):
            raise ValueError(f"demotion must be minimal|belief, got {demotion!r}")
        if demotion == "belief" and int(belief_top_k) < 1:
            raise ValueError("belief_top_k must be >= 1 for belief demotion")
        if float(mask_weight_power) < 0.0:
            raise ValueError(f"mask_weight_power must be >= 0, got {mask_weight_power}")
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.select_method = select_method
        self.sel_mc = int(sel_mc)
        self.loss_mc = int(loss_mc)
        self.mask_mode = mask_mode
        self.retain_frac = float(retain_frac)
        self.select_temp = float(select_temp)
        # In "cover" mode each draw masks a rho(t) fraction of the answer and splits it
        # retain_frac / (1 - retain_frac) between R and A_hat, drawing forget positions
        # weighted by their PCSS score. With few draws the highest-score forget tokens
        # dominate and low-score ones are rarely masked, so coverage of A_hat stays thin.
        if mask_mode == "cover" and self.loss_mc < 4:
            warnings.warn(
                f"mask_mode='cover' with loss_mc={self.loss_mc} < 4: few Monte-Carlo "
                f"draws give thin score-weighted coverage of A_hat. Consider loss_mc >= 4.",
                stacklevel=2,
            )
        self.drop_question_overlap = bool(drop_question_overlap)
        self.topk_frac = float(topk_frac)
        self.otsu_relax = float(otsu_relax)
        self.demotion = demotion
        self.belief_top_k = int(belief_top_k)
        # Emphasis on high-mask (few-revealed) states: per-example loss weight is the
        # answer-mask fraction raised to this power. 0.0 = uniform (original behaviour).
        # Higher values up-weight draws where most of the answer is still masked -- the
        # early-generation commitments a decoder makes first, hence the states that must
        # be content-free for order-robust semantic erasure (see docstring).
        self.mask_weight_power = float(mask_weight_power)
        self.mask_id = self.processing_class.mask_token_id
        self.pad_id = self.processing_class.pad_token_id
        self.vocab_size = len(self.processing_class)
        lexical_vocab_size = self.vocab_size - len(
            self._forbidden_unlearning_token_ids()
        )
        if lexical_vocab_size < 2:
            raise ValueError(
                "SimpleTSD requires at least two non-special tokenizer tokens"
            )
        # Belief demotion must leave at least one surviving token to renormalise over.
        if demotion == "belief" and self.belief_top_k + 1 >= lexical_vocab_size:
            raise ValueError(
                f"belief_top_k={self.belief_top_k} leaves no surviving vocabulary "
                f"(lexical_vocab_size={lexical_vocab_size})"
            )
        self.ref_model = self._prepare_ref_model(self.model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self._pending_metrics = {}

    @torch.no_grad()
    def _teacher_log_probs(self, y, attention_mask):
        out = self.ref_model(input_ids=y, attention_mask=attention_mask)
        return self._log_probs(self._postprocess_outputs(out).logits)

    def _prompt_mask(self, input_ids, answer_mask, attention_mask):
        """Prompt positions to blank out for ``x_empty``: every attended, non-answer,
        non-pad token. Pad is kept (pad == eos on Dream/LLaDA, blanking it injects noise).
        """
        prompt = ~answer_mask & (input_ids != self.pad_id)
        if attention_mask is not None:
            prompt = prompt & attention_mask.bool()
        return prompt

    @torch.no_grad()
    def _select_forget_positions(self, input_ids, answer_mask, attention_mask):
        """``A_hat`` and per-token prompt-dependence scores ``s_i`` (PCSS, Alg. 1).

        Average over ``sel_mc`` partial masks of ``log p0(y_i | x, y_M) -
        log p0(y_i | x_empty, y_M)`` at masked positions, then threshold and drop
        question-copies.
        """
        b, l = answer_mask.shape
        device = input_ids.device
        prompt = self._prompt_mask(input_ids, answer_mask, attention_mask)
        selectable = answer_mask & self._unlearning_position_mask(input_ids)
        gold = input_ids.unsqueeze(-1)

        s_sum = torch.zeros(b, l, device=device)
        cnt = torch.zeros(b, l, device=device)
        for _ in range(self.sel_mc):
            t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(b, device=device)
            p_mask = (1.0 - self.scheduler(t)).unsqueeze(1)  # rho, per example
            m = (torch.rand((b, l), device=device) < p_mask) & selectable
            if not bool(m.any()):
                continue
            y = torch.where(m, self.mask_id, input_ids)
            log_cond = self._condition_on_unlearning_vocabulary(
                self._teacher_log_probs(y, attention_mask)
            ).gather(-1, gold).squeeze(-1)
            y_base = torch.where(prompt, self.mask_id, y)
            log_base = self._condition_on_unlearning_vocabulary(
                self._teacher_log_probs(y_base, attention_mask)
            ).gather(-1, gold).squeeze(-1)
            s_sum = s_sum + torch.where(m, log_cond - log_base, torch.zeros_like(s_sum))
            cnt = cnt + m.float()
        s = s_sum / cnt.clamp_min(1.0)  # never-masked tokens keep s = 0

        a_hat = self._threshold_scores(s, selectable)
        if self.drop_question_overlap:
            a_hat = a_hat & ~self._question_overlap_mask(input_ids, answer_mask, prompt)
        return a_hat, s

    @staticmethod
    def _otsu_threshold(vals, relax=0.0):
        """1-D Otsu split (maximise between-class variance); caller selects ``s >= thr``.
        ``relax`` in [0, 1) lowers the cut toward the minimum: ``thr' = thr - relax*(thr - min)``.
        """
        xs, _ = torch.sort(vals)
        n = xs.numel()
        if n < 2 or float(xs[0]) == float(xs[-1]):
            return xs[0] if n else vals.new_tensor(0.0)
        total = xs.sum()
        csum = torch.cumsum(xs, 0)[:-1]
        cn = torch.arange(1, n, device=xs.device, dtype=xs.dtype)
        w0 = cn / n
        m0 = csum / cn
        m1 = (total - csum) / (n - cn)
        score = w0 * (1.0 - w0) * (m0 - m1) ** 2
        i = int(torch.argmax(score))
        thr = 0.5 * (xs[i] + xs[i + 1])
        return thr - relax * (thr - xs[0]) if relax > 0.0 else thr

    def _threshold_scores(self, s, answer_mask):
        """Boolean ``A_hat`` from scores ``s``: fixed ``tau``, per-example Otsu, or top-``topk_frac``."""
        if self.select_method == "tau":
            return (s >= self.tau) & answer_mask
        a_hat = torch.zeros_like(answer_mask)
        for bi in range(s.shape[0]):
            idx = answer_mask[bi].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            vals = s[bi, idx]
            if self.select_method == "topk":
                k = max(1, int(round(self.topk_frac * idx.numel())))
                thr = torch.topk(vals, k).values.min()
            else:
                thr = self._otsu_threshold(vals, self.otsu_relax)
            a_hat[bi, idx[vals >= thr]] = True
        return a_hat

    def _question_overlap_mask(self, input_ids, answer_mask, prompt):
        """Answer words that merely echo a word already in the question (no secret to
        forget). A word is dropped iff every content fragment it contains is a question
        word; word-grouping catches names split across subword tokens.
        """
        tok = self.processing_class
        overlap = torch.zeros_like(answer_mask)
        for bi in range(input_ids.shape[0]):
            q_ids = input_ids[bi][prompt[bi]].tolist()
            if not q_ids:
                continue
            q_text = tok.decode(q_ids, skip_special_tokens=True).lower()
            qwords = {w for w in re.findall(r"[a-z0-9]+", q_text) if len(w) >= 2}
            if not qwords:
                continue
            ans_idx = answer_mask[bi].nonzero(as_tuple=True)[0]
            if ans_idx.numel() == 0:
                continue
            surfaces = [tok.decode([int(t)]) for t in input_ids[bi][ans_idx]]
            for grp in _group_word_indices(surfaces):
                frags = _content_frags("".join(surfaces[j] for j in grp))
                if frags and all(f in qwords for f in frags):
                    for j in grp:
                        overlap[bi, ans_idx[j]] = True
        return overlap

    def _sample_mask(self, answer_mask, a_hat):
        """Answer-stratified mask over the whole answer A (as in ``DiffusionTSD``): draw
        ``k ~ Uniform{1..|A|}`` and mask ``k`` positions, so both ``A_hat`` (forget) and
        its complement (self-distill) get masked. If no ``A_hat`` position is drawn, one
        is forced in so every state with a nonempty ``A_hat`` carries forget signal.
        """
        b, l = answer_mask.shape
        device = answer_mask.device
        count = answer_mask.sum(-1)  # |A| per example
        scores = torch.rand(b, l, device=device).masked_fill(~answer_mask, -1.0)
        order = scores.argsort(dim=-1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(-1, order, torch.arange(l, device=device).expand(b, l))
        u = torch.rand(b, device=device)
        k = (u * count.clamp_min(1).float()).floor().long() + 1  # in [1, |A|]
        k = torch.minimum(k, count.clamp_min(1))
        masked = (rank < k.unsqueeze(-1)) & answer_mask

        need = a_hat.any(-1) & ~(masked & a_hat).any(-1)
        if bool(need.any()):
            pick = scores.masked_fill(~a_hat, -1.0).argmax(-1, keepdim=True)
            forced = torch.zeros_like(masked).scatter_(-1, pick, True)
            masked = masked | (forced & need.unsqueeze(-1))
        return masked

    def _score_logits(self, scores, bi, idx, temp):
        """Selection logits for positions ``idx`` of example ``bi``: the PCSS scores
        standardised within this stratum and divided by ``temp`` (so the temperature is
        scale-free w.r.t. the raw log-ratio scale). Returns zeros -- i.e. uniform
        selection -- when no scores are available (manual-span ``A_hat``) or the stratum
        has a single position.
        """
        n = idx.numel()
        if scores is None or n <= 1:
            return torch.zeros(n, device=idx.device)
        s = scores[bi, idx]
        s = (s - s.mean()) / (s.std() + 1e-8)
        return s / temp

    @staticmethod
    def _weighted_topk(logits, k):
        """Sample ``k`` distinct indices without replacement with probability
        proportional to ``exp(logits)`` (Gumbel-top-k). All-zero ``logits`` -> uniform.
        """
        u = torch.rand(logits.numel(), device=logits.device).clamp_min(1e-12)
        gumbel = -torch.log(-torch.log(u))
        return (logits + gumbel).topk(k).indices

    def _sample_masks_cover(self, answer_mask, a_hat, scores=None):
        """``loss_mc`` score-weighted masks, one per Monte-Carlo draw (Alg. 2, multi-draw).

        Per example, per draw we follow ordinary MDLM masking with a forget/retain
        composition split:

        1. Pick a timestep ``t ~ U(eps, 1)`` and derive the mask rate ``rho = 1 -
           scheduler(t)``; the draw masks ``n_mask = round(rho * |A|)`` answer tokens.
        2. Split the budget by ``retain_frac`` (= q): ``kf = round((1 - q) * n_mask)``
           come from the forget set ``A_hat`` and ``kr = n_mask - kf`` from the retain
           set ``R = A \\ A_hat``. At least one forget position is masked when ``A_hat``
           is nonempty, so every draw carries a forget signal.
        3. Draw the ``kf`` forget positions *weighted by their PCSS score* (higher
           prompt-dependence -> higher probability; Gumbel-top-k, temperature
           ``select_temp``); retain positions are drawn uniformly. With no scores
           (manual span) forget selection is uniform too.

        Coverage of ``A_hat`` across draws is therefore soft: high-score forget tokens
        are masked often, low-score ones rarely -- there is no hard guarantee that every
        position is hit (unlike an explicit coverage floor). Returns a bool tensor
        ``(loss_mc, b, l)``.
        """
        b, l = answer_mask.shape
        device = answer_mask.device
        L = self.loss_mc
        q = self.retain_frac  # retain share of each draw's masked budget
        masks = torch.zeros(L, b, l, dtype=torch.bool, device=device)
        for bi in range(b):
            f_idx = (a_hat[bi] & answer_mask[bi]).nonzero(as_tuple=False).flatten()
            r_idx = (answer_mask[bi] & ~a_hat[bi]).nonzero(as_tuple=False).flatten()
            n_ans = int(answer_mask[bi].sum())
            n_f, n_r = f_idx.numel(), r_idx.numel()
            if n_ans == 0:
                continue
            # Forget logits (score-weighted) are fixed per example; retain is uniform.
            # Only the Gumbel noise inside _weighted_topk is redrawn per draw.
            f_logits = self._score_logits(scores, bi, f_idx, self.select_temp)
            r_logits = torch.zeros(n_r, device=device)
            for i in range(L):
                t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(1, device=device)
                rho = float(1.0 - self.scheduler(t))
                n_mask = int(round(rho * n_ans))
                kf = int(round((1.0 - q) * n_mask))
                kf = min(max(kf, 1), n_f) if n_f > 0 else 0  # >=1 forget/draw when A_hat nonempty
                kr = min(max(n_mask - kf, 0), n_r)
                if kf > 0:
                    masks[i, bi, f_idx[self._weighted_topk(f_logits, kf)]] = True
                if kr > 0:
                    masks[i, bi, r_idx[self._weighted_topk(r_logits, kr)]] = True
        return masks

    def _demote_mask(self, log_p0, gold):
        """Vocabulary tokens to demote per position (matches ``DiffusionTSD``): the gold
        token, plus the teacher's ``belief_top_k`` most-probable tokens when belief.
        """
        demote = torch.zeros_like(log_p0, dtype=torch.bool)
        demote.scatter_(-1, gold.unsqueeze(-1), True)
        if self.demotion == "belief":
            topk = log_p0.topk(min(self.belief_top_k, log_p0.shape[-1]), dim=-1).indices
            demote.scatter_(-1, topk, True)
        return demote

    def _forget_kl(self, log_p, log_p0, y_m, ctx):
        """Per-position forget objective ``KL(target || p_theta)`` on the ``A_hat`` set,
        plus the teacher p0-mass removed per position (diagnostic ``removed_mass``).

        Default (``DiffusionSimpleTSD``) target is the *edited teacher* ``q``: take the
        with-question teacher ``p0(.|x, y_M)``, zero the demoted set (gold, and the
        belief top-K under belief demotion) and renormalise over the survivors in log
        space. Subclasses override this to install a different forget target while
        reusing the whole selection / masking / weighting pipeline unchanged.

        Args:
            log_p:  student log-probs ``log p_theta(.|x, y_M)``  ``[b, l, V]``.
            log_p0: with-question teacher log-probs ``log p0(.|x, y_M)`` ``[b, l, V]``.
            y_m:    masked input ids ``[b, l]`` (the state the forwards were run on).
            ctx:    the ``_forget_context`` dict (``input_ids``, ``attention_mask``,
                    ``answer_mask``, ``a_hat``, ...).
        Returns ``(kl_f, removed)``, each ``[b, l]``.
        """
        log_p_forget = self._condition_on_unlearning_vocabulary(log_p)
        log_p0_forget = self._condition_on_unlearning_vocabulary(log_p0)
        demote = self._demote_mask(log_p0_forget, ctx["input_ids"])
        log_q = log_p0_forget.masked_fill(demote, float("-inf"))
        log_q = log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)
        q = log_q.exp()
        support = ~(demote | self._forbidden_unlearning_vocab_mask(log_q.device))
        safe_log_q = log_q.masked_fill(~support, 0.0)
        safe_log_p = log_p_forget.masked_fill(~support, 0.0)
        kl_f = (q * (safe_log_q - safe_log_p)).sum(-1)
        removed = (log_p0_forget.exp() * demote).sum(-1)
        # Verification diagnostics (read by _draw_loss): gold mass under the RAW teacher
        # (what it remembers) vs under the edited TARGET the student is fit to. The target
        # gold prob is 0 by construction here (gold set to -inf), proving the demotion
        # happens on the teacher-side target, not the student.
        with torch.no_grad():
            gold_ids = ctx["input_ids"].unsqueeze(-1)
            self._forget_diag = {
                "gold_prob_teacher_raw": log_p0_forget.gather(-1, gold_ids)
                .squeeze(-1)
                .exp(),
                "gold_prob_target": log_q.gather(-1, gold_ids).squeeze(-1).exp(),
            }
        return kl_f, removed

    def _forget_context(self, forget_inputs):
        """Mask-independent per-batch context reused across every ``loss_mc`` draw: the
        preprocessed inputs, the answer span ``A``, the selected forget set ``A_hat`` (and
        its PCSS scores), plus the joint cover masks when ``mask_mode='cover'``. Building
        this once keeps the expensive selection (``sel_mc`` teacher forwards) and cover
        planning off the per-draw path so only the student/teacher forward repeats.
        """
        prep = self._preprocess_inputs(
            {
                "input_ids": forget_inputs["input_ids"],
                "labels": forget_inputs["labels"],
                "attention_mask": forget_inputs.get("attention_mask", None),
            }
        )
        input_ids = prep["input_ids"]
        attention_mask = prep.get("attention_mask", None)
        answer_mask = prep["labels"] != -100

        if bool(answer_mask.any()) and int(input_ids[answer_mask].max()) >= self.vocab_size:
            raise ValueError("gold token id >= vocab_size; check tokenizer/collator alignment.")

        # A_hat: explicit span (manual option) or PCSS selection; answer positions only.
        provided = forget_inputs.get("answer_mask", None)
        if provided is not None:
            provided = provided.to(answer_mask.device).bool()
            if provided.shape != answer_mask.shape:
                raise ValueError(
                    f"answer_mask shape {tuple(provided.shape)} != inputs {tuple(answer_mask.shape)}."
                )
            a_hat = (
                provided
                & answer_mask
                & self._unlearning_position_mask(input_ids)
            )
            scores = None
        else:
            a_hat, scores = self._select_forget_positions(input_ids, answer_mask, attention_mask)

        cover_masks = (
            self._sample_masks_cover(answer_mask, a_hat, scores)
            if self.mask_mode == "cover" else None
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_mask": answer_mask,
            "a_hat": a_hat,
            "scores": scores,
            "cover_masks": cover_masks,
        }

    def _draw_loss(self, model, ctx, draw_idx):
        """Loss for a single mask draw over the fixed ``A_hat`` (context from
        ``_forget_context``). Returns ``(loss, outputs, metrics)``; ``loss`` is the
        per-example loss averaged over the active examples (those with a selected forget
        position). The caller either averages several draws (eval / single-draw training)
        or backprops each draw independently (multi-draw training, see ``training_step``).
        """
        input_ids = ctx["input_ids"]
        attention_mask = ctx["attention_mask"]
        answer_mask = ctx["answer_mask"]
        a_hat = ctx["a_hat"]
        cover_masks = ctx["cover_masks"]

        masked = (
            cover_masks[draw_idx] if cover_masks is not None
            else self._sample_mask(answer_mask, a_hat)
        )
        y_m = torch.where(masked, self.mask_id, input_ids)

        outputs = self._postprocess_outputs(model(input_ids=y_m, attention_mask=attention_mask))
        log_p = self._log_probs(outputs.logits)
        log_p0 = self._teacher_log_probs(y_m, attention_mask)

        # Forget target (overridable via _forget_kl); default = edited teacher
        # (demote gold + renorm). kl_f = KL(target || p_theta) on forget positions;
        # kl_d = KL(p0 || p_theta) on the rest.
        kl_f, removed = self._forget_kl(log_p, log_p0, y_m, ctx)
        p0 = log_p0.exp()
        kl_d = (p0 * (log_p0 - log_p)).sum(-1)
        return self._combine_draw(model, ctx, masked, outputs, log_p, kl_f, kl_d, removed)

    def _combine_draw(self, model, ctx, masked, outputs, log_p, kl_f, kl_d, removed):
        """Combine the per-position forget loss ``kl_f`` (on ``F = M ∩ A_hat``) and
        self-distillation ``kl_d`` (on ``R = M \\ A_hat``) into the per-draw scalar loss,
        with the ``active``/``mask_weight_power`` weighting and training metrics. Shared by
        the base draw and any subclass whose forward path differs (e.g. a hidden-state
        objective that still wants the identical composition and logging).
        """
        input_ids = ctx["input_ids"]
        answer_mask = ctx["answer_mask"]
        a_hat = ctx["a_hat"]
        gold = input_ids.unsqueeze(-1)

        f_pos = masked & a_hat  # F = M ∩ A_hat
        r_pos = masked & ~a_hat  # R = M \ A_hat
        f_cnt = f_pos.sum(-1).float()
        r_cnt = r_pos.sum(-1).float()
        loss_f = (kl_f * f_pos).sum(-1) / f_cnt.clamp_min(1)
        loss_d = (kl_d * r_pos).sum(-1) / r_cnt.clamp_min(1)
        has_r = r_cnt > 0
        loss_ex = torch.where(has_r, self.alpha * loss_f + (1 - self.alpha) * loss_d, loss_f)

        # An example contributes iff it has a selected forget position. Keyed on A_hat
        # (not this draw's f_cnt) so it is correct in cover mode when a draw happens to
        # mask no A_hat position. In stratified mode _sample_mask forces >=1 A_hat
        # position per draw, so this equals f_cnt > 0.
        active = a_hat.any(-1)  # drop examples with no selected position (Alg. 2)
        if self.mask_weight_power == 0.0:
            loss = loss_ex[active].mean() if bool(active.any()) else log_p.sum() * 0.0
        else:
            # Up-weight examples whose answer is mostly masked in this draw (few tokens
            # revealed = the early-generation states that must be content-free). Weight =
            # (masked answer fraction)^power, restricted to active examples, then a
            # weighted mean so the loss scale is unchanged when the fractions are equal.
            ans_cnt = answer_mask.sum(-1).float()
            frac = (masked & answer_mask).sum(-1).float() / ans_cnt.clamp_min(1)
            w = frac.pow(self.mask_weight_power) * active.float()
            wsum = w.sum()
            loss = (w * loss_ex).sum() / wsum if bool(wsum > 0) else log_p.sum() * 0.0

        metrics = {}
        if model.training:
            with torch.no_grad():
                act, hr, hf = bool(active.any()), bool(has_r.any()), bool(f_pos.any())
                denom = answer_mask.sum().clamp_min(1)
                gold_prob = log_p.gather(-1, gold).squeeze(-1).exp()
                metrics = {
                    "simple_tsd/loss": float(loss.detach()),
                    "simple_tsd/forget_kl": float(loss_f[active].mean()) if act else 0.0,
                    "simple_tsd/distill_kl": float(loss_d[has_r].mean()) if hr else 0.0,
                    "simple_tsd/selected_frac": float(a_hat.sum().float() / denom),
                    "simple_tsd/skipped_examples": float((~active).sum()),
                    "simple_tsd/mask_rate": float(masked.sum().float() / denom),
                    "simple_tsd/removed_mass": float(removed[f_pos].mean()) if hf else 0.0,
                    "simple_tsd/gold_prob_student": float(gold_prob[f_pos].mean()) if hf else 0.0,
                }
                # Verification that the forget TARGET is gold-clean (set by _forget_kl):
                # gold mass under the raw teacher (remembered) vs under the target the
                # student is trained toward. gold_prob_target ~ 0 for the default
                # (gold-demoted) target; > 0 for the QMask promptless target reveals the
                # residual leak the demotion alone does not remove.
                diag = getattr(self, "_forget_diag", None)
                if diag is not None and hf and "gold_prob_teacher_raw" in diag:
                    metrics["simple_tsd/gold_prob_teacher_raw"] = float(
                        diag["gold_prob_teacher_raw"][f_pos].mean()
                    )
                    metrics["simple_tsd/gold_prob_target"] = float(
                        diag["gold_prob_target"][f_pos].mean()
                    )
                if ctx["scores"] is not None and bool(answer_mask.any()):
                    sc = ctx["scores"]
                    metrics["simple_tsd/score_mean"] = float(sc[answer_mask].mean())
                    if hf:
                        # Score-weighted masking should prefer high-PCSS forget tokens, so
                        # this sits above score_mean when select_temp is doing its job.
                        metrics["simple_tsd/masked_forget_score"] = float(sc[f_pos].mean())
                # Subclass-supplied scalar metrics (already namespaced), e.g. CSD's
                # concept-direction diagnostics. Base sets nothing.
                extra = getattr(self, "_method_metrics", None)
                if extra:
                    metrics.update(extra)
        return loss, outputs, metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Average the loss over ``loss_mc`` mask draws. Used for evaluation (``no_grad``,
        so no graph is retained) and for single-draw training (``loss_mc == 1``, the
        original path). Multi-draw *training* does not go through here -- it uses
        ``training_step``'s per-draw backward so activation memory stays at one student
        graph regardless of ``loss_mc``; averaging several live graphs here would OOM.
        """
        ctx = self._forget_context(inputs["forget"])
        loss_sum, outputs, metrics = None, None, {}
        for i in range(self.loss_mc):
            loss_i, outputs, metrics = self._draw_loss(model, ctx, i)
            loss_sum = loss_i if loss_sum is None else loss_sum + loss_i
        mean_loss = loss_sum / self.loss_mc
        if model.training:
            metrics["simple_tsd/loss"] = float(mean_loss.detach())
            self._pending_metrics.update(metrics)
        return (self.gamma * mean_loss, outputs) if return_outputs else self.gamma * mean_loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Single draw: identical to the original path -- let HF drive forward/backward.
        if self.loss_mc <= 1:
            loss = super().training_step(model, inputs, num_items_in_batch)
            if model.training and self.accelerator.sync_gradients and self._pending_metrics:
                self.log(self._pending_metrics)
                self._pending_metrics = {}
            return loss

        # Multi-draw: forward+backward one draw at a time so only a single student graph
        # is alive at once. Holding all loss_mc graphs until one backward scales activation
        # memory by loss_mc and OOMs under ZeRO-3. Gradients from each draw accumulate in
        # DeepSpeed's buffer; the ZeRO reduce-scatter + engine.step() fire only on the
        # final contribution of this optimizer update. accelerate 1.11 routes
        # accelerator.backward -> DeepSpeedEngineWrapper.backward(sync_gradients=
        # accelerator.sync_gradients), which sets the accumulation boundary and gates
        # engine.step() on that flag. So we force sync_gradients=False on every draw but
        # the last (accumulate, no step), and let the last draw sync+step -- but only on
        # HF's own boundary micro-step (step_sync); on non-boundary micro-steps all draws
        # stay False and simply accumulate, exactly as ordinary grad accumulation does.
        model.train()
        inputs = self._prepare_inputs(inputs)
        ctx = self._forget_context(inputs["forget"])
        grad_state = self.accelerator.gradient_state
        step_sync = self.accelerator.sync_gradients
        n = self.loss_mc
        total = None
        last_metrics = {}
        for i in range(n):
            grad_state._set_sync_gradients(step_sync and i == n - 1)
            with self.compute_loss_context_manager():
                loss_i, _outputs, last_metrics = self._draw_loss(model, ctx, i)
                loss_i = self.gamma * loss_i / n  # sum over draws == mean, matches single-draw scale
            self.accelerator.backward(loss_i)
            total = loss_i.detach() if total is None else total + loss_i.detach()
        grad_state._set_sync_gradients(step_sync)  # restore HF's view for the outer loop

        if model.training and step_sync:
            last_metrics["simple_tsd/loss"] = float(total)
            self._pending_metrics.update(last_metrics)
            if self._pending_metrics:
                self.log(self._pending_metrics)
                self._pending_metrics = {}
        return total
