"""Simple Trajectory-Selective Distillation (SimpleTSD) for masked-diffusion LM unlearning.

Implements ``docs/simple_tsd_algorithms.tex``. SimpleTSD is retain-set-free and
runs entirely on the forget set. It differs from the full ``DiffusionTSD`` in two
deliberate ways that make it "simple":

1. **Prompt-conditional saliency selection (PCSS, Alg. 1).** Rather than treating
   the whole answer as forget, SimpleTSD scores each answer token by how much the
   prompt raises its teacher probability under **partial** masking, averaged over
   ``sel_mc`` schedule-sampled masks,
   ``s_i = mean_M [ log p0(y_i | x, y_M) - log p0(y_i | x_empty, y_M) ]`` with ``i``
   masked, and keeps ``A_hat`` via an adaptive per-example threshold (Otsu by
   default). Partial masking is essential: scoring over the *fully* masked answer
   collapses (the promptless teacher is near-uniform on every position, so every
   token scores as prompt-bound and ``A_hat`` covers ~everything, leaving ``R``
   empty and self-distillation inert). Answer tokens that merely copy a token from
   the question are then subtracted -- the prompt already reveals that surface form,
   so it is not secret to forget. A caller may instead supply an explicit span via
   ``answer_mask`` (the manual-span option), which skips the scoring pass.

2. **Self-distillation on the *non-selected answer* tokens.** Within one partially
   masked answer state, masked positions split into ``F = M ∩ A_hat`` (demote the
   forget-token set, renormalise, ``KL(q || p_theta)``) and ``R = M \\ A_hat``
   (``KL(p0 || p_theta)`` — copy the teacher). The full ``DiffusionTSD`` only
   self-distils *non-answer* positions, which are vacuous on TOFU; SimpleTSD instead
   preserves the generic answer tokens, so fluency survives without a retain set.

At each ``F`` position the demoted set is ``{y_i}`` (``demotion="minimal"``, default)
or the teacher's ``belief_top_k`` highest-probability tokens (``demotion="belief"``) --
the belief variant zeroes the teacher's most-confident predictions (which at a fact
position include the gold token) to curb paraphrase leakage.

The per-example objective is ``L = alpha * L_f + (1 - alpha) * L_d`` when ``R`` is
nonempty, else ``L = L_f``. ``alpha`` (default 0.5) is the forget weight, *not* a
retain-NLL weight; there is no retain dataset. Masking follows the model's ordinary
diffusion schedule (independent Bernoulli(rho) per answer position, rho = 1 -
scheduler(t)); if a draw masks no selected position, one random ``A_hat`` position
is added so every trained state carries forget signal.
"""

import re

import torch

from trainer.dlm.unlearn.base import DiffusionUnlearnTrainer


def _group_word_indices(surfaces):
    """Group token positions into whitespace-delimited words. A token whose decoded
    surface begins with whitespace (or the first token) opens a new word; subword
    continuations attach to the current word. Used to reassemble a name split across
    subword tokens before checking it against the question."""
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


# Special tokens (<|eot_id|>, <|endoftext|>, <eos>, ...) decode with no leading space,
# so they glue onto the final content word; strip them before extracting fragments,
# else their pieces ("eot", "endoftext") keep a copied word (question-echo) in A_hat.
_SPECIAL_RE = re.compile(r"<\|?[^<>]*\|?>")


def _content_frags(surface, min_len=2):
    surface = _SPECIAL_RE.sub(" ", surface).lower()
    return [f for f in re.findall(r"[a-z0-9]+", surface) if len(f) >= min_len]


class DiffusionSimpleTSD(DiffusionUnlearnTrainer):
    def __init__(
        self,
        alpha=0.5,
        tau=0.0,
        gamma=1.0,
        select_method="otsu",
        sel_mc=8,
        drop_question_overlap=True,
        topk_frac=0.5,
        otsu_relax=0.0,
        demotion="minimal",
        belief_top_k=5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if select_method not in ("otsu", "tau", "topk"):
            raise ValueError(
                f"select_method must be otsu|tau|topk, got {select_method!r}"
            )
        if not (0.0 < topk_frac <= 1.0):
            raise ValueError(f"topk_frac must be in (0, 1], got {topk_frac}")
        if int(sel_mc) < 1:
            raise ValueError(f"sel_mc must be >= 1, got {sel_mc}")
        if not (0.0 <= otsu_relax < 1.0):
            raise ValueError(f"otsu_relax must be in [0, 1), got {otsu_relax}")
        if demotion not in ("minimal", "belief"):
            raise ValueError(f"demotion must be minimal|belief, got {demotion!r}")
        if demotion == "belief" and int(belief_top_k) < 1:
            raise ValueError("belief_top_k must be >= 1 for belief demotion")
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.select_method = select_method
        self.sel_mc = int(sel_mc)
        self.drop_question_overlap = bool(drop_question_overlap)
        self.topk_frac = float(topk_frac)
        self.otsu_relax = float(otsu_relax)
        self.demotion = demotion
        self.belief_top_k = int(belief_top_k)
        self.mask_id = self.processing_class.mask_token_id
        self.pad_id = self.processing_class.pad_token_id
        self.vocab_size = len(self.processing_class)
        # Belief demotion must leave at least one surviving token, else the edited
        # target renormalises over an empty support (log-sum-exp of -inf = -inf).
        if demotion == "belief" and self.belief_top_k + 1 >= self.vocab_size:
            raise ValueError(
                f"belief_top_k={self.belief_top_k} leaves no surviving vocabulary "
                f"(vocab_size={self.vocab_size})"
            )
        self.ref_model = self._prepare_ref_model(self.model)
        # The teacher is a frozen snapshot: forwarded under no_grad below, but also
        # detach its parameters so an accidental grad path cannot train it.
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self._pending_metrics = {}

    # -- teacher log-probs on a given (already preprocessed) noised sequence -----
    @torch.no_grad()
    def _teacher_log_probs(self, y, attention_mask):
        out = self.ref_model(input_ids=y, attention_mask=attention_mask)
        return self._log_probs(self._postprocess_outputs(out).logits)

    # -- prompt-dependence selection (docs/simple_tsd_algorithms.tex Alg. 1) ----
    def _prompt_mask(self, input_ids, answer_mask, attention_mask):
        """Positions holding prompt content, to blank out for ``x_empty``.

        The prompt is every attended, non-answer, non-pad position (question tokens
        plus the chat-template scaffold). Trailing pad is left untouched -- on Dream
        and LLaDA ``pad_token_id == eos_token_id``, so replacing it would inject
        spurious tokens rather than remove conditioning.
        """
        prompt = answer_mask.new_zeros(answer_mask.shape)
        prompt |= ~answer_mask & (input_ids != self.pad_id)
        if attention_mask is not None:
            prompt &= attention_mask.bool()
        return prompt

    @torch.no_grad()
    def _select_forget_positions(self, input_ids, answer_mask, attention_mask):
        """``A_hat`` and the per-token scores ``s_i`` via prompt-conditional
        saliency selection (PCSS).

        Each answer token is scored by prompt-dependence under **partial** masking,
        averaged over ``sel_mc`` schedule-sampled masks:

            s_i = mean over draws where i is masked of
                  log p0(y_i | x, y_M) - log p0(y_i | x_empty, y_M).

        The original Alg. 1 masked the *whole* answer, which collapses: with no
        answer context, the promptless teacher is near-uniform on every position, so
        every token looks prompt-bound and ``A_hat`` covers ~all of the answer (``R``
        empty, no self-distillation). Partial masks leave answer neighbours, so
        format/filler tokens are recoverable *without* the prompt (small ``s``) while
        fact-bearing tokens still need it (large ``s``) -- a genuine dynamic range.

        Selection then applies an adaptive threshold (Otsu by default, since ``s`` is
        almost always >= 0 so a fixed ``tau`` is uninformative) and subtracts answer
        tokens that merely copy a token from the question -- the prompt already
        reveals that surface form, so it is not secret to forget.
        """
        b, l = answer_mask.shape
        device = input_ids.device
        prompt = self._prompt_mask(input_ids, answer_mask, attention_mask)
        gold = input_ids.unsqueeze(-1)

        s_sum = torch.zeros(b, l, device=device)
        cnt = torch.zeros(b, l, device=device)
        for _ in range(self.sel_mc):
            t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
                b, device=device
            )
            p_mask = (1.0 - self.scheduler(t)).unsqueeze(1)  # rho, per example
            m = (torch.rand((b, l), device=device) < p_mask) & answer_mask
            if not bool(m.any()):
                continue
            y = torch.where(m, self.mask_id, input_ids)
            log_cond = (
                self._teacher_log_probs(y, attention_mask).gather(-1, gold).squeeze(-1)
            )
            y_base = torch.where(prompt, self.mask_id, y)
            log_base = (
                self._teacher_log_probs(y_base, attention_mask)
                .gather(-1, gold)
                .squeeze(-1)
            )
            s_draw = log_cond - log_base
            s_sum = s_sum + torch.where(m, s_draw, torch.zeros_like(s_draw))
            cnt = cnt + m.float()
        # Tokens never masked across draws keep s = 0 (neutral; unlikely selected).
        s = s_sum / cnt.clamp_min(1.0)

        a_hat = self._threshold_scores(s, answer_mask)
        if self.drop_question_overlap:
            a_hat = a_hat & ~self._question_overlap_mask(input_ids, answer_mask, prompt)
        return a_hat, s

    @staticmethod
    def _otsu_threshold(vals, relax=0.0):
        """1-D Otsu split of a score tensor: the threshold maximising between-class
        variance ``w0*w1*(m0-m1)^2``. Returns the threshold; caller selects ``s >= thr``.

        ``relax`` in [0, 1) loosens the cut toward the low tail so more tokens enter
        ``A_hat``: ``thr' = thr - relax * (thr - min)``. relax=0 is pure Otsu; relax->1
        pushes the threshold to the minimum (select every answer token)."""
        xs, _ = torch.sort(vals)
        n = xs.numel()
        if n < 2 or float(xs[0]) == float(xs[-1]):
            return xs[0] if n else vals.new_tensor(0.0)
        total = xs.sum()
        csum = torch.cumsum(xs, 0)[:-1]  # sum of first i+1 elems, i = 0..n-2
        cn = torch.arange(1, n, device=xs.device, dtype=xs.dtype)
        w0 = cn / n
        w1 = 1.0 - w0
        m0 = csum / cn
        m1 = (total - csum) / (n - cn)
        score = w0 * w1 * (m0 - m1) ** 2
        i = int(torch.argmax(score))
        thr = 0.5 * (xs[i] + xs[i + 1])
        if relax > 0.0:
            thr = thr - relax * (thr - xs[0])
        return thr

    def _threshold_scores(self, s, answer_mask):
        """Boolean ``A_hat`` from scores ``s`` restricted to answer positions.

        ``tau``: fixed absolute threshold. ``otsu``/``topk``: adaptive per example
        (Otsu 2-class split, or keep the top ``topk_frac`` by score).
        """
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
            else:  # otsu
                thr = self._otsu_threshold(vals, self.otsu_relax)
            a_hat[bi, idx[vals >= thr]] = True
        return a_hat

    def _question_overlap_mask(self, input_ids, answer_mask, prompt):
        """Answer tokens that reproduce a WORD/PHRASE already present in the question.

        Such tokens merely echo surface form the prompt already exposes (a name or
        term copied from ``x``), so they carry no secret to forget even when ``s_i`` is
        high (it is high precisely *because* the prompt supplies it).

        Detection is at WORD granularity, mirroring the probe: answer tokens are
        decoded and grouped into whitespace-delimited words (a token whose decoded
        surface begins with a space opens a new word), and every token of a word is
        dropped iff every alphanumeric fragment the word contains (len >= 2) is a
        question word. Grouping is essential -- a copied name is split across subword
        tokens ("Hs"+"iao" -> "Hsiao", " Yun"+"-H"+"wa" -> "Yun-Hwa") whose individual
        pieces do NOT equal a whole question word, so the earlier per-token id match
        left most of the name in ``A_hat``. Matching the reassembled word instead
        drops the whole copied span.
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

    # -- schedule-driven Bernoulli masking with a guaranteed forget hit ---------
    def _sample_mask(self, answer_mask, a_hat):
        """Answer-stratified mask over the WHOLE answer A (as in TSD's sampler).

        Draw ``k ~ Uniform{1..|A|}`` per example and mask ``k`` uniformly chosen
        answer positions, covering every level of partial answer visibility
        (``k = |A|`` is the fully masked answer). The mask spans all of A, so the
        selected set ``A_hat`` supplies forget positions (``masked cap A_hat``) and
        the complement ``A \\ A_hat`` supplies self-distillation positions -- unlike
        a mask confined to ``A_hat``, which would starve the distill term.

        If a row's draw covers no selected position, one random ``A_hat`` position
        is forced in, so every state with a nonempty ``A_hat`` trains a forget
        decision (Alg. 2). Rows with empty ``A_hat`` are left as drawn and dropped
        downstream.
        """
        b, l = answer_mask.shape
        device = answer_mask.device
        count = answer_mask.sum(-1)  # |A| per example
        # Rank answer positions by a random score; keep the top-k as masked.
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
            forced = torch.zeros_like(masked)
            forced.scatter_(-1, pick, True)
            masked = masked | (forced & need.unsqueeze(-1))
        return masked

    def _demote_mask(self, log_p0, gold):
        """Boolean set ``B_i`` of vocabulary tokens to demote at each F position.

        ``minimal`` demotes only the gold token ``y_i``. ``belief`` instead demotes the
        teacher's ``belief_top_k`` highest-probability tokens (their mass is set to 0 and
        the rest renormalised) -- gold is NOT force-added. At a fact-bearing F position
        the teacher is confident on the gold token, so the top-K already contains ``y_i``;
        belief additionally zeroes the most-likely paraphrase alternatives. ``gold`` is
        the clean id at each position, shape ``[b, l]``.
        """
        demote = torch.zeros_like(log_p0, dtype=torch.bool)
        if self.demotion == "belief":
            topk = log_p0.topk(min(self.belief_top_k, log_p0.shape[-1]), dim=-1).indices
            demote.scatter_(-1, topk, True)
        else:  # minimal
            demote.scatter_(-1, gold.unsqueeze(-1), True)
        return demote

    def _simple_tsd_loss(self, model, forget_inputs):
        prep = self._preprocess_inputs(
            {
                "input_ids": forget_inputs["input_ids"],
                "labels": forget_inputs["labels"],
                "attention_mask": forget_inputs.get("attention_mask", None),
            }
        )
        input_ids, labels = prep["input_ids"], prep["labels"]
        attention_mask = prep.get("attention_mask", None)
        answer_mask = labels != -100

        # Gold ids index the vocabulary axis; surface an out-of-range id rather than
        # silently clamping onto the last token (which would demote the wrong token).
        if bool(answer_mask.any()):
            max_gold = int(input_ids[answer_mask].max())
            if max_gold >= self.vocab_size:
                raise ValueError(
                    f"gold token id {max_gold} >= vocab_size {self.vocab_size}; "
                    "check the tokenizer / collator alignment."
                )

        # A_hat: an explicit span skips Alg. 1 (manual-span option); otherwise select
        # by prompt-dependence. Restricted to answer positions in both cases.
        provided = forget_inputs.get("answer_mask", None)
        if provided is not None:
            provided = provided.to(answer_mask.device).bool()
            if provided.shape != answer_mask.shape:
                raise ValueError(
                    "answer_mask must be pre-aligned with the preprocessed inputs "
                    f"(got {tuple(provided.shape)}, expected {tuple(answer_mask.shape)})."
                )
            a_hat = provided & answer_mask
            scores = None
        else:
            a_hat, scores = self._select_forget_positions(
                input_ids, answer_mask, attention_mask
            )

        masked = self._sample_mask(answer_mask, a_hat)
        y_m = torch.where(masked, self.mask_id, input_ids)

        outputs = self._postprocess_outputs(
            model(input_ids=y_m, attention_mask=attention_mask)
        )
        log_p = self._log_probs(outputs.logits)  # student log p_theta(.|x,y_M,M)
        log_p0 = self._teacher_log_probs(y_m, attention_mask)

        # Edited teacher target q: drop the demoted token set B_i and renormalise
        # EXACTLY over the survivors in log space (set demoted logits to -inf, subtract
        # the survivors' log-sum-exp). ``minimal`` removes only the gold token y_i;
        # ``belief`` also removes the teacher's top-K (paraphrase-leak ablation).
        # Renormalising in log space guarantees sum_v q(v) == 1 for any teacher
        # confidence, unlike the naive 1/(1 - removed) rescale which under-normalises
        # near certainty.
        gold = input_ids.unsqueeze(-1)
        demote = self._demote_mask(log_p0, input_ids)  # B_i (gold; or top-K if belief)
        log_q = log_p0.masked_fill(demote, float("-inf"))
        log_q = log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)
        q = log_q.exp()  # exactly 0 on the demoted set, sums to 1 over survivors

        # KL(q || p_theta): the demoted terms are 0 * -inf, masked to 0 before NaN.
        q_log_q = torch.where(demote, torch.zeros_like(log_q), q * log_q)
        kl_f = q_log_q.sum(-1) - (q * log_p).sum(-1)  # forget positions
        p0 = log_p0.exp()
        kl_d = (p0 * (log_p0 - log_p)).sum(-1)  # self-distillation positions

        # F = M ∩ A_hat, R = M \ A_hat (both inside the answer, since M ⊆ answer).
        f_pos = masked & a_hat
        r_pos = masked & ~a_hat
        f_cnt = f_pos.sum(-1).float()
        r_cnt = r_pos.sum(-1).float()
        loss_f = (kl_f * f_pos).sum(-1) / f_cnt.clamp_min(1)
        loss_d = (kl_d * r_pos).sum(-1) / r_cnt.clamp_min(1)
        has_r = r_cnt > 0
        loss_ex = torch.where(
            has_r, self.alpha * loss_f + (1 - self.alpha) * loss_d, loss_f
        )

        # Drop examples with no selected-and-masked position (Alg. 2 "continue").
        active = f_cnt > 0
        loss = loss_ex[active].mean() if bool(active.any()) else log_p.sum() * 0.0

        metrics = {}
        if model.training:
            with torch.no_grad():
                removed = (p0 * demote).sum(-1)  # teacher mass demoted (gold + top-K)
                gold_prob = log_p.gather(-1, gold).squeeze(-1).exp()  # student on gold
                has_f = bool(f_pos.any())
                metrics = {
                    "simple_tsd/loss": float(loss.detach()),
                    "simple_tsd/forget_kl": float(loss_f[active].mean())
                    if bool(active.any())
                    else 0.0,
                    "simple_tsd/distill_kl": float(loss_d[has_r].mean())
                    if bool(has_r.any())
                    else 0.0,
                    "simple_tsd/selected_frac": float(
                        a_hat.sum().float() / answer_mask.sum().clamp_min(1)
                    ),
                    "simple_tsd/sel_mc": float(self.sel_mc),
                    "simple_tsd/skipped_examples": float((~active).sum()),
                    "simple_tsd/mask_rate": float(
                        masked.sum().float() / answer_mask.sum().clamp_min(1)
                    ),
                    "simple_tsd/removed_mass": float(removed[f_pos].mean())
                    if has_f
                    else 0.0,
                    "simple_tsd/gold_prob_student": float(gold_prob[f_pos].mean())
                    if has_f
                    else 0.0,
                }
                if scores is not None and bool(answer_mask.any()):
                    metrics["simple_tsd/score_mean"] = float(scores[answer_mask].mean())
        return loss, outputs, metrics

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss, outputs, metrics = self._simple_tsd_loss(model, inputs["forget"])
        loss = self.gamma * loss
        if model.training:
            self._pending_metrics.update(metrics)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if model.training and self.accelerator.sync_gradients and self._pending_metrics:
            self.log(self._pending_metrics)
            self._pending_metrics = {}
        return loss
