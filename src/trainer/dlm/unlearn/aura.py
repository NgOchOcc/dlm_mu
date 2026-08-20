"""AURA-DLM: retain-free association unlearning for masked diffusion LMs.

Implements ``docs/aura_dlm_iclr_algorithms.tex``. AURA removes the association
between a forget prompt ``x`` and the responses the model can still recover,
while preserving the frozen model's behaviour under a *minimal* prompt
intervention ``x^-`` (the forget prompt with its most influential small span
masked out).

Relative to its retain-free sibling GAP (``gap.py``), which masks the *whole*
prompt and self-distils a closed-form projected target, AURA changes three
things:

* **Minimal intervention.** ``x^-`` masks only the forget-key span ``s*`` -- the
  small contiguous prompt span whose removal most reduces reconstruction of the
  forget response (Alg 1). GAP masks every prompt token. The search is over the
  frozen teacher only, so ``s*`` is deterministic per example and is computed on
  first touch and cached across epochs, exactly like the reveal order in
  ``traj_rmdu.py`` (no separate precompute script or data-pipeline change).
* **Margin-hinge erase.** The association excess at a masked response position is
  ``a_i = log p_theta(z_i | x, z_t) - log p_theta0(z_i | x^-, z_t)``; the erase
  loss penalises only ``[a_i - delta]_+`` (Eq. erase). Association below the
  margin is left alone.
* **Explicit preservation.** ``KL(p_theta0(.|x^-,z_t) || p_theta(.|x^-,z_t))``
  distils the frozen teacher on the intervened prompt (Eq. preserve), keeping the
  student faithful wherever the forget key is absent.

The three distributions per step come from two student forwards batched together
-- ``p^+ = p_theta(.|x,z_t)`` and ``p^- = p_theta(.|x^-,z_t)`` -- and one frozen
teacher forward ``q = p_theta0(.|x^-,z_t)``; all share the same noised response
``z_t`` (impl. note "Batching"). Optionally the response set is augmented with
``K`` on-policy samples from the current model (``num_on_policy``); the ground
truth ``y`` is always included.

MRO note: subclasses ``DiffusionGradDiff`` for the ``gamma``/``alpha``/retain
plumbing and the ``ref_model`` helper, as GAP/MDU/TrajRMDU do. ``alpha`` defaults
to 0 (retain-free); set it and supply a retain dataset only for the retain-NLL
ablation.
"""

import torch
import torch.nn.functional as F

from trainer.dlm.unlearn.grad_diff import DiffusionGradDiff
from trainer.dlm.unlearn.rollout import denoise_logits, generate_answers


class DiffusionAURA(DiffusionGradDiff):
    def __init__(
        self,
        delta=0.0,
        lam=1.0,
        l_max=4,
        span_samples=4,
        max_candidates=64,
        num_on_policy=0,
        on_policy_steps=32,
        rollout_temperature=1.0,
        span_chunk_size=64,
        cache_spans=True,
        alpha=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(alpha=alpha, *args, **kwargs)
        if lam < 0.0:
            raise ValueError(f"lam (preservation weight) must be >= 0, got {lam}")
        if l_max < 1:
            raise ValueError(f"l_max must be a positive span length, got {l_max}")
        if span_samples < 1:
            raise ValueError(f"span_samples (R) must be >= 1, got {span_samples}")
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {max_candidates}")
        if num_on_policy < 0:
            raise ValueError(f"num_on_policy (K) must be >= 0, got {num_on_policy}")
        self.delta = delta
        self.lam = lam
        self.l_max = l_max
        self.span_samples = span_samples
        self.max_candidates = max_candidates
        self.num_on_policy = num_on_policy
        self.on_policy_steps = on_policy_steps
        self.rollout_temperature = rollout_temperature
        self.span_chunk_size = span_chunk_size
        self.cache_spans = cache_spans
        self.mask_id = self.processing_class.mask_token_id
        self.vocab_size = len(self.processing_class)
        self.ref_model = self._prepare_ref_model(self.model)
        self._span_cache = {}
        self._pending_metrics = {}

    # -- cross-rank coordination ------------------------------------------------
    def _sync_max(self, values):
        """Element-wise max of ``values`` across ranks.

        Every frozen forward the span search issues is a ZeRO-3 collective, so the
        decision of whether to run the search and how many candidate columns to
        pad to must be identical on every rank even though each rank's batch and
        cache state differ (same rationale as ``traj_rmdu.py::_sync_max``).
        """
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return values
        buf = torch.tensor(values, device=self.accelerator.device, dtype=torch.long)
        torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.MAX)
        return buf.tolist()

    # -- forward masking --------------------------------------------------------
    def _forward_mask(self, answer_mask):
        """Bernoulli forward-process mask over answer positions (one draw)."""
        b, l = answer_mask.shape
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=answer_mask.device
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)
        return (torch.rand((b, l), device=answer_mask.device) < p_mask) & answer_mask

    @torch.no_grad()
    def _teacher_logp_gold(self, seqs, gold, attention_mask):
        """Frozen log-prob of each ``gold`` token, chunked over the batch.

        ``seqs``/``gold`` are ``(N, l)``; returns ``(N, l)``. The chunk count is
        ``ceil(N / span_chunk_size)`` -- a function of configured sizes only, so it
        is identical across ranks and never desynchronises the collectives.
        """
        out = []
        for s in range(0, seqs.shape[0], self.span_chunk_size):
            sl = slice(s, s + self.span_chunk_size)
            am = None if attention_mask is None else attention_mask[sl]
            logits = denoise_logits(
                self.ref_model, seqs[sl], am, self.right_shift_logits, self.vocab_size
            )
            log_p = F.log_softmax(logits, dim=-1)
            g = gold[sl].clamp(max=self.vocab_size - 1).unsqueeze(-1)
            out.append(log_p.gather(-1, g).squeeze(-1))
        return torch.cat(out, dim=0)

    # -- Algorithm 1: minimal prompt intervention -------------------------------
    def _candidate_spans(self, prompt_positions):
        """Contiguous prompt spans of length ``1..l_max`` as absolute-index lists.

        Capped at ``max_candidates`` by an even stride so the candidate count is
        bounded regardless of prompt length.
        """
        positions = prompt_positions.tolist()
        n = len(positions)
        spans = [
            positions[s : s + length]
            for length in range(1, self.l_max + 1)
            for s in range(0, n - length + 1)
        ]
        if len(spans) > self.max_candidates:
            stride = len(spans) / self.max_candidates
            spans = [spans[int(i * stride)] for i in range(self.max_candidates)]
        return spans

    @torch.no_grad()
    def _compute_spans(self, input_ids, answer_mask, attention_mask, num_cols):
        """Argmax-``A(s)/|s|`` intervention mask per row (Eq. keyspan), ``(b, l)``.

        For each example, candidate spans are scored by the drop in the frozen
        teacher's reconstruction of the gold answer when the span is masked from
        the prompt, averaged over ``span_samples`` forward-noised responses -- the
        estimator ``A(s)`` of the proposal. ``num_cols`` is the rank-synced number
        of candidate columns; padding columns are marked invalid and excluded.
        """
        b, l = input_ids.shape
        device = input_ids.device
        prompt_mask = ~answer_mask
        if attention_mask is not None:
            prompt_mask = prompt_mask & attention_mask.bool()

        cand_mask = torch.zeros(b, num_cols, l, dtype=torch.bool, device=device)
        cand_len = torch.ones(b, num_cols, device=device)
        valid = torch.zeros(b, num_cols, dtype=torch.bool, device=device)
        for i in range(b):
            spans = self._candidate_spans(prompt_mask[i].nonzero(as_tuple=True)[0])
            for j, idxs in enumerate(spans[:num_cols]):
                cand_mask[i, j, idxs] = True
                cand_len[i, j] = len(idxs)
                valid[i, j] = True

        # A(s): mean over span_samples of the masked-position mean of
        # (log p0(y_i | x, y_t) - log p0(y_i | x_{\s}, y_t)).
        gold_flat = input_ids.repeat_interleave(num_cols, dim=0)  # (b*C, l)
        am_flat = (
            None
            if attention_mask is None
            else attention_mask.repeat_interleave(num_cols, dim=0)
        )
        score = torch.zeros(b, num_cols, device=device)
        for _ in range(self.span_samples):
            masked = self._forward_mask(answer_mask)  # M_t, (b, l)
            y_t = torch.where(masked, self.mask_id, input_ids)
            base_lp = self._teacher_logp_gold(y_t, input_ids, attention_mask)  # (b, l)

            # Same noised response, one prompt span masked per candidate column.
            interv = torch.where(
                cand_mask, self.mask_id, y_t.unsqueeze(1)
            ).reshape(b * num_cols, l)
            cand_lp = self._teacher_logp_gold(interv, gold_flat, am_flat).reshape(
                b, num_cols, l
            )

            diff = (base_lp.unsqueeze(1) - cand_lp) * masked.unsqueeze(1)
            denom = masked.sum(-1, keepdim=True).clamp_min(1)  # (b, 1)
            score += diff.sum(-1) / denom  # (b, C)
        score = score / self.span_samples

        # A(s)/|s|, invalid columns removed; ties and empty prompts -> no span.
        normed = (score / cand_len).masked_fill(~valid, float("-inf"))
        best = normed.argmax(dim=-1)  # (b,)
        any_valid = valid.any(dim=-1)
        chosen = cand_mask[torch.arange(b, device=device), best]  # (b, l)
        return chosen & any_valid.unsqueeze(-1)

    def _select_spans(self, input_ids, answer_mask, attention_mask):
        """Cached forget-key intervention mask per example, ``(b, l)`` bool.

        Deterministic-per-example under the frozen teacher, so it is computed once
        on first touch and reused across epochs. When any rank has a cache miss,
        every rank recomputes its whole batch (fixed forward count, rank-synced
        column width) to keep the ZeRO-3 collectives aligned; only genuinely new
        rows are written, so a row's span stays fixed after its first draw.
        """
        keys = [row.cpu().numpy().tobytes() for row in input_ids]
        missing = any(k not in self._span_cache for k in keys)

        prompt_mask = ~answer_mask
        if attention_mask is not None:
            prompt_mask = prompt_mask & attention_mask.bool()
        local_cols = max(
            (len(self._candidate_spans(prompt_mask[i].nonzero(as_tuple=True)[0]))
             for i in range(input_ids.shape[0])),
            default=1,
        )
        any_missing, cols = self._sync_max([int(missing), local_cols])
        cols = max(min(cols, self.max_candidates), 1)

        if any_missing:
            chosen = self._compute_spans(input_ids, answer_mask, attention_mask, cols)
            if not self.cache_spans:
                return chosen
            for key, row in zip(keys, chosen):
                self._span_cache.setdefault(key, row.cpu())

        return torch.stack([self._span_cache[k] for k in keys]).to(input_ids.device)

    # -- on-policy responses ----------------------------------------------------
    def _response_set(self, model, input_ids, answer_mask, attention_mask):
        """``{y}`` plus ``num_on_policy`` samples from the current model.

        The gold response is always first (impl. note "On-policy sampling"). Each
        sample runs a fixed number of reveal steps, so the collective count does
        not depend on the data.
        """
        responses = [input_ids]
        for _ in range(self.num_on_policy):
            z = generate_answers(
                model,
                input_ids,
                answer_mask,
                attention_mask,
                self.on_policy_steps,
                self.mask_id,
                self.right_shift_logits,
                self.vocab_size,
                temperature=self.rollout_temperature,
            )
            responses.append(z.detach())
        return responses

    # -- Algorithm 2: erase + preservation on one response ----------------------
    def _forget_terms(self, model, clean_ids, span_mask, answer_mask, attention_mask):
        """``(erase, preserve, outputs, sum_a, count)`` for one response ``z``.

        ``erase`` and ``preserve`` are reduced over the masked response positions
        ``M_t`` and averaged over active examples (the ``1/|M_t|`` of Alg 2). The
        two student conditionals are batched into a single forward.
        """
        masked = self._forward_mask(answer_mask)  # M_t
        z_t = torch.where(masked, self.mask_id, clean_ids)
        z_t_interv = torch.where(span_mask, self.mask_id, z_t)  # prompt span removed

        stu_in = torch.cat([z_t, z_t_interv], dim=0)
        stu_am = None if attention_mask is None else torch.cat(
            [attention_mask, attention_mask], dim=0
        )
        outputs = self._postprocess_outputs(
            model(input_ids=stu_in, attention_mask=stu_am)
        )
        log_p_full, log_p_interv = self._log_probs(outputs.logits).chunk(2, dim=0)

        with torch.no_grad():
            teacher = self._postprocess_outputs(
                self.ref_model(input_ids=z_t_interv, attention_mask=attention_mask)
            )
            log_q = self._log_probs(teacher.logits)  # detached (frozen teacher)

        # Erase: association excess at the gold token, hinged at the margin.
        gold = clean_ids.clamp(max=self.vocab_size - 1).unsqueeze(-1)
        a = (log_p_full.gather(-1, gold) - log_q.gather(-1, gold)).squeeze(-1)
        erase_pt = F.relu(a - self.delta)

        # Preservation: teacher-to-student KL on the intervened prompt.
        q = log_q.exp()
        preserve_pt = (q * (log_q - log_p_interv)).sum(-1)

        ones = torch.ones_like(erase_pt)
        erase = self._reduce(erase_pt, masked, ones)
        preserve = self._reduce(preserve_pt, masked, ones)
        active = masked.to(a.dtype)
        return erase, preserve, outputs, float((a * active).sum()), float(active.sum())

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
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

        span_mask = self._select_spans(input_ids, answer_mask, attention_mask)
        responses = self._response_set(model, input_ids, answer_mask, attention_mask)

        erase_sum = preserve_sum = 0.0
        a_sum = a_count = 0.0
        outputs = None
        for z in responses:
            erase, preserve, outputs, s_a, s_c = self._forget_terms(
                model, z, span_mask, answer_mask, attention_mask
            )
            erase_sum = erase_sum + erase
            preserve_sum = preserve_sum + preserve
            a_sum += s_a
            a_count += s_c
        n = len(responses)
        erase = erase_sum / n
        preserve = preserve_sum / n
        forget_loss = erase + self.lam * preserve

        retain_loss = self.optional_retain_loss(model, inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss

        if model.training:
            self._pending_metrics.update(
                {
                    "aura/erase": float(erase.detach()),
                    "aura/preserve": float(preserve.detach()),
                    "aura/association": a_sum / max(a_count, 1.0),
                    "aura/span_len": float(span_mask.sum(-1).float().mean()),
                    "aura/cached_spans": float(len(self._span_cache)),
                    "aura/response_set": float(n),
                }
            )
            if not isinstance(retain_loss, float):
                self._pending_metrics["aura/retain_loss"] = float(retain_loss.detach())

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if model.training and self.accelerator.sync_gradients and self._pending_metrics:
            self.log(self._pending_metrics)
            self._pending_metrics = {}
        return loss
