"""Concept-Subspace Demotion (CSD) for masked-diffusion LM unlearning.

Where ``DiffusionSimpleTSD`` demotes the gold token (and a belief top-K) in the teacher's
*output distribution* and renormalises, the ``simple_tsd_viz_target`` probe showed that
mass simply re-flows to semantic neighbours (case variants, morphology, same-field words):
token-level demotion is whack-a-mole because a concept is not a token -- it is a *subspace*
of hidden-state space that every lexical realisation reads out from. CSD therefore erases
the concept at the representation level instead of the vocabulary level.

Concept subspace (mode ``repr``). For each example we probe the frozen teacher under
``subspace_mc`` partial masks of the answer, in two conditions per draw on the same noised
answer ``y_M``:

    h0      = h_teacher(x,        y_M)   # with the question
    h0_base = h_teacher(x_empty,  y_M)   # question span blanked (the PCSS ``log_base`` state)

Each masked forget position contributes a **prompt-dependence vector** ``g = h0 - h0_base``
-- the part of the hidden state that exists only because the question is present, the
memorised association PCSS scores lifted to representation space. Stacking these vectors
across draws and forget positions and taking the top-``subspace_rank`` right singular
vectors gives a per-example orthonormal basis ``P`` for the concept subspace. A single
draw's ``g`` is rank-1 and noisy (and its rank-1 erasure reduces exactly to the promptless
target, i.e. to ``DiffusionSimpleTSDQMask``); the SVD over many draws denoises it and spans
the directions all lexical variants of the concept read from, so erasing the whole subspace
collapses them together.

Forget objective (per selected position ``A_hat``). With ``c(v) = P v`` the projection onto
the concept subspace, the student (which *does* see the question) is trained so its hidden
state ``h_s`` matches the promptless teacher inside the subspace while staying faithful to
the with-question teacher outside it:

    L_f = || P h_s - P h0_base ||^2                                   # erase: subspace
          + complement_weight * || (I - P^T P)(h_s - h0) ||^2 / d     # component -> promptless
                                                                      #  preserve the rest

The first term drives every concept-subspace component to its promptless value; the second
is a LEACE-style minimal-change term pinning the orthogonal complement to the with-question
teacher, so fluency and unrelated content are preserved. At ``subspace_rank == 1`` this is
exactly the original rank-1 CSD. Both the student and the two teacher passes request hidden
states (activations are not ZeRO-3 partitioned, so no parameter gather is needed);
``hidden_states[-1]`` is the post-final-LN state the unembedding reads, so erasing there
controls the output distribution most directly.

The retain / self-distillation branch on ``R = A \\ A_hat`` (``KL(p0(.|x,y_M) || p_theta)``)
and the whole selection / masking / weighting pipeline are inherited unchanged from the
base; only the forget-position objective is replaced.

``mode='dist'`` (project ``h0`` off the concept subspace and re-unembed, ``softmax(W_U P h0)``,
matched with a forward KL) is left as future work: it needs the unembedding matrix ``W_U``,
which is sharded under ZeRO-3 and expensive to gather per step. The ``simple_tsd_viz_target``
probe renders that projected target on a single GPU (where ``W_U`` is cheap) for pre-training
verification across subspace ranks.
"""

import torch

from trainer.dlm.unlearn.simple_tsd import DiffusionSimpleTSD


class DiffusionCSD(DiffusionSimpleTSD):
    def __init__(
        self,
        erase_layer=-1,
        complement_weight=1.0,
        min_dir_norm=1e-3,
        subspace_rank=8,
        subspace_mc=8,
        mode="repr",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if mode not in ("repr", "dist"):
            raise ValueError(f"mode must be repr|dist, got {mode!r}")
        if mode == "dist":
            raise NotImplementedError(
                "mode='dist' (W_U projection target) is future work; the unembedding is "
                "ZeRO-3 sharded. Use scripts/dlm/tofu/simple_tsd_viz_target.py "
                "(VIZ_MODE=subspace) to visualise it on a single GPU."
            )
        if self.right_shift_logits:
            raise ValueError(
                "DiffusionCSD does not support right_shift_logits: the logit shift would "
                "misalign hidden_states with the answer span. Disable it for CSD."
            )
        if float(complement_weight) < 0.0:
            raise ValueError(f"complement_weight must be >= 0, got {complement_weight}")
        if not (float(min_dir_norm) > 0.0):
            raise ValueError(f"min_dir_norm must be > 0, got {min_dir_norm}")
        if int(subspace_rank) < 1:
            raise ValueError(f"subspace_rank must be >= 1, got {subspace_rank}")
        if int(subspace_mc) < 1:
            raise ValueError(f"subspace_mc must be >= 1, got {subspace_mc}")
        self.erase_layer = int(erase_layer)
        self.complement_weight = float(complement_weight)
        self.min_dir_norm = float(min_dir_norm)
        self.subspace_rank = int(subspace_rank)
        self.subspace_mc = int(subspace_mc)
        self.mode = mode

    @torch.no_grad()
    def _teacher_hidden(self, y, attention_mask):
        """``(log p0, h)`` for the frozen teacher on ``y``: tokenizer-vocab log-probs and
        the hidden state at ``erase_layer`` (``hidden_states[-1]`` = post-final-LN input to
        the unembedding). Detached -- the concept subspace is a fixed target."""
        out = self.ref_model(
            input_ids=y, attention_mask=attention_mask, output_hidden_states=True
        )
        log_p0 = self._log_probs(self._postprocess_outputs(out).logits)
        return log_p0, out.hidden_states[self.erase_layer]

    @torch.no_grad()
    def _teacher_hidden_only(self, y, attention_mask):
        """Just the ``erase_layer`` hidden state (no unembedding) -- for subspace building."""
        out = self.ref_model(
            input_ids=y, attention_mask=attention_mask, output_hidden_states=True
        )
        return out.hidden_states[self.erase_layer]

    @torch.no_grad()
    def _build_concept_subspace(self, input_ids, attention_mask, answer_mask, a_hat):
        """Per-example orthonormal basis ``P [b, r, d]`` for the concept subspace, plus the
        actual rank per example ``[b]`` and the fraction of prompt-dependence energy the
        top-r captures (diagnostic ``[b]``).

        For ``subspace_mc`` partial answer masks we forward the frozen teacher with and
        without the question and collect ``g = h0 - h0_base`` at the forget positions
        (``A_hat``) that were masked in that draw. The stacked, norm-filtered vectors are
        reduced by SVD (float32) to their top-``subspace_rank`` right singular vectors.
        """
        b, l = answer_mask.shape
        d = None
        r = self.subspace_rank
        device = input_ids.device
        prompt = self._prompt_mask(input_ids, answer_mask, attention_mask)
        vecs = [[] for _ in range(b)]  # collected g per example

        for _ in range(self.subspace_mc):
            t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(b, device=device)
            p_mask = (1.0 - self.scheduler(t)).unsqueeze(1)  # rho, per example
            m = (torch.rand((b, l), device=device) < p_mask) & answer_mask
            fm = m & a_hat  # forget positions actually masked this draw
            if not bool(fm.any()):
                continue
            y = torch.where(m, self.mask_id, input_ids)
            h0 = self._teacher_hidden_only(y, attention_mask)
            y_base = torch.where(prompt, self.mask_id, y)
            h0_base = self._teacher_hidden_only(y_base, attention_mask)
            g = (h0 - h0_base).float()  # [b, l, d]
            if d is None:
                d = g.shape[-1]
            for bi in range(b):
                idx = fm[bi].nonzero(as_tuple=True)[0]
                if idx.numel():
                    vecs[bi].append(g[bi, idx])

        if d is None:  # no forget position was ever masked (empty A_hat everywhere)
            d = int(self.model.get_output_embeddings().weight.shape[1])
        basis = torch.zeros(b, r, d, device=device)
        ranks = torch.zeros(b, device=device)
        energy = torch.zeros(b, device=device)
        for bi in range(b):
            if not vecs[bi]:
                continue
            G = torch.cat(vecs[bi], dim=0)  # [N, d], float32
            G = G[G.norm(dim=-1) >= self.min_dir_norm]
            if G.numel() == 0:
                continue
            # Right singular vectors are the principal concept directions; S their energy.
            _, S, Vh = torch.linalg.svd(G, full_matrices=False)
            k = min(r, Vh.shape[0])
            basis[bi, :k] = Vh[:k].to(basis.dtype)
            ranks[bi] = k
            s2 = (S ** 2)
            energy[bi] = float(s2[:k].sum() / s2.sum().clamp_min(1e-12))
        return basis, ranks, energy

    def _concept_repr_loss(self, h_s, h0, h0_base, answer_mask, basis, basis_rank):
        """Per-position subspace-erasure loss ``[b, l]`` and the erased prompt-dependence
        magnitude (diagnostic ``removed``). ``h_s`` carries the student graph; ``h0`` /
        ``h0_base`` are the detached with-/without-question teacher states; ``basis`` is the
        orthonormal concept basis ``[b, r, d]`` (zero-padded rows contribute nothing, so
        ``subspace_rank == 1`` reduces to the original rank-1 objective).
        """
        # Projections onto the concept subspace: c(v) = P v, shape [b, l, r].
        basis = basis.to(h_s.dtype)  # SVD basis is float32; match the states' dtype
        c_s = torch.einsum("brd,bld->blr", basis, h_s)
        c_base = torch.einsum("brd,bld->blr", basis, h0_base)
        erase = ((c_s - c_base) ** 2).sum(-1)  # || P(h_s - h0_base) ||^2  [b, l]

        diff = h_s - h0                          # match with-question teacher off-subspace
        c_diff = torch.einsum("brd,bld->blr", basis, diff)
        proj = torch.einsum("brd,blr->bld", basis, c_diff)  # P^T P diff
        comp = diff - proj
        d = h_s.shape[-1]
        preserve = (comp ** 2).sum(-1) / d       # [b, l], normalised by hidden dim

        valid = answer_mask & (basis_rank > 0).unsqueeze(-1)  # [b, l]
        kl_f = (erase + self.complement_weight * preserve) * valid.to(erase.dtype)

        # Diagnostic: how far the with-question teacher sits from promptless inside the
        # subspace (the displacement the student is asked to erase), where erased.
        with torch.no_grad():
            c0 = torch.einsum("brd,bld->blr", basis, h0)
            removed = ((c0 - c_base) ** 2).sum(-1).sqrt() * valid.to(h0.dtype)
        return kl_f, removed

    def _forget_context(self, forget_inputs):
        """Base context plus the per-example concept subspace, built once per batch off the
        per-draw path (like the base selection). Adds ``concept_basis``/``basis_rank``.
        """
        ctx = super()._forget_context(forget_inputs)
        basis, ranks, energy = self._build_concept_subspace(
            ctx["input_ids"], ctx["attention_mask"], ctx["answer_mask"], ctx["a_hat"]
        )
        ctx["concept_basis"] = basis
        ctx["basis_rank"] = ranks
        ctx["basis_energy"] = energy
        return ctx

    def _draw_loss(self, model, ctx, draw_idx):
        """Single mask draw with a hidden-state forget objective. Mirrors the base draw
        (same masking, teacher log-probs for the retain KL, same composition via
        ``_combine_draw``) but forwards the student and both teacher conditions with hidden
        states and replaces ``kl_f`` with the concept-subspace erasure loss.
        """
        input_ids = ctx["input_ids"]
        attention_mask = ctx["attention_mask"]
        answer_mask = ctx["answer_mask"]
        a_hat = ctx["a_hat"]
        cover_masks = ctx["cover_masks"]
        basis = ctx["concept_basis"]
        basis_rank = ctx["basis_rank"]

        masked = (
            cover_masks[draw_idx] if cover_masks is not None
            else self._sample_mask(answer_mask, a_hat)
        )
        y_m = torch.where(masked, self.mask_id, input_ids)

        # Student forward with hidden states (grad). Activations are not ZeRO-3 sharded.
        s_out = model(input_ids=y_m, attention_mask=attention_mask, output_hidden_states=True)
        outputs = self._postprocess_outputs(s_out)
        log_p = self._log_probs(outputs.logits)
        h_s = s_out.hidden_states[self.erase_layer]
        # The SVD basis is built in float32; match the model dtype for the einsums below.
        basis = basis.to(h_s.dtype)

        # Teacher with the question, and with the question blanked (x_empty), same y_M.
        log_p0, h0 = self._teacher_hidden(y_m, attention_mask)
        prompt = self._prompt_mask(input_ids, answer_mask, attention_mask)
        y_base = torch.where(prompt, self.mask_id, y_m)
        _, h0_base = self._teacher_hidden(y_base, attention_mask)

        kl_f, removed = self._concept_repr_loss(
            h_s, h0, h0_base, answer_mask, basis, basis_rank
        )
        kl_d = (log_p0.exp() * (log_p0 - log_p)).sum(-1)  # retain self-distillation

        # Diagnostics: subspace rank / captured energy, and how far the student's subspace
        # component still sits from the promptless target at the masked forget positions.
        with torch.no_grad():
            f_pos = masked & a_hat
            if bool(f_pos.any()):
                c_s = torch.einsum("brd,bld->blr", basis, h_s)
                c_base = torch.einsum("brd,bld->blr", basis, h0_base)
                resid = ((c_s - c_base) ** 2).sum(-1).sqrt()
                self._method_metrics = {
                    "csd/basis_rank": float(basis_rank.float().mean()),
                    "csd/captured_energy": float(ctx["basis_energy"].mean()),
                    "csd/subspace_residual": float(resid[f_pos].mean()),
                    "csd/subspace_disp": float(removed[f_pos].mean()),
                }
            else:
                self._method_metrics = None
        self._forget_diag = None  # gold-prob target metrics are meaningless for CSD

        return self._combine_draw(model, ctx, masked, outputs, log_p, kl_f, kl_d, removed)
