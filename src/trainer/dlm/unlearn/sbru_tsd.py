"""SBRU on the SimpleTSD selection pipeline (SBRU-TSD) for masked-diffusion LM unlearning.

Combines two mechanisms we already have:

* the ``DiffusionSimpleTSD`` **position selection + self-distillation** pipeline --
  PCSS scores each answer token by prompt-dependence, keeps the forget set ``A_hat``
  by an adaptive threshold, and *self-distils the teacher* on the non-selected answer
  tokens ``R = M \\ A_hat``. That self-distillation is the retain-free utility guard:
  fluency and unrelated content survive without ever touching the retain corpus.
* the ``DiffusionSBRU`` **soft-embedding repulsion** as the forget objective, replacing
  SimpleTSD's demote-the-gold-token KL. At the selected forget positions the model's
  soft predicted embedding ``e_hat = E^T p_theta = sum_v p_theta(v) E_v`` is pushed out
  of a bounded kernel ball around the forget centre and attracted toward the promptless
  anchor.

Per selected forget position (``F = M ∩ A_hat``) the objective is the bounded
    kl_f = k_rep(e_hat, c)  -  gamma_anc * k_sigma_anc(e_hat, a)          in [-gamma_anc, 1]
with an RBF kernel, forget centre ``c`` (the gold embedding ``E[y_f]`` by default, or the
frozen teacher's soft embedding under ``centre_mode="ref_soft"``) and promptless anchor
``a = E^T p_theta0(.|m, y_t)``. The repulsion gradient vanishes once ``e_hat`` is far from
``c`` relative to the bandwidth -- a built-in stopping point, so the loss is bounded and
cannot collapse.

The repulsion bandwidth follows the SVGD **median heuristic** (Liu & Wang, 2016): the
kernel is ``k(x, x') = exp(-||x - x'||^2 / h)`` with ``h = med^2 / log(n)``, where ``med``
is the median pairwise distance between the current forget-position soft embeddings
``{e_hat_i}_{i=1}^n`` in the batch, recomputed each step (``n`` = number of selected forget
positions). This adaptively matches the bandwidth to the embedding scale, so
``sum_j k(x_i, x_j) ≈ n exp(-med^2 / h) = 1``. Passing a positive float for ``sigma``
instead uses a fixed ``k_sigma(u, v) = exp(-||u - v||^2 / 2 sigma^2)``; the anchor term
always uses the fixed ``sigma_anc``. When the batch has fewer than two forget positions (or
all coincide) the heuristic degenerates and falls back to ``sigma_fallback``.

The base then composes it exactly as SimpleTSD does:
    L_ex = alpha * mean_F kl_f  +  (1 - alpha) * mean_R KL(p0(.|x,y_M) || p_theta)
so ``alpha`` trades repulsion (forget) against self-distillation (retain), and the whole
selection / cover-masking / balanced-weighting / multi-draw machinery is inherited
unchanged. Only ``_forget_kl`` is overridden; the unembedding ``E`` is gathered once per
batch in ``_forget_context`` (detached fixed geometry, ZeRO-3 safe).

Special/control tokens inherit SimpleTSD's exclusion from forget positions and are also
removed from SBRU's soft embeddings, promptless anchors, and semantic-ball neighbours.

Unlike ``DiffusionSBRU`` (which subclasses ``DiffusionGradDiff`` and relies on a retain
NLL / the promptless anchor alone for utility -- measured to collapse retain-free on
LLaDA), here the SimpleTSD self-distillation on ``R`` supplies retain-free utility
directly, which is the intended fix for that collapse.
"""

import math

import torch

from trainer.dlm.unlearn.simple_tsd import DiffusionSimpleTSD


class DiffusionSBRUTSD(DiffusionSimpleTSD):
    def __init__(
        self,
        sigma="median",
        sigma_anc=1.0,
        sigma_fallback=0.05,
        gamma_anc=1.0,
        centre_mode="gold",
        ball_k=0,
        *args,
        **kwargs,
    ):
        # Validate the cheap scalars before super().__init__ builds the frozen ref model.
        median = isinstance(sigma, str) and sigma.lower() == "median"
        if not median and not (float(sigma) > 0.0):
            raise ValueError(f"sigma must be > 0 or 'median', got {sigma!r}")
        if not (float(sigma_fallback) > 0.0):
            raise ValueError(f"sigma_fallback must be > 0, got {sigma_fallback}")
        if sigma_anc is not None and not (float(sigma_anc) > 0.0):
            raise ValueError(f"sigma_anc must be > 0, got {sigma_anc}")
        if centre_mode not in ("gold", "ref_soft"):
            raise ValueError(f"centre_mode must be gold|ref_soft, got {centre_mode!r}")
        if int(ball_k) < 0:
            raise ValueError(f"ball_k must be >= 0, got {ball_k}")
        super().__init__(*args, **kwargs)  # builds + freezes self.ref_model, sets mask/pad/vocab
        # Repulsion bandwidth: the SVGD median heuristic (h = med^2 / log n, recomputed each
        # step) when sigma == "median"; a fixed sigma otherwise. sigma_fallback is the fixed
        # sigma used only when the heuristic degenerates (< 2 forget positions, or all equal).
        self.median_bandwidth = median
        self.sigma = None if median else float(sigma)
        self.sigma_fallback = float(sigma_fallback)
        # Anchor bandwidth: wider than the repulsion sigma so the attraction basin still
        # reaches e_hat after repulsion pushes it off the forget centre. Always fixed.
        self.sigma_anc = (
            float(sigma_anc)
            if sigma_anc is not None
            else (self.sigma if self.sigma is not None else self.sigma_fallback)
        )
        self.gamma_anc = float(gamma_anc)
        self.centre_mode = centre_mode
        self.ball_k = int(ball_k)
        lexical_vocab_size = self.vocab_size - len(
            self._forbidden_unlearning_token_ids()
        )
        if self.ball_k > lexical_vocab_size:
            raise ValueError(
                f"ball_k={self.ball_k} exceeds lexical vocabulary size "
                f"{lexical_vocab_size}"
            )

    # -- geometry helpers (ported from DiffusionSBRU) ----------------------------
    def _gather_E(self, model):
        """The unembedding matrix ``E [V, d]``, detached (fixed geometry). Under ZeRO-3 the
        parameter is sharded, so it is materialised with ``GatheredParameters`` and cloned
        inside the context; sliced to the tokenizer vocabulary to match ``_log_probs``."""
        W = model.get_output_embeddings().weight
        if self.is_deepspeed_enabled:
            import deepspeed

            with deepspeed.zero.GatheredParameters([W], modifier_rank=None):
                return W[: self.vocab_size].detach().clone()
        return W[: self.vocab_size].detach()

    def _soft_emb(self, log_p, E):
        """Probability-weighted centroid ``e_hat = p @ E`` at every position ``[b, l, d]``,
        computed in ``E``'s dtype; callers cast to float32 for the distance/kernel."""
        return log_p.exp().to(E.dtype) @ E

    def _kernel(self, u, v, sigma=None, h=None):
        """RBF kernel reduced over ``d`` -> ``[b, l]``. With ``h`` given, the SVGD-style
        ``k(u, v) = exp(-||u - v||^2 / h)``; otherwise the fixed ``exp(-||u - v||^2 /
        2 sigma^2)`` (``sigma`` defaults to the repulsion bandwidth, the anchor passes
        ``sigma_anc``)."""
        sq = ((u - v) ** 2).sum(-1)
        if h is not None:
            return torch.exp(-sq / h)
        s = self.sigma if sigma is None else sigma
        return torch.exp(-sq / (2.0 * s**2))

    @staticmethod
    def _median_bandwidth(points):
        """SVGD median heuristic ``h = med^2 / log(n)`` for ``k(x, x') = exp(-||x - x'||^2 /
        h)``, where ``med`` is the median pairwise distance between the ``points [n, d]``.
        Returns a detached scalar tensor, or ``None`` when the heuristic degenerates
        (``n < 2`` or all points coincide -> ``med = 0``)."""
        n = points.shape[0]
        if n < 2:
            return None
        with torch.no_grad():
            med = torch.pdist(points.float()).median()
            if not (float(med) > 0.0):
                return None
            return med**2 / math.log(n)

    @torch.no_grad()
    def _ball_centre(self, gold_emb, E):
        """Semantic-ball centre: mean of the ``ball_k`` cosine-nearest neighbours of the gold
        embedding in ``E`` (``ball_k=0`` -> just the gold embedding)."""
        if self.ball_k <= 0:
            return gold_emb
        b, l, d = gold_emb.shape
        En = torch.nn.functional.normalize(E.float(), dim=-1)
        q = torch.nn.functional.normalize(gold_emb.reshape(-1, d).float(), dim=-1)
        sims = q @ En.T  # [b*l, V]
        sims[:, self._forbidden_unlearning_vocab_mask(sims.device)] = float("-inf")
        nn = sims.topk(self.ball_k, dim=-1).indices  # [b*l, k]
        centre = E[nn].float().mean(dim=1)  # [b*l, d]
        return centre.reshape(b, l, d)

    @torch.no_grad()
    def _teacher_soft_emb(self, y, attention_mask, E):
        """Frozen-teacher soft embedding on ``y`` (detached, fixed geometry target)."""
        log_p0 = self._teacher_log_probs(y, attention_mask)
        log_p0 = self._condition_on_unlearning_vocabulary(log_p0)
        return self._soft_emb(log_p0, E).float()

    # -- overridden forget objective ---------------------------------------------
    def _forget_kl(self, log_p, log_p0, y_m, ctx):
        """SBRU repulsion + anchor as the per-position forget objective on ``A_hat``.

        Returns ``(kl_f, removed)``, each ``[b, l]``. ``kl_f = L_rep - gamma_anc * L_anc``
        with the bounded kernels; ``removed`` reports ``||e_hat - c||`` (how far the student's
        soft embedding has moved off the forget centre -- the base logs its forget-position
        mean as ``removed_mass``). The base ``_combine_draw`` reduces ``kl_f`` over
        ``F = M ∩ A_hat`` and adds self-distillation KL on ``R = M \\ A_hat``.
        """
        input_ids = ctx["input_ids"]
        attention_mask = ctx["attention_mask"]
        answer_mask = ctx["answer_mask"]
        E = ctx["E"]  # [V, d], detached fixed geometry, gathered once per batch

        log_p_forget = self._condition_on_unlearning_vocabulary(log_p)
        e_hat = self._soft_emb(log_p_forget, E).float()  # [b, l, d], differentiable

        # Forget-region centre c.
        if self.centre_mode == "gold":
            c = self._ball_centre(E[input_ids], E).float()
        else:  # ref_soft: the frozen teacher's soft embedding on the same noised state
            c = self._teacher_soft_emb(y_m, attention_mask, E)

        # Promptless anchor a = E^T p_theta0(.|m, y_t): teacher with the prompt blanked.
        prompt = self._prompt_mask(input_ids, answer_mask, attention_mask)
        y_base = torch.where(prompt, self.mask_id, y_m)
        a = self._teacher_soft_emb(y_base, attention_mask, E)

        sel = ctx["a_hat"] & answer_mask
        # Repulsion bandwidth: SVGD median heuristic over the current forget-position soft
        # embeddings, recomputed each draw; fall back to the fixed sigma_fallback when the
        # batch is degenerate. h uses the exp(-||.||^2 / h) convention (h = 2 sigma^2).
        h_rep = None
        if self.median_bandwidth:
            h_rep = self._median_bandwidth(e_hat[sel])
            if h_rep is None:
                h_rep = e_hat.new_tensor(2.0 * self.sigma_fallback**2)
        L_rep = self._kernel(e_hat, c, h=h_rep)              # [b, l], in [0, 1]
        L_anc = self._kernel(e_hat, a, sigma=self.sigma_anc)  # [b, l], in (0, 1]
        kl_f = L_rep - self.gamma_anc * L_anc                 # bounded in [-gamma_anc, 1]

        with torch.no_grad():
            removed = (e_hat - c).norm(dim=-1)  # dist_c, logged as removed_mass by the base
            if bool(sel.any()):
                sigma_rep = (
                    float((h_rep / 2.0) ** 0.5)
                    if h_rep is not None
                    else float(self.sigma)
                )
                self._method_metrics = {
                    "sbru_tsd/L_rep": float(L_rep[sel].mean()),
                    "sbru_tsd/L_anc": float(L_anc[sel].mean()),
                    "sbru_tsd/dist_c": float(removed[sel].mean()),
                    "sbru_tsd/dist_a": float((e_hat - a).norm(dim=-1)[sel].mean()),
                    "sbru_tsd/sigma_rep": sigma_rep,
                }
            else:
                self._method_metrics = None
        # Gold-prob target diagnostics are meaningless for a repulsion objective.
        self._forget_diag = None
        return kl_f, removed

    def _forget_context(self, forget_inputs):
        """Base context (selection / masking) plus the detached unembedding ``E`` gathered
        once per batch, so ``_forget_kl`` reuses it across every ``loss_mc`` draw without a
        per-draw ZeRO-3 gather."""
        ctx = super()._forget_context(forget_inputs)
        ctx["E"] = self._gather_E(self.model)
        return ctx
