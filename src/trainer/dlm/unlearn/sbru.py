"""Semantic-Ball Repulsion Unlearning (SBRU) for masked-diffusion LMs.

The discrete-diffusion, weight-time, embedding-space analogue of a repulsive
MMD field (see ``docs/idea.tex``). Instead of gradient ascent on the forget
token's log-probability -- which diverges and lets the freed mass leak to
synonyms -- SBRU pushes the model's *soft predicted embedding* out of a bounded
kernel ball around the forget region, while attracting it toward a generic
promptless anchor and anchoring utility with the retain NLL.

At a masked answer position ``i`` the soft predicted embedding is the
probability-weighted centroid of the token embeddings,

    e_hat_i = E^T p_theta(. | x_f, y_t) = sum_v p_theta(v) E_v   in R^d,

where ``E [V, d]`` is the (frozen-geometry) unembedding matrix. It moves
continuously and encodes *which region of meaning* the model emits, so a
synonym stays inside the ball and does not lower the loss. The objective has
three bounded terms:

    L_rep = mean_i k_sigma(e_hat_i, c_i)          in [0, 1]   (repulsion)
    L_anc = -mean_i k_sigma(e_hat_i, a_i)         in [-1, 0]  (anchor attraction)
    L_ret = diffusion-NLL on the retain set                   (utility)
    L_SBRU = gamma * (L_rep + gamma_anc * L_anc) + lam_ret * L_ret

with kernel ``k_sigma(u, v) = exp(-||u - v||^2 / 2 sigma^2) in (0, 1]``, whose
gradient ``-(1/sigma^2) k_sigma (u - v)`` vanishes once ``||u - v|| >> sigma`` --
a built-in stopping point that bounds the loss below by 0, so there is no
``-inf`` to chase and no collapse.

``c_i`` is the forget-region centre: the gold forget-token embedding
``E[y_f_i]`` (``centre_mode="gold"``, default) or the frozen reference soft
embedding ``e_hat0_i`` (``centre_mode="ref_soft"``). ``a_i`` is the promptless
unconditional soft embedding ``E^T p_theta0(. | m, y_t)`` -- the teacher with the
prompt blanked out, i.e. the generic linguistic prior region (MDU's anchor).

``E`` is a fixed geometry here: it is detached before use so the repulsion
gradient flows only into the policy distribution, never into the embeddings
themselves (which would admit the degenerate "just move the embeddings"
solution). Under ZeRO-3 the unembedding parameter is sharded, so it is gathered
with ``deepspeed.zero.GatheredParameters`` each step. The retain cross-entropy
still trains ``E`` normally through the real LM head.
"""

import torch

from trainer.dlm.unlearn.grad_diff import DiffusionGradDiff


class DiffusionSBRU(DiffusionGradDiff):
    def __init__(
        self,
        sigma=1.0,
        sigma_anc=None,
        gamma_anc=1.0,
        lam_ret=1.0,
        centre_mode="gold",
        ball_k=0,
        *args,
        **kwargs,
    ):
        # Validate cheap scalars before super().__init__ builds the frozen ref model.
        if not (float(sigma) > 0.0):
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if sigma_anc is not None and not (float(sigma_anc) > 0.0):
            raise ValueError(f"sigma_anc must be > 0, got {sigma_anc}")
        if centre_mode not in ("gold", "ref_soft"):
            raise ValueError(f"centre_mode must be gold|ref_soft, got {centre_mode!r}")
        if int(ball_k) < 0:
            raise ValueError(f"ball_k must be >= 0, got {ball_k}")
        super().__init__(*args, **kwargs)
        self.sigma = float(sigma)
        # Anchor bandwidth: wider than the repulsion sigma so the attraction basin
        # still reaches e_hat after repulsion has pushed it off the forget centre
        # (retain-free: L_anc is the only utility guard). Defaults to sigma.
        self.sigma_anc = float(sigma_anc) if sigma_anc is not None else float(sigma)
        self.gamma_anc = float(gamma_anc)
        self.lam_ret = float(lam_ret)
        self.centre_mode = centre_mode
        self.ball_k = int(ball_k)
        self.mask_id = self.processing_class.mask_token_id
        self.pad_id = self.processing_class.pad_token_id
        self.vocab_size = len(self.processing_class)
        # Frozen teacher for the promptless anchor (and the ref_soft centre).
        self.ref_model = self._prepare_ref_model(self.model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    # -- geometry helpers --------------------------------------------------------
    def _gather_E(self, model):
        """The unembedding matrix ``E [V, d]``, detached (fixed geometry).

        Under ZeRO-3 the parameter is sharded across ranks, so it is materialised
        with ``GatheredParameters`` and cloned inside the context (outside it, the
        storage reverts to the local shard). Sliced to the tokenizer vocabulary to
        match ``_log_probs``.
        """
        W = model.get_output_embeddings().weight
        if self.is_deepspeed_enabled:
            import deepspeed

            with deepspeed.zero.GatheredParameters([W], modifier_rank=None):
                return W[: self.vocab_size].detach().clone()
        return W[: self.vocab_size].detach()

    def _soft_emb(self, log_p, E):
        """Probability-weighted centroid ``e_hat = p @ E`` at every position ``[b, l, d]``.

        Computed in ``E``'s dtype (the model dtype) to match the backbone; callers
        cast to float32 for the distance/kernel.
        """
        return log_p.exp().to(E.dtype) @ E

    def _kernel(self, u, v, sigma=None):
        """``k_sigma(u, v) = exp(-||u - v||^2 / 2 sigma^2)`` reduced over ``d`` -> ``[b, l]``.

        ``sigma`` defaults to the repulsion bandwidth ``self.sigma``; the anchor term
        passes ``self.sigma_anc`` for a wider attraction basin. Dtype-agnostic; callers
        upcast the (bf16) embeddings to float32 first so the squared distance is accurate.
        """
        s = self.sigma if sigma is None else sigma
        sq = ((u - v) ** 2).sum(-1)
        return torch.exp(-sq / (2.0 * s**2))

    def _prompt_mask(self, input_ids, answer_mask, attention_mask):
        """Prompt positions to blank for the promptless anchor: attended, non-answer,
        non-pad tokens (pad == eos on LLaDA/Dream, blanking it would inject noise)."""
        prompt = ~answer_mask & (input_ids != self.pad_id)
        if attention_mask is not None:
            prompt = prompt & attention_mask.bool()
        return prompt

    @torch.no_grad()
    def _teacher_soft_emb(self, y, attention_mask, E):
        out = self.ref_model(input_ids=y, attention_mask=attention_mask)
        log_p0 = self._log_probs(self._postprocess_outputs(out).logits)
        return self._soft_emb(log_p0, E)

    @torch.no_grad()
    def _ball_centre(self, gold_emb, E):
        """Semantic-ball centre: mean of the ``ball_k`` cosine-nearest neighbours of
        the gold embedding in ``E`` (``ball_k=0`` -> just the gold embedding).

        Captures synonyms explicitly: a whole neighbourhood of meaning is repelled,
        not a single token.
        """
        if self.ball_k <= 0:
            return gold_emb
        b, l, d = gold_emb.shape
        En = torch.nn.functional.normalize(E.float(), dim=-1)
        q = torch.nn.functional.normalize(gold_emb.reshape(-1, d).float(), dim=-1)
        sims = q @ En.T  # [b*l, V]
        nn = sims.topk(self.ball_k, dim=-1).indices  # [b*l, k]
        centre = E[nn].float().mean(dim=1)  # [b*l, d]
        return centre.reshape(b, l, d)

    # -- objective ---------------------------------------------------------------
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget = inputs["forget"]
        art = self._sample_and_denoise(
            model,
            forget["input_ids"],
            forget["labels"],
            forget.get("attention_mask", None),
        )
        log_p = self._log_probs(art["logits"])  # [b, l, V], differentiable
        input_ids = art["input_ids"]
        noised = art["noised_input_ids"]
        attn = art["attention_mask"]
        masked = art["masked_mask"]
        maskable = art["maskable_mask"]
        loss_weights = art["loss_weights"]

        E = self._gather_E(model)  # [V, d], detached fixed geometry
        e_hat = self._soft_emb(log_p, E).float()  # [b, l, d], differentiable

        # Forget-region centre c.
        if self.centre_mode == "gold":
            c = self._ball_centre(E[input_ids], E).float()
        else:  # ref_soft: the frozen reference's soft embedding on the same noised state
            c = self._teacher_soft_emb(noised, attn, E).float()

        # Promptless anchor a = E^T p_theta0(. | m, y_t): teacher with the prompt blanked.
        prompt = self._prompt_mask(input_ids, maskable, attn)
        y_base = torch.where(prompt, self.mask_id, noised)
        a = self._teacher_soft_emb(y_base, attn, E).float()

        # Bounded terms, reduced (weighted mean over examples) on the masked positions.
        L_rep = self._reduce(self._kernel(e_hat, c), masked, loss_weights)
        L_anc = -self._reduce(
            self._kernel(e_hat, a, sigma=self.sigma_anc), masked, loss_weights
        )
        L_ret = self.optional_retain_loss(model, inputs)

        loss = self.gamma * (L_rep + self.gamma_anc * L_anc) + self.lam_ret * L_ret

        if model.training:
            with torch.no_grad():
                self.log(
                    {
                        "sbru/L_rep": float(L_rep),
                        "sbru/L_anc": float(L_anc),
                        "sbru/L_ret": float(L_ret),
                        "sbru/dist_c": float(
                            self._reduce(
                                (e_hat - c).norm(dim=-1), masked, loss_weights
                            )
                        ),
                        "sbru/dist_a": float(
                            self._reduce(
                                (e_hat - a).norm(dim=-1), masked, loss_weights
                            )
                        ),
                    }
                )
        return (loss, art["outputs"]) if return_outputs else loss
