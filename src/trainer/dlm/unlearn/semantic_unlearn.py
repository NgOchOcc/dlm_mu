"""Semantic_Unlearn for masked-diffusion language models.

At each masked forget-answer position, the frozen teacher distribution is tilted
using either an RBF embedding-neighbourhood score or cosine similarity:

    q_rbf*(v) propto p_theta0(v | x_f, y_t) exp(-eta k_i(v)),
    k_i(v) = exp(-||E_v - E_yi||^2 / h_i),

    q_cos*(v) propto p_theta0(v | x_f, y_t) exp(-eta (cos(E_v, E_yi) - 1)).

The RBF bandwidth ``h_i`` is the median squared distance from the gold-token
centre to the lexical tokenizer vocabulary, divided by ``log(|V_lex|)``.
Special/control tokens are excluded from target support and target positions.
The student and teacher consume the same sampled diffusion state, and training
minimizes only ``KL(q* || p_theta)`` averaged over masked lexical answer
positions. There is no retain loss and no denoising cross-entropy term.
"""

import math

import torch
import torch.nn.functional as F

from trainer.dlm.unlearn.base import DiffusionUnlearnTrainer


class DiffusionSemanticUnlearn(DiffusionUnlearnTrainer):
    def __init__(
        self,
        eta=1.0,
        target_mode="rbf",
        sigma_fallback=0.05,
        center_chunk_size=8,
        vocab_chunk_size=4096,
        *args,
        **kwargs,
    ):
        # Validate cheap arguments before the Trainer builds the frozen teacher.
        eta = float(eta)
        target_mode = str(target_mode).lower()
        sigma_fallback = float(sigma_fallback)
        if target_mode not in {"rbf", "cosine"}:
            raise ValueError(
                f"target_mode must be one of ('rbf', 'cosine'), got {target_mode!r}"
            )
        if isinstance(center_chunk_size, bool) or int(center_chunk_size) != float(
            center_chunk_size
        ):
            raise ValueError(
                f"center_chunk_size must be a positive integer, got {center_chunk_size}"
            )
        if isinstance(vocab_chunk_size, bool) or int(vocab_chunk_size) != float(
            vocab_chunk_size
        ):
            raise ValueError(
                f"vocab_chunk_size must be a positive integer, got {vocab_chunk_size}"
            )
        center_chunk_size = int(center_chunk_size)
        vocab_chunk_size = int(vocab_chunk_size)
        if not math.isfinite(eta) or eta < 0.0:
            raise ValueError(f"eta must be finite and >= 0, got {eta}")
        if not math.isfinite(sigma_fallback) or sigma_fallback <= 0.0:
            raise ValueError(
                f"sigma_fallback must be finite and > 0, got {sigma_fallback}"
            )
        if center_chunk_size <= 0:
            raise ValueError(
                f"center_chunk_size must be a positive integer, got {center_chunk_size}"
            )
        if vocab_chunk_size <= 0:
            raise ValueError(
                f"vocab_chunk_size must be a positive integer, got {vocab_chunk_size}"
            )

        super().__init__(*args, **kwargs)
        self.eta = eta
        self.target_mode = target_mode
        self.sigma_fallback = sigma_fallback
        self.center_chunk_size = center_chunk_size
        self.vocab_chunk_size = vocab_chunk_size
        self.vocab_size = len(self.processing_class)
        lexical_vocab_size = self.vocab_size - len(
            self._forbidden_unlearning_token_ids()
        )
        if lexical_vocab_size < 2:
            raise ValueError(
                "Semantic_Unlearn requires at least two non-special tokenizer tokens"
            )

        self.ref_model = self._prepare_ref_model(self.model)
        for parameter in self.ref_model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _unwrap_embeddings_model(model):
        """Find the wrapped Hugging Face module that exposes input embeddings."""
        while not hasattr(model, "get_input_embeddings") and hasattr(model, "module"):
            model = model.module
        if not hasattr(model, "get_input_embeddings"):
            raise AttributeError("reference model does not expose get_input_embeddings()")
        return model

    def _gather_E(self):
        """Frozen teacher input embeddings ``E[:|V|]``, materialized under ZeRO-3."""
        ref_module = self._unwrap_embeddings_model(self.ref_model)
        weight = ref_module.get_input_embeddings().weight
        if self.is_deepspeed_enabled:
            import deepspeed

            with deepspeed.zero.GatheredParameters([weight], modifier_rank=None):
                return weight[: self.vocab_size].detach().clone()
        return weight[: self.vocab_size].detach()

    def _squared_distances(self, centers, E):
        """Squared distances from ``centers [C,d]`` to all ``E [V,d]`` in float32."""
        centers = centers.float()
        center_norm = centers.square().sum(-1, keepdim=True)
        chunks = []
        for start in range(0, E.shape[0], self.vocab_chunk_size):
            rows = E[start : start + self.vocab_chunk_size].float()
            distance = center_norm + rows.square().sum(-1).unsqueeze(0)
            distance = distance - 2.0 * (centers @ rows.T)
            chunks.append(distance.clamp_min_(0.0))
        return torch.cat(chunks, dim=-1)

    @torch.no_grad()
    def _semantic_kernels(self, center_ids, E):
        """Return one vocabulary-wide RBF kernel and bandwidth per center token."""
        if center_ids.ndim != 1:
            raise ValueError("center_ids must be a one-dimensional tensor")
        if center_ids.numel() == 0:
            empty = E.new_empty((0, self.vocab_size), dtype=torch.float32)
            return empty, E.new_empty((0,), dtype=torch.float32)
        if bool(((center_ids < 0) | (center_ids >= self.vocab_size)).any()):
            raise ValueError("center_ids contains an id outside the tokenizer vocabulary")

        kernels = []
        bandwidths = []
        allowed = ~self._forbidden_unlearning_vocab_mask(E.device)
        log_vocab = math.log(int(allowed.sum()))
        fallback_h = 2.0 * self.sigma_fallback**2
        for start in range(0, center_ids.numel(), self.center_chunk_size):
            ids = center_ids[start : start + self.center_chunk_size]
            distances = self._squared_distances(E[ids], E)
            median_d2 = distances[:, allowed].median(dim=-1).values
            bandwidth = median_d2 / log_vocab
            valid = torch.isfinite(bandwidth) & (bandwidth > 0.0)
            bandwidth = torch.where(
                valid,
                bandwidth,
                bandwidth.new_full(bandwidth.shape, fallback_h),
            )
            kernels.append(torch.exp(-distances / bandwidth.unsqueeze(-1)))
            bandwidths.append(bandwidth)
        return torch.cat(kernels, dim=0), torch.cat(bandwidths, dim=0)

    @torch.no_grad()
    def _cosine_scores(self, center_ids, E):
        """Return ``cos(E_v, E_center) - 1`` for each center and token."""
        if center_ids.ndim != 1:
            raise ValueError("center_ids must be a one-dimensional tensor")
        if center_ids.numel() == 0:
            return E.new_empty((0, self.vocab_size), dtype=torch.float32)
        if bool(((center_ids < 0) | (center_ids >= self.vocab_size)).any()):
            raise ValueError("center_ids contains an id outside the tokenizer vocabulary")

        centers = F.normalize(E[center_ids].float(), dim=-1)
        chunks = []
        for start in range(0, E.shape[0], self.vocab_chunk_size):
            rows = F.normalize(
                E[start : start + self.vocab_chunk_size].float(), dim=-1
            )
            cosine = (centers @ rows.T).clamp_(-1.0, 1.0)
            chunks.append(cosine - 1.0)
        return torch.cat(chunks, dim=-1)

    @torch.no_grad()
    def _semantic_scores(self, center_ids, E):
        """Return the configured vocabulary-wide score for each center token."""
        if self.target_mode == "cosine":
            return self._cosine_scores(center_ids, E)
        kernels, _ = self._semantic_kernels(center_ids, E)
        return kernels

    @torch.no_grad()
    def _tilted_target(self, log_p0, semantic_scores):
        """Normalize ``log p0 - eta * score`` and detach the resulting target."""
        log_q = log_p0.float() - self.eta * semantic_scores.float()
        log_q = self._condition_on_unlearning_vocabulary(log_q)
        q = log_q.exp()
        return log_q.detach(), q.detach()

    @torch.no_grad()
    def _semantic_target(self, log_p0, center_ids, E):
        """Construct targets for arbitrary positions; used by tests and diagnostics."""
        if log_p0.ndim != 2 or log_p0.shape[-1] != self.vocab_size:
            raise ValueError(
                f"log_p0 must have shape [N, {self.vocab_size}], got {tuple(log_p0.shape)}"
            )
        if center_ids.shape != log_p0.shape[:1]:
            raise ValueError("center_ids must contain one token id per target position")
        if not bool(self._unlearning_position_mask(center_ids).all()):
            raise ValueError("center_ids must not contain special/control token ids")
        unique_ids, inverse = torch.unique(center_ids, sorted=True, return_inverse=True)
        scores = self._semantic_scores(unique_ids, E)
        return self._tilted_target(log_p0, scores[inverse])

    def _masked_kl(self, student_logits, teacher_logits, center_ids, E):
        """Forward KL for masked positions, processed by distinct-center chunks."""
        if center_ids.numel() == 0:
            return student_logits[..., 0].float() * 0.0

        unique_ids, inverse = torch.unique(center_ids, sorted=True, return_inverse=True)
        positions = []
        losses = []
        for start in range(0, unique_ids.numel(), self.center_chunk_size):
            stop = min(start + self.center_chunk_size, unique_ids.numel())
            member = (inverse >= start) & (inverse < stop)
            position = member.nonzero(as_tuple=True)[0]
            scores = self._semantic_scores(unique_ids[start:stop], E)
            position_scores = scores[inverse[position] - start]

            log_p_theta = self._condition_on_unlearning_vocabulary(
                student_logits[position, : self.vocab_size]
            )
            with torch.no_grad():
                log_p0 = self._condition_on_unlearning_vocabulary(
                    teacher_logits[position, : self.vocab_size]
                )
                log_q, q = self._tilted_target(log_p0, position_scores)
            allowed = ~self._forbidden_unlearning_vocab_mask(log_p_theta.device)
            losses.append(
                (
                    q[..., allowed]
                    * (log_q[..., allowed] - log_p_theta[..., allowed])
                ).sum(-1)
            )
            positions.append(position)

        position = torch.cat(positions)
        loss = torch.cat(losses)
        return loss[position.argsort()]

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget = inputs["forget"]
        # This student pass samples y_t once. The frozen teacher below reuses exactly
        # those noised ids, attention mask, and Dream/LLaDA logit postprocessing.
        art = self._sample_and_denoise(
            model,
            forget["input_ids"],
            forget["labels"],
            forget.get("attention_mask", None),
        )
        with torch.no_grad():
            teacher = self._postprocess_outputs(
                self.ref_model(
                    input_ids=art["noised_input_ids"],
                    attention_mask=art["attention_mask"],
                )
            )
            E = self._gather_E()

        masked = art["masked_mask"] & self._unlearning_position_mask(
            art["input_ids"]
        )
        center_ids = art["input_ids"][masked]
        masked_kl = self._masked_kl(
            art["logits"][masked], teacher.logits[masked], center_ids, E
        )
        per_token = art["logits"][..., 0].float() * 0.0
        per_token = per_token.masked_scatter(masked, masked_kl)
        loss = self._reduce(per_token, masked, torch.ones_like(per_token))
        return (loss, art["outputs"]) if return_outputs else loss
