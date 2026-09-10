"""Bootstrapping-Token unlearning for masked-diffusion language models."""

import math

import torch
import torch.nn.functional as F

from trainer.dlm.unlearn.base import DiffusionUnlearnTrainer


class DiffusionBST(DiffusionUnlearnTrainer):
    """Gradient ascent on a detached mixture of gold and top-k model targets."""

    def __init__(self, lambda_bst=0.5, top_k=10, *args, **kwargs):
        lambda_bst = float(lambda_bst)
        top_k = int(top_k)
        if not math.isfinite(lambda_bst) or not 0.0 <= lambda_bst <= 1.0:
            raise ValueError(
                "lambda_bst must be finite and satisfy 0 <= lambda_bst <= 1, "
                f"got {lambda_bst}"
            )
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        super().__init__(*args, **kwargs)
        self.lambda_bst = lambda_bst
        self.top_k = top_k
        self.vocab_size = len(self.processing_class)
        if self.top_k > self.vocab_size:
            raise ValueError(
                f"top_k must not exceed tokenizer vocabulary size {self.vocab_size}, "
                f"got {self.top_k}"
            )

    def _bootstrap_cross_entropy(self, logits, gold_ids):
        logits = logits[..., : self.vocab_size]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        gold_nll = -log_probs.gather(-1, gold_ids.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            values, indices = logits.float().topk(self.top_k, dim=-1)
            topk_probs = F.softmax(values, dim=-1)

        topk_nll = -(topk_probs * log_probs.gather(-1, indices)).sum(-1)
        return (
            (1.0 - self.lambda_bst) * gold_nll
            + self.lambda_bst * topk_nll
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget = inputs["forget"]
        artifacts = self._sample_and_denoise(
            model,
            forget["input_ids"],
            forget["labels"],
            forget.get("attention_mask", None),
        )
        masked = artifacts["masked_mask"]
        per_token = artifacts["logits"][..., 0].float() * 0.0
        if bool(masked.any()):
            token_loss = self._bootstrap_cross_entropy(
                artifacts["logits"][masked], artifacts["input_ids"][masked]
            )
            per_token = per_token.masked_scatter(masked, token_loss)

        weighted = per_token * artifacts["loss_weights"] * masked.to(per_token.dtype)
        forget_loss = self._normalize_token_nll(
            weighted, artifacts["maskable_mask"], artifacts["input_ids"].shape[0]
        )
        loss = -forget_loss
        return (loss, artifacts["outputs"]) if return_outputs else loss
