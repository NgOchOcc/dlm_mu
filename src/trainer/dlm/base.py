"""Masked diffusion language modeling trainer for open-unlearning.

This is the DLM counterpart of ``trainer/base.py::FinetuneTrainer``. Instead of
the autoregressive cross-entropy that HuggingFace ``Trainer`` computes from
``model(**inputs).loss``, the training objective here is the masked diffusion
loss (Sahoo et al. 2024; Nie et al. 2025): a diffusion timestep ``t`` is
sampled, answer tokens are stochastically replaced by ``[MASK]`` with
probability ``1 - alpha(t)``, and a (timestep-weighted) cross-entropy is
computed on the masked positions only.

Low-level diffusion primitives (alpha scheduler, CART loss weights, BOS
prepend for right-shifted logits) are reused from the ``dllm`` package so this
framework does not duplicate model-specific logic.
"""

import re

import torch
import torch.nn.functional as F

from trainer.base import FinetuneTrainer


class DiffusionFinetuneTrainer(FinetuneTrainer):
    """Finetune a diffusion LM with the masked diffusion objective.

    Diffusion hyperparameters are passed as ``method_args`` from the trainer
    config, mirroring how the autoregressive unlearning trainers receive
    ``gamma``/``alpha``/etc.

    Args:
        scheduler_name: Name of the alpha scheduler (e.g. ``LinearAlphaScheduler``).
        time_epsilon: Lower bound on the sampled timestep to avoid degenerate t.
        loss_weight_type: ``"scheduler"``, ``"uniform"`` or ``"cart[geo_p:0.3]"``.
        loss_norm_type: ``"token"``, ``"sequence"`` or ``"batch"`` normalization.
        right_shift_logits: Set True for Dream (predicts token at position i-1);
            False for LLaDA.
    """

    def __init__(
        self,
        scheduler_name="LinearAlphaScheduler",
        time_epsilon=1e-3,
        loss_weight_type="scheduler",
        loss_norm_type="token",
        right_shift_logits=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from dllm.core.schedulers import make_alpha_scheduler

        if not (0.0 < time_epsilon < 1.0):
            raise ValueError("time_epsilon must be in (0, 1)")
        self.scheduler = make_alpha_scheduler(scheduler_name)
        self.time_epsilon = time_epsilon
        self.loss_weight_type = loss_weight_type
        self.loss_norm_type = loss_norm_type
        self.right_shift_logits = right_shift_logits

    # -- input/output shifting for right_shift_logits models (Dream) ------------
    def _preprocess_inputs(self, inputs):
        if self.right_shift_logits:
            from dllm.utils.data import prepend_bos

            labels = inputs.get("labels", None)
            # If every sequence already starts with -100, treat as-is.
            if labels is not None and torch.all(labels[:, 0] == -100):
                return inputs
            inputs = prepend_bos(
                inputs,
                bos_token_id=self.processing_class.bos_token_id,
                label_pad_token_id=-100,
            )
        return inputs

    def _postprocess_outputs(self, outputs):
        if self.right_shift_logits:
            logits = outputs.logits
            outputs.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return outputs

    def _compute_loss_weights(self, t, input_ids, masked_mask):
        b, l = input_ids.shape
        if self.loss_weight_type.startswith("cart"):
            from dllm.pipelines.dream.trainer import cart_weight

            match = re.search(r"geo_p:(0\.\d+)", self.loss_weight_type)
            geo_p = float(match.group(1)) if match else 0.3
            return cart_weight(masked_mask, t, p=geo_p)
        elif self.loss_weight_type == "scheduler":
            return self.scheduler.weight(t).unsqueeze(1).repeat(1, l)
        elif self.loss_weight_type == "uniform":
            return torch.ones_like(input_ids, dtype=torch.float)
        raise NotImplementedError(f"Unknown loss_weight_type: {self.loss_weight_type}")

    def _sample_and_denoise(self, model, input_ids, labels, attention_mask=None):
        """Sample a diffusion timestep, mask answer tokens, and run the model.

        Returns the artifacts needed to build any masked-diffusion objective
        (finetune NLL, gradient ascent/difference, NPO, null-anchor KL). Keeping
        this separate lets the unlearning trainers reuse the exact same noising
        and forward pass (and, where needed, replay ``noised_input_ids`` through
        a reference/unconditional model).
        """
        assert self.processing_class.padding_side == "right"
        inputs = {"input_ids": input_ids, "labels": labels}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels = inputs["input_ids"], inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)

        b, l = input_ids.shape
        maskable_mask = labels != -100  # answer positions only

        # 1. Sample diffusion timesteps and derive per-token masking probability.
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)

        # 2. Stochastically mask answer tokens.
        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        # 3. Predict clean tokens from the noised input.
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)

        loss_weights = self._compute_loss_weights(t, input_ids, masked_mask)
        return {
            "outputs": outputs,
            "logits": outputs.logits,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "t": t,
            "masked_mask": masked_mask,
            "maskable_mask": maskable_mask,
            "noised_input_ids": noised_input_ids,
            "loss_weights": loss_weights,
        }

    def _normalize_token_nll(self, token_nll, maskable_mask, b):
        """Reduce a ``(b, l)`` weighted per-token NLL to a scalar loss."""
        if self.loss_norm_type == "token":
            token_nll = token_nll / maskable_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll = token_nll / (
                maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b
            )
        elif self.loss_norm_type == "batch":
            token_nll = token_nll / b
        else:
            raise ValueError(f"Invalid loss_norm_type: {self.loss_norm_type}")
        return token_nll.sum()

    def diffusion_nll(
        self, model, input_ids, labels, attention_mask=None, return_outputs=False
    ):
        """Masked diffusion negative log-likelihood on answer (label) positions.

        This is the shared building block for both finetuning and the DLM
        unlearning trainers.
        """
        art = self._sample_and_denoise(model, input_ids, labels, attention_mask)
        input_ids, logits = art["input_ids"], art["logits"]
        masked_mask, maskable_mask = art["masked_mask"], art["maskable_mask"]

        # Weighted cross-entropy on masked positions only.
        token_nll = F.cross_entropy(
            logits.transpose(1, 2), input_ids, reduction="none"
        )
        token_nll = token_nll * art["loss_weights"] * masked_mask.to(token_nll.dtype)
        loss = self._normalize_token_nll(token_nll, maskable_mask, input_ids.shape[0])

        return (loss, art["outputs"]) if return_outputs else loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss, outputs = self.diffusion_nll(
            model,
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
            return_outputs=True,
        )
        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Use the diffusion loss for evaluation, mirroring MDLMTrainer.

        Custom evaluators (see evals/) are the primary DLM evaluation path; this
        keeps the default HF eval loop functional when no evaluator is set.
        """
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)
        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()
        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().contiguous()
        return (loss.detach(), logits, labels)
