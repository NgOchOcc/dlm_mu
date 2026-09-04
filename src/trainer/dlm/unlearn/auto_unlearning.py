"""Selective anti-conditional unlearning for masked-diffusion LMs.

For a shared noised answer state ``y_t``, let ``p_cond`` be the frozen teacher
with the forget prompt visible and ``p_uncond`` the same teacher with the prompt
replaced by mask tokens. For ``0 <= lam < 1``, set

    beta = lam / (1 - lam)

    delta_+(v) = max(0, log p_cond(v) - log p_uncond(v)),
    q_forget(v) = softmax(log p_cond(v) - beta * delta_+(v)).

The conditional distribution remains the target baseline, preserving the
teacher's normal behavior except where the prompt-added score applies selective
suppression. The conditional student is trained with ``KL(p_theta || q_forget)``
at masked answer positions. The method uses no retain examples or denoising NLL.
"""

import math

import torch

from trainer.dlm.unlearn.base import DiffusionUnlearnTrainer


class DiffusionAutoUnlearning(DiffusionUnlearnTrainer):
    def __init__(self, lam=0.5, *args, **kwargs):
        lam = float(lam)
        if not math.isfinite(lam) or not 0.0 <= lam < 1.0:
            raise ValueError(f"lam must be finite and satisfy 0 <= lam < 1, got {lam}")

        super().__init__(*args, **kwargs)
        self.lam = lam
        self.beta = lam / (1.0 - lam)
        self.mask_id = self.processing_class.mask_token_id
        self.pad_id = self.processing_class.pad_token_id
        self.vocab_size = len(self.processing_class)
        if self.vocab_size < 2:
            raise ValueError(
                "Auto Unlearning requires a tokenizer vocabulary with at least two tokens"
            )

        self.ref_model = self._prepare_ref_model(self.model)
        for parameter in self.ref_model.parameters():
            parameter.requires_grad_(False)

    def _prompt_mask(self, input_ids, answer_mask, attention_mask):
        """Attended, non-answer, non-padding positions replaced for ``p_uncond``."""
        prompt = ~answer_mask & (input_ids != self.pad_id)
        if attention_mask is not None:
            prompt = prompt & attention_mask.bool()
        return prompt

    @torch.no_grad()
    def _teacher_log_probs(self, input_ids, attention_mask, positions):
        outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self._postprocess_outputs(outputs).logits[positions]
        return self._log_probs(logits)

    @torch.no_grad()
    def _forget_target(self, log_p_cond, log_p_uncond):
        """Return the detached conditional target with prompt-added suppression."""
        log_p_cond = log_p_cond.float()
        prompt_added = (log_p_cond - log_p_uncond.float()).clamp_min(0.0)
        target_logits = log_p_cond - self.beta * prompt_added
        log_q = target_logits - torch.logsumexp(
            target_logits, dim=-1, keepdim=True
        )
        return log_q.detach(), log_q.exp().detach()

    @staticmethod
    def _reverse_kl(log_p, log_q):
        """Tokenwise ``KL(p || q)`` over the vocabulary dimension."""
        p = log_p.exp()
        return (p * (log_p - log_q)).sum(-1)

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

        input_ids = art["input_ids"]
        noised = art["noised_input_ids"]
        attention_mask = art["attention_mask"]
        answer_mask = art["maskable_mask"]
        masked = art["masked_mask"]
        log_p = self._log_probs(art["logits"][masked])

        prompt = self._prompt_mask(input_ids, answer_mask, attention_mask)
        promptless = torch.where(prompt, self.mask_id, noised)
        with torch.no_grad():
            log_p_cond = self._teacher_log_probs(noised, attention_mask, masked)
            log_p_uncond = self._teacher_log_probs(
                promptless, attention_mask, masked
            )
            log_q, _ = self._forget_target(log_p_cond, log_p_uncond)

        masked_kl = self._reverse_kl(log_p, log_q)
        per_token = art["logits"][..., 0].float() * 0.0
        per_token = per_token.masked_scatter(masked, masked_kl)
        loss = self._reduce(per_token, masked, torch.ones_like(per_token))
        return (loss, art["outputs"]) if return_outputs else loss
