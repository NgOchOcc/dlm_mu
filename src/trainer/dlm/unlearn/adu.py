"""Auto-unlearning combined with secret-free generative distillation."""

import math

import torch
import torch.nn.functional as F

from trainer.dlm.unlearn.repel_gendistill import DiffusionRepelGenDistill


class DiffusionADU(DiffusionRepelGenDistill):
    """Combine selective anti-conditional KL with generative distillation.

    For each diffusion mask draw, the objective is

        auto_weight * L_auto + (1 - auto_weight) * L_gendistill.

    ``L_auto`` is the reverse KL target from ``DiffusionAutoUnlearning`` and
    ``L_gendistill`` is ``loss_d`` from ``DiffusionRepelGenDistill``. The RBF
    repulsion branch is intentionally disabled.
    """

    def __init__(self, lam=0.5, auto_weight=0.5, *args, **kwargs):
        lam = float(lam)
        auto_weight = float(auto_weight)
        if not math.isfinite(lam) or not 0.0 <= lam < 1.0:
            raise ValueError(f"lam must be finite and satisfy 0 <= lam < 1, got {lam}")
        if not math.isfinite(auto_weight) or not 0.0 <= auto_weight <= 1.0:
            raise ValueError(
                "auto_weight must be finite and satisfy 0 <= auto_weight <= 1, "
                f"got {auto_weight}"
            )
        if "repel_alpha" in kwargs:
            raise ValueError("DiffusionADU does not use repel_alpha; use auto_weight")

        super().__init__(repel_alpha=0.0, *args, **kwargs)
        self.lam = lam
        self.beta = lam / (1.0 - lam)
        self.auto_weight = auto_weight

    @torch.no_grad()
    def _auto_target(self, log_p_cond, log_p_uncond):
        log_p_cond = log_p_cond.float()
        prompt_added = (log_p_cond - log_p_uncond.float()).clamp_min(0.0)
        target_logits = log_p_cond - self.beta * prompt_added
        return target_logits - torch.logsumexp(target_logits, dim=-1, keepdim=True)

    @staticmethod
    def _reverse_kl(log_p, log_q):
        return (log_p.exp() * (log_p - log_q)).sum(-1)

    def _draw_loss(self, model, ctx):
        input_ids = ctx["input_ids"]
        attention_mask = ctx["attention_mask"]
        answer_mask = ctx["answer_mask"]
        a_hat = ctx["a_hat"]
        gen_ans = ctx["gen_ans"]
        tl_base = ctx["tl_base"]
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        timesteps = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            batch_size, device=device
        )
        mask_prob = 1.0 - self.scheduler(timesteps).unsqueeze(1).expand(
            batch_size, seq_len
        )
        masked = (torch.rand((batch_size, seq_len), device=device) < mask_prob) & answer_mask
        noised = torch.where(masked, self.mask_id, input_ids)

        outputs = self._postprocess_outputs(
            model(input_ids=noised, attention_mask=attention_mask)
        )
        logits = outputs.logits[..., : self.vocab_size].float()

        gendistill_loss = logits.new_zeros(())
        if self.auto_weight < 1.0 and bool(answer_mask.any()):
            if self.distill_mode == "teacher":
                top_values, top_indices = tl_base[answer_mask].topk(
                    self.distill_topk, dim=-1
                )
                target_probs = torch.nan_to_num(F.softmax(top_values, dim=-1))
                log_probs = F.log_softmax(
                    torch.nan_to_num(logits[answer_mask]), dim=-1
                )
                selected_log_probs = log_probs.gather(-1, top_indices).clamp(min=-30.0)
                per_position = -(target_probs * selected_log_probs).sum(-1)
            else:
                per_position = F.cross_entropy(
                    torch.nan_to_num(logits[answer_mask]),
                    gen_ans[answer_mask],
                    reduction="none",
                )

            if self.loss_d_ahat_weight != 1.0:
                weights = torch.where(
                    a_hat[answer_mask],
                    per_position.new_full((), self.loss_d_ahat_weight),
                    per_position.new_ones(()),
                )
                gendistill_loss = (per_position * weights).sum() / weights.sum().clamp(min=1.0)
            else:
                gendistill_loss = per_position.mean()

        auto_loss = logits.new_zeros(())
        if self.auto_weight > 0.0 and bool(masked.any()):
            prompt = self._prompt_mask(input_ids, answer_mask, attention_mask)
            promptless = torch.where(prompt, self.mask_id, noised)
            with torch.no_grad():
                log_p_cond = self._teacher_log_probs(noised, attention_mask)[masked]
                log_p_uncond = self._teacher_log_probs(promptless, attention_mask)[masked]
                log_q = self._auto_target(log_p_cond, log_p_uncond)
            log_p = F.log_softmax(logits[masked], dim=-1)
            per_token = logits[..., 0] * 0.0
            per_token = per_token.masked_scatter(
                masked, self._reverse_kl(log_p, log_q)
            )
            auto_loss = self._reduce(per_token, masked, torch.ones_like(per_token))

        loss = (
            self.auto_weight * auto_loss
            + (1.0 - self.auto_weight) * gendistill_loss
            + 0.0 * logits.sum()
        )

        metrics = {}
        if model.training:
            with torch.no_grad():
                denom = answer_mask.sum().clamp_min(1)
                metrics = {
                    "adu/loss": float(loss.detach()),
                    "adu/loss_auto": float(auto_loss.detach()),
                    "adu/loss_gendistill": float(gendistill_loss.detach()),
                    "adu/selected_frac": float(a_hat.sum().float() / denom),
                    "adu/mask_rate": float(masked.sum().float() / denom),
                }
        return loss, outputs, metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ctx = self._build_context(model, inputs["forget"])
        loss_sum, outputs, metrics = None, None, {}
        for _ in range(self.loss_mc):
            loss_i, outputs, metrics = self._draw_loss(model, ctx)
            loss_sum = loss_i if loss_sum is None else loss_sum + loss_i
        forget_loss = loss_sum / self.loss_mc

        retain_loss = self.optional_retain_loss(model, inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        if model.training:
            metrics["adu/loss"] = float(forget_loss.detach())
            self._pending_metrics.update(metrics)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.loss_mc <= 1:
            loss = super().training_step(model, inputs, num_items_in_batch)
            if model.training and self.accelerator.sync_gradients and self._pending_metrics:
                self.log(self._pending_metrics)
                self._pending_metrics = {}
            return loss

        model.train()
        inputs = self._prepare_inputs(inputs)
        ctx = self._build_context(model, inputs["forget"])
        grad_state = self.accelerator.gradient_state
        step_sync = self.accelerator.sync_gradients
        total = None
        last_metrics = {}
        for draw_index in range(self.loss_mc):
            grad_state._set_sync_gradients(step_sync and draw_index == self.loss_mc - 1)
            with self.compute_loss_context_manager():
                loss_i, _outputs, last_metrics = self._draw_loss(model, ctx)
                loss_i = self.gamma * loss_i / self.loss_mc
            self.accelerator.backward(loss_i)
            total = loss_i.detach() if total is None else total + loss_i.detach()
        grad_state._set_sync_gradients(step_sync)

        if model.training and step_sync:
            last_metrics["adu/loss"] = float(total)
            self._pending_metrics.update(last_metrics)
            if self._pending_metrics:
                self.log(self._pending_metrics)
                self._pending_metrics = {}
        return total
