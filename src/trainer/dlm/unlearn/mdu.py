import math
import torch

from trainer.dlm.unlearn.grad_diff import DiffusionGD


class DiffusionMDU(DiffusionGD):
    def __init__(self, tau=0.5, eta=0.0, kl_direction="forward", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kl_direction not in ("forward", "reverse", "js"):
            raise ValueError(f"Unknown kl_direction: {kl_direction}")
        self.tau = tau
        self.eta = eta
        self.kl_direction = kl_direction
        self.vocab_size = len(self.processing_class)
        self.ref_model = self._prepare_ref_model(self.model)

    def _null_input_ids(self, art):
        prompt_mask = ~art["maskable_mask"]
        if art["attention_mask"] is not None:
            prompt_mask = prompt_mask & art["attention_mask"].bool()
        return torch.where(
            prompt_mask, self.processing_class.mask_token_id, art["noised_input_ids"]
        )

    def _anchor_kl(self, logits_c, logits_u):
        if self.eta == 0.0:
            target = logits_u
        else:
            target = logits_u - self.eta * (logits_c - logits_u)
        log_pc = self._log_probs(logits_c)
        log_pt = self._log_probs(self.tau * target.detach()).detach()

        if self.kl_direction == "forward":
            return (log_pc.exp() * (log_pc - log_pt)).sum(-1)
        if self.kl_direction == "reverse":
            return (log_pt.exp() * (log_pt - log_pc)).sum(-1)
        log_pm = torch.logaddexp(log_pc, log_pt) - math.log(2.0)
        return 0.5 * (
            (log_pc.exp() * (log_pc - log_pm)).sum(-1) +
            (log_pt.exp() * (log_pt - log_pm)).sum(-1)
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
        art = self._sample_and_denoise(
            model,
            forget_inputs["input_ids"],
            forget_inputs["labels"],
            forget_inputs.get("attention_mask", None),
        )
        with torch.no_grad():
            anchor = self.ref_model(input_ids=self._null_input_ids(art), attention_mask=art["attention_mask"])
            anchor = self._postprocess_outputs(anchor)

        divergence = self._anchor_kl(art["logits"], anchor.logits)
        divergence = (divergence * art["loss_weights"] * art["masked_mask"].to(divergence.dtype))
        forget_loss = self._normalize_token_nll(divergence, art["maskable_mask"], art["input_ids"].shape[0])

        retain_loss = self.retain_nll(model, inputs["retain"])
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, art["outputs"]) if return_outputs else loss
