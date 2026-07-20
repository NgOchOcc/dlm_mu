import torch
import torch.nn.functional as F

from trainer.dlm.unlearn.grad_diff import DiffusionGradDiff


class DiffusionNullAnchor(DiffusionGradDiff):
    """Null-anchor (CFG-style) forgetting for masked diffusion LMs.

    On the noised forget sequence the policy produces conditional logits. We
    then additionally mask the *prompt* positions and run the policy again to
    get unconditional logits. Minimizing the KL from the conditional to the
    (detached) unconditional distribution pulls the answer prediction toward
    what the model would say without the question, i.e. removes the specific
    question->answer association. The retain term is the plain diffusion NLL.
    """

    def __init__(self, tau=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau

    def _null_anchor_kl(self, logits, logits_u):
        log_pc = F.log_softmax(logits / self.tau, dim=-1)
        log_pt = F.log_softmax(logits_u / self.tau, dim=-1)
        pc = log_pc.exp()
        return (pc * (log_pc - log_pt)).sum(-1)

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
        attention_mask = art["attention_mask"]
        maskable_mask = art["maskable_mask"]

        # Prompt = attended, non-answer positions.
        if attention_mask is not None:
            q_mask = (~maskable_mask) & attention_mask.bool()
        else:
            q_mask = ~maskable_mask
        noised_u = torch.where(
            q_mask, self.processing_class.mask_token_id, art["noised_input_ids"]
        )
        with torch.no_grad():
            outputs_u = model(input_ids=noised_u, attention_mask=attention_mask)
            outputs_u = self._postprocess_outputs(outputs_u)

        kl_per_token = self._null_anchor_kl(art["logits"], outputs_u.logits)
        kl_per_token = kl_per_token * art["loss_weights"] * art["masked_mask"].to(
            kl_per_token.dtype
        )
        forget_loss = self._normalize_token_nll(
            kl_per_token, maskable_mask, art["input_ids"].shape[0]
        )

        retain_loss = self.retain_nll(model, inputs["retain"])
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, art["outputs"]) if return_outputs else loss
