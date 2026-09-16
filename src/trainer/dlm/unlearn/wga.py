import torch.nn.functional as F

from trainer.dlm.unlearn.grad_diff import DiffusionGradDiff


class DiffusionWGA(DiffusionGradDiff):
    """Weighted Gradient Ascent for masked diffusion LMs.

    On the masked answer positions each token's cross-entropy is weighted by
    ``p(gt)^beta`` (detached), up-weighting tokens the model is already
    confident about, then ascended. The retain term is the plain diffusion NLL.
    """

    def __init__(self, beta=1.0, gamma=1.0, alpha=1.0, *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, *args, **kwargs)
        self.beta = beta

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
        token_ce = F.cross_entropy(
            art["logits"].transpose(1, 2), art["input_ids"], reduction="none"
        )
        weight_ce = ((-token_ce).exp().detach()) ** self.beta
        masked = art["masked_mask"].to(token_ce.dtype)
        forget_loss = -(weight_ce * token_ce * masked).sum() / masked.sum().clamp_min(1)

        retain_loss = self.optional_retain_loss(model, inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, art["outputs"]) if return_outputs else loss
