import torch.nn.functional as F

from trainer.dlm.unlearn.grad_diff import DiffusionGD


class DiffusionSimNPO(DiffusionGD):
    """Reference-free SimNPO with the masked-diffusion objective.

    Like DiffusionNPO but without a reference model: the per-example diffusion
    NLL (averaged over answer tokens) is offset by ``delta`` and passed through
    the NPO log-sigmoid. The retain term is the plain diffusion NLL.
    """

    def __init__(self, delta=0.0, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
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
        per_example_nll = self._per_example_nll(
            art["logits"],
            art["input_ids"],
            art["loss_weights"],
            art["masked_mask"],
            art["maskable_mask"],
        )
        forget_loss = per_example_nll - self.delta
        forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

        retain_loss = self.optional_retain_loss(model, inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, art["outputs"]) if return_outputs else loss
