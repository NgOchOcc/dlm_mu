from trainer.dlm.unlearn.base import DiffusionUnlearnTrainer


class DiffusionGD(DiffusionUnlearnTrainer):
    """Gradient difference: ascend on forget, descend on retain.

    ``retain_loss_type`` only supports ``"NLL"`` for DLMs. The autoregressive
    ``"KL"`` variant relies on ``model(**inputs).loss`` / next-token logits and
    has no direct masked-diffusion analogue here.
    """

    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        if retain_loss_type != "NLL":
            raise NotImplementedError(
                f"retain_loss_type={retain_loss_type} not supported for DLM GradDiff; "
                "use 'NLL'."
            )
        self.retain_loss_type = retain_loss_type

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_loss, forget_outputs = self.forget_nll(
            model, inputs["forget"], return_outputs=True
        )
        forget_loss = -forget_loss
        retain_loss = self.optional_retain_loss(model, inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
