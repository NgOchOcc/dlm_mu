from trainer.dlm.unlearn.base import DiffusionUnlearnTrainer


class DiffusionGradAscent(DiffusionUnlearnTrainer):
    """Gradient ascent on the masked-diffusion forget loss."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_loss, forget_outputs = self.forget_nll(
            model, inputs["forget"], return_outputs=True
        )
        loss = -forget_loss
        return (loss, forget_outputs) if return_outputs else loss
