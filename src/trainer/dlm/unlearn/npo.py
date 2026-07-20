import torch
import torch.nn.functional as F

from trainer.dlm.unlearn.grad_diff import DiffusionGradDiff


class DiffusionNPO(DiffusionGradDiff):
    """Negative Preference Optimization with the masked-diffusion objective.

    The policy and the frozen reference model see the *same* noised forget
    sequence (shared timestep and masking), so their per-example diffusion NLLs
    are directly comparable in the NPO log-sigmoid term. The retain term is the
    plain diffusion NLL, as in ``DiffusionGradDiff``.
    """

    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.ref_model = self._prepare_ref_model(self.model)

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
        policy_nll = self._per_example_nll(
            art["logits"],
            art["input_ids"],
            art["loss_weights"],
            art["masked_mask"],
            art["maskable_mask"],
        )
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=art["noised_input_ids"],
                attention_mask=art["attention_mask"],
            )
            ref_outputs = self._postprocess_outputs(ref_outputs)
            ref_nll = self._per_example_nll(
                ref_outputs.logits,
                art["input_ids"],
                art["loss_weights"],
                art["masked_mask"],
                art["maskable_mask"],
            )

        # NPO: lower policy NLL (better memorization) is penalized relative to ref.
        forget_loss = (
            -F.logsigmoid(self.beta * (policy_nll - ref_nll)).mean() * 2 / self.beta
        )

        retain_loss = self.retain_nll(model, inputs["retain"])
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, art["outputs"]) if return_outputs else loss
