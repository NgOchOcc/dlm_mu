"""Base class for masked-diffusion unlearning trainers.

DLM counterpart of ``trainer/unlearn/base.py``. Whereas the autoregressive
unlearning trainers get their per-example loss from ``model(**inputs).loss``,
the DLM trainers build their objective from the masked-diffusion NLL defined in
``trainer/dlm/base.py::DiffusionFinetuneTrainer``. This class only adds the
plumbing shared across the concrete methods (reference-model preparation,
DeepSpeed sharding of the reference model) and a small helper that returns the
per-example forget/retain diffusion NLL.
"""

import copy

import torch.nn.functional as F

from trainer.dlm.base import DiffusionFinetuneTrainer
from trainer.unlearn.base import UnlearnTrainer


class DiffusionUnlearnTrainer(DiffusionFinetuneTrainer, UnlearnTrainer):
    """Shared behaviour for DLM forget/retain unlearning trainers.

    MRO: ``DiffusionUnlearnTrainer -> DiffusionFinetuneTrainer ->
    UnlearnTrainer -> FinetuneTrainer``. ``compute_loss``/``prediction_step``
    resolve to the diffusion implementations, while ``_prepare_deepspeed``
    comes from ``UnlearnTrainer``.
    """

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def forget_nll(self, model, forget_inputs, return_outputs=False):
        return self.diffusion_nll(
            model,
            forget_inputs["input_ids"],
            forget_inputs["labels"],
            forget_inputs.get("attention_mask", None),
            return_outputs=return_outputs,
        )

    def retain_nll(self, model, retain_inputs, return_outputs=False):
        return self.diffusion_nll(
            model,
            retain_inputs["input_ids"],
            retain_inputs["labels"],
            retain_inputs.get("attention_mask", None),
            return_outputs=return_outputs,
        )

    def _per_example_nll(
        self, logits, input_ids, loss_weights, masked_mask, maskable_mask
    ):
        """Per-example weighted diffusion NLL, averaged over answer tokens.

        Shared by DiffusionNPO (policy vs reference) and DiffusionSimNPO
        (reference-free). Operates on artifacts from ``_sample_and_denoise`` so
        callers can compare policy/reference on an identical noised sequence.
        """
        token_nll = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)
        return token_nll.sum(-1) / maskable_mask.sum(-1).clamp_min(1)
