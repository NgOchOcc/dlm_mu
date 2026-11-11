"""WHP (Eldan & Russinovich 2023, "Who's Harry Potter?") for masked-diffusion LMs.

Fine-tunes on the original forget text with generic-prediction labels as
targets (scripts/dlm/whp_generic_labels.py): the student sees the usual noised
forget answer and, at every masked answer position, is trained with cross
entropy towards the generic label instead of the original token. Positions
whose label is -1 (consistency rule i) stay maskable but carry no loss; the
terminator/EOS block keeps its original targets.
"""

import json

import torch
import torch.nn.functional as F

from trainer.dlm.unlearn.base import DiffusionUnlearnTrainer


class DiffusionWHP(DiffusionUnlearnTrainer):
    def __init__(self, generic_labels_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(generic_labels_path) as f:
            self.generic_labels = {int(k): v for k, v in json.load(f)["labels"].items()}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget = inputs["forget"]
        art = self._sample_and_denoise(
            model, forget["input_ids"], forget["labels"], forget.get("attention_mask", None)
        )
        targets = art["input_ids"].clone()
        supervised = art["maskable_mask"].clone()
        starts = art["maskable_mask"].int().argmax(-1).tolist()
        for row, index in enumerate(forget["index"].tolist()):
            generic = torch.tensor(self.generic_labels[index], device=targets.device)
            span = slice(starts[row], starts[row] + len(generic))
            targets[row, span] = generic.clamp_min(0)
            supervised[row, span] &= generic >= 0

        token_nll = F.cross_entropy(art["logits"].transpose(1, 2), targets, reduction="none")
        token_nll = token_nll * art["loss_weights"] * (art["masked_mask"] & supervised).to(token_nll.dtype)
        loss = self._normalize_token_nll(token_nll, art["maskable_mask"], targets.shape[0])
        return (loss, art["outputs"]) if return_outputs else loss
