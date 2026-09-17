"""Selective unlearning with offline Monte Carlo query mask selection (SDU).

``DiffusionSDU`` behaves exactly like ``DiffusionSelectiveUnlearning`` — same
query/answer masking roles, same counterfactual-teacher construction, same
selective-KL/preserve loss split — but on the query side it replaces the
hand-written ``is_entity_surface`` heuristic with a fixed selection computed
offline (``scripts/dlm/sdu_precompute.py``) via Monte Carlo occlusion
importance (``trainer.dlm.unlearn.query_importance``). The answer side is
unchanged: ``answer_entity_mask`` is inherited as-is from
``DiffusionSelectiveUnlearning``, still driven by ``is_entity_surface``.

The cache is keyed by the dataset's own stable per-example ``"index"`` field
(``data.qa.QADataset``), which requires training with a collator that emits
it (``configs/collator/DataCollatorForSupervisedDatasetwithIndex.yaml``). The
cache stores WORD ranks (not raw token positions): at train time this
trainer re-derives the same whitespace-delimited word groups
(``build_word_groups``) that ``sdu_precompute.py`` used, then marks positions
belonging to a cached word index.
"""

import json

import torch

from trainer.dlm.unlearn.query_importance import build_word_groups
from trainer.dlm.unlearn.selective_unlearning import DiffusionSelectiveUnlearning


def _load_cache(mask_cache_path):
    with open(mask_cache_path) as f:
        raw = json.load(f)
    entries = raw["entries"]
    return {int(idx): set(entry["query"]) for idx, entry in entries.items()}


class DiffusionSDU(DiffusionSelectiveUnlearning):
    """Selective unlearning whose query entity mask comes from an offline MC cache."""

    def __init__(self, mask_cache_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sdu_cache = _load_cache(mask_cache_path)
        self._current_index = None

    def sample_selective_state(self, model, forget):
        if "index" not in forget:
            raise ValueError(
                "DiffusionSDU requires an index-emitting collator "
                "(configs/collator/DataCollatorForSupervisedDatasetwithIndex.yaml)"
            )
        self._current_index = forget["index"]
        return super().sample_selective_state(model, forget)

    def query_entity_mask(self, input_ids, answer_mask, attention_mask):
        """Query positions selected offline as leaking the memorized answer."""
        tokenizer = self.processing_class
        prompt_mask = self.prompt_mask(input_ids, answer_mask, attention_mask)
        entity_mask = torch.zeros_like(prompt_mask)
        for batch_index in range(input_ids.shape[0]):
            dataset_index = int(self._current_index[batch_index])
            selected = self._sdu_cache.get(dataset_index)
            if selected is None:
                raise KeyError(
                    f"DiffusionSDU mask cache has no entry for dataset index "
                    f"{dataset_index}; rerun scripts/dlm/sdu_precompute.py to "
                    "cover this split/index range"
                )
            if not selected:
                continue
            positions = prompt_mask[batch_index].nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                continue
            words = build_word_groups(tokenizer, input_ids[batch_index], positions)
            for word_index in selected:
                if word_index >= len(words):
                    continue
                for pos in words[word_index]["positions"]:
                    entity_mask[batch_index, pos] = True
        return entity_mask & prompt_mask
