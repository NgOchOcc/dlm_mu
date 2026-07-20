"""Masked-diffusion memorization metrics (DLM counterpart of memorization.py).

Only the model-dependent metrics need DLM-specific handlers: ``probability``
becomes ``probability_diffusion`` (MC likelihood estimate) and ``rouge``
becomes ``rouge_diffusion`` (iterative-denoising generation). The downstream
aggregation metrics (``truth_ratio``, ``probability_w_options``) operate purely
on the ``{"prob", "avg_loss"}`` payloads and are reused as-is from
``memorization.py``.
"""

import logging
import numpy as np
from torch.utils.data import DataLoader

from evals.metrics.utils import aggregate_to_1D, run_batchwise_evals
from evals.metrics.dlm_utils import (
    evaluate_probability_diffusion,
    eval_text_similarity_diffusion,
)
from evals.metrics.base import unlearning_metric

logging.getLogger("absl").setLevel(logging.WARNING)
logger = logging.getLogger("evaluator")


@unlearning_metric(name="probability_diffusion")
def probability_diffusion(model, **kwargs):
    """MC-estimated answer probability under a masked diffusion model."""
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    tokenizer = kwargs["tokenizer"]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    fun_args = {
        "scheduler_name": kwargs.get("scheduler_name", "LinearAlphaScheduler"),
        "mc_num": kwargs.get("mc_num", 128),
        "time_epsilon": kwargs.get("time_epsilon", 1e-3),
        "right_shift_logits": kwargs.get("right_shift_logits", False),
        "mask_token_id": tokenizer.mask_token_id,
        "mc_batch_size": kwargs.get("mc_batch_size", 128),
    }
    scores_by_index = run_batchwise_evals(
        model, dataloader, evaluate_probability_diffusion, fun_args, "Calculating DLM loss"
    )
    prob_values = np.array(
        [e["prob"] for e in scores_by_index.values() if e["prob"] is not None]
    )
    prob_values = aggregate_to_1D(prob_values)
    return {"agg_value": np.mean(prob_values), "value_by_index": scores_by_index}


@unlearning_metric(name="rouge_diffusion")
def rouge_diffusion(model, **kwargs):
    """ROUGE against iterative-denoising generations."""
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    generation_args = kwargs["generation_args"]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    fun_args = {
        "tokenizer": tokenizer,
        "generation_args": generation_args,
        "scheduler_name": kwargs.get("scheduler_name", "LinearAlphaScheduler"),
        "right_shift_logits": kwargs.get("right_shift_logits", False),
    }
    scores_by_index = run_batchwise_evals(
        model,
        dataloader,
        eval_text_similarity_diffusion,
        fun_args,
        "Calculating DLM text similarity",
    )
    rouge_values = np.array(
        [
            e[kwargs["rouge_type"]]
            for e in scores_by_index.values()
            if e[kwargs["rouge_type"]] is not None
        ]
    )
    rouge_values = aggregate_to_1D(rouge_values)
    return {"agg_value": np.mean(rouge_values), "value_by_index": scores_by_index}
