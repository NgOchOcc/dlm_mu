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
import torch
from torch.utils.data import DataLoader

from evals.metrics.utils import aggregate_to_1D, run_batchwise_evals
from evals.metrics.dlm_utils import (
    evaluate_probability_diffusion,
    eval_text_similarity_diffusion,
    tokenwise_vocab_logprobs_diffusion,
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


def _dlm_logprob_fun_args(kwargs):
    tokenizer = kwargs["tokenizer"]
    return {
        "right_shift_logits": kwargs.get("right_shift_logits", False),
        "mask_token_id": tokenizer.mask_token_id,
    }


@unlearning_metric(name="exact_memorization_diffusion")
def exact_memorization_diffusion(model, **kwargs):
    """Greedy one-shot reconstruction accuracy under a masked diffusion model.

    DLM counterpart of ``exact_memorization``: fully mask the answer, take the
    per-position argmax, and report the fraction of answer tokens recovered.
    """
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    def _exact_memorization(model, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs_diffusion(
            model, batch, return_labels=True, **_dlm_logprob_fun_args(kwargs)
        )
        em_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            valid_len = len(labels)
            if valid_len == 0:
                logger.warning(
                    "EM score for an instance is marked None, due to "
                    "tokenization issues that resulted in no valid target tokens."
                )
                em_batch.append({"score": None})
            else:
                preds = torch.argmax(log_probs, dim=-1)
                em_score = (preds == labels).sum() / valid_len
                em_batch.append({"score": em_score.item()})
        return em_batch

    scores_by_index = run_batchwise_evals(
        model, dataloader, _exact_memorization, {}, "Calculating DLM EM"
    )
    em_values = np.array(
        [e["score"] for e in scores_by_index.values() if e["score"] is not None]
    )
    em_values = aggregate_to_1D(em_values)
    return {"agg_value": np.mean(em_values), "value_by_index": scores_by_index}


@unlearning_metric(name="extraction_strength_diffusion")
def extraction_strength_diffusion(model, **kwargs):
    """Extraction strength under a masked diffusion model.

    DLM counterpart of ``extraction_strength``: the shortest gold prefix length
    ``k`` after which the greedy one-shot reconstruction matches the remaining
    answer suffix exactly, reported as ``1 - k/valid_len``.
    """
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    def _extraction_strength(model, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs_diffusion(
            model, batch, return_labels=True, **_dlm_logprob_fun_args(kwargs)
        )
        es_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            valid_len = len(labels)
            if valid_len == 0:
                logger.warning(
                    "ES score for an instance is marked None, due to "
                    "tokenization issues that resulted in no valid target tokens."
                )
                es_batch.append({"score": None})
                continue
            preds = torch.argmax(log_probs, dim=-1)
            k = valid_len
            for k in range(valid_len):
                if torch.equal(preds[k:], labels[k:]):
                    break
            es_batch.append({"score": 1 - (k / valid_len)})
        return es_batch

    scores_by_index = run_batchwise_evals(
        model, dataloader, _extraction_strength, {}, "Calculating DLM ES"
    )
    es_values = np.array(
        [e["score"] for e in scores_by_index.values() if e["score"] is not None]
    )
    es_values = aggregate_to_1D(es_values)
    return {"agg_value": np.mean(es_values), "value_by_index": scores_by_index}
