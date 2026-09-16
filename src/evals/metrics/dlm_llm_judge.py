"""LLM-judge metrics for masked diffusion language models (Dream, LLaDA).

Two generation-quality metrics scored by an external LLM judge over the API:

* ``naturalness_diffusion`` — rates language fluency/naturalness of the
  generated answer alone (0-5, higher = more natural human language).
* ``similarity_diffusion`` — rates semantic similarity between the generated
  answer and the ground-truth answer to be forgotten (0-5, higher = more
  similar). Lower is better for unlearning.

Both reuse the iterative-denoising generation path in
``generate_answers_diffusion`` so decoding matches ``rouge_diffusion`` exactly;
only the scoring differs. Judge API primitives live in ``llm_judge_api`` so the
standalone post-eval script shares them. Kept DLM-specific (``_diffusion``
suffix, own module) per the repo's DLM/AR separation.
"""

import logging

import numpy as np
from torch.utils.data import DataLoader

from evals.metrics.base import unlearning_metric
from evals.metrics.dlm_utils import generate_answers_diffusion
from evals.metrics.llm_judge_api import score_naturalness_many, score_similarity_many
from evals.metrics.utils import aggregate_to_1D, run_batchwise_evals

logger = logging.getLogger("evaluator")


def _judge_fun_args(kwargs):
    return {
        "tokenizer": kwargs["tokenizer"],
        "generation_args": kwargs["generation_args"],
        "scheduler_name": kwargs.get("scheduler_name", "LinearAlphaScheduler"),
        "right_shift_logits": kwargs.get("right_shift_logits", False),
    }


def _run_judge_metric(model, kwargs, batch_eval_fn, eval_msg):
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    scores_by_index = run_batchwise_evals(
        model, dataloader, batch_eval_fn, _judge_fun_args(kwargs), eval_msg
    )
    values = np.array(
        [e["score"] for e in scores_by_index.values() if e["score"] is not None]
    )
    values = aggregate_to_1D(values)
    agg_value = float(np.mean(values)) if values.size else None
    return {"agg_value": agg_value, "value_by_index": scores_by_index}


def _entry(item, score):
    return {
        "score": score,
        "input": item["input"],
        "ground_truth": item["ground_truth"],
        "generation": item["generation"],
    }


@unlearning_metric(name="naturalness_diffusion")
def naturalness_diffusion(model, **kwargs):
    """LLM-judged language naturalness (0-5) of iterative-denoising generations."""

    def _naturalness(model, batch, **fun_args):
        generations = generate_answers_diffusion(model, batch=batch, **fun_args)
        scores = score_naturalness_many([g["generation"] for g in generations])
        return [_entry(item, score) for item, score in zip(generations, scores)]

    return _run_judge_metric(model, kwargs, _naturalness, "Calculating DLM naturalness")


@unlearning_metric(name="similarity_diffusion")
def similarity_diffusion(model, **kwargs):
    """LLM-judged semantic similarity (0-5) to the ground truth; lower is better."""

    def _similarity(model, batch, **fun_args):
        generations = generate_answers_diffusion(model, batch=batch, **fun_args)
        scores = score_similarity_many(
            [(g["input"], g["ground_truth"], g["generation"]) for g in generations]
        )
        return [_entry(item, score) for item, score in zip(generations, scores)]

    return _run_judge_metric(model, kwargs, _similarity, "Calculating DLM similarity")
