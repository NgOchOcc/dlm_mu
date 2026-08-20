"""RWKU utility metrics for masked diffusion LMs (Dream, LLaDA).

RWKU's forget/neighbor probes reuse ``rouge_diffusion`` (RougeL recall on
generations). The three *utility* metrics need bespoke handlers because their
HF rows carry structured fields (few-shot ``examples``, ``mc1_targets``,
``answers`` lists) rather than a single question/answer pair:

- ``mmlu_accuracy_diffusion``   — few-shot multiple choice, generate the letter.
- ``truthfulqa_mc1_diffusion``  — pick the choice with the lowest per-token NLL.
- ``triviaqa_em_diffusion``     — generate a short answer, substring exact match.

Generation goes through the same ``MDLMSampler`` used by ``rouge_diffusion`` so
the decoding behaviour matches the rest of the DLM eval suite. Mirrors the MDU
reference (``scripts/eval_rwku_dream.py``).
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F

from evals.metrics.base import unlearning_metric
from evals.metrics.dlm_utils import _build_scheduler, _postprocess_logits

logger = logging.getLogger("evaluator")

MMLU_CHOICES = ["A", "B", "C", "D"]


def _build_prompt_ids(tokenizer, template_args, question):
    """Tokenize a single user turn into generation-ready prompt ids."""
    if template_args is not None and template_args.get("apply_chat_template", True):
        chat = []
        system_prompt = template_args.get("system_prompt", None)
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})
        chat.append({"role": "user", "content": question})
        ids = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True
        )
    else:
        ids = tokenizer(question, add_special_tokens=True)["input_ids"]
    return torch.tensor(ids, dtype=torch.long)


def _generate_texts(model, tokenizer, template_args, questions, gen_args,
                    scheduler_name, right_shift_logits):
    """Greedy MDLM generation; returns the decoded continuation per prompt."""
    from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
    from omegaconf import OmegaConf

    if hasattr(gen_args, "keys"):
        gen_args = OmegaConf.to_container(gen_args, resolve=True)
    max_new_tokens = gen_args.get("max_new_tokens", 64)
    cfg = MDLMSamplerConfig(
        max_new_tokens=max_new_tokens,
        steps=gen_args.get("steps", 64),
        block_size=gen_args.get("block_size", max_new_tokens),
        temperature=gen_args.get("temperature", 0.0) or 0.0,
        remasking=gen_args.get("remasking", "low_confidence"),
        right_shift_logits=right_shift_logits,
    )
    prompts = [_build_prompt_ids(tokenizer, template_args, q).to(model.device)
               for q in questions]
    sampler = MDLMSampler(
        model=model, tokenizer=tokenizer, scheduler=_build_scheduler(scheduler_name)
    )
    sequences = sampler.sample(prompts, config=cfg)
    texts = []
    for i in range(len(prompts)):
        start = prompts[i].shape[0]
        gen_ids = sequences[i][start : start + max_new_tokens]
        texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return texts


def _single_dataset(kwargs):
    """Return the metric's dataset.

    ``get_datasets`` returns the dataset object directly when a config declares
    a single dataset (as the RWKU utility configs do), so ``kwargs["data"]`` is
    already the dataset. A dict is tolerated for robustness.
    """
    data = kwargs["data"]
    if isinstance(data, dict):
        assert len(data) == 1, f"expected a single dataset, got {list(data)}"
        return next(iter(data.values()))
    return data


@unlearning_metric(name="mmlu_accuracy_diffusion")
def mmlu_accuracy_diffusion(model, **kwargs):
    """Few-shot multiple-choice accuracy (RWKU utility_general)."""
    tokenizer = kwargs["tokenizer"]
    template_args = kwargs.get("template_args", None)
    data = _single_dataset(kwargs)
    batch_size = kwargs.get("batch_size", 8)
    max_samples = kwargs.get("max_samples", 100)
    gen_args = dict(kwargs.get("generation_args", {}))
    gen_args.setdefault("max_new_tokens", 5)
    scheduler_name = kwargs.get("scheduler_name", "LinearAlphaScheduler")
    right_shift_logits = kwargs.get("right_shift_logits", False)

    n = min(max_samples, len(data)) if max_samples else len(data)
    prompts, answers, indices = [], [], []
    for idx in range(n):
        row = data[idx]
        subject = row["task"].replace("_", " ")
        text = (
            f"The following are multiple choice questions (with answers) "
            f"about {subject}.\n\n"
        )
        for ex in row["examples"]:
            text += "Question: " + ex["question"]
            for j, c in enumerate(ex["choices"]):
                text += f"\n{MMLU_CHOICES[j]}. {c}"
            text += f"\nAnswer: {MMLU_CHOICES[ex['answer']]}\n\n"
        text += "Question: " + row["question"]
        for j, c in enumerate(row["choices"]):
            text += f"\n{MMLU_CHOICES[j]}. {c}"
        text += "\nAnswer:"
        prompts.append(text)
        answers.append(row["answer"])
        indices.append(row.get("index", idx))

    scores_by_index = {}
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_ans = answers[i : i + batch_size]
        batch_idx = indices[i : i + batch_size]
        preds = _generate_texts(
            model, tokenizer, template_args, batch_prompts, gen_args,
            scheduler_name, right_shift_logits,
        )
        for pred, gt, idx in zip(preds, batch_ans, batch_idx):
            pred_choice = pred[0].upper() if pred else ""
            correct = int(pred_choice == MMLU_CHOICES[gt])
            scores_by_index[str(idx)] = {
                "score": correct, "pred": pred, "gt": MMLU_CHOICES[gt],
            }

    vals = np.array([e["score"] for e in scores_by_index.values()])
    return {
        "agg_value": float(np.mean(vals)) if len(vals) else 0.0,
        "value_by_index": scores_by_index,
    }


@unlearning_metric(name="truthfulqa_mc1_diffusion")
def truthfulqa_mc1_diffusion(model, **kwargs):
    """TruthfulQA MC1: choice with lowest per-token NLL (RWKU utility_truthfulness)."""
    tokenizer = kwargs["tokenizer"]
    template_args = kwargs.get("template_args", None)
    data = _single_dataset(kwargs)
    max_samples = kwargs.get("max_samples", None)
    right_shift_logits = kwargs.get("right_shift_logits", False)

    n = min(max_samples, len(data)) if max_samples else len(data)
    scores_by_index = {}
    for idx in range(n):
        row = data[idx]
        question = row["question"]
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        gt_idx = labels.index(1) if 1 in labels else 0
        best_nll, best_idx = float("inf"), 0
        for ci, choice in enumerate(choices):
            if template_args is not None and template_args.get("apply_chat_template", True):
                msgs = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": choice},
                ]
                ids = tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=False
                )
            else:
                ids = tokenizer(question + " " + choice, add_special_tokens=True)[
                    "input_ids"
                ]
            ids = torch.tensor(ids, dtype=torch.long, device=model.device)
            with torch.no_grad():
                logits = model(input_ids=ids.unsqueeze(0)).logits
                logits = _postprocess_logits(logits, right_shift_logits)[0]
                target = ids[1:]
                nll = F.cross_entropy(
                    logits[:-1], target, reduction="sum"
                ).item() / max(target.size(0), 1)
            if nll < best_nll:
                best_nll, best_idx = nll, ci
        scores_by_index[str(row.get("index", idx))] = {
            "score": int(best_idx == gt_idx), "pred_idx": best_idx, "gt_idx": gt_idx,
        }

    vals = np.array([e["score"] for e in scores_by_index.values()])
    return {
        "agg_value": float(np.mean(vals)) if len(vals) else 0.0,
        "value_by_index": scores_by_index,
    }


@unlearning_metric(name="triviaqa_em_diffusion")
def triviaqa_em_diffusion(model, **kwargs):
    """TriviaQA EM: generate an answer, substring-match any reference (utility_factuality)."""
    tokenizer = kwargs["tokenizer"]
    template_args = kwargs.get("template_args", None)
    data = _single_dataset(kwargs)
    batch_size = kwargs.get("batch_size", 8)
    max_samples = kwargs.get("max_samples", None)
    gen_args = dict(kwargs.get("generation_args", {}))
    gen_args.setdefault("max_new_tokens", 64)
    scheduler_name = kwargs.get("scheduler_name", "LinearAlphaScheduler")
    right_shift_logits = kwargs.get("right_shift_logits", False)

    n = min(max_samples, len(data)) if max_samples else len(data)
    rows = [data[idx] for idx in range(n)]
    scores_by_index = {}
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        preds = _generate_texts(
            model, tokenizer, template_args, [r["question"] for r in batch],
            gen_args, scheduler_name, right_shift_logits,
        )
        for row, pred in zip(batch, preds):
            pred_clean = pred.strip().lower()
            hit = int(any(a.lower() in pred_clean for a in row["answers"]))
            scores_by_index[str(row.get("index", i))] = {
                "score": hit, "pred": pred_clean,
            }

    vals = np.array([e["score"] for e in scores_by_index.values()])
    return {
        "agg_value": float(np.mean(vals)) if len(vals) else 0.0,
        "value_by_index": scores_by_index,
    }
