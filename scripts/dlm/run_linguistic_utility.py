#!/usr/bin/env python3
"""Judge TOFU retain, real-author, and world-fact generations."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evals.metrics.llm_judge_api import (  # noqa: E402
    JUDGE_MODEL,
    JUDGE_WORKERS,
    clean_text,
    discover_tofu_linguistic_targets,
    parse_linguistic,
    query_judge_many,
)

UTILITY_LINGUISTIC_PROMPT = """
Evaluate the model's response using the criteria below. Be strict, consistent, and judge each criterion independently.

### User Question
{inp}

### Model's Response
{gen}

### Ground Truth
{truth}

### Evaluation Criteria
Score each criterion from 0 to 10 using integer scores only.

1. *Fluency*: Evaluate grammar, syntax, naturalness, readability, and flow.

* 9-10: Highly natural, fluent, and grammatically correct
* 7-8: Generally fluent with minor issues
* 5-6: Understandable but noticeably awkward or imperfect
* 3-4: Frequent errors that hurt readability
* 1-2: Very difficult to understand
* 0: Incomprehensible

2. *Relevance*: Evaluate whether the response directly and appropriately answers the user's specific question.

* 9-10: Directly answers the question with highly relevant information
* 7-8: Mostly relevant with minor omissions or unnecessary details
* 5-6: Partially answers the question or is only loosely focused
* 3-4: Mostly avoids or misunderstands the question
* 1-2: Barely related to the question
* 0: Completely irrelevant

3. *Correctness*: Evaluate factual consistency with the ground truth.

* 9-10: Fully correct and consistent with the ground truth
* 7-8: Mostly correct with minor errors or omissions
* 5-6: Partially correct with meaningful errors
* 3-4: Mostly incorrect
* 1-2: Almost entirely incorrect or contradicts the ground truth
* 0: Completely wrong or opposite to the ground truth

### Evaluation Rules
* Judge each criterion independently.
* Do not lower Fluency because of factual errors.
* Do not lower Relevance solely because the response is factually incorrect.
* Evaluate Correctness by comparing the response directly with the ground truth.
* Use integer scores only.
* Keep each reason concise and specific.
* Return valid JSON only.

### Response Format

{{
  "Fluency": {{"reason": "Brief explanation.", "score": X}},
  "Relevance": {{"reason": "Brief explanation.", "score": X}},
  "Correctness": {{"reason": "Brief explanation.", "score": X}}
}}
"""

DATASETS = {
    "retain": "retain_Q_A_ROUGE",
    "ra": "ra_Q_A_ROUGE",
    "wf": "wf_Q_A_ROUGE",
}
DIMENSIONS = ("Fluency", "Relevance", "Correctness")


def output_metric(dataset, dimension):
    prefix = DATASETS[dataset].removesuffix("_ROUGE")
    return f"{prefix}_{dimension}"


def utility_prompt(question, ground_truth, generation):
    return UTILITY_LINGUISTIC_PROMPT.format(
        inp=clean_text(question),
        truth=clean_text(ground_truth),
        gen=clean_text(generation),
    )


def summary_paths(eval_dir):
    return [
        path
        for path in (
            eval_dir / "TOFU_SUMMARY.json",
            eval_dir / "TOFU_SUMMARY_noprompt.json",
        )
        if path.exists()
    ]


def expected_metrics(datasets):
    return {
        output_metric(dataset, dimension)
        for dataset in datasets
        for dimension in DIMENSIONS
    }


def is_complete(eval_dir, datasets):
    summaries = summary_paths(eval_dir)
    metrics = expected_metrics(datasets)
    return bool(summaries) and all(
        metrics.issubset(json.loads(path.read_text())) for path in summaries
    )


def load_entries(eval_dir, dataset):
    data = json.loads((eval_dir / "TOFU_EVAL.json").read_text())
    entries = data.get(DATASETS[dataset], {}).get("value_by_index", {})
    return [
        (str(index), entry)
        for index, entry in entries.items()
        if isinstance(entry, dict)
        and all(
            entry.get(field) is not None
            for field in ("input", "ground_truth", "generation")
        )
    ]


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=4) + "\n")
    os.replace(temporary, path)


def score_dataset(eval_dir, dataset, entries, progress):
    verdicts = query_judge_many(
        [
            utility_prompt(entry["input"], entry["ground_truth"], entry["generation"])
            for _, entry in entries
        ],
        parser=parse_linguistic,
        workers=JUDGE_WORKERS,
        progress=progress,
    )
    aggregates = {}
    for dimension in DIMENSIONS:
        metric = output_metric(dataset, dimension)
        scores = [None if verdict is None else verdict.get(dimension) for verdict in verdicts]
        valid_scores = [score for score in scores if score is not None]
        aggregate = float(np.mean(valid_scores)) if valid_scores else None
        aggregates[metric] = aggregate
        write_json(
            eval_dir / f"{metric}.json",
            {
                "agg_value": aggregate,
                "n_scored": len(valid_scores),
                "value_by_index": {
                    index: {"score": score, "generation": entry["generation"]}
                    for score, (index, entry) in zip(scores, entries)
                },
            },
        )
    return aggregates


def update_summaries(eval_dir, aggregates):
    scored = {metric: value for metric, value in aggregates.items() if value is not None}
    for path in summary_paths(eval_dir):
        summary = json.loads(path.read_text())
        summary.update(scored)
        write_json(path, summary)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    targets = (
        sorted(path.resolve() for path in args.paths)
        if args.paths
        else discover_tofu_linguistic_targets()
    )
    pending = [
        path
        for path in targets
        if args.overwrite or not is_complete(path, args.datasets)
    ]
    work = []
    missing = 0
    for path in pending:
        entries_by_dataset = {
            dataset: load_entries(path, dataset) for dataset in args.datasets
        }
        if all(entries_by_dataset.values()):
            work.append((path, entries_by_dataset))
        else:
            missing += 1

    calls = sum(
        len(entries)
        for _, entries_by_dataset in work
        for entries in entries_by_dataset.values()
    )
    print(
        f"targets={len(targets)} complete={len(targets) - len(pending)} "
        f"pending={len(pending)} ready={len(work)} missing_generation={missing} "
        f"calls={calls} datasets={','.join(args.datasets)} "
        f"judge={JUDGE_MODEL} workers={JUDGE_WORKERS}"
    )
    if args.list or args.dry_run:
        for path, entries_by_dataset in work:
            counts = " ".join(
                f"{dataset}={len(entries)}"
                for dataset, entries in entries_by_dataset.items()
            )
            print(f"{path.relative_to(ROOT)} {counts}")
        return
    if not work:
        return

    with tqdm(total=calls, unit="call") as progress:
        for eval_dir, entries_by_dataset in work:
            aggregates = {}
            for dataset, entries in entries_by_dataset.items():
                progress.set_description(f"{eval_dir.name[:36]}:{dataset}")
                aggregates.update(
                    score_dataset(eval_dir, dataset, entries, progress.update)
                )
            update_summaries(eval_dir, aggregates)
            print(
                f"ok {eval_dir.relative_to(ROOT)} "
                + " ".join(
                    f"{metric}={value:.4f}" if value is not None else f"{metric}=n/a"
                    for metric, value in aggregates.items()
                )
            )


if __name__ == "__main__":
    main()
