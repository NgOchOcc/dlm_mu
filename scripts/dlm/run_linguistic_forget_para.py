#!/usr/bin/env python3
"""Judge TOFU paraphrased forget-question generations."""

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
    discover_tofu_linguistic_targets,
    score_linguistic_many,
)

SOURCE_METRIC = "forget_Q_A_PARA_ROUGE"
METRICS = {
    "Fluency": "forget_Q_A_PARA_Fluency",
    "Relevance": "forget_Q_A_PARA_Relevance",
    "Correctness": "forget_Q_A_PARA_Correctness",
}


def summary_paths(eval_dir):
    return [
        path
        for path in (
            eval_dir / "TOFU_SUMMARY.json",
            eval_dir / "TOFU_SUMMARY_noprompt.json",
        )
        if path.exists()
    ]


def is_complete(eval_dir):
    summaries = summary_paths(eval_dir)
    return bool(summaries) and all(
        all(metric in json.loads(path.read_text()) for metric in METRICS.values())
        for path in summaries
    )


def load_entries(eval_dir):
    data = json.loads((eval_dir / "TOFU_EVAL.json").read_text())
    entries = data.get(SOURCE_METRIC, {}).get("value_by_index", {})
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


def score_directory(eval_dir, entries, progress):
    verdicts = score_linguistic_many(
        [
            (entry["input"], entry["ground_truth"], entry["generation"])
            for _, entry in entries
        ],
        workers=JUDGE_WORKERS,
        progress=progress,
    )
    aggregates = {}
    for dimension, metric in METRICS.items():
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
    pending = [path for path in targets if args.overwrite or not is_complete(path)]
    ready = [(path, load_entries(path)) for path in pending]
    ready = [(path, entries) for path, entries in ready if entries]
    unavailable = len(pending) - len(ready)

    print(
        f"targets={len(targets)} complete={len(targets) - len(pending)} "
        f"pending={len(pending)} ready={len(ready)} missing_para_generation={unavailable} "
        f"judge={JUDGE_MODEL} workers={JUDGE_WORKERS}"
    )
    if args.list or args.dry_run:
        for path, _ in ready:
            print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
        return
    if not ready:
        raise SystemExit(
            f"No {SOURCE_METRIC} generations found. Re-run TOFU evaluation first."
        )

    with tqdm(total=sum(len(entries) for _, entries in ready), unit="call") as progress:
        for eval_dir, entries in ready:
            progress.set_description(eval_dir.name[:48])
            aggregates = score_directory(eval_dir, entries, progress.update)
            update_summaries(eval_dir, aggregates)
            values = " ".join(
                f"{dimension}={aggregates[metric]:.4f}"
                if aggregates[metric] is not None
                else f"{dimension}=n/a"
                for dimension, metric in METRICS.items()
            )
            print(f"ok {eval_dir.relative_to(ROOT)} {values} n={len(entries)}")


if __name__ == "__main__":
    main()
