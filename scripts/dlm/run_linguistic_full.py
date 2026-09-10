#!/usr/bin/env python3
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
    linguistic_prompt,
    parse_linguistic,
    query_judge_many,
)

EVAL_ROOT = ROOT / "saves/eval"
SOURCE_METRIC = "forget_Q_A_ROUGE"
METRICS = {
    "Fluency": "forget_Q_A_Fluency",
    "Relevance": "forget_Q_A_Relevance",
    "Hallucination": "forget_Q_A_Hallucination",
    "Correctness": "forget_Q_A_Correctness",
}
FORGET_ONLY_METHODS = {
    "DiffusionGA_forgetonly",
    "DiffusionGradAscent_forgetonly",
    "DiffusionNPO_forgetonly",
    "DiffusionSimNPO_forgetonly",
    "DiffusionWGA_forgetonly",
    "DiffusionBST",
    "DiffusionRDU",
    "DiffusionRepelGenDistill",
    "DiffusionTSD",
    "DiffusionADU",
}
MODELS = ("tofu_Dream-v0-Instruct-7B_", "tofu_LLaDA-8B-Instruct_")
SPLITS = ("forget10", "forget05", "forget01")


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
    paths = summary_paths(eval_dir)
    if not paths:
        return False
    return all(
        all(metric in json.loads(path.read_text()) for metric in METRICS.values())
        for path in paths
    )


def discover_targets():
    targets = set()
    for split in SPLITS:
        split_dir = EVAL_ROOT / split
        for model_prefix in MODELS:
            for method in FORGET_ONLY_METHODS:
                path = split_dir / f"{model_prefix}{method}"
                if (path / "TOFU_EVAL.json").exists():
                    targets.add(path)

    retain_for_split = {"forget10": "retain90", "forget05": "retain95", "forget01": "retain99"}
    for split in SPLITS:
        for model_prefix in MODELS:
            for suffix in (f"retain100_{split}", retain_for_split[split]):
                path = EVAL_ROOT / f"{model_prefix}{suffix}"
                if (path / "TOFU_EVAL.json").exists():
                    targets.add(path)
    return sorted(targets)


def load_entries(eval_dir):
    data = json.loads((eval_dir / "TOFU_EVAL.json").read_text())
    by_index = data.get(SOURCE_METRIC, {}).get("value_by_index", {})
    return [
        (str(index), entry)
        for index, entry in by_index.items()
        if isinstance(entry, dict) and entry.get("generation") is not None
    ]


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=4))
    os.replace(temporary, path)


def score_directory(eval_dir, entries, progress):
    verdicts = query_judge_many(
        [
            linguistic_prompt(entry["input"], entry["ground_truth"], entry["generation"])
            for _, entry in entries
        ],
        parser=parse_linguistic,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    targets = [path.resolve() for path in args.paths] if args.paths else discover_targets()
    missing = [path for path in targets if args.overwrite or not is_complete(path)]

    print(
        f"targets={len(targets)} complete={len(targets) - len(missing)} pending={len(missing)} "
        f"judge={JUDGE_MODEL} workers={JUDGE_WORKERS}"
    )
    if args.list or args.dry_run:
        for path in missing:
            print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    if args.list or args.dry_run or not missing:
        return

    work = []
    for eval_dir in missing:
        entries = load_entries(eval_dir)
        if entries:
            work.append((eval_dir, entries))
        else:
            print(f"skip {eval_dir}: no {SOURCE_METRIC} generations")

    total = sum(len(entries) for _, entries in work)
    with tqdm(total=total, unit="call") as progress:
        for eval_dir, entries in work:
            progress.set_description(eval_dir.name[:48])
            aggregates = score_directory(eval_dir, entries, progress.update)
            update_summaries(eval_dir, aggregates)
            values = " ".join(
                f"{metric.rsplit('_', 1)[-1]}={value:.4f}" if value is not None else f"{metric.rsplit('_', 1)[-1]}=n/a"
                for metric, value in aggregates.items()
            )
            print(f"ok {eval_dir.relative_to(ROOT)} {values} n={len(entries)}")


if __name__ == "__main__":
    main()
