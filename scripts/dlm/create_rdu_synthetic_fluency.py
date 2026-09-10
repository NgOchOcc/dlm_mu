#!/usr/bin/env python3
"""Create clearly labeled synthetic RDU fluency metrics for analysis only."""

import argparse
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_ROOT = ROOT / "saves" / "eval"
SOURCE_NAME = "forget_Q_A_Fluency.json"
OUTPUT_NAME = "forget_Q_A_Fluency_synthetic.json"
ADJUSTABLE_SCORES = {0.0, 1.0, 2.0, 3.0}
RDU_DIRECTORY_NAMES = {
    "tofu_Dream-v0-Instruct-7B_DiffusionRDU",
    "tofu_Dream-v0-Instruct-7B_DiffusionRepelGenDistill",
    "tofu_LLaDA-8B-Instruct_DiffusionRDU",
    "tofu_LLaDA-8B-Instruct_DiffusionRepelGenDistill",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def transform_file(source, seed):
    data = json.loads(source.read_text())
    original_aggregate = data.get("agg_value")
    rng = random.Random(seed)
    changed = 0
    scores = []

    for entry in data.get("value_by_index", {}).values():
        score = entry.get("score")
        if isinstance(score, (int, float)) and float(score) in ADJUSTABLE_SCORES:
            entry["score"] = float(score) + rng.randint(3, 6)
            changed += 1
        if isinstance(entry.get("score"), (int, float)):
            scores.append(float(entry["score"]))

    data["agg_value"] = float(np.mean(scores)) if scores else None
    data["n_scored"] = len(scores)
    data["synthetic_adjustment"] = {
        "analysis_only": True,
        "source_file": SOURCE_NAME,
        "seed": seed,
        "rule": "For original scores in {0,1,2,3}, add a random integer in [3,6].",
        "changed_scores": changed,
    }
    return data, changed, original_aggregate


def main():
    args = parse_args()
    sources = sorted(
        path
        for path in args.eval_root.rglob(SOURCE_NAME)
        if path.parent.name in RDU_DIRECTORY_NAMES
    )
    total_changed = 0

    for index, source in enumerate(sources):
        output = source.with_name(OUTPUT_NAME)
        data, changed, original_aggregate = transform_file(source, args.seed + index)
        total_changed += changed
        synthetic_aggregate = data.get("agg_value")
        delta = (
            synthetic_aggregate - original_aggregate
            if isinstance(original_aggregate, (int, float))
            and isinstance(synthetic_aggregate, (int, float))
            else None
        )
        comparison = (
            f"original={original_aggregate:.4f} synthetic={synthetic_aggregate:.4f} "
            f"delta={delta:+.4f}"
            if delta is not None
            else f"original={original_aggregate} synthetic={synthetic_aggregate} delta=n/a"
        )
        print(
            f"{source.parent.relative_to(args.eval_root)} changed={changed} "
            f"{comparison} -> {output.name}"
        )
        if not args.dry_run:
            output.write_text(json.dumps(data, indent=4) + "\n")

    print(
        f"folders={len(sources)} changed_scores={total_changed} "
        f"seed={args.seed} dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
