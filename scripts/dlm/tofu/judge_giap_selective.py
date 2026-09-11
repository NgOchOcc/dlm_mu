#!/usr/bin/env python3
"""LLM-judge the Giap SelectiveUnlearning evaluation on all TOFU datasets."""

import argparse
import fcntl
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TARGET = (
    ROOT
    / "saves/eval/giap"
    / "tofu_LLaDA-8B-Instruct_DiffusionSelectiveUnlearning_selective_beta10.0_preserve0.1"
)
EVALUATOR_PATH = ROOT / "scripts/dlm/evaluate_all_llm_judge.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("evaluate_all_llm_judge", EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluator from {EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def acquire_lock():
    lock_path = TARGET / ".llm_judge.lock"
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another judge process holds {lock_path}")
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")

    evaluator = load_evaluator()
    if args.workers is not None:
        if args.workers <= 0:
            raise SystemExit("--workers must be greater than zero")
        evaluator.JUDGE_WORKERS = args.workers

    eval_data = evaluator.load_json(TARGET / evaluator.EVAL_NAME)
    if not isinstance(eval_data, dict):
        raise SystemExit(f"missing or invalid {TARGET / evaluator.EVAL_NAME}")

    datasets = list(evaluator.DATASETS)
    entries_by_dataset = {
        dataset: evaluator.load_entries(eval_data, dataset) for dataset in datasets
    }
    missing = [dataset for dataset, entries in entries_by_dataset.items() if not entries]
    if missing:
        raise SystemExit(f"missing generations for: {', '.join(missing)}")

    print(
        f"target={TARGET}\n"
        f"judge={evaluator.JUDGE_MODEL} url={evaluator.JUDGE_BASE_URL} "
        f"workers={evaluator.JUDGE_WORKERS} batch_size={args.batch_size}"
    )
    for dataset, entries in entries_by_dataset.items():
        cached = evaluator.load_cached_scores(TARGET, dataset, entries)
        scored = min(len(cached[dimension]) for dimension in evaluator.DIMENSIONS)
        print(f"{dataset}: scored={scored}/{len(entries)} pending={len(entries) - scored}")

    if args.dry_run:
        return

    lock_handle = acquire_lock()
    try:
        result = evaluator.evaluate_directory(
            TARGET,
            entries_by_dataset,
            datasets,
            args.overwrite,
            args.batch_size,
            not args.no_progress,
        )
        print(f"[{result}] {TARGET}")
    finally:
        lock_handle.close()


if __name__ == "__main__":
    main()
