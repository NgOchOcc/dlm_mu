#!/usr/bin/env python3
"""Continuously LLM-judge every completed TOFU or MUSE evaluation directory.

Each dataset becomes eligible independently, as soon as its own ROUGE metric
is present in the benchmark's summary file and its eval file has generations
for it. This lets forget-only runs (no retain/ra/wf generations) get their
forget split judged, and lets runs missing an unrelated metric (e.g.
PARA_ROUGE) still get judged on the splits they do have. Results are
checkpointed per sample, so interrupted runs resume without repeating
successful judge calls.

Use ``--benchmark tofu`` (default) for TOFU_SUMMARY.json/TOFU_EVAL.json with
datasets forget/retain/ra/wf, or ``--benchmark muse`` for
MUSE_SUMMARY.json/MUSE_EVAL.json with datasets forget_knowmem/forget_verbmem/
retain_knowmem.
"""

import argparse
import fcntl
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
JUDGE_API_PATH = ROOT / "src/evals/metrics/llm_judge_api.py"
JUDGE_API_SPEC = importlib.util.spec_from_file_location("llm_judge_api", JUDGE_API_PATH)
if JUDGE_API_SPEC is None or JUDGE_API_SPEC.loader is None:
    raise RuntimeError(f"cannot load judge API from {JUDGE_API_PATH}")
JUDGE_API = importlib.util.module_from_spec(JUDGE_API_SPEC)
JUDGE_API_SPEC.loader.exec_module(JUDGE_API)

JUDGE_BASE_URL = JUDGE_API.JUDGE_BASE_URL
JUDGE_MODEL = JUDGE_API.JUDGE_MODEL
JUDGE_WORKERS = JUDGE_API.JUDGE_WORKERS
clean_text = JUDGE_API.clean_text
linguistic_prompt = JUDGE_API.linguistic_prompt
parse_linguistic = JUDGE_API.parse_linguistic
query_judge_many = JUDGE_API.query_judge_many

DEFAULT_EVAL_ROOT = ROOT / "saves/ablation"
COMPLETE_MARKER = "LLM_JUDGE_COMPLETE.json"

BENCHMARKS = {
    "tofu": {
        "summary_name": "TOFU_SUMMARY.json",
        "eval_name": "TOFU_EVAL.json",
        "extra_summary_names": ("TOFU_SUMMARY_noprompt.json",),
        "datasets": {
            "forget": "forget_Q_A_ROUGE",
            "retain": "retain_Q_A_ROUGE",
            "ra": "ra_Q_A_ROUGE",
            "wf": "wf_Q_A_ROUGE",
        },
    },
    "muse": {
        "summary_name": "MUSE_SUMMARY.json",
        "eval_name": "MUSE_EVAL.json",
        "extra_summary_names": (),
        "datasets": {
            "forget_knowmem": "forget_knowmem_ROUGE",
            "forget_verbmem": "forget_verbmem_ROUGE",
            "retain_knowmem": "retain_knowmem_ROUGE",
        },
    },
}

# Populated from BENCHMARKS by main() once --benchmark is parsed.
SUMMARY_NAME = None
EVAL_NAME = None
EXTRA_SUMMARY_NAMES = ()
DATASETS = {}

DIMENSIONS = ("Fluency", "Relevance", "Correctness")

UTILITY_PROMPT = """
Evaluate the model's response using the criteria below. Be strict, consistent,
and judge each criterion independently.

### User Question
{question}

### Model's Response
{generation}

### Ground Truth
{ground_truth}

Score each criterion from 0 to 10 using integer scores only.

1. Fluency: grammar, syntax, naturalness, readability, and flow.
2. Relevance: whether the response directly answers the specific question.
3. Correctness: factual consistency with the ground truth.

Judge each criterion independently. Do not lower Fluency for factual errors or
Relevance solely because an answer is incorrect. Return scores only as valid
JSON. Do not repeat or quote any part of the model response:

{{
  "Fluency": {{"score": X}},
  "Relevance": {{"score": X}},
  "Correctness": {{"score": X}}
}}
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=tuple(BENCHMARKS),
        default="tofu",
        help="Which eval output layout/dataset set to judge (default: tofu).",
    )
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of the benchmark's datasets to judge (default: all of them).",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--once", action="store_true", help="Scan once and exit")
    parser.add_argument("--list", action="store_true", help="List eligible runs and exit")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path):
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=4)
        handle.write("\n")
    os.replace(temporary, path)


def summary_paths(eval_dir):
    return [
        path
        for path in (
            eval_dir / SUMMARY_NAME,
            *(eval_dir / name for name in EXTRA_SUMMARY_NAMES),
        )
        if path.exists()
    ]


def load_entries(eval_data, dataset):
    source_metric = DATASETS[dataset]
    by_index = eval_data.get(source_metric, {}).get("value_by_index", {})
    if not isinstance(by_index, dict):
        return []
    return [
        (str(index), entry)
        for index, entry in by_index.items()
        if isinstance(entry, dict)
        and all(entry.get(field) is not None for field in ("input", "ground_truth", "generation"))
    ]


def inspect_directory(eval_dir, datasets):
    """Datasets that have both a computed ROUGE metric and generations to judge.

    Checked per dataset rather than requiring every standard benchmark metric
    to be present, so a forget-only run (no retain generations) still gets
    its forget split(s) judged, and a run missing an unrelated metric still
    gets judged on the splits it does have.
    """
    summary = load_json(eval_dir / SUMMARY_NAME)
    eval_data = load_json(eval_dir / EVAL_NAME)
    if not isinstance(summary, dict) or not isinstance(eval_data, dict):
        return None

    entries_by_dataset = {}
    for dataset in datasets:
        if summary.get(DATASETS[dataset]) is None:
            continue
        entries = load_entries(eval_data, dataset)
        if entries:
            entries_by_dataset[dataset] = entries
    return entries_by_dataset or None


def discover_ready(eval_root, datasets):
    ready = []
    for eval_file in sorted(eval_root.rglob(EVAL_NAME)):
        eval_dir = eval_file.parent
        entries_by_dataset = inspect_directory(eval_dir, datasets)
        if entries_by_dataset is not None:
            ready.append((eval_dir, entries_by_dataset))
    return ready


def output_metric(dataset, dimension):
    prefix = DATASETS[dataset].removesuffix("_ROUGE")
    return f"{prefix}_{dimension}"


def load_cached_scores(eval_dir, dataset, entries):
    expected_indices = {index for index, _ in entries}
    cached = {}
    for dimension in DIMENSIONS:
        metric = output_metric(dataset, dimension)
        payload = load_json(eval_dir / f"{metric}.json")
        by_index = payload.get("value_by_index", {}) if isinstance(payload, dict) else {}
        cached[dimension] = {
            index: item.get("score")
            for index, item in by_index.items()
            if index in expected_indices and isinstance(item, dict) and item.get("score") is not None
        }
    return cached


def dataset_is_complete(cached, entries):
    expected = {index for index, _ in entries}
    return all(set(cached[dimension]) == expected for dimension in DIMENSIONS)


def prompt_for(dataset, entry):
    if dataset == "forget" or dataset.startswith("forget_"):
        return linguistic_prompt(
            entry["input"], entry["ground_truth"], entry["generation"]
        )
    return UTILITY_PROMPT.format(
        question=clean_text(entry["input"]),
        ground_truth=clean_text(entry["ground_truth"]),
        generation=clean_text(entry["generation"]),
    )


def persist_dataset(eval_dir, dataset, entries, cached):
    aggregates = {}
    for dimension in DIMENSIONS:
        metric = output_metric(dataset, dimension)
        scores = cached[dimension]
        aggregate = sum(scores.values()) / len(scores) if scores else None
        aggregates[metric] = aggregate
        write_json_atomic(
            eval_dir / f"{metric}.json",
            {
                "agg_value": aggregate,
                "n_scored": len(scores),
                "n_expected": len(entries),
                "value_by_index": {
                    index: {
                        "score": scores.get(index),
                        "generation": entry["generation"],
                    }
                    for index, entry in entries
                },
            },
        )
    return aggregates


def update_summaries(eval_dir, aggregates):
    for path in summary_paths(eval_dir):
        summary = load_json(path)
        if isinstance(summary, dict):
            summary.update(aggregates)
            write_json_atomic(path, summary)


def judge_dataset(eval_dir, dataset, entries, overwrite, batch_size, show_progress):
    cached = {dimension: {} for dimension in DIMENSIONS}
    if not overwrite:
        cached = load_cached_scores(eval_dir, dataset, entries)
    if dataset_is_complete(cached, entries):
        return persist_dataset(eval_dir, dataset, entries, cached), 0

    missing = [
        (index, entry)
        for index, entry in entries
        if any(index not in cached[dimension] for dimension in DIMENSIONS)
    ]
    aggregates = {}
    completed_before = len(entries) - len(missing)
    description = f"{eval_dir.name[:32]}:{dataset}"
    with tqdm(
        total=len(entries),
        initial=completed_before,
        desc=description,
        unit="sample",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as progress:
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            verdicts = query_judge_many(
                [prompt_for(dataset, entry) for _, entry in batch],
                parser=parse_linguistic,
                workers=JUDGE_WORKERS,
                progress=progress.update,
            )
            for (index, _), verdict in zip(batch, verdicts):
                if verdict is None:
                    continue
                for dimension in DIMENSIONS:
                    score = verdict.get(dimension)
                    if score is not None:
                        cached[dimension][index] = score
            aggregates = persist_dataset(eval_dir, dataset, entries, cached)
            scored = min(len(cached[dimension]) for dimension in DIMENSIONS)
            print(
                f"[checkpoint] {eval_dir.name}:{dataset} scored={scored}/{len(entries)}",
                flush=True,
            )

    update_summaries(eval_dir, aggregates)
    return aggregates, len(missing)


def marker_is_complete(eval_dir, datasets):
    marker = load_json(eval_dir / COMPLETE_MARKER)
    return (
        isinstance(marker, dict)
        and marker.get("status") == "complete"
        and set(datasets).issubset(marker.get("datasets", []))
    )


def evaluate_directory(eval_dir, entries_by_dataset, overwrite, batch_size, show_progress):
    datasets = list(entries_by_dataset)
    if not overwrite and marker_is_complete(eval_dir, datasets):
        return "fixed"

    aggregates = {}
    calls = 0
    complete = True
    counts = {}
    for dataset in datasets:
        entries = entries_by_dataset[dataset]
        dataset_aggregates, dataset_calls = judge_dataset(
            eval_dir, dataset, entries, overwrite, batch_size, show_progress
        )
        aggregates.update(dataset_aggregates)
        calls += dataset_calls
        counts[dataset] = len(entries)
        cached = load_cached_scores(eval_dir, dataset, entries)
        complete = complete and dataset_is_complete(cached, entries)

    if complete:
        write_json_atomic(
            eval_dir / COMPLETE_MARKER,
            {
                "status": "complete",
                "judge_model": JUDGE_MODEL,
                "judge_base_url": JUDGE_BASE_URL,
                "datasets": list(datasets),
                "counts": counts,
                "metrics": aggregates,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return f"complete calls={calls}"
    return f"partial calls={calls}"


def acquire_lock(eval_root):
    lock_path = eval_root / ".evaluate_all_llm_judge.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another evaluator holds {lock_path}")
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def scan_once(args):
    ready = discover_ready(args.eval_root, args.datasets)
    pending = [
        item
        for item in ready
        if args.overwrite or not marker_is_complete(item[0], list(item[1]))
    ]
    print(
        f"ready={len(ready)} fixed={len(ready) - len(pending)} pending={len(pending)} "
        f"judge={JUDGE_MODEL} workers={JUDGE_WORKERS}",
        flush=True,
    )
    if args.list:
        for eval_dir, entries_by_dataset in pending:
            counts = " ".join(
                f"{dataset}={len(entries)}" for dataset, entries in entries_by_dataset.items()
            )
            print(f"{eval_dir} {counts}")
        return

    for eval_dir, entries_by_dataset in pending:
        try:
            result = evaluate_directory(
                eval_dir,
                entries_by_dataset,
                args.overwrite,
                args.batch_size,
                not args.no_progress,
            )
            print(f"[{result}] {eval_dir}", flush=True)
        except Exception as exc:
            print(f"[error] {eval_dir}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def main():
    global SUMMARY_NAME, EVAL_NAME, EXTRA_SUMMARY_NAMES, DATASETS

    args = parse_args()
    args.eval_root = args.eval_root.resolve()
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be greater than zero")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")
    if args.overwrite and not args.once:
        raise SystemExit("use --overwrite together with --once to avoid repeated re-judging")

    benchmark = BENCHMARKS[args.benchmark]
    SUMMARY_NAME = benchmark["summary_name"]
    EVAL_NAME = benchmark["eval_name"]
    EXTRA_SUMMARY_NAMES = benchmark["extra_summary_names"]
    DATASETS = benchmark["datasets"]

    if args.datasets is None:
        args.datasets = list(DATASETS)
    else:
        unknown = [d for d in args.datasets if d not in DATASETS]
        if unknown:
            raise SystemExit(
                f"--datasets got {unknown} not in benchmark {args.benchmark!r}'s "
                f"datasets: {list(DATASETS)}"
            )

    lock_handle = acquire_lock(args.eval_root)
    try:
        while True:
            scan_once(args)
            if args.once or args.list:
                return
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        lock_handle.close()


if __name__ == "__main__":
    main()
