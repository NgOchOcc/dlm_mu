"""Post-hoc LLM-judge scoring of saved DLM evals (no model, GPU or sampling needed).

Reads the forget-set generations already stored in each checkpoint's
``TOFU_EVAL.json`` (under ``forget_Q_A_ROUGE.value_by_index[*]``) and re-judges
that text with the API judge. Three rubrics run per generation, each one API
call, each writing its own metrics:

* ``quality``     — readability / helpfulness / truthfulness (1-5).
* ``leak``        — binary YES/NO verdict on whether the answer leaks the
  forgotten answer, so the mean is the fraction of answers that leak (0-1).
* ``linguistic``  — Fluency / Relevance / Hallucination / Correctness (0-10).
  Higher is better on all four; Correctness is intentionally inverted
  (10 = completely wrong), and Relevance rewards refusals.

All dirs' prompts share one pool of ``JUDGE_WORKERS`` (default 8) threads, since
the cost is API round-trip latency rather than compute.

Per dir it writes one ``forget_Q_A_<Metric>.json`` (agg + per-index breakdown)
per dimension and folds every aggregate into each ``TOFU_SUMMARY*.json`` present,
which is where ``show_results.py`` reads its judge columns from. Writes happen
per dir and atomically, so an interrupted run only loses the dir in flight.
A rubric whose metrics are already in every summary is skipped, so adding a new
rubric does not re-pay for the ones already scored.

Run from the repo root:
    python scripts/dlm/run_judge.py                 # every dir, unscored rubrics only
    OVERWRITE=1 python scripts/dlm/run_judge.py     # re-judge everything
    JUDGE_SUITE=leak,linguistic python scripts/dlm/run_judge.py   # pick rubrics
    python scripts/dlm/run_judge.py <dir> ...       # only these dirs
Naming dirs selects which ones to visit; it does not by itself force re-judging,
so a dir that already has a rubric's scores still skips it. Combine with
OVERWRITE=1 to redo scores that are already there.
"""

import json
import os
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evals.metrics.llm_judge_api import (  # noqa: E402
    JUDGE_WORKERS,
    linguistic_prompt,
    parse_leak_verdict,
    parse_linguistic,
    parse_quality,
    privacy_leak_prompt,
    quality_prompt,
    query_judge_many,
)

EVAL_DIR = Path("saves/eval")
SOURCE_METRIC = "forget_Q_A_ROUGE"  # holds the forget-set generations

# A rubric: how to build its prompt from an eval entry, how to parse the verdict
# into dimension -> score, and which output metric each dimension lands in. One
# API call per (entry, spec); a multi-dimension parser fills several metrics from
# that single call.
JudgeSpec = namedtuple("JudgeSpec", "name prompt_fn parser dims")


def _leak_dims(text):
    """Adapt the scalar leak verdict to the dimension-dict contract of a spec."""
    verdict = parse_leak_verdict(text)
    return None if verdict is None else {"leak": verdict}


SPECS = (
    JudgeSpec(
        "quality",
        # The forgotten ground truth doubles as the `target` the model must not
        # discuss, which is what the quality rubric scores helpfulness against.
        lambda e: quality_prompt(
            e["input"], e["ground_truth"], e["generation"], e["ground_truth"]
        ),
        parse_quality,
        {
            "readability": "forget_Q_A_Readability",
            "helpfulness": "forget_Q_A_Helpfulness",
            "truthfulness": "forget_Q_A_Truthfulness",
        },
    ),
    JudgeSpec(
        "leak",
        lambda e: privacy_leak_prompt(e["input"], e["ground_truth"], e["generation"]),
        _leak_dims,
        {"leak": "forget_Q_A_Leak"},
    ),
    JudgeSpec(
        "linguistic",
        lambda e: linguistic_prompt(e["input"], e["ground_truth"], e["generation"]),
        parse_linguistic,
        {
            "Fluency": "forget_Q_A_Fluency",
            "Relevance": "forget_Q_A_Relevance",
            "Hallucination": "forget_Q_A_Hallucination",
            "Correctness": "forget_Q_A_Correctness",
        },
    ),
)

# JUDGE_SUITE picks a subset of rubrics by name, so a newly added rubric can be
# backfilled over old eval dirs without re-running the others.
_SUITE = os.environ.get("JUDGE_SUITE", "")
_WANTED = [s.strip() for s in _SUITE.split(",") if s.strip()]
if _WANTED:
    unknown = set(_WANTED) - {s.name for s in SPECS}
    if unknown:
        raise SystemExit(
            f"unknown JUDGE_SUITE entries {sorted(unknown)}; "
            f"known rubrics: {[s.name for s in SPECS]}"
        )
ACTIVE_SPECS = tuple(s for s in SPECS if not _WANTED or s.name in _WANTED)

# Trainers removed from the repo; their saved evals are kept but not re-judged.
SKIP_METHODS = ("DiffusionGRPELBO", "DiffusionPCTU", "DiffusionTRTD", "DiffusionKLPU")

# Seconds between progress lines when output is a log file rather than a tty.
LOG_INTERVAL = float(os.environ.get("JUDGE_LOG_INTERVAL", 30))


class _LogStream:
    """Turn tqdm's ``\\r`` redraws into newline-terminated lines.

    tqdm redraws in place, which collapses a redirected log into one enormous
    line; splitting on the carriage return keeps the bar (with its rate and ETA)
    readable under ``tail -f``.
    """

    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        text = text.strip("\r\n ")
        if text:
            self.stream.write(text + "\n")
            self.stream.flush()

    def flush(self):
        self.stream.flush()


def find_eval_dirs(argv):
    """Dirs to score: CLI args if given, else every dir under saves/eval that has
    a TOFU_EVAL.json and is not a removed method's leftovers."""
    if argv:
        return [Path(a) for a in argv]
    dirs = sorted({p.parent for p in EVAL_DIR.rglob("TOFU_EVAL.json")})
    return [d for d in dirs if not d.name.endswith(SKIP_METHODS)]


def summary_paths(eval_dir):
    return [
        p
        for p in (
            eval_dir / "TOFU_SUMMARY.json",
            eval_dir / "TOFU_SUMMARY_noprompt.json",
        )
        if p.exists()
    ]


def pending_specs(eval_dir):
    """Active rubrics whose metrics are not already in every summary of the dir.

    A dir with no summary at all is treated as fully pending: there is nothing to
    fold the aggregates into yet, but the per-metric JSON files are still worth
    writing.
    """
    summaries = [json.loads(p.read_text()) for p in summary_paths(eval_dir)]
    if not summaries:
        return list(ACTIVE_SPECS)
    return [
        spec
        for spec in ACTIVE_SPECS
        if not all(m in s for s in summaries for m in spec.dims.values())
    ]


def load_entries(eval_dir):
    """(index, entry) pairs with a generation, or None if there are none."""
    eval_file = eval_dir / "TOFU_EVAL.json"
    if not eval_file.exists():
        return None
    data = json.loads(eval_file.read_text())
    by_index = data.get(SOURCE_METRIC, {}).get("value_by_index", {})
    pairs = [
        (idx, entry)
        for idx, entry in by_index.items()
        if isinstance(entry, dict) and entry.get("generation") is not None
    ]
    return pairs or None


def write_json(path, payload):
    """Atomic write, so an interrupted run never leaves a truncated JSON file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=4))
    os.replace(tmp, path)


def score_dir(eval_dir, entries, specs, progress):
    """Judge every entry once per rubric in ``specs``.

    A multi-dimension rubric yields all of its dimensions from that single call;
    the verdict dict is then split into one output metric per dimension.
    Returns {metric: agg_value} across all rubrics.
    """
    indices = [idx for idx, _ in entries]
    aggs = {}
    for spec in specs:
        verdicts = query_judge_many(
            [spec.prompt_fn(entry) for _, entry in entries],
            parser=spec.parser,
            workers=JUDGE_WORKERS,
            progress=progress,
        )
        for dim, metric in spec.dims.items():
            metric_scores = [None if v is None else v.get(dim) for v in verdicts]
            valid = [s for s in metric_scores if s is not None]
            agg = float(np.mean(valid)) if valid else None
            aggs[metric] = agg
            write_json(
                eval_dir / f"{metric}.json",
                {
                    "agg_value": agg,
                    "n_scored": len(valid),
                    "value_by_index": {
                        str(idx): {"score": score, "generation": entry["generation"]}
                        for idx, score, (_, entry) in zip(
                            indices, metric_scores, entries
                        )
                    },
                },
            )
    return aggs


def update_summaries(eval_dir, aggs):
    """Fold the aggregates into every summary in the dir.

    A rubric that scored nothing at all (judge unreachable, every verdict
    unparseable) is left out: writing its ``null`` would make the dir look
    already-scored and skip the retry on the next run.
    """
    scored = {m: v for m, v in aggs.items() if v is not None}
    if not scored:
        return
    for path in summary_paths(eval_dir):
        summary = json.loads(path.read_text())
        summary.update(scored)
        write_json(path, summary)


def main():
    argv = sys.argv[1:]
    forced = os.environ.get("OVERWRITE") == "1"
    eval_dirs = find_eval_dirs(argv)

    work = []
    for eval_dir in eval_dirs:
        specs = list(ACTIVE_SPECS) if forced else pending_specs(eval_dir)
        if not specs:
            print(f"skip  {eval_dir}  (already scored)", flush=True)
            continue
        entries = load_entries(eval_dir)
        if entries is None:
            print(f"skip  {eval_dir}  (no {SOURCE_METRIC} generations)", flush=True)
            continue
        work.append((eval_dir, entries, specs))

    if not work:
        print(f"Nothing to judge under {EVAL_DIR}")
        return

    total_calls = sum(len(e) * len(s) for _, e, s in work)
    print(
        f"Judging {len(work)} dirs, {total_calls} calls, "
        f"{JUDGE_WORKERS} workers, rubrics={[s.name for s in ACTIVE_SPECS]} ...",
        flush=True,
    )
    # In a log file tqdm's in-place redraws become one huge line, so they are
    # split into throttled lines instead of being dropped.
    bar_kwargs = {"total": total_calls, "unit": "call"}
    if not sys.stderr.isatty():
        bar_kwargs.update(file=_LogStream(sys.stderr), mininterval=LOG_INTERVAL)
    with tqdm(**bar_kwargs) as bar:
        for eval_dir, entries, specs in work:
            bar.set_description(eval_dir.name[:48])
            aggs = score_dir(eval_dir, entries, specs, bar.update)
            update_summaries(eval_dir, aggs)
            shown = "  ".join(
                f"{metric.split('_')[-1]}="
                + ("n/a" if value is None else f"{value:.4f}")
                for metric, value in aggs.items()
            )
            print(f"ok    {eval_dir}  {shown}  (n={len(entries)})", flush=True)


if __name__ == "__main__":
    main()
