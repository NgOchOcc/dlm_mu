"""Aggregate TOFU_SUMMARY.json files under saves/eval into a single CSV.

One row per (model, method); columns are the TOFU summary metrics.
Run from the repo root: python scripts/dlm/aggregate_results.py
"""

import csv
import json
import re
from pathlib import Path

EVAL_DIR = Path("saves/eval")
OUT_CSV = EVAL_DIR / "tofu_results_summary.csv"

METRIC_COLS = [
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "retain_Q_A_Prob",
    "retain_Q_A_ROUGE",
    "ra_Q_A_Prob",
    "ra_Q_A_ROUGE",
    "wf_Q_A_Prob",
    "wf_Q_A_ROUGE",
]

METHOD_ORDER = {
    "base": 0, "full": 1, "full_cont": 2,
    "GradAscent": 3, "GradDiff": 4, "NPO": 5, "SimNPO": 6, "WGA": 7,
}


def parse_task(task_name):
    """tofu_<model>_<method> -> (model, method)."""
    name = task_name[len("tofu_"):]
    m = re.search(r"_forget\d+_Diffusion(\w+)$", name)
    if m:
        model = name[: m.start()]
        method = m.group(1)
    else:
        # base / full / full_cont
        for suffix in ("_full_cont", "_full", "_base"):
            if name.endswith(suffix):
                model = name[: -len(suffix)]
                method = suffix.lstrip("_")
                break
        else:
            model, method = name, ""
    return model, method


def main():
    rows = []
    for summary in sorted(EVAL_DIR.glob("*/TOFU_SUMMARY.json")):
        task_name = summary.parent.name
        model, method = parse_task(task_name)
        data = json.loads(summary.read_text())
        row = {"model": model, "method": method}
        for col in METRIC_COLS:
            val = data.get(col, "")
            row[col] = round(val * 100, 2) if isinstance(val, (int, float)) else ""
        rows.append(row)

    rows.sort(key=lambda r: (r["model"], METHOD_ORDER.get(r["method"], 99)))

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "method"] + METRIC_COLS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
