#!/usr/bin/env python
"""Recompute the truth-ratio metrics of finished TOFU evals, without a GPU.

Why this exists: truth_ratio used to be computed as wrong_prob / (correct_prob +
1e-10) from exp(-avg_loss) of each side. A collapsed unlearned model reaches
avg_loss ~200 nats, so correct_prob underflows far below that 1e-10 guard, the
guard replaces correct_prob outright, and the reported ratio comes out ~1e-80
instead of O(1). memorization.py now works from the loss difference instead.

Every input the metric needs (the per-index avg_loss of the correct and wrong
answer sets) is already stored in TOFU_EVAL.json, so finished evals can be
corrected in place rather than re-run on GPU.

  usage:
    python scripts/dlm/fix_truth_ratio.py [glob ...]          # report only
    python scripts/dlm/fix_truth_ratio.py --write [glob ...]  # rewrite the files

  default glob: saves/eval/*/*/TOFU_EVAL.json

--write only touches metrics the epsilon actually corrupted (see EPS_LOSS): on a
healthy arm the recomputed value differs from the stored one by MC noise only,
because the stored top-level forget_Q_A_PARA_Prob is a second 128-sample draw
that overwrote the one the ratio consumed (eval/tofu_dlm.yaml now orders PARA
before the ratio so new runs no longer diverge). Those numbers are not wrong, so
leave them alone; --force overrides. Both TOFU_EVAL.json and TOFU_SUMMARY.json
are updated, after saving <file>.bak once.
"""

import argparse
import glob as globmod
import json
import os
import shutil

import numpy as np

# avg_loss above which the old formula's 1e-10 epsilon supplied >1% of the
# denominator, i.e. p_correct < 1e-8. Below it the epsilon is negligible and the
# stored value is sound.
EPS_LOSS = -np.log(1e-8)

# metric name -> (correct prob metric, wrong prob metric, aggregator)
# Mirrors configs/eval/tofu_dlm_metrics/*_Truth_Ratio.yaml.
SPECS = {
    "forget_truth_ratio": (
        "forget_Q_A_PARA_Prob",
        "forget_Q_A_PERT_Prob",
        "closer_to_1_better",
    ),
    "retain_Truth_Ratio": (
        "retain_Q_A_PARA_Prob",
        "retain_Q_A_PERT_Prob",
        "true_better",
    ),
    "ra_Truth_Ratio": ("ra_Q_A_Prob", "ra_Q_A_PERT_Prob", "true_better"),
    "wf_Truth_Ratio": ("wf_Q_A_Prob", "wf_Q_A_PERT_Prob", "true_better"),
}


def aggregate(scores, aggregator):
    """Same three aggregators as memorization.truth_ratio."""
    if aggregator == "closer_to_1_better":
        return float(np.mean(np.minimum(scores, 1 / (scores + 1e-10))))
    if aggregator == "true_better":
        return float(np.mean(np.maximum(0, 1 - scores)))
    if aggregator == "prob_mean":
        return float(np.mean(scores))
    raise ValueError(f"unknown aggregator: {aggregator}")


def recompute(evals, metric):
    """Return (indices, tr, agg_value, eps_frac) or None if inputs are missing.

    eps_frac is the share of indices whose correct-answer loss was high enough
    for the old epsilon to dominate -- the evidence that the stored value is an
    artifact rather than a differently-sampled but valid estimate.
    """
    correct_key, wrong_key, aggregator = SPECS[metric]
    if correct_key not in evals or wrong_key not in evals:
        return None
    correct = evals[correct_key]["value_by_index"]
    wrong = evals[wrong_key]["value_by_index"]

    indices = [
        i
        for i in correct
        if correct[i] is not None and wrong.get(i) is not None
    ]
    if not indices:
        return None
    # aggregate_to_1D: the wrong side holds one loss per perturbed answer.
    loss_correct = np.array([np.mean(correct[i]["avg_loss"]) for i in indices])
    loss_wrong = np.array([np.mean(wrong[i]["avg_loss"]) for i in indices])

    log_ratio = loss_correct - loss_wrong  # = log(wrong_prob / correct_prob)
    if aggregator == "prob_mean":
        from scipy.special import expit

        scores = expit(-log_ratio)
    else:
        scores = np.exp(np.clip(log_ratio, -700, 700))
    eps_frac = float(np.mean(loss_correct > EPS_LOSS))
    return indices, scores, aggregate(scores, aggregator), eps_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="*", default=None)
    ap.add_argument("--write", action="store_true", help="rewrite the eval files")
    ap.add_argument(
        "--force",
        action="store_true",
        help="rewrite even where the epsilon was negligible (MC-noise-only diffs)",
    )
    args = ap.parse_args()

    patterns = args.globs or ["saves/eval/*/*/TOFU_EVAL.json"]
    paths = sorted({p for pat in patterns for p in globmod.glob(pat)})
    if not paths:
        print(f"no eval files matched {patterns}")
        return

    header = f"{'run':<52}{'metric':<20}{'old':>12}{'new':>10}{'eps%':>7}  action"
    print(header)
    changed = 0
    for path in paths:
        try:
            with open(path) as fh:
                evals = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"{path}: unreadable ({exc})")
            continue

        run = os.path.basename(os.path.dirname(path)).replace("tofu_", "")
        updates = {}
        for metric in SPECS:
            if metric not in evals:
                continue
            got = recompute(evals, metric)
            if got is None:
                continue
            indices, scores, agg, eps_frac = got
            old = evals[metric].get("agg_value")
            moved = old is None or not np.isclose(old, agg, rtol=1e-9, atol=0)
            corrupted = eps_frac > 0
            if not moved:
                action = "unchanged"
            elif corrupted or args.force:
                action = "FIX"
                updates[metric] = (indices, scores, agg)
            else:
                action = "keep (MC noise)"
            old_str = "None" if old is None else f"{old:.4g}"
            print(
                f"{run:<52}{metric:<20}{old_str:>12}{agg:>10.4f}"
                f"{100 * eps_frac:>6.1f}%  {action}"
            )

        if not updates or not args.write:
            continue
        if not os.path.exists(path + ".bak"):
            shutil.copy2(path, path + ".bak")
        for metric, (indices, scores, agg) in updates.items():
            evals[metric]["agg_value"] = agg
            evals[metric]["value_by_index"] = {
                i: {"score": float(s)} for i, s in zip(indices, scores)
            }
        with open(path, "w") as fh:
            json.dump(evals, fh, indent=4)

        summary_path = os.path.join(os.path.dirname(path), "TOFU_SUMMARY.json")
        if os.path.exists(summary_path):
            with open(summary_path) as fh:
                summary = json.load(fh)
            for metric, (_, _, agg) in updates.items():
                if metric in summary:
                    summary[metric] = agg
            if not os.path.exists(summary_path + ".bak"):
                shutil.copy2(summary_path, summary_path + ".bak")
            with open(summary_path, "w") as fh:
                json.dump(summary, fh, indent=4)
        changed += 1

    if args.write:
        print(f"\nrewrote {changed} run(s); originals kept as *.bak")
    else:
        print("\nreport only -- pass --write to rewrite the eval files")


if __name__ == "__main__":
    main()
