"""Compare the heuristic query-entity selection (``_is_entity_surface``)
against each Monte Carlo top-k% selection, aggregated over every example in
a ``query_leak_probe.py --json`` output.

For each k in the report's ``k_percents``, treats the heuristic-selected
word set as the reference and the MC top-k% word set (after
``drop_stopwords``) as the candidate, and reports, pooled across all
examples (micro-averaged, i.e. summed intersection/union/etc. before
dividing - robust to examples with very few or zero selected words):
  - Jaccard = |heuristic ∩ mc_k| / |heuristic ∪ mc_k|
  - Precision = |heuristic ∩ mc_k| / |mc_k]  (of what MC masks, how much the
    heuristic would also mask)
  - Recall = |heuristic ∩ mc_k| / |heuristic|  (of what the heuristic masks,
    how much MC top-k% also masks)
  - F1 = harmonic mean of precision/recall

Usage:
    python scripts/dlm/analyze_heuristic_mc_overlap.py \\
        scripts/dlm/query_leak_probe_forget01_full.json
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path")
    args = ap.parse_args()

    results = json.loads(Path(args.json_path).read_text())
    if not results:
        print("no examples in report")
        return

    k_percents = results[0]["k_percents"]
    n_examples = len(results)

    stats = {k: {"inter": 0, "union": 0, "mc": 0, "heur": 0} for k in k_percents}
    n_heur_empty = 0

    for r in results:
        heur = set(r["heuristic_selected"])
        if not heur:
            n_heur_empty += 1
        row_by_key = {row["key"]: row for row in r["rows"]}
        for k in k_percents:
            mc = set(row_by_key[f"mc_top{k}"]["selected"])
            s = stats[k]
            s["inter"] += len(heur & mc)
            s["union"] += len(heur | mc)
            s["mc"] += len(mc)
            s["heur"] += len(heur)

    print(f"{n_examples} examples ({n_heur_empty} with empty heuristic selection)\n")
    header = f"{'k%':>5}  {'jaccard':>8}  {'precision':>9}  {'recall':>7}  {'f1':>6}  {'|mc|':>6}  {'|heur|':>7}  {'|inter|':>8}"
    print(header)
    print("-" * len(header))
    for k in k_percents:
        s = stats[k]
        jaccard = s["inter"] / s["union"] if s["union"] else float("nan")
        precision = s["inter"] / s["mc"] if s["mc"] else float("nan")
        recall = s["inter"] / s["heur"] if s["heur"] else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall) > 0
            else float("nan")
        )
        print(
            f"{k:>5}  {jaccard:>8.3f}  {precision:>9.3f}  {recall:>7.3f}  {f1:>6.3f}  "
            f"{s['mc']:>6}  {s['heur']:>7}  {s['inter']:>8}"
        )


if __name__ == "__main__":
    main()
