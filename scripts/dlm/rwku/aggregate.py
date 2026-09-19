import argparse
import json
import re
from pathlib import Path

# (metric key fragment, direction) -- direction is what "better" means after
# unlearning: forget probes should drop, everything else should hold.
LOWER_IS_BETTER = ("forget_level1", "forget_level2", "forget_level3")

ORDER = [
    "forget_level1_ROUGE",
    "forget_level2_ROUGE",
    "forget_level3_ROUGE",
    "neighbor_level1_ROUGE",
    "neighbor_level2_ROUGE",
    "mmlu",
    "truthfulqa_mc1",
    "triviaqa_em",
]


def load_runs(root, setting):
    runs = {}
    for summary in sorted(Path(root).glob(f"*/{setting}/*/RWKU_SUMMARY.json")):
        model_method = summary.parts[-4]
        entity = summary.parts[-2]
        try:
            data = json.loads(summary.read_text())
        except json.JSONDecodeError:
            print(f"  ! unreadable: {summary}")
            continue
        runs.setdefault(model_method, {})[entity] = data
    return runs


def entity_sort_key(name):
    m = re.match(r"^(\d+)_", name)
    return (int(m.group(1)) if m else 1 << 30, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval/RWKU")
    ap.add_argument("--setting", default="train_v1")
    ap.add_argument("--per-entity", action="store_true",
                    help="also print each entity's scores, not just the mean")
    args = ap.parse_args()

    runs = load_runs(args.root, args.setting)
    if not runs:
        print(f"no RWKU_SUMMARY.json under {args.root}/*/{args.setting}/*/")
        return

    metrics = [m for m in ORDER
               if any(m in d for e in runs.values() for d in e.values())]
    extra = sorted({k for e in runs.values() for d in e.values() for k in d}
                   - set(metrics))
    metrics += extra

    width = max(len(k) for k in runs) + 2
    header = f"{'run':<{width}}{'n':>3}  " + "  ".join(f"{m:>22}" for m in metrics)
    print(header)
    print("-" * len(header))

    for model_method in sorted(runs):
        entities = runs[model_method]
        cells = []
        for m in metrics:
            vals = [d[m] for d in entities.values()
                    if isinstance(d.get(m), (int, float))]
            cells.append(f"{sum(vals)/len(vals):>22.4f}" if vals else f"{'-':>22}")
        print(f"{model_method:<{width}}{len(entities):>3}  " + "  ".join(cells))

        if args.per_entity:
            for entity in sorted(entities, key=entity_sort_key):
                d = entities[entity]
                cells = [f"{d[m]:>22.4f}" if isinstance(d.get(m), (int, float))
                         else f"{'-':>22}" for m in metrics]
                print(f"  {entity:<{width-2}}{'':>3}  " + "  ".join(cells))

    print()
    print("lower is better: " + ", ".join(LOWER_IS_BETTER))
    print("higher is better: neighbor_*, mmlu, truthfulqa_mc1, triviaqa_em")
    incomplete = {k: len(v) for k, v in runs.items() if len(v) < 10}
    if incomplete:
        print("incomplete (fewer than 10 entities): "
              + ", ".join(f"{k}={n}" for k, n in sorted(incomplete.items())))


if __name__ == "__main__":
    main()
