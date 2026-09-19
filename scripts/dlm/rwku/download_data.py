import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "jinzhuoran/RWKU"
ROWS_URL = "https://datasets-server.huggingface.co/rows"

# config -> (split, role) for everything the RWKU pipeline reads.
CONFIGS = {
    "train_pair_llama3": ("train", "train: per-entity unlearning (answer = response[1])"),
    "train_original_passage": ("train", "train: whole-corpus passages (ablation only)"),
    "forget_level1": ("test", "eval: forget probe, cloze"),
    "forget_level2": ("test", "eval: forget probe, QA"),
    "forget_level3": ("test", "eval: forget probe, adversarial"),
    "neighbor_level1": ("test", "eval: neighbor probe, cloze"),
    "neighbor_level2": ("test", "eval: neighbor probe, QA"),
    "utility_general": ("test", "eval: MMLU"),
    "utility_factuality": ("test", "eval: TriviaQA (disabled in configs)"),
    "utility_truthfulness": ("test", "eval: TruthfulQA (disabled in configs)"),
}


def fetch_page(config, split, offset, length, retries=8):
    qs = urllib.parse.urlencode(
        {"dataset": DATASET, "config": config, "split": split,
         "offset": offset, "length": length}
    )
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{ROWS_URL}?{qs}", timeout=120) as r:
                return json.load(r)["rows"]
        except urllib.error.HTTPError as exc:
            last = exc
            # The rows endpoint rate-limits well before a large config is fully
            # paged, and recovers on the order of a minute -- back off far
            # harder for 429 than for a transient network error.
            time.sleep(min(90, 20 * (attempt + 1)) if exc.code == 429
                       else 3 * (attempt + 1))
        except Exception as exc:
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{config}/{split} offset={offset}: {last}")


def download(config, split, subjects, out_dir, page=100, max_rows=None):
    """Stream one config, keeping rows whose `subject` is wanted.

    The rows endpoint has no server-side filter, so every page is fetched and
    filtered here. Configs without a `subject` column are kept whole.
    """
    out = Path(out_dir) / f"{config}.json"
    state_path = Path(out_dir) / f".{config}.progress.json"

    # Resume support: the rows endpoint rate-limits partway through the larger
    # configs, so progress is checkpointed and a re-run picks up where it left
    # off instead of re-paging from zero.
    kept, seen, offset = [], 0, 0
    if state_path.exists():
        state = json.loads(state_path.read_text())
        kept, seen, offset = state["kept"], state["seen"], state["offset"]
        print(f"    {config}: resuming at offset {offset} ({len(kept)} kept)")

    while True:
        rows = fetch_page(config, split, offset, page)
        if not rows:
            break
        for r in rows:
            row = r["row"]
            seen += 1
            if subjects is None or row.get("subject") in subjects:
                kept.append(row)
        offset += len(rows)
        state_path.write_text(json.dumps(
            {"kept": kept, "seen": seen, "offset": offset}))
        if max_rows is not None and seen >= max_rows:
            break
        print(f"    {config}: scanned {seen}, kept {len(kept)}", end="\r", flush=True)

    out.write_text(json.dumps(kept, ensure_ascii=False, indent=2))
    state_path.unlink(missing_ok=True)
    print(f"    {config}: scanned {seen}, kept {len(kept)} -> {out}    ")
    return len(kept), seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/rwku")
    ap.add_argument("--subjects-file", default="scripts/dlm/rwku/subjects.txt")
    ap.add_argument("--num-subjects", type=int, default=5)
    ap.add_argument("--all-subjects", action="store_true",
                    help="keep every entity instead of the first --num-subjects")
    ap.add_argument("--configs", default=None,
                    help="comma-separated subset of configs (default: all)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="stop after scanning this many source rows per config")
    args = ap.parse_args()

    subjects = None
    if not args.all_subjects:
        names = [ln.strip() for ln in open(args.subjects_file) if ln.strip()
                 and not ln.startswith("#")]
        subjects = set(names[: args.num_subjects])

    wanted = CONFIGS
    if args.configs:
        names = [c.strip() for c in args.configs.split(",") if c.strip()]
        missing = [c for c in names if c not in CONFIGS]
        if missing:
            raise SystemExit(f"unknown config(s): {missing}")
        wanted = {c: CONFIGS[c] for c in names}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset : {DATASET}")
    print(f"subjects: {'ALL' if subjects is None else sorted(subjects)}")
    print(f"out     : {out_dir}/\n")

    manifest = {}
    for config, (split, role) in wanted.items():
        done = out_dir / f"{config}.json"
        if done.exists() and not (out_dir / f".{config}.progress.json").exists():
            rows = json.loads(done.read_text())
            print(f"  {config}: already downloaded ({len(rows)} rows), skipping")
            manifest[config] = {"split": split, "role": role,
                                "rows_kept": len(rows), "rows_scanned": None}
            continue
        print(f"  {config} ({role})")
        kept, seen = download(config, split, subjects, out_dir,
                              max_rows=args.max_rows)
        manifest[config] = {"split": split, "role": role,
                            "rows_kept": kept, "rows_scanned": seen}

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(
        {"dataset": DATASET,
         "subjects": sorted(subjects) if subjects else "all",
         "configs": manifest}, ensure_ascii=False, indent=2))
    print(f"\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
