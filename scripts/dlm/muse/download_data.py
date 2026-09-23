"""Download every MUSE dataset the pipeline reads, into a local directory.

Saved as Arrow via `save_to_disk`, one directory per (repo, config, split),
so a later run can load them with `HF_DATASETS_OFFLINE=1` and never contact
the Hub. Point the configs at the result with --patch-configs, or keep the
hub paths and rely on the HF cache being warm.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

# (repo, config, split) triples used by the MUSE experiment and eval configs.
# `knowmem` carries the QA splits, `verbmem` the verbatim probe, `raw` the
# corpus; the *_icl splits are the few-shot prompts the knowmem metrics use.
WANTED = {
    "muse-bench/MUSE-{corpus}": [
        ("knowmem", "forget_qa"),
        ("knowmem", "forget_qa_icl"),
        ("knowmem", "retain_qa"),
        ("knowmem", "retain_qa_icl"),
        ("verbmem", "forget"),
        ("raw", "forget"),
        ("raw", "retain1"),
        ("raw", "retain2"),
        ("raw", "holdout"),
    ],
    "tamarsonha/MUSE-{corpus}-Train": [
        (None, "full"),
        (None, "retain"),
    ],
}


def target_dir(root, repo, config, split):
    return Path(root) / repo.replace("/", "__") / (config or "default") / split


def download(root, corpora, overwrite=False):
    manifest = {}
    for template, entries in WANTED.items():
        for corpus in corpora:
            repo = template.format(corpus=corpus)
            for config, split in entries:
                out = target_dir(root, repo, config, split)
                key = f"{repo}/{config or 'default'}/{split}"
                if out.exists() and not overwrite:
                    print(f"  skip   {key} (already saved)")
                    manifest[key] = str(out)
                    continue
                try:
                    kwargs = {"split": split}
                    if config is not None:
                        kwargs["name"] = config
                    dataset = load_dataset(repo, **kwargs)
                except Exception as exc:
                    # A missing split is not fatal: the *_icl and holdout
                    # splits do not exist for every corpus.
                    print(f"  MISS   {key}: {type(exc).__name__}")
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                dataset.save_to_disk(str(out))
                print(f"  saved  {key}  ({len(dataset)} rows) -> {out}")
                manifest[key] = str(out)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/muse")
    parser.add_argument("--corpora", default="News Books")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    corpora = args.corpora.split()
    print(f"corpora : {corpora}")
    print(f"out     : {args.out_dir}\n")

    manifest = download(args.out_dir, corpora, args.overwrite)

    path = Path(args.out_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {path} ({len(manifest)} datasets)")


if __name__ == "__main__":
    main()
