"""Post-hoc forget-set fluency scoring for saved DLM eval outputs.

Reads the forget-set generations already stored in each checkpoint's
``TOFU_EVAL.json`` (under ``forget_Q_A_ROUGE.value_by_index[*].generation``),
runs them through a gibberish detector, and records the mean probability of the
"clean" class as ``forget_Q_A_gibberish`` (higher = more fluent). No model
re-evaluation and no diffusion sampling — this only re-reads text that eval
already produced.

The score is written back into each ``TOFU_SUMMARY.json`` (and
``TOFU_SUMMARY_noprompt.json`` when present) so ``show_results.py`` can display
the ``f_Flu`` column. A per-index breakdown is also written to
``FLUENCY_forget.json`` next to the summary.

Run from the repo root:  python scripts/dlm/run_fluency.py
Dirs already holding the fluency score are skipped. Pass explicit checkpoint
dirs to force re-scoring:  python scripts/dlm/run_fluency.py <dir> ...
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EVAL_DIR = Path("saves/eval")
CLASSIFIER = "madhurjindal/autonlp-Gibberish-Detector-492513457"
METRIC_KEY = "forget_Q_A_gibberish"
SOURCE_METRIC = "forget_Q_A_ROUGE"  # holds the forget-set generations
CLEAN_CLASS_ID = 0  # id2label: {0: clean, 1: mild gibberish, 2: noise, 3: word salad}
BATCH_SIZE = 32
MAX_LENGTH = 256


def find_eval_dirs(argv):
    """Checkpoint dirs to score: CLI args if given, else every dir under
    saves/eval that has a TOFU_EVAL.json."""
    if argv:
        return [Path(a) for a in argv]
    return sorted({p.parent for p in EVAL_DIR.rglob("TOFU_EVAL.json")})


def already_scored(eval_dir):
    """True if any summary in the dir already holds the fluency metric."""
    for name in ("TOFU_SUMMARY.json", "TOFU_SUMMARY_noprompt.json"):
        path = eval_dir / name
        if path.exists() and METRIC_KEY in json.loads(path.read_text()):
            return True
    return False


def load_generations(eval_dir):
    """(index, generation) pairs from the forget-set ROUGE metric, or None if
    the eval file / generations are absent."""
    eval_file = eval_dir / "TOFU_EVAL.json"
    if not eval_file.exists():
        return None
    data = json.loads(eval_file.read_text())
    by_index = data.get(SOURCE_METRIC, {}).get("value_by_index", {})
    pairs = [
        (idx, entry["generation"])
        for idx, entry in by_index.items()
        if isinstance(entry, dict) and entry.get("generation") is not None
    ]
    return pairs or None


@torch.no_grad()
def clean_probs(texts, tokenizer, classifier, device):
    """Probability of the clean class for each text (list of floats)."""
    scores = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)
        logits = classifier(**inputs).logits
        probs = F.softmax(logits, dim=-1)[:, CLEAN_CLASS_ID]
        scores.extend(probs.cpu().tolist())
    return scores


def update_summaries(eval_dir, agg_value):
    """Write METRIC_KEY (forget-set fluency) into every summary JSON present in
    the dir. The Utility metric is derived downstream in show_results.py."""
    for name in ("TOFU_SUMMARY.json", "TOFU_SUMMARY_noprompt.json"):
        path = eval_dir / name
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        summary[METRIC_KEY] = agg_value
        path.write_text(json.dumps(summary, indent=4))


def main():
    forced = bool(sys.argv[1:]) or os.environ.get("OVERWRITE") == "1"
    eval_dirs = find_eval_dirs(sys.argv[1:])
    if not forced:
        for eval_dir in eval_dirs:
            if already_scored(eval_dir):
                print(f"skip  {eval_dir}  (already scored)")
        eval_dirs = [d for d in eval_dirs if not already_scored(d)]
    if not eval_dirs:
        print(f"Nothing to score under {EVAL_DIR}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading gibberish detector on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER)
    classifier = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER)
    classifier = classifier.to(device).eval()

    for eval_dir in eval_dirs:
        pairs = load_generations(eval_dir)
        if pairs is None:
            print(f"skip  {eval_dir}  (no {SOURCE_METRIC} generations)")
            continue

        indices, texts = zip(*pairs)
        scores = clean_probs(list(texts), tokenizer, classifier, device)
        agg_value = float(np.mean(scores))

        breakdown = {
            "agg_value": agg_value,
            "value_by_index": {
                str(idx): {"score": s, "generation": t}
                for idx, s, t in zip(indices, scores, texts)
            },
        }
        (eval_dir / "FLUENCY_forget.json").write_text(json.dumps(breakdown, indent=4))
        update_summaries(eval_dir, agg_value)
        print(f"ok    {eval_dir}  fluency={agg_value * 100:6.2f}  (n={len(scores)})")


if __name__ == "__main__":
    main()
