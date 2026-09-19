import argparse
import json
import random
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.qa import QADataset  # noqa: E402
from model import get_model  # noqa: E402
from trainer.dlm.unlearn.selective_unlearning import DiffusionSelectiveUnlearning  # noqa: E402
from trainer.dlm.unlearn.query_importance import (  # noqa: E402
    build_word_groups,
    calibrate_entity_importance,
    drop_stopwords,
    expand_name_spans,
    mc_query_leak_importance,
    topk_word_indices,
)


def precompute_example(model, tokenizer, mask_id, vocab_size, device, item, args, rng):
    input_ids = item["input_ids"].to(device)
    labels = item["labels"].to(device)
    attention_mask = item.get("attention_mask")
    attention_mask = attention_mask.to(device) if attention_mask is not None else torch.ones_like(input_ids)

    answer_pos = (labels != -100).nonzero(as_tuple=True)[0]
    answer_targets = input_ids[answer_pos]
    query_pos = (labels == -100).nonzero(as_tuple=True)[0]

    query_words = build_word_groups(tokenizer, input_ids, query_pos)

    query_importance = mc_query_leak_importance(
        model, vocab_size, mask_id, input_ids, attention_mask,
        answer_pos, answer_targets, query_words,
        args.num_mc_trials, rng,
    )
    query_importance = calibrate_entity_importance(
        query_importance, query_words, DiffusionSelectiveUnlearning._QUERY_KEEP,
        boost=args.entity_boost,
    )

    query_selected = topk_word_indices(query_importance, args.k_percent_query)
    query_selected = drop_stopwords(
        query_selected, query_words, DiffusionSelectiveUnlearning._QUERY_KEEP
    )
    query_selected = expand_name_spans(
        query_selected, query_words, DiffusionSelectiveUnlearning._QUERY_KEEP
    )

    return {
        "query": sorted(query_selected),
        "n_query_words": len(query_words),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dataset-path", default="locuslab/TOFU")
    ap.add_argument(
        "--dataset-name", default="forget10",
        help="HF dataset config name (TOFU: forget10/forget05/forget01; MUSE: knowmem)",
    )
    ap.add_argument(
        "--split", default="train",
        help="HF dataset split (TOFU: train; MUSE knowmem: forget_qa/retain_qa)",
    )
    ap.add_argument("--question-key", default="question")
    ap.add_argument("--answer-key", default="answer")
    ap.add_argument("--filter-column", default=None)
    ap.add_argument("--filter-value", default=None)
    ap.add_argument("--answer-index", type=int, default=None)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=None, help="default: whole split")
    ap.add_argument("--num-mc-trials", type=int, default=32)
    ap.add_argument("--k-percent-query", type=int, default=20)
    ap.add_argument(
        "--entity-boost", type=float, default=1.6,
        help="Boost capitalized/numeric query words before ",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    model_cfg = OmegaConf.create({
        "model_type": "dlm",
        "model_args": {"pretrained_model_name_or_path": args.model_dir, "torch_dtype": "bfloat16"},
    })
    model, tokenizer = get_model(model_cfg)
    model.eval()
    device = next(model.parameters()).device
    mask_id = tokenizer.mask_token_id
    vocab_size = len(tokenizer)

    # The filter must match the one used at training time: the cache is keyed by
    # the dataset index, which is assigned AFTER filtering, so precomputing on
    # the unfiltered split would misalign every entry.
    hf_args = {"name": args.dataset_name, "split": args.split, "path": args.dataset_path}
    if args.filter_column is not None and args.filter_value is not None:
        hf_args["filter_column"] = args.filter_column
        hf_args["filter_value"] = args.filter_value

    dataset = QADataset(
        hf_args=hf_args,
        template_args={"apply_chat_template": True, "system_prompt": None},
        tokenizer=tokenizer,
        question_key=args.question_key,
        answer_key=args.answer_key,
        answer_index=args.answer_index,
        max_length=args.max_length,
    )

    num_samples = args.num_samples if args.num_samples is not None else len(dataset)
    end_index = min(args.start_index + num_samples, len(dataset))

    entries = {}
    for i in range(args.start_index, end_index):
        item = dataset[i]
        dataset_index = int(dataset.data[i]["index"])
        entry = precompute_example(model, tokenizer, mask_id, vocab_size, device, item, args, rng)
        entries[str(dataset_index)] = entry
        print(
            f"[dataset idx {dataset_index}, {i - args.start_index + 1}/{end_index - args.start_index}] "
            f"query {len(entry['query'])}/{entry['n_query_words']}"
        )

    cache = {
        "meta": {
            "model_dir": args.model_dir,
            "dataset_path": args.dataset_path,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "filter_column": args.filter_column,
            "filter_value": args.filter_value,
            "answer_index": args.answer_index,
            "k_percent_query": args.k_percent_query,
            "entity_boost": args.entity_boost,
            "num_mc_trials": args.num_mc_trials,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "entries": entries,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cache, indent=2))
    print(f"wrote {out_path} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
