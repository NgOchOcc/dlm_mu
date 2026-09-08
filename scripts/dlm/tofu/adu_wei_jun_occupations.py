#!/usr/bin/env python
"""Visualize generation-time top-k candidates for Wei-Jun Chen's parents' jobs."""

from __future__ import annotations

import gc
import json
import os
import sys
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from repel_gendistill_viz import build_trainer, tokenize
from repel_gentime_topk import generate_and_capture


QUESTION = "What were the occupations of Wei-Jun Chen's parents?"
ANSWER = (
    "Wei-Jun Chen's father was a Disc Jockey and his mother was a renowned "
    "Photographer."
)
BEFORE = "saves/finetune/tofu_LLaDA-8B-Instruct_retain100_150epochs"
AFTER = "saves/unlearn/forget10/tofu_LLaDA-8B-Instruct_DiffusionADU"
OUT_DIR = "outputs/adu_wei_jun_occupations"
TOPK = 10
GEN_STEPS = 128
SEED = 0


def load_model(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(BEFORE, trust_remote_code=True)
    model_class = get_class_from_dynamic_module(
        "modeling_llada.LLaDAModelLM", BEFORE
    )
    model = model_class.from_pretrained(
        path,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def find_subsequence(sequence, subsequence, allowed):
    for start in range(len(sequence) - len(subsequence) + 1):
        positions = list(range(start, start + len(subsequence)))
        if all(allowed[position] for position in positions):
            if sequence[start : start + len(subsequence)] == subsequence:
                return positions
    raise ValueError(f"Token sequence not found: {subsequence}")


def occupation_positions(tokenizer, input_ids, answer_mask):
    sequence = input_ids[0].tolist()
    allowed = answer_mask[0].tolist()
    father_ids = tokenizer.encode(" Disc Jockey", add_special_tokens=False)
    mother_ids = tokenizer.encode(" Photographer", add_special_tokens=False)
    father = find_subsequence(sequence, father_ids, allowed)
    mother = find_subsequence(sequence, mother_ids, allowed)
    labels = ["Father: Disc", "Father: J", "Father: ockey", "Mother: Photographer"]
    positions = father + mother
    if len(positions) != len(labels):
        raise ValueError(f"Unexpected occupation tokenization: {positions}")
    return positions, dict(zip(positions, labels))


def capture_checkpoint(name, path, device):
    model, tokenizer = load_model(path, device)
    args = SimpleNamespace(
        select_method="topk",
        topk_frac=0.8,
        sel_mc=16,
        gen_steps=GEN_STEPS,
        keep_answer=False,
        keep_subject_name=False,
    )
    trainer = build_trainer(model, tokenizer, args)
    input_ids, answer_mask, attention_mask, _ = tokenize(
        tokenizer, QUESTION, ANSWER, device
    )
    positions, labels = occupation_positions(tokenizer, input_ids, answer_mask)

    generated, captures = generate_and_capture(
        trainer,
        tokenizer,
        input_ids,
        answer_mask,
        attention_mask,
        positions,
        TOPK,
    )
    missing = [position for position in positions if position not in captures]
    if missing:
        raise RuntimeError(f"Positions were not captured for {name}: {missing}")

    embedding = model.get_output_embeddings().weight.detach().float()
    records = []
    vectors = []
    for position in positions:
        capture = captures[position]
        candidate_ids = capture["ids"]
        vectors.append(embedding[candidate_ids].cpu().numpy())
        records.append(
            {
                "position": position,
                "label": labels[position],
                "gold_id": int(input_ids[0, position]),
                "gold": tokenizer.decode([int(input_ids[0, position])]).strip(),
                "step": capture["step"],
                "chosen_id": capture["chosen"],
                "chosen": capture["chosen_surf"],
                "candidate_ids": candidate_ids,
                "candidates": capture["surf"],
                "probabilities": capture["probs"],
            }
        )

    answer_positions = answer_mask[0].nonzero(as_tuple=True)[0].tolist()
    generated_answer = tokenizer.decode(
        [int(generated[0, position]) for position in answer_positions]
    ).strip()
    result = {
        "name": name,
        "checkpoint": path,
        "question": QUESTION,
        "gold_answer": ANSWER,
        "generated_answer": generated_answer,
        "records": records,
        "vectors": np.concatenate(vectors),
    }

    del trainer, embedding, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def normalized_joint_tsne(before, after):
    vectors = np.concatenate([before["vectors"], after["vectors"]])
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return TSNE(
        n_components=2,
        metric="cosine",
        init="pca",
        perplexity=15,
        random_state=SEED,
    ).fit_transform(vectors)


def plot_model(result, coordinates, output_path):
    colors = ["#1f77b4", "#17becf", "#9467bd", "#d62728"]
    fig, axis = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("white")
    offset = 0

    for color, record in zip(colors, result["records"]):
        count = len(record["candidate_ids"])
        points = coordinates[offset : offset + count]
        offset += count
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=145,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
            label=record["label"],
        )
        for index, (surface, probability) in enumerate(
            zip(record["candidates"], record["probabilities"])
        ):
            candidate_id = record["candidate_ids"][index]
            marker = "★" if candidate_id == record["chosen_id"] else ""
            gold = " [gold]" if candidate_id == record["gold_id"] else ""
            label = surface.encode("ascii", "ignore").decode().strip() or "·"
            axis.annotate(
                f"{marker}{label}{gold}\n{probability:.3f}",
                points[index],
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="#222222",
            )
        center = np.median(points, axis=0)
        axis.text(
            center[0],
            center[1],
            record["label"],
            color=color,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color},
        )

    axis.set_title(
        f"{result['name']} — generation-time top-{TOPK} occupation candidates\n"
        f"Q: {QUESTION}\nGenerated: {result['generated_answer']}",
        fontsize=13,
        fontweight="bold",
        pad=18,
    )
    axis.set_xlabel("Joint t-SNE dimension 1")
    axis.set_ylabel("Joint t-SNE dimension 2")
    axis.legend(loc="best", framealpha=0.95)
    axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def serializable(result):
    return {key: value for key, value in result.items() if key != "vectors"}


def main():
    os.chdir(REPO)
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0")

    before = capture_checkpoint("Before ADU unlearning", BEFORE, device)
    after = capture_checkpoint("After ADU unlearning", AFTER, device)
    coordinates = normalized_joint_tsne(before, after)
    split = len(before["vectors"])

    before_path = os.path.join(OUT_DIR, "before_adu.png")
    after_path = os.path.join(OUT_DIR, "after_adu.png")
    plot_model(before, coordinates[:split], before_path)
    plot_model(after, coordinates[split:], after_path)

    report_path = os.path.join(OUT_DIR, "captures.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"before": serializable(before), "after": serializable(after)},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(before_path)
    print(after_path)
    print(report_path)


if __name__ == "__main__":
    main()
