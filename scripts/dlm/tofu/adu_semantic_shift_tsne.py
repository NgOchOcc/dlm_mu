#!/usr/bin/env python
"""Plot word-level semantic shift from pre-ADU top-k to post-ADU predictions."""

from __future__ import annotations

import json
import os
import re

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
BASE = os.path.join(
    REPO, "saves/finetune/tofu_LLaDA-8B-Instruct_retain100_150epochs"
)
REPORT = os.path.join(REPO, "outputs/adu_wei_jun_occupations/captures.json")
OUTPUT = os.path.join(
    REPO, "outputs/adu_wei_jun_occupations/semantic_shift_tsne.png"
)
PDF_OUTPUT = os.path.join(
    REPO, "outputs/adu_wei_jun_occupations/semantic_shift_tsne.pdf"
)


PANELS = {
    "Father occupation": {
        "before": [
            "Disc Jockey",
            "DJ",
            "Photographer",
            "professional",
            "Research",
            "Marine",
            "acclaimed",
            "renowned",
        ],
        "after": [
            "Psychiatrist",
            "Counselor",
            "Engineer",
            "Psychologist",
            "Civil",
            "professions",
            "renowned",
        ],
        "generated": "Psychiatrist",
        "gold": "Disc Jockey",
    },
    "Mother occupation": {
        "before": [
            "Photographer",
            "Reporter",
            "Producer",
            "artist",
            "photography",
            "chef",
            "physicist",
            "renowned",
        ],
        "after": [
            "layer",
            "mechanic",
            "Engineer",
            "Counselor",
            "nurse",
            "Psychologist",
            "professions",
            "Brick",
        ],
        "generated": "layer",
        "gold": "Photographer",
    },
}


def load_base_embedding():
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    config = AutoConfig.from_pretrained(BASE, trust_remote_code=True)
    model_class = get_class_from_dynamic_module(
        "modeling_llada.LLaDAModelLM", BASE
    )
    model = model_class.from_pretrained(
        BASE,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    embedding = model.get_output_embeddings().weight.detach().float().cpu()
    del model
    return tokenizer, embedding


def phrase_embedding(tokenizer, embedding, phrase):
    token_ids = tokenizer.encode(" " + phrase, add_special_tokens=False)
    rows = embedding[torch.tensor(token_ids)]
    return F.normalize(rows.mean(0), dim=0).numpy()


def validate_words(report):
    after = report["after"]["records"]
    father = "".join(record["chosen"] for record in after[:3]).strip()
    mother = after[3]["chosen"].strip()
    if father != PANELS["Father occupation"]["generated"]:
        raise ValueError(f"Unexpected father generation: {father}")
    if mother != PANELS["Mother occupation"]["generated"]:
        raise ValueError(f"Unexpected mother generation: {mother}")


def plot_panel(axis, tokenizer, embedding, title, panel, seed):
    labels = panel["before"] + panel["after"]
    vectors = np.stack(
        [phrase_embedding(tokenizer, embedding, label) for label in labels]
    )
    coordinates = TSNE(
        n_components=2,
        metric="cosine",
        init="pca",
        perplexity=5,
        random_state=seed,
    ).fit_transform(vectors)
    before_count = len(panel["before"])
    before_xy = coordinates[:before_count]
    after_xy = coordinates[before_count:]
    before_vectors = vectors[:before_count]
    generated_index = panel["after"].index(panel["generated"])
    generated_vector = vectors[before_count + generated_index]
    centroid = F.normalize(torch.tensor(before_vectors).mean(0), dim=0).numpy()
    cosine_similarity = float(np.dot(centroid, generated_vector))
    cosine_distance = 1.0 - cosine_similarity

    axis.scatter(
        before_xy[:, 0],
        before_xy[:, 1],
        s=125,
        color="#D55E00",
        edgecolor="white",
        linewidth=0.8,
        label="Top-k before ADU",
        zorder=3,
    )
    axis.scatter(
        after_xy[:, 0],
        after_xy[:, 1],
        s=125,
        marker="D",
        color="#0072B2",
        edgecolor="white",
        linewidth=0.8,
        label="Top-k after ADU",
        zorder=3,
    )
    star = after_xy[generated_index]
    projected_centroid = before_xy.mean(0)
    axis.scatter(
        [projected_centroid[0]],
        [projected_centroid[1]],
        s=220,
        marker="X",
        color="#333333",
        edgecolor="white",
        linewidth=1.0,
        label="Before-cluster centroid",
        zorder=4,
    )
    axis.plot(
        [projected_centroid[0], star[0]],
        [projected_centroid[1], star[1]],
        linestyle="--",
        linewidth=2,
        color="#555555",
        alpha=0.8,
        zorder=2,
    )
    axis.scatter(
        [star[0]],
        [star[1]],
        s=520,
        marker="*",
        color="#CC79A7",
        edgecolor="black",
        linewidth=1.2,
        label="Generated after ADU",
        zorder=5,
    )

    for point, label in zip(before_xy, panel["before"]):
        weight = "bold" if label == panel["gold"] else "normal"
        axis.annotate(
            label,
            point,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight=weight,
            color="#8B2E00",
        )
    for point, label in zip(after_xy, panel["after"]):
        weight = "bold" if label == panel["generated"] else "normal"
        axis.annotate(
            label,
            point,
            xytext=(5, -12 if weight == "bold" else 5),
            textcoords="offset points",
            fontsize=9,
            fontweight=weight,
            color="#005A87",
        )

    axis.set_title(
        f"{title}\nDistance from generated word to before-cluster centroid: "
        f"{cosine_distance:.3f}",
        fontsize=13,
        fontweight="bold",
    )
    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")
    axis.grid(alpha=0.15)
    return cosine_distance


def main():
    with open(REPORT, encoding="utf-8") as handle:
        report = json.load(handle)
    validate_words(report)
    tokenizer, embedding = load_base_embedding()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    distances = {}
    for seed, (axis, (title, panel)) in enumerate(zip(axes, PANELS.items())):
        distances[title] = plot_panel(axis, tokenizer, embedding, title, panel, seed)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, framealpha=0.95)
    fig.suptitle(
        "Semantic shift of occupation predictions after DiffusionADU\n"
        "All words are embedded with the same pre-unlearning LLaDA output embedding",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.91))
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    fig.savefig(PDF_OUTPUT, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(distances, indent=2))
    print(OUTPUT)
    print(PDF_OUTPUT)


if __name__ == "__main__":
    main()
