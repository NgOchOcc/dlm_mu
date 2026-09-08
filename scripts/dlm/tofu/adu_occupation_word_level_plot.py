#!/usr/bin/env python
"""Create a paper-friendly word-level comparison from generation-time captures."""

from __future__ import annotations

import itertools
import json
import math
import os
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
REPORT = os.path.join(REPO, "outputs/adu_wei_jun_occupations/captures.json")
TOKENIZER = os.path.join(
    REPO, "saves/finetune/tofu_LLaDA-8B-Instruct_retain100_150epochs"
)
OUTPUT = os.path.join(
    REPO, "outputs/adu_wei_jun_occupations/word_level_comparison.png"
)
PDF_OUTPUT = os.path.join(
    REPO, "outputs/adu_wei_jun_occupations/word_level_comparison.pdf"
)
PAPER_OUTPUT = os.path.join(
    REPO, "outputs/adu_wei_jun_occupations/paper_occupation_shift.png"
)
PAPER_PDF = os.path.join(
    REPO, "outputs/adu_wei_jun_occupations/paper_occupation_shift.pdf"
)


def clean_phrase(text):
    text = re.sub(r"\s+", " ", text).strip(" ,.;:'\"()[]{}")
    return text if re.fullmatch(r"[A-Za-z][A-Za-z -]{2,}", text) else None


def top_span_phrases(records, tokenizer, limit=4):
    candidates = {}
    for choices in itertools.product(*[range(len(record["candidate_ids"])) for record in records]):
        token_ids = [record["candidate_ids"][choice] for record, choice in zip(records, choices)]
        phrase = clean_phrase(tokenizer.decode(token_ids))
        if phrase is None:
            continue
        log_score = sum(
            math.log(max(record["probabilities"][choice], 1e-30))
            for record, choice in zip(records, choices)
        )
        key = phrase.casefold()
        if key not in candidates or log_score > candidates[key][1]:
            candidates[key] = (phrase, log_score)
    ranked = sorted(candidates.values(), key=lambda item: item[1], reverse=True)[:limit]
    if not ranked:
        return []
    normalizer = np.logaddexp.reduce([score for _, score in ranked])
    return [(phrase, math.exp(score - normalizer)) for phrase, score in ranked]


def top_single_words(record, tokenizer, limit=4):
    candidates = {}
    for token_id, probability in zip(record["candidate_ids"], record["probabilities"]):
        decoded = tokenizer.decode([token_id])
        if token_id != record["chosen_id"] and not decoded[:1].isspace():
            continue
        phrase = clean_phrase(decoded)
        if phrase is None:
            continue
        key = phrase.casefold()
        if key not in candidates or probability > candidates[key][1]:
            candidates[key] = (phrase, probability)
    ranked = sorted(candidates.values(), key=lambda item: item[1], reverse=True)[:limit]
    total = sum(probability for _, probability in ranked) or 1.0
    return [(phrase, probability / total) for phrase, probability in ranked]


def occupation_predictions(side, tokenizer):
    records = side["records"]
    return {
        "Father occupation": top_span_phrases(records[:3], tokenizer),
        "Mother occupation": top_single_words(records[3], tokenizer),
    }


def committed_phrase(records, tokenizer):
    token_ids = [record["chosen_id"] for record in records]
    probability = 1.0
    for record in records:
        chosen_index = record["candidate_ids"].index(record["chosen_id"])
        probability *= record["probabilities"][chosen_index]
    return tokenizer.decode(token_ids).strip(), probability


def draw_paper_summary(report, tokenizer):
    before_father = committed_phrase(report["before"]["records"][:3], tokenizer)
    after_father = committed_phrase(report["after"]["records"][:3], tokenizer)
    before_mother = committed_phrase(report["before"]["records"][3:], tokenizer)
    after_mother = committed_phrase(report["after"]["records"][3:], tokenizer)
    rows = [
        ("Father occupation", "Disc Jockey", before_father, after_father),
        ("Mother occupation", "Photographer", before_mother, after_mother),
    ]

    fig, axis = plt.subplots(figsize=(13, 5.8))
    axis.axis("off")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.text(0.34, 0.89, "Before ADU", ha="center", fontsize=15, fontweight="bold")
    axis.text(0.74, 0.89, "After ADU", ha="center", fontsize=15, fontweight="bold")

    for y, (parent, gold, before, after) in zip((0.63, 0.28), rows):
        axis.text(0.015, y, parent, va="center", fontsize=12, fontweight="bold")
        axis.text(
            0.34,
            y,
            f"{before[0]}\nP = {before[1]:.3f}",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="#A63D00",
            bbox={"boxstyle": "round,pad=0.55", "fc": "#FFF1E8", "ec": "#D55E00", "lw": 2},
        )
        axis.annotate(
            "",
            xy=(0.59, y),
            xytext=(0.47, y),
            arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#555555"},
        )
        axis.text(0.53, y + 0.055, "DiffusionADU", ha="center", fontsize=10)
        axis.text(
            0.74,
            y,
            f"{after[0]}\nP = {after[1]:.3f}",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="#005A87",
            bbox={"boxstyle": "round,pad=0.55", "fc": "#EAF6FB", "ec": "#0072B2", "lw": 2},
        )
        axis.text(0.74, y - 0.13, f"Gold: {gold}", ha="center", va="center", fontsize=10, color="#555555")

    fig.suptitle(
        "Occupation predictions for Wei-Jun Chen's parents",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    axis.text(
        0.5,
        0.06,
        "Complete words/phrases reconstructed from the tokens committed at generation time. "
        "P is the product of committed-token probabilities.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.savefig(PAPER_OUTPUT, dpi=240, bbox_inches="tight")
    fig.savefig(PAPER_PDF, bbox_inches="tight")
    plt.close(fig)


def draw_bars(axis, predictions, title, gold):
    labels = [label for label, _ in predictions][::-1]
    values = [value for _, value in predictions][::-1]
    colors = ["#D55E00" if label.casefold() == gold.casefold() else "#0072B2" for label in labels]
    bars = axis.barh(labels, values, color=colors, alpha=0.9)
    axis.set_xlim(0, max(values + [0.01]) * 1.2)
    axis.set_title(title, fontsize=12, fontweight="bold")
    axis.set_xlabel("Normalized probability among displayed phrases")
    axis.grid(axis="x", alpha=0.18)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_width() + axis.get_xlim()[1] * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )


def main():
    with open(REPORT, encoding="utf-8") as handle:
        report = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    before = occupation_predictions(report["before"], tokenizer)
    after = occupation_predictions(report["after"], tokenizer)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    draw_bars(axes[0, 0], before["Father occupation"], "Before ADU", "Disc Jockey")
    draw_bars(axes[0, 1], after["Father occupation"], "After ADU", "Disc Jockey")
    draw_bars(axes[1, 0], before["Mother occupation"], "Before ADU", "Photographer")
    draw_bars(axes[1, 1], after["Mother occupation"], "After ADU", "Photographer")

    axes[0, 0].set_ylabel("Father occupation", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Mother occupation", fontsize=12, fontweight="bold")
    fig.suptitle(
        "Word-level occupation predictions before and after DiffusionADU\n"
        "Question: What were the occupations of Wei-Jun Chen's parents?",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.015,
        "Orange = gold occupation. Subtoken probabilities are composed into complete phrases.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    fig.savefig(PDF_OUTPUT, bbox_inches="tight")
    draw_paper_summary(report, tokenizer)

    print("Before father:", before["Father occupation"])
    print("After father:", after["Father occupation"])
    print("Before mother:", before["Mother occupation"])
    print("After mother:", after["Mother occupation"])
    print(OUTPUT)
    print(PDF_OUTPUT)
    print(PAPER_OUTPUT)
    print(PAPER_PDF)


if __name__ == "__main__":
    main()
