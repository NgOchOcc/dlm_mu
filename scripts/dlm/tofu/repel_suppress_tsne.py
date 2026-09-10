#!/usr/bin/env python
"""Diagnose whether suppress_gold merely swaps the secret for a SAME-SEMANTIC token.

RDU's ``suppress_gold`` sets the gold token logit to -inf at each selected
forget position (A_hat), so the distillation anchor / repulsion cannot copy the secret
back. This script tests the failure mode: if forbidding the gold token just makes the
teacher pick a NEAR-SYNONYM, then suppression removes the surface token but NOT the
semantic content -- and the repulsion in loss_f (which lives in the output-embedding
space E) is fighting a target that is already semantically next to the gold.

At every content A_hat position we take, in the CLEAN gold context (a single teacher
forward, no generation needed):
    gold  = the true secret token
    alt   = argmax_{v != gold} teacher_logits           (gold forced to -inf)
and compare their rows in E (``model.get_output_embeddings().weight``), the EXACT space
loss_f's RBF repulsion operates in.

Outputs a two-panel figure:
  * t-SNE (cosine metric) of {E[gold]} ∪ {E[alt]}, gold vs alt colored, each pair joined
    by a thin segment. Short/overlapping segments => the fallback lands in the gold's own
    semantic neighborhood.
  * histogram of cos(E[gold], E[alt]) vs a baseline cos(E[gold], E[random other gold]).
    A right-shifted "suppress" distribution is the quantitative evidence.

    CUDA_VISIBLE_DEVICES=5 python scripts/dlm/tofu/repel_suppress_tsne.py [--n 150]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for repel_gendistill_viz

from datasets import load_dataset  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from repel_gendistill_viz import build_trainer, tokenize  # noqa: E402

_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def collect(t, tok, ds, n, device):
    """Per content-A_hat position: (gold_id, alt_id, gold_surf, alt_surf, p_gold, p_alt)."""
    E = t.ref_model.get_output_embeddings().weight.detach().float()[: t.vocab_size]
    rows = []
    with torch.no_grad():
        for i in range(min(n, len(ds))):
            q, a = ds[i]["question"], ds[i]["answer"]
            input_ids, answer_mask, attn, _ = tokenize(tok, q, a, device)
            a_hat = t._select_forget_positions(input_ids, answer_mask, attn)
            # clean-context teacher logits: isolates the suppression effect (full true
            # context, gold visible) from any query-blanking / generation confound.
            logits = t._teacher_logits(input_ids, attn).float()[..., : t.vocab_size]
            probs = F.softmax(logits, dim=-1)
            pos = (a_hat[0] & answer_mask[0]).nonzero(as_tuple=True)[0].tolist()
            for p in pos:
                gid = int(input_ids[0, p])
                gsurf = tok.decode([gid])
                if not _WORD_RE.search(gsurf):
                    continue  # only content (word) tokens carry a "secret"
                lg = logits[0, p].clone()
                lg[gid] = float("-inf")
                aid = int(lg.argmax())
                rows.append({
                    "ex": i, "pos": p,
                    "gold_id": gid, "alt_id": aid,
                    "gold": gsurf, "alt": tok.decode([aid]),
                    "p_gold": float(probs[0, p, gid]),
                    "p_alt": float(probs[0, p, aid]),
                })
            print(f"[{i}] A_hat_content={len(pos)} cum_pairs={len(rows)}", flush=True)
    return rows, E


def cos(E, ids_a, ids_b):
    a = F.normalize(E[torch.tensor(ids_a)], dim=-1)
    b = F.normalize(E[torch.tensor(ids_b)], dim=-1)
    return (a * b).sum(-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saves/finetune/tofu_LLaDA-8B-Instruct_retain100_150epochs")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--select_method", default="topk")
    ap.add_argument("--topk_frac", type=float, default=0.8)
    ap.add_argument("--sel_mc", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/repel_suppress_tsne.png")
    args = ap.parse_args()
    os.chdir(REPO)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # build_trainer needs these arg fields; generation knobs are unused here.
    ba = SimpleNamespace(select_method=args.select_method, topk_frac=args.topk_frac,
                         sel_mc=args.sel_mc, gen_steps=1, keep_answer=False,
                         keep_subject_name=False)
    t = build_trainer(model, tok, ba)

    ds = load_dataset("locuslab/TOFU", "forget10", split="train")
    rows, E = collect(t, tok, ds, args.n, device)
    if len(rows) < 5:
        raise SystemExit(f"too few content A_hat positions ({len(rows)})")

    gold_ids = [r["gold_id"] for r in rows]
    alt_ids = [r["alt_id"] for r in rows]
    # suppress pair similarity vs baseline (gold vs a shuffled other gold: same token pool)
    perm = np.random.permutation(len(gold_ids))
    perm[perm == np.arange(len(perm))] = (perm[perm == np.arange(len(perm))] + 1) % len(perm)
    cos_supp = cos(E, gold_ids, alt_ids)
    cos_base = cos(E, gold_ids, [gold_ids[j] for j in perm])

    # fraction where the forbidden-fallback is a real word (not subword/punct)
    frac_word = np.mean([1.0 if _WORD_RE.search(r["alt"]) else 0.0 for r in rows])
    # fraction where alt is merely a case/whitespace variant of gold (same word, e.g.
    # 'civil'->'Civil'): suppression did not even change the SECRET, only its surface form.
    frac_case = np.mean([1.0 if r["gold"].strip().casefold() == r["alt"].strip().casefold()
                         else 0.0 for r in rows])

    print("\n=== suppress_gold semantic-leak diagnostic ===")
    print(f"content A_hat positions: {len(rows)}")
    print(f"cos(gold, suppress-alt)  mean={cos_supp.mean():.3f}  median={np.median(cos_supp):.3f}")
    print(f"cos(gold, random-gold)   mean={cos_base.mean():.3f}  median={np.median(cos_base):.3f}")
    print(f"alt is a real word: {frac_word:.1%}")
    print(f"alt is a case/whitespace variant of gold (same word): {frac_case:.1%}")
    print("examples (gold -> alt):")
    for r in rows[:25]:
        print(f"   {r['gold']!r:>18} -> {r['alt']!r:<18} "
              f"cos={cos(E,[r['gold_id']],[r['alt_id']])[0]:.2f} "
              f"p_gold={r['p_gold']:.2f} p_alt={r['p_alt']:.2f}")

    # ---- t-SNE of gold ∪ alt (cosine) ----
    ids = gold_ids + alt_ids
    emb = E[torch.tensor(ids)].cpu().numpy()
    perplex = max(5, min(30, len(gold_ids) - 1))
    xy = TSNE(n_components=2, metric="cosine", init="pca", perplexity=perplex,
              random_state=args.seed).fit_transform(emb)
    g_xy, a_xy = xy[: len(gold_ids)], xy[len(gold_ids):]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    for i in range(len(gold_ids)):
        ax1.plot([g_xy[i, 0], a_xy[i, 0]], [g_xy[i, 1], a_xy[i, 1]],
                 "-", color="#bbb", lw=0.4, zorder=1)
    ax1.scatter(g_xy[:, 0], g_xy[:, 1], s=18, c="#1f77b4", label="gold (secret)", zorder=3)
    ax1.scatter(a_xy[:, 0], a_xy[:, 1], s=18, c="#ff7f0e", marker="^",
                label="suppress-alt (gold=-inf argmax)", zorder=3)
    ax1.set_title(f"t-SNE (cosine) of E rows · {len(gold_ids)} A_hat positions\n"
                  f"short segments = fallback in gold's semantic neighborhood")
    ax1.legend(loc="best", fontsize=9)
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2.hist(cos_base, bins=40, alpha=0.6, color="#888", label=f"baseline gold·random-gold (μ={cos_base.mean():.2f})")
    ax2.hist(cos_supp, bins=40, alpha=0.6, color="#d62728", label=f"gold·suppress-alt (μ={cos_supp.mean():.2f})")
    ax2.axvline(cos_supp.mean(), color="#d62728", ls="--", lw=1)
    ax2.axvline(cos_base.mean(), color="#555", ls="--", lw=1)
    ax2.set_xlabel("cosine similarity in E space")
    ax2.set_ylabel("count")
    ax2.set_title("suppress-fallback vs random baseline\nright-shift => same-semantic swap")
    ax2.legend(loc="best", fontsize=9)

    fig.suptitle("Does suppress_gold remove the SECRET or just the TOKEN? "
                 "(LLaDA SFT, forget10, clean context)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
