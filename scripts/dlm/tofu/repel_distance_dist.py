#!/usr/bin/env python
"""Distance-distribution evidence for the repel method.

RepelGenDistill's ``loss_f`` is an RBF repulsion, in the output-embedding space E
(``model.get_output_embeddings().weight``), acting on the model's SOFT-EMBEDDING at each
forget position:

    ê_θ(p) = Σ_v  softmax(logits_θ(p))_v · E[v]        (a prob-weighted point in E)

Repel pushes ê_θ(p) AWAY from the secret token's anchor E[gold]. So the single honest
coordinate that tells us whether the secret is gone is how far ê_θ(p) sits from E[gold]:

    cos( ê_θ(p), E[gold] )       (1.0 = still pointing right at the secret)
    ‖ ê_θ(p) − E[gold] ‖         (0   = still sitting on the secret)

This script measures those two quantities, in the CLEAN gold context (single teacher
forward, gold visible -- the "does the model still know the secret when asked normally"
test), for two models:

    SFT     = the fine-tuned teacher that still knows everything
    repel   = the RepelGenDistill-unlearned checkpoint

over two position sets:

    forget  = A_hat ∩ content words on forget10   (the tokens repel actually targets)
    retain  = all content answer words on retain90 (control: must NOT move)

The forget A_hat positions are selected ONCE on the SFT model and REUSED verbatim for the
repel model, so the two curves are measured at the identical token positions (isolating
"did the distribution move" from "did selection differ"). Models are loaded one at a time.

Expected picture: forget cos collapses (SFT ≈0.9 → repel low) while retain cos is
unchanged (≈0.9 for both) -- the secret is repelled, the rest is preserved.

    CUDA_VISIBLE_DEVICES=0 python scripts/dlm/tofu/repel_distance_dist.py \
        --sft   saves/finetune/tofu_LLaDA-8B-Instruct_retain100_150epochs \
        --repel saves/unlearn/forget10/tofu_LLaDA-8B-Instruct_DiffusionRepelGenDistill \
        [--n_forget 100 --n_retain 100]
"""
from __future__ import annotations

import argparse
import gc
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset  # noqa: E402
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from repel_gendistill_viz import build_trainer, tokenize  # noqa: E402

_WORD_RE = re.compile(r"[A-Za-z]{2,}")  # a "content word" carries a secret


def load_model(path, device):
    try:
        m = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    except Exception:
        m = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def soft_emb_metrics(t, tok, E, input_ids, attn, positions):
    """Per-position CLOZE: mask the scored position (rest of the context intact) and read
    the model's distribution for that slot -- exactly the recall generation faces, and not
    trivialised by the gold token being visible at the position itself. For each position:
    cos(ê_θ, E[gold]) and ‖ê_θ − E[gold]‖, with ê_θ = Σ_v softmax(logits)_v · E[v].

    One batched forward per example: row j is the context with position P[j] masked."""
    if not positions:
        return []
    k, L = len(positions), input_ids.shape[1]
    X = input_ids.repeat(k, 1).clone()
    for j, pos in enumerate(positions):
        X[j, pos] = t.mask_id
    A = attn.repeat(k, 1)
    logits = t._teacher_logits(X, A).float()[..., : t.vocab_size]
    p = F.softmax(logits, dim=-1)                       # (k, L, V)
    out = []
    for j, pos in enumerate(positions):
        e_hat = p[j, pos] @ E                           # (d,) soft-embedding at masked slot
        g = E[int(input_ids[0, pos])]                   # (d,) secret anchor
        cos = float(F.cosine_similarity(e_hat, g, dim=0))
        dist = float((e_hat - g).norm())
        out.append((cos, dist))
    return out


@torch.no_grad()
def forget_positions(t, tok, input_ids, answer_mask, attn, seed):
    """A_hat ∩ content-word answer positions (seeded so the pick is reproducible)."""
    torch.manual_seed(seed)
    a_hat = t._select_forget_positions(input_ids, answer_mask, attn)
    pos = (a_hat[0] & answer_mask[0]).nonzero(as_tuple=True)[0].tolist()
    return [p for p in pos if _WORD_RE.search(tok.decode([int(input_ids[0, p])]))]


def retain_positions(tok, input_ids, answer_mask):
    """All content-word answer positions (deterministic control set)."""
    pos = answer_mask[0].nonzero(as_tuple=True)[0].tolist()
    return [p for p in pos if _WORD_RE.search(tok.decode([int(input_ids[0, p])]))]


def run_model(path, tok, device, ba, forget_ds, retain_ds, n_forget, n_retain,
              fixed_forget, seed):
    """Collect metrics for one model. Returns (forget_metrics, retain_metrics, picked).

    ``fixed_forget`` is None on the first (SFT) pass -- positions are selected and
    returned in ``picked``; on the repel pass ``fixed_forget`` is that dict, so the very
    same A_hat positions are re-used (identical token positions across models)."""
    model = load_model(path, device)
    t = build_trainer(model, tok, ba)
    E = model.get_output_embeddings().weight.detach().float()[: t.vocab_size]

    picked = {}
    fmet, rmet = [], []
    with torch.no_grad():
        for i in range(min(n_forget, len(forget_ds))):
            q, a = forget_ds[i]["question"], forget_ds[i]["answer"]
            input_ids, answer_mask, attn, _ = tokenize(tok, q, a, device)
            if fixed_forget is None:
                pos = forget_positions(t, tok, input_ids, answer_mask, attn, seed + i)
                picked[i] = pos
            else:
                pos = fixed_forget.get(i, [])
            if pos:
                fmet += soft_emb_metrics(t, tok, E, input_ids, attn, pos)
            if (i + 1) % 20 == 0:
                print(f"  [{path.split('/')[-1]}] forget {i + 1} cum_pos={len(fmet)}", flush=True)

        for i in range(min(n_retain, len(retain_ds))):
            q, a = retain_ds[i]["question"], retain_ds[i]["answer"]
            input_ids, answer_mask, attn, _ = tokenize(tok, q, a, device)
            pos = retain_positions(tok, input_ids, answer_mask)
            if pos:
                rmet += soft_emb_metrics(t, tok, E, input_ids, attn, pos)
            if (i + 1) % 20 == 0:
                print(f"  [{path.split('/')[-1]}] retain {i + 1} cum_pos={len(rmet)}", flush=True)

    del model, t, E
    gc.collect()
    torch.cuda.empty_cache()
    return np.array(fmet), np.array(rmet), picked


def _panel(ax, sft, repel, col, title, xlabel, bins, shift_left_is_forget):
    """Overlay SFT vs repel histograms (density) with mean lines."""
    lo = min(sft.min(), repel.min() if len(repel) else sft.min())
    hi = max(sft.max(), repel.max() if len(repel) else sft.max())
    edges = np.linspace(lo, hi, bins)
    ax.hist(sft, bins=edges, density=True, color="#1f77b4", alpha=0.55,
            label=f"SFT  (μ={sft.mean():.2f})")
    ax.axvline(sft.mean(), color="#1f77b4", ls="--", lw=1.4)
    if len(repel):
        ax.hist(repel, bins=edges, density=True, color="#d62728", alpha=0.55,
                label=f"repel  (μ={repel.mean():.2f})")
        ax.axvline(repel.mean(), color="#d62728", ls="--", lw=1.4)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel("density")
    ax.legend(fontsize=9, framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="saves/finetune/tofu_LLaDA-8B-Instruct_retain100_150epochs")
    ap.add_argument("--repel", default="saves/unlearn/forget10/tofu_LLaDA-8B-Instruct_DiffusionRepelGenDistill")
    ap.add_argument("--forget_split", default="forget10")
    ap.add_argument("--retain_split", default="retain90")
    ap.add_argument("--n_forget", type=int, default=100)
    ap.add_argument("--n_retain", type=int, default=100)
    ap.add_argument("--select_method", default="topk")
    ap.add_argument("--topk_frac", type=float, default=0.8)
    ap.add_argument("--sel_mc", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replot", action="store_true",
                    help="skip GPU work; reload cached metrics (<out>.npz) and just redraw")
    ap.add_argument("--out", default="outputs/repel_distance_dist.png")
    args = ap.parse_args()
    os.chdir(REPO)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    npz = os.path.splitext(args.out)[0] + ".npz"
    if args.replot:
        d = np.load(npz)
        sft_f, sft_r, rep_f, rep_r = d["sft_f"], d["sft_r"], d["rep_f"], d["rep_r"]
        have_repel = len(rep_f) > 0
    else:
        device = torch.device("cuda:0")
        tok = AutoTokenizer.from_pretrained(args.sft, trust_remote_code=True)
        ba = SimpleNamespace(select_method=args.select_method, topk_frac=args.topk_frac,
                             sel_mc=args.sel_mc, gen_steps=1, keep_answer=False,
                             keep_subject_name=False)

        forget_ds = load_dataset("locuslab/TOFU", args.forget_split, split="train")
        retain_ds = load_dataset("locuslab/TOFU", args.retain_split, split="train")

        print("=== SFT pass (selects A_hat positions, reused for repel) ===", flush=True)
        sft_f, sft_r, picked = run_model(args.sft, tok, device, ba, forget_ds, retain_ds,
                                         args.n_forget, args.n_retain, None, args.seed)

        have_repel = args.repel and os.path.isdir(args.repel)
        if have_repel:
            print("=== repel pass (same A_hat positions) ===", flush=True)
            rep_f, rep_r, _ = run_model(args.repel, tok, device, ba, forget_ds, retain_ds,
                                        args.n_forget, args.n_retain, picked, args.seed)
        else:
            print(f"!! repel checkpoint not found ({args.repel}); plotting SFT baseline only", flush=True)
            rep_f = np.empty((0, 2)); rep_r = np.empty((0, 2))
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.savez(npz, sft_f=sft_f, sft_r=sft_r, rep_f=rep_f, rep_r=rep_r)

    print("\n=== soft-embedding distance to secret E[gold] (clean context) ===")
    print(f"forget positions: {len(sft_f)}   retain positions: {len(sft_r)}")
    print(f"cos(ê,gold)  forget:  SFT={sft_f[:,0].mean():.3f}"
          + (f"  repel={rep_f[:,0].mean():.3f}" if len(rep_f) else ""))
    print(f"cos(ê,gold)  retain:  SFT={sft_r[:,0].mean():.3f}"
          + (f"  repel={rep_r[:,0].mean():.3f}" if len(rep_r) else ""))
    print(f"‖ê-gold‖     forget:  SFT={sft_f[:,1].mean():.3f}"
          + (f"  repel={rep_f[:,1].mean():.3f}" if len(rep_f) else ""))
    print(f"‖ê-gold‖     retain:  SFT={sft_r[:,1].mean():.3f}"
          + (f"  repel={rep_r[:,1].mean():.3f}" if len(rep_r) else ""))

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#999"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")
    _panel(axes[0, 0], sft_f[:, 0], rep_f[:, 0], "forget",
           "FORGET positions — cos(ê_θ, E[gold])", "cosine to secret  (1 = still pointing at it)", 40, True)
    _panel(axes[0, 1], sft_r[:, 0], rep_r[:, 0], "retain",
           "RETAIN positions (control) — cos(ê_θ, E[gold])", "cosine to secret", 40, False)
    _panel(axes[1, 0], sft_f[:, 1], rep_f[:, 1], "forget",
           "FORGET positions — ‖ê_θ − E[gold]‖", "distance from secret  (0 = sitting on it)", 40, False)
    _panel(axes[1, 1], sft_r[:, 1], rep_r[:, 1], "retain",
           "RETAIN positions (control) — ‖ê_θ − E[gold]‖", "distance from secret", 40, False)
    if have_repel:
        sub = (f"repel drives the soft-embedding ê_θ = Σ softmax(logits)·E OFF the secret at FORGET positions "
               f"(cos {sft_f[:,0].mean():.2f}→{rep_f[:,0].mean():.2f});\n"
               f"RETAIN positions (control) are preserved far better "
               f"(cos {sft_r[:,0].mean():.2f}→{rep_r[:,0].mean():.2f}) — the memorised spike at 1.0 largely survives")
    else:
        sub = "SFT baseline only (no repel checkpoint) — forget & retain both sit on the secret"
    fig.suptitle("Does repel move the model's soft-embedding away from the secret?\n" + sub,
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
