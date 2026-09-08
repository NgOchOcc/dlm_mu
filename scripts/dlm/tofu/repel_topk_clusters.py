#!/usr/bin/env python
"""Show that the top-k tokens the model would GENERATE at each forget position form a
tight SAME-SEMANTIC cluster -- so forbidding the gold token (suppress_gold) is futile:
the whole top-k neighbourhood carries the same secret.

At each selected content A_hat position (clean gold context, single teacher forward) we
take the teacher's top-k candidate tokens and their rows in E
(``model.get_output_embeddings().weight``, the space loss_f's RBF repulsion lives in).
Pooling a handful of representative positions, we t-SNE the candidate embeddings into 2D
and 3D. Points are coloured by their position: if each position's top-k lands in one
compact, well-separated blob (e.g. {civil, Civil, engineering, Engineer, ...}), the model
is choosing among synonyms/inflections of the same concept -- exactly what makes killing
a single gold token id ineffective.

    CUDA_VISIBLE_DEVICES=6 python scripts/dlm/tofu/repel_topk_clusters.py [--n_pos 12 --topk 8]

Writes two figures: ``outputs/repel_topk_clusters_2d.png`` and ``..._3d.png``.
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  (registers 3d projection)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from repel_gendistill_viz import build_trainer, tokenize  # noqa: E402

_WORD_RE = re.compile(r"^[A-Za-z]{4,}$")
_CAND_RE = re.compile(r"^[A-Za-z]{3,}$")  # candidate must be a plain word (no subword frags)
_STOP = {"with", "being", "that", "this", "which", "their", "there", "from", "have",
         "were", "will", "would", "into", "been", "they", "them", "your", "when",
         "what", "then", "these", "those", "about", "also", "some", "such", "than",
         "over", "under", "while", "where", "both", "each", "other", "more", "most"}


def collect(t, tok, ds, n_scan, n_pos, topk, device, E, min_coh):
    """Pick <=n_pos distinct content A_hat positions (<=1 per example, sampled evenly
    across the whole dataset so concepts come from many authors and separate cleanly).

    Keep only positions whose top-k form a COHERENT set (mean pairwise cosine in E >=
    min_coh): incoherent picks are name/subword-fragment artifacts (e.g. 'ammad' ->
    aku/akh/ova) that are neither synonyms nor visually separable -- the very property
    we are illustrating filters them out."""
    picks, seen = [], set()
    idxs = sorted(set(np.linspace(0, len(ds) - 1, n_scan).astype(int).tolist()))
    with torch.no_grad():
        for i in idxs:
            if len(picks) >= n_pos:
                break
            q, a = ds[i]["question"], ds[i]["answer"]
            input_ids, answer_mask, attn, _ = tokenize(tok, q, a, device)
            a_hat = t._select_forget_positions(input_ids, answer_mask, attn)
            logits = t._teacher_logits(input_ids, attn).float()[..., : t.vocab_size]
            pos = (a_hat[0] & answer_mask[0]).nonzero(as_tuple=True)[0].tolist()
            for p in pos:                       # at most ONE position from this example
                gid = int(input_ids[0, p])
                gsurf = tok.decode([gid]).strip()
                key = gsurf.casefold()
                if not _WORD_RE.match(gsurf) or key in _STOP or key in seen:
                    continue
                # take the top word-candidates (skip subword frags / punctuation / CJK)
                topv, topi = logits[0, p].topk(40)
                cand_ids, cand_surf = [], []
                for c in topi.tolist():
                    s = tok.decode([c]).strip()
                    if _CAND_RE.match(s):
                        cand_ids.append(c); cand_surf.append(s)
                    if len(cand_ids) >= topk:
                        break
                if len(cand_ids) < 4:
                    continue
                # coherence gate: mean off-diagonal cosine of this position's candidates
                rows = F.normalize(E[torch.tensor(cand_ids)], dim=-1)
                sim = rows @ rows.T
                off = sim[~torch.eye(len(cand_ids), dtype=torch.bool)]
                coh = float(off.mean())
                if coh < min_coh:
                    print(f"skip ex{i} gold={gsurf!r} coh={coh:.2f} (fragment)", flush=True)
                    continue
                seen.add(key)
                picks.append({
                    "ex": int(i), "gold": gsurf, "gold_id": gid,
                    "cand_ids": cand_ids, "cand_surf": cand_surf, "coh": coh,
                })
                print(f"pick[{len(picks)}] ex{i} gold={gsurf!r} coh={coh:.2f} topk={cand_surf}", flush=True)
                break
    return picks


def within_between(E, picks):
    """Mean within-cluster vs between-cluster PAIRWISE cosine (apples-to-apples: both are
    raw token-pair cosines, so centroid-averaging bias cannot inflate the baseline)."""
    ids, owner = [], []
    for ci, pk in enumerate(picks):
        ids += pk["cand_ids"]; owner += [ci] * len(pk["cand_ids"])
    rows = F.normalize(E[torch.tensor(ids)], dim=-1)
    sim = (rows @ rows.T)
    own = torch.tensor(owner)
    same = own[:, None] == own[None, :]
    eye = torch.eye(len(ids), dtype=torch.bool)
    within = float(sim[same & ~eye].mean())
    between = float(sim[~same].mean())
    return within, between


def _label(s):
    """Drop non-ASCII (CJK top-k tokens the plot font cannot render) for clean labels."""
    a = "".join(c for c in s if ord(c) < 128).strip()
    return a if a else "·"


def _hull(pts):
    """Convex hull vertices (Andrew's monotone chain, numpy only, no scipy)."""
    p = sorted(map(tuple, pts))
    if len(p) < 3:
        return np.array(p)
    def half(ps):
        h = []
        for q in ps:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) \
                    - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(q)
        return h[:-1]
    return np.array(half(p) + half(p[::-1]))


def cluster_region(ax, pts, color):
    """Soft convex-hull fill around a cluster's DENSE CORE (drop the farthest ~30% from the
    median so a single t-SNE-scattered candidate can't stretch the hull across the figure)."""
    if len(pts) < 3:
        return
    med = np.median(pts, 0)
    keep = np.argsort(((pts - med) ** 2).sum(1))[: max(3, int(np.ceil(0.6 * len(pts))))]
    core = pts[keep]
    hull = _hull(core)
    cen = hull.mean(0)
    hull = cen + (hull - cen) * 1.15       # pad so core points sit inside the shaded region
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(hull, closed=True, facecolor=color, alpha=0.13,
                         edgecolor=color, lw=1.2, zorder=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saves/finetune/tofu_LLaDA-8B-Instruct_retain100_150epochs")
    ap.add_argument("--n_scan", type=int, default=60)
    ap.add_argument("--n_pos", type=int, default=12)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--min_coh", type=float, default=0.12,
                    help="min mean pairwise cosine among a position's top-k to keep it "
                         "(drops incoherent name/subword-fragment positions)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out2d", default="outputs/repel_topk_clusters_2d.png")
    ap.add_argument("--out3d", default="outputs/repel_topk_clusters_3d.png")
    args = ap.parse_args()
    os.chdir(REPO)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ba = SimpleNamespace(select_method="topk", topk_frac=0.8, sel_mc=16,
                         gen_steps=1, keep_answer=False, keep_subject_name=False)
    t = build_trainer(model, tok, ba)
    ds = load_dataset("locuslab/TOFU", "forget10", split="train")
    E = model.get_output_embeddings().weight.detach().float()[: t.vocab_size]
    picks = collect(t, tok, ds, args.n_scan, args.n_pos, args.topk, device, E, args.min_coh)
    if len(picks) < 3:
        raise SystemExit(f"too few positions ({len(picks)})")

    wc, bc = within_between(E, picks)
    print(f"\nwithin-cluster cos (top-k of same position): {wc:.3f}")
    print(f"between-cluster cos (across positions):       {bc:.3f}")

    # pooled candidate embeddings + labels
    all_ids, owner, surfaces, is_gold = [], [], [], []
    for ci, pk in enumerate(picks):
        for cid, cs in zip(pk["cand_ids"], pk["cand_surf"]):
            all_ids.append(cid); owner.append(ci); surfaces.append(cs)
            is_gold.append(cid == pk["gold_id"])
    emb = E[torch.tensor(all_ids)].cpu().numpy()
    owner = np.array(owner); is_gold = np.array(is_gold)
    perplex = max(5, min(30, len(all_ids) - 1))
    cmap = plt.get_cmap("tab20" if len(picks) > 10 else "tab10")
    colors = [cmap(i % cmap.N) for i in range(len(picks))]
    metric_txt = (f"within-cluster cos = {wc:.2f}   vs   between-cluster cos = {bc:.2f}   "
                  f"(random content-word pairs ≈ 0.05)\n"
                  f"→ each position's top-{args.topk} sits in ONE semantic region "
                  f"(killing the gold id alone leaves the concept intact)")

    # ---------- 2D ----------
    xy = TSNE(n_components=2, metric="cosine", init="pca", perplexity=perplex,
              random_state=args.seed).fit_transform(emb)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#999"})
    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor("white")
    for ci, pk in enumerate(picks):
        m = owner == ci
        pts = xy[m]
        lab = np.median(pts, 0)  # label at median: robust to a stray candidate
        cluster_region(ax, pts, colors[ci])   # soft hull groups the cluster (no crossing spokes)
        ax.scatter(pts[:, 0], pts[:, 1], s=95, color=colors[ci], edgecolor="white",
                   linewidth=0.8, zorder=3)
        gm = m & is_gold
        if gm.any():
            ax.scatter(xy[gm, 0], xy[gm, 1], s=430, marker="*", color=colors[ci],
                       edgecolor="black", linewidth=1.1, zorder=5)
        ax.text(lab[0], lab[1], pk["gold"], fontsize=13, fontweight="bold",
                color=colors[ci], ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[ci], alpha=0.9))
    for j, s in enumerate(surfaces):
        if not is_gold[j]:
            ax.annotate(_label(s), (xy[j, 0], xy[j, 1]), fontsize=7, color="#333",
                        alpha=0.85, xytext=(3, 3), textcoords="offset points", zorder=4)
    ax.set_title("Top-k generation candidates cluster by MEANING at each forget position",
                 fontsize=15, fontweight="bold", pad=52)
    ax.text(0.5, 1.012, metric_txt, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color="#444")
    ax.scatter([], [], s=200, marker="*", color="#555", edgecolor="black", label="gold (secret) token")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out2d), exist_ok=True)
    fig.savefig(args.out2d, dpi=150)
    print(f"wrote {args.out2d}")

    # ---------- 3D ----------
    xyz = TSNE(n_components=3, metric="cosine", init="pca", perplexity=perplex,
               random_state=args.seed).fit_transform(emb)
    gcen = xyz.mean(0)
    fig = plt.figure(figsize=(12.5, 10))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    for ci, pk in enumerate(picks):
        m = owner == ci
        pts = xyz[m]
        med = np.median(pts, 0)
        # spokes only among the dense core (drop farthest ~40%) so a stray t-SNE point
        # can't fling a line across the whole cube
        d = ((pts - med) ** 2).sum(1)
        core = pts[np.argsort(d)[: max(3, int(np.ceil(0.6 * len(pts))))]]
        ccen = core.mean(0)
        for pt in core:
            ax.plot([ccen[0], pt[0]], [ccen[1], pt[1]], [ccen[2], pt[2]],
                    color=colors[ci], lw=0.5, alpha=0.55, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=55, color=colors[ci],
                   edgecolor="white", linewidth=0.5, depthshade=True)
        gm = m & is_gold
        if gm.any():
            ax.scatter(xyz[gm, 0], xyz[gm, 1], xyz[gm, 2], s=300, marker="*",
                       color=colors[ci], edgecolor="black", linewidth=1.0)
        # push the label radially outward from the global centre so labels don't pile up
        lab = med + (med - gcen) * 0.55
        ax.text(lab[0], lab[1], lab[2], pk["gold"], fontsize=11, fontweight="bold",
                color=colors[ci], ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=colors[ci], alpha=0.85))
    ax.set_title("Top-k generation candidates in 3-D semantic space\n"
                 f"within-cluster cos {wc:.2f} vs between-cluster cos {bc:.2f} "
                 "— top-k of a position = one concept",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2"); ax.set_zlabel("t-SNE-3")
    ax.view_init(elev=18, azim=-60)
    ax.xaxis.pane.set_alpha(0.04); ax.yaxis.pane.set_alpha(0.04); ax.zaxis.pane.set_alpha(0.04)
    fig.tight_layout()
    fig.savefig(args.out3d, dpi=150)
    print(f"wrote {args.out3d}")


if __name__ == "__main__":
    main()
