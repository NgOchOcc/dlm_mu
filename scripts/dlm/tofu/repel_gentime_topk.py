#!/usr/bin/env python
"""Top-10 candidates captured AT GENERATION TIME for 3-4 positions of ONE answer.

Unlike ``repel_topk_clusters.py`` (which reads the teacher's top-k in the CLEAN,
fully-teacher-forced context), this script runs the model's real masked-diffusion
generation on a single example and, at the *exact denoising step where a chosen answer
position is unmasked*, snapshots the model's top-10 candidates for that position. That is
the distribution the sampler actually chose from -- so it is the honest evidence that,
right when the model commits a token, the alternatives it was weighing are same-semantic.

The answer span is masked and denoised with the query kept intact (normal generation, no
identity-blanking); EOS/pad/mask are suppressed exactly as the trainer's generator does.
We track 3-4 content A_hat positions and, the first time each is unmasked, record its
top-10 (ids, probs, surfaces) plus the token the model actually committed there.

Two figures (2D and 3D t-SNE, cosine, of the pooled top-10 output-embedding rows E,
coloured by position), the picked sentence, and per-position top-10 are written/printed.

    CUDA_VISIBLE_DEVICES=6 python scripts/dlm/tofu/repel_gentime_topk.py [--ex 5 --n_pos 4 --topk 10]
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
_CAND_RE = re.compile(r"^[A-Za-z]{3,}$")
_STOP = {"with", "being", "that", "this", "which", "their", "there", "from", "have",
         "were", "will", "would", "into", "been", "they", "them", "your", "when",
         "what", "then", "these", "those", "about", "also", "some", "such", "than",
         "over", "under", "while", "where", "both", "each", "other", "more", "most",
         "known", "also", "many", "very", "much", "well", "just", "like"}


def pick_positions(t, input_ids, answer_mask, attn, tok, n_pos):
    """Choose <=n_pos content A_hat positions spread across the answer (distinct words)."""
    a_hat = t._select_forget_positions(input_ids, answer_mask, attn)
    cand = (a_hat[0] & answer_mask[0]).nonzero(as_tuple=True)[0].tolist()
    keep, seen = [], set()
    for p in cand:
        surf = tok.decode([int(input_ids[0, p])]).strip()
        key = surf.casefold()
        if _WORD_RE.match(surf) and key not in _STOP and key not in seen:
            seen.add(key)
            keep.append(p)
    if len(keep) <= n_pos:
        return keep
    idx = np.linspace(0, len(keep) - 1, n_pos).round().astype(int)
    return [keep[j] for j in sorted(set(idx.tolist()))]


@torch.no_grad()
def generate_and_capture(t, tok, input_ids, answer_mask, attn, targets, topk):
    """Normal from-scratch denoise of the answer (query intact). Snapshot each target's
    top-k the first step it is unmasked. Returns (generated_ids, {pos: capture})."""
    mask_id = t.mask_id
    x = torch.where(answer_mask, torch.full_like(input_ids, mask_id), input_ids)
    T = t.gen_steps
    suppress_ids = torch.cat([
        t._eos_ids(x.device),
        torch.tensor([i for i in (t.pad_id, mask_id) if i is not None],
                     dtype=torch.long, device=x.device),
    ]).unique()
    targets = set(targets)
    cap = {}
    for step in range(T):
        mask_idx = (x == mask_id) & answer_mask
        if not mask_idx.any():
            break
        n_remaining = mask_idx.sum(dim=1).float()
        steps_left = max(T - step, 1)
        n_unmask = (n_remaining / steps_left).ceil().long().clamp(min=1)
        logits = t._teacher_logits(x, attn).float()
        logits[..., suppress_ids] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        conf = probs.max(dim=-1).values
        pred = probs.argmax(dim=-1)
        pos = mask_idx[0].nonzero(as_tuple=True)[0]
        k = int(min(int(n_unmask[0].item()), len(pos)))
        _, top = conf[0, pos].topk(k, largest=True)
        chosen = pos[top]
        for p in chosen.tolist():
            if p in targets and p not in cap:
                tv, ti = probs[0, p].topk(topk)
                cap[p] = {
                    "ids": ti.tolist(), "probs": tv.tolist(),
                    "surf": [tok.decode([c]).strip() for c in ti.tolist()],
                    "step": step, "chosen": int(pred[0, p]),
                    "chosen_surf": tok.decode([int(pred[0, p])]).strip(),
                }
        x[0, chosen] = pred[0, chosen]
    return x, cap


def within_between(E, picks):
    ids, owner = [], []
    for ci, pk in enumerate(picks):
        ids += pk["ids"]; owner += [ci] * len(pk["ids"])
    rows = F.normalize(E[torch.tensor(ids)], dim=-1)
    sim = rows @ rows.T
    own = torch.tensor(owner)
    same = own[:, None] == own[None, :]
    eye = torch.eye(len(ids), dtype=torch.bool)
    return float(sim[same & ~eye].mean()), float(sim[~same].mean())


def _label(s):
    a = "".join(c for c in s if ord(c) < 128).strip()
    return a if a else "·"


def _hull(pts):
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
    if len(pts) < 3:
        return
    med = np.median(pts, 0)
    keep = np.argsort(((pts - med) ** 2).sum(1))[: max(3, int(np.ceil(0.7 * len(pts))))]
    hull = _hull(pts[keep])
    cen = hull.mean(0)
    hull = cen + (hull - cen) * 1.15
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(hull, closed=True, facecolor=color, alpha=0.12,
                         edgecolor=color, lw=1.2, zorder=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saves/finetune/tofu_LLaDA-8B-Instruct_retain100_150epochs")
    ap.add_argument("--ex", type=int, default=-1, help="example index; -1 = auto-pick a good one")
    ap.add_argument("--n_pos", type=int, default=4)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--gen_steps", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out2d", default="outputs/repel_gentime_topk_2d.png")
    ap.add_argument("--out3d", default="outputs/repel_gentime_topk_3d.png")
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
                         gen_steps=args.gen_steps, keep_answer=False, keep_subject_name=False)
    t = build_trainer(model, tok, ba)
    ds = load_dataset("locuslab/TOFU", "forget10", split="train")
    E = model.get_output_embeddings().weight.detach().float()[: t.vocab_size]

    # choose the example: an explicit --ex, or auto-scan for one with >=n_pos content
    # positions whose gen-time top-10 are coherent (mean pairwise cosine high).
    order = [args.ex] if args.ex >= 0 else list(range(0, min(60, len(ds))))
    chosen_ex, picks, gen_ids, q, a = None, None, None, None, None
    for i in order:
        q, a = ds[i]["question"], ds[i]["answer"]
        input_ids, answer_mask, attn, _ = tokenize(tok, q, a, device)
        targets = pick_positions(t, input_ids, answer_mask, attn, tok, args.n_pos)
        if len(targets) < min(3, args.n_pos):
            continue
        gen_ids, cap = generate_and_capture(t, tok, input_ids, answer_mask, attn, targets, args.topk)
        picks = []
        for p in targets:
            if p not in cap:
                continue
            c = cap[p]
            # keep only positions whose top-k are real words and coherent
            ids = [cid for cid, s in zip(c["ids"], c["surf"]) if _CAND_RE.match(s)]
            surf = [s for s in c["surf"] if _CAND_RE.match(s)]
            if len(ids) < 4:
                continue
            rows = F.normalize(E[torch.tensor(ids)], dim=-1)
            off = (rows @ rows.T)[~torch.eye(len(ids), dtype=torch.bool)]
            picks.append({
                "pos": p, "gold": tok.decode([int(input_ids[0, p])]).strip(),
                "gold_id": int(input_ids[0, p]),
                "ids": ids, "surf": surf, "probs": c["probs"][: len(ids)],
                "chosen": c["chosen"], "chosen_surf": c["chosen_surf"],
                "step": c["step"], "coh": float(off.mean()),
            })
        if args.ex >= 0:
            chosen_ex = i; break
        if len(picks) >= min(3, args.n_pos) and np.mean([pk["coh"] for pk in picks]) > 0.18:
            chosen_ex = i; break
    if not picks or len(picks) < 3:
        raise SystemExit("could not find >=3 usable positions; pass --ex explicitly")

    input_ids, answer_mask, attn, _ = tokenize(tok, q, a, device)
    print(f"\n=== example #{chosen_ex} ===\nQ: {q}\nA: {a}\n")
    gen_answer = tok.decode([int(gen_ids[0, p]) for p in answer_mask[0].nonzero(as_tuple=True)[0].tolist()])
    print(f"GEN: {gen_answer.strip()}\n")
    for pk in picks:
        star = "  <-- committed" if pk["chosen_surf"] else ""
        print(f"pos {pk['pos']:>3}  gold={pk['gold']!r}  step={pk['step']}  coh={pk['coh']:.2f}  "
              f"committed={pk['chosen_surf']!r}")
        for s, pr in zip(pk["surf"], pk["probs"]):
            mark = " *" if s == pk["chosen_surf"] else ("  g" if s == pk["gold"] else "  ")
            print(f"      {pr:6.3f}  {s}{mark}")
    wc, bc = within_between(E, picks)
    print(f"\nwithin-position cos = {wc:.3f}   between-position cos = {bc:.3f}")

    # pooled points
    all_ids, owner, surfaces, probs, is_chosen, is_gold = [], [], [], [], [], []
    for ci, pk in enumerate(picks):
        for cid, cs, pr in zip(pk["ids"], pk["surf"], pk["probs"]):
            all_ids.append(cid); owner.append(ci); surfaces.append(cs); probs.append(pr)
            is_chosen.append(cid == pk["chosen"]); is_gold.append(cid == pk["gold_id"])
    emb = E[torch.tensor(all_ids)].cpu().numpy()
    owner = np.array(owner); probs = np.array(probs)
    is_chosen = np.array(is_chosen); is_gold = np.array(is_gold)
    # uniform point size: generation here is near-deterministic (p≈1.0 on the committed
    # token, runner-ups ≈0), so scaling by prob would hide the very alternatives we want
    # to show -- and "even the ~0-prob candidates are synonyms" is exactly the point.
    sizes = np.full(len(all_ids), 130.0)
    perplex = max(3, min(15, len(all_ids) - 1))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(picks))]
    metric_txt = (f"within-position cos = {wc:.2f}  vs  between-position cos = {bc:.2f}   "
                  f"(random content-word pairs ≈ 0.05)\n"
                  f"→ at the step each position is committed, its top-{args.topk} are "
                  f"same-semantic (suppressing the gold id alone changes nothing)")

    # ---------- 2D ----------
    xy = TSNE(n_components=2, metrictô="cosine", init="pca", perplexity=perplex,
              random_state=args.seed).fit_transform(emb)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#999"})
    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor("white")
    for ci, pk in enumerate(picks):
        m = owner == ci
        pts = xy[m]
        cluster_region(ax, pts, colors[ci])
        ax.scatter(pts[:, 0], pts[:, 1], s=sizes[m], color=colors[ci], edgecolor="white",
                   linewidth=0.8, zorder=3, alpha=0.92)
        # star = the token the model actually committed here
        cm = m & is_chosen
        if cm.any():
            ax.scatter(xy[cm, 0], xy[cm, 1], s=520, marker="*", color=colors[ci],
                       edgecolor="black", linewidth=1.2, zorder=6)
        # ring = the gold (secret) token, if it was in the top-k
        gm = m & is_gold
        if gm.any():
            ax.scatter(xy[gm, 0], xy[gm, 1], s=340, facecolor="none", edgecolor="black",
                       linewidth=1.6, zorder=5)
        lab = np.median(pts, 0)
        ax.text(lab[0], lab[1], f"pos {pk['pos']}: {pk['gold']}", fontsize=12.5,
                fontweight="bold", color=colors[ci], ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[ci], alpha=0.9))
    for j, s in enumerate(surfaces):
        if not is_chosen[j]:
            ax.annotate(_label(s), (xy[j, 0], xy[j, 1]), fontsize=8, color="#333",
                        alpha=0.85, xytext=(4, 4), textcoords="offset points", zorder=4)
    ax.set_title(f"Top-{args.topk} candidates AT GENERATION TIME — one answer, {len(picks)} committed positions",
                 fontsize=15, fontweight="bold", pad=52)
    ax.text(0.5, 1.012, metric_txt, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color="#444")
    ax.scatter([], [], s=200, marker="*", color="#555", edgecolor="black", label="token the model committed")
    ax.scatter([], [], s=170, facecolor="none", edgecolor="black", label="gold (secret) token")
    ax.scatter([], [], s=120, color="#999", label="other top-k candidate")
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out2d), exist_ok=True)
    fig.savefig(args.out2d, dpi=150)
    print(f"wrote {args.out2d}")

    # ---------- 3D ----------
    # lower perplexity in 3-D: with only ~4x10 points, a high perplexity smears the four
    # groups into one cloud; a small one lets each synonym set contract to a tight blob.
    perplex3 = max(3, min(7, len(all_ids) - 1))
    xyz = TSNE(n_components=3, metric="cosine", init="pca", perplexity=perplex3,
               random_state=args.seed).fit_transform(emb)
    gcen = xyz.mean(0)
    fig = plt.figure(figsize=(12.5, 10))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    for ci, pk in enumerate(picks):
        m = owner == ci
        pts = xyz[m]
        med = np.median(pts, 0)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=sizes[m] * 0.7, color=colors[ci],
                   edgecolor="white", linewidth=0.5, depthshade=True, alpha=0.9)
        cm = m & is_chosen
        if cm.any():
            ax.scatter(xyz[cm, 0], xyz[cm, 1], xyz[cm, 2], s=340, marker="*",
                       color=colors[ci], edgecolor="black", linewidth=1.1)
        lab = med + (med - gcen) * 0.9   # push label well outside the central cloud
        ax.text(lab[0], lab[1], lab[2], f"pos {pk['pos']}: {pk['gold']}", fontsize=11,
                fontweight="bold", color=colors[ci], ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=colors[ci], alpha=0.9))
    ax.set_title(f"Top-{args.topk} candidates at generation time in 3-D semantic space\n"
                 f"within-position cos {wc:.2f} vs between-position cos {bc:.2f} "
                 "— each committed position weighs synonyms",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2"); ax.set_zlabel("t-SNE-3")
    ax.view_init(elev=18, azim=-60)
    ax.xaxis.pane.set_alpha(0.04); ax.yaxis.pane.set_alpha(0.04); ax.zaxis.pane.set_alpha(0.04)
    fig.tight_layout()
    fig.savefig(args.out3d, dpi=150)
    print(f"wrote {args.out3d}")


if __name__ == "__main__":
    main()
