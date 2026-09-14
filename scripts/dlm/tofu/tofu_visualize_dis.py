"""Visualise the SimpleTSD forget TARGET across several demotion depths. At the selected
forget positions ``A_hat`` we mask them, run the frozen teacher once, and compare the RAW
teacher distribution ``p0(.|x, y_M)`` against the EDITED target ``q_k`` for each demotion
depth k in ``VIZ_BELIEF_KS`` (gold + teacher top-K removed, then renormalised) that the
student would be trained toward. This shows where the demoted gold mass is redistributed
as we demote more tokens -- in particular whether it keeps flowing into semantic
neighbours (synonyms / same-field words) no matter how deep we cut.

Reuses the real training pipeline (``get_model`` / ``get_data`` / ``get_collators``,
same Hydra config as ``src/train.py``) so tokenisation, chat template and answer span
match training exactly. No DeepSpeed trainer is imported or built -- the visualisation
uses an inference-only probe with the same preprocessing and PCSS selection operations.

Launch (teacher is LLaDA-8B bf16 ~16 GB):

    CUDA_VISIBLE_DEVICES=4 VIZ_BELIEF_KS=1,5,10,20 \
    python scripts/dlm/tofu/tofu_visualize_dis.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/tofu/dlm_simple_tsd \
        model=LLaDA-8B-Instruct \
        model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_LLaDA-8B-Instruct_retain100_150epochs \
        forget_split=forget10 \
        task_name=viz_simpletsd_target_forget10

``VIZ_BELIEF_KS`` = comma list of belief_top_k depths to compare (default 1,5,10,20).
Note top-1 ~ demote gold only (gold is already the teacher argmax), i.e. the "minimal"
target. Other knobs:
    VIZ_REVEAL       (default: forget)   # forget = mask only A_hat; all = mask whole answer
    VIZ_UNCOND       (default: 0)        # 1 = also mask the whole prompt -> p0 is the
                                         # UNCONDITIONAL (promptless) teacher; output goes to
                                         # target_ks_<split>_uncond.{jsonl,html}
    VIZ_TOPK         (default: 12)       # how many vocab tokens to show per distribution
    VIZ_LIMIT        (default: 8)        # number of examples to dump (0 = all)
    VIZ_BATCH_SIZE   (default: 4)
    VIZ_OUT_DIR      (default: logs/simpletsd_viz)
"""

import html
import importlib
import json
import os
import random
import sys
import types
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# Direct script launches need both source trees on sys.path. Some dllm revisions
# eagerly import the optional lm-eval benchmark stack from package __init__ files;
# this visualization only needs leaf model/scheduler modules.
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "dllm"))

try:
    importlib.import_module("dllm.core.schedulers")
except ModuleNotFoundError as exc:
    if exc.name != "lm_eval":
        raise
    for module_name in list(sys.modules):
        if module_name == "dllm" or module_name.startswith("dllm."):
            del sys.modules[module_name]
    for package_name, package_path in (
        ("dllm", _REPO / "dllm" / "dllm"),
        ("dllm.core", _REPO / "dllm" / "dllm" / "core"),
    ):
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        package.__package__ = package_name
        sys.modules[package_name] = package

from data import get_collators, get_data  # noqa: E402
from dllm.core.schedulers import make_alpha_scheduler  # noqa: E402
from model import get_model  # noqa: E402


def seed_everything(seed=42):
    """Seed this inference-only script without importing the Trainer package."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _SimpleTSDProbe:
    """Inference-only subset of ``DiffusionSimpleTSD`` used by this report.

    Importing the Trainer class pulls in DeepSpeed and performs CUDA build checks at
    module import time. The report never constructs a Trainer, so keep the exact
    preprocessing and PCSS selection operations here without importing that stack.
    """

    def _preprocess_inputs(self, inputs):
        if self.right_shift_logits:
            from dllm.utils.data import prepend_bos

            labels = inputs.get("labels", None)
            if labels is not None and torch.all(labels[:, 0] == -100):
                return inputs
            inputs = prepend_bos(
                inputs,
                bos_token_id=self.processing_class.bos_token_id,
                label_pad_token_id=-100,
            )
        return inputs

    def _postprocess_outputs(self, outputs):
        if self.right_shift_logits:
            logits = outputs.logits
            outputs.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return outputs

    def _log_probs(self, logits):
        return F.log_softmax(logits[..., : self.vocab_size].float(), dim=-1)

    @torch.no_grad()
    def _teacher_log_probs(self, y, attention_mask):
        out = self.ref_model(input_ids=y, attention_mask=attention_mask)
        return self._log_probs(self._postprocess_outputs(out).logits)

    def _prompt_mask(self, input_ids, answer_mask, attention_mask):
        prompt = ~answer_mask & (input_ids != self.pad_id)
        if attention_mask is not None:
            prompt = prompt & attention_mask.bool()
        return prompt

    @torch.no_grad()
    def _select_forget_positions(self, input_ids, answer_mask, attention_mask):
        b, l = answer_mask.shape
        device = input_ids.device
        prompt = self._prompt_mask(input_ids, answer_mask, attention_mask)
        gold = input_ids.unsqueeze(-1)

        s_sum = torch.zeros(b, l, device=device)
        cnt = torch.zeros(b, l, device=device)
        for _ in range(self.sel_mc):
            t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(b, device=device)
            p_mask = (1.0 - self.scheduler(t)).unsqueeze(1)
            m = (torch.rand((b, l), device=device) < p_mask) & answer_mask
            if not bool(m.any()):
                continue
            y = torch.where(m, self.mask_id, input_ids)
            log_cond = self._teacher_log_probs(y, attention_mask).gather(-1, gold).squeeze(-1)
            y_base = torch.where(prompt, self.mask_id, y)
            log_base = self._teacher_log_probs(y_base, attention_mask).gather(-1, gold).squeeze(-1)
            s_sum = s_sum + torch.where(m, log_cond - log_base, torch.zeros_like(s_sum))
            cnt = cnt + m.float()
        scores = s_sum / cnt.clamp_min(1.0)

        selected = self._threshold_scores(scores, answer_mask)
        if self.drop_question_overlap:
            selected = selected & ~self._question_overlap_mask(input_ids, answer_mask, prompt)
        return selected, scores

    @staticmethod
    def _otsu_threshold(vals, relax=0.0):
        xs, _ = torch.sort(vals)
        n = xs.numel()
        if n < 2 or float(xs[0]) == float(xs[-1]):
            return xs[0] if n else vals.new_tensor(0.0)
        total = xs.sum()
        csum = torch.cumsum(xs, 0)[:-1]
        cn = torch.arange(1, n, device=xs.device, dtype=xs.dtype)
        w0 = cn / n
        m0 = csum / cn
        m1 = (total - csum) / (n - cn)
        score = w0 * (1.0 - w0) * (m0 - m1) ** 2
        i = int(torch.argmax(score))
        threshold = 0.5 * (xs[i] + xs[i + 1])
        return threshold - relax * (threshold - xs[0]) if relax > 0.0 else threshold

    def _threshold_scores(self, scores, answer_mask):
        if self.select_method == "tau":
            return (scores >= self.tau) & answer_mask
        selected = torch.zeros_like(answer_mask)
        for bi in range(scores.shape[0]):
            idx = answer_mask[bi].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            vals = scores[bi, idx]
            if self.select_method == "topk":
                k = max(1, int(round(self.topk_frac * idx.numel())))
                threshold = torch.topk(vals, k).values.min()
            else:
                threshold = self._otsu_threshold(vals, self.otsu_relax)
            selected[bi, idx[vals >= threshold]] = True
        return selected

    def _question_overlap_mask(self, input_ids, answer_mask, prompt):
        import re

        special_re = re.compile(r"<\|?[^<>]*\|?>")
        overlap = torch.zeros_like(answer_mask)
        for bi in range(input_ids.shape[0]):
            q_ids = input_ids[bi][prompt[bi]].tolist()
            if not q_ids:
                continue
            q_text = self.processing_class.decode(q_ids, skip_special_tokens=True).lower()
            qwords = {w for w in re.findall(r"[a-z0-9]+", q_text) if len(w) >= 2}
            if not qwords:
                continue
            ans_idx = answer_mask[bi].nonzero(as_tuple=True)[0]
            if ans_idx.numel() == 0:
                continue
            surfaces = [self.processing_class.decode([int(t)]) for t in input_ids[bi][ans_idx]]
            groups, current = [], []
            for i, surface in enumerate(surfaces):
                if surface[:1].isspace() and current:
                    groups.append(current)
                    current = [i]
                else:
                    current.append(i)
            if current:
                groups.append(current)
            for group in groups:
                surface = "".join(surfaces[j] for j in group)
                surface = special_re.sub(" ", surface).lower()
                fragments = [
                    frag for frag in re.findall(r"[a-z0-9]+", surface) if len(frag) >= 2
                ]
                if fragments and all(fragment in qwords for fragment in fragments):
                    for j in group:
                        overlap[bi, ans_idx[j]] = True
        return overlap


def _env_or(cfg_val, env_key, cast):
    v = os.getenv(env_key)
    return cast(v) if v not in (None, "") else cfg_val


def _build_probe(cfg, tokenizer, model):
    """Build the inference-only probe state read by selection and demotion."""
    ma = cfg.trainer.method_args
    t = _SimpleTSDProbe()
    t.processing_class = tokenizer
    t.mask_id = tokenizer.mask_token_id
    t.pad_id = tokenizer.pad_token_id
    t.vocab_size = len(tokenizer)
    t.ref_model = model
    t.select_method = _env_or(ma.get("select_method", "otsu"), "PROBE_SELECT_METHOD", str)
    t.otsu_relax = _env_or(float(ma.get("otsu_relax", 0.0)), "PROBE_OTSU_RELAX", float)
    t.sel_mc = _env_or(int(ma.get("sel_mc", 8)), "PROBE_SEL_MC", int)
    t.tau = float(ma.get("tau", 0.0))
    t.topk_frac = float(ma.get("topk_frac", 0.5))
    t.drop_question_overlap = bool(ma.get("drop_question_overlap", True))
    t.demotion = "belief"
    t.belief_top_k = int(ma.get("belief_top_k", 5))
    t.right_shift_logits = bool(ma.get("right_shift_logits", False))
    t.time_epsilon = float(ma.get("time_epsilon", 1e-3))
    t.scheduler = make_alpha_scheduler(
        ma.get("scheduler_name", "LinearAlphaScheduler")
    )
    return t


def _tok(tokenizer, i):
    """Decode a single id to a display surface (visible spaces)."""
    s = tokenizer.decode([int(i)])
    return s if s.strip() else (s.replace(" ", "␣") or "∅")


def _topk_rows(log_dist, tokenizer, k, gold_id):
    """Sorted (token, prob, id, is_gold) rows for the top-k of one distribution, plus
    the gold row appended if it is not already in the top-k (so gold is always shown)."""
    probs = log_dist.exp()
    vals, idx = probs.topk(min(k, probs.numel()))
    rows = [
        {"tok": _tok(tokenizer, i), "prob": float(v), "id": int(i), "is_gold": int(i) == gold_id}
        for v, i in zip(vals.tolist(), idx.tolist())
    ]
    if gold_id not in idx.tolist():
        rows.append(
            {"tok": _tok(tokenizer, gold_id), "prob": float(probs[gold_id]),
             "id": gold_id, "is_gold": True, "below_topk": True}
        )
    return rows


def _remove_topk(log_p0, k):
    """Remaining distribution after deleting the k highest-probability tokens per
    position (pure prob-rank removal) and renormalising over the survivors."""
    if k < 1:
        raise ValueError(f"removed top-K must be positive, got {k}")
    finite_count = torch.isfinite(log_p0).sum(dim=-1)
    if bool((finite_count <= k).any()):
        raise ValueError(
            f"cannot remove top-{k} from a distribution with only "
            f"{int(finite_count.min().item())} finite entries"
        )
    idx = log_p0.topk(k, dim=-1).indices
    log_q = log_p0.scatter(-1, idx, float("-inf"))
    return log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)


@torch.no_grad()
def _collect_concept_basis(model, probe, input_ids, attn, answer_mask, a_hat,
                           erase_layer, min_dir_norm, subspace_mc, max_rank):
    """Per-example orthonormal concept basis (up to ``max_rank`` rows) built exactly as
    ``DiffusionCSD._build_concept_subspace``: over ``subspace_mc`` partial answer masks,
    collect ``g = h0 - h0_base`` at the masked forget positions and SVD-reduce them. Returns
    a list of ``[R_j, d]`` float tensors (one per example, possibly empty)."""
    b, l = answer_mask.shape
    device = input_ids.device
    prompt = probe._prompt_mask(input_ids, answer_mask, attn)
    vecs = [[] for _ in range(b)]
    for _ in range(subspace_mc):
        t = probe.time_epsilon + (1 - probe.time_epsilon) * torch.rand(b, device=device)
        p_mask = (1.0 - probe.scheduler(t)).unsqueeze(1)
        m = (torch.rand((b, l), device=device) < p_mask) & answer_mask
        fm = m & a_hat
        if not bool(fm.any()):
            continue
        y = torch.where(m, probe.mask_id, input_ids)
        h0 = model(input_ids=y, attention_mask=attn,
                   output_hidden_states=True).hidden_states[erase_layer]
        y_base = torch.where(prompt, probe.mask_id, y)
        h0b = model(input_ids=y_base, attention_mask=attn,
                    output_hidden_states=True).hidden_states[erase_layer]
        g = (h0 - h0b).float()
        for bi in range(b):
            idx = fm[bi].nonzero(as_tuple=True)[0]
            if idx.numel():
                vecs[bi].append(g[bi, idx])
    bases = []
    for bi in range(b):
        if not vecs[bi]:
            bases.append(input_ids.new_zeros(0, 0, dtype=torch.float32))
            continue
        G = torch.cat(vecs[bi], dim=0)
        G = G[G.norm(dim=-1) >= min_dir_norm]
        if G.numel() == 0:
            bases.append(input_ids.new_zeros(0, 0, dtype=torch.float32))
            continue
        _, _, Vh = torch.linalg.svd(G, full_matrices=False)
        bases.append(Vh[:max_rank])
    return bases


@torch.no_grad()
def run_subspace(cfg, tokenizer, model, probe, forget_ds, collator, device,
                 split, out_dir):
    """VIZ_MODE=subspace: pre-training verification of DiffusionCSD's erased target across
    a sweep of concept-subspace ranks.

    For each example we build the per-example concept subspace ``P`` from ``VIZ_SUBSPACE_MC``
    partial-mask teacher forwards (top-r right singular vectors of the collected
    prompt-dependence vectors ``g = h0 - h0_base``), exactly as
    ``DiffusionCSD._build_concept_subspace``. At the displayed (forget-masked) state we form
    the match-promptless target ``h_target = h0 - P^T P (h0 - h0_base)`` for each rank in
    ``VIZ_SUBSPACE_RANKS`` and RE-UNEMBED it on this single GPU (where the unembedding is
    cheap, vs ZeRO-3-sharded in training). Comparing gold prob / entropy vs the raw teacher
    across ranks is the go/no-go signal: does a higher-rank subspace collapse the fact tokens
    that rank-1 (== the promptless / QMask target) leaves standing? A one-off sanity check
    confirms ``W_U @ h0`` reproduces the teacher logits, validating the re-unembedding.
    """
    from torch.utils.data import DataLoader

    erase_layer = int(os.getenv("VIZ_ERASE_LAYER", "-1"))
    min_dir_norm = float(os.getenv("VIZ_MIN_DIR_NORM", "1e-3"))
    ranks = [int(x) for x in os.getenv("VIZ_SUBSPACE_RANKS", "1,4,8,16").split(",") if x.strip()]
    subspace_mc = int(os.getenv("VIZ_SUBSPACE_MC", "8"))
    max_rank = max(ranks)
    topk = int(os.getenv("VIZ_TOPK", "12"))
    limit = int(os.getenv("VIZ_LIMIT", "8"))
    batch_size = int(os.getenv("VIZ_BATCH_SIZE", "4"))
    emb = model.get_output_embeddings()  # Linear[V, d]; reproduces the teacher logits
    loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    jsonl_path = os.path.join(out_dir, f"target_subspace_{split}.jsonl")
    html_path = os.path.join(out_dir, f"target_subspace_{split}.html")
    labels = ["raw"] + [f"r={r}" for r in ranks]
    print(f"[viz:subspace] split={split} erase_layer={erase_layer} ranks={ranks} "
          f"subspace_mc={subspace_mc} min_dir_norm={min_dir_norm} topk={topk} "
          f"limit={limit} device={device}", flush=True)

    records, n_ex, n_pos = [], 0, 0
    agg = {lab: 0.0 for lab in labels}         # mean gold prob per column
    logit_check = None                         # max|W_U h0 - teacher logits|, sanity

    for batch in loader:
        forget = batch["forget"]
        input_ids = forget["input_ids"].to(device)
        labels_t = forget["labels"].to(device)
        attn = forget.get("attention_mask", None)
        attn = attn.to(device) if attn is not None else None

        prep = probe._preprocess_inputs(
            {"input_ids": input_ids, "labels": labels_t, "attention_mask": attn}
        )
        input_ids = prep["input_ids"]
        attn = prep.get("attention_mask", None)
        answer_mask = prep["labels"] != -100
        a_hat, _ = probe._select_forget_positions(input_ids, answer_mask, attn)

        # Concept subspace from many draws (the training recipe), per example.
        bases = _collect_concept_basis(
            model, probe, input_ids, attn, answer_mask, a_hat,
            erase_layer, min_dir_norm, subspace_mc, max_rank,
        )

        # Display state: mask only the selected forget positions.
        y_m = torch.where(a_hat, probe.mask_id, input_ids)
        out = model(input_ids=y_m, attention_mask=attn, output_hidden_states=True)
        log_p0 = probe._log_probs(probe._postprocess_outputs(out).logits)
        h0 = out.hidden_states[erase_layer]
        prompt = probe._prompt_mask(input_ids, answer_mask, attn)
        y_base = torch.where(prompt, probe.mask_id, y_m)
        out_b = model(input_ids=y_base, attention_mask=attn, output_hidden_states=True)
        h0_base = out_b.hidden_states[erase_layer]
        g = (h0 - h0_base).float()

        if logit_check is None:
            logit_check = float((emb(h0) - out.logits).abs().max())
            print(f"[viz:subspace] sanity max|W_U h0 - logits|={logit_check:.2e}", flush=True)

        b = input_ids.shape[0]
        for j in range(b):
            fpos = (a_hat[j] & answer_mask[j]).nonzero(as_tuple=True)[0]
            if fpos.numel() == 0:
                continue
            P = bases[j]  # [R, d] float32 (may be empty)
            q_text = tokenizer.decode(
                input_ids[j][prompt[j]].tolist(), skip_special_tokens=True
            )
            ans_idx = answer_mask[j].nonzero(as_tuple=True)[0]
            ans_text = tokenizer.decode(input_ids[j][ans_idx].tolist(), skip_special_tokens=True)

            positions = []
            for pi in fpos.tolist():
                gold_id = int(input_ids[j, pi])
                g_pi = g[j, pi]                       # [d]
                raw_rows = _topk_rows(log_p0[j, pi], tokenizer, topk, gold_id)
                cols = [{"label": "raw", "rows": raw_rows,
                         "gold_prob": round(float(log_p0[j, pi, gold_id].exp()), 6),
                         "top1": round(raw_rows[0]["prob"], 4),
                         "entropy": round(_entropy(log_p0[j, pi]), 4)}]
                agg["raw"] += float(log_p0[j, pi, gold_id].exp())
                for r in ranks:
                    lab = f"r={r}"
                    if P.numel() == 0:
                        h_t = h0[j, pi]                # no basis -> no erasure
                        used_r = 0
                    else:
                        Pr = P[:r]                     # [r', d], float32
                        proj = Pr.t() @ (Pr @ g_pi)    # P^T P (h0 - h0_base)  [d], float32
                        h_t = (h0[j, pi].float() - proj).to(h0.dtype)
                        used_r = Pr.shape[0]
                    log_q = probe._log_probs(emb(h_t.unsqueeze(0)))[0]
                    rows = _topk_rows(log_q, tokenizer, topk, gold_id)
                    cols.append({"label": lab, "rows": rows,
                                 "gold_prob": round(float(log_q[gold_id].exp()), 6),
                                 "top1": round(rows[0]["prob"], 4),
                                 "entropy": round(_entropy(log_q), 4),
                                 "used_rank": used_r})
                    agg[lab] += float(log_q[gold_id].exp())
                positions.append({"pos": pi, "gold": _tok(tokenizer, gold_id),
                                  "gold_id": gold_id, "dir_norm": round(float(g_pi.norm()), 4),
                                  "cols": cols})
                n_pos += 1

            records.append({"example": n_ex, "question": q_text, "answer": ans_text,
                            "n_forget_pos": len(positions), "positions": positions})
            n_ex += 1
            if n_ex <= 3:
                p0 = positions[0]
                print(f"--- ex {n_ex-1} | forget[0]='{p0['gold']}' "
                      f"||g||={p0['dir_norm']:.3f} | Q: {q_text[:60]}", flush=True)
                for c in p0["cols"]:
                    top = ", ".join(f"{r['tok']}={r['prob']:.3f}" for r in c["rows"][:5])
                    print(f"    {c['label']:>6}: gold={c['gold_prob']:.4f} top1={c['top1']:.3f}"
                          f" H={c['entropy']:.2f}nats | {top}", flush=True)
                print("", flush=True)
            if limit and n_ex >= limit:
                break
        if limit and n_ex >= limit:
            break

    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "split": split, "mode": "subspace", "erase_layer": erase_layer,
        "ranks": ranks, "subspace_mc": subspace_mc, "min_dir_norm": min_dir_norm,
        "n_examples": n_ex, "n_forget_positions": n_pos,
        "unembed_logit_max_abs_err": logit_check,
        "mean_gold_prob": {lab: round(v / max(n_pos, 1), 8) for lab, v in agg.items()},
        "mean_top1": {lab: round(sum(p["cols"][i]["top1"] for r in records for p in r["positions"])
                                 / max(n_pos, 1), 4)
                      for i, lab in enumerate(labels)},
        "mean_entropy": {lab: round(sum(p["cols"][i]["entropy"] for r in records for p in r["positions"])
                                    / max(n_pos, 1), 4)
                         for i, lab in enumerate(labels)},
    }
    _write_subspace_html(html_path, records, summary, labels)
    print("\n[viz:subspace] DONE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[viz:subspace] per-position JSONL : {jsonl_path}", flush=True)
    print(f"[viz:subspace] HTML               : {html_path}", flush=True)



def _knockout_bias(b, l, device, forget_mask, ctx_mask, neg=-1e9):
    """Additive attention bias ``[b, 1, l, l]`` (float) that SEVERS every edge between the
    forget positions and their fact-context, in BOTH directions: a forget position may no
    longer attend to any context token and no context token may attend to it (across all
    layers -- LLaDA takes this bias per block). Everything else stays at 0 (LLaDA is
    bidirectional, so the default bias is all-zero). The padding mask is added on top inside
    the model, and since forget/ctx are real (non-pad) tokens no entry becomes -inf."""
    bias = torch.zeros(b, 1, l, l, device=device, dtype=torch.float)
    for j in range(b):
        F = forget_mask[j].nonzero(as_tuple=True)[0]
        C = ctx_mask[j].nonzero(as_tuple=True)[0]
        if F.numel() and C.numel():
            bias[j, 0, F[:, None], C[None, :]] = neg
            bias[j, 0, C[:, None], F[None, :]] = neg
    return bias


@torch.no_grad()
def run_attn_knockout(cfg, tokenizer, model, probe, forget_ds, collator, device,
                      split, out_dir):
    """VIZ_MODE=attn_knockout: go/no-go probe for the attention-isolation idea.

    Mask the selected forget positions ``A_hat`` (reveal the rest of the answer -- the
    worst-case, answer-context-pinned setting the CSD viz used). On the frozen teacher we
    then re-decode those positions under two attention knockouts, injected as a per-example
    additive ``attention_bias`` that LLaDA applies at every layer:

      * ``kout_q``  : sever forget <-> question edges only (the attention-level analogue of
                      QMask -- if the fact still survives here it is pinned by the answer's
                      own revealed tokens, exactly the leak the CSD viz measured).
      * ``kout_qa`` : sever forget <-> (question + other answer) edges -- full isolation of
                      the forget position from all fact-bearing context.

    Reading: if ``kout_q`` leaves the fact token high but ``kout_qa`` collapses it, the fact
    flows through attention and a training method that isolates BOTH directions could beat
    QMask. If even ``kout_qa`` leaves the fact standing, the fact is written into the
    position's own residual/MLP independent of context attention -> no attention method can
    erase it (the CATU failure mode), and this whole family is a dead end. Cheap: teacher
    only, a few examples.
    """
    from torch.utils.data import DataLoader

    topk = int(os.getenv("VIZ_TOPK", "12"))
    limit = int(os.getenv("VIZ_LIMIT", "4"))
    batch_size = int(os.getenv("VIZ_BATCH_SIZE", "4"))
    loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    labels = ["raw", "kout_q", "kout_qa"]
    print(f"[viz:attn_knockout] split={split} topk={topk} limit={limit} device={device}",
          flush=True)

    records, n_ex, n_pos = [], 0, 0
    agg = {lab: 0.0 for lab in labels}

    for batch in loader:
        forget = batch["forget"]
        input_ids = forget["input_ids"].to(device)
        labels_t = forget["labels"].to(device)
        attn = forget.get("attention_mask", None)
        attn = attn.to(device) if attn is not None else None

        prep = probe._preprocess_inputs(
            {"input_ids": input_ids, "labels": labels_t, "attention_mask": attn}
        )
        input_ids = prep["input_ids"]
        attn = prep.get("attention_mask", None)
        answer_mask = prep["labels"] != -100
        a_hat, _ = probe._select_forget_positions(input_ids, answer_mask, attn)
        prompt = probe._prompt_mask(input_ids, answer_mask, attn)

        b, l = input_ids.shape
        y_m = torch.where(a_hat, probe.mask_id, input_ids)   # reveal all but forget positions

        # raw teacher (full bidirectional attention)
        log_raw = probe._log_probs(
            probe._postprocess_outputs(
                model(input_ids=y_m, attention_mask=attn, output_hidden_states=True)
            ).logits
        )
        # knockout forwards: forget <-> question, and forget <-> (question + other answer)
        other_ans = answer_mask & ~a_hat
        bias_q = _knockout_bias(b, l, device, a_hat, prompt)
        bias_qa = _knockout_bias(b, l, device, a_hat, prompt | other_ans)
        log_kq = probe._log_probs(probe._postprocess_outputs(
            model(input_ids=y_m, attention_mask=attn, attention_bias=bias_q,
                  output_hidden_states=True)).logits)
        log_kqa = probe._log_probs(probe._postprocess_outputs(
            model(input_ids=y_m, attention_mask=attn, attention_bias=bias_qa,
                  output_hidden_states=True)).logits)
        dists = {"raw": log_raw, "kout_q": log_kq, "kout_qa": log_kqa}

        for j in range(b):
            fpos = (a_hat[j] & answer_mask[j]).nonzero(as_tuple=True)[0]
            if fpos.numel() == 0:
                continue
            q_text = tokenizer.decode(input_ids[j][prompt[j]].tolist(), skip_special_tokens=True)
            ans_idx = answer_mask[j].nonzero(as_tuple=True)[0]
            ans_text = tokenizer.decode(input_ids[j][ans_idx].tolist(), skip_special_tokens=True)

            positions = []
            for pi in fpos.tolist():
                gold_id = int(input_ids[j, pi])
                cols = []
                for lab in labels:
                    lp = dists[lab][j, pi]
                    rows = _topk_rows(lp, tokenizer, topk, gold_id)
                    cols.append({"label": lab, "rows": rows,
                                 "gold_prob": round(float(lp[gold_id].exp()), 6),
                                 "top1": round(rows[0]["prob"], 4),
                                 "entropy": round(_entropy(lp), 4)})
                    agg[lab] += float(lp[gold_id].exp())
                positions.append({"pos": pi, "gold": _tok(tokenizer, gold_id),
                                  "gold_id": gold_id, "cols": cols})
                n_pos += 1

            records.append({"example": n_ex, "question": q_text, "answer": ans_text,
                            "n_forget_pos": len(positions), "positions": positions})
            n_ex += 1
            if n_ex <= 4:
                p0 = positions[0]
                print(f"--- ex {n_ex-1} | forget[0]='{p0['gold']}' | Q: {q_text[:70]}",
                      flush=True)
                for c in p0["cols"]:
                    top = ", ".join(f"{r['tok']}={r['prob']:.3f}" for r in c["rows"][:5])
                    print(f"    {c['label']:>8}: gold={c['gold_prob']:.4f} top1={c['top1']:.3f}"
                          f" H={c['entropy']:.2f}nats | {top}", flush=True)
                print("", flush=True)
            if limit and n_ex >= limit:
                break
        if limit and n_ex >= limit:
            break

    jsonl_path = os.path.join(out_dir, f"attn_knockout_{split}.jsonl")
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "split": split, "mode": "attn_knockout", "n_examples": n_ex,
        "n_forget_positions": n_pos,
        "mean_gold_prob": {lab: round(v / max(n_pos, 1), 6) for lab, v in agg.items()},
    }
    print("\n[viz:attn_knockout] DONE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[viz:attn_knockout] per-position JSONL : {jsonl_path}", flush=True)


def _entropy(log_dist):
    """Shannon entropy (nats) of a log-distribution, ignoring -inf support."""
    neg = torch.isneginf(log_dist)
    p = log_dist.exp()
    plogp = torch.where(neg, torch.zeros_like(log_dist), p * log_dist)
    return float(-plogp.sum())


@torch.no_grad()
def run_compare(cfg, tokenizer, model, probe, forget_ds, collator, device, split, out_dir):
    """VIZ_COMPARE=1: the remove-top-k analysis run for BOTH the conditional (query
    visible) and the unconditional (query masked) teacher, side by side in one HTML.

    Identical masking/removal to the default ``ks`` path -- mask the selected forget
    positions ``A_hat`` (reveal=forget) or the whole answer (reveal=all), read the
    teacher, then for each k in ``VIZ_BELIEF_KS`` delete the k highest-prob tokens and
    renormalise (``_remove_topk``) and show the top-``VIZ_TOPK`` survivors. The only
    difference between the two conditions is whether the prompt (query) is also masked,
    so per position you can read off directly how much of the surviving mass was
    query-driven vs recoverable from the answer's own co-text."""
    ks = [int(x) for x in os.getenv("VIZ_BELIEF_KS", "1,5,10,20").split(",") if x.strip()]
    reveal = os.getenv("VIZ_REVEAL", "forget")   # forget | all
    topk = int(os.getenv("VIZ_TOPK", "10"))
    limit = int(os.getenv("VIZ_LIMIT", "8"))
    batch_size = int(os.getenv("VIZ_BATCH_SIZE", "4"))
    loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    jsonl_path = os.path.join(out_dir, f"target_ks_{split}_compare.jsonl")
    html_path = os.path.join(out_dir, f"target_ks_{split}_compare.html")
    conds = ["cond", "uncond"]
    labels = ["raw"] + [f"k={k}" for k in ks]
    print(f"[viz:compare] split={split} belief_ks={ks} reveal={reveal} topk={topk} "
          f"limit={limit} device={device}", flush=True)

    records = []
    agg_gold = {c: {"raw": 0.0, **{k: 0.0 for k in ks}} for c in conds}
    n_pos, n_ex = 0, 0

    for batch in loader:
        forget = batch["forget"]
        input_ids = forget["input_ids"].to(device)
        labels_t = forget["labels"].to(device)
        attn = forget.get("attention_mask", None)
        attn = attn.to(device) if attn is not None else None

        prep = probe._preprocess_inputs(
            {"input_ids": input_ids, "labels": labels_t, "attention_mask": attn}
        )
        input_ids = prep["input_ids"]
        attn = prep.get("attention_mask", None)
        answer_mask = prep["labels"] != -100
        a_hat, _ = probe._select_forget_positions(input_ids, answer_mask, attn)

        to_mask = answer_mask if reveal == "all" else a_hat
        y_base = torch.where(to_mask, probe.mask_id, input_ids)         # query visible
        prompt_all = probe._prompt_mask(input_ids, answer_mask, attn)
        y_unc = torch.where(prompt_all, probe.mask_id, y_base)          # query masked too
        log_p0 = {
            "cond": probe._teacher_log_probs(y_base, attn),
            "uncond": probe._teacher_log_probs(y_unc, attn),
        }
        log_qs = {c: {k: _remove_topk(log_p0[c], k) for k in ks} for c in conds}

        b = input_ids.shape[0]
        for j in range(b):
            fpos = (a_hat[j] & answer_mask[j]).nonzero(as_tuple=True)[0]
            if fpos.numel() == 0:
                continue
            prompt = probe._prompt_mask(
                input_ids[j : j + 1], answer_mask[j].unsqueeze(0),
                None if attn is None else attn[j : j + 1],
            )[0]
            q_text = tokenizer.decode(input_ids[j][prompt].tolist(), skip_special_tokens=True)
            ans_idx = answer_mask[j].nonzero(as_tuple=True)[0]
            ans_text = tokenizer.decode(input_ids[j][ans_idx].tolist(), skip_special_tokens=True)

            positions = []
            for pi in fpos.tolist():
                gold_id = int(input_ids[j, pi])
                groups = {}
                for c in conds:
                    lp = log_p0[c][j, pi]
                    raw_rows = _topk_rows(lp, tokenizer, topk, gold_id)
                    cols = [{"label": "raw", "rows": raw_rows,
                             "gold_prob": round(float(lp[gold_id].exp()), 6),
                             "top1": round(raw_rows[0]["prob"], 4),
                             "entropy": round(_entropy(lp), 4)}]
                    agg_gold[c]["raw"] += float(lp[gold_id].exp())
                    for k in ks:
                        lq = log_qs[c][k][j, pi]
                        rows = _topk_rows(lq, tokenizer, topk, gold_id)
                        cols.append({"label": f"k={k}", "rows": rows,
                                     "gold_prob": round(float(lq.exp()[gold_id]), 8),
                                     "top1": round(rows[0]["prob"], 4),
                                     "entropy": round(_entropy(lq), 4),
                                     "sum": round(float(lq.exp().sum()), 6)})
                        agg_gold[c][k] += float(lq.exp()[gold_id])
                    groups[c] = cols
                positions.append({"pos": pi, "gold": _tok(tokenizer, gold_id),
                                  "gold_id": gold_id, "groups": groups})
                n_pos += 1

            records.append({"example": n_ex, "question": q_text, "answer": ans_text,
                            "n_forget_pos": len(positions), "positions": positions})
            n_ex += 1
            if n_ex <= 3:
                p0 = positions[0]
                print(f"--- ex {n_ex-1} | forget[0]='{p0['gold']}' | Q: {q_text[:60]}", flush=True)
                for c in conds:
                    print(f"  [{c}]", flush=True)
                    for col in p0["groups"][c]:
                        top = ", ".join(f"{r['tok']}={r['prob']:.3f}" for r in col["rows"][:5])
                        print(f"    {col['label']:>5}: top1={col['top1']:.3f} "
                              f"H={col['entropy']:.2f}nats | {top}", flush=True)
                print("", flush=True)
            if limit and n_ex >= limit:
                break
        if limit and n_ex >= limit:
            break

    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _col_mean(c, i, field):  # mean over positions of cols[i][field] within condition c
        vals = [p["groups"][c][i][field] for r in records for p in r["positions"]]
        return round(sum(vals) / max(len(vals), 1), 4)

    summary = {
        "split": split, "belief_ks": ks, "reveal": reveal, "mode": "compare",
        "n_examples": n_ex, "n_forget_positions": n_pos,
        "mean_gold_prob": {c: {str(kk): round(v / max(n_pos, 1), 8)
                               for kk, v in agg_gold[c].items()} for c in conds},
        "mean_top1": {c: {lab: _col_mean(c, i, "top1") for i, lab in enumerate(labels)}
                      for c in conds},
        "mean_entropy": {c: {lab: _col_mean(c, i, "entropy") for i, lab in enumerate(labels)}
                         for c in conds},
    }
    _write_compare_html(html_path, records, summary, ks)
    print("\n[viz:compare] DONE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[viz:compare] per-position JSONL : {jsonl_path}", flush=True)
    print(f"[viz:compare] HTML               : {html_path}", flush=True)


@torch.no_grad()
@hydra.main(version_base=None, config_path="../../../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = get_model(cfg.model)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    template_args = cfg.model.template_args
    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer, template_args=template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    forget_ds = data["train"]

    probe = _build_probe(cfg, tokenizer, model)
    out_dir = os.path.abspath(os.getenv("VIZ_OUT_DIR", "logs/simpletsd_viz"))
    os.makedirs(out_dir, exist_ok=True)
    split = str(cfg.get("forget_split", "forget"))

    if os.getenv("VIZ_MODE", "ks") == "subspace":
        run_subspace(cfg, tokenizer, model, probe, forget_ds, collator, device,
                     split, out_dir)
        return

    if os.getenv("VIZ_MODE", "ks") == "attn_knockout":
        run_attn_knockout(cfg, tokenizer, model, probe, forget_ds, collator, device,
                          split, out_dir)
        return

    if os.getenv("VIZ_COMPARE", "0") == "1":
        run_compare(cfg, tokenizer, model, probe, forget_ds, collator, device,
                    split, out_dir)
        return

    ks = [int(x) for x in os.getenv("VIZ_BELIEF_KS", "1,5,10,20").split(",") if x.strip()]
    reveal = os.getenv("VIZ_REVEAL", "forget")   # forget | all
    uncond = os.getenv("VIZ_UNCOND", "0") == "1"  # mask the prompt -> unconditional p0
    topk = int(os.getenv("VIZ_TOPK", "12"))
    limit = int(os.getenv("VIZ_LIMIT", "8"))
    batch_size = int(os.getenv("VIZ_BATCH_SIZE", "4"))
    loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    suffix = "_uncond" if uncond else ""
    jsonl_path = os.path.join(out_dir, f"target_ks_{split}{suffix}.jsonl")
    html_path = os.path.join(out_dir, f"target_ks_{split}{suffix}.html")

    print(f"[viz] split={split} belief_ks={ks} reveal={reveal} uncond={uncond} "
          f"topk={topk} limit={limit} device={device}", flush=True)

    records = []
    # aggregate gold prob under each k (index 0 = raw teacher)
    agg_gold = {"raw": 0.0, **{k: 0.0 for k in ks}}
    n_pos = 0
    n_ex = 0

    for batch in loader:
        forget = batch["forget"]
        input_ids = forget["input_ids"].to(device)
        labels = forget["labels"].to(device)
        attn = forget.get("attention_mask", None)
        attn = attn.to(device) if attn is not None else None

        prep = probe._preprocess_inputs(
            {"input_ids": input_ids, "labels": labels, "attention_mask": attn}
        )
        input_ids = prep["input_ids"]
        attn = prep.get("attention_mask", None)
        answer_mask = prep["labels"] != -100
        a_hat, scores = probe._select_forget_positions(input_ids, answer_mask, attn)

        to_mask = answer_mask if reveal == "all" else a_hat
        y_m = torch.where(to_mask, probe.mask_id, input_ids)
        if uncond:
            # Unconditional: blank the whole prompt (query) so p0 is promptless.
            prompt_all = probe._prompt_mask(input_ids, answer_mask, attn)
            y_m = torch.where(prompt_all, probe.mask_id, y_m)
        log_p0 = probe._teacher_log_probs(y_m, attn)  # log p0(.|x, y_M)  [b, l, V]
        log_qs = {k: _remove_topk(log_p0, k) for k in ks}

        b = input_ids.shape[0]
        for j in range(b):
            fpos = (a_hat[j] & answer_mask[j]).nonzero(as_tuple=True)[0]
            if fpos.numel() == 0:
                continue
            prompt = probe._prompt_mask(
                input_ids[j : j + 1], answer_mask[j].unsqueeze(0),
                None if attn is None else attn[j : j + 1],
            )[0]
            q_text = tokenizer.decode(input_ids[j][prompt].tolist(), skip_special_tokens=True)
            ans_idx = answer_mask[j].nonzero(as_tuple=True)[0]
            ans_text = tokenizer.decode(input_ids[j][ans_idx].tolist(), skip_special_tokens=True)

            positions = []
            for pi in fpos.tolist():
                gold_id = int(input_ids[j, pi])
                raw_rows = _topk_rows(log_p0[j, pi], tokenizer, topk, gold_id)
                cols = [{"label": "raw", "rows": raw_rows,
                         "gold_prob": round(float(log_p0[j, pi, gold_id].exp()), 6),
                         "top1": round(raw_rows[0]["prob"], 4),
                         "entropy": round(_entropy(log_p0[j, pi]), 4)}]
                agg_gold["raw"] += float(log_p0[j, pi, gold_id].exp())
                for k in ks:
                    lq = log_qs[k][j, pi]
                    rows = _topk_rows(lq, tokenizer, topk, gold_id)
                    cols.append({
                        "label": f"k={k}", "rows": rows,
                        "gold_prob": round(float(lq.exp()[gold_id]), 8),
                        "top1": round(rows[0]["prob"], 4),
                        "entropy": round(_entropy(lq), 4),
                        "sum": round(float(lq.exp().sum()), 6),
                    })
                    agg_gold[k] += float(lq.exp()[gold_id])
                positions.append({"pos": pi, "gold": _tok(tokenizer, gold_id),
                                  "gold_id": gold_id, "cols": cols})
                n_pos += 1

            records.append({"example": n_ex, "question": q_text, "answer": ans_text,
                            "n_forget_pos": len(positions), "positions": positions})
            n_ex += 1
            if n_ex <= 3:
                p0 = positions[0]
                print(f"--- ex {n_ex-1} | forget[0]='{p0['gold']}' | Q: {q_text[:60]}", flush=True)
                for c in p0["cols"]:
                    top = ", ".join(f"{r['tok']}={r['prob']:.3f}" for r in c["rows"][:5])
                    print(f"    {c['label']:>5}: top1={c['top1']:.3f} H={c['entropy']:.2f}nats"
                          f" | {top}", flush=True)
                print("", flush=True)
            if limit and n_ex >= limit:
                break
        if limit and n_ex >= limit:
            break

    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "split": split, "belief_ks": ks, "reveal": reveal, "uncond": uncond,
        "n_examples": n_ex, "n_forget_positions": n_pos,
        "mean_gold_prob": {str(kk): round(v / max(n_pos, 1), 8) for kk, v in agg_gold.items()},
        "mean_top1": {lab: round(sum(p["cols"][i]["top1"] for r in records for p in r["positions"])
                                 / max(n_pos, 1), 4)
                      for i, lab in enumerate(["raw"] + [f"k={k}" for k in ks])},
        "mean_entropy": {lab: round(sum(p["cols"][i]["entropy"] for r in records for p in r["positions"])
                                    / max(n_pos, 1), 4)
                         for i, lab in enumerate(["raw"] + [f"k={k}" for k in ks])},
    }
    _write_html(html_path, records, summary, ks)
    print("\n[viz] DONE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[viz] per-position JSONL : {jsonl_path}", flush=True)
    print(f"[viz] HTML               : {html_path}", flush=True)


def _bars(rows):
    """Horizontal prob bars for one distribution's top-k rows; gold in red."""
    mx = max((r["prob"] for r in rows), default=1e-9) or 1e-9
    out = []
    for r in rows:
        w = max(1.0, 100.0 * r["prob"] / mx)
        cls = "gold" if r["is_gold"] else "tok"
        note = " ·<topk" if r.get("below_topk") else ""
        out.append(
            f'<div class="row {cls}"><span class="lbl">{html.escape(r["tok"])}{note}</span>'
            f'<span class="bar" style="width:{w:.1f}%"></span>'
            f'<span class="p">{r["prob"]:.4f}</span></div>'
        )
    return "".join(out)


def _write_html(path, records, summary, ks):
    ncol = 1 + len(ks)
    css = f"""
    body{{font:13px/1.5 system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}}
    header{{position:sticky;top:0;background:#161a22;padding:12px 20px;border-bottom:1px solid #2a2f3a;z-index:2}}
    .wrap{{padding:16px 20px;max-width:1500px;margin:auto}}
    .ex{{border:1px solid #2a2f3a;border-radius:10px;margin:14px 0;padding:12px 14px;background:#131722}}
    .q{{color:#9ecbff}}.a{{color:#b7f5c0;margin:4px 0 10px}}
    .pos{{border-top:1px dashed #2a2f3a;padding:10px 0}}
    .cols{{display:grid;grid-template-columns:repeat({ncol},1fr);gap:14px}}
    .col h4{{margin:0 0 6px;font-weight:600;color:#c9c9c9;font-size:12px}}
    .col .gp{{color:#9aa4b2;font-size:11px;margin:-4px 0 6px}}
    .row{{display:grid;grid-template-columns:110px 1fr 52px;align-items:center;gap:6px;margin:2px 0}}
    .lbl{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:ui-monospace,monospace}}
    .bar{{height:11px;background:#3b82f6;border-radius:3px}}
    .tok .bar{{background:#3b82f6}}.gold .bar{{background:#ef4444}}
    .gold .lbl{{color:#ff8a8a;font-weight:700}}
    .p{{text-align:right;color:#9aa4b2;font-variant-numeric:tabular-nums;font-size:11px}}
    .meta{{color:#9aa4b2;font-size:12px;margin-top:4px}}
    code{{background:#1e2430;padding:1px 5px;border-radius:4px}}
    """
    mg = summary["mean_gold_prob"]
    gold_line = " ".join(
        f"{lab}=<code>{mg.get(lab, mg.get(str(lab), 0)):.2e}</code>"
        for lab in (["raw"] + [str(k) for k in ks])
    )
    head = (
        f"<b>SimpleTSD forget target vs demotion depth</b> &mdash; "
        f"belief_ks=<code>{ks}</code> reveal=<code>{summary['reveal']}</code> "
        f"cond=<code>{'UNCONDITIONAL (query masked)' if summary.get('uncond') else 'conditional'}</code> "
        f"over {summary['n_forget_positions']} positions &mdash; mean gold prob: {gold_line}"
    )
    body = []
    for r in records:
        pos_html = []
        for p in r["positions"]:
            cols_html = []
            for c in p["cols"]:
                title = "Raw teacher p0" if c["label"] == "raw" else f"Removed top-{c['label'][2:]}"
                gp = (f'top1={c["top1"]:.3f} · H={c["entropy"]:.2f} nats'
                      + (f' · &Sigma;={c["sum"]:.3f}' if "sum" in c else ""))
                cols_html.append(
                    f'<div class="col"><h4>{title}</h4><div class="gp">{gp}</div>{_bars(c["rows"])}</div>'
                )
            pos_html.append(
                f'<div class="pos"><div class="meta">forget token <b>{html.escape(p["gold"])}</b> '
                f'@pos {p["pos"]}</div><div class="cols">{"".join(cols_html)}</div></div>'
            )
        body.append(
            f'<div class="ex"><div class="q">Q: {html.escape(r["question"])}</div>'
            f'<div class="a">A: {html.escape(r["answer"])}</div>{"".join(pos_html)}</div>'
        )
    doc = (
        f"<!doctype html><meta charset='utf-8'><style>{css}</style>"
        f"<header>{head}</header><div class='wrap'>{''.join(body)}</div>"
    )
    with open(path, "w") as f:
        f.write(doc)


def _cols_grid(cols, ncol):
    """One row of ``ncol`` distribution columns (raw + each k), reusing ``_bars``."""
    cells = []
    for c in cols:
        title = "Raw (k=0)" if c["label"] == "raw" else f"Removed top-{c['label'][2:]}"
        gp = (f'gold={c["gold_prob"]:.4f} · top1={c["top1"]:.3f} · H={c["entropy"]:.2f} nats'
              + (f' · &Sigma;={c["sum"]:.3f}' if "sum" in c else ""))
        cells.append(
            f'<div class="col"><h4>{title}</h4><div class="gp">{gp}</div>{_bars(c["rows"])}</div>'
        )
    return f'<div class="cols" style="grid-template-columns:repeat({ncol},1fr)">{"".join(cells)}</div>'


def _write_compare_html(path, records, summary, ks):
    """Conditional vs unconditional remove-top-k, one HTML. Per forget position: two
    labelled rows (query visible / query masked), each a grid of raw + removed-top-k
    columns in the SAME order, so each k lines up vertically across the two conditions."""
    ncol = 1 + len(ks)
    css = f"""
    body{{font:13px/1.5 system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}}
    header{{position:sticky;top:0;background:#161a22;padding:12px 20px;border-bottom:1px solid #2a2f3a;z-index:2}}
    .wrap{{padding:16px 20px;max-width:1600px;margin:auto}}
    .ex{{border:1px solid #2a2f3a;border-radius:10px;margin:14px 0;padding:12px 14px;background:#131722}}
    .q{{color:#9ecbff}}.a{{color:#b7f5c0;margin:4px 0 10px}}
    .pos{{border-top:1px dashed #2a2f3a;padding:10px 0}}
    .condrow{{margin:8px 0}}
    .condlbl{{font-size:12px;font-weight:700;margin:0 0 4px;padding:2px 8px;border-radius:5px;display:inline-block}}
    .condlbl.cond{{background:#173a2a;color:#8ff0b8}}
    .condlbl.uncond{{background:#3a1730;color:#f0a8d8}}
    .cols{{display:grid;gap:14px}}
    .col h4{{margin:0 0 6px;font-weight:600;color:#c9c9c9;font-size:12px}}
    .col .gp{{color:#9aa4b2;font-size:11px;margin:-4px 0 6px}}
    .row{{display:grid;grid-template-columns:110px 1fr 52px;align-items:center;gap:6px;margin:2px 0}}
    .lbl{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:ui-monospace,monospace}}
    .bar{{height:11px;background:#3b82f6;border-radius:3px}}
    .tok .bar{{background:#3b82f6}}.gold .bar{{background:#ef4444}}
    .gold .lbl{{color:#ff8a8a;font-weight:700}}
    .p{{text-align:right;color:#9aa4b2;font-variant-numeric:tabular-nums;font-size:11px}}
    .meta{{color:#9aa4b2;font-size:12px;margin-top:4px}}
    code{{background:#1e2430;padding:1px 5px;border-radius:4px}}
    """
    mg = summary["mean_gold_prob"]
    labels_k = ["raw"] + [str(k) for k in ks]

    def _mgline(c):
        return " ".join(f"{lab}=<code>{mg[c].get(lab, 0):.2e}</code>" for lab in labels_k)

    head = (
        f"<b>Remove-top-k target: conditional vs unconditional</b> &mdash; "
        f"belief_ks=<code>{ks}</code> reveal=<code>{summary['reveal']}</code> over "
        f"{summary['n_forget_positions']} positions<br>"
        f"<span style='color:#8ff0b8'>cond (query visible)</span> mean gold: {_mgline('cond')}<br>"
        f"<span style='color:#f0a8d8'>uncond (query masked)</span> mean gold: {_mgline('uncond')}"
    )
    cond_titles = {"cond": "Conditional — query visible",
                   "uncond": "Unconditional — query masked"}
    body = []
    for r in records:
        pos_html = []
        for p in r["positions"]:
            rows_html = []
            for c in ("cond", "uncond"):
                rows_html.append(
                    f'<div class="condrow"><div class="condlbl {c}">{cond_titles[c]}</div>'
                    f'{_cols_grid(p["groups"][c], ncol)}</div>'
                )
            pos_html.append(
                f'<div class="pos"><div class="meta">forget token '
                f'<b>{html.escape(p["gold"])}</b> @pos {p["pos"]}</div>{"".join(rows_html)}</div>'
            )
        body.append(
            f'<div class="ex"><div class="q">Q: {html.escape(r["question"])}</div>'
            f'<div class="a">A: {html.escape(r["answer"])}</div>{"".join(pos_html)}</div>'
        )
    doc = (
        f"<!doctype html><meta charset='utf-8'><style>{css}</style>"
        f"<header>{head}</header><div class='wrap'>{''.join(body)}</div>"
    )
    with open(path, "w") as f:
        f.write(doc)


def _write_subspace_html(path, records, summary, labels):
    """HTML with one column per rank (raw teacher + each ``r`` in the sweep) for
    VIZ_MODE=subspace, so fact-token collapse can be read off across ranks side by side."""
    ncol = len(labels)
    css = f"""
    body{{font:13px/1.5 system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}}
    header{{position:sticky;top:0;background:#161a22;padding:12px 20px;border-bottom:1px solid #2a2f3a;z-index:2}}
    .wrap{{padding:16px 20px;max-width:1600px;margin:auto}}
    .ex{{border:1px solid #2a2f3a;border-radius:10px;margin:14px 0;padding:12px 14px;background:#131722}}
    .q{{color:#9ecbff}}.a{{color:#b7f5c0;margin:4px 0 10px}}
    .pos{{border-top:1px dashed #2a2f3a;padding:10px 0}}
    .cols{{display:grid;grid-template-columns:repeat({ncol},1fr);gap:14px}}
    .col h4{{margin:0 0 6px;font-weight:600;color:#c9c9c9;font-size:12px}}
    .col .gp{{color:#9aa4b2;font-size:11px;margin:-4px 0 6px}}
    .row{{display:grid;grid-template-columns:110px 1fr 52px;align-items:center;gap:6px;margin:2px 0}}
    .lbl{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:ui-monospace,monospace}}
    .bar{{height:11px;background:#3b82f6;border-radius:3px}}
    .tok .bar{{background:#3b82f6}}.gold .bar{{background:#ef4444}}
    .gold .lbl{{color:#ff8a8a;font-weight:700}}
    .p{{text-align:right;color:#9aa4b2;font-variant-numeric:tabular-nums;font-size:11px}}
    .meta{{color:#9aa4b2;font-size:12px;margin-top:4px}}
    code{{background:#1e2430;padding:1px 5px;border-radius:4px}}
    """
    mg = summary["mean_gold_prob"]
    mg_str = " ".join(f"{lab}=<code>{mg[lab]:.2e}</code>" for lab in labels)
    head = (
        f"<b>CSD subspace-projected forget target (rank sweep)</b> &mdash; erase_layer="
        f"<code>{summary['erase_layer']}</code> · ranks=<code>{summary['ranks']}</code> · "
        f"subspace_mc=<code>{summary['subspace_mc']}</code> over "
        f"{summary['n_forget_positions']} positions &mdash; mean gold prob {mg_str} "
        f"&mdash; unembed sanity max|W_U h0 - logits|="
        f"<code>{summary['unembed_logit_max_abs_err']:.2e}</code>"
    )
    body = []
    for r in records:
        pos_html = []
        for p in r["positions"]:
            cols_html = []
            for c in p["cols"]:
                title = ("Raw teacher p0" if c["label"] == "raw"
                         else f"Subspace target {c['label']}")
                gp = (f'gold={c["gold_prob"]:.4f} · top1={c["top1"]:.3f} · '
                      f'H={c["entropy"]:.2f} nats'
                      + (f' · used_r={c["used_rank"]}' if "used_rank" in c else ""))
                cols_html.append(
                    f'<div class="col"><h4>{html.escape(title)}</h4>'
                    f'<div class="gp">{gp}</div>{_bars(c["rows"])}</div>'
                )
            pos_html.append(
                f'<div class="pos"><div class="meta">forget token '
                f'<b>{html.escape(p["gold"])}</b> @pos {p["pos"]} · '
                f'||g||={p["dir_norm"]:.3f}</div>'
                f'<div class="cols">{"".join(cols_html)}</div></div>'
            )
        body.append(
            f'<div class="ex"><div class="q">Q: {html.escape(r["question"])}</div>'
            f'<div class="a">A: {html.escape(r["answer"])}</div>{"".join(pos_html)}</div>'
        )
    doc = (
        f"<!doctype html><meta charset='utf-8'><style>{css}</style>"
        f"<header>{head}</header><div class='wrap'>{''.join(body)}</div>"
    )
    with open(path, "w") as f:
        f.write(doc)


if __name__ == "__main__":
    main()
