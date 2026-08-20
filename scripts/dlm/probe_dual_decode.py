"""Token-level confidence trace of two TOFU-finetuned DLM checkpoints on one canvas.

Compares a checkpoint that memorised a forget split (``retain100``) against the
oracle that never saw it (``retain95`` for ``forget05``). The interesting
quantity is not each model's generation in isolation -- those diverge after the
first disagreement and stop being comparable token-for-token -- but what the two
models believe *at the same denoising state*.

So the decode is driven by one model (``driver``) exactly as ``MDLMSampler``
would (greedy, low-confidence remasking, single block), while the other model
(``scorer``) is run on the identical canvas at every step. For each token the
driver commits we therefore get a matched pair of distributions and can report:

* ``p`` of the committed token under both models, hence the per-token log-odds
  ``log p_driver - log p_scorer`` -- a memorisation score localised to a token;
* the rank the committed token holds under the scorer (1 vs 50000 separates
  "the oracle would also say this" from "the oracle has no idea");
* full-distribution Jensen-Shannon divergence at that position, which catches
  disagreement the committed token alone misses;
* predictive entropy under each model, and what the scorer would have said
  instead (top-k alternatives).

Run it in both directions so each model's own trajectory is traced with the
counterpart cross-scored on it.

Usage:
    python scripts/dlm/probe_dual_decode.py \
        --model-a saves/finetune/tofu_LLaDA-8B-Instruct_retain100_150epochs \
        --model-b saves/finetune/tofu_LLaDA-8B-Instruct_retain95_150epochs \
        --tag-a retain100 --tag-b retain95 \
        --split forget05 --n 20 --out saves/probe/dual_decode_llada_forget05
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

IGNORE_INDEX = -100


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
def load_dlm(ckpt, device):
    from omegaconf import OmegaConf

    from model import get_model

    model, tokenizer = get_model(
        OmegaConf.create(
            {
                "model_type": "dlm",
                "model_args": {
                    "pretrained_model_name_or_path": ckpt,
                    "torch_dtype": "bfloat16",
                },
                "tokenizer_args": {"pretrained_model_name_or_path": ckpt},
            }
        )
    )
    return model.to(device).eval(), tokenizer


def load_examples(tokenizer, split, n):
    """Prompt-only ``input_ids`` plus the trimmed ground-truth answer, per example."""
    from evals.metrics.dlm_utils import derive_stop_token_ids
    from data.qa import QADataset

    ds = QADataset(
        hf_args={"path": "locuslab/TOFU", "name": split, "split": "train"},
        template_args={"apply_chat_template": True, "system_prompt": None},
        tokenizer=tokenizer,
        question_key="question",
        answer_key="answer",
        max_length=512,
        predict_with_generate=True,
    )
    stop_ids = sorted(derive_stop_token_ids(tokenizer))
    stop = torch.tensor(stop_ids) if stop_ids else None

    def trim(ids):
        if stop is None:
            return ids
        hit = torch.isin(ids, stop).nonzero(as_tuple=True)[0]
        return ids[: hit[0]] if hit.numel() else ids

    out = []
    for i in range(min(n, len(ds))):
        item = ds[i]
        prompt_ids = item["input_ids"]
        labels = item["labels"]
        # `labels` holds the whole conversation with no IGNORE_INDEX over the
        # prompt span, so slice by prompt length before trimming (see
        # generate_answers_diffusion for why trimming the raw labels is wrong).
        gt_ids = trim(labels[labels != IGNORE_INDEX][prompt_ids.shape[0] :])
        out.append(
            {
                "index": int(item["index"]),
                "question": ds.data[i]["question"],
                "answer": ds.data[i]["answer"],
                "prompt_ids": prompt_ids,
                "gt_ids": gt_ids,
                "ground_truth": tokenizer.decode(gt_ids, skip_special_tokens=True).strip(),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# the traced decode
# --------------------------------------------------------------------------- #
def _entropy(logp):
    """Shannon entropy in nats from log-probabilities over the last dim."""
    return -(logp.exp() * logp).sum(-1)


def _js(logp, logq):
    """Jensen-Shannon divergence in nats between two log-prob vectors."""
    p, q = logp.exp(), logq.exp()
    logm = ((p + q) / 2).clamp_min(1e-12).log()
    return 0.5 * ((p * (logp - logm)).sum(-1) + (q * (logq - logm)).sum(-1))


@torch.no_grad()
def traced_decode(driver, scorer, tokenizer, prompt_ids, gen_cfg, scheduler, topk=5):
    """Greedy low-confidence MDLM decode by ``driver``, cross-scored by ``scorer``.

    Reproduces ``MDLMSampler.sample`` for a single sequence and a single block
    (``block_size >= max_new_tokens``, which is the eval default), recording one
    row per committed token. ``scorer`` never influences the trajectory.
    """
    from dllm.core.samplers.utils import get_num_transfer_tokens

    max_new = gen_cfg["max_new_tokens"]
    steps = gen_cfg["steps"]
    assert gen_cfg["block_size"] >= max_new, "single-block decode only"
    assert gen_cfg["temperature"] == 0.0, "greedy decode only (temperature=0)"

    mask_id, eos_id = tokenizer.mask_token_id, tokenizer.eos_token_id
    dev = driver.device
    p_len = prompt_ids.shape[0]
    T = p_len + max_new

    x = torch.full((1, T), eos_id, dtype=torch.long, device=dev)
    x[0, :p_len] = prompt_ids.to(dev)
    x[0, p_len:] = mask_id
    attn = torch.ones((1, T), dtype=torch.long, device=dev)

    num_transfer = get_num_transfer_tokens(
        mask_index=(x[:, p_len:] == mask_id),
        steps=steps,
        scheduler=scheduler,
        stochastic=False,
    )

    rows, step_summary = [], []
    for s in range(num_transfer.size(1)):
        k = int(num_transfer[0, s].item())
        if k <= 0:
            continue
        masked = x[0] == mask_id

        logp_d = F.log_softmax(driver(x, attention_mask=attn).logits[0].float(), dim=-1)
        logp_s = F.log_softmax(
            scorer(x.to(scorer.device), attention_mask=attn.to(scorer.device))
            .logits[0]
            .float()
            .to(dev),
            dim=-1,
        )

        # Driver picks the argmax token per position and commits the k masked
        # positions where it is most confident.
        conf_d, x0 = logp_d.exp().max(dim=-1)
        confidence = torch.where(masked, conf_d, torch.full_like(conf_d, -float("inf")))
        sel = torch.topk(confidence, k=k).indices

        ent_d, ent_s = _entropy(logp_d[sel]), _entropy(logp_s[sel])
        js = _js(logp_d[sel], logp_s[sel])
        top_d = logp_d[sel].topk(topk, dim=-1)
        top_s = logp_s[sel].topk(topk, dim=-1)

        for r, pos in enumerate(sel.tolist()):
            tok = int(x0[pos].item())
            lp_d = float(logp_d[pos, tok])
            lp_s = float(logp_s[pos, tok])
            rows.append(
                {
                    "step": s,
                    "pos": pos - p_len,  # position within the answer
                    "token_id": tok,
                    "piece": tokenizer.convert_ids_to_tokens([tok])[0],
                    "text": tokenizer.decode([tok]),
                    "n_masked_before": int(masked.sum().item()),
                    "driver": {
                        "logp": lp_d,
                        "entropy": float(ent_d[r]),
                        "top": [
                            {"text": tokenizer.decode([t]), "logp": float(v)}
                            for t, v in zip(top_d.indices[r].tolist(), top_d.values[r])
                        ],
                    },
                    "scorer": {
                        "logp": lp_s,
                        "entropy": float(ent_s[r]),
                        # rank of the committed token under the scorer, 1-based
                        "rank": int((logp_s[pos] > lp_s).sum().item()) + 1,
                        "argmax_agrees": bool(int(logp_s[pos].argmax()) == tok),
                        "top": [
                            {"text": tokenizer.decode([t]), "logp": float(v)}
                            for t, v in zip(top_s.indices[r].tolist(), top_s.values[r])
                        ],
                    },
                    "log_odds": lp_d - lp_s,
                    "js": float(js[r]),
                }
            )

        step_summary.append(
            {
                "step": s,
                "k": k,
                "n_masked": int(masked.sum().item()),
                "mean_conf_masked": float(conf_d[masked].mean()),
                "mean_js_committed": float(js.mean()),
            }
        )
        x[0, sel] = x0[sel]

    return x[0, p_len:], rows, step_summary


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="checkpoint that saw the forget split")
    ap.add_argument("--model-b", required=True, help="oracle that never saw it")
    ap.add_argument("--tag-a", default="retain100")
    ap.add_argument("--tag-b", default="retain95")
    ap.add_argument("--split", default="forget05")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="saves/probe/dual_decode")
    ap.add_argument("--device-a", default="cuda:0")
    ap.add_argument("--device-b", default="cuda:1")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    from rouge_score import rouge_scorer

    from dllm.core.schedulers import make_alpha_scheduler
    from evals.metrics.dlm_utils import derive_stop_token_ids

    gen_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "steps": args.steps,
        "block_size": args.max_new_tokens,
        "temperature": 0.0,
    }
    scheduler = make_alpha_scheduler("LinearAlphaScheduler")

    print(f"[load] {args.tag_a} <- {args.model_a} on {args.device_a}")
    model_a, tokenizer = load_dlm(args.model_a, args.device_a)
    print(f"[load] {args.tag_b} <- {args.model_b} on {args.device_b}")
    model_b, tok_b = load_dlm(args.model_b, args.device_b)
    assert tokenizer.get_vocab() == tok_b.get_vocab(), "checkpoints must share a vocab"

    examples = load_examples(tokenizer, args.split, args.n)
    print(f"[data] {len(examples)} examples from {args.split}")

    stop_ids = sorted(derive_stop_token_ids(tokenizer))
    stop = torch.tensor(stop_ids) if stop_ids else None
    scorer_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def trim_gen(ids):
        if stop is None:
            return ids
        hit = torch.isin(ids.cpu(), stop).nonzero(as_tuple=True)[0]
        return ids[: hit[0]] if hit.numel() else ids

    directions = [
        (f"{args.tag_a}_driven", model_a, model_b, args.tag_a, args.tag_b),
        (f"{args.tag_b}_driven", model_b, model_a, args.tag_b, args.tag_a),
    ]

    out = {
        "meta": {
            "split": args.split,
            "models": {args.tag_a: args.model_a, args.tag_b: args.model_b},
            "tag_a": args.tag_a,
            "tag_b": args.tag_b,
            "generation": gen_cfg,
            "scheduler": "LinearAlphaScheduler",
        },
        "examples": [],
    }

    for ei, ex in enumerate(examples):
        rec = {k: ex[k] for k in ("index", "question", "answer", "ground_truth")}
        rec["runs"] = {}
        for name, driver, scorer, d_tag, s_tag in directions:
            gen_ids, rows, summary = traced_decode(
                driver, scorer, tokenizer, ex["prompt_ids"], gen_cfg, scheduler, args.topk
            )
            gen = tokenizer.decode(trim_gen(gen_ids), skip_special_tokens=True).strip()
            rs = scorer_rouge.score(ex["ground_truth"], gen)
            rec["runs"][name] = {
                "driver": d_tag,
                "scorer": s_tag,
                "generation": gen,
                "rougeL_recall": rs["rougeL"].recall,
                "rougeL_f1": rs["rougeL"].fmeasure,
                "n_committed": len(rows),
                "tokens": rows,
                "steps": summary,
            }
            print(
                f"[{ei:3d}/{len(examples)}] {name:>18s} rougeL_r={rs['rougeL'].recall:.3f} "
                f"tok={len(rows):3d} | {gen[:70]!r}"
            )
        out["examples"].append(rec)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "trace.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[done] wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
