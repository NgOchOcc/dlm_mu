"""Diagnose why Dream (but not LLaDA) loses retain performance after unlearning.

Sweeps the masking rate and reports, per rate, the *unweighted* answer-token
cross-entropy and the EOS mass the model puts on masked positions. The training
objective weights those same positions very differently on the two backbones
(Dream: ``cart[geo_p:0.3]``, LLaDA: ``scheduler`` = 1/t), so comparing a base
checkpoint against its unlearned counterpart at matched mask rates shows which
part of the diffusion time axis the retain anchor actually protected.

Usage:
    python scripts/dlm/diag_retain_collapse.py --model dream \
        --ckpt saves/finetune/tofu_Dream-v0-Instruct-7B_retain100_150_epochs \
        --tag dream_base --out /tmp/diag
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

MODEL_CFG = {
    "dream": {"name": "Dream-v0-Instruct-7B", "right_shift": True, "weight": "cart"},
    "llada": {"name": "LLaDA-8B-Instruct", "right_shift": False, "weight": "scheduler"},
}
MASK_RATES = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]


def cart_weight(masked_mask, p=0.3):
    """Dream's CART loss weight: geometric mass of nearby *unmasked* context."""
    _, l = masked_mask.shape
    idx = torch.arange(l, device=masked_mask.device)
    dist = idx[None, :] - idx[:, None]
    geo = (
        math.log(p) + (dist.abs() - 1).clamp(min=0) * math.log(1 - p)
    ).exp() * 0.5
    geo = geo.masked_fill(dist == 0, 0.0)
    return ((~masked_mask).float() @ geo.T) * masked_mask.float()


def build_batches(tokenizer, split, n, batch_size):
    from data.collators import DataCollatorForSupervisedDataset
    from data.qa import QADataset

    ds = QADataset(
        hf_args={"path": "locuslab/TOFU", "name": split, "split": "train"},
        template_args={"apply_chat_template": True, "system_prompt": None},
        tokenizer=tokenizer,
        question_key="question",
        answer_key="answer",
        max_length=512,
    )
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, index="index")
    items = [ds[i] for i in range(min(n, len(ds)))]
    return [collator(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


@torch.no_grad()
def sweep(model, tokenizer, batches, right_shift, mask_id, eos_id, seed=0):
    """Unweighted CE / EOS mass / objective weight, per masking rate."""
    out = {}
    for rate in MASK_RATES:
        ce_sum = w_ce_sum = eos_sum = n_tok = 0.0
        ce_all = []
        for bi, batch in enumerate(batches):
            torch.manual_seed(seed + bi)  # same mask pattern across checkpoints
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            attn = batch["attention_mask"].to(model.device)

            # The collator derives attention_mask as input_ids.ne(pad), and
            # pad == eos on both backbones, so the answer's own terminal EOS is
            # labelled but unattended. Masking it yields a ~30-nat outlier that
            # swamps the mean, so restrict to attended answer positions.
            maskable = (labels != -100) & attn.bool()
            masked = (torch.rand(input_ids.shape, device=model.device) < rate) & maskable
            noised = torch.where(masked, mask_id, input_ids)

            logits = model(input_ids=noised, attention_mask=attn).logits
            if right_shift:
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            ce = F.cross_entropy(
                logits.transpose(1, 2).float(), input_ids, reduction="none"
            )
            logp = F.log_softmax(logits.float(), dim=-1)
            p_eos = logp[..., eos_id].exp()

            m = masked.float()
            # Objective weight the trainer would apply at this rate.
            w = cart_weight(masked) if right_shift else torch.full_like(m, 1.0 / rate)

            ce_sum += (ce * m).sum().item()
            w_ce_sum += (ce * w * m).sum().item()
            eos_sum += (p_eos * m).sum().item()
            n_tok += m.sum().item()
            ce_all.append(ce[masked].float().cpu())

        ce_cat = torch.cat(ce_all) if ce_all else torch.zeros(1)
        out[rate] = {
            "ce_masked": ce_sum / max(n_tok, 1),
            "ce_median": ce_cat.median().item(),
            "weighted_ce_masked": w_ce_sum / max(n_tok, 1),
            "p_eos_masked": eos_sum / max(n_tok, 1),
            "n_masked_tokens": n_tok,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_CFG), required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/tmp/diag")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    cfg = MODEL_CFG[args.model]
    from omegaconf import OmegaConf

    from model import get_model

    model, tokenizer = get_model(
        OmegaConf.create(
            {
                "model_type": "dlm",
                "model_args": {
                    "pretrained_model_name_or_path": args.ckpt,
                    "torch_dtype": "bfloat16",
                },
                "tokenizer_args": {"pretrained_model_name_or_path": args.ckpt},
            }
        )
    )
    model = model.to("cuda").eval()
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    print(f"[{args.tag}] mask_id={mask_id} eos_id={eos_id} pad={tokenizer.pad_token_id}")

    result = {"tag": args.tag, "ckpt": args.ckpt, "model": args.model, "splits": {}}
    for split in ["retain90", "forget10"]:
        batches = build_batches(tokenizer, split, args.n, args.batch_size)
        result["splits"][split] = sweep(
            model, tokenizer, batches, cfg["right_shift"], mask_id, eos_id
        )
        for rate, v in result["splits"][split].items():
            print(
                f"[{args.tag}] {split} t={rate:<5} ce={v['ce_masked']:9.4f} "
                f"med={v['ce_median']:8.4f} w_ce={v['weighted_ce_masked']:10.4f} "
                f"p_eos={v['p_eos_masked']:.4f}"
            )

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.tag}] wrote {path}")


if __name__ == "__main__":
    main()
