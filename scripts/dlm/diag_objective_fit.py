"""Measure an SFT checkpoint's fit under the objective it was actually trained on.

The earlier diagnostics masked positions by raw ``labels != -100``, which is *not*
what the trainer supervises: with ``eos_pad_to`` set, ``_extend_response_with_eos``
rewrites everything past the turn terminator into an EOS run out to a fixed block
width. Measuring the raw span therefore scores the model on the chat template's
post-answer scaffold, which was never a training target -- producing ~24-nat
"unlearnable token" artifacts.

This script binds the trainer's own ``_extend_response_with_eos`` (so the
transformation cannot drift from training) and reports CE split by region:

    content    answer tokens up to and including the last non-scaffold token
    terminator the single turn-end token (im_end / eot_id)
    eos_run    the EOS padding out to ``eos_pad_to``

A backbone that is genuinely under-fit on retain shows it in *content*. A high
number confined to *eos_run* or *terminator* is a formatting effect, not a
knowledge-retention deficit.

Usage:
    python scripts/dlm/diag_objective_fit.py --model dream \
        --ckpt saves/finetune/tofu_Dream-v0-Instruct-7B_retain100_150_epochs \
        --split retain90
"""

import argparse
import os
import sys
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

TOKENIZER_HUB = {
    "dream": "Dream-org/Dream-v0-Instruct-7B",
    "llada": "GSAI-ML/LLaDA-8B-Instruct",
}


def make_extender(tokenizer, eos_pad_to):
    """A minimal stand-in carrying the trainer's real extension logic."""
    from trainer.dlm.base import DiffusionFinetuneTrainer

    shim = types.SimpleNamespace(
        processing_class=tokenizer, eos_pad_to=eos_pad_to
    )
    for name in ("_extend_response_with_eos", "_get_terminator_tensor"):
        setattr(shim, name, getattr(DiffusionFinetuneTrainer, name).__get__(shim))
    return shim


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(TOKENIZER_HUB), required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="retain90")
    ap.add_argument("--rates", default="0.1,0.5,0.9")
    ap.add_argument("--eos-pad-to", type=int, default=128)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument(
        "--by-offset",
        type=int,
        default=0,
        help="If >0, also bucket content positions into this many offset bins and "
        "report CE and P(EOS) per bin. Early-terminating generations show up as "
        "P(EOS) rising with offset, which teacher-forced CE alone cannot reveal.",
    )
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from data.collators import DataCollatorForSupervisedDataset
    from data.qa import QADataset
    from model import get_model

    right_shift = args.model == "dream"
    model, tokenizer = get_model(
        OmegaConf.create(
            {
                "model_type": "dlm",
                "model_args": {
                    "pretrained_model_name_or_path": args.ckpt,
                    "torch_dtype": "bfloat16",
                },
                "tokenizer_args": {
                    "pretrained_model_name_or_path": TOKENIZER_HUB[args.model]
                },
            }
        )
    )
    model = model.to("cuda").eval()

    ds = QADataset(
        hf_args={"path": "locuslab/TOFU", "name": args.split, "split": "train"},
        template_args={"apply_chat_template": True, "system_prompt": None},
        tokenizer=tokenizer,
        question_key="question",
        answer_key="answer",
        max_length=512,
    )
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, index="index")
    batch = collator([ds[i] for i in range(args.n)])

    ids = batch["input_ids"].to(model.device)
    labels = batch["labels"].to(model.device)
    attn = batch["attention_mask"].to(model.device)

    shim = make_extender(tokenizer, args.eos_pad_to)
    term = shim._get_terminator_tensor(ids.device)

    # Region tags computed from the pre-extension span: content is everything up to
    # the terminator, the terminator itself is its own bucket, the rest is EOS pad.
    b = ids.shape[0]
    orig_sup = labels != -100
    term_pos = []
    for i in range(b):
        sup = orig_sup[i].nonzero(as_tuple=True)[0]
        hit = torch.isin(ids[i, sup], term).nonzero(as_tuple=True)[0] if term is not None else torch.tensor([], dtype=torch.long)
        term_pos.append(int(sup[int(hit[0])]) if hit.numel() else None)

    ids, labels, attn = shim._extend_response_with_eos(ids, labels, attn)
    if right_shift:
        from dllm.utils.data import prepend_bos

        out = prepend_bos(
            {"input_ids": ids, "labels": labels, "attention_mask": attn},
            bos_token_id=tokenizer.bos_token_id,
            label_pad_token_id=-100,
        )
        ids, labels, attn = out["input_ids"], out["labels"], out["attention_mask"]
        term_pos = [None if p is None else p + 1 for p in term_pos]

    region = torch.zeros_like(ids)  # 0 = not supervised
    sup = labels != -100
    for i in range(b):
        s = sup[i].nonzero(as_tuple=True)[0]
        if not s.numel():
            continue
        tp = term_pos[i]
        if tp is None:
            region[i, s] = 1
            continue
        region[i, s[s <= tp]] = 1  # content (incl. terminator, split off next)
        region[i, tp] = 2  # terminator
        region[i, s[s > tp]] = 3  # eos run
    names = {1: "content", 2: "terminator", 3: "eos_run"}

    eos_id = tokenizer.eos_token_id
    print(f"\n{args.model}  ckpt={args.ckpt}  split={args.split}  n={b}")
    print(f"  supervised/row: total={int(sup.sum())/b:.1f}  "
          + "  ".join(f"{names[k]}={int((region==k).sum())/b:.1f}" for k in names))

    for rate in [float(x) for x in args.rates.split(",")]:
        torch.manual_seed(0)
        masked = (torch.rand(ids.shape, device=ids.device) < rate) & sup
        noised = torch.where(masked, tokenizer.mask_token_id, ids)
        logits = model(input_ids=noised, attention_mask=attn).logits
        if right_shift:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        ce = F.cross_entropy(logits.transpose(1, 2).float(), ids, reduction="none")
        p_eos = logits.float().softmax(-1)[..., eos_id]

        print(f"  rate={rate:g}")
        for k in names:
            sel = masked & (region == k)
            if not sel.any():
                print(f"    {names[k]:11s} n=0")
                continue
            v = ce[sel]
            print(f"    {names[k]:11s} n={v.numel():5d} mean={v.mean():8.4f} "
                  f"median={v.median():8.4f} max={v.max():9.2f} "
                  f"sum_share={v.sum()/ce[masked].sum():6.1%} "
                  f"p_eos={p_eos[sel].mean():.3f}")

        if not args.by_offset:
            continue
        # Offset within the content span, so the buckets mean the same thing across
        # rows of differing answer length.
        nb = args.by_offset
        content = masked & (region == 1)
        rows, cols = content.nonzero(as_tuple=True)
        span = [(region[i] == 1).nonzero(as_tuple=True)[0] for i in range(b)]
        frac = torch.tensor(
            [
                (span[int(r)] < int(c)).sum().item() / max(len(span[int(r)]), 1)
                for r, c in zip(rows, cols)
            ]
        )
        bucket = (frac * nb).clamp(max=nb - 1).long()
        print(f"    content by offset ({nb} bins, fraction through the answer):")
        for j in range(nb):
            m = bucket == j
            if not m.any():
                continue
            rr, cc = rows[m], cols[m]
            print(f"      [{j/nb:.2f}-{(j+1)/nb:.2f}) n={int(m.sum()):4d} "
                  f"ce={ce[rr, cc].mean():7.4f}  p_eos={p_eos[rr, cc].mean():.4f}  "
                  f"p_eos>0.5={100*(p_eos[rr, cc] > 0.5).float().mean():5.1f}%")


if __name__ == "__main__":
    main()
