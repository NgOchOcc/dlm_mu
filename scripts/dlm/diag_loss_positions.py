"""Locate which answer positions carry the Dream SFT checkpoint's residual loss.

The SFT log reports a CART-weighted loss of ~5e-4 on retain100, which implies a
raw CE of ~6e-3, yet sweeping the mask rate measures a *mean* CE near 1 nat with
a median of 0. So the loss is concentrated in a few positions. This prints the
worst ones with their tokens, which distinguishes a real fit deficit from a
scaffold/measurement artifact.

Usage:
    python scripts/dlm/diag_loss_positions.py --model dream \
        --ckpt saves/finetune/tofu_Dream-v0-Instruct-7B_retain100_150_epochs
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

TOKENIZER_HUB = {
    "dream": "Dream-org/Dream-v0-Instruct-7B",
    "llada": "GSAI-ML/LLaDA-8B-Instruct",
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(TOKENIZER_HUB), required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="retain90")
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=16)
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

    torch.manual_seed(0)
    maskable = (labels != -100) & attn.bool()
    masked = (torch.rand(ids.shape, device=ids.device) < args.rate) & maskable
    noised = torch.where(masked, tokenizer.mask_token_id, ids)

    logits = model(input_ids=noised, attention_mask=attn).logits
    if right_shift:
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    ce = F.cross_entropy(logits.transpose(1, 2).float(), ids, reduction="none")

    vals = ce[masked]
    print(f"masked tokens={vals.numel()} mean={vals.mean():.4f} "
          f"median={vals.median():.4f} max={vals.max():.4f}")
    for q in [0.5, 0.9, 0.99]:
        print(f"  quantile {q}: {vals.float().quantile(q):.4f}")

    # Worst positions, with the token and where it sits in the response block.
    rows = masked.nonzero()
    order = ce[masked].argsort(descending=True)[:15]
    print("\n worst positions:  ce      row  pos  offset_in_answer  token")
    for k in order.tolist():
        r, p = rows[k].tolist()
        ans = (labels[r] != -100).nonzero().flatten()
        off = int((ans == p).nonzero()[0]) if (ans == p).any() else -1
        print(f"   {ce[r, p].item():10.2f}  {r:4d} {p:4d} {off:4d}/{ans.numel():<4d}"
              f"  {tokenizer.decode([ids[r, p]])!r}")

    # Aggregate by token identity to see if a specific scaffold token dominates.
    print("\n by token (top 8 by total CE):")
    tot = {}
    for k in range(rows.shape[0]):
        r, p = rows[k].tolist()
        tid = int(ids[r, p])
        e = tot.setdefault(tid, [0.0, 0])
        e[0] += ce[r, p].item()
        e[1] += 1
    for tid, (s, c) in sorted(tot.items(), key=lambda kv: -kv[1][0])[:8]:
        print(f"   {tokenizer.decode([tid])!r:24s} n={c:4d} total={s:10.2f} mean={s/c:8.3f}")

    # What the model puts at the single worst position instead of the gold token.
    r, p = rows[order[0]].tolist()
    lp = F.log_softmax(logits[r, p].float(), dim=-1)
    top = lp.topk(5)
    ans = (labels[r] != -100).nonzero().flatten()
    print(f"\n worst position detail: row={r} pos={p} "
          f"offset={int((ans == p).nonzero()[0])}/{ans.numel()}")
    print(f"   gold={tokenizer.decode([ids[r, p]])!r} logp={lp[ids[r, p]]:.2f} "
          f"p={lp[ids[r, p]].exp():.3e}")
    print("   context: " + repr(tokenizer.decode(ids[r, max(0, p - 4) : p + 3])))
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"   pred {tokenizer.decode([i])!r:22s} p={torch.tensor(v).exp():.4f}")


if __name__ == "__main__":
    main()
