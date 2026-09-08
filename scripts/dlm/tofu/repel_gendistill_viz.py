#!/usr/bin/env python
"""Visualize DiffusionRepelGenDistill's SECRET-FREE teacher generation on TOFU forget10.

Calls the REAL trainer methods (no stubs) against a real LLaDA SFT teacher on a single
GPU, so it exercises exactly the code path used during training -- minus DeepSpeed
(single process, so ``_gather_E`` / ZeRO-3 collectives are irrelevant here; this only
covers selection + generation):

    _select_forget_positions   -> A_hat (PCSS, topk=0.8, sel_mc=16 by default)
    _query_identity_mask       -> which QUERY tokens get blanked before generating
    _generate_secret_free      -> the teacher's secret-free answer with those tokens masked

For each example the HTML shows:
  * the question with BLANKED query tokens struck through in red (the entities/identity
    the teacher no longer sees),
  * the gold answer with A_hat forget-positions highlighted,
  * the teacher's generated secret-free answer,
  * a per-forget-position leak check (did the teacher reproduce the gold token, or swap it).

The trainer instance is built with ``object.__new__`` and only the attributes the three
methods read are bound, so no HF Trainer / accelerate machinery is needed.

    CUDA_VISIBLE_DEVICES=0 python scripts/dlm/tofu/repel_gendistill_viz.py [--n 12]
"""
from __future__ import annotations

import argparse
import html
import os
import re
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))

from datasets import load_dataset  # noqa: E402
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from trainer.dlm.unlearn.repel_gendistill import DiffusionRepelGenDistill  # noqa: E402

_SPECIAL_RE = re.compile(r"<\|?[^<>]*\|?>")
_WORD_RE = re.compile(r"[A-Za-z0-9]{2,}")


def build_trainer(model, tokenizer, args):
    """Bind only what the selection/generation methods read onto a bare instance."""
    t = object.__new__(DiffusionRepelGenDistill)
    t.processing_class = tokenizer
    t.ref_model = model
    t.mask_id = tokenizer.mask_token_id or getattr(model.config, "mask_token_id", 126336)
    t.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else t.mask_id
    t.vocab_size = len(tokenizer)
    t.time_epsilon = 1e-3
    t.scheduler = lambda tt: 1.0 - tt          # LinearAlphaScheduler: alpha(t)=1-t
    t.right_shift_logits = False               # LLaDA
    t._postprocess_outputs = lambda o: o       # identity when right_shift is off
    # PCSS selection knobs (defaults = config: topk / 0.8 / sel_mc 16)
    t.select_method = args.select_method
    t.topk_frac = args.topk_frac
    t.tau = 0.0
    t.otsu_relax = 0.0
    t.sel_mc = args.sel_mc
    t.drop_question_overlap = True
    # generation knobs
    t.gen_steps = args.gen_steps
    t.keep_answer = args.keep_answer
    t.query_blank_identity = True
    t.keep_subject_name = args.keep_answer and args.keep_subject_name
    t._special_ids_cache = None
    t._eos_ids_cache = None
    return t


def tokenize(tokenizer, question, answer, device):
    msgs = [{"role": "user", "content": question},
            {"role": "assistant", "content": answer}]
    full = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    prompt_ids = tokenizer.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True)
    plen = len(prompt_ids)
    input_ids = torch.tensor([full], device=device)
    b, l = input_ids.shape
    answer_mask = torch.zeros(b, l, dtype=torch.bool, device=device)
    answer_mask[0, plen:] = True
    # drop structural / eos tokens from the answer span
    for pos in range(plen, l):
        surf = tokenizer.decode([int(input_ids[0, pos])])
        if int(input_ids[0, pos]) in tokenizer.all_special_ids or not _SPECIAL_RE.sub("", surf).strip():
            answer_mask[0, pos] = False
    attn = torch.ones_like(input_ids)
    return input_ids, answer_mask, attn, plen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saves/finetune/tofu_LLaDA-8B-Instruct_retain100_150epochs")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--select_method", default="topk", choices=["topk", "otsu", "tau"])
    ap.add_argument("--topk_frac", type=float, default=0.8)
    ap.add_argument("--sel_mc", type=int, default=16)
    ap.add_argument("--gen_steps", type=int, default=128)
    ap.add_argument("--keep_answer", action="store_true")
    ap.add_argument("--keep_subject_name", action="store_true")
    ap.add_argument("--out", default="outputs/repel_gendistill_gen_viz.html")
    args = ap.parse_args()
    os.chdir(REPO)

    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    t = build_trainer(model, tok, args)
    ds = load_dataset("locuslab/TOFU", "forget10", split="train")

    records = []
    with torch.no_grad():
        for i in range(min(args.n, len(ds))):
            q, a = ds[i]["question"], ds[i]["answer"]
            input_ids, answer_mask, attn, plen = tokenize(tok, q, a, device)

            a_hat = t._select_forget_positions(input_ids, answer_mask, attn)
            identity = t._query_identity_mask(input_ids, answer_mask, attn)
            gen = t._generate_secret_free(input_ids, answer_mask, attn, a_hat)

            # question tokens with the blanked (identity) ones marked
            prompt = t._prompt_mask(input_ids, answer_mask, attn)
            q_pos = prompt[0].nonzero(as_tuple=True)[0].tolist()
            q_toks = [(tok.decode([int(input_ids[0, p])]), bool(identity[0, p])) for p in q_pos]

            # answer tokens with A_hat marked; generated answer; leak check
            a_pos = answer_mask[0].nonzero(as_tuple=True)[0].tolist()
            gold_toks = [(tok.decode([int(input_ids[0, p])]), bool(a_hat[0, p])) for p in a_pos]
            gen_toks = [tok.decode([int(gen[0, p])]) for p in a_pos]
            n_content = n_leak = 0
            for p in a_pos:
                if not bool(a_hat[0, p]):
                    continue
                gsf = tok.decode([int(input_ids[0, p])])
                if not _WORD_RE.search(gsf):
                    continue
                n_content += 1
                if int(gen[0, p]) == int(input_ids[0, p]):
                    n_leak += 1
            records.append({
                "idx": i, "question": q, "answer": a,
                "q_toks": q_toks, "gold_toks": gold_toks, "gen_toks": gen_toks,
                "n_ahat": int(a_hat[0].sum()), "n_ans": len(a_pos),
                "n_content": n_content, "n_leak": n_leak,
                "leak_rate": round(n_leak / n_content, 3) if n_content else 0.0,
            })
            print(f"[{i}] |ans|={len(a_pos)} |A_hat|={int(a_hat[0].sum())} "
                  f"|blanked_q|={int(identity[0].sum())} leak={n_leak}/{n_content}", flush=True)

    write_html(records, args)


def write_html(records, args):
    def esc(s):
        return html.escape(s or "")

    def q_span(toks):
        out = []
        for surf, blank in toks:
            s = esc(surf)
            if blank:
                out.append(f"<span class='blank'>{s}</span>")
            else:
                out.append(s)
        return "".join(out)

    def ans_span(gold_toks, gen_toks):
        gold = []
        for surf, is_f in gold_toks:
            gold.append(f"<span class='fpos'>{esc(surf)}</span>" if is_f else esc(surf))
        gen = []
        for (gsurf, is_f), gnsurf in zip(gold_toks, gen_toks):
            if is_f and _WORD_RE.search(gsurf):
                cls = "leak" if gnsurf == gsurf else "swap"
                gen.append(f"<span class='{cls}'>{esc(gnsurf)}</span>")
            else:
                gen.append(esc(gnsurf))
        return "".join(gold), "".join(gen)

    rows = []
    for r in records:
        gold_html, gen_html = ans_span(r["gold_toks"], r["gen_toks"])
        badge = ("leak-hi" if r["leak_rate"] >= 0.5 else "leak-ok")
        rows.append(
            f"<div class='card'>"
            f"<div class='hd'>#{r['idx']} · A_hat {r['n_ahat']}/{r['n_ans']} · "
            f"leak <span class='{badge}'>{r['n_leak']}/{r['n_content']} ({r['leak_rate']})</span></div>"
            f"<div class='lab'>Q (blanked = <span class='blank'>đỏ gạch</span> = teacher không thấy):</div>"
            f"<div class='txt'>{q_span(r['q_toks'])}</div>"
            f"<div class='lab'>Gold (<span class='fpos'>A_hat</span>):</div><div class='txt'>{gold_html}</div>"
            f"<div class='lab'>GEN secret-free (<span class='leak'>leak</span>/<span class='swap'>swap</span> tại A_hat):</div>"
            f"<div class='txt'>{gen_html}</div>"
            f"</div>"
        )
    n_leak_hi = sum(1 for r in records if r["leak_rate"] >= 0.5)
    doc = f"""<!doctype html><meta charset='utf-8'><title>RepelGenDistill secret-free gen</title>
<style>
 body{{font:14px system-ui;margin:20px;background:#faf9f7;color:#111;line-height:1.6}}
 .wrap{{max-width:1000px;margin:0 auto}}
 h1{{font-size:19px}} .sub{{color:#666;font-size:13px}}
 .card{{background:#fff;border:1px solid #e2e0d8;border-radius:10px;padding:12px 16px;margin:12px 0}}
 .hd{{font-weight:600;font-size:13px;margin-bottom:8px}}
 .lab{{font-size:11px;color:#888;margin-top:8px}}
 .txt{{font-family:ui-monospace,Menlo,monospace;font-size:13px;white-space:pre-wrap;word-break:break-word}}
 .blank{{color:#c00;text-decoration:line-through;background:#fde2e2;border-radius:3px}}
 .fpos{{background:#fff2b8;border-radius:3px}}
 .leak{{background:#fde2e2;color:#a00;font-weight:600;border-radius:3px}}
 .swap{{background:#dcf5dc;color:#070;border-radius:3px}}
 .leak-hi{{color:#a00;font-weight:700}} .leak-ok{{color:#070}}
</style>
<div class='wrap'>
<h1>RepelGenDistill — teacher sinh secret-free (mask entity trong query)</h1>
<p class='sub'>model: <code>{esc(args.model)}</code> · select={args.select_method} topk_frac={args.topk_frac} sel_mc={args.sel_mc}
 gen_steps={args.gen_steps} keep_answer={args.keep_answer} · query_blank_identity=True<br>
 {len(records)} ví dụ · leak_rate ≥ 0.5: <b>{n_leak_hi}</b>. Gọi trực tiếp
 <code>_select_forget_positions → _query_identity_mask → _generate_secret_free</code> của trainer.</p>
{''.join(rows)}
</div>"""
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"\nwrote {args.out} ({len(records)} examples)")


if __name__ == "__main__":
    main()
