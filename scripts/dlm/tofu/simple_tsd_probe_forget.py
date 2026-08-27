"""Run the SimpleTSD PCSS pre-processing (Alg. 1) over the whole forget set and
dump the selected forget set ``A_hat`` per example, so we can inspect *which*
answer tokens the algorithm decides to unlearn.

This reuses the real training pipeline (``get_model`` / ``get_data`` /
``get_collators``, driven by the same Hydra config as ``src/train.py``) so the
tokenisation, chat template and answer span match training exactly. It does NOT
build the full DeepSpeed trainer: PCSS only needs the frozen teacher, so we bind
the attributes ``DiffusionSimpleTSD._select_forget_positions`` reads onto a bare
instance (``object.__new__``) with the loaded SFT model as ``ref_model`` -- one
model copy, no optimizer, fits on a single GPU.

Launch (pick a GPU with free memory; teacher is LLaDA-8B bf16 ~16 GB):

    CUDA_VISIBLE_DEVICES=4 python scripts/dlm/tofu/simple_tsd_probe_forget.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/tofu/dlm_simple_tsd \
        model=LLaDA-8B-Instruct \
        model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_LLaDA-8B-Instruct_retain100_150epochs \
        forget_split=forget10 \
        task_name=probe_simpletsd_forget10

Tunable via env:
    PROBE_SEL_MC       (default: trainer.method_args.sel_mc)  # partial-mask draws when scoring
    PROBE_OTSU_RELAX   (default: trainer.method_args.otsu_relax)
    PROBE_SELECT_METHOD(default: trainer.method_args.select_method)  # otsu|topk|tau
    PROBE_BATCH_SIZE   (default: 8)
    PROBE_LIMIT        (default: 0 = all forget examples)
    PROBE_OUT_DIR      (default: logs/simpletsd_probe)
"""

import json
import os

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data import get_collators, get_data
from dllm.core.schedulers import make_alpha_scheduler
from model import get_model
from trainer.dlm.unlearn.simple_tsd import DiffusionSimpleTSD
from trainer.utils import seed_everything


def _env_or(cfg_val, env_key, cast):
    v = os.getenv(env_key)
    return cast(v) if v not in (None, "") else cfg_val


def _build_probe(cfg, tokenizer, model):
    """A bare ``DiffusionSimpleTSD`` carrying only what ``_select_forget_positions``
    (and the teacher/threshold helpers it calls) reads -- no Trainer/DeepSpeed init."""
    ma = cfg.trainer.method_args
    t = object.__new__(DiffusionSimpleTSD)
    t.processing_class = tokenizer
    t.mask_id = tokenizer.mask_token_id
    t.pad_id = tokenizer.pad_token_id
    t.vocab_size = len(tokenizer)
    t.ref_model = model
    # Selection knobs -- default to the trainer config, overridable via env.
    t.select_method = _env_or(ma.get("select_method", "otsu"), "PROBE_SELECT_METHOD", str)
    t.otsu_relax = _env_or(float(ma.get("otsu_relax", 0.0)), "PROBE_OTSU_RELAX", float)
    t.sel_mc = _env_or(int(ma.get("sel_mc", 8)), "PROBE_SEL_MC", int)
    t.tau = float(ma.get("tau", 0.0))
    t.topk_frac = float(ma.get("topk_frac", 0.5))
    t.drop_question_overlap = bool(ma.get("drop_question_overlap", True))
    # Base-trainer attributes the teacher/masking path reads.
    t.right_shift_logits = bool(ma.get("right_shift_logits", False))
    t.time_epsilon = float(cfg.trainer.args.get("time_epsilon", 1e-3))
    t.scheduler = make_alpha_scheduler(
        cfg.trainer.args.get("scheduler_name", "LinearAlphaScheduler")
    )
    return t


def _decode_words(tokenizer, ids):
    """Per-token decoded surfaces (leading space marks a new whitespace word)."""
    return [tokenizer.decode([int(i)]) for i in ids]


def _render_answer(surfaces, sel):
    """Answer text with A_hat tokens wrapped in >>...<< so they stand out."""
    out = []
    for s, is_sel in zip(surfaces, sel):
        out.append(f">>{s.strip() or s}<<" if is_sel else s)
    return "".join(out)


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
    forget_ds = data["train"]  # ForgetRetainDataset (retain=null -> forget-only)

    batch_size = int(os.getenv("PROBE_BATCH_SIZE", "8"))
    limit = int(os.getenv("PROBE_LIMIT", "0"))
    loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    probe = _build_probe(cfg, tokenizer, model)

    out_dir = os.path.abspath(os.getenv("PROBE_OUT_DIR", "logs/simpletsd_probe"))
    os.makedirs(out_dir, exist_ok=True)
    split = str(cfg.get("forget_split", "forget"))
    jsonl_path = os.path.join(out_dir, f"ahat_{split}.jsonl")
    txt_path = os.path.join(out_dir, f"ahat_{split}.txt")

    print(
        f"[probe] split={split} select_method={probe.select_method} "
        f"otsu_relax={probe.otsu_relax} sel_mc={probe.sel_mc} "
        f"drop_question_overlap={probe.drop_question_overlap} bs={batch_size} "
        f"device={device}",
        flush=True,
    )

    n_ex = 0
    tot_ans = tot_sel = 0
    frac_sum = 0.0
    jf = open(jsonl_path, "w")
    tf = open(txt_path, "w")
    for bi_batch, batch in enumerate(loader):
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

        b = input_ids.shape[0]
        for j in range(b):
            am = answer_mask[j]
            ans_idx = am.nonzero(as_tuple=True)[0]
            if ans_idx.numel() == 0:
                continue
            prompt = probe._prompt_mask(input_ids[j : j + 1], am.unsqueeze(0),
                                        None if attn is None else attn[j : j + 1])[0]
            q_ids = input_ids[j][prompt].tolist()
            q_text = tokenizer.decode(q_ids, skip_special_tokens=True)

            ans_ids = input_ids[j][ans_idx]
            surfaces = _decode_words(tokenizer, ans_ids)
            sel = a_hat[j][ans_idx].tolist()
            sc = scores[j][ans_idx].tolist()
            ans_text = tokenizer.decode(ans_ids, skip_special_tokens=True)

            n_sel = int(sum(sel))
            n_ans = int(len(sel))
            forget_tokens = [surfaces[k].strip() or surfaces[k] for k in range(n_ans) if sel[k]]

            n_ex += 1
            tot_ans += n_ans
            tot_sel += n_sel
            frac_sum += (n_sel / n_ans) if n_ans else 0.0

            rec = {
                "example": n_ex - 1,
                "question": q_text,
                "answer": ans_text,
                "n_answer_tokens": n_ans,
                "n_forget_tokens": n_sel,
                "selected_frac": round(n_sel / n_ans, 4) if n_ans else 0.0,
                "forget_tokens": forget_tokens,
                "tokens": [
                    {"tok": surfaces[k], "score": round(float(sc[k]), 4), "in_a_hat": bool(sel[k])}
                    for k in range(n_ans)
                ],
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            tf.write(f"### Example {n_ex - 1}\n")
            tf.write(f"Q: {q_text}\n")
            tf.write(f"A: {ans_text}\n")
            tf.write(f"A_hat ({n_sel}/{n_ans} tokens): {forget_tokens}\n")
            tf.write(f"marked: {_render_answer(surfaces, sel)}\n\n")

            if n_ex <= 5:
                print(f"--- example {n_ex - 1} ---", flush=True)
                print(f"Q: {q_text}", flush=True)
                print(f"A: {ans_text}", flush=True)
                print(f"A_hat ({n_sel}/{n_ans}): {forget_tokens}", flush=True)
                print(f"marked: {_render_answer(surfaces, sel)}\n", flush=True)

        if (bi_batch + 1) % 5 == 0:
            print(f"[probe] processed {n_ex} examples ...", flush=True)
        if limit and n_ex >= limit:
            break

    jf.close()
    tf.close()
    summary = {
        "split": split,
        "n_examples": n_ex,
        "total_answer_tokens": tot_ans,
        "total_forget_tokens": tot_sel,
        "token_level_selected_frac": round(tot_sel / tot_ans, 4) if tot_ans else 0.0,
        "mean_per_example_selected_frac": round(frac_sum / n_ex, 4) if n_ex else 0.0,
        "select_method": probe.select_method,
        "otsu_relax": probe.otsu_relax,
        "sel_mc": probe.sel_mc,
        "drop_question_overlap": probe.drop_question_overlap,
    }
    with open(os.path.join(out_dir, f"summary_{split}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[probe] DONE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[probe] per-example JSONL : {jsonl_path}", flush=True)
    print(f"[probe] readable dump     : {txt_path}", flush=True)


if __name__ == "__main__":
    main()
