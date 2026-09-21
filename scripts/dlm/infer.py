#!/usr/bin/env python
"""Single-sample inference for a masked diffusion LM (Dream, LLaDA) checkpoint.

Loads a checkpoint through the same path the eval pipeline uses
(``model/dlm.py`` -> ``dllm.utils.get_model``) and denoises one prompt with
``dllm``'s ``MDLMSampler``, so the answer matches what ``src/eval.py`` would
produce for that prompt.

Usage:
    scripts/dlm/infer.py --model-dir <ckpt> --prompt "Who is Ngoc?"
    scripts/dlm/infer.py --model-dir <ckpt> --prompt-file q.txt --model dream
    echo "Who is Ngoc?" | scripts/dlm/infer.py --model-dir <ckpt>

``--model {dream,llada,llada1.5}`` only sets ``right_shift_logits`` (True for
Dream, False for LLaDA); it is auto-detected from the checkpoint config when
omitted. Generation defaults mirror ``configs/generation/dlm_default.yaml``.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from model import get_model  # noqa: E402
from evals.metrics.dlm_utils import derive_stop_token_ids  # noqa: E402


# right_shift_logits per family: Dream predicts the token at i-1, LLaDA at i.
# Mirrors dlm_config() in scripts/dlm/lib.sh.
RIGHT_SHIFT_BY_MODEL = {"dream": True, "llada": False, "llada1.5": False}


def detect_right_shift(model, model_dir):
    """True when the checkpoint is a Dream-family model (right-shifted logits)."""
    haystack = str(model_dir).lower()
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    cls_name = type(model).__name__.lower()
    return "dream" in haystack or "dream" in model_type or "dream" in cls_name


def build_prompt_ids(tokenizer, prompt, apply_chat_template, system_prompt):
    """Tokenize the prompt the way data/utils.py does for eval batches."""
    if not apply_chat_template:
        return tokenizer(prompt, return_tensors=None)["input_ids"], prompt
    chat = []
    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})
    chat.append({"role": "user", "content": prompt})
    ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return ids, text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True,
                    help="checkpoint dir or HF id, e.g. saves/finetune/... or Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--tokenizer-dir", default=None,
                    help="defaults to --model-dir (matches how eval scripts override both)")
    ap.add_argument("--prompt", default=None, help="the sample to infer; else --prompt-file or stdin")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--model", choices=sorted(RIGHT_SHIFT_BY_MODEL), default=None,
                    help="family, only used to set right_shift_logits (auto-detected if omitted)")
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--no-chat-template", action="store_true",
                    help="feed the raw prompt instead of the instruct chat template")
    ap.add_argument("--system-prompt", default=None,
                    help="None by default, matching configs/model/*.yaml for DLMs")
    # Generation args: defaults from configs/generation/dlm_default.yaml.
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--remasking", default="low_confidence")
    ap.add_argument("--scheduler-name", default="LinearAlphaScheduler")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-trim", action="store_true",
                    help="keep everything after the turn terminator (debug the raw denoised block)")
    ap.add_argument("--json", default=None, help="also write the result to this JSON file")
    args = ap.parse_args()

    if args.prompt is not None:
        prompt = args.prompt
    elif args.prompt_file:
        prompt = Path(args.prompt_file).read_text()
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read()
    else:
        ap.error("give the sample via --prompt, --prompt-file, or stdin")
    prompt = prompt.strip()
    if not prompt:
        ap.error("empty prompt")

    torch.manual_seed(args.seed)

    model_cfg = OmegaConf.create({
        "model_type": "dlm",
        "model_args": {
            "pretrained_model_name_or_path": args.model_dir,
            "torch_dtype": args.torch_dtype,
        },
        "tokenizer_args": {
            "pretrained_model_name_or_path": args.tokenizer_dir or args.model_dir,
        },
    })
    model, tokenizer = get_model(model_cfg)
    model.eval()
    device = next(model.parameters()).device

    right_shift = (RIGHT_SHIFT_BY_MODEL[args.model] if args.model
                   else detect_right_shift(model, args.model_dir))

    prompt_ids, wrapped_prompt = build_prompt_ids(
        tokenizer, prompt, not args.no_chat_template, args.system_prompt
    )
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
    from dllm.core.schedulers import make_alpha_scheduler

    cfg = MDLMSamplerConfig(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        block_size=args.block_size,
        temperature=args.temperature,
        remasking=args.remasking,
        right_shift_logits=right_shift,
    )
    sampler = MDLMSampler(
        model=model,
        tokenizer=tokenizer,
        scheduler=make_alpha_scheduler(args.scheduler_name),
    )
    with torch.no_grad():
        sequences = sampler.sample([prompt_tensor], config=cfg)

    # The sampler returns [prompt | generated | pad]; keep the generated block only.
    gen_ids = sequences[0][prompt_tensor.shape[0]: prompt_tensor.shape[0] + args.max_new_tokens]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Masked diffusion fills the whole block, so it emits its turn terminator at
    # the true answer end and keeps denoising after it (usually repeating the
    # answer). Cut there, exactly as generate_answers_diffusion does for ROUGE.
    if args.no_trim:
        answer = raw
    else:
        stop_ids = derive_stop_token_ids(tokenizer)
        if stop_ids:
            stop_tensor = torch.tensor(sorted(stop_ids), device=gen_ids.device)
            hit = torch.isin(gen_ids, stop_tensor).nonzero(as_tuple=True)[0]
            if hit.numel() > 0:
                gen_ids = gen_ids[: hit[0]]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    print(f"=== model       : {args.model_dir}")
    print(f"=== right_shift : {right_shift}")
    print(f"=== prompt      : {prompt}")
    print(f"=== answer      :\n{answer}")
    if not args.no_trim and raw != answer:
        print(f"\n=== untrimmed block (after the turn terminator) :\n{raw}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "model_dir": args.model_dir,
            "right_shift_logits": right_shift,
            "prompt": prompt,
            "wrapped_prompt": wrapped_prompt,
            "answer": answer,
            "raw_block": raw,
            "generation_args": {
                "max_new_tokens": args.max_new_tokens,
                "steps": args.steps,
                "block_size": args.block_size,
                "temperature": args.temperature,
                "remasking": args.remasking,
                "scheduler_name": args.scheduler_name,
            },
        }, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
