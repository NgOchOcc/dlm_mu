"""Evaluation primitives for masked diffusion language models (Dream, LLaDA).

DLM counterparts of the autoregressive helpers in ``evals/metrics/utils.py``:

* ``evaluate_probability_diffusion`` — a Monte-Carlo estimate of the answer
  likelihood under the masked-diffusion model (LLaDA Eq.14 conditional
  reconstruction loss), returning the same ``{"prob", "avg_loss"}`` payload as
  the autoregressive ``evaluate_probability`` so the existing
  ``probability``/``truth_ratio`` metrics can consume it unchanged.
* ``eval_text_similarity_diffusion`` — ROUGE against iterative-denoising
  generations produced by ``dllm``'s ``MDLMSampler``.

Low-level diffusion primitives (alpha scheduler, BOS prepend for right-shifted
logits) are imported from the ``dllm`` package rather than reimplemented.
"""

from rouge_score import rouge_scorer
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from data.utils import IGNORE_INDEX


def _build_scheduler(scheduler_name):
    from dllm.core.schedulers import make_alpha_scheduler

    return make_alpha_scheduler(scheduler_name)


def _postprocess_logits(logits, right_shift_logits):
    """Right-shift logits for Dream-style models (predicts token at i-1)."""
    if right_shift_logits:
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits


@torch.no_grad()
def evaluate_probability_diffusion(
    model,
    batch,
    scheduler_name="LinearAlphaScheduler",
    mc_num=128,
    time_epsilon=1e-3,
    right_shift_logits=False,
    mask_token_id=None,
    mc_batch_size=128,
):
    """Eq.(14) conditional reconstruction loss under masked diffusion.

    Replicates the MDU benchmark estimator (LLaDA Appendix A.2, Eq.14): for each
    MC sample draw ``l ~ Uniform{1, L}`` over the ``L`` answer tokens, mask
    exactly ``l`` random answer positions, predict them, and accumulate the
    masked cross-entropy sum divided by ``l``. The per-example loss is the mean
    over ``mc_num`` samples (no ``1/p_mask`` importance weight and no division by
    answer length); ``prob = exp(-loss)`` mirrors the autoregressive metric.

    ``scheduler_name`` and ``time_epsilon`` are accepted for config
    compatibility but unused by the discrete Eq.14 scheme.
    """
    assert mask_token_id is not None, "mask_token_id must be provided for DLM probability"
    device = model.device
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    if right_shift_logits:
        from dllm.utils.data import prepend_bos

        shifted = prepend_bos(
            {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask},
            bos_token_id=model.config.bos_token_id
            if hasattr(model, "config")
            else None,
            label_pad_token_id=IGNORE_INDEX,
        )
        input_ids, labels = shifted["input_ids"], shifted["labels"]
        attention_mask = shifted.get("attention_mask", attention_mask)

    b, seq_len = input_ids.shape
    maskable = labels != IGNORE_INDEX

    results = []
    for i in range(b):
        example_ids = input_ids[i]
        example_attn = attention_mask[i] if attention_mask is not None else None
        answer_indices = torch.nonzero(maskable[i], as_tuple=False).flatten()
        L = answer_indices.shape[0]
        if L == 0:
            # Matches MDU: empty answer → loss 0.0 → prob = exp(0) = 1.0.
            results.append({"prob": 1.0, "avg_loss": 0.0})
            continue

        # Pre-generate all masking patterns (one per MC sample).
        all_noised = example_ids.unsqueeze(0).expand(mc_num, -1).clone()
        mask_flags = torch.zeros(mc_num, seq_len, device=device, dtype=torch.bool)
        ls = []
        for s in range(mc_num):
            l = int(torch.randint(1, L + 1, (1,), device=device).item())
            ls.append(l)
            perm = torch.randperm(L, device=device)[:l]
            positions = answer_indices[perm]
            all_noised[s, positions] = mask_token_id
            mask_flags[s, positions] = True

        target_ids = example_ids.unsqueeze(0).expand(mc_num, -1)
        attn_expand = (
            example_attn.unsqueeze(0).expand(mc_num, -1)
            if example_attn is not None
            else None
        )
        total_score = 0.0
        for start in range(0, mc_num, mc_batch_size):
            end = min(start + mc_batch_size, mc_num)
            chunk_attn = attn_expand[start:end] if attn_expand is not None else None
            logits = model(
                input_ids=all_noised[start:end], attention_mask=chunk_attn
            ).logits
            logits = _postprocess_logits(logits, right_shift_logits)
            token_nll = F.cross_entropy(
                logits.transpose(1, 2), target_ids[start:end], reduction="none"
            )
            for j in range(end - start):
                masked_ce_sum = token_nll[j][mask_flags[start + j]].sum().item()
                total_score += masked_ce_sum / ls[start + j]

        avg_loss = total_score / mc_num
        results.append({"prob": float(torch.exp(torch.tensor(-avg_loss))), "avg_loss": avg_loss})
    return results


def _extract_prompts(input_ids, labels, attention_mask):
    """Prompt tokens for generation.

    Uses ``predict_with_generate`` batches, where ``input_ids`` already holds
    just the question/prompt (right-padded) and ``labels`` separately holds the
    ground-truth answer — so the two need not share a sequence length. The
    prompt is therefore the attended prefix of ``input_ids``, not the
    ``labels == -100`` span.
    """
    prompts = []
    for i in range(input_ids.shape[0]):
        if attention_mask is not None:
            prompt_mask = attention_mask[i].bool()
        else:
            prompt_mask = torch.ones_like(input_ids[i], dtype=torch.bool)
        prompts.append(input_ids[i][prompt_mask])
    return prompts


@torch.no_grad()
def eval_text_similarity_diffusion(
    model,
    tokenizer,
    batch,
    generation_args,
    scheduler_name="LinearAlphaScheduler",
    right_shift_logits=False,
):
    """ROUGE between iterative-denoising generations and ground-truth answers."""
    from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig

    device = model.device
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    input_texts = tokenizer.batch_decode(
        [ids[ids != tokenizer.pad_token_id] if tokenizer.pad_token_id is not None else ids
         for ids in input_ids],
        skip_special_tokens=True,
    )
    gt_tokens = [label[label != IGNORE_INDEX] for label in labels]
    ground_truths = tokenizer.batch_decode(gt_tokens, skip_special_tokens=True)

    prompts = _extract_prompts(input_ids, labels, attention_mask)

    gen_args = OmegaConf.to_container(generation_args, resolve=True)
    cfg = MDLMSamplerConfig(
        max_new_tokens=gen_args.get("max_new_tokens", 128),
        steps=gen_args.get("steps", 256),
        block_size=gen_args.get("block_size", 128),
        temperature=gen_args.get("temperature", 0.0) or 0.0,
        remasking=gen_args.get("remasking", "low_confidence"),
        right_shift_logits=right_shift_logits,
    )
    sampler = MDLMSampler(model=model, tokenizer=tokenizer, scheduler=_build_scheduler(scheduler_name))
    sequences = sampler.sample(prompts, config=cfg)

    # ROUGE over prompt+generation vs prompt+ground-truth (full sequence,
    # prompt included) rather than the generated answer alone.
    gen_texts = []
    for i in range(len(prompts)):
        gen_texts.append(
            tokenizer.decode(sequences[i], skip_special_tokens=True).strip()
        )
    references = [f"{inp} {gt}".strip() for inp, gt in zip(input_texts, ground_truths)]

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = []
    for gen, gt, inp in zip(gen_texts, references, input_texts):
        rs = scorer.score(gt, gen)
        scores.append(
            {
                "rouge1_recall": rs["rouge1"].recall,
                "rougeL_f1": rs["rougeL"].fmeasure,
                "rougeL_recall": rs["rougeL"].recall,
                "input": inp,
                "ground_truth": gt,
                "generation": gen,
            }
        )
    return scores
