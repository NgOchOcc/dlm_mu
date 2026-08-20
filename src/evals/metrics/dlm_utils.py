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


@torch.no_grad()
def tokenwise_vocab_logprobs_diffusion(
    model,
    batch,
    right_shift_logits=False,
    mask_token_id=None,
    return_labels=False,
):
    """DLM counterpart of ``tokenwise_vocab_logprobs`` for greedy-reconstruction
    metrics (exact memorization, extraction strength).

    A masked diffusion model predicts the token *at* a masked position rather
    than the *next* token, so there is no teacher-forced shift. We fully mask
    every answer position (``labels != IGNORE_INDEX``) in a single forward pass
    — the one-shot reconstruction regime — and read off the per-position vocab
    log-probabilities at those positions. This yields, per example, the model's
    argmax reconstruction of each answer token, matching the semantics of the
    autoregressive helper (distribution over the true target token at each
    labelled position).

    Returns:
        log_probs_batch (List[Tensor]): per example, shape (N, V) over the N
            answer positions.
        labels_batch (List[Tensor]): per example, the gold answer tokens (N,).
            Returned only when ``return_labels`` is True.
    """
    assert mask_token_id is not None, "mask_token_id must be provided for DLM EM/ES"
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
            bos_token_id=model.config.bos_token_id if hasattr(model, "config") else None,
            label_pad_token_id=IGNORE_INDEX,
        )
        input_ids, labels = shifted["input_ids"], shifted["labels"]
        attention_mask = shifted.get("attention_mask", attention_mask)

    maskable = labels != IGNORE_INDEX
    noised = input_ids.clone()
    noised[maskable] = mask_token_id

    logits = model(input_ids=noised, attention_mask=attention_mask).logits
    logits = _postprocess_logits(logits, right_shift_logits)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    V = log_probs.shape[-1]
    log_probs_batch = []
    labels_batch = []
    for i in range(input_ids.shape[0]):
        answer_idx = maskable[i].nonzero(as_tuple=True)[0]
        if answer_idx.numel() == 0:
            log_probs_batch.append(torch.zeros(0, V, device=device))
            labels_batch.append(torch.tensor([], device=device, dtype=torch.long))
            continue
        log_probs_batch.append(log_probs[i, answer_idx])
        labels_batch.append(input_ids[i, answer_idx])

    return (log_probs_batch, labels_batch) if return_labels else log_probs_batch


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


_STOP_IDS_CACHE = {}


def _stop_cache_key(tokenizer):
    """Stable-across-GC cache key: tokenizer's name_or_path when available, else id()."""
    return getattr(tokenizer, "name_or_path", None) or id(tokenizer)


def derive_stop_token_ids(tokenizer):
    """Token ids marking the end of an assistant turn.

    Masked diffusion fills a fixed-length block, so the model emits its turn
    terminator at the true answer end and then keeps denoising the leftover
    positions (repeating the answer). The terminator is model-specific and, on
    both Dream and LLaDA, is NOT ``eos_token_id`` (that is ``<|endoftext|>``) and
    is NOT even flagged in ``all_special_ids`` — which is why ``skip_special_tokens``
    leaves it (and the repeat after it) in the decoded string. So derive it from
    the chat template itself: render two assistant turns with different content
    and take the first token of their common suffix — the token the template
    appends right after the answer (``<|im_end|>`` for Dream, ``<|eot_id|>`` for
    LLaDA). Union with ``eos_token_id`` gives a model-agnostic stop set.

    Returned set includes both the turn terminator and ``eos_token_id`` — callers
    that need only the terminator (e.g. to trim training labels while KEEPING the
    terminator itself as a supervised target) should exclude ``eos_token_id``.
    """
    key = _stop_cache_key(tokenizer)
    if key in _STOP_IDS_CACHE:
        return _STOP_IDS_CACHE[key]
    stops = set()
    if tokenizer.eos_token_id is not None:
        stops.add(tokenizer.eos_token_id)
    try:
        r1 = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "AAAA"}],
            tokenize=True,
            add_generation_prompt=False,
        )
        r2 = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "BBBBBBBB"}],
            tokenize=True,
            add_generation_prompt=False,
        )
        k = 0
        while k < min(len(r1), len(r2)) and r1[-1 - k] == r2[-1 - k]:
            k += 1
        if k:  # first token of the common suffix is the terminator after content
            stops.add(r1[len(r1) - k])
    except Exception:
        pass  # eos_token_id alone is the fallback
    _STOP_IDS_CACHE[key] = stops
    return stops


# Back-compat alias for the private name used by the previous revision.
_derive_stop_token_ids = derive_stop_token_ids


@torch.no_grad()
def generate_answers_diffusion(
    model,
    tokenizer,
    batch,
    generation_args,
    scheduler_name="LinearAlphaScheduler",
    right_shift_logits=False,
):
    """Iterative-denoising generations paired with prompts and ground truths.

    Returns a list of per-example ``{"input", "ground_truth", "generation"}``
    dicts. Shared by ``eval_text_similarity_diffusion`` (ROUGE) and the LLM-judge
    metrics so the expensive ``MDLMSampler`` decoding is defined in one place.
    """
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
    # Stop set (turn terminator ∪ eos) used to trim BOTH the ground truth and the
    # generation to the answer content. Without this the ground truth carries the
    # chat template's post-answer scaffolding into ROUGE / the judge -- Dream's
    # supervised span ends "answer<|im_end|>\n<|endoftext|>" (adds a literal
    # "<|im_end|>" to the decoded string since it isn't in all_special_ids), and
    # LLaDA's ends "answer<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<|endoftext|>"
    # (adds the word "assistant" to the decoded string). Both would nudge ROUGE
    # in different directions run-to-run; trimming once here fixes GT and
    # generation with the same logic.
    stop_ids = derive_stop_token_ids(tokenizer)
    stop_tensor = (
        torch.tensor(sorted(stop_ids), device=device) if stop_ids else None
    )

    def _trim_at_stop(ids_1d):
        """Return ids up to (excluding) the first stop token, or the input unchanged."""
        if stop_tensor is None:
            return ids_1d
        hit = torch.isin(ids_1d, stop_tensor).nonzero(as_tuple=True)[0]
        return ids_1d[: hit[0]] if hit.numel() > 0 else ids_1d

    # In eval mode (predict_with_generate=True) `input_ids` holds only the
    # prompt while `labels` holds the entire conversation with NO IGNORE_INDEX
    # over the prompt span, so we slice by prompt length before trimming --
    # trimming the raw `labels != IGNORE_INDEX` would cut at the first
    # terminator inside the prompt itself (e.g. the "<|im_end|>" that closes
    # Dream's system prompt) and return system-prompt text as the "answer".
    pad_id = tokenizer.pad_token_id
    prompt_lens = [
        int((ids != pad_id).sum()) if pad_id is not None else int(ids.numel())
        for ids in input_ids
    ]
    gt_tokens = []
    for i, label in enumerate(labels):
        non_pad = label[label != IGNORE_INDEX]
        gt_tokens.append(_trim_at_stop(non_pad[prompt_lens[i] :]))
    ground_truths = tokenizer.batch_decode(gt_tokens, skip_special_tokens=True)
    ground_truths = [gt.strip() for gt in ground_truths]

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

    # The sampler returns [prompt | generated | pad], so slice off each prompt to
    # keep the response only. Then cut at the first turn terminator or EOS so
    # ROUGE / the judge see just the intended answer, not answer+repeat.
    max_new_tokens = cfg.max_new_tokens
    results = []
    for i in range(len(prompts)):
        gen_ids = sequences[i][prompts[i].shape[0] : prompts[i].shape[0] + max_new_tokens]
        gen_ids = _trim_at_stop(gen_ids)
        results.append(
            {
                "input": input_texts[i],
                "ground_truth": ground_truths[i],
                "generation": tokenizer.decode(gen_ids, skip_special_tokens=True).strip(),
            }
        )
    return results


def eval_text_similarity_diffusion(
    model,
    tokenizer,
    batch,
    generation_args,
    scheduler_name="LinearAlphaScheduler",
    right_shift_logits=False,
):
    """ROUGE between iterative-denoising generations and ground-truth answers."""
    generations = generate_answers_diffusion(
        model,
        tokenizer,
        batch,
        generation_args,
        scheduler_name=scheduler_name,
        right_shift_logits=right_shift_logits,
    )

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = []
    for item in generations:
        rs = scorer.score(item["ground_truth"], item["generation"])
        scores.append(
            {
                "rouge1_recall": rs["rouge1"].recall,
                "rougeL_f1": rs["rougeL"].fmeasure,
                "rougeL_recall": rs["rougeL"].recall,
                "input": item["input"],
                "ground_truth": item["ground_truth"],
                "generation": item["generation"],
            }
        )
    return scores
