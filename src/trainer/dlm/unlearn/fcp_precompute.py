"""Offline counterfactual atom-bank and cap construction for FCP-MDLM.

Implements Algorithm 1 of docs/FCP_MDLM_training_algorithm.tex. Given the
finetuned model ``theta0`` (has seen the forget facts) and a pre-exposure model
``theta_b`` (has not), we build, per forget fact:

  * the fact-bearing span ``K`` (positions whose target evidence *increased*
    after forget exposure and are not copied from the prompt), and
  * per-position probability caps ``rho_a`` for the recall and infill states.

The result is cached to disk; training (``fcp.py``) then needs only the cached
caps + states, never ``theta_b``.

``FrameHelper`` mirrors the three tiny frame methods of
``DiffusionFinetuneTrainer`` (``_preprocess_inputs``, ``_postprocess_outputs``,
``_mask_tokens``) so cap construction happens in exactly the same
post-preprocess frame the trainer uses. Keep the two in sync.
"""

import torch

from trainer.dlm.unlearn.fcp_common import (
    target_vs_rest_logodds,
    target_prob,
    select_span_K,
    apply_strict_cap,
)


class FrameHelper:
    """Post-preprocess frame handling shared with DiffusionFinetuneTrainer."""

    def __init__(self, tokenizer, right_shift_logits):
        self.processing_class = tokenizer
        self.right_shift_logits = right_shift_logits

    def _preprocess_inputs(self, inputs):
        # Mirrors trainer/dlm/base.py::_preprocess_inputs.
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
        # Mirrors trainer/dlm/base.py::_postprocess_outputs.
        if self.right_shift_logits:
            logits = outputs.logits
            outputs.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return outputs

    def _mask_tokens(self, input_ids, positions):
        return torch.where(positions, self.processing_class.mask_token_id, input_ids)


@torch.no_grad()
def _frozen_logits(model, helper, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    outputs = helper._postprocess_outputs(outputs)
    return outputs.logits[0].float()  # (l, V) for the single sequence


def build_fact_cache(
    theta0,
    theta_b,
    forget_dataset,
    tokenizer,
    right_shift_logits,
    kappa=0.9,
    strict_epsilon=None,
    device="cuda",
    log_every=50,
):
    """Run Algorithm 1 over every forget fact; return a dict keyed by fact index.

    Each entry stores the post-preprocess ``input_ids``/``attention_mask``, the
    full ``answer_mask`` (to rebuild the recall state), the selected span
    ``K_pos`` with gold tokens ``y_K``, and recall/infill caps.
    """
    helper = FrameHelper(tokenizer, right_shift_logits)
    theta0.eval()
    theta_b.eval()
    cache = {}
    n_zero_evidence = 0
    k_sizes = []

    for j in range(len(forget_dataset)):
        item = forget_dataset[j]
        fact_id = int(item["index"])
        input_ids = torch.as_tensor(item["input_ids"], device=device).unsqueeze(0)
        labels = torch.as_tensor(item["labels"], device=device).unsqueeze(0)
        attn = item.get("attention_mask", None)
        attn = (
            torch.as_tensor(attn, device=device).unsqueeze(0)
            if attn is not None
            else torch.ones_like(input_ids)
        )

        inputs = helper._preprocess_inputs(
            {"input_ids": input_ids, "labels": labels, "attention_mask": attn}
        )
        input_ids, labels = inputs["input_ids"], inputs["labels"]
        attn = inputs["attention_mask"]

        answer_mask = labels != -100  # (1, l)
        answer_pos = answer_mask[0].nonzero(as_tuple=True)[0]  # (n,)
        if answer_pos.numel() == 0:
            continue
        y = input_ids[0, answer_pos]  # (n,)

        prompt_mask = (~answer_mask[0]) & attn[0].bool()
        prompt_tokens = set(input_ids[0, prompt_mask].tolist())
        prompt_copied = torch.tensor(
            [int(tok) in prompt_tokens for tok in y.tolist()],
            device=device,
            dtype=torch.bool,
        )

        # --- recall state: all answer tokens masked -------------------------
        recall_ids = helper._mask_tokens(input_ids, answer_mask)
        logits0 = _frozen_logits(theta0, helper, recall_ids, attn)
        logitsb = _frozen_logits(theta_b, helper, recall_ids, attn)
        m0 = target_vs_rest_logodds(logits0, answer_pos, y)
        mb = target_vs_rest_logodds(logitsb, answer_pos, y)
        d = (m0 - mb).clamp_min(0.0)  # acquired evidence d(a)
        rho_recall_all = target_prob(logitsb, answer_pos, y)  # cap on recall state

        K_local = select_span_K(d, prompt_copied, kappa)  # into answer_pos
        K_pos = answer_pos[K_local]
        y_K = y[K_local]
        rho_recall = rho_recall_all[K_local]

        # --- infill state: only span K masked -------------------------------
        infill_mask = torch.zeros_like(answer_mask)
        infill_mask[0, K_pos] = True
        infill_ids = helper._mask_tokens(input_ids, infill_mask)
        logitsb_inf = _frozen_logits(theta_b, helper, infill_ids, attn)
        rho_infill = target_prob(logitsb_inf, K_pos, y_K)

        k = int(K_pos.numel())
        rho_recall = apply_strict_cap(rho_recall, strict_epsilon, k)
        rho_infill = apply_strict_cap(rho_infill, strict_epsilon, k)

        zero_ev = bool(d.sum().item() <= 0)
        n_zero_evidence += int(zero_ev)
        k_sizes.append(k)

        cache[fact_id] = {
            "input_ids": input_ids[0].cpu(),
            "attention_mask": attn[0].cpu(),
            "answer_mask": answer_mask[0].cpu(),
            "K_pos": K_pos.cpu(),
            "y_K": y_K.cpu(),
            "rho_recall": rho_recall.cpu(),
            "rho_infill": rho_infill.cpu(),
            "zero_evidence": zero_ev,
        }

        if log_every and (j + 1) % log_every == 0:
            print(
                f"[fcp-precompute] {j + 1}/{len(forget_dataset)} facts | "
                f"mean|K|={sum(k_sizes) / len(k_sizes):.2f} | "
                f"zero-evidence={n_zero_evidence}",
                flush=True,
            )

    meta = {
        "kappa": kappa,
        "strict_epsilon": strict_epsilon,
        "right_shift_logits": right_shift_logits,
        "n_facts": len(cache),
        "n_zero_evidence": n_zero_evidence,
        "mean_K": (sum(k_sizes) / len(k_sizes)) if k_sizes else 0.0,
    }
    return {"facts": cache, "meta": meta}
