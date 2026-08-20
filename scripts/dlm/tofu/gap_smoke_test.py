"""Smoke test for DiffusionGAP's projected teacher target.

Verifies the six sanity checks in docs/gap_dlm_algorithm.tex Sec. "Sanity Checks"
against the real class methods, without loading a model: the projection is pure
math over teacher logits, so we bind only the attributes it reads onto an
instance built with ``__new__``.

    python scripts/dlm/tofu/gap_smoke_test.py
"""

import torch

from trainer.dlm.unlearn.gap import DiffusionGAP


def make_gap(lam=1.0, copy_ngram=2, vocab_size=8, epsilon=1e-8):
    gap = object.__new__(DiffusionGAP)
    gap.lam = lam
    gap.copy_ngram = copy_ngram
    gap.vocab_size = vocab_size
    gap.gap_epsilon = epsilon
    return gap


def main():
    torch.manual_seed(0)
    V = 8
    gap = make_gap(vocab_size=V)

    # One sequence: 2 prompt positions + 3 answer positions. maskable_mask marks
    # the answer; input_ids holds the gold tokens. Prompt = [1, 2]; answer = [2, 3, 4]
    # so the answer bigram (2,3) is NOT copied but the single token 2 is echoed.
    input_ids = torch.tensor([[1, 2, 2, 3, 4]])
    maskable = torch.tensor([[False, False, True, True, True]])
    attn = torch.ones_like(input_ids)
    logits_c = torch.randn(1, 5, V)
    logits_u = torch.randn(1, 5, V)
    art = {
        "input_ids": input_ids,
        "maskable_mask": maskable,
        "attention_mask": attn,
        "noised_input_ids": input_ids,
    }

    # Check 5 + 4 + 6 on random teachers (lam=1, copy_ngram=2 -> no copy match here).
    log_q, q = gap._projected_target(logits_c, logits_u, art)
    sums = q.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), sums
    assert torch.allclose(log_q.exp(), q, atol=1e-5)
    assert not q.requires_grad and not log_q.requires_grad, "q* must be detached"
    # For v in S+ (g>0): q*(v) <= u(v).
    log_pc = gap._log_probs(logits_c)
    log_pu = gap._log_probs(logits_u)
    g = log_pc - log_pu
    positive = g > 0
    true_ids = input_ids.clamp(max=V - 1).unsqueeze(-1)
    g_true = g.gather(-1, true_ids).squeeze(-1)
    kappa = gap._copy_safe_indicator(art)
    alpha = (gap.lam * kappa * (1 - torch.exp(-g_true.clamp_min(0.0)))).unsqueeze(-1)
    u = ((1 - alpha) * log_pc + alpha * log_pu).exp()
    assert (q[positive] <= u[positive] + 1e-5).all(), "cap q*(v) <= u(v) violated on S+"
    print("[ok] check 4 (q*<=u on S+), 5 (sum=1), 6 (detached)")

    # Check 1: p^c(y_i) == p^u(y_i) at every token -> alpha == 0 -> q* == p^c.
    same = torch.randn(1, 5, V)
    log_q2, q2 = gap._projected_target(same, same, art)
    pc = gap._log_probs(same).exp()
    assert torch.allclose(q2, pc, atol=1e-5), "p^c==p^u must give q*=p^c"
    print("[ok] check 1 (p^c=p^u -> alpha=0) and 3 (alpha=0 -> q*=p^c)")

    # Check 2: copy_ngram=1 marks the echoed token 2 as copy-safe (kappa=0). At a
    # copy-safe position q* must equal p^c even when the prompt amplifies tokens.
    gap1 = make_gap(lam=1.0, copy_ngram=1, vocab_size=V)
    kappa1 = gap1._copy_safe_indicator(art)
    # answer positions are 2,3,4 with tokens 2,3,4; only token 2 is in the prompt.
    assert kappa1[0, 2] == 0.0 and kappa1[0, 3] == 1.0 and kappa1[0, 4] == 1.0, kappa1
    log_qk, qk = gap1._projected_target(logits_c, logits_u, art)
    pc_k = gap1._log_probs(logits_c).exp()
    assert torch.allclose(qk[0, 2], pc_k[0, 2], atol=1e-5), "kappa=0 must give q*=p^c"
    print("[ok] check 2 (kappa=0 -> q*=p^c) and copy-span matching")

    print("\nAll GAP-DLM sanity checks passed.")


if __name__ == "__main__":
    main()
