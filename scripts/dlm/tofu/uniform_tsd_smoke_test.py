"""Smoke test for DiffusionUTSD's uniform-target forget objective.

Verifies the ``_forget_kl`` override in ``src/trainer/dlm/unlearn/uniform_tsd.py``
against hand-computed KL values without loading a model: an instance is built with
``object.__new__`` and only the attributes ``_forget_kl`` reads are bound. Checks

  mix_beta=1.0 -> target is uniform u=1/V:
      gold_prob_target == 1/V everywhere; kl_f == KL(u||p_theta) >= 0; finite.
  mix_beta=0.0 -> reduces EXACTLY to DiffusionSimpleTSD (edited-teacher target q).
  mix_beta in (0,1) -> probability-space mix m = (1-beta) q + beta u, kl_f matches
      the direct KL(m||p_theta).

    python scripts/dlm/tofu/uniform_tsd_smoke_test.py
"""

import math

import torch

from trainer.dlm.unlearn.simple_tsd import DiffusionSimpleTSD
from trainer.dlm.unlearn.uniform_tsd import DiffusionUTSD


def make_utsd(mix_beta, demotion="minimal", belief_top_k=5):
    t = object.__new__(DiffusionUTSD)
    t.mix_beta = float(mix_beta)
    t.demotion = demotion
    t.belief_top_k = int(belief_top_k)
    return t


def make_base(demotion="minimal", belief_top_k=5):
    t = object.__new__(DiffusionSimpleTSD)
    t.demotion = demotion
    t.belief_top_k = int(belief_top_k)
    return t


def rand_log_probs(b, l, V, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(b, l, V, generator=g, dtype=torch.float64)
    return torch.log_softmax(logits, dim=-1)


def main():
    torch.manual_seed(0)
    b, l, V = 2, 4, 32
    log_p = rand_log_probs(b, l, V, seed=1)   # student
    log_p0 = rand_log_probs(b, l, V, seed=2)  # teacher
    input_ids = torch.randint(0, V, (b, l), generator=torch.Generator().manual_seed(3))
    ctx = {"input_ids": input_ids}
    gold = input_ids.unsqueeze(-1)

    # --- mix_beta = 1.0: pure uniform target ------------------------------------
    t1 = make_utsd(mix_beta=1.0)
    kl_f, removed = t1._forget_kl(log_p, log_p0, None, ctx)
    # gold target is exactly 1/V everywhere.
    gt = t1._forget_diag["gold_prob_target"]
    assert torch.allclose(gt, torch.full_like(gt, 1.0 / V)), gt
    # kl_f == KL(u||p) = sum_v (1/V)(log(1/V) - log p) computed directly.
    u = torch.full_like(log_p, 1.0 / V)
    kl_ref = (u * (math.log(1.0 / V) - log_p)).sum(-1)
    assert torch.allclose(kl_f, kl_ref, atol=1e-9), (kl_f - kl_ref).abs().max()
    assert bool((kl_f >= -1e-9).all()), kl_f.min()  # forward KL is non-negative
    assert torch.isfinite(kl_f).all()
    # removed = teacher gold mass - 1/V, floored at 0.
    p0_gold = log_p0.gather(-1, gold).squeeze(-1).exp()
    assert torch.allclose(removed, (p0_gold - 1.0 / V).clamp_min(0.0)), removed
    print(f"[ok] mix_beta=1.0: gold_prob_target==1/V ({1.0/V:.4g}); kl_f==KL(u||p)>=0, finite")

    # --- mix_beta = 0.0: must reduce EXACTLY to DiffusionSimpleTSD ---------------
    for demotion, btk in (("minimal", 5), ("belief", 4)):
        t0 = make_utsd(mix_beta=0.0, demotion=demotion, belief_top_k=btk)
        base = make_base(demotion=demotion, belief_top_k=btk)
        base._forget_diag = None
        kl0, rem0 = t0._forget_kl(log_p, log_p0, None, ctx)
        klb, remb = base._forget_kl(log_p, log_p0, None, ctx)
        assert torch.allclose(kl0, klb, atol=1e-9), (demotion, (kl0 - klb).abs().max())
        gt0 = t0._forget_diag["gold_prob_target"]
        gtb = base._forget_diag["gold_prob_target"]
        assert torch.allclose(gt0, gtb, atol=1e-9), (demotion, (gt0 - gtb).abs().max())
        print(f"[ok] mix_beta=0.0 ({demotion}): kl_f and gold_prob_target match SimpleTSD exactly")

    # --- mix_beta in (0,1): kl_f == direct KL(m||p) with m=(1-b)q+b*u -----------
    beta = 0.3
    tm = make_utsd(mix_beta=beta, demotion="minimal")
    kl_m, _ = tm._forget_kl(log_p, log_p0, None, ctx)
    demote = tm._demote_mask(log_p0, input_ids)
    log_q = log_p0.masked_fill(demote, float("-inf"))
    log_q = log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)
    q = log_q.exp()
    m = (1.0 - beta) * q + beta * (1.0 / V)
    kl_m_ref = (m * (m.clamp_min(1e-12).log() - log_p)).sum(-1)
    assert torch.allclose(kl_m, kl_m_ref, atol=1e-9), (kl_m - kl_m_ref).abs().max()
    assert bool((kl_m >= -1e-9).all()), kl_m.min()
    # target mass on the demoted set is beta*u (uniform floor), not 0 as in base.
    gt_m = tm._forget_diag["gold_prob_target"]
    assert torch.allclose(gt_m[demote.gather(-1, gold).squeeze(-1)],
                          torch.tensor(beta / V, dtype=torch.float64), atol=1e-9)
    print(f"[ok] mix_beta={beta}: kl_f==KL(m||p) with m=(1-b)q+b*u; gold floor=b/V")

    # --- validation: __init__ rejects mix_beta out of [0,1] (guard runs before the
    # super().__init__ that would need a real model, so we can catch it in isolation).
    for bad in (-0.1, 1.5):
        try:
            DiffusionUTSD(mix_beta=bad)
        except ValueError:
            pass
        except TypeError:
            # super().__init__() reached with no model -> the mix_beta guard already
            # passed, which for an out-of-range value is a failure of the guard.
            raise AssertionError(f"mix_beta={bad} was not rejected before super().__init__")
        else:
            raise AssertionError(f"mix_beta={bad} should have been rejected")
    print("[ok] mix_beta outside [0,1] rejected")

    print("\nAll DiffusionUTSD uniform-target sanity checks passed.")


if __name__ == "__main__":
    main()
