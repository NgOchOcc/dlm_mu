"""Smoke test for DiffusionSBRU's embedding-space geometry (docs/idea.tex).

Verifies the pure-math helpers of ``src/trainer/dlm/unlearn/sbru.py`` against
hand-computed values without loading a model: an instance is built with
``object.__new__`` and only the attributes the helpers read are bound.

  _soft_emb  : one-hot p -> E_v exactly; uniform p -> mean_v E_v (the centroid).
  _kernel    : k(a,a)==1; k in (0,1]; strictly decreasing in ||a-b||; sigma->inf -> 1.
  gradient   : autograd d/du k_sigma(u,c) == -(1/sigma^2) k_sigma (u-c)  (SGF field).
  _ball_centre: ball_k=0 -> gold embedding; ball_k=1 -> the gold token itself (self is
               its own nearest neighbour), so centre == gold embedding.
  L_rep, L_anc: L_rep in (0,1], L_anc in [-1,0) on random masked input via _reduce.
  centre_mode : gold vs ref_soft produce different centres.

    python scripts/dlm/tofu/sbru_smoke_test.py
"""

import math

import torch

from trainer.dlm.unlearn.sbru import DiffusionSBRU


def make_sbru(sigma=1.0, ball_k=0, pad_id=0):
    t = object.__new__(DiffusionSBRU)
    t.sigma = float(sigma)
    t.ball_k = int(ball_k)
    t.pad_id = int(pad_id)
    return t


def main():
    torch.manual_seed(0)
    b, l, V, d = 2, 4, 16, 8
    t = make_sbru(sigma=1.0)
    E = torch.randn(V, d, dtype=torch.float64)

    # --- _soft_emb: one-hot p -> E_v; uniform p -> mean_v E_v --------------------
    # One-hot at token 5 (as log-probs): e_hat == E[5].
    log_p = torch.full((1, 1, V), -1e30, dtype=torch.float64)
    log_p[0, 0, 5] = 0.0
    e = t._soft_emb(log_p, E)
    assert torch.allclose(e[0, 0], E[5], atol=1e-6), (e[0, 0] - E[5]).abs().max()
    # Uniform -> centroid.
    log_u = torch.log(torch.full((1, 1, V), 1.0 / V, dtype=torch.float64))
    eu = t._soft_emb(log_u, E)
    assert torch.allclose(eu[0, 0], E.mean(0), atol=1e-6), (eu[0, 0] - E.mean(0)).abs().max()
    print("[ok] _soft_emb: one-hot -> E_v; uniform -> mean_v E_v")

    # --- _kernel: k(a,a)==1, in (0,1], strictly decreasing in distance ----------
    a = torch.randn(3, 2, d, dtype=torch.float64)
    assert torch.allclose(t._kernel(a, a), torch.ones(3, 2, dtype=torch.float64), atol=1e-9)
    c = torch.zeros_like(a)
    near = t._kernel(a, c)
    far = t._kernel(a * 5.0, c)  # farther from c -> smaller kernel
    assert bool(((near > 0) & (near <= 1)).all())
    assert bool((far < near).all()), (far, near)
    # sigma -> inf collapses the kernel to 1.
    t_big = make_sbru(sigma=1e6)
    assert torch.allclose(t_big._kernel(a, c), torch.ones(3, 2, dtype=torch.float64), atol=1e-6)
    print("[ok] _kernel: k(a,a)=1; in (0,1]; decreasing in ||a-b||; sigma->inf -> 1")

    # --- gradient field: d/du k(u,c) == -(1/sigma^2) k (u-c) ---------------------
    u = torch.randn(2, 3, d, dtype=torch.float64, requires_grad=True)
    cc = torch.randn(2, 3, d, dtype=torch.float64)
    k = t._kernel(u, cc)
    k.sum().backward()
    with torch.no_grad():
        analytic = -(1.0 / t.sigma**2) * t._kernel(u, cc).unsqueeze(-1) * (u - cc)
    assert torch.allclose(u.grad, analytic, atol=1e-8), (u.grad - analytic).abs().max()
    print("[ok] gradient: autograd matches -(1/sigma^2) k_sigma (u-c) (repulsive field)")

    # --- _ball_centre: ball_k=0 -> gold; ball_k=1 -> gold (self is own 1-NN) -----
    gold_emb = E[torch.randint(0, V, (b, l))]
    t0 = make_sbru(ball_k=0)
    assert torch.allclose(t0._ball_centre(gold_emb, E), gold_emb)
    # ball_k=1: nearest neighbour of a real token embedding under cosine is itself.
    ids = torch.randint(0, V, (b, l))
    t1 = make_sbru(ball_k=1)
    cb = t1._ball_centre(E[ids], E)
    assert torch.allclose(cb, E[ids].to(cb.dtype), atol=1e-6), (cb - E[ids].to(cb.dtype)).abs().max()
    print("[ok] _ball_centre: k=0 -> gold; k=1 -> gold token (self is its own NN)")

    # --- L_rep in (0,1], L_anc in [-1,0) via _reduce ----------------------------
    log_p = torch.log_softmax(torch.randn(b, l, V, dtype=torch.float64), dim=-1)
    e_hat = t._soft_emb(log_p, E)
    c = E[torch.randint(0, V, (b, l))]
    aanc = E[torch.randint(0, V, (b, l))]
    mask = torch.zeros(b, l, dtype=torch.bool)
    mask[:, 1:3] = True  # some masked positions
    w = torch.ones(b, l, dtype=torch.float64)
    L_rep = t._reduce(t._kernel(e_hat, c), mask, w)
    L_anc = -t._reduce(t._kernel(e_hat, aanc), mask, w)
    assert 0.0 < float(L_rep) <= 1.0, float(L_rep)
    assert -1.0 <= float(L_anc) < 0.0, float(L_anc)
    print(f"[ok] L_rep={float(L_rep):.4g} in (0,1];  L_anc={float(L_anc):.4g} in [-1,0)")

    # --- centre_mode: gold vs ref_soft differ -----------------------------------
    gold_centre = E[ids]
    ref_soft_centre = t._soft_emb(log_p, E)  # a soft (non one-hot) centroid
    assert not torch.allclose(gold_centre, ref_soft_centre, atol=1e-3)
    print("[ok] centre_mode: gold (E[y_f]) and ref_soft (centroid) differ")

    # --- __init__ rejects bad scalars before it needs a model -------------------
    for kwargs in ({"sigma": 0.0}, {"sigma": -1.0}, {"centre_mode": "bad"}, {"ball_k": -1}):
        try:
            DiffusionSBRU(**kwargs)
        except ValueError:
            pass
        except TypeError:
            raise AssertionError(f"{kwargs} bypassed the guard (reached super().__init__)")
        else:
            raise AssertionError(f"{kwargs} should have been rejected")
    print("[ok] __init__ rejects sigma<=0 / bad centre_mode / ball_k<0")

    print("\nAll DiffusionSBRU geometry sanity checks passed.")


if __name__ == "__main__":
    main()
