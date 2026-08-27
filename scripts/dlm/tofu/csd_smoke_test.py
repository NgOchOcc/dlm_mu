"""Smoke test for DiffusionCSD's concept-subspace erasure loss.

Exercises the novel core of ``src/trainer/dlm/unlearn/concept_subspace.py`` --
``_concept_repr_loss`` -- with hand-built orthonormal bases and hidden states (no model
load), verifying the erase / preserve decomposition over an r-dim subspace, the validity
mask, the ``removed`` diagnostic, and that ``subspace_rank == 1`` reduces to the original
rank-1 objective. The subspace *building* (SVD over teacher forwards) needs a model and is
verified by the GPU rank-sweep in ``simple_tsd_viz_target.py`` (VIZ_MODE=subspace).

    PYTHONPATH=src python scripts/dlm/tofu/csd_smoke_test.py
"""

import torch

from trainer.dlm.unlearn.concept_subspace import DiffusionCSD


def make_csd(complement_weight=1.0, min_dir_norm=1e-3):
    """A bare DiffusionCSD carrying only what ``_concept_repr_loss`` reads."""
    t = object.__new__(DiffusionCSD)
    t.complement_weight = float(complement_weight)
    t.min_dir_norm = float(min_dir_norm)
    return t


def _basis(rows, b=1):
    """Orthonormal basis tensor [b, r, d] from a list of (already orthonormal) row lists."""
    P = torch.tensor(rows, dtype=torch.float32)
    return P.unsqueeze(0).expand(b, *P.shape).contiguous()


def main():
    torch.manual_seed(0)
    d = 6
    # pos0: answer, with-question teacher has concept-subspace content (2,3) along e0,e1 and
    # off-subspace content 5 along e4. pos1: not an answer position.
    h0 = torch.zeros(1, 2, d)
    h0[0, 0] = torch.tensor([2.0, 3.0, 0.0, 0.0, 5.0, 0.0])
    h0_base = torch.zeros(1, 2, d)          # promptless teacher (decoupled for the unit test)
    answer_mask = torch.tensor([[True, False]])
    P2 = _basis([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])  # rank-2 subspace span{e0,e1}
    rank2 = torch.tensor([2.0])

    t = make_csd(complement_weight=1.0)
    t0 = make_csd(complement_weight=0.0)

    # --- Erase term isolation (cw=0), h_s = h0 + off-subspace perturbation along e2.
    # erase = ||P(h_s - h0_base)||^2 = 2^2 + 3^2 = 13 (subspace part unchanged); preserve off.
    h_pert = h0.clone(); h_pert[0, 0, 2] += 7.0
    kl, removed = t0._concept_repr_loss(h_pert, h0, h0_base, answer_mask, P2, rank2)
    assert kl.shape == (1, 2) and removed.shape == (1, 2), (kl.shape, removed.shape)
    assert torch.isfinite(kl).all()
    assert abs(float(kl[0, 0]) - 13.0) < 1e-4, kl
    assert float(kl[0, 1]) == 0.0, kl                         # non-answer -> 0
    assert abs(float(removed[0, 0]) - 13.0 ** 0.5) < 1e-4, removed  # ||P(h0-h0_base)||
    print("[ok] erase = ||P(h_s - h0_base)||^2 over the subspace; non-answer zeroed")

    # --- Preserve term (cw=1): the same e2 perturbation is off-subspace -> preserve=49/d.
    kl_p, _ = t._concept_repr_loss(h_pert, h0, h0_base, answer_mask, P2, rank2)
    assert abs(float(kl_p[0, 0]) - (13.0 + 49.0 / d)) < 1e-4, kl_p
    print("[ok] off-subspace drift adds complement_weight * ||comp||^2 / d only")

    # --- Ideal target: subspace component driven to promptless (0 here) AND off-subspace
    # kept at h0 -> both terms vanish. h_s = h0 with its subspace part removed.
    h_star = h0.clone(); h_star[0, 0, 0] = 0.0; h_star[0, 0, 1] = 0.0
    kl_s, _ = t._concept_repr_loss(h_star, h0, h0_base, answer_mask, P2, rank2)
    assert float(kl_s[0, 0]) < 1e-6, kl_s
    print("[ok] subspace->promptless + off-subspace->h0 gives loss 0")

    # --- Validity: an example with empty basis (rank 0) contributes nothing.
    kl_z, _ = t._concept_repr_loss(h_pert, h0, h0_base, answer_mask, P2, torch.tensor([0.0]))
    assert float(kl_z[0, 0]) == 0.0, kl_z
    print("[ok] basis_rank=0 zeroes the forget loss")

    # --- rank-1 equivalence: a single unit row must match the closed-form scalar objective
    # (ghat.h_s - ghat.h0_base)^2 + cw * ||off-axis||^2 / d used by the original rank-1 CSD.
    ghat = torch.tensor([0.6, 0.8, 0.0, 0.0, 0.0, 0.0])  # unit vector in span{e0,e1}
    P1 = _basis([ghat.tolist()])
    h_s = torch.randn(1, 2, d)
    kl1, rem1 = t._concept_repr_loss(h_s, h0, h0_base, answer_mask, P1, torch.tensor([1.0]))
    erase_ref = ((ghat * h_s[0, 0]).sum() - (ghat * h0_base[0, 0]).sum()) ** 2
    diff = h_s[0, 0] - h0[0, 0]
    comp = diff - ghat * (ghat * diff).sum()
    ref = erase_ref + 1.0 * (comp ** 2).sum() / d
    assert abs(float(kl1[0, 0]) - float(ref)) < 1e-4, (float(kl1[0, 0]), float(ref))
    print("[ok] subspace_rank=1 reproduces the original rank-1 CSD objective")

    print("\nAll DiffusionCSD concept-subspace-loss checks passed.")


if __name__ == "__main__":
    main()
