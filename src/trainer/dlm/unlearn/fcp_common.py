"""Shared math for Fact-wise Counterfactual Projection (FCP-MDLM).

These helpers are used by *both* the offline cap-construction pass
(``fcp_precompute.py``) and the online trainer (``fcp.py``). Keeping them here,
free of any HuggingFace ``Trainer`` dependency, guarantees the precompute and
training paths compute the target probability ``p_a`` and the fact-bearing span
``K`` in an identical way (see docs/FCP_MDLM_training_algorithm.tex).

Frame convention: every function operates on tensors that are already in the
model's *post-preprocess* frame (BOS prepended / labels shifted for Dream) and
on logits that have already been passed through ``_postprocess_outputs``.
"""

import torch
import torch.nn.functional as F


def target_vs_rest_logodds(logits, positions, target_tokens):
    """m_theta(a) = z_i(y_i) - logsumexp_{v != y_i} z_i(v).

    Args:
        logits: (l, V) logits for a single sequence (post-processed).
        positions: (n,) long, answer positions to score.
        target_tokens: (n,) long, gold token id y_i at each position.

    Returns:
        (n,) target-versus-rest log-odds m at each requested position.
    """
    z = logits[positions].float()  # (n, V); upcast the small slice for precision
    tgt = z.gather(1, target_tokens.unsqueeze(1)).squeeze(1)  # (n,)
    # logsumexp over all v, then subtract the target's contribution.
    lse_all = torch.logsumexp(z, dim=1)  # (n,)
    # log(sum_{v!=y} exp z_v) = log(exp(lse_all) - exp(z_y)); stable form:
    lse_rest = lse_all + torch.log1p(-torch.exp(tgt - lse_all).clamp(max=1 - 1e-6))
    return tgt - lse_rest


def target_prob(logits, positions, target_tokens):
    """p_a = softmax(z_i)[y_i] at each requested position. Shapes as above."""
    z = logits[positions].float()  # (n, V); upcast the small slice for precision
    logp = F.log_softmax(z, dim=-1)
    return logp.gather(1, target_tokens.unsqueeze(1)).squeeze(1).exp()  # (n,)


def projection_loss(p_a, rho):
    """One-sided minimum-change projection loss (Eq. projection-loss).

    ``0`` when ``p_a <= rho``; otherwise ``KL(Bern(rho) || Bern(p_a))``. The
    gradient w.r.t. the target-vs-rest logit equals ``p_a - rho`` on the active
    branch, i.e. it vanishes exactly at the forgetting boundary. ``rho`` is a
    constant (detached cap).

    Args:
        p_a: (...,) target probabilities (require grad through the policy).
        rho: (...,) per-atom caps, broadcastable to ``p_a``.
    """
    eps = 1e-6
    p = p_a.clamp(eps, 1 - eps)
    r = rho.clamp(eps, 1 - eps)
    active = p_a > rho
    kl = r * (r.log() - p.log()) + (1 - r) * ((1 - r).log() - (1 - p).log())
    return torch.where(active, kl, torch.zeros_like(kl))


def select_span_K(d_scores, prompt_copied, kappa):
    """Shortest prefix of positions (sorted by acquired evidence) covering a
    fraction ``kappa`` of total evidence, dropping prompt-copied positions.

    Implements the Alg. 1 selection. Operates on the *candidate* answer
    positions of one fact.

    Args:
        d_scores: (n,) acquired-evidence d(a) >= 0 for each answer position.
        prompt_copied: (n,) bool, True if the answer token also appears in the
            prompt (excluded, "not copied from the prompt").
        kappa: coverage fraction in (0, 1].

    Returns:
        (k,) long tensor of selected local indices into the answer-position
        array (a subset, in descending-evidence order). Falls back to all
        non-copied candidates when total evidence is zero.
    """
    n = d_scores.shape[0]
    cand = (~prompt_copied).nonzero(as_tuple=True)[0]
    if cand.numel() == 0:
        cand = torch.arange(n, device=d_scores.device)  # degenerate: keep all
    d = d_scores[cand]
    total = d.sum()
    order = torch.argsort(d, descending=True)
    cand_sorted = cand[order]
    if total <= 0:
        return cand_sorted  # zero acquired evidence -> monitor everything
    d_sorted = d[order]
    csum = torch.cumsum(d_sorted, dim=0)
    # smallest prefix whose cumulative evidence reaches kappa * total
    reached = (csum >= kappa * total).nonzero(as_tuple=True)[0]
    k = int(reached[0].item()) + 1 if reached.numel() > 0 else cand_sorted.numel()
    return cand_sorted[:k]


def apply_strict_cap(rho, strict_epsilon, k):
    """rho_a <- min(rho_cf, epsilon^(1/k)) for strict-erasure mode (optional)."""
    if strict_epsilon is None:
        return rho
    bar = float(strict_epsilon) ** (1.0 / max(int(k), 1))
    return torch.minimum(rho, torch.full_like(rho, bar))
