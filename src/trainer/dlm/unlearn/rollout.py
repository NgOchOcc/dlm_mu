"""Frozen-model rollout primitives shared by the rollout-based DLM unlearners.

Used by DPDU (``docs/dpdu_unlearning_algorithms.tex``, Alg 1-2), which needs the
same things from the frozen model at every training step: candidate next states
``A^(n) ~ K_{0,t}(. | y_t, x)``, completions ``hat Y_0 ~ P_0(. | A^(n), x)``, and
partially denoised states drawn from an actual reverse trajectory rather than from
forward masking. The probes reuse :func:`generate_answers` and
:func:`denoise_logits` so their generations visit the same states.

Everything here is ``no_grad`` tensor plumbing; the reveal schedule and the
low-confidence remasking follow the same convention as ``dllm``'s sampler, so the
training rollouts and the eval-time generations visit comparable states.

The model forward is factored out (:func:`denoise_logits`) so the trainer can
reuse the *same* base logits for particle sampling, teacher smoothing and the
top-K variant instead of paying for three forwards.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def denoise_logits(model, x, attention_mask, right_shift_logits, vocab_size):
    """One denoising forward, right-shift corrected and clipped to the tokenizer.

    The LM head is padded wider than the tokenizer (Dream: 152064 vs 151667) and
    ids past the tokenizer decode to ``None``, so the tail is dropped here rather
    than at every call site.
    """
    logits = model(input_ids=x, attention_mask=attention_mask).logits
    if right_shift_logits:
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits[..., :vocab_size].float()


def select_reveal(scores, candidates, n_reveal):
    """Top-``n_reveal`` candidate positions per row, as a boolean mask."""
    scores = scores.masked_fill(~candidates, float("-inf"))
    rank = scores.argsort(dim=-1, descending=True).argsort(dim=-1)
    return (rank < n_reveal.unsqueeze(-1)) & candidates


def schedule_reveal_count(remaining, steps_left):
    """Positions to reveal this step so all of them are revealed after the schedule."""
    return torch.ceil(remaining.float() / max(steps_left, 1)).long()


def sample_next_states(
    base_logits,
    masked,
    reveal_fraction,
    num_particles,
    temperature=1.0,
    selection="entropy",
    proposal_top_k=0,
):
    """Alg 1: one shared reveal set ``U_t`` plus ``N`` token draws on it.

    ``U_t`` is shared across particles on purpose: the projected marginal
    ``hat r_{i,t}`` is only well defined if every particle updates position ``i``.
    Particles therefore differ in the *tokens* they commit, which is what the
    safety weights are meant to discriminate.

    Both the position choice and the token draw fight the same measured problem: a
    memorised checkpoint is essentially deterministic. On the LLaDA TOFU SFT model
    the max base probability at the revealed positions was ``0.9999``, so
    confidence-ranked positions sampled at temperature 1 gave ``N`` *identical*
    particles and a teacher equal to the base transition.

    * ``selection="entropy"`` therefore ranks positions by the base entropy --
      where redirection is actually possible -- while ``"confidence"`` reproduces
      the deployed sampler's order.
    * ``temperature > 1`` draws particles from a flattened proposal ``q``. The
      returned ``log_ratio = sum_{i in U_t} log p_0 - log q`` lets the caller
      correct the weights by self-normalized importance sampling, so the teacher
      still estimates the marginal under the true transition ``p_0``.
    * ``proposal_top_k`` confines ``q`` to the ``K`` most likely tokens of ``p_0``.
      A flat proposal over the *whole* vocabulary buys diversity with gibberish:
      at ``T = 5`` the surviving candidates scored ``0.6-0.97`` degeneracy, and the
      teacher preferred them because the leakage they avoid outweighs the fluency
      they lose. Restricting the support is the candidate set ``C_i`` of Alg 3, so
      the estimated tilt is the one restricted to ``C_i``.

    Args:
        base_logits: ``(b, l, V)`` frozen logits at the current state ``y_t``.
        masked: ``(b, l)`` positions currently masked *and* revealable.
        reveal_fraction: Fraction of masked positions promoted to ``U_t``.
        num_particles: Number of candidate next states ``N``; ``0`` returns the
            reveal set only (the top-K teacher of Alg 3 needs no particles).
        temperature: Proposal temperature for the token draws.
        selection: ``"entropy"`` or ``"confidence"`` position ranking.
        proposal_top_k: Restrict the proposal to the top ``K`` tokens of ``p_0``
            (``0`` = the whole vocabulary).

    Returns:
        ``(reveal, tokens, log_ratio)``: ``reveal`` a ``(b, l)`` bool mask
        (``U_t``), ``tokens`` a ``(b, l, N)`` long tensor of sampled ids, and
        ``log_ratio`` a ``(b, N)`` importance log-ratio (both ``None`` when
        ``num_particles == 0``; ``log_ratio`` is all-zero at ``temperature == 1``).
    """
    if selection not in ("entropy", "confidence"):
        raise ValueError("selection must be 'entropy' or 'confidence'")
    b, l, vocab = base_logits.shape
    log_p = F.log_softmax(base_logits, dim=-1)
    probs = log_p.exp()

    remaining = masked.sum(-1)
    n_reveal = torch.ceil(remaining.float() * reveal_fraction).long()
    n_reveal = torch.where(remaining > 0, n_reveal.clamp_min(1), remaining)
    if selection == "entropy":
        score = -(probs * log_p).sum(-1)
    else:
        score = probs.max(dim=-1).values
    reveal = select_reveal(score, masked, n_reveal)

    if num_particles <= 0:
        return reveal, None, None

    log_q = (
        log_p
        if temperature == 1.0
        else F.log_softmax(base_logits / temperature, dim=-1)
    )
    if 0 < proposal_top_k < vocab:
        kept = base_logits >= base_logits.topk(proposal_top_k, dim=-1).values[..., -1:]
        log_q = F.log_softmax(log_q.masked_fill(~kept, float("-inf")), dim=-1)
    tokens = torch.multinomial(
        log_q.exp().reshape(-1, vocab), num_particles, replacement=True
    ).view(b, l, num_particles)
    ratio = (log_p - log_q).gather(-1, tokens) * reveal.unsqueeze(-1)
    return reveal, tokens, ratio.sum(1)


@torch.no_grad()
def complete_states(
    model,
    x,
    revealable,
    attention_mask,
    steps,
    mask_id,
    right_shift_logits,
    vocab_size,
    temperature=1.0,
    chunk_size=0,
):
    """Finish every still-masked revealable position with the given model.

    ``steps`` reveal rounds are used regardless of how many positions remain, so a
    small ``steps`` is a coarse (cheaper, lower-quality) completion rather than a
    truncated one -- the final round reveals whatever is left.

    The loop deliberately has no data-dependent early exit: under ZeRO-3 every
    forward is a collective, so a rank that finished early would hang the others.
    ``chunk_size`` splits the rollout batch to bound the transient
    ``(n, l, vocab)`` logits; the chunk count depends only on configured sizes, so
    it stays rank-invariant for the same reason.
    """
    from dllm.core.samplers import add_gumbel_noise

    n = x.shape[0]
    if chunk_size and n > chunk_size:
        return torch.cat(
            [
                complete_states(
                    model,
                    x[s : s + chunk_size],
                    revealable[s : s + chunk_size],
                    None if attention_mask is None else attention_mask[s : s + chunk_size],
                    steps,
                    mask_id,
                    right_shift_logits,
                    vocab_size,
                    temperature=temperature,
                )
                for s in range(0, n, chunk_size)
            ]
        )

    for step in range(steps):
        masked = (x == mask_id) & revealable
        remaining = masked.sum(-1)

        logits = denoise_logits(
            model, x, attention_mask, right_shift_logits, vocab_size
        )
        x0 = add_gumbel_noise(logits, temperature=temperature).argmax(dim=-1)
        confidence = F.softmax(logits, dim=-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)

        n_reveal = schedule_reveal_count(remaining, steps - step)
        reveal = select_reveal(confidence, masked, n_reveal)
        x = torch.where(reveal, x0, x)
    return x


@torch.no_grad()
def reverse_trajectory_state(
    model,
    input_ids,
    answer_mask,
    attention_mask,
    traj_steps,
    mask_id,
    right_shift_logits,
    vocab_size,
    temperature=1.0,
):
    """A partially denoised state from a real reverse trajectory.

    Starts from a fully masked answer and runs a per-row random number of reverse
    steps ``r ~ U{1, ..., traj_steps - 1}``, so the revealed tokens are
    *model-generated* rather than gold. This is the state source the proposal's
    Failure Check #3 asks for: forward-masked gold answers never appear at
    inference time, and a teacher fitted only on them transfers poorly.

    Returns ``(y_t, still_masked, mask_fraction)`` where ``mask_fraction`` is the
    per-row fraction of answer positions left masked, used by the caller to pick a
    loss weight for the state.

    Like :func:`complete_states`, the loop runs a fixed number of forwards so all
    ZeRO-3 ranks issue the same collectives; rows that have already stopped are
    excluded through ``active`` instead of by breaking out.
    """
    from dllm.core.samplers import add_gumbel_noise

    x = torch.where(answer_mask, torch.full_like(input_ids, mask_id), input_ids)
    stop_after = torch.randint(
        1, max(traj_steps, 2), (input_ids.shape[0],), device=input_ids.device
    )

    for step in range(traj_steps):
        active = (stop_after > step) & (answer_mask.sum(-1) > 0)
        masked = (x == mask_id) & answer_mask
        remaining = masked.sum(-1)

        logits = denoise_logits(
            model, x, attention_mask, right_shift_logits, vocab_size
        )
        x0 = add_gumbel_noise(logits, temperature=temperature).argmax(dim=-1)
        confidence = F.softmax(logits, dim=-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)

        n_reveal = schedule_reveal_count(remaining, traj_steps - step)
        reveal = select_reveal(confidence, masked, n_reveal) & active.unsqueeze(-1)
        x = torch.where(reveal, x0, x)

    still_masked = (x == mask_id) & answer_mask
    answer_count = answer_mask.sum(-1).clamp_min(1)
    return x, still_masked, still_masked.sum(-1).float() / answer_count


@torch.no_grad()
def teacher_forced_reveal_order(
    model,
    input_ids,
    answer_mask,
    attention_mask,
    steps,
    mask_id,
    right_shift_logits,
    vocab_size,
):
    """Step at which the deployment policy reveals each answer position.

    Alg 1 of ``docs/traj_rmdu_training_algorithms.tex``: the position schedule is
    the deployed decoder's (confidence ranking, ``ceil(remaining / steps_left)``
    per step) while the revealed tokens are the gold ones, so the trajectory
    carries the decoder's mask topology without its generation errors.

    Because every revealed token is gold, a state is fully determined by which
    positions are still masked, so the whole trajectory compresses into this one
    integer tensor: state ``k`` masks the positions whose order is ``>= k``. The
    order is deterministic (frozen model, ``argmax`` confidence), which is what
    makes it cacheable across epochs.

    ``steps`` must be passed in already synchronised across ranks: the loop body
    is a collective under ZeRO-3, so a rank running fewer steps would hang.
    """
    x = torch.where(answer_mask, torch.full_like(input_ids, mask_id), input_ids)
    order = torch.full_like(input_ids, steps - 1)

    for step in range(steps):
        masked = (x == mask_id) & answer_mask
        logits = denoise_logits(
            model, x, attention_mask, right_shift_logits, vocab_size
        )
        confidence = F.softmax(logits, dim=-1).max(dim=-1).values

        n_reveal = schedule_reveal_count(masked.sum(-1), steps - step)
        reveal = select_reveal(confidence, masked, n_reveal)
        order = torch.where(reveal, torch.full_like(order, step), order)
        x = torch.where(reveal, input_ids, x)
    return order


@torch.no_grad()
def generate_answers(
    model,
    input_ids,
    answer_mask,
    attention_mask,
    steps,
    mask_id,
    right_shift_logits,
    vocab_size,
    temperature=1.0,
):
    """Unconstrained answers from the fully masked state (dual update, Alg 4)."""
    x = torch.where(answer_mask, torch.full_like(input_ids, mask_id), input_ids)
    return complete_states(
        model,
        x,
        answer_mask,
        attention_mask,
        steps,
        mask_id,
        right_shift_logits,
        vocab_size,
        temperature=temperature,
    )


def decode_answers(tokenizer, x, answer_mask):
    """Decoded answer span of each row, in the order the rows appear."""
    return [
        tokenizer.decode(row[row_mask], skip_special_tokens=True).strip()
        for row, row_mask in zip(x, answer_mask)
    ]
