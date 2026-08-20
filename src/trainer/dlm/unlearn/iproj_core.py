"""Selective belief-mass projection: the target side of Traj-IProj.

Implements Algorithm 1 of ``docs/traj_iproj_algorithms.tex`` as pure tensor
functions plus one tokenizer-derived table, so the trainer
(``traj_iproj.py``) and the pre-training gate (``src/gate_traj_iproj.py``) share
one implementation and cannot drift apart.

The target is an I-projection of the frozen teacher ``b_i = p_0(. | x, z)``:
everything the method does not explicitly forbid keeps the teacher's exact odds.
Three decisions define it, in the order Algorithm 1 makes them:

**Where** -- :func:`active_positions` keeps the top ``eta`` fraction of masked
positions by prompt-induced gold residual, *within each example* (Eq. 4). A fixed
nat threshold is not comparable across state distributions: the mean gold residual
on decoder-aligned states is an order of magnitude above that on forward-corrupted
states at matched mask rate, so one constant makes the gate a no-op on one arm of
the state ablation and near-noise on the other.

**What** -- :class:`SpellingTable` and :func:`belief_set` build ``B_i`` at the level
of *strings* (Eq. 6). A cap on the gold id alone is escapable by re-spelling; the
measured reroute onto case, leading-space and sub-token variants of the same name
averages 0.16 of the displaced mass and exceeds half of it at one position in
eight.

**How far** -- :func:`cap_and_project` composes the knowledge cap with the
*decision-margin* cap (Eq. 9-11). This is the piece that makes the objective
non-vacuous: deployment decoding is greedy, so lowering the mass on ``B_i``
without moving the argmax is measured as zero forgetting, and on a memorised
checkpoint (``b(y) ~ 0.999``, runner-up ``~3e-4``) the argmax only leaves ``B_i``
once the cap drops to ~0.23, far below any anchor mass.

Everything is dense over the vocabulary. That is deliberate: it is the same memory
profile the sibling trainers already run at, and it keeps the code a transcription
of the equations. The one exception is :func:`masked_stats`, which chunks over rows
so the two masked reductions never hold a second full ``(b, l, V)`` temporary.
"""

import string
from collections import defaultdict

import torch

# Byte-level BPE (LLaDA, Dream) marks a word start with U+0120; sentencepiece uses
# U+2581. Newline pieces carry U+010A.
_SPACE_MARKERS = ("Ġ", "▁")
_NEWLINE_MARKER = "Ċ"
_PUNCTUATION = string.punctuation + "’‘“”–—"


def _piece_text(piece):
    """A piece's surface string with the word-start marker turned back into space."""
    for marker in _SPACE_MARKERS:
        piece = piece.replace(marker, " ")
    return piece.replace(_NEWLINE_MARKER, "\n")


def _opens_word(piece):
    return any(piece.startswith(marker) for marker in _SPACE_MARKERS)


class SpellingTable:
    """``Spell(y)``: the vocabulary ids that could start re-spelling the answer.

    A vocabulary id belongs to ``Spell`` at position ``i`` when its piece string
    spells a prefix of the protected substring still to be produced there. The
    substring is scoped to the current *word*: the gold pieces from ``i`` up to the
    next piece that opens a new word. So for the LLaDA tokenization
    ``['ĠYun', '-H', 'wa']`` of "Yun-Hwa", position 0 protects ``"yun-hwa"`` and
    admits ``Y``, ``Yu``, ``Yun``, ``yun``, ``ĠYun``, ``Yun-``, ... while position 1
    protects ``"-hwa"`` and admits ``-``, ``-H``, ``-h``, ``-Hw``, ...

    Matching is case-insensitive and marker-insensitive, which is what folds the
    case and leading-space variants into one set. With ``strip_leading_punct`` a
    piece is also admitted after dropping its leading punctuation, so ``-Y`` counts
    as a spelling of ``"yun..."`` -- the model reaches for those when the
    unpunctuated form is capped.

    ``max_ids`` truncates, so the order matters. Exact prefix matches come first and
    re-punctuated ones after, each longest-piece-first. Ranking by length alone is
    wrong: at LLaDA's vocabulary the letter ``s`` shares its bucket with 45
    punctuated variants (``'s``, ``.s``, ``-s``, ...), all of them two characters
    long, so a pure length sort truncates the gold piece out of its own spelling set.

    Measured on 30 TOFU names (138 positions), the set holds mean 12 / p95 21 /
    max 24 ids without the punctuation tier and mean 46 / p95 67 / max 72 with it,
    so the default ``max_ids`` clears the observed maximum -- i.e. ``Spell`` is
    ~5e-4 of the 126k vocabulary and truncation never fires on this data.

    The table is built once per tokenizer (~0.5 s at LLaDA's 126k vocabulary) and
    lookups are dictionary hits on the prefixes of a short word, so the per-batch
    cost is microseconds. Special tokens are never admitted.
    """

    def __init__(
        self,
        tokenizer,
        vocab_size,
        max_ids=80,
        max_chars=32,
        strip_leading_punct=True,
    ):
        pieces = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
        special = set(tokenizer.all_special_ids)

        self.max_ids = max_ids
        self.max_chars = max_chars
        self.texts = [_piece_text(p) if p is not None else "" for p in pieces]
        self.opens_word = [bool(p) and _opens_word(p) for p in pieces]

        self._exact = defaultdict(list)
        self._bare = defaultdict(list)
        for token_id, text in enumerate(self.texts):
            key = text.strip().lower()
            if not key or token_id in special:
                continue
            self._exact[key].append(token_id)
            if strip_leading_punct:
                bare = key.lstrip(_PUNCTUATION)
                if bare and bare != key:
                    self._bare[bare].append(token_id)

        # Longest piece first within a tier: when the cap bites, the most specific
        # spellings are the ones worth keeping.
        for table in (self._exact, self._bare):
            for ids in table.values():
                ids.sort(key=lambda i: (-len(self.texts[i].strip()), i))

    def spell_ids(self, target):
        """Ids whose piece spells a prefix of ``target`` (a normalized word)."""
        target = target[: self.max_chars]
        found, seen = [], set()
        for table in (self._exact, self._bare):
            tier = []
            for length in range(1, len(target) + 1):
                for token_id in table.get(target[:length], ()):
                    if token_id not in seen:
                        seen.add(token_id)
                        tier.append(token_id)
            tier.sort(key=lambda i: (-len(self.texts[i].strip()), i))
            found.extend(tier)
        return found[: self.max_ids]

    def _remaining_word(self, gold_ids, answer_mask, start):
        """The normalized protected substring still to be spelled at ``start``."""
        parts = []
        for position in range(start, len(gold_ids)):
            if position > start and (
                not answer_mask[position] or self.opens_word[gold_ids[position]]
            ):
                break
            text = self.texts[gold_ids[position]]
            if position > start and not any(ch.isalnum() for ch in text):
                break
            parts.append(text)
        return "".join(parts).strip().lower()

    def row_spell_ids(self, gold_ids, answer_mask):
        """``(l, max_ids)`` int64 of spell ids per position, ``-1`` padded.

        Rows outside the answer are all ``-1``: the belief set is only ever read at
        masked answer positions.
        """
        gold_ids = [int(i) for i in gold_ids]
        answer_mask = [bool(m) for m in answer_mask]
        out = torch.full((len(gold_ids), self.max_ids), -1, dtype=torch.long)
        for position, keep in enumerate(answer_mask):
            if not keep:
                continue
            ids = self.spell_ids(self._remaining_word(gold_ids, answer_mask, position))
            if ids:
                out[position, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out


def allowed_vocab_mask(tokenizer, vocab_size, device):
    """``(V,)`` bool mask of the ids the deployment decoder may emit.

    Special tokens are stripped from every normalized distribution -- both the
    belief set and the projected target -- because a target that spends its freed
    mass on ``[MASK]`` or ``<|endoftext|>`` buys a low leakage score with output the
    decoder can never produce.
    """
    allowed = torch.ones(vocab_size, dtype=torch.bool, device=device)
    special = [i for i in tokenizer.all_special_ids if 0 <= i < vocab_size]
    if special:
        allowed[torch.tensor(special, device=device)] = False
    return allowed


def active_positions(residual, masked, eta):
    """Top-``eta`` fraction of ``masked`` positions per row by ``residual`` (Eq. 4).

    ``eta >= 1`` returns ``masked`` unchanged, which is the "all positions, soft
    constraint" ablation. Ties at the threshold are all admitted, so the realized
    fraction can exceed ``eta`` -- the equation asks for ``|A| ~ eta |M|``, and
    breaking ties arbitrarily would make the gate depend on index order.
    """
    if eta >= 1.0:
        return masked.clone()
    count = masked.sum(-1)
    keep = torch.ceil(count.float() * eta).long().clamp_min(1)
    scores = residual.masked_fill(~masked, float("-inf"))
    ordered = scores.sort(-1, descending=True).values
    index = (keep - 1).clamp(0, residual.shape[-1] - 1).unsqueeze(-1)
    return masked & (residual >= ordered.gather(-1, index))


def belief_set(
    topk_index,
    topk_log_b,
    topk_log_a,
    spell_ids,
    gold_ids,
    keep,
    vocab_size,
    delta_tok,
    log_prob_floor,
    allowed=None,
):
    """``B_i`` as a dense ``(b, l, V)`` bool mask (Eq. 6).

    A token enters either by *lift* -- it is in the teacher's top-``K``, the prompt
    raised it by more than ``delta_tok`` nats, and it carries at least
    ``exp(log_prob_floor)`` mass -- or by *spelling*. The mass floor is not
    cosmetic: the teacher's entropy on a memorised forget answer is ~1e-3 nats, so
    a lift computed on the tail is numerical noise, and without the floor the set
    fills up with ids the model would never emit.

    Only the top-``K`` slice of the lift is needed, so the ``(b, l, V)`` difference
    ``log b - log a`` is never materialized.
    """
    batch, length, _ = topk_index.shape
    forbidden = torch.zeros(
        (batch, length, vocab_size), dtype=torch.bool, device=topk_index.device
    )
    by_lift = (topk_log_b - topk_log_a > delta_tok) & (topk_log_b > log_prob_floor)
    forbidden.scatter_(-1, topk_index, by_lift)

    gold = gold_ids.unsqueeze(-1)
    # Padding slots point at the gold id, which the next line forbids anyway, so
    # the scatter needs no separate validity mask.
    forbidden.scatter_(-1, torch.where(spell_ids >= 0, spell_ids, gold), True)
    forbidden.scatter_(-1, gold, True)

    if allowed is not None:
        forbidden &= allowed
    return forbidden & keep.unsqueeze(-1)


def masked_stats(log_b, forbidden, chunk=1):
    """``(log_s, log_1ms, log_mu, log_lead)``: the four reductions Eq. 9 needs.

    ``s`` is the belief mass, ``1-s`` the safe mass, ``mu`` the best safe token and
    ``lead`` the *largest* teacher mass inside the belief set. The last one is what
    the cap has to be keyed to: the greedy decoder compares the leading forbidden
    token against the leading allowed one, and the gold id is only the leading
    forbidden token when the checkpoint has memorised the answer.

    Each side is summed under its own mask rather than as ``total - forbidden``: on
    a memorised checkpoint ``s`` reaches ``1 - 1e-7``, and the subtraction then
    cancels away the entire safe term. Chunking over rows keeps the two masked
    copies at one row's worth of vocabulary instead of a second full batch.
    """
    neg_inf = torch.finfo(log_b.dtype).min
    parts = []
    for start in range(0, log_b.shape[0], max(chunk, 1)):
        rows = slice(start, start + max(chunk, 1))
        row_log_b, row_forbidden = log_b[rows], forbidden[rows]
        safe = row_log_b.masked_fill(row_forbidden, neg_inf)
        unsafe = row_log_b.masked_fill(~row_forbidden, neg_inf)
        parts.append(
            (
                unsafe.logsumexp(-1),
                safe.logsumexp(-1),
                safe.amax(-1),
                unsafe.amax(-1),
            )
        )
    return tuple(torch.cat(side) for side in zip(*parts))


def compose_cap(
    log_s,
    log_1ms,
    log_mu,
    log_lead,
    anchor_mass=None,
    *,
    rho=1.0,
    kappa=1.0,
    margin_cap=True,
):
    """``c_i`` (Eq. 11) and the two shape scalars behind it, all in float64.

    ``omega = max_{v in B} b(v)/s`` and ``mu_bar = mu/(1-s)`` are the odds-preserving
    shapes inside and outside ``B_i``. Because the projection rescales each side by a
    constant, the post-projection leading forbidden token gets ``c * omega`` and the
    best allowed token ``(1-c) * mu_bar``; requiring the first to fall below ``kappa``
    times the second is what Eq. 10 solves for.

    ``omega`` is keyed to the *leading* forbidden token, not to the gold one. The
    gold id is always in ``B_i``, so this can only tighten the cap, and it is what
    makes the guarantee hold for the quantity the decoder actually compares -- keying
    it to ``b(y)`` alone lets a non-gold member of ``B_i`` (a case variant, a
    sub-token spelling) inherit the argmax and the fact survives verbatim.

    Note ``kappa=1`` is the *equality* boundary: the cap solves ``c*omega =
    kappa*(1-c)*mu_bar``, so the leading forbidden token comes out exactly tied with
    the best allowed one and which of the two wins the greedy argmax is decided by
    index order. ``kappa < 1`` is what strictly moves the decode.
    """
    log_s, log_1ms, log_mu = log_s.double(), log_1ms.double(), log_mu.double()
    mass = log_s.exp()
    omega = (log_lead.double() - log_s).exp()
    mu_bar = (log_mu - log_1ms).exp()

    caps = [mass]
    margin = torch.full_like(mass, float("nan"))
    if margin_cap:
        margin = kappa * mu_bar / (omega + kappa * mu_bar)
        caps.append(margin)
    knowledge = torch.full_like(mass, float("nan"))
    if anchor_mass is not None:
        knowledge = rho * anchor_mass.double()
        caps.append(knowledge)

    cap = torch.stack(caps, dim=0).amin(dim=0)
    return {
        "cap": cap,
        "mass": mass,
        "omega": omega,
        "mu_bar": mu_bar,
        "cap_margin": margin,
        "cap_knowledge": knowledge,
        "log_s": log_s,
        "log_1ms": log_1ms,
    }


def project(log_b, forbidden, cap, log_s, log_1ms, active):
    """``log q*`` (Eq. 12), detached, plus the projection diagnostics.

    ``q*`` scales ``B_i`` by ``c/s`` and its complement by ``(1-c)/(1-s)``, which is
    exactly the I-projection: every odds ratio within each side survives. Positions
    that are inactive, or whose teacher already satisfies the cap, get ``q* = b``
    bitwise -- the shift is multiplied by the binding mask rather than selected by a
    second ``where``, so no extra full-vocabulary temporary is allocated.

    ``log(s/c)`` is returned because it is the whole contribution of the projection:
    when the student is confident inside the belief set,
    ``KL(p||q*) - KL(p||b) ~ log(s/c)`` (Eq. 17), so it answers "is this objective
    doing anything" without a downstream metric.
    """
    binds = active & (log_s.exp() > cap)
    cap = cap.clamp(1e-12, 1.0 - 1e-12)
    log_cap = cap.log()
    shift_forbidden = log_cap - log_s
    shift_safe = torch.log1p(-cap) - log_1ms

    shift = torch.where(
        forbidden, shift_forbidden.unsqueeze(-1), shift_safe.unsqueeze(-1)
    ).to(log_b.dtype)
    shift = torch.nan_to_num(shift) * binds.unsqueeze(-1).to(log_b.dtype)
    log_q = (log_b + shift).detach()

    argmax = log_q.argmax(-1, keepdim=True)
    return log_q, {
        "binds": binds,
        "log_ratio": torch.nan_to_num(log_s - log_cap) * binds.double(),
        "argmax_escaped": ~forbidden.gather(-1, argmax).squeeze(-1),
    }
