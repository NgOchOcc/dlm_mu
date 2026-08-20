"""CPU verification of the Traj-IProj target (docs/traj_iproj_algorithms.tex).

Runs the identities the doc asks to check before any training, plus the end-to-end
loss wiring against a stub model, so all of it is reproducible without a GPU or a
checkpoint:

* the projection is a distribution, hits the cap exactly, preserves odds ratios
  within each side of ``B_i``, and is bitwise the teacher wherever it does not bind;
* Eq. 17 -- the closed form for ``KL(p||q*) - KL(p||b)`` -- matches the realized gap;
* the margin cap moves the greedy argmax out of ``B_i`` and the knowledge cap alone
  does not, which is the whole reason the margin cap exists;
* ``Spell(y)`` on the real LLaDA tokenizer admits the measured reroute variants,
  never drops the gold piece, never admits a special token, and fits ``max_ids``;
* ``_forget_loss`` produces a finite loss and a finite gradient for every state
  source, both KL directions, and both backbones' ``right_shift_logits`` settings.

Usage (needs only torch + transformers; the tokenizer half needs the LLaDA
tokenizer in the HF cache):

    python scripts/dlm/verify_traj_iproj.py            # everything
    python scripts/dlm/verify_traj_iproj.py --no-tokenizer
"""

import argparse
import os
import statistics as st
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trainer.dlm.unlearn.iproj_core import (  # noqa: E402
    SpellingTable,
    active_positions,
    allowed_vocab_mask,
    belief_set,
    compose_cap,
    masked_stats,
    project,
)

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


# -- Algorithm 1: the projection identities -----------------------------------
def verify_projection():
    print("\n== projection identities ==")
    torch.manual_seed(0)
    batch, length, vocab, top_k = 3, 7, 64, 8

    logits = torch.randn(batch, length, vocab) * 2
    gold_ids = torch.randint(0, vocab, (batch, length))
    # Row 0 stands in for a memorised checkpoint: b(y) ~ 0.999.
    logits[0].scatter_(-1, gold_ids[0].unsqueeze(-1), 9.0)
    log_b = logits.log_softmax(-1)

    masked = torch.zeros(batch, length, dtype=torch.bool)
    masked[:, 2:] = True
    topk_log_b, topk_index = log_b.topk(top_k, -1)
    log_a = torch.randn(batch, length, vocab).log_softmax(-1)
    topk_log_a = log_a.gather(-1, topk_index)
    log_b_gold = log_b.gather(-1, gold_ids.unsqueeze(-1)).squeeze(-1)
    residual = (
        log_b_gold - log_a.gather(-1, gold_ids.unsqueeze(-1)).squeeze(-1)
    ).relu()

    spell = torch.full((batch, length, 4), -1, dtype=torch.long)
    spell[..., 0] = (gold_ids + 1) % vocab
    floor = float(torch.log(torch.tensor(1e-4)))

    active = active_positions(residual, masked, 0.5)
    forbidden = belief_set(
        topk_index, topk_log_b, topk_log_a, spell, gold_ids, masked, vocab, 0.5, floor
    )
    log_s, log_1ms, log_mu, log_lead = masked_stats(log_b, forbidden, chunk=2)
    anchor = torch.full((batch, length), 0.45)
    caps = compose_cap(log_s, log_1ms, log_mu, log_lead, anchor, rho=1.0, kappa=0.5)
    log_q, diag = project(
        log_b, forbidden, caps["cap"], caps["log_s"], caps["log_1ms"], active
    )
    binds = diag["binds"]
    neg_inf = torch.finfo(log_q.dtype).min

    norm = log_q.logsumexp(-1).exp()
    check(
        "q* rows normalize",
        torch.allclose(norm, torch.ones_like(norm), atol=1e-5),
        f"max|1-sum|={float((norm - 1).abs().max()):.2e}",
    )

    q_mass = log_q.masked_fill(~forbidden, neg_inf).logsumexp(-1).exp()
    err = (q_mass.double() - caps["cap"])[binds].abs()
    check(
        "q*(B) == cap where binding",
        bool(err.numel()) and float(err.max()) < 1e-6,
        f"n={int(binds.sum())} max err={float(err.max()):.2e}",
    )

    untouched = ~binds
    check(
        "q* == b bitwise off the binding set",
        bool((log_q[untouched] == log_b[untouched]).all()),
        f"n={int(untouched.sum())}",
    )

    # The projection rescales each side by a constant, so the log shift is flat
    # within B_i -- that flatness *is* the odds preservation.
    shift = log_q - log_b
    spread = (
        shift.masked_fill(~forbidden, -1e30).amax(-1)
        - shift.masked_fill(~forbidden, 1e30).amin(-1)
    )
    check(
        "shift constant inside B (odds preserved)",
        float(spread[binds].abs().max()) < 1e-5,
        f"max spread={float(spread[binds].abs().max()):.2e}",
    )

    memorised = binds[0] & active[0]
    escaped = diag["argmax_escaped"][0][memorised]
    check(
        "margin cap moves argmax out of B (memorised row)",
        bool(memorised.any()) and bool(escaped.all()),
        f"escaped {int(escaped.sum())}/{int(memorised.sum())}",
    )
    every = binds & active
    check(
        "margin cap moves argmax out of B (all binding positions)",
        bool(diag["argmax_escaped"][every].all()),
        f"escaped {int(diag['argmax_escaped'][every].sum())}/{int(every.sum())}",
    )

    # The predicted no-op: the anchor sits far above the level the decoder sees.
    caps_k = compose_cap(
        log_s, log_1ms, log_mu, log_lead, anchor, rho=1.0, margin_cap=False
    )
    _, diag_k = project(
        log_b, forbidden, caps_k["cap"], caps_k["log_s"], caps_k["log_1ms"], active
    )
    mem_k = diag_k["binds"][0] & active[0]
    check(
        "knowledge cap alone leaves argmax inside B",
        bool(mem_k.any()) and not bool(diag_k["argmax_escaped"][0][mem_k].any()),
        f"cap {float(caps_k['cap'][0][mem_k].mean()):.3f} (knowledge) vs "
        f"{float(caps['cap'][0][memorised].mean()):.3f} (margin)",
    )

    loose = compose_cap(
        log_s, log_1ms, log_mu, log_lead, anchor_mass=None, margin_cap=False
    )
    log_q_l, diag_l = project(
        log_b, forbidden, loose["cap"], loose["log_s"], loose["log_1ms"], active
    )
    check(
        "cap == s is a no-op",
        bool((log_q_l == log_b).all()) and not bool(diag_l["binds"].any()),
    )

    # Eq. 17 in full, before the p(B) ~ 1 approximation.
    log_p = torch.randn(batch, length, vocab).log_softmax(-1)
    kl = lambda log_target: (log_p.exp() * (log_p - log_target)).sum(-1)  # noqa: E731
    realized = (kl(log_q) - kl(log_b)).double()
    p_in = log_p.masked_fill(~forbidden, neg_inf).logsumexp(-1).exp().double()
    cap = caps["cap"].clamp(1e-12, 1 - 1e-12)
    predicted = (
        -p_in * (cap.log() - caps["log_s"])
        - (1 - p_in) * (torch.log1p(-cap) - caps["log_1ms"])
    ) * binds.double()
    check(
        "Eq. 17 closed form matches the realized KL gap",
        float((realized - predicted)[binds].abs().max()) < 1e-4,
        f"max err={float((realized - predicted)[binds].abs().max()):.2e}",
    )

    for eta in (0.15, 0.3, 0.5, 1.0):
        got = active_positions(residual, masked, eta)
        assert bool((got <= masked).all())
        print(f"      eta={eta}: realized active fraction {float(got.sum() / masked.sum()):.3f}")
    check(
        "active subset of masked, eta=1 is identity",
        bool((active_positions(residual, masked, 1.0) == masked).all()),
    )


# -- belief-set construction and degenerate inputs ----------------------------
def verify_belief_set():
    print("\n== belief set and degenerate inputs ==")
    torch.manual_seed(1)
    batch, length, vocab, top_k = 2, 5, 40, 6
    log_b = (torch.randn(batch, length, vocab) * 2).log_softmax(-1)
    gold = torch.randint(0, vocab, (batch, length))
    masked = torch.ones(batch, length, dtype=torch.bool)
    tk_b, tk_i = log_b.topk(top_k, -1)
    tk_a = tk_b - 5.0  # every top-K token counts as lifted
    floor = float(torch.log(torch.tensor(1e-4)))

    pad = torch.full((batch, length, 4), -1, dtype=torch.long)
    as_gold = gold.unsqueeze(-1).expand(batch, length, 4).contiguous()
    f_pad = belief_set(tk_i, tk_b, tk_a, pad, gold, masked, vocab, 0.5, floor)
    f_gold = belief_set(tk_i, tk_b, tk_a, as_gold, gold, masked, vocab, 0.5, floor)
    check("spell -1 padding forbids nothing extra", bool((f_pad == f_gold).all()))
    check("gold always in B", bool(f_pad.gather(-1, gold.unsqueeze(-1)).all()))

    unmasked = torch.zeros(batch, length, dtype=torch.bool)
    check(
        "B empty where not masked",
        not bool(
            belief_set(tk_i, tk_b, tk_a, pad, gold, unmasked, vocab, 0.5, floor).any()
        ),
    )

    allowed = torch.ones(vocab, dtype=torch.bool)
    allowed[gold[0, 0]] = False
    f_allow = belief_set(
        tk_i, tk_b, tk_a, pad, gold, masked, vocab, 0.5, floor, allowed
    )
    check("allowed mask strips ids from B", not bool(f_allow[0, 0, gold[0, 0]]))

    one = masked_stats(log_b, f_pad, chunk=1)
    whole = masked_stats(log_b, f_pad, chunk=batch)
    check(
        "masked_stats chunk-invariant",
        all(torch.equal(x, y) for x, y in zip(one, whole)),
    )

    log_s, log_1ms, log_mu, log_lead = one
    prob = log_b.exp().double()
    worst = max(
        float((log_s.exp().double() - (prob * f_pad).sum(-1)).abs().max()),
        float((log_1ms.exp().double() - (prob * ~f_pad).sum(-1)).abs().max()),
        float((log_mu.exp().double() - prob.masked_fill(f_pad, -1).amax(-1)).abs().max()),
        float(
            (log_lead.exp().double() - prob.masked_fill(~f_pad, -1).amax(-1)).abs().max()
        ),
    )
    check("masked_stats == dense reference", worst < 1e-6, f"max err={worst:.2e}")

    for margin in (True, False):
        for anchor in (None, torch.full((batch, length), 0.3)):
            caps = compose_cap(
                log_s, log_1ms, log_mu, log_lead, anchor, kappa=0.5, margin_cap=margin
            )
            log_q, _ = project(
                log_b, f_pad, caps["cap"], caps["log_s"], caps["log_1ms"], masked
            )
            check(
                f"cap finite, q* clean (margin={margin}, anchor={anchor is not None})",
                bool(torch.isfinite(caps["cap"]).all())
                and not bool(torch.isnan(log_q).any()),
                f"mean cap={float(caps['cap'].mean()):.4f}",
            )

    # B == V leaves no safe side at all: 1 - s == 0, so the safe rescale is 1/0.
    everything = torch.ones(batch, length, vocab, dtype=torch.bool)
    stats = masked_stats(log_b, everything)
    caps = compose_cap(*stats, anchor_mass=torch.full((batch, length), 0.3), kappa=0.5)
    log_q, _ = project(
        log_b, everything, caps["cap"], caps["log_s"], caps["log_1ms"], masked
    )
    check(
        "degenerate B == V stays finite",
        bool(torch.isfinite(log_q).all()),
        f"cap={float(caps['cap'].mean()):.3f}",
    )


# -- Spell(y) on the real tokenizer -------------------------------------------
NAMES = [
    "Hsiao Yun-Hwa", "Basil Mahfouz Al-Kuwaiti", "Nikolai Abilov", "Jaime Vasquez",
    "Xin Lee Williams", "Moshe Ben-David", "Kalkidan Abera", "Takashi Nakamura",
    "Raven Marais", "Aysha Al-Hashim", "Edward Patrick Sullivan", "Behrouz Rohani",
    "Wei-Jun Chen", "Tae-ho Park", "Hina Ameen", "Rajeev Majumdar",
    "Jad Ambrose Al-Shamary", "Adib Jarrah", "Ji-Yeon Park", "Sebastian Marino",
    "Maria Estela Gutierrez", "Nadir Hafeez", "Yevgeny Grimkov", "Anara Yusifova",
    "Jordan Sinclair", "Elliot Patrick Benson", "Alejandro Tomasino",
    "Ingrid Christensen",
]


def verify_spelling(model_id="GSAI-ML/LLaDA-8B-Instruct"):
    print(f"\n== Spell(y) on {model_id} ==")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    vocab = len(tokenizer)
    table = SpellingTable(tokenizer, vocab)

    sizes, dropped = [], 0
    for name in NAMES:
        ids = tokenizer(" " + name, add_special_tokens=False)["input_ids"]
        rows = table.row_spell_ids(ids, [True] * len(ids))
        for position, token_id in enumerate(ids):
            sizes.append(int((rows[position] >= 0).sum()))
            # A piece must always spell its own position, or the cap it protects is
            # escapable by emitting the gold token itself.
            dropped += token_id not in rows[position].tolist()
    ordered = sorted(sizes)
    check("gold piece always in Spell at its own position", dropped == 0,
          f"{len(sizes)} positions, {dropped} dropped")
    check(
        "Spell fits max_ids",
        max(sizes) <= table.max_ids,
        f"mean={st.mean(sizes):.1f} med={st.median(sizes)} "
        f"p95={ordered[int(0.95 * len(ordered))]} max={max(sizes)} "
        f"(cap {table.max_ids}, {max(sizes) / vocab * 100:.4f}% of vocab)",
    )

    ids = tokenizer(" Hsiao Yun-Hwa", add_special_tokens=False)["input_ids"]
    pieces = tokenizer.convert_ids_to_tokens(ids)
    rows = table.row_spell_ids(ids, [True] * len(ids))
    print(f"      tokenization: {pieces}")
    # The measured reroute targets: case, leading-space and punctuation variants.
    probes = [
        ("ĠYun", "ĠYun"), ("ĠYun", "yun"), ("ĠYun", "Yu"), ("ĠYun", "-Y"),
        ("-H", "-H"), ("-H", "-h"), ("wa", "WA"),
    ]
    admitted = True
    for piece, variant in probes:
        variant_id = tokenizer.convert_tokens_to_ids(variant)
        if variant_id is None:
            print(f"      SKIP {variant!r} absent from this vocabulary")
            continue
        position = pieces.index(piece)
        hit = variant_id in rows[position].tolist()
        admitted &= hit
        print(f"      {'ok ' if hit else 'NO '} {piece!r} admits {variant!r}")
    check("Spell admits the measured reroute variants", admitted)

    special = set(tokenizer.all_special_ids)
    check(
        "no special id in Spell",
        not any(int(i) in special for i in rows.flatten().tolist()),
        f"{len(special)} special ids",
    )
    blocked = int((~allowed_vocab_mask(tokenizer, vocab, torch.device("cpu"))).sum())
    check(
        "allowed_vocab_mask blocks exactly the special ids",
        blocked == len([i for i in special if 0 <= i < vocab]),
        f"{blocked} blocked",
    )


# -- end-to-end loss wiring against a stub model ------------------------------
VOCAB, LENGTH, BATCH, MASK_ID, PROMPT = 64, 48, 3, 63, 8


class _Tokenizer:
    """Byte-BPE-shaped stub: 'Ġa' opens a word, 'a' continues, '-a' is punctuated."""

    mask_token_id = MASK_ID
    bos_token_id = 0
    padding_side = "right"
    all_special_ids = [0, 1, MASK_ID]

    def __len__(self):
        return VOCAB

    def convert_ids_to_tokens(self, ids):
        out = []
        for i in ids:
            letter = chr(ord("a") + i % 26)
            if i in self.all_special_ids:
                out.append(f"<s{i}>")
            elif i < 27:
                out.append("Ġ" + letter)
            elif i < 53:
                out.append(letter)
            else:
                out.append("-" + letter)
        return out


class _StubModel(nn.Module):
    """Logits that depend on the whole sequence, so masking the prompt matters.

    Without the mixing term a position's logits are a function of its own token
    only, the context anchor equals the teacher at every answer position, and the
    gold residual is identically zero -- every position ties and the quantile gate
    degenerates to "admit all", which hides the split-normalizer path.
    """

    def __init__(self, seed, peak):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(VOCAB, VOCAB)
        self.peak = peak

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        logits = self.emb(input_ids) * 0.5
        position = torch.arange(input_ids.shape[1], device=input_ids.device).float()
        logits = logits + (position.view(1, -1, 1) * 0.31).sin()
        context = self.emb(input_ids).mean(1, keepdim=True)
        logits = logits + 1.5 * context * (position.view(1, -1, 1) * 0.17).cos()
        logits.scatter_add_(
            -1,
            (input_ids % VOCAB).unsqueeze(-1),
            torch.full_like(input_ids, self.peak, dtype=logits.dtype).unsqueeze(-1),
        )
        return types.SimpleNamespace(logits=logits)


def _make_trainer():
    """A DiffusionTrajIProj with only the attributes the loss path reads.

    Bypasses ``Trainer.__init__`` deliberately: no accelerator, optimizer or
    checkpoint is needed to exercise the target construction and the reductions.
    """
    from trainer.dlm.unlearn.traj_iproj import DiffusionTrajIProj

    trainer = DiffusionTrajIProj.__new__(DiffusionTrajIProj)
    tokenizer = _Tokenizer()
    trainer.processing_class = tokenizer
    trainer.vocab_size = VOCAB
    trainer.mask_id = MASK_ID
    trainer.right_shift_logits = False
    trainer.time_epsilon = 1e-3
    trainer.scheduler = lambda t: torch.full_like(t, 0.5)  # fixed 50% mask rate
    trainer.eta, trainer.kappa, trainer.rho = 0.3, 0.5, 1.0
    trainer.top_k, trainer.delta_tok = 8, 0.5
    trainer.log_prob_floor = float(torch.log(torch.tensor(1e-4)))
    trainer.margin_cap = True
    trainer.knowledge_anchor, trainer.anchor_top_k = "cached", 16
    trainer.gamma, trainer.gamma_local, trainer.alpha = 1.0, 1.0, 1.0
    trainer.kl_direction = "forward"
    trainer.lambda_traj, trainer.lambda_on = 0.0, 0.0
    trainer.trajectory_steps, trainer.rollout_steps = 0, 4
    trainer.rollout_temperature = 1.0
    trainer.retain_loss_type = "KL"
    trainer.cache_trajectories = True
    trainer.spelling = SpellingTable(tokenizer, VOCAB, max_ids=8)
    trainer.allowed = allowed_vocab_mask(tokenizer, VOCAB, torch.device("cpu"))
    trainer._reveal_cache = {}
    trainer._spell_cache = {}
    trainer._anchor_cache = {}
    trainer._pending_metrics = {}
    trainer.ref_model = _StubModel(0, peak=9.0).eval()
    return trainer


def _grad_norm(model):
    return sum(float(p.grad.norm()) for p in model.parameters() if p.grad is not None)


def _step(trainer, student, inputs):
    student.zero_grad()
    loss, _, metrics = trainer._forget_loss(student, inputs)
    loss.backward()
    return float(loss), _grad_norm(student), metrics


def verify_loss():
    print("\n== end-to-end loss wiring (stub model) ==")
    trainer = _make_trainer()
    student = _StubModel(1, peak=2.0)

    torch.manual_seed(7)
    input_ids = torch.randint(2, MASK_ID, (BATCH, LENGTH))
    labels = input_ids.clone()
    labels[:, :PROMPT] = -100
    inputs = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones(BATCH, LENGTH, dtype=torch.long),
    }

    loss, grad, metrics = _step(trainer, student, inputs)
    check("forget loss finite", torch.isfinite(torch.tensor(loss)), f"loss={loss:.4f}")
    # A -inf column reaching the KL arithmetic survives the forward and only shows
    # up here, as 0 * (-inf + inf) in the backward.
    check("gradient finite and non-zero",
          grad > 0 and torch.isfinite(torch.tensor(grad)), f"|grad|={grad:.4f}")
    check("frozen teacher takes no gradient",
          all(p.grad is None for p in trainer.ref_model.parameters()))

    print("      metrics:")
    for key, value in sorted(metrics.items()):
        print(f"        {key:26s} {value: .6f}")

    check("erasure and local terms both non-zero",
          metrics["iproj/erasure_kl"] > 0 and metrics["iproj/local_kl"] > 0)
    check("projection changes the objective",
          metrics["iproj/kl_gain"] > 0, f"kl_gain={metrics['iproj/kl_gain']:.4f}")
    check("log(s/c) > 0 where the cap binds", metrics["iproj/log_ratio"] > 0)
    check("active rate tracks eta",
          0.2 <= metrics["iproj/active_rate"] <= 0.45,
          f"eta=0.3 -> {metrics['iproj/active_rate']:.3f}")
    check("argmax escapes B on active positions",
          metrics["iproj/argmax_escaped"] > 0.5,
          f"{metrics['iproj/argmax_escaped']:.3f}")

    trainer.margin_cap = False
    loss_k, _, metrics_k = _step(trainer, student, inputs)
    check("knowledge-only arm is a distinct, finite objective",
          torch.isfinite(torch.tensor(loss_k))
          and abs(metrics["iproj/cap"] - metrics_k["iproj/cap"]) > 1e-9,
          f"cap {metrics['iproj/cap']:.4f} (margin) vs {metrics_k['iproj/cap']:.4f}")
    trainer.margin_cap = True

    cached = len(trainer._anchor_cache)
    trainer._forget_loss(student, inputs)
    check("anchor cache reused across steps",
          cached > 0 and len(trainer._anchor_cache) == cached, f"{cached} entries")

    trainer.kl_direction = "kd"
    loss_kd, grad_kd, _ = _step(trainer, student, inputs)
    check("kd direction runs and differs",
          torch.isfinite(torch.tensor(loss_kd)) and grad_kd > 0
          and abs(loss_kd - loss) > 1e-6, f"forward={loss:.4f} kd={loss_kd:.4f}")
    trainer.kl_direction = "forward"

    for lambda_traj, lambda_on, name in (
        (1.0, 0.0, "trajectory only"),
        (0.5, 0.5, "trajectory + on-policy"),
        (0.0, 1.0, "on-policy only"),
    ):
        trainer.lambda_traj, trainer.lambda_on = lambda_traj, lambda_on
        trainer._reveal_cache = {}
        try:
            value, grad, metrics_state = _step(trainer, student, inputs)
            check(f"state source: {name}",
                  torch.isfinite(torch.tensor(value)) and grad > 0,
                  f"loss={value:.4f} |grad|={grad:.4f} "
                  f"mask_rate={metrics_state['iproj/mask_rate']:.3f}")
        except Exception as error:  # noqa: BLE001 - report, do not mask
            check(f"state source: {name}", False, f"{type(error).__name__}: {error}")

    trainer.lambda_traj, trainer.lambda_on = 1.0, 0.0
    trainer._reveal_cache = {}
    trainer._forget_loss(student, inputs)
    cached = len(trainer._reveal_cache)
    trainer._forget_loss(student, inputs)
    check("reveal order cached across steps",
          cached > 0 and len(trainer._reveal_cache) == cached, f"{cached} rows")

    # Dream prepends BOS, which shifts every mask and spell index by one.
    trainer.right_shift_logits = True
    trainer.lambda_traj, trainer.lambda_on = 0.5, 0.0
    trainer._reveal_cache, trainer._spell_cache, trainer._anchor_cache = {}, {}, {}
    try:
        value, grad, _ = _step(trainer, student, inputs)
        check("right_shift_logits=True (Dream) path",
              torch.isfinite(torch.tensor(value)) and grad > 0,
              f"loss={value:.4f} |grad|={grad:.4f}")
    except Exception as error:  # noqa: BLE001
        check("right_shift_logits=True (Dream) path", False,
              f"{type(error).__name__}: {error}")

    trainer.right_shift_logits = False
    trainer.allowed = None
    trainer._spell_cache, trainer._anchor_cache = {}, {}
    value, grad, _ = _step(trainer, student, inputs)
    check("mask_special_tokens=False stays finite",
          torch.isfinite(torch.tensor(value)) and grad > 0,
          f"loss={value:.4f} |grad|={grad:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="skip the Spell(y) checks, which need the LLaDA tokenizer cached",
    )
    parser.add_argument("--model-id", default="GSAI-ML/LLaDA-8B-Instruct")
    args = parser.parse_args()

    verify_projection()
    verify_belief_set()
    if not args.no_tokenizer:
        verify_spelling(args.model_id)
    verify_loss()

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
    else:
        print("ALL PASS")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
