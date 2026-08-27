"""Smoke test for DiffusionSimpleTSD's PCSS forget-position selection.

Verifies the selection path in ``src/trainer/dlm/unlearn/simple_tsd.py`` against the
real class methods without loading a model: the teacher is stubbed with a fixed
prompt-dependence profile, so we bind only the attributes the selection reads onto
an instance built with ``object.__new__`` and exercise

  _select_forget_positions -> {_threshold_scores(_otsu_threshold), _question_overlap_mask}

    python scripts/dlm/tofu/simple_tsd_smoke_test.py
"""

import types

import torch

from trainer.dlm.unlearn.simple_tsd import DiffusionSimpleTSD

MASK, PAD, V = 100, 0, 128


class _FakeTok:
    """Minimal stand-in for the HF tokenizer: maps ids to decoded surfaces so the
    word-group question-overlap logic can run without a real model. Each surface is
    supplied verbatim (leading spaces mark word starts, no space = subword join)."""

    def __init__(self, words):
        self._w = words

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._w.get(int(i), "") for i in ids)


# Default vocab for the selection tests: every id is its own whitespace word.
_DEFAULT_WORDS = {0: "", MASK: "", 1: " who", 2: " beta", 3: " gamma", 4: " delta"}


def make_tsd(select_method="otsu", drop_question_overlap=True, sel_mc=30, otsu_relax=0.0,
             words=None, demotion="minimal", belief_top_k=5):
    t = object.__new__(DiffusionSimpleTSD)
    t.mask_id, t.pad_id, t.vocab_size = MASK, PAD, V
    t.processing_class = _FakeTok(words or _DEFAULT_WORDS)
    t.select_method = select_method
    t.drop_question_overlap = drop_question_overlap
    t.sel_mc = sel_mc
    t.topk_frac = 0.5
    t.tau = 0.0
    t.otsu_relax = otsu_relax
    t.demotion = demotion
    t.belief_top_k = belief_top_k
    t.time_epsilon = 1e-3
    # rho = 1 - scheduler(t) = 0.5 -> each answer token masked ~half the draws.
    t.scheduler = lambda tt: torch.full_like(tt, 0.5)
    return t


def bind_fake_teacher(t):
    """A teacher whose gold log-prob at the two 'prompt-bound' answer positions
    (2, 3) drops sharply when the prompt is masked (s ~ 4.5 nats), while the
    'filler' position 4 is prompt-independent (s ~ 0)."""

    def fake(self, y, attention_mask):
        b, l = y.shape
        lp = y.new_full((b, l, V), -20.0, dtype=torch.float32)
        for bi in range(b):
            prompt_masked = bool(y[bi, 0] == self.mask_id)  # base pass masks prompt
            for pos in range(l):
                g = int(y[bi, pos]) if y[bi, pos] != self.mask_id else 0
                if pos in (2, 3):  # prompt-bound (fact + copied-name)
                    val = -5.0 if prompt_masked else -0.5
                elif pos == 4:  # filler, prompt-independent
                    val = -1.0
                else:
                    val = -1.0
                gold = int(_gold_at(self, pos, bi))
                lp[bi, pos, gold] = val
        return lp

    t._teacher_log_probs = types.MethodType(fake, t)


# The stub must score the GOLD token, which is the clean input id at each position
# (masking replaces it in y, so recover it from the closure-held clean sequence).
_CLEAN = None


def _gold_at(self, pos, bi):
    return _CLEAN[bi, pos]


def _make_cover_stub(loss_mc, retain_frac, select_temp, rho):
    """Instance carrying only what ``_sample_masks_cover`` reads. ``scheduler`` returns a
    constant so ``rho = 1 - scheduler(t)`` is fixed, making per-draw counts deterministic."""
    t = object.__new__(DiffusionSimpleTSD)
    t.loss_mc = loss_mc
    t.retain_frac = retain_frac
    t.select_temp = select_temp
    t.time_epsilon = 1e-3
    t.scheduler = lambda tt: torch.full_like(tt, 1.0 - rho)
    return t


def test_cover_masking():
    """Cover-mode masking: fixed rho/q composition, score-weighted forget selection,
    >=1 forget per draw, uniform fallback, and answer-only masking."""
    torch.manual_seed(0)
    L = 10
    answer_mask = torch.ones(1, L, dtype=torch.bool)
    a_hat = torch.zeros(1, L, dtype=torch.bool)
    a_hat[0, :6] = True  # positions 0..5 forget, 6..9 retain
    scores = torch.zeros(1, L)
    scores[0, :6] = torch.arange(6, dtype=torch.float32)  # pos5 highest prompt-dependence

    # rho=0.5 -> n_mask=round(0.5*10)=5; q=0.4 -> kf=round(0.6*5)=3 forget, kr=2 retain.
    stub = _make_cover_stub(loss_mc=4000, retain_frac=0.4, select_temp=1.0, rho=0.5)
    masks = DiffusionSimpleTSD._sample_masks_cover(stub, answer_mask, a_hat, scores)
    assert masks.shape == (4000, 1, L), masks.shape
    f_counts = masks[:, 0, :6].sum(-1)
    r_counts = masks[:, 0, 6:].sum(-1)
    assert (f_counts == 3).all(), f_counts.unique()
    assert (r_counts == 2).all(), r_counts.unique()
    # never mask outside the answer span (structural: only f_idx/r_idx are set).
    assert not bool((masks[:, 0] & ~answer_mask[0]).any())
    print("[ok] cover: n_mask=round(rho|A|), split kf/kr = (1-q)/q of n_mask")

    # Score weighting: mask frequency rises with the PCSS score across forget positions.
    freq = masks[:, 0, :6].float().mean(0)
    assert int(freq.argmax()) == 5 and int(freq.argmin()) == 0, freq
    assert bool((freq[1:] >= freq[:-1] - 0.03).all()), freq  # weakly monotone in score
    assert float(freq[5] - freq[0]) > 0.3, freq  # clear preference for high score
    print("[ok] cover: forget positions masked weighted by PCSS score (high -> more often)")

    # No scores (manual span) -> uniform: every forget position ~ kf/n_f = 3/6 = 0.5.
    stub_u = _make_cover_stub(loss_mc=4000, retain_frac=0.4, select_temp=1.0, rho=0.5)
    masks_u = DiffusionSimpleTSD._sample_masks_cover(stub_u, answer_mask, a_hat, None)
    freq_u = masks_u[:, 0, :6].float().mean(0)
    assert float((freq_u - 0.5).abs().max()) < 0.05, freq_u
    print("[ok] cover: no scores -> uniform forget selection")

    # rho=0 -> n_mask=0 but >=1 forget still masked; nothing from retain.
    stub_0 = _make_cover_stub(loss_mc=64, retain_frac=0.5, select_temp=1.0, rho=0.0)
    masks_0 = DiffusionSimpleTSD._sample_masks_cover(stub_0, answer_mask, a_hat, scores)
    assert (masks_0[:, 0, :6].sum(-1) == 1).all(), masks_0[:, 0, :6].sum(-1).unique()
    assert (masks_0[:, 0, 6:].sum(-1) == 0).all(), masks_0[:, 0, 6:].sum(-1).unique()
    print("[ok] cover: rho=0 forces exactly one forget token, no retain")

    # rho=1 -> whole answer masked every draw (counts clamp to availability).
    stub_1 = _make_cover_stub(loss_mc=16, retain_frac=0.4, select_temp=1.0, rho=1.0)
    masks_1 = DiffusionSimpleTSD._sample_masks_cover(stub_1, answer_mask, a_hat, scores)
    assert bool(masks_1[:, 0].all()), masks_1[:, 0].sum(-1).unique()
    print("[ok] cover: rho=1 masks the whole answer")

    # No forget positions (empty A_hat) -> the whole answer is "retain", masked at
    # rho|A|=5 per draw with no forget floor; must not crash on the empty forget stratum.
    empty_a = torch.zeros(1, L, dtype=torch.bool)
    stub_e = _make_cover_stub(loss_mc=16, retain_frac=0.4, select_temp=1.0, rho=0.5)
    masks_e = DiffusionSimpleTSD._sample_masks_cover(stub_e, answer_mask, empty_a, None)
    assert bool((masks_e[:, 0].sum(-1) == 5).all()), masks_e[:, 0].sum(-1).unique()
    print("[ok] cover: empty A_hat masks only retain, no forget floor")

    # select_temp -> 0 approaches deterministic top-kf by score (top-3 = pos 3,4,5).
    stub_t = _make_cover_stub(loss_mc=200, retain_frac=0.4, select_temp=0.01, rho=0.5)
    masks_t = DiffusionSimpleTSD._sample_masks_cover(stub_t, answer_mask, a_hat, scores)
    top3 = masks_t[:, 0, 3:6].float().mean(0)
    assert bool((top3 > 0.98).all()), top3  # highest-score 3 forget positions nearly always
    print("[ok] cover: small select_temp -> near-deterministic top-score forget masking")


def main():
    global _CLEAN
    torch.manual_seed(0)

    # prompt = [1, 2]; answer positions 2,3,4 hold ids [3, 2, 4].
    # pos2 (id 3) = a genuine fact token; pos3 (id 2) copies prompt token P2=2;
    # pos4 (id 4) = filler. s is high on 2 and 3, ~0 on 4.
    input_ids = torch.tensor([[1, 2, 3, 2, 4]])
    _CLEAN = input_ids
    answer_mask = torch.tensor([[False, False, True, True, True]])
    attn = torch.ones_like(input_ids)

    # --- Otsu + overlap drop: {pos2, pos3} pass Otsu, overlap removes pos3 (copies q).
    t = make_tsd(select_method="otsu", drop_question_overlap=True)
    bind_fake_teacher(t)
    a_hat, s = t._select_forget_positions(input_ids, answer_mask, attn)
    assert s[0, 2] > 3.0 and s[0, 3] > 3.0, s  # prompt-bound
    assert abs(float(s[0, 4])) < 1.0, s  # prompt-independent
    assert a_hat[0].tolist() == [False, False, True, False, False], a_hat
    print("[ok] PCSS+Otsu selects prompt-bound tokens; overlap drops the copied one")

    # --- Same, overlap OFF: both prompt-bound tokens survive.
    t2 = make_tsd(select_method="otsu", drop_question_overlap=False)
    bind_fake_teacher(t2)
    a_hat2, _ = t2._select_forget_positions(input_ids, answer_mask, attn)
    assert a_hat2[0].tolist() == [False, False, True, True, False], a_hat2
    print("[ok] disabling overlap-drop keeps the question-copy token")

    # --- Otsu never selects outside the answer, even if a prompt token scored high.
    assert not bool((a_hat & ~answer_mask).any())
    print("[ok] A_hat is restricted to answer positions")

    # --- Static Otsu split on a bimodal score vector.
    thr = DiffusionSimpleTSD._otsu_threshold(torch.tensor([0.1, 0.0, 4.8, 5.1, 5.0]))
    assert 0.2 < float(thr) < 4.7, thr
    print("[ok] _otsu_threshold splits a bimodal score vector")

    # --- otsu_relax lowers the threshold toward the minimum (loosen -> select more).
    vec = torch.tensor([0.1, 0.0, 4.8, 5.1, 5.0])
    thr0 = float(DiffusionSimpleTSD._otsu_threshold(vec, relax=0.0))
    thr_r = float(DiffusionSimpleTSD._otsu_threshold(vec, relax=0.5))
    assert thr_r < thr0, (thr0, thr_r)
    assert thr_r >= float(vec.min()) - 1e-6, thr_r
    print("[ok] otsu_relax lowers the threshold (forgets more) but stays >= min")

    # --- Question-overlap drops a WHOLE copied name split across subword tokens.
    # Question exposes "Hsiao Yun-Hwa"; the answer repeats it as subword tokens
    # ("Hs","iao"," Yun","-H","wa") plus a genuine forget word (" engineer"). The
    # per-token id match used to keep "Hs","iao","-H","wa" (their fragments are not
    # whole question words); word-grouping must now drop the entire name and keep
    # only " engineer".
    words = {21: "what is the name hsiao yun hwa", 5: "Hs", 6: "iao",
             7: " Yun", 8: "-H", 9: "wa", 40: " engineer"}
    tname = make_tsd(words=words)
    ids = torch.tensor([[21, 5, 6, 7, 8, 9, 40]])
    ans = torch.tensor([[False, True, True, True, True, True, True]])
    prm = tname._prompt_mask(ids, ans, torch.ones_like(ids))
    ovl = tname._question_overlap_mask(ids, ans, prm)
    assert ovl[0].tolist() == [False, True, True, True, True, True, False], ovl
    print("[ok] question-overlap drops the whole copied name, keeps the fact token")

    # --- A copied word glued to trailing special tokens is still dropped. Answer ends
    # "award<|eot_id|><|endoftext|>"; "award" is a question word, so despite the glued
    # specials the whole group must be flagged (special pieces stripped before match).
    words2 = {21: "which award did she win", 50: " award", 51: "<|eot_id|>",
              52: "<|endoftext|>", 53: " unemployed"}
    tsp = make_tsd(words=words2)
    ids2 = torch.tensor([[21, 53, 50, 51, 52]])
    ans2 = torch.tensor([[False, True, True, True, True]])
    prm2 = tsp._prompt_mask(ids2, ans2, torch.ones_like(ids2))
    ovl2 = tsp._question_overlap_mask(ids2, ans2, prm2)
    # pos1 " unemployed" (fact, not in Q) kept; pos2-4 "award<eot><eos>" dropped.
    assert ovl2[0].tolist() == [False, False, True, True, True], ovl2
    print("[ok] copied word glued to special tokens is dropped; fact word kept")

    # --- Belief vs minimal demotion of the edited teacher target q. belief demotes the
    # gold token AND the top-K HIGHEST-PROBABILITY teacher tokens (matching TSD). Teacher
    # mass: id0=0.5, id1=0.2, ...; top-2 = {0, 1}. gold=0 (the argmax, as at a real fact
    # position) so belief removes gold plus the next alternative; minimal removes only gold.
    log_p0 = torch.log(torch.tensor([[[0.5, 0.2, 0.15, 0.1, 0.03, 0.02]]]))  # (1,1,6)
    tb = make_tsd(demotion="belief", belief_top_k=2)
    demote_b = tb._demote_mask(log_p0, torch.tensor([[0]]))
    assert demote_b[0, 0].tolist() == [True, True, False, False, False, False], demote_b
    # gold OUTSIDE the top-K is force-added on top of the top-K: {0,1} U {3}.
    demote_b2 = tb._demote_mask(log_p0, torch.tensor([[3]]))
    assert demote_b2[0, 0].tolist() == [True, True, False, True, False, False], demote_b2
    tm = make_tsd(demotion="minimal")
    demote_m = tm._demote_mask(log_p0, torch.tensor([[3]]))
    assert demote_m[0, 0].tolist() == [False, False, False, True, False, False], demote_m
    # q must renormalise EXACTLY over the survivors and vanish on the demoted set.
    log_q = log_p0.masked_fill(demote_b, float("-inf"))
    log_q = log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)
    q = log_q.exp()
    assert torch.allclose(q.sum(-1), torch.ones(1, 1), atol=1e-6), q.sum(-1)
    assert float(q[0, 0][demote_b[0, 0]].sum()) < 1e-6, q
    print("[ok] belief demotes gold + top-K highest-prob teacher tokens; q renormalises to 1")

    test_cover_masking()

    print("\nAll SimpleTSD PCSS-selection sanity checks passed.")


if __name__ == "__main__":
    main()
