#!/usr/bin/env python
"""Single-token prompt-dependence probe for masked diffusion LMs (Dream, LLaDA).

For a QA pair ``(x, y)`` this measures, *token by token*, how much the prompt
``x`` helps the model reconstruct each ground-truth answer token ``y_i``.

Protocol (differs from SimpleTSD's Alg. 1 selection, which masks the WHOLE
answer at once):

  * Mask exactly ONE answer token ``y_i`` at a time, keeping every other answer
    token visible, and read the model's confidence ``p(y_i | ...)`` at that
    position (single forward per position, batched).
  * Do this under two conditions:
      cond  — the real prompt ``x`` is visible;
      base  — the prompt is replaced by ``[MASK]`` tokens (``x_empty``).
  * Report the per-token gap:
      gap_prob = p_cond(y_i) - p_base(y_i)          (probability space)
      s_i      = log p_cond(y_i) - log p_base(y_i)   (nats; Alg.1-style score)

A large positive gap at position ``i`` means the prompt is what makes the model
confident about ``y_i`` — i.e. that token is prompt-bound (memorized-through-x),
which is exactly what a retain-free forget method wants to target.

Usage:
    python scripts/dlm/tofu/single_token_probe.py --model llada
    python scripts/dlm/tofu/single_token_probe.py --model dream --ckpt saves/finetune/... --index 0 --num 3
    python scripts/dlm/tofu/single_token_probe.py --model llada \
        --question "Who wrote ...?" --answer "..."

Notes:
  * For TOFU memorization you almost certainly want ``--ckpt <SFT dir>`` (the
    TOFU-finetuned checkpoint is the Base/target model), not the raw instruct
    weights that are the default here.
  * Dream needs right-shifted logits (predicts token at i-1); LLaDA does not.
    This is set automatically from ``--model``.
"""

import argparse
import os
import re
import sys

import torch
import torch.nn.functional as F

# Keep this DLM diagnostic self-contained but reuse the framework's tokenization
# and diffusion logit conventions (src/ is not installed as a package).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_REPO, "src"))

from data.utils import IGNORE_INDEX, preprocess_chat_instance  # noqa: E402
from evals.metrics.dlm_utils import _postprocess_logits  # noqa: E402


MODEL_PRESETS = {
    "llada": {
        "pretrained": "GSAI-ML/LLaDA-8B-Instruct",
        "right_shift_logits": False,
    },
    "dream": {
        "pretrained": "Dream-org/Dream-v0-Instruct-7B",
        "right_shift_logits": True,
    },
}


def load_dlm(pretrained, dtype="bfloat16"):
    """Load a diffusion LM + tokenizer via the dllm package (mirrors model/dlm.py)."""
    from dllm.utils import get_model, get_tokenizer
    from dllm.utils.configs import ModelArguments

    # Register dllm's bundled LLaDA/Dream classes with the Auto* factories so the
    # tokenizer's diffusion [MASK] token is set (see model/dlm.py for the why).
    for pipeline in ("llada", "dream"):
        try:
            __import__(f"dllm.pipelines.{pipeline}.models")
        except ImportError:
            pass

    args = ModelArguments(model_name_or_path=pretrained, dtype=dtype)
    model = get_model(model_args=args)
    tokenizer = get_tokenizer(model_args=args)
    model.eval()
    return model, tokenizer


def build_pair(tokenizer, question, answer, max_length=512):
    """Chat-templated (prompt+answer) ids/labels/attention_mask for one QA pair."""
    template_args = {"apply_chat_template": True, "system_prompt": None}
    item = preprocess_chat_instance(
        tokenizer, template_args, [question], [answer], max_length,
        predict_with_generate=False,
    )
    return item  # tensors: input_ids (L,), labels (L,), attention_mask (L,)


@torch.no_grad()
def _confidence_at_positions(model, base_ids, attn, answer_pos, extra_mask, mask_id,
                             right_shift_logits, chunk):
    """Gold-token log-prob at each answer position, masking that position (plus any
    positions in ``extra_mask``, e.g. the whole prompt for the base condition).

    Builds one masked variant per answer position and forwards them batched.
    Returns a (N,) tensor of log p(gold_i) aligned with ``answer_pos``.
    """
    device = model.device
    L = base_ids.shape[0]
    N = answer_pos.shape[0]
    gold = base_ids[answer_pos]  # (N,)

    variants = base_ids.unsqueeze(0).repeat(N, 1).clone()  # (N, L)
    if extra_mask is not None and extra_mask.any():
        variants[:, extra_mask] = mask_id
    variants[torch.arange(N), answer_pos] = mask_id
    attn_batch = attn.unsqueeze(0).repeat(N, 1) if attn is not None else None

    out = torch.empty(N, device=device)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        a = attn_batch[start:end] if attn_batch is not None else None
        logits = model(input_ids=variants[start:end], attention_mask=a).logits
        logits = _postprocess_logits(logits, right_shift_logits)
        logp = F.log_softmax(logits.float(), dim=-1)  # (chunk, L, V)
        rows = torch.arange(end - start)
        out[start:end] = logp[rows, answer_pos[start:end], gold[start:end]]
    return out


@torch.no_grad()
def _confidence_whole_answer(model, base_ids, attn, answer_pos, extra_mask, mask_id,
                             right_shift_logits):
    """Gold-token log-prob at every answer position when the WHOLE answer is masked
    at once (single forward). This is exactly SimpleTSD's Alg. 1 selection regime
    (``_select_forget_positions``): ``y_tilde`` masks all answer tokens; ``extra_mask``
    additionally blanks the prompt for the ``x_empty`` pass.

    Returns a (N,) tensor of log p(gold_i) aligned with ``answer_pos``.
    """
    device = model.device
    gold = base_ids[answer_pos]  # (N,)
    noised = base_ids.clone()
    if extra_mask is not None and extra_mask.any():
        noised[extra_mask] = mask_id
    noised[answer_pos] = mask_id  # mask the entire answer span
    a = attn.unsqueeze(0) if attn is not None else None
    logits = model(input_ids=noised.unsqueeze(0), attention_mask=a).logits
    logits = _postprocess_logits(logits, right_shift_logits)
    logp = F.log_softmax(logits.float(), dim=-1)[0]  # (L, V)
    return logp[answer_pos, gold]


@torch.no_grad()
def _confidence_partial(model, base_ids, attn, answer_pos, answer_mask, extra_mask,
                        mask_id, right_shift_logits, scheduler, mc, time_epsilon, chunk):
    """Prompt-CMI over PARTIAL, schedule-sampled answer masks (the proposed score).

    For each of ``mc`` Monte-Carlo draws, sample a masking rate ``rho = 1 -
    scheduler(t)``, ``t ~ U(eps, 1)`` (the same law the trainer uses), mask each
    answer position independently w.p. ``rho``, and read the gold log-prob at the
    masked positions — once with the real prompt (cond) and once with ``extra_mask``
    additionally blanked (base). Per answer token we average over the draws in which
    it was masked, giving

        s_i = E_{m ~ schedule, i in m} [ log p(y_i|x,y_rev) - log p(y_i|x-,y_rev) ]

    i.e. the prompt's contribution to token i *conditioned on realistic partial
    answer context*, not the degenerate empty canvas. Every answer position is
    guaranteed to be masked in at least one draw.

    Returns (logp_cond_mean, logp_base_mean, s_mean), each (N,).
    """
    device = base_ids.device
    N = answer_pos.shape[0]
    gold = base_ids[answer_pos]

    # Sample mc partial masks over the answer span.
    t = time_epsilon + (1 - time_epsilon) * torch.rand(mc, device=device)
    rho = (1.0 - scheduler(t)).clamp(1e-4, 1.0).unsqueeze(1)  # (mc,1)
    masks = (torch.rand(mc, base_ids.shape[0], device=device) < rho) & answer_mask.unsqueeze(0)
    # Guarantee every answer position is covered by >=1 draw (else its s is undefined).
    covered = masks[:, answer_pos].any(0)
    if not bool(covered.all()):
        for k_local, j in enumerate(answer_pos[~covered].tolist()):
            masks[k_local % mc, j] = True

    cond_ids = base_ids.unsqueeze(0).repeat(mc, 1)
    cond_ids = cond_ids.masked_fill(masks, mask_id)
    base_batch = cond_ids.clone()
    if extra_mask is not None and extra_mask.any():
        base_batch[:, extra_mask] = mask_id
    attn_b = attn.unsqueeze(0).repeat(mc, 1) if attn is not None else None

    def _forward_gold(ids):
        out = torch.empty(mc, N, device=device)
        for start in range(0, mc, chunk):
            end = min(start + chunk, mc)
            a = attn_b[start:end] if attn_b is not None else None
            logits = model(input_ids=ids[start:end], attention_mask=a).logits
            logits = _postprocess_logits(logits, right_shift_logits)
            logp = F.log_softmax(logits.float(), dim=-1)
            out[start:end] = logp[:, answer_pos, gold]  # (chunk, N)
        return out

    logp_cond = _forward_gold(cond_ids)   # (mc, N)
    logp_base = _forward_gold(base_batch)
    hit = masks[:, answer_pos].float()    # (mc, N): 1 where token i was masked
    cnt = hit.sum(0).clamp_min(1)
    mean_cond = (logp_cond * hit).sum(0) / cnt
    mean_base = (logp_base * hit).sum(0) / cnt
    s_mean = ((logp_cond - logp_base) * hit).sum(0) / cnt
    return mean_cond, mean_base, s_mean


_GLINER = {}


def _load_gliner(model_name):
    if model_name not in _GLINER:
        from gliner import GLiNER

        _GLINER[model_name] = GLiNER.from_pretrained(model_name)
    return _GLINER[model_name]


def _token_char_spans(tokenizer, ids):
    """Approximate (start, end) char offsets per token via cumulative single-token
    decode. Good enough to map an entity substring onto covering tokens."""
    spans, text = [], ""
    for tid in ids.tolist():
        piece = tokenizer.decode([tid])
        spans.append((len(text), len(text) + len(piece)))
        text += piece
    return text, spans


def _subject_positions(tokenizer, input_ids, prompt_mask, question, subject,
                       gliner_model):
    """Boolean mask (over the full sequence) of prompt tokens covering the subject
    entity. If ``subject`` is None, extract entities from ``question`` with GLiNER.
    Returns None if nothing is found (caller falls back to full-prompt ablation)."""
    prompt_pos = prompt_mask.nonzero(as_tuple=True)[0]
    if prompt_pos.numel() == 0:
        return None
    prompt_ids = input_ids[prompt_pos]
    text, spans = _token_char_spans(tokenizer, prompt_ids)
    low = text.lower()

    subjects = []
    if subject:
        subjects = [subject]
    else:
        try:
            ents = _load_gliner(gliner_model).predict_entities(
                question, ["person", "organization", "location", "book", "work"],
                threshold=0.3,
            )
            subjects = [e["text"] for e in ents]
        except Exception as e:  # network/model issues -> caller falls back
            print(f"[warn] GLiNER failed ({e!r}); using full-prompt ablation", file=sys.stderr)
            return None

    sel = torch.zeros_like(prompt_mask)
    found = False
    for subj in subjects:
        c = low.find(subj.lower())
        while c != -1:
            end = c + len(subj)
            for k, (a, b) in enumerate(spans):
                if a < end and b > c:  # token overlaps the entity char span
                    sel[prompt_pos[k]] = True
                    found = True
            c = low.find(subj.lower(), c + 1)
    return sel if found else None


def probe_pair(model, tokenizer, question, answer, right_shift_logits, mask_id,
               chunk=16, max_length=512, mode="single",
               ablate="prompt", subject=None, gliner_model=None,
               scheduler=None, mc=16, time_epsilon=1e-3,
               select="none", topk=0.5, sel_thr=0.0, otsu_relax=0.0,
               drop_question_overlap=True):
    """Two-condition prompt-dependence probe for one (x, y) pair.

    mode="single":  mask ONE answer token at a time (rest visible).
    mode="whole":   mask the WHOLE answer at once — SimpleTSD's current Alg. 1 score.
    mode="partial": mask a schedule-sampled SUBSET of the answer, averaged over
                    ``mc`` draws — the proposed diffusion-native prompt-CMI score.

    ablate="prompt":  the base condition masks the entire prompt (x_empty).
    ablate="subject": the base condition masks only the subject-entity tokens of
                      the prompt, isolating the fact from the question format.
    """
    item = build_pair(tokenizer, question, answer, max_length)
    device = model.device
    input_ids = item["input_ids"].to(device)
    labels = item["labels"].to(device)
    attn = item["attention_mask"].to(device)
    pad_id = tokenizer.pad_token_id

    if right_shift_logits:
        from dllm.utils.data import prepend_bos

        bos = model.config.bos_token_id if hasattr(model, "config") else None
        shifted = prepend_bos(
            {"input_ids": input_ids.unsqueeze(0),
             "labels": labels.unsqueeze(0),
             "attention_mask": attn.unsqueeze(0)},
            bos_token_id=bos, label_pad_token_id=IGNORE_INDEX,
        )
        input_ids = shifted["input_ids"][0]
        labels = shifted["labels"][0]
        attn = shifted.get("attention_mask", attn.unsqueeze(0))[0]

    answer_mask = labels != IGNORE_INDEX
    answer_pos = answer_mask.nonzero(as_tuple=True)[0]
    if answer_pos.numel() == 0:
        raise ValueError("no answer tokens (labels all IGNORE_INDEX)")

    # Base-condition ablation mask: full prompt, or just the subject entity span.
    prompt_mask = (~answer_mask) & (input_ids != pad_id)
    if attn is not None:
        prompt_mask = prompt_mask & attn.bool()
    ablate_mask = prompt_mask
    if ablate == "subject":
        sub = _subject_positions(tokenizer, input_ids, prompt_mask, question,
                                 subject, gliner_model)
        if sub is not None:
            ablate_mask = sub
        else:
            print("[warn] no subject span located; falling back to full-prompt ablation",
                  file=sys.stderr)

    if mode == "whole":
        logp_cond = _confidence_whole_answer(
            model, input_ids, attn, answer_pos, None, mask_id, right_shift_logits)
        logp_base = _confidence_whole_answer(
            model, input_ids, attn, answer_pos, ablate_mask, mask_id, right_shift_logits)
    elif mode == "partial":
        logp_cond, logp_base, _ = _confidence_partial(
            model, input_ids, attn, answer_pos, answer_mask, ablate_mask, mask_id,
            right_shift_logits, scheduler, mc, time_epsilon, chunk)
    else:
        logp_cond = _confidence_at_positions(
            model, input_ids, attn, answer_pos, None, mask_id, right_shift_logits, chunk)
        logp_base = _confidence_at_positions(
            model, input_ids, attn, answer_pos, ablate_mask, mask_id, right_shift_logits, chunk)
    gold = input_ids[answer_pos]
    s_nats = (logp_cond - logp_base).tolist()
    tokens = [tokenizer.decode([t]) for t in gold.tolist()]
    # A token that merely copies a word/phrase already present in the question is
    # not secret information to forget (the prompt reveals it), even if s_i is high.
    # Subtract this lexical overlap from the selected forget span.
    overlap = _question_overlap_mask(tokens, question) if drop_question_overlap \
        else [False] * len(tokens)
    selected = _select_span(s_nats, select, topk, sel_thr, otsu_relax)
    selected = [sel and not ov for sel, ov in zip(selected, overlap)]
    return {
        "tokens": tokens,
        "p_cond": logp_cond.exp().tolist(),
        "p_base": logp_base.exp().tolist(),
        "logp_cond": logp_cond.tolist(),
        "logp_base": logp_base.tolist(),
        "s_nats": s_nats,
        "overlap": overlap,
        "selected": selected,
    }


def _otsu_threshold(vals, relax=0.0):
    """1-D Otsu split of a list of scores: the threshold maximizing between-class
    variance ``w0*w1*(m0-m1)^2``. Returns the threshold; select ``s >= thr``.

    ``relax`` in [0, 1) loosens the cut toward the low tail so more tokens are
    selected: ``thr' = thr - relax * (thr - min)``. relax=0 is pure Otsu."""
    xs = sorted(vals)
    n = len(xs)
    if n < 2 or xs[0] == xs[-1]:
        return xs[0] if n else 0.0
    total = sum(xs)
    best_t, best_score, csum, cn = xs[0], -1.0, 0.0, 0
    for i in range(n - 1):
        csum += xs[i]
        cn += 1
        w0, w1 = cn / n, (n - cn) / n
        m0 = csum / cn
        m1 = (total - csum) / (n - cn)
        score = w0 * w1 * (m0 - m1) ** 2
        if score > best_score:
            best_score, best_t = score, 0.5 * (xs[i] + xs[i + 1])
    if relax > 0.0:
        best_t = best_t - relax * (best_t - xs[0])
    return best_t


def _group_word_indices(surfaces):
    """Group token positions into whitespace-delimited words. A token whose decoded
    surface begins with whitespace (or the first token) opens a new word; subword
    continuations attach to the current word."""
    groups, cur = [], []
    for i, s in enumerate(surfaces):
        if s[:1].isspace() and cur:
            groups.append(cur)
            cur = [i]
        else:
            cur.append(i)
    if cur:
        groups.append(cur)
    return groups


# Special tokens (<|eot_id|>, <|endoftext|>, <eos>, ...) decode with no leading space,
# so they glue onto the final content word ("award" -> "award.<|eot_id|>..."). Strip
# them before extracting fragments, else their pieces ("eot", "endoftext") make the
# all-fragments-are-question-words test fail and a copied word leaks into A_hat.
_SPECIAL_RE = re.compile(r"<\|?[^<>]*\|?>")


def _content_frags(surface, min_len=2):
    surface = _SPECIAL_RE.sub(" ", surface).lower()
    return [f for f in re.findall(r"[a-z0-9]+", surface) if len(f) >= min_len]


def _question_overlap_mask(tokens, question, min_len=2):
    """Mark answer tokens that reproduce a WORD/PHRASE already in the question.

    Grouping is at WORD granularity so a name split across subword tokens is caught
    as a whole ("Hs"+"iao" -> "Hsiao"; " Yun"+"-H"+"wa" -> "Yun-Hwa"): tokens are
    grouped into whitespace-delimited words, and every token of a word is flagged iff
    every alphanumeric fragment the word contains (len >= ``min_len``) is a question
    word. Such tokens merely echo surface form the prompt already exposes, so they
    carry no secret to forget even when ``s_i`` is high (it is high precisely because
    the prompt supplies it). Matching a full word — not each subword against a whole
    question word — is what keeps a copied name from leaking into A_hat.
    """
    qwords = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) >= min_len}
    mask = [False] * len(tokens)
    if not qwords:
        return mask
    for grp in _group_word_indices(tokens):
        frags = _content_frags("".join(tokens[i] for i in grp), min_len)
        if frags and all(f in qwords for f in frags):
            for i in grp:
                mask[i] = True
    return mask


def _select_span(s_list, method="none", topk=0.5, thr=0.0, otsu_relax=0.0):
    """Boolean forget-selection over answer tokens from their scores ``s_list``.

    method="none": no selection (visualization is the continuous gradient only).
    method="otsu": adaptive per-example 2-class split (recommended, threshold-free);
                   ``otsu_relax`` in [0, 1) loosens it to select more tokens.
    method="topk": keep the top ``topk`` fraction by score.
    method="thr":  keep ``s_i >= thr`` (fixed absolute threshold).
    """
    n = len(s_list)
    if n == 0 or method == "none":
        return [False] * n
    if method == "otsu":
        t = _otsu_threshold(s_list, otsu_relax)
        return [v >= t for v in s_list]
    if method == "topk":
        k = max(1, int(round(topk * n)))
        cutoff = sorted(s_list, reverse=True)[k - 1]
        return [v >= cutoff for v in s_list]
    if method == "thr":
        return [v >= thr for v in s_list]
    raise ValueError(f"unknown select method {method}")


def print_report(question, answer, res, tau=0.0):
    s = torch.tensor(res["s_nats"])
    pc = torch.tensor(res["p_cond"])
    pb = torch.tensor(res["p_base"])
    sel = res.get("selected", [False] * len(res["tokens"]))
    ovl = res.get("overlap", [False] * len(res["tokens"]))
    gap = pc - pb
    print("=" * 88)
    print(f"Q: {question}")
    print(f"A: {answer}")
    print("-" * 88)
    print(f"{'i':>3} {'*':>1} {'token':<14} {'p_cond':>9} {'p_base':>9} {'gap_prob':>9} {'s(nats)':>9}")
    for i, tok in enumerate(res["tokens"]):
        t = tok.replace("\n", "\\n")[:14]
        mark = "*" if sel[i] else ("q" if ovl[i] else " ")
        print(f"{i:>3} {mark:>1} {t:<14} {pc[i]:>9.4f} {pb[i]:>9.4f} {gap[i]:>9.4f} {s[i]:>9.3f}")
    print("-" * 88)
    n_sel = sum(sel)
    n_ovl = sum(ovl)
    print(f"tokens={len(res['tokens'])}  selected={n_sel} "
          f"({n_sel / max(len(sel), 1):.2f})  dropped_q={n_ovl}  "
          f"mean_gap_prob={gap.mean():.4f}  median_gap_prob={gap.median():.4f}")
    print(f"mean_s={s.mean():.3f} nats  median_s={s.median():.3f}  "
          f"frac(s>=tau={tau})={(s >= tau).float().mean():.3f}")
    print("=" * 88)


def _html_escape(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def _color_for(s_val, s_max):
    """Blue→white→red diverging color for a nats score, normalized by |s_max|."""
    m = max(s_max, 1e-6)
    x = max(-1.0, min(1.0, s_val / m))
    if x >= 0:  # prompt-dependent → red
        r, g, b = 255, int(255 * (1 - x)), int(255 * (1 - x))
    else:       # prompt makes it worse → blue
        r, g, b = int(255 * (1 + x)), int(255 * (1 + x)), 255
    return f"rgb({r},{g},{b})"


def render_html(pairs_results, model_name, pretrained, tau):
    """Self-contained HTML: colored answer tokens (by s_i) + per-token detail table."""
    blocks = []
    for idx, (q, a, res) in enumerate(pairs_results):
        s = res["s_nats"]
        pc, pb = res["p_cond"], res["p_base"]
        sel = res.get("selected", [False] * len(s))
        ovl = res.get("overlap", [False] * len(s))
        s_max = max((abs(v) for v in s), default=1.0)
        # Colored token strip (hover shows numbers). Selected tokens (the forget
        # span Â) get a solid outline; unselected tokens are dimmed; tokens dropped
        # for copying the question are struck through.
        spans = []
        for i, tok in enumerate(res["tokens"]):
            disp = _html_escape(tok).replace(" ", "&nbsp;") or "·"
            title = (f"s={s[i]:.3f} nats | p_cond={pc[i]:.4f} p_base={pb[i]:.4f} "
                     f"gap={pc[i]-pb[i]:.4f}" + (" | copies question" if ovl[i] else ""))
            cls = "tok sel" if sel[i] else "tok uns"
            if ovl[i]:
                cls += " ovl"
            spans.append(
                f'<span class="{cls}" style="background:{_color_for(s[i], s_max)}" '
                f'title="{title}">{disp}</span>'
            )
        # Detail rows.
        rows = []
        for i, tok in enumerate(res["tokens"]):
            gap = pc[i] - pb[i]
            cls = "sel" if sel[i] else ("ovl" if ovl[i] else "")
            mark = "&#10003;" if sel[i] else ("q" if ovl[i] else "")
            rows.append(
                f'<tr class="{cls}"><td>{i}</td><td class="mono">{_html_escape(tok)}</td>'
                f'<td>{pc[i]:.4f}</td><td>{pb[i]:.4f}</td><td>{gap:+.4f}</td>'
                f'<td>{s[i]:+.3f}</td><td>{mark}</td></tr>'
            )
        import statistics
        mean_s = statistics.fmean(s) if s else 0.0
        n_sel = sum(sel)
        n_ovl = sum(ovl)
        frac_sel = (n_sel / len(sel)) if sel else 0.0
        mean_gap = statistics.fmean([pc[i] - pb[i] for i in range(len(pc))]) if pc else 0.0
        # Reconstruct the two SimpleTSD groups verbatim: FORGET = Â (selected, gold
        # demoted via KL(q||p_theta)); RETAIN = every other answer token (unselected
        # or dropped-as-question-copy), self-distilled via KL(p0||p_theta).
        toks = res["tokens"]
        forget_txt = _html_escape("".join(t for t, f in zip(toks, sel) if f)) or "<em>(empty)</em>"
        retain_txt = _html_escape("".join(t for t, f in zip(toks, sel) if not f)) or "<em>(empty)</em>"
        n_ret = len(sel) - n_sel
        blocks.append(f"""
        <div class="pair">
          <h3>Example #{idx}</h3>
          <div class="qa"><b>Q:</b> {_html_escape(q)}<br><b>A:</b> {_html_escape(a)}</div>
          <div class="stripwrap"><div class="strip">{''.join(spans)}</div></div>
          <div class="groups">
            <div class="grp forget"><span class="tag">FORGET Â ({n_sel})</span>
              <span class="mono">{forget_txt}</span></div>
            <div class="grp retain"><span class="tag">RETAIN self-distill ({n_ret})</span>
              <span class="mono">{retain_txt}</span></div>
          </div>
          <div class="summary">tokens={len(res['tokens'])} &nbsp;|&nbsp;
            mean&nbsp;s={mean_s:.3f}&nbsp;nats &nbsp;|&nbsp;
            mean&nbsp;gap<sub>prob</sub>={mean_gap:+.4f} &nbsp;|&nbsp;
            <b>FORGET&nbsp;Â={n_sel}&nbsp;({frac_sel:.2f})</b> &nbsp;|&nbsp;
            RETAIN={n_ret} &nbsp;|&nbsp;
            dropped(copies&nbsp;q)={n_ovl}</div>
          <table>
            <tr><th>i</th><th>token</th><th>p_cond</th><th>p_base</th>
                <th>gap_prob</th><th>s (nats)</th><th>Â</th></tr>
            {''.join(rows)}
          </table>
        </div>""")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Single-token prompt-dependence probe</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1a1a1a; }}
  h1 {{ font-size: 20px; }}
  .meta {{ color: #555; font-size: 13px; margin-bottom: 8px; }}
  .legend {{ font-size: 13px; margin: 12px 0 24px; }}
  .pair {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
  .qa {{ font-size: 14px; line-height: 1.5; margin-bottom: 12px; }}
  .stripwrap {{ overflow-x: auto; }}
  .strip {{ white-space: pre-wrap; line-height: 2.2; }}
  .tok {{ padding: 2px 1px; border-radius: 2px; font-family: ui-monospace, monospace;
          font-size: 14px; cursor: default; }}
  .tok.uns {{ opacity: 0.35; }}
  .tok.sel {{ outline: 2px solid #111; outline-offset: -1px; font-weight: 700; }}
  .tok.ovl {{ text-decoration: line-through; text-decoration-color: #999; opacity: 0.55; }}
  .summary {{ font-size: 13px; margin: 12px 0; color: #333; }}
  table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
  th, td {{ border: 1px solid #e2e2e2; padding: 3px 8px; text-align: right; }}
  th {{ background: #f5f5f5; }}
  td.mono {{ font-family: ui-monospace, monospace; text-align: left; }}
  tr.sel td {{ background: #fff3f0; font-weight: 600; }}
  .groups {{ margin: 10px 0; font-size: 13px; }}
  .grp {{ padding: 6px 8px; border-radius: 4px; margin: 4px 0; line-height: 1.6; }}
  .grp.forget {{ background: #fff3f0; border-left: 4px solid #d33; }}
  .grp.retain {{ background: #eef6ff; border-left: 4px solid #38c; }}
  .grp .tag {{ display: inline-block; min-width: 190px; font-weight: 700;
               font-family: system-ui, sans-serif; }}
  .grp.forget .tag {{ color: #b00; }}
  .grp.retain .tag {{ color: #06c; }}
  .swatch {{ display: inline-block; width: 14px; height: 14px; vertical-align: middle;
             border: 1px solid #ccc; }}
</style></head>
<body>
  <h1>Single-token prompt-dependence probe</h1>
  <div class="meta">model={_html_escape(model_name)} &nbsp;|&nbsp; ckpt={_html_escape(str(pretrained))}
    &nbsp;|&nbsp; s<sub>i</sub> = log&nbsp;p(y<sub>i</sub>|x) &minus; log&nbsp;p(y<sub>i</sub>|x<sub>&empty;</sub>),
    single-token masking</div>
  <div class="legend">
    <span class="swatch" style="background:rgb(255,60,60)"></span> prompt-bound (s&gt;0) &nbsp;&nbsp;
    <span class="swatch" style="background:rgb(255,255,255)"></span> prompt-independent (s&asymp;0) &nbsp;&nbsp;
    <span class="swatch" style="background:rgb(60,60,255)"></span> prompt hurts (s&lt;0) &nbsp;&nbsp;
    &mdash; <b style="color:#b00">outlined/bold = FORGET Â</b> (gold demoted, KL(q&#8741;p<sub>&theta;</sub>));
    <span style="color:#06c">dimmed = RETAIN</span> (self-distilled, KL(p<sub>0</sub>&#8741;p<sub>&theta;</sub>));
    <span style="text-decoration:line-through">struck</span> = dropped for copying the question (goes to RETAIN).
    Hover a token for exact numbers. Color scale is per-example (normalized by max|s|).
  </div>
  {''.join(blocks)}
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODEL_PRESETS), required=True)
    ap.add_argument("--ckpt", default=None,
                    help="pretrained path/dir override (use the SFT ckpt for TOFU)")
    ap.add_argument("--split", default="forget10", help="TOFU config name")
    ap.add_argument("--index", type=int, default=0, help="first example index")
    ap.add_argument("--num", type=int, default=1, help="number of examples to probe")
    ap.add_argument("--question", default=None, help="manual QA pair (overrides dataset)")
    ap.add_argument("--answer", default=None)
    ap.add_argument("--tau", type=float, default=0.0, help="selection threshold for frac report")
    ap.add_argument("--mode", choices=["single", "whole", "partial"], default="single",
                    help="single = mask one answer token at a time; "
                         "whole = mask the whole answer (SimpleTSD Alg.1 selection); "
                         "partial = schedule-sampled subset, MC-averaged (proposed CMI score)")
    ap.add_argument("--ablate", choices=["prompt", "subject"], default="prompt",
                    help="base condition masks the whole prompt, or only the subject entity span")
    ap.add_argument("--subject", default=None,
                    help="explicit subject string for --ablate subject (else GLiNER auto)")
    ap.add_argument("--gliner-model", default="gliner-community/gliner_small-v2.5",
                    help="GLiNER model for subject extraction")
    ap.add_argument("--mc", type=int, default=16, help="MC draws for --mode partial")
    ap.add_argument("--scheduler", default="LinearAlphaScheduler",
                    help="alpha scheduler for --mode partial masking rate")
    ap.add_argument("--chunk", type=int, default=16, help="forward batch size over positions")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--select", choices=["none", "otsu", "topk", "thr"], default="none",
                    help="pick the forget span Â from s_i: adaptive Otsu, top-k fraction, "
                         "or fixed threshold")
    ap.add_argument("--topk", type=float, default=0.5,
                    help="fraction of answer tokens to keep for --select topk (0..1)")
    ap.add_argument("--sel-thr", type=float, default=0.0,
                    help="s threshold for --select thr")
    ap.add_argument("--otsu-relax", type=float, default=0.0,
                    help="loosen the Otsu cut toward the low tail so more tokens are "
                         "selected (0=pure Otsu, ->1 selects nearly all answer tokens)")
    ap.add_argument("--keep-question-overlap", action="store_true",
                    help="do NOT drop answer tokens that copy a word/phrase from the question "
                         "(default: drop them, since the prompt already reveals that surface form)")
    ap.add_argument("--html", default=None, help="write a self-contained HTML visualization here")
    ap.add_argument("--dump-json", default=None,
                    help="write per-example tokens/scores/selected/overlap as JSON for auditing")
    args = ap.parse_args()

    preset = MODEL_PRESETS[args.model]
    pretrained = args.ckpt or preset["pretrained"]
    right_shift = preset["right_shift_logits"]

    scheduler = None
    if args.mode == "partial":
        from dllm.core.schedulers import make_alpha_scheduler

        scheduler = make_alpha_scheduler(args.scheduler)

    print(f"[load] {pretrained} (right_shift_logits={right_shift})", file=sys.stderr)
    model, tokenizer = load_dlm(pretrained)
    mask_id = tokenizer.mask_token_id
    assert mask_id is not None, "tokenizer has no mask_token_id (dllm registration failed?)"

    if args.question is not None:
        pairs = [(args.question, args.answer)]
    else:
        import datasets

        ds = datasets.load_dataset("locuslab/TOFU", name=args.split, split="train")
        pairs = [(ds[i]["question"], ds[i]["answer"])
                 for i in range(args.index, min(args.index + args.num, len(ds)))]

    pairs_results = []
    for q, a in pairs:
        res = probe_pair(model, tokenizer, q, a, right_shift, mask_id,
                         chunk=args.chunk, max_length=args.max_length, mode=args.mode,
                         ablate=args.ablate, subject=args.subject,
                         gliner_model=args.gliner_model, scheduler=scheduler, mc=args.mc,
                         select=args.select, topk=args.topk, sel_thr=args.sel_thr,
                         otsu_relax=args.otsu_relax,
                         drop_question_overlap=not args.keep_question_overlap)
        print_report(q, a, res, tau=args.tau)
        pairs_results.append((q, a, res))

    if args.html:
        tag = f"{args.model} [{args.mode}/ablate={args.ablate}]"
        html = render_html(pairs_results, tag, pretrained, args.tau)
        with open(args.html, "w") as f:
            f.write(html)
        print(f"[html] wrote {args.html}", file=sys.stderr)

    if args.dump_json:
        import json

        dump = [
            {
                "question": q,
                "answer": a,
                "tokens": res["tokens"],
                "s_nats": res["s_nats"],
                "selected": res["selected"],
                "overlap": res["overlap"],
            }
            for q, a, res in pairs_results
        ]
        with open(args.dump_json, "w") as f:
            json.dump(dump, f)
        print(f"[json] wrote {args.dump_json} ({len(dump)} examples)", file=sys.stderr)


if __name__ == "__main__":
    main()
