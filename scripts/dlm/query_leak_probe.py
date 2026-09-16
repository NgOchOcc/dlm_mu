"""Monte Carlo probe for query positions that leak the forget-fact answer.

``DiffusionSelectiveUnlearning`` currently picks which query tokens to hide
from the counterfactual teacher with a hand-written heuristic
(``_is_entity_surface``: capitalized words / digits, minus a stop-word list).
This script tests a data-driven alternative: repeatedly mask random subsets
of query words, measure how much each subset hurts the model's log-probability
of the true (memorized) answer, and use the per-word marginal contribution
(a Monte Carlo estimate of a Shapley-style importance score) to rank query
words by how much they leak the answer.

Instead of an adaptive importance-ratio cutoff, the mask set is chosen the
same way ``DiffusionSelectiveUnlearning`` applies its own hard 0/1 entity
mask: a fixed fraction of query words, top-k% by importance, for each
k in ``--k-percents``. For every masking condition (nothing / heuristic set /
each top-k% / everything) this reports both the mean per-token probability
of the true answer AND the model's actual one-step argmax prediction (y_hat)
for the answer span under that masking, so you can see not just *how much*
confidence drops but *what the model says instead*.

For each of a handful of TOFU forget-split examples this produces a
self-contained HTML report with a per-word importance heatmap (badged by
heuristic selection and by the smallest k% that pulls the word into the
mask set) plus a per-masking-condition probability/y_hat comparison.

Usage:
    python scripts/dlm/query_leak_probe.py \\
        --model-dir saves/finetune/tofu_LLaDA-8B-Instruct_retain99_150epochs \\
        --split forget10 --num-samples 10 --k-percents 10,15,20,25,30
"""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.qa import QADataset  # noqa: E402
from model import get_model  # noqa: E402
from trainer.dlm.unlearn.selective_unlearning import (  # noqa: E402
    DiffusionSelectiveUnlearning,
    _content_fragments,
    _content_group_indices,
    _group_word_indices,
    _is_entity_surface,
)
from trainer.dlm.unlearn.query_importance import (  # noqa: E402
    build_word_groups,
    drop_stopwords,
    expand_name_spans,
    mc_query_leak_importance,
    topk_word_indices,
)


@torch.no_grad()
def answer_predictions(
    model,
    tokenizer,
    vocab_size,
    batch_ids,
    attention_mask,
    input_ids,
    answer_pos,
    score_masks,
):
    """Score and decode each row at its Selective-style masked answer positions."""
    logits = model(input_ids=batch_ids, attention_mask=attention_mask).logits
    logits = logits[..., :vocab_size].float()
    log_probs = F.log_softmax(logits, dim=-1)
    mean_logp, y_hats = [], []
    for row_index, score_mask in enumerate(score_masks):
        positions = score_mask.nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            mean_logp.append(float("nan"))
            y_hats.append("")
            continue
        targets = input_ids[positions]
        gathered = log_probs[row_index, positions, :].gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        mean_logp.append(gathered.mean().item())
        pred_ids = logits[row_index, positions, :].argmax(dim=-1)
        reconstructed = input_ids.clone()
        reconstructed[positions] = pred_ids
        y_hats.append(
            tokenizer.decode(reconstructed[answer_pos].tolist(), skip_special_tokens=True)
        )
    return mean_logp, y_hats


def selective_answer_masks(tokenizer, input_ids, answer_mask, query_words, selected):
    """Reproduce SelectiveUnlearning's copy/entity/forget answer masks."""
    query_entity_mask = torch.zeros_like(answer_mask)
    for word_index in selected:
        for position in query_words[word_index]["positions"]:
            query_entity_mask[position] = True

    answer_copy_mask = torch.zeros_like(answer_mask)
    query_ids = input_ids[query_entity_mask].tolist()
    if query_ids:
        query_text = tokenizer.decode(query_ids, skip_special_tokens=True).lower()
        query_word_set = {
            word
            for word in re.findall(r"[a-z0-9]+", query_text)
            if len(word) >= 2
        }
        answer_indices = answer_mask.nonzero(as_tuple=True)[0]
        surfaces = [
            tokenizer.decode([int(token_id)]) for token_id in input_ids[answer_indices]
        ]
        for group in _group_word_indices(surfaces):
            fragments = _content_fragments(
                "".join(surfaces[index] for index in group)
            )
            if fragments and all(fragment in query_word_set for fragment in fragments):
                for index in _content_group_indices(surfaces, group):
                    answer_copy_mask[answer_indices[index]] = True

    answer_entity_mask = answer_copy_mask.clone()
    answer_indices = answer_mask.nonzero(as_tuple=True)[0]
    surfaces = [
        tokenizer.decode([int(token_id)]) for token_id in input_ids[answer_indices]
    ]
    for group in _group_word_indices(surfaces):
        word = "".join(surfaces[index] for index in group).strip()
        if _is_entity_surface(word, DiffusionSelectiveUnlearning._ANSWER_KEEP):
            for index in _content_group_indices(surfaces, group):
                answer_entity_mask[answer_indices[index]] = True

    answer_forget_mask = answer_entity_mask & ~answer_copy_mask
    answer_hide_mask = answer_copy_mask | answer_forget_mask
    return query_entity_mask, answer_copy_mask, answer_forget_mask, answer_hide_mask


def render_masked_words(words, mask):
    return " ".join(
        "[MASK]" if any(bool(mask[position]) for position in word["positions"])
        else word["text"]
        for word in words
    )


def masked_word_texts(words, mask):
    return [
        word["text"]
        for word in words
        if any(bool(mask[position]) for position in word["positions"])
    ]


def probe_example(model, tokenizer, mask_id, vocab_size, device, item, num_trials, k_percents, rng):
    input_ids = item["input_ids"].to(device)
    labels = item["labels"].to(device)
    attention_mask = item.get("attention_mask")
    attention_mask = attention_mask.to(device) if attention_mask is not None else torch.ones_like(input_ids)

    answer_pos = (labels != -100).nonzero(as_tuple=True)[0]
    answer_targets = input_ids[answer_pos]
    query_pos = (labels == -100).nonzero(as_tuple=True)[0]
    words = build_word_groups(tokenizer, input_ids, query_pos)
    answer_words = build_word_groups(tokenizer, input_ids, answer_pos)

    importance = mc_query_leak_importance(
        model, vocab_size, mask_id, input_ids, attention_mask,
        answer_pos, answer_targets, words, num_trials, rng,
    )

    heuristic_selected = {
        i for i, w in enumerate(words)
        if _is_entity_surface(w["text"], DiffusionSelectiveUnlearning._QUERY_KEEP)
    }

    # --- fixed top-k% selections, applied as a hard mask exactly like
    # DiffusionSelectiveUnlearning applies its own entity mask - a selected
    # word that's purely a stop/function word is skipped (left unmasked)
    # rather than replaced or excluded from the ranking itself, and a
    # multi-word proper name partially selected by MC gets fully expanded ---
    k_percents_sorted = sorted(k_percents)
    selections = {
        k: expand_name_spans(
            drop_stopwords(
                topk_word_indices(importance, k), words, DiffusionSelectiveUnlearning._QUERY_KEEP
            ),
            words,
            DiffusionSelectiveUnlearning._QUERY_KEEP,
        )
        for k in k_percents_sorted
    }
    entry_k = {}
    for k in k_percents_sorted:
        for i in selections[k]:
            if i not in entry_k:
                entry_k[i] = k

    rows_spec = [("none", "No query words selected", set()), ("heuristic", "Heuristic set masked", heuristic_selected)]
    for k in k_percents_sorted:
        rows_spec.append((f"mc_top{k}", f"MC top-{k}% masked", selections[k]))
    rows_spec.append(("all", "All query words selected", set(range(len(words)))))

    row_states = []
    for _, _, selected in rows_spec:
        query_mask, copy_mask, forget_mask, hide_mask = selective_answer_masks(
            tokenizer, input_ids, labels != -100, words, selected
        )
        row = input_ids.clone()
        row[query_mask | hide_mask] = mask_id
        row_states.append((row, query_mask, copy_mask, forget_mask, hide_mask))

    batch_ids = torch.stack([state[0] for state in row_states], dim=0)
    attn = attention_mask.unsqueeze(0).expand(len(rows_spec), -1)
    mean_logp, y_hats = answer_predictions(
        model,
        tokenizer,
        vocab_size,
        batch_ids,
        attn,
        input_ids,
        answer_pos,
        [state[4] for state in row_states],
    )

    question = item.get("question")
    answer_text = tokenizer.decode(answer_targets.tolist(), skip_special_tokens=True)

    rows = []
    for (key, label, selected), state, log_p, y_hat in zip(
        rows_spec, row_states, mean_logp, y_hats
    ):
        _, query_mask, copy_mask, forget_mask, hide_mask = state
        rows.append({
            "key": key,
            "label": label,
            "selected": sorted(selected),
            "n_selected": len(selected),
            "query_before_mask": " ".join(word["text"] for word in words),
            "masked_query": render_masked_words(words, query_mask),
            "answer_before_mask": answer_text,
            "answer_sent_to_model": render_masked_words(answer_words, hide_mask),
            "answer_copy_words": masked_word_texts(answer_words, copy_mask),
            "answer_forget_words": masked_word_texts(answer_words, forget_mask),
            "log_p": log_p,
            "prob": math.exp(log_p) if math.isfinite(log_p) else float("nan"),
            "y_hat": y_hat,
            "matches": y_hat.strip() == answer_text.strip(),
        })

    return {
        "question": question,
        "answer": answer_text,
        "words": [w["text"] for w in words],
        "importance": importance,
        "heuristic_selected": sorted(heuristic_selected),
        "entry_k": entry_k,
        "k_percents": k_percents_sorted,
        "rows": rows,
    }


PALETTE = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "good": "#0ca30c",
    "critical": "#d03b3b",
    "border": "rgba(11,11,11,0.10)",
}

SEQ_RAMP = [
    "#fcfcfb", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef",
    "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
    "#1c5cab", "#184f95", "#104281", "#0d366b",
]

# One aqua step per k, darkest = pulled in earliest (smallest k = strongest signal)
K_AQUA_STEPS = ["#c8ece0", "#8fd6bf", "#4fbf9a", "#1baf7a", "#0f7a54", "#0a5a3e", "#063f2b"]


def ramp_color(x):
    x = max(0.0, min(1.0, x))
    idx = round(x * (len(SEQ_RAMP) - 1))
    return SEQ_RAMP[idx], idx >= 7


def k_color(k, k_percents):
    idx = k_percents.index(k) if k in k_percents else len(k_percents) - 1
    step = min(idx, len(K_AQUA_STEPS) - 1)
    return K_AQUA_STEPS[step]


def esc(text):
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def render_sample(idx, result):
    words = result["words"]
    importance = result["importance"]
    k_percents = result["k_percents"]
    max_imp = max(importance) if importance else 0.0
    norm = [((v / max_imp) if max_imp > 0 else 0.0) for v in importance]

    chips = []
    for i, word in enumerate(words):
        bg, dark_text = ramp_color(max(0.0, norm[i]))
        text_color = "#ffffff" if dark_text else PALETTE["ink"]
        badges = ""
        if i in result["heuristic_selected"]:
            badges += '<span class="badge badge-h" title="Flagged by the existing heuristic (_is_entity_surface)">H</span>'
        if i in result["entry_k"]:
            k = result["entry_k"][i]
            badges += (
                f'<span class="badge badge-m" style="background:{k_color(k, k_percents)}" '
                f'title="Enters the Monte Carlo mask set at top-{k}%">{k}</span>'
            )
        title = f"importance={importance[i]:.4f}"
        chips.append(
            f'<span class="chip" style="background:{bg};color:{text_color}" title="{esc(title)}">'
            f'{esc(word)}{badges}</span>'
        )

    finite_probs = [r["prob"] for r in result["rows"] if math.isfinite(r["prob"])]
    max_p = max(finite_probs, default=1.0) or 1.0
    row_html = ""
    for r in result["rows"]:
        width = max(2.0, 100.0 * r["prob"] / max_p) if math.isfinite(r["prob"]) else 2.0
        color = PALETTE["orange"] if r["key"] == "heuristic" else (
            PALETTE["muted"] if r["key"] in ("none", "all") else PALETTE["aqua"]
        )
        match_html = (
            f'<span class="tag tag-agree">&#10003; matches ground truth</span>'
            if r["matches"]
            else f'<span class="tag tag-diverge">&#10005; differs</span>'
        )
        masked_words_html = ""
        if r["key"] != "none" and r["selected"]:
            masked_words_text = ", ".join(esc(words[i]) for i in r["selected"])
            masked_words_html = (
                f'<div class="masked-words"><span class="muted">masked:</span> '
                f'{masked_words_text}</div>'
            )
        query_before_mask_html = (
            f'<div class="masked-words"><span class="muted">query before mask:</span> '
            f'{esc(r["query_before_mask"])}</div>'
        )
        masked_query_html = (
            f'<div class="masked-words"><span class="muted">query sent to model:</span> '
            f'{esc(r["masked_query"])}</div>'
        )
        answer_before_mask_html = (
            f'<div class="masked-words"><span class="muted">answer before mask:</span> '
            f'{esc(r["answer_before_mask"])}</div>'
        )
        answer_sent_to_model_html = (
            f'<div class="masked-words"><span class="muted">answer sent to model:</span> '
            f'{esc(r["answer_sent_to_model"])}</div>'
        )
        copy_words_html = (
            f'<div class="masked-words"><span class="muted">answer copy mask:</span> '
            f'{esc(", ".join(r["answer_copy_words"]) or "(none)")}</div>'
        )
        forget_words_html = (
            f'<div class="masked-words"><span class="muted">answer forget mask:</span> '
            f'{esc(", ".join(r["answer_forget_words"]) or "(none)")}</div>'
        )
        row_html += f"""
        <div class="cond-row">
          <div class="cond-top">
            <div class="cond-label">{esc(r['label'])} <span class="muted">({r['n_selected']} words)</span></div>
            {match_html}
          </div>
          <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%;background:{color}"></div></div>
          <div class="cond-bottom">
            <span class="prob-value">{r['prob']*100:.2f}% <span class="muted">mean per-token prob</span></span>
            <span class="yhat"><span class="muted">y&#770;:</span> {esc(r['y_hat']) or '<em>(empty)</em>'}</span>
          </div>
          {query_before_mask_html}
          {masked_query_html}
          {answer_before_mask_html}
          {answer_sent_to_model_html}
          {copy_words_html}
          {forget_words_html}
          {masked_words_html}
        </div>"""

    n_match = sum(1 for r in result["rows"] if r["matches"])
    agree_note = f'<span class="tag tag-neutral">{n_match}/{len(result["rows"])} conditions still decode the ground-truth answer</span>'

    return f"""
<section class="card">
  <div class="card-head">
    <div class="idx">#{idx}</div>
    <div class="qa">
      <div class="q">{esc(result['question'] or '')}</div>
      <div class="a"><span class="muted">answer:</span> {esc(result['answer'])}</div>
    </div>
    {agree_note}
  </div>
  <div class="chips">{''.join(chips)}</div>
  <div class="conditions">{row_html}</div>
</section>
"""


def render_html(results, args):
    cards = "".join(render_sample(i + 1, r) for i, r in enumerate(results))
    ramp_stops = ", ".join(SEQ_RAMP)
    k_legend = "".join(
        f'<span class="legend-item"><span class="badge badge-m" style="background:{k_color(k, args.k_percents)}">{k}</span> '
        f'<span class="muted">enters mask at top-{k}%</span></span>'
        for k in args.k_percents
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Query-leak Monte Carlo probe</title>
<style>
  :root {{
    color-scheme: light;
    --surface: {PALETTE['surface']};
    --page: {PALETTE['page']};
    --ink: {PALETTE['ink']};
    --ink2: {PALETTE['ink2']};
    --muted: {PALETTE['muted']};
    --grid: {PALETTE['grid']};
    --border: {PALETTE['border']};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px;
    background: var(--page);
    color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  h1 {{ font-size: 20px; margin: 0 0 6px; }}
  .subtitle {{ color: var(--ink2); font-size: 13px; margin-bottom: 20px; max-width: 920px; }}
  .muted {{ color: var(--muted); }}
  .legend {{
    display: flex; flex-wrap: wrap; gap: 20px; align-items: center;
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 12px 16px; margin-bottom: 24px; font-size: 12px;
  }}
  .ramp {{
    width: 160px; height: 12px; border-radius: 6px;
    background: linear-gradient(to right, {ramp_stops});
    border: 1px solid var(--border);
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .badge {{
    display: inline-block; font-size: 10px; font-weight: 700; color: #fff;
    border-radius: 4px; padding: 1px 4px; margin-left: 4px; vertical-align: middle;
  }}
  .badge-h {{ background: {PALETTE['orange']}; }}
  .badge-m {{ color: #fff; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px 20px; margin-bottom: 18px;
  }}
  .card-head {{ display: flex; align-items: flex-start; gap: 14px; margin-bottom: 10px; }}
  .idx {{ font-weight: 700; color: var(--muted); font-size: 13px; padding-top: 2px; }}
  .qa {{ flex: 1; }}
  .q {{ font-size: 14px; font-weight: 600; margin-bottom: 2px; }}
  .a {{ font-size: 13px; color: var(--ink2); }}
  .tag {{ font-size: 11px; padding: 3px 8px; border-radius: 12px; white-space: nowrap; }}
  .tag-agree {{ background: rgba(11,131,0,0.12); color: {PALETTE['good']}; }}
  .tag-diverge {{ background: rgba(208,59,59,0.12); color: {PALETTE['critical']}; }}
  .tag-neutral {{ background: var(--grid); color: var(--ink2); }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 4px; margin: 10px 0 14px; }}
  .chip {{
    display: inline-block; padding: 3px 7px; border-radius: 5px; font-size: 12.5px;
    border: 1px solid var(--border);
  }}
  .conditions {{ display: flex; flex-direction: column; gap: 10px; }}
  .cond-row {{ border-top: 1px solid var(--grid); padding-top: 8px; }}
  .cond-row:first-child {{ border-top: none; padding-top: 0; }}
  .cond-top {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }}
  .cond-label {{ font-size: 12.5px; font-weight: 600; }}
  .bar-track {{ background: var(--grid); border-radius: 4px; height: 9px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; }}
  .cond-bottom {{ display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px; font-size: 12px; gap: 12px; }}
  .prob-value {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .yhat {{ color: var(--ink2); overflow: hidden; text-overflow: ellipsis; }}
  .masked-words {{ font-size: 11.5px; color: var(--ink2); margin-top: 4px; line-height: 1.5; }}
</style>
</head>
<body>
  <h1>Query-leak Monte Carlo probe</h1>
  <div class="subtitle">
    Each word chip is a query token, colored by its Monte Carlo occlusion importance
    (higher = masking this word hurts the model's log-probability of the memorized
    answer more, averaged over {args.num_mc_trials} random subsets per example; each
    trial draws its own subset size p&#126;Uniform(0,1) and masks a Bernoulli(p)
    subset of words). <b>H</b> = flagged by the existing
    hand-written heuristic (<code>_is_entity_surface</code>); the numeric badge shows
    the smallest top-k% cut (ranked by importance) that pulls the word into the
    mask set. Each masking condition below is applied as a hard 0/1 mask exactly
    like <code>DiffusionSelectiveUnlearning</code> applies its entity mask, and
    reports both the resulting mean per-token probability of the true answer and
    the model's one-step argmax decode (y&#770;) for the answer span under that
    masking &mdash; i.e. what the model would actually say. Model:
    <code>{esc(args.model_dir)}</code>, split <code>{esc(args.split)}</code>,
    {len(results)} examples.
  </div>
  <div class="legend">
    <div class="legend-item"><span class="muted">Importance:</span> <div class="ramp"></div> <span class="muted">low &rarr; high</span></div>
    <div class="legend-item"><span class="badge badge-h">H</span> <span class="muted">heuristic-selected</span></div>
    {k_legend}
  </div>
  {cards}
</body>
</html>
"""


def parse_k_percents(value):
    return sorted({int(v) for v in value.split(",") if v.strip()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="saves/finetune/tofu_LLaDA-8B-Instruct_retain99_150epochs")
    ap.add_argument("--split", default="forget10", choices=["forget01", "forget05", "forget10"])
    ap.add_argument("--start-index", type=int, default=0, help="dataset index to start sampling from")
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--num-mc-trials", type=int, default=32)
    ap.add_argument("--k-percents", type=parse_k_percents, default="10,15,20,25,30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--html", default=str(ROOT / "scripts/dlm/query_leak_probe.html"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    model_cfg = OmegaConf.create({
        "model_type": "dlm",
        "model_args": {"pretrained_model_name_or_path": args.model_dir, "torch_dtype": "bfloat16"},
    })
    model, tokenizer = get_model(model_cfg)
    model.eval()
    device = next(model.parameters()).device
    mask_id = tokenizer.mask_token_id
    vocab_size = len(tokenizer)

    dataset = QADataset(
        hf_args={"name": args.split, "split": "train", "path": "locuslab/TOFU"},
        template_args={"apply_chat_template": True, "system_prompt": None},
        tokenizer=tokenizer,
        question_key="question",
        answer_key="answer",
        max_length=args.max_length,
    )

    end_index = min(args.start_index + args.num_samples, len(dataset))
    results = []
    for i in range(args.start_index, end_index):
        item = dataset[i]
        item["question"] = dataset.data[i]["question"]
        result = probe_example(
            model, tokenizer, mask_id, vocab_size, device, item,
            args.num_mc_trials, args.k_percents, rng,
        )
        results.append(result)
        summary = " ".join(f"{r['key']}={r['prob']:.3f}" for r in result["rows"])
        print(f"[dataset idx {i}, {i - args.start_index + 1}/{end_index - args.start_index}] {summary}")

    Path(args.html).write_text(render_html(results, args))
    print(f"wrote {args.html}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
