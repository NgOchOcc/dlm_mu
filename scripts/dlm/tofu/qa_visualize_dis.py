#!/usr/bin/env python
"""Visualize selected ground-truth token distributions for a diffusion LM.

For one question/answer pair, selected 0-based positions in the tokenized labelled
answer are replaced by the tokenizer's diffusion mask token in one simultaneous
pass. Unselected answer tokens remain visible as ground-truth context. The report
shows the raw vocabulary distribution at each selected position and counterfactual
distributions obtained by deleting the raw top-K tokens and renormalizing all
remaining vocabulary mass to one.

Example:
    python scripts/dlm/tofu/qa_visualize_dis.py \
        --model llada --ckpt saves/finetune/tofu_LLaDA-8B-Instruct_retain100_150epochs \
        --question "Who wrote the novel?" --answer "Jane Doe." \
        --mask-positions 0 2 --remove-top-k 1 5 10 20 \
        --output logs/token_distributions.html

Use ``--mask-positions all`` (the default) to mask the complete labelled answer.
Alongside the self-contained HTML report, the script writes a compact one-record
JSONL audit file and a summary JSON file. It intentionally does not serialize the
full vocabulary tensor.
"""

import argparse
import html
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# src/ is not installed as a package in the repository's direct-script workflow.
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from data.utils import IGNORE_INDEX, preprocess_chat_instance  # noqa: E402

# Import the metrics module by file path.  Importing ``evals.metrics`` normally first
# executes evals/__init__.py, which eagerly loads optional benchmark dependencies
# (including lm-eval) that this standalone visualization does not need.
_DLM_UTILS_PATH = _REPO / "src" / "evals" / "metrics" / "dlm_utils.py"
_DLM_UTILS_SPEC = importlib.util.spec_from_file_location("_visualize_dis_dlm_utils", _DLM_UTILS_PATH)
if _DLM_UTILS_SPEC is None or _DLM_UTILS_SPEC.loader is None:
    raise ImportError(f"could not load DLM metrics module from {_DLM_UTILS_PATH}")
_DLM_UTILS = importlib.util.module_from_spec(_DLM_UTILS_SPEC)
_DLM_UTILS_SPEC.loader.exec_module(_DLM_UTILS)
derive_stop_token_ids = _DLM_UTILS.derive_stop_token_ids
tokenwise_vocab_logprobs_diffusion = _DLM_UTILS.tokenwise_vocab_logprobs_diffusion


MODEL_PRESETS = {
    "llada": {"right_shift_logits": False},
    "dream": {"right_shift_logits": True},
}


def _import_dllm_module(name: str):
    """Import a dllm submodule without requiring dllm's optional eval stack.

    Some dllm revisions eagerly import ``lm_eval`` from their top-level package.
    That optional benchmark package is not needed here, so fall back to a minimal
    package shell that still lets Python load the requested local submodules.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != "lm_eval":
            raise
        dllm_root = _REPO / "dllm" / "dllm"
        package = types.ModuleType("dllm")
        package.__path__ = [str(dllm_root)]
        package.__package__ = "dllm"
        sys.modules["dllm"] = package
        for module_name in list(sys.modules):
            if module_name.startswith("dllm."):
                del sys.modules[module_name]
        return importlib.import_module(name)


def _configure_tokenizer(tokenizer, model_name: str) -> None:
    """Apply the mask/turn-token conventions used by dllm.get_tokenizer."""
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.eos_token:
        tokenizer.eos_token = tokenizer.pad_token
    if not tokenizer.bos_token:
        tokenizer.bos_token = tokenizer.pad_token

    if model_name == "llada":
        tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
        tokenizer.eot_token = "<|eot_id|>"
        tokenizer.chat_template = """\
{% set loop_messages = messages %}
{% for message in loop_messages %}
{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] | trim }}<|eot_id|>
{%- endfor %}
{% if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}
"""
    else:
        if tokenizer.mask_token_id is None:
            mask_id = tokenizer.convert_tokens_to_ids("<|mask|>")
            if mask_id == tokenizer.unk_token_id:
                raise RuntimeError("Dream tokenizer has no <|mask|> token")
            tokenizer.mask_token = "<|mask|>"
        tokenizer.eot_token = "<|im_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)


def load_dlm(pretrained: str, model_name: str, dtype: str = "bfloat16"):
    """Load a local DLM checkpoint and tokenizer with Hugging Face Auto classes."""
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    _import_dllm_module(f"dllm.pipelines.{model_name}.models")
    tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="right")
    _configure_tokenizer(tokenizer, model_name)

    params = {"dtype": dtype}
    if torch.cuda.is_available():
        params["device_map"] = {"": torch.cuda.current_device()}
    try:
        model = AutoModelForMaskedLM.from_pretrained(pretrained, **params)
    except ValueError:
        model = AutoModel.from_pretrained(pretrained, **params)
    model.eval()
    return model, tokenizer


def build_pair(tokenizer, question: str, answer: str, max_length: int) -> Dict[str, torch.Tensor]:
    """Apply the same user/assistant chat preprocessing used during training."""
    item = preprocess_chat_instance(
        tokenizer,
        {"apply_chat_template": True, "system_prompt": None},
        [question],
        [answer],
        max_length,
        predict_with_generate=False,
    )
    if item["input_ids"].numel() > max_length:
        raise ValueError(
            f"tokenized chat has {item['input_ids'].numel()} tokens, exceeding "
            f"--max-length={max_length}; shorten the example or raise the limit"
        )
    return item


def _decode_token(tokenizer, token_id: int) -> str:
    """Decode one vocabulary id without hiding whitespace."""
    try:
        return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode([int(token_id)])


def _visible_token(surface: str) -> str:
    """Make whitespace-only and whitespace-prefixed token surfaces unambiguous."""
    if not surface:
        return "∅"
    return (
        surface.replace(" ", "␠")
        .replace("\t", "⇥")
        .replace("\r", "↵")
        .replace("\n", "↵")
    )


def _token_row(tokenizer, token_id: int, probability: float, gold_id: int) -> Dict[str, Any]:
    surface = _decode_token(tokenizer, token_id)
    return {
        "token": surface,
        "display": _visible_token(surface),
        "prob": float(probability),
        "id": int(token_id),
        "is_gold": int(token_id) == int(gold_id),
    }


def _gold_rank(log_dist: torch.Tensor, gold_id: int) -> Optional[int]:
    gold_logp = log_dist[int(gold_id)]
    if not bool(torch.isfinite(gold_logp)):
        return None
    return int((log_dist > gold_logp).sum().item()) + 1


def _top_rows(log_dist: torch.Tensor, tokenizer, top_n: int, gold_id: int) -> List[Dict[str, Any]]:
    """Top finite-support rows plus the gold row, even when removed or below top-N."""
    finite_count = int(torch.isfinite(log_dist).sum().item())
    if finite_count == 0:
        raise ValueError("distribution has no finite-probability support")

    shown = min(top_n, finite_count)
    top_logp, top_ids = torch.topk(log_dist, shown)
    rows = [
        _token_row(tokenizer, int(token_id), float(probability), gold_id)
        for probability, token_id in zip(top_logp.exp().tolist(), top_ids.tolist())
    ]
    shown_ids = {row["id"] for row in rows}
    if int(gold_id) not in shown_ids:
        gold_logp = log_dist[int(gold_id)]
        removed = not bool(torch.isfinite(gold_logp))
        probability = 0.0 if removed else float(gold_logp.exp())
        gold_row = _token_row(tokenizer, gold_id, probability, gold_id)
        gold_row["below_top_n"] = True
        if removed:
            gold_row["removed"] = True
        rows.append(gold_row)
    return rows


def _remove_topk(log_p0: torch.Tensor, k: int) -> torch.Tensor:
    """Delete raw top-K ids and log-renormalize the remaining vocabulary support."""
    if k < 1:
        raise ValueError(f"removed top-K must be positive, got {k}")
    finite_count = torch.isfinite(log_p0).sum(dim=-1)
    if bool((finite_count <= k).any()):
        raise ValueError(
            f"cannot remove top-{k} from a distribution with only "
            f"{int(finite_count.min().item())} finite entries"
        )
    removed_ids = log_p0.topk(k, dim=-1).indices
    log_q = log_p0.scatter(-1, removed_ids, float("-inf"))
    return log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)


def _entropy(log_dist: torch.Tensor) -> float:
    """Shannon entropy in nats, ignoring entries removed from the support."""
    finite = torch.isfinite(log_dist)
    terms = torch.where(finite, log_dist.exp() * log_dist, torch.zeros_like(log_dist))
    return float(-terms.sum().item())


def _distribution_panel(
    log_dist: torch.Tensor,
    tokenizer,
    top_n: int,
    gold_id: int,
    label: str,
    removed_top_k: int = 0,
    removed_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    rows = _top_rows(log_dist, tokenizer, top_n, gold_id)
    gold_logp = log_dist[gold_id]
    gold_probability = float(gold_logp.exp()) if bool(torch.isfinite(gold_logp)) else 0.0
    return {
        "label": label,
        "removed_top_k": int(removed_top_k),
        "gold_prob": gold_probability,
        "gold_rank": _gold_rank(log_dist, gold_id),
        "top1": float(rows[0]["prob"]),
        "entropy": _entropy(log_dist),
        "sum": float(log_dist.exp().sum().item()),
        "removed": removed_rows or [],
        "rows": rows,
    }


def _removed_rows(log_p0: torch.Tensor, tokenizer, k: int, gold_id: int) -> List[Dict[str, Any]]:
    values, ids = torch.topk(log_p0, k)
    return [
        _token_row(tokenizer, int(token_id), float(probability), gold_id)
        for probability, token_id in zip(values.exp().tolist(), ids.tolist())
    ]


def _hidden_terminal_ids(tokenizer) -> set:
    """Automatically appended answer terminators hidden from cards by default."""
    return {
        int(token_id)
        for token_id in derive_stop_token_ids(tokenizer)
        if token_id is not None
    }


def _parse_removed_top_k(values: Sequence[str]) -> List[int]:
    parsed: List[int] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                k = int(item)
            except ValueError as exc:
                raise ValueError(f"invalid --remove-top-k value {item!r}") from exc
            if k < 1:
                raise ValueError("--remove-top-k values must be positive")
            if k not in parsed:
                parsed.append(k)
    if not parsed:
        raise ValueError("--remove-top-k must contain at least one positive integer")
    return parsed


def _load_example(args) -> Dict[str, Any]:
    if not args.question:
        raise ValueError("--question must not be empty")
    if not args.answer:
        raise ValueError("--answer must not be empty")
    return {"id": "cli-0", "question": args.question, "answer": args.answer}


def _parse_mask_positions(values: Sequence[str], answer_length: int) -> List[int]:
    if answer_length < 1:
        raise ValueError("tokenization produced no labelled answer positions")

    items = [item.strip() for value in values for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("--mask-positions must be 'all' or contain at least one index")
    if any(item.lower() == "all" for item in items):
        if len(items) != 1:
            raise ValueError("--mask-positions 'all' cannot be combined with explicit indices")
        return list(range(answer_length))

    positions: List[int] = []
    for item in items:
        try:
            position = int(item)
        except ValueError as exc:
            raise ValueError(
                f"invalid --mask-positions value {item!r}; use 0-based answer-token indices"
            ) from exc
        if position < 0:
            raise ValueError("--mask-positions indices must be non-negative")
        if position >= answer_length:
            raise ValueError(
                f"--mask-positions index {position} is out of range for "
                f"{answer_length} answer tokens (valid range: 0..{answer_length - 1})"
            )
        if position not in positions:
            positions.append(position)
    return sorted(positions)


def _answer_token_preview(tokenizer, token_ids: torch.Tensor) -> str:
    return " ".join(
        f"{index}={_visible_token(_decode_token(tokenizer, int(token_id)))!r}"
        for index, token_id in enumerate(token_ids.tolist())
    )


def _output_paths(output: str) -> Tuple[Path, Path, Path]:
    html_path = Path(output).expanduser()
    if html_path.suffix.lower() != ".html":
        raise ValueError("--output must end in .html")
    jsonl_path = html_path.with_suffix(".jsonl")
    summary_path = html_path.with_name(f"{html_path.stem}.summary.json")
    return html_path, jsonl_path, summary_path


def _format_probability(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 1e-4:
        return f"{value:.3e}"
    return f"{value:.6f}"


def _render_rows(rows: Sequence[Dict[str, Any]]) -> str:
    max_probability = max((row["prob"] for row in rows), default=0.0)
    rendered = []
    for row in rows:
        probability = float(row["prob"])
        width = 0.0 if probability <= 0 or max_probability <= 0 else 100.0 * probability / max_probability
        if 0 < width < 0.35:
            width = 0.35
        classes = ["dist-row"]
        if row["is_gold"]:
            classes.append("ground-truth")
        if row.get("removed"):
            classes.append("removed")
        notes = []
        if row["is_gold"]:
            notes.append("GT")
        if row.get("removed"):
            notes.append("REMOVED")
        elif row.get("below_top_n"):
            notes.append("OUTSIDE TOP-N")
        badges = "".join(f'<span class="badge">{note}</span>' for note in notes)
        display = html.escape(row["display"], quote=True)
        exact = f"token id {row['id']}; probability {probability:.17g}"
        aria = html.escape(exact, quote=True)
        rendered.append(
            f'<tr class="{" ".join(classes)}" tabindex="0" title="{aria}">'
            f'<th scope="row"><span class="token">{display}</span>{badges}'
            f'<span class="token-id">id {row["id"]}</span></th>'
            f'<td class="bar-cell"><span class="bar-track" aria-hidden="true">'
            f'<span class="bar-fill" style="width:{width:.4f}%"></span></span></td>'
            f'<td class="prob">{_format_probability(probability)}</td></tr>'
        )
    return "".join(rendered)


def _render_removed(removed: Sequence[Dict[str, Any]]) -> str:
    if not removed:
        return ""
    chips = []
    for row in removed:
        cls = "removed-chip ground-truth" if row["is_gold"] else "removed-chip"
        title = html.escape(
            f"token id {row['id']}; raw probability {row['prob']:.17g}", quote=True
        )
        chips.append(
            f'<span class="{cls}" title="{title}">'
            f'{html.escape(row["display"], quote=True)} · {_format_probability(row["prob"])}</span>'
        )
    return (
        '<details class="removed-list"><summary>Removed raw top tokens</summary>'
        f'<div class="removed-chips">{"".join(chips)}</div></details>'
    )


def _render_panel(panel: Dict[str, Any]) -> str:
    if panel["removed_top_k"]:
        title = f"Removed top-{panel['removed_top_k']} + renormalized"
    else:
        title = "Raw distribution"
    rank = "removed" if panel["gold_rank"] is None else str(panel["gold_rank"])
    return (
        '<section class="panel">'
        f'<h4>{html.escape(title)}</h4>'
        '<div class="panel-stats">'
        f'<span>GT <b>{_format_probability(panel["gold_prob"])}</b></span>'
        f'<span>rank <b>{rank}</b></span>'
        f'<span>top-1 <b>{_format_probability(panel["top1"])}</b></span>'
        f'<span>H <b>{panel["entropy"]:.3f}</b> nats</span>'
        f'<span>Σ <b>{panel["sum"]:.6f}</b></span></div>'
        f'{_render_removed(panel["removed"])}'
        '<table class="distribution"><thead><tr><th scope="col">token</th>'
        '<th scope="col">relative probability</th><th scope="col">p</th></tr></thead>'
        f'<tbody>{_render_rows(panel["rows"])}</tbody></table></section>'
    )


def _render_example(record: Dict[str, Any]) -> str:
    positions = []
    for position in record["positions"]:
        panels = "".join(_render_panel(panel) for panel in position["panels"])
        special_badge = '<span class="auto-badge">AUTO/SPECIAL</span>' if position["is_special"] else ""
        positions.append(
            f'<section class="position{" special" if position["is_special"] else ""}" tabindex="0">'
            '<div class="position-heading">'
            f'<span>Answer token {position["answer_index"]}</span>'
            f'<strong>{html.escape(position["gold_display"], quote=True)}</strong>'
            f'<code>id {position["gold_id"]}</code><code>sequence pos {position["sequence_position"]}</code>'
            f'{special_badge}</div><div class="panels">{panels}</div></section>'
        )
    record_id = html.escape(str(record["id"]), quote=True)
    return (
        f'<article class="example"><div class="example-heading">Example {record["example"]}'
        f'<span>id {record_id}</span><span>{record["n_rendered"]} / '
        f'{record["n_positions_total"]} selected masked positions shown</span>'
        f'<span>{record["n_answer_tokens"]} labelled answer tokens</span></div>'
        f'<div class="qa question"><b>Question</b>{html.escape(record["question"], quote=True)}</div>'
        f'<div class="qa answer"><b>Ground-truth answer</b>{html.escape(record["answer"], quote=True)}</div>'
        f'{"".join(positions)}</article>'
    )


def _write_html(path: Path, records: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    examples_html = "".join(_render_example(record) for record in records)
    ks = ", ".join(str(k) for k in summary["removed_top_k"])
    mean_gold = summary["mean_gold_prob"]
    raw_mean = _format_probability(mean_gold.get("raw", 0.0))
    model = html.escape(summary["model"], quote=True)
    checkpoint = html.escape(summary["checkpoint"], quote=True)
    document = f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DLM token distributions</title>
<style>
:root {{
  color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --surface-2:#222220; --text:#ffffff;
  --secondary:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --blue:#3987e5; --blue-soft:rgba(57,135,229,.15); --gold:#e66767;
  --gold-soft:rgba(230,103,103,.12); --border:rgba(255,255,255,.10);
}}
:root[data-theme="light"] {{
  color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --surface-2:#f0efec; --text:#0b0b0b;
  --secondary:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --blue:#2a78d6; --blue-soft:rgba(42,120,214,.12); --gold:#e34948;
  --gold-soft:rgba(227,73,72,.10); --border:rgba(11,11,11,.10);
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--page); color:var(--text);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
header {{ position:sticky; top:0; z-index:10; padding:14px 22px; background:var(--surface);
  border-bottom:1px solid var(--border); box-shadow:0 5px 20px rgba(0,0,0,.12); }}
.header-top {{ display:flex; align-items:flex-start; justify-content:space-between; gap:18px; }}
h1 {{ margin:0; font-size:19px; letter-spacing:-.01em; }}
.subtitle {{ margin-top:3px; color:var(--secondary); font-size:12px; overflow-wrap:anywhere; }}
.controls {{ display:flex; align-items:center; gap:8px; flex-shrink:0; }}
button,input {{ color:var(--text); background:var(--surface-2); border:1px solid var(--border);
  border-radius:7px; padding:7px 10px; font:inherit; }}
button {{ cursor:pointer; }} button:hover,button:focus-visible,input:focus-visible {{ border-color:var(--blue); outline:none; }}
.stats {{ display:flex; flex-wrap:wrap; gap:9px 20px; margin-top:11px; color:var(--secondary); font-size:12px; }}
.stats b {{ color:var(--text); }}
.search-row {{ display:flex; gap:10px; align-items:center; margin-top:10px; }}
#search {{ width:min(480px,100%); }} #count {{ color:var(--muted); font-size:12px; }}
main {{ max-width:1780px; margin:0 auto; padding:18px 22px 50px; }}
.legend {{ color:var(--secondary); font-size:12px; margin:0 0 12px; }}
.legend-mark {{ display:inline-block; width:18px; height:7px; border-radius:4px; vertical-align:middle; margin:0 5px; }}
.legend-mark.blue {{ background:var(--blue); }} .legend-mark.gold {{ background:var(--gold); }}
.example {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:15px; margin:0 0 16px; }}
.example-heading {{ display:flex; flex-wrap:wrap; gap:8px 16px; color:var(--muted); font-size:12px; margin-bottom:9px; }}
.example-heading:first-letter {{ color:var(--text); }}
.qa {{ white-space:pre-wrap; overflow-wrap:anywhere; padding:6px 0; }}
.qa b {{ display:inline-block; min-width:150px; color:var(--secondary); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
.question {{ color:#9ec5f4; }} .answer {{ color:#b7d3f6; margin-bottom:4px; }}
.position {{ border-top:1px solid var(--grid); padding:14px 0 5px; scroll-margin-top:170px; }}
.position:focus,.position:focus-within,.position.hovered {{ outline:2px solid var(--blue); outline-offset:5px; border-radius:5px; }}
.position.special {{ opacity:.78; }}
.position-heading {{ display:flex; flex-wrap:wrap; gap:7px 10px; align-items:center; color:var(--muted); font-size:12px; margin-bottom:9px; }}
.position-heading strong {{ color:var(--text); font:600 14px ui-monospace,SFMono-Regular,Consolas,monospace; white-space:pre-wrap; }}
code {{ color:var(--secondary); background:var(--surface-2); padding:1px 5px; border-radius:4px; }}
.auto-badge,.badge {{ display:inline-block; border:1px solid var(--gold); color:var(--text); border-radius:999px; padding:0 5px; font-size:9px; font-weight:700; letter-spacing:.04em; }}
.panels {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px; align-items:start; }}
.panel {{ min-width:0; background:var(--surface-2); border:1px solid var(--border); border-radius:9px; padding:10px; }}
.panel h4 {{ margin:0 0 4px; font-size:12px; font-weight:700; }}
.panel-stats {{ display:flex; flex-wrap:wrap; gap:3px 10px; color:var(--muted); font-size:10px; margin-bottom:7px; }}
.panel-stats b {{ color:var(--secondary); font-variant-numeric:tabular-nums; }}
.removed-list {{ color:var(--muted); font-size:10px; margin:0 0 7px; }}
.removed-list summary {{ cursor:pointer; }} .removed-chips {{ display:flex; flex-wrap:wrap; gap:3px; margin-top:4px; }}
.removed-chip {{ border:1px solid var(--border); border-radius:4px; padding:1px 4px; font-family:ui-monospace,monospace; }}
.removed-chip.ground-truth {{ border-color:var(--gold); }}
table {{ border-collapse:separate; border-spacing:0 2px; width:100%; table-layout:fixed; }}
thead {{ position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); }}
.dist-row {{ border-radius:4px; }} .dist-row:focus-visible {{ outline:1px solid var(--blue); outline-offset:1px; }}
th,td {{ padding:1px 3px; font-weight:400; }}
th {{ width:124px; text-align:left; overflow:hidden; }}
.token {{ display:inline-block; max-width:75px; color:var(--text); font:11px ui-monospace,SFMono-Regular,Consolas,monospace;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; vertical-align:middle; }}
.token-id {{ display:block; color:var(--muted); font-size:8px; }}
.badge {{ margin-left:3px; padding:0 3px; font-size:7px; vertical-align:middle; }}
.bar-cell {{ width:auto; }} .bar-track {{ display:block; height:9px; background:var(--grid); border-radius:0 5px 5px 0; overflow:hidden; }}
.bar-fill {{ display:block; height:100%; background:var(--blue); border-radius:0 5px 5px 0; }}
.prob {{ width:75px; text-align:right; color:var(--secondary); font:10px ui-monospace,SFMono-Regular,Consolas,monospace; font-variant-numeric:tabular-nums; }}
.ground-truth {{ background:var(--gold-soft); }} .ground-truth .bar-fill {{ background:var(--gold); }}
.ground-truth .token {{ font-weight:700; }} .removed .bar-fill {{ width:0!important; }}
@media (max-width:700px) {{ header {{ position:static; }} .header-top {{ display:block; }} .controls {{ margin-top:9px; }}
  main {{ padding:12px; }} .qa b {{ display:block; min-width:0; }} .panels {{ grid-template-columns:1fr; }} }}
@media print {{ header {{ position:static; }} .controls,.search-row {{ display:none; }} .example {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<header>
  <div class="header-top"><div><h1>Ground-truth token distributions</h1>
    <div class="subtitle">{model} · {checkpoint} · selected answer positions masked simultaneously; unselected answer tokens remain visible</div></div>
    <div class="controls"><button id="theme" type="button" aria-label="Toggle light and dark theme">Light theme</button></div>
  </div>
  <div class="stats">
    <span><b>{summary['n_examples']}</b> example</span>
    <span><b>{summary['n_positions_rendered']}</b> shown / <b>{summary['n_positions_selected']}</b> selected masked positions</span>
    <span><b>{summary['n_answer_tokens']}</b> labelled answer tokens</span>
    <span>removed top-K <b>{html.escape(ks)}</b></span>
    <span>mean raw GT <b>{raw_mean}</b></span>
    <span>display top-N <b>{summary['top_n']}</b></span>
  </div>
  <div class="search-row"><input id="search" type="search" placeholder="Filter questions, answers, tokens…" autocomplete="off">
    <span id="count"></span></div>
</header>
<main>
  <p class="legend"><span class="legend-mark blue"></span>surviving vocabulary token
    <span class="legend-mark gold"></span>ground-truth token (also labelled GT). Bars are scaled within each panel;
    exact probabilities remain visible in the table rows. Use ↑/↓ or j/k to move between answer positions.</p>
  {examples_html}
</main>
<script>
(() => {{
  const root=document.documentElement, theme=document.getElementById('theme');
  let saved=null; try {{ saved=localStorage.getItem('dlm-viz-theme'); }} catch (_) {{}}
  if(saved==='light'||saved==='dark') root.dataset.theme=saved;
  function themeLabel() {{ theme.textContent=root.dataset.theme==='dark'?'Light theme':'Dark theme'; }}
  themeLabel();
  theme.addEventListener('click', () => {{
    root.dataset.theme=root.dataset.theme==='dark'?'light':'dark'; themeLabel();
    try {{ localStorage.setItem('dlm-viz-theme',root.dataset.theme); }} catch (_) {{}}
  }});

  const search=document.getElementById('search'), count=document.getElementById('count');
  const examples=[...document.querySelectorAll('.example')];
  function filter() {{
    const query=search.value.trim().toLowerCase(); let shown=0;
    for(const example of examples) {{
      const hit=!query||example.textContent.toLowerCase().includes(query);
      example.hidden=!hit; if(hit) shown++;
    }}
    count.textContent=`${{shown}} / ${{examples.length}} examples`;
  }}
  search.addEventListener('input',filter); filter();

  function visiblePositions() {{ return [...document.querySelectorAll('.position')].filter(p => !p.closest('.example').hidden); }}
  function move(delta) {{
    const positions=visiblePositions(); if(!positions.length) return;
    let index=positions.indexOf(document.activeElement.closest?.('.position'));
    if(index<0) index=delta>0?-1:0;
    positions[Math.max(0,Math.min(positions.length-1,index+delta))].focus();
  }}
  document.addEventListener('keydown', event => {{
    const tag=document.activeElement.tagName;
    if(tag==='INPUT'||tag==='BUTTON') {{ if(event.key==='Escape') document.activeElement.blur(); return; }}
    if(event.key==='ArrowDown'||event.key==='j') {{ event.preventDefault(); move(1); }}
    else if(event.key==='ArrowUp'||event.key==='k') {{ event.preventDefault(); move(-1); }}
    else if(event.key==='Home') {{ const p=visiblePositions(); if(p.length) {{ event.preventDefault(); p[0].focus(); }} }}
    else if(event.key==='End') {{ const p=visiblePositions(); if(p.length) {{ event.preventDefault(); p[p.length-1].focus(); }} }}
    else if(event.key==='/') {{ event.preventDefault(); search.focus(); }}
  }});
  for(const position of document.querySelectorAll('.position')) {{
    position.addEventListener('mouseenter',()=>position.classList.add('hovered'));
    position.addEventListener('mouseleave',()=>position.classList.remove('hovered'));
  }}
}})();
</script>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _summarize(
    records: Sequence[Dict[str, Any]],
    model_name: str,
    checkpoint: str,
    dtype: str,
    right_shift_logits: bool,
    removed_top_k: Sequence[int],
    top_n: int,
    include_special_tokens: bool,
) -> Dict[str, Any]:
    labels = ["raw"] + [f"removed_top_{k}" for k in removed_top_k]
    aggregates = {
        label: {"gold_prob": 0.0, "top1": 0.0, "entropy": 0.0, "sum": 0.0}
        for label in labels
    }
    n_positions = 0
    n_selected = 0
    n_answer_tokens = 0
    for record in records:
        n_selected += record["n_positions_total"]
        n_answer_tokens += record["n_answer_tokens"]
        for position in record["positions"]:
            n_positions += 1
            for panel in position["panels"]:
                aggregate = aggregates[panel["label"]]
                for metric in aggregate:
                    aggregate[metric] += panel[metric]

    def means(metric: str) -> Dict[str, float]:
        denominator = max(n_positions, 1)
        return {label: aggregates[label][metric] / denominator for label in labels}

    return {
        "model": model_name,
        "checkpoint": checkpoint,
        "dtype": dtype,
        "right_shift_logits": bool(right_shift_logits),
        "masking": "selected labelled answer positions simultaneously; unselected answer tokens visible",
        "mask_positions": records[0]["mask_positions"] if records else [],
        "removed_top_k": list(removed_top_k),
        "top_n": int(top_n),
        "include_special_tokens": bool(include_special_tokens),
        "n_examples": len(records),
        "n_answer_tokens": n_answer_tokens,
        "n_positions_selected": n_selected,
        "n_positions_total": n_selected,
        "n_positions_rendered": n_positions,
        "mean_gold_prob": means("gold_prob"),
        "mean_top1": means("top1"),
        "mean_entropy": means("entropy"),
        "mean_probability_sum": means("sum"),
    }


@torch.no_grad()
def _analyze_example(
    model,
    tokenizer,
    example: Dict[str, Any],
    example_index: int,
    right_shift_logits: bool,
    removed_top_k: Sequence[int],
    mask_position_values: Sequence[str],
    top_n: int,
    max_length: int,
    include_special_tokens: bool,
    hidden_terminal_ids: set,
) -> Dict[str, Any]:
    item = build_pair(tokenizer, example["question"], example["answer"], max_length)
    original_answer_positions = (item["labels"] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
    if original_answer_positions.numel() == 0:
        raise ValueError(f"example {example_index}: tokenization produced no answer positions")

    answer_token_ids = item["input_ids"][original_answer_positions]
    print(f"[viz] answer tokens: {_answer_token_preview(tokenizer, answer_token_ids)}", flush=True)
    selected_answer_indices = _parse_mask_positions(
        mask_position_values, int(original_answer_positions.numel())
    )
    print(
        f"[viz] masking answer token positions: "
        f"{', '.join(str(position) for position in selected_answer_indices)}",
        flush=True,
    )

    selected_sequence_positions = original_answer_positions[selected_answer_indices]
    mask_override = torch.zeros_like(item["labels"], dtype=torch.bool)
    mask_override[selected_sequence_positions] = True
    batch = {key: value.unsqueeze(0) for key, value in item.items()}
    log_probs_batch, labels_batch = tokenwise_vocab_logprobs_diffusion(
        model,
        batch,
        right_shift_logits=right_shift_logits,
        mask_token_id=tokenizer.mask_token_id,
        return_labels=True,
        mask_override=mask_override.unsqueeze(0),
    )
    log_probs = log_probs_batch[0]
    gold_ids = labels_batch[0]
    if log_probs.shape[0] != len(selected_answer_indices) or gold_ids.numel() != log_probs.shape[0]:
        raise RuntimeError(
            f"example {example_index}: selected answer/logit alignment mismatch "
            f"({len(selected_answer_indices)} selected vs {log_probs.shape[0]} distributions)"
        )
    if any(k >= log_probs.shape[-1] for k in removed_top_k):
        raise ValueError(
            f"--remove-top-k must be smaller than vocabulary size {log_probs.shape[-1]}"
        )

    positions = []
    for selected_index, answer_index in enumerate(selected_answer_indices):
        gold_id = int(gold_ids[selected_index].item())
        is_special = gold_id in hidden_terminal_ids
        if is_special and not include_special_tokens:
            continue
        raw_log_dist = log_probs[selected_index]
        raw_panel = _distribution_panel(
            raw_log_dist, tokenizer, top_n, gold_id, label="raw"
        )
        panels = [raw_panel]
        for k in removed_top_k:
            removed = _removed_rows(raw_log_dist, tokenizer, k, gold_id)
            renormalized = _remove_topk(raw_log_dist, k)
            panels.append(
                _distribution_panel(
                    renormalized,
                    tokenizer,
                    top_n,
                    gold_id,
                    label=f"removed_top_{k}",
                    removed_top_k=k,
                    removed_rows=removed,
                )
            )
        gold_surface = _decode_token(tokenizer, gold_id)
        positions.append(
            {
                "answer_index": answer_index,
                # Dream prepends BOS inside the canonical helper before masking.
                "sequence_position": int(original_answer_positions[answer_index].item())
                + int(right_shift_logits),
                "gold": gold_surface,
                "gold_display": _visible_token(gold_surface),
                "gold_id": gold_id,
                "is_special": is_special,
                "panels": panels,
            }
        )

    if not positions:
        raise ValueError(
            f"example {example_index}: no displayable answer tokens remain after hiding special "
            "tokens; pass --include-special-tokens or provide a non-empty textual answer"
        )
    return {
        "example": example_index,
        "id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "mask_positions": selected_answer_indices,
        "n_answer_tokens": int(original_answer_positions.numel()),
        "n_positions_total": len(selected_answer_indices),
        "n_rendered": len(positions),
        "positions": positions,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), required=True)
    parser.add_argument(
        "--ckpt", required=True,
        help="local fine-tuned Hugging Face checkpoint directory",
    )
    parser.add_argument("--question", required=True, help="question for the QA example")
    parser.add_argument("--answer", required=True, help="ground-truth answer for the QA example")
    parser.add_argument(
        "--mask-positions",
        nargs="+",
        default=["all"],
        metavar="INDEX",
        help=(
            "0-based tokenized-answer positions to mask simultaneously, or 'all'; "
            "accepts space- or comma-separated indices"
        ),
    )
    parser.add_argument(
        "--remove-top-k", nargs="+", default=["1", "5", "10", "20"], metavar="K",
        help=(
            "raw top-K depths to remove independently before renormalizing survivors; "
            "accepts space- or comma-separated values"
        ),
    )
    parser.add_argument("--top-n", type=int, default=12, help="vocabulary rows shown per panel")
    parser.add_argument("--max-length", type=int, default=512, help="maximum tokenized chat length")
    parser.add_argument("--dtype", default="bfloat16", help="model dtype passed to dllm")
    parser.add_argument(
        "--include-special-tokens", action="store_true",
        help="render selected answer terminator/EOS positions",
    )
    parser.add_argument(
        "--output", default="logs/token_distributions.html",
        help="self-contained HTML output; .jsonl and .summary.json sidecars share its stem",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.top_n < 1:
            raise ValueError("--top-n must be positive")
        if args.max_length < 1:
            raise ValueError("--max-length must be positive")
        removed_top_k = _parse_removed_top_k(args.remove_top_k)
        example = _load_example(args)
        html_path, jsonl_path, summary_path = _output_paths(args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    checkpoint = str(Path(args.ckpt).expanduser())
    if not Path(checkpoint).exists():
        parser.error(f"checkpoint does not exist: {checkpoint}")

    preset = MODEL_PRESETS[args.model]
    right_shift_logits = preset["right_shift_logits"]
    print(
        f"[viz] loading {checkpoint} (model={args.model}, dtype={args.dtype}, "
        f"right_shift_logits={right_shift_logits})",
        flush=True,
    )
    model, tokenizer = load_dlm(checkpoint, args.model, dtype=args.dtype)
    if tokenizer.mask_token_id is None:
        raise RuntimeError("tokenizer has no mask_token_id (dllm model registration failed)")

    hidden_terminal_ids = _hidden_terminal_ids(tokenizer)
    print(f"[viz] example id={example['id']!r}", flush=True)
    try:
        record = _analyze_example(
            model,
            tokenizer,
            example,
            0,
            right_shift_logits,
            removed_top_k,
            args.mask_positions,
            args.top_n,
            args.max_length,
            args.include_special_tokens,
            hidden_terminal_ids,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        parser.error(f"while processing example id={example['id']!r}: {exc}")
    records = [record]

    summary = _summarize(
        records,
        args.model,
        checkpoint,
        args.dtype,
        right_shift_logits,
        removed_top_k,
        args.top_n,
        args.include_special_tokens,
    )

    html_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_html(html_path, records, summary)

    print("[viz] DONE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[viz] HTML        : {html_path.resolve()}", flush=True)
    print(f"[viz] JSONL       : {jsonl_path.resolve()}", flush=True)
    print(f"[viz] summary JSON: {summary_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
