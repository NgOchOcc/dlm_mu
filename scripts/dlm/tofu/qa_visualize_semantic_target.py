#!/usr/bin/env python
"""Visualize the exact Semantic_Unlearn target for selected QA answer tokens.

Selected 0-based positions in the tokenized labelled answer are masked together in
one frozen-teacher pass while unselected answer tokens remain visible. For every
selected position, the report compares the normalized teacher distribution ``p0``
with the production Semantic_Unlearn target

    q*(v) = p0(v) exp(-eta k(v)) / Z,
    k(v) = exp(-||E_v - E_gold||^2 / h).

The production trainer's own ``_semantic_kernels`` and ``_tilted_target`` methods
construct the target. The model's frozen input embeddings supply ``E`` and both
logits and embeddings are restricted to ``len(tokenizer)`` exactly as in training.
A self-contained HTML report, compact JSONL audit record, and summary JSON are
written under the same output stem.
"""

import argparse
import html
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import qa_visualize_dis as base  # noqa: E402
from data.utils import IGNORE_INDEX  # noqa: E402


def _load_semantic_unlearn_class():
    """Load production target math without importing the full Trainer package.

    Importing ``trainer`` executes its registry and eagerly imports DeepSpeed. This
    standalone diagnostic needs only the methods defined in semantic_unlearn.py, so
    install a minimal base-class shell for that module's one inheritance import.
    """
    module_name = "_semantic_target_production"
    module_path = _REPO / "src" / "trainer" / "dlm" / "unlearn" / "semantic_unlearn.py"
    base_name = "trainer.dlm.unlearn.base"
    saved_base = sys.modules.get(base_name)
    shell = types.ModuleType(base_name)
    shell.DiffusionUnlearnTrainer = type("DiffusionUnlearnTrainer", (), {})
    sys.modules[base_name] = shell
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load Semantic_Unlearn module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.DiffusionSemanticUnlearn
    finally:
        if saved_base is None:
            del sys.modules[base_name]
        else:
            sys.modules[base_name] = saved_base


DiffusionSemanticUnlearn = _load_semantic_unlearn_class()


ETA_ZERO_ATOL = 2e-6
GOLD_KERNEL_ATOL = 2e-6
NORMALIZATION_ATOL = 2e-5


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value}")
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value}") from exc
    if integer != numeric or integer <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return integer


def _parse_etas(values: Sequence[str]) -> List[float]:
    etas: List[float] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                eta = float(item)
            except ValueError as exc:
                raise ValueError(f"invalid --eta value {item!r}") from exc
            if not math.isfinite(eta) or eta < 0.0:
                raise ValueError(f"--eta values must be finite and >= 0, got {item!r}")
            if eta not in etas:
                etas.append(eta)
    if not etas:
        raise ValueError("--eta must contain at least one finite non-negative value")
    return etas


def _eta_text(eta: float) -> str:
    return f"{eta:.12g}"


def _eta_label(eta: float) -> str:
    return f"eta_{_eta_text(eta)}"


def _load_dlm(
    pretrained: str,
    model_name: str,
    dtype: str,
    device_map_mode: str,
    gpu_memory: Optional[str],
    cpu_memory: Optional[str],
):
    """Load a teacher, optionally splitting its weights across GPU and CPU."""
    if device_map_mode == "single":
        if not torch.cuda.is_available():
            raise RuntimeError("--device-map single requires an available CUDA GPU")
        return base.load_dlm(pretrained, model_name, dtype=dtype)

    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    base._import_dllm_module(f"dllm.pipelines.{model_name}.models")
    tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="right")
    base._configure_tokenizer(tokenizer, model_name)

    if device_map_mode == "cpu" or not torch.cuda.is_available():
        device_map: Any = {"": "cpu"}
        max_memory = None
    else:
        device_map = "auto"
        max_memory = {}
        if gpu_memory:
            for index in range(torch.cuda.device_count()):
                max_memory[index] = gpu_memory
        if cpu_memory:
            max_memory["cpu"] = cpu_memory
        if not max_memory:
            max_memory = None

    params: Dict[str, Any] = {"dtype": dtype, "device_map": device_map}
    if max_memory is not None:
        params["max_memory"] = max_memory

    # LLaDA's Hugging Face wrapper currently hardcodes CUDA for inner-module
    # construction. That defeats Transformers' meta-device loading, which is
    # required for CPU loading and GPU/CPU dispatch. Patch only during model
    # construction; loaded modules and the production model code are unchanged.
    llada_module = None
    original_init = None
    if model_name == "llada":
        import dllm.pipelines.llada.models.modeling_llada as llada_module

        original_init = llada_module.LLaDAModel.__init__

        def meta_aware_init(self, config, init_params=True):
            if getattr(config, "init_device", None) == "cuda":
                config.init_device = "meta"
            return original_init(self, config, init_params=init_params)

        llada_module.LLaDAModel.__init__ = meta_aware_init

    try:
        try:
            model = AutoModelForMaskedLM.from_pretrained(pretrained, **params)
        except ValueError:
            model = AutoModel.from_pretrained(pretrained, **params)
    finally:
        if llada_module is not None and original_init is not None:
            llada_module.LLaDAModel.__init__ = original_init

    model.eval()
    return model, tokenizer


def _build_probe(
    vocab_size: int,
    eta: float,
    sigma_fallback: float,
    center_chunk_size: int,
    vocab_chunk_size: int,
) -> DiffusionSemanticUnlearn:
    """Bind production target methods without constructing Trainer/DeepSpeed."""
    if vocab_size < 2:
        raise ValueError("Semantic_Unlearn requires a vocabulary with at least two tokens")
    if not math.isfinite(sigma_fallback) or sigma_fallback <= 0.0:
        raise ValueError(
            f"--sigma-fallback must be finite and > 0, got {sigma_fallback}"
        )
    probe = object.__new__(DiffusionSemanticUnlearn)
    probe.vocab_size = int(vocab_size)
    probe.eta = float(eta)
    probe.sigma_fallback = float(sigma_fallback)
    probe.center_chunk_size = _positive_integer(center_chunk_size, "--center-chunk-size")
    probe.vocab_chunk_size = _positive_integer(vocab_chunk_size, "--vocab-chunk-size")
    return probe


def _rank(probabilities: torch.Tensor, token_id: int) -> int:
    probability = probabilities[int(token_id)]
    return int((probabilities > probability).sum().item()) + 1


def _entropy(probabilities: torch.Tensor) -> float:
    positive = probabilities > 0
    return float(-(probabilities[positive] * probabilities[positive].log()).sum().item())


def _top_rows(
    probabilities: torch.Tensor,
    raw_probabilities: torch.Tensor,
    kernel: torch.Tensor,
    cosine_similarity: torch.Tensor,
    tokenizer,
    top_n: int,
    gold_id: int,
) -> List[Dict[str, Any]]:
    shown = min(top_n, probabilities.numel())
    values, ids = torch.topk(probabilities, shown)
    rows: List[Dict[str, Any]] = []
    for probability, token_id in zip(values.tolist(), ids.tolist()):
        surface = base._decode_token(tokenizer, int(token_id))
        rows.append(
            {
                "token": surface,
                "display": base._visible_token(surface),
                "id": int(token_id),
                "is_gold": int(token_id) == int(gold_id),
                "prob": float(probability),
                "raw_prob": float(raw_probabilities[token_id].item()),
                "kernel": float(kernel[token_id].item()),
                "cosine": float(cosine_similarity[token_id].item()),
            }
        )
    if int(gold_id) not in {row["id"] for row in rows}:
        surface = base._decode_token(tokenizer, gold_id)
        rows.append(
            {
                "token": surface,
                "display": base._visible_token(surface),
                "id": int(gold_id),
                "is_gold": True,
                "below_top_n": True,
                "prob": float(probabilities[gold_id].item()),
                "raw_prob": float(raw_probabilities[gold_id].item()),
                "kernel": float(kernel[gold_id].item()),
                "cosine": float(cosine_similarity[gold_id].item()),
            }
        )
    return rows


def _panel(
    probabilities: torch.Tensor,
    raw_probabilities: torch.Tensor,
    kernel: torch.Tensor,
    cosine_similarity: torch.Tensor,
    tokenizer,
    top_n: int,
    gold_id: int,
    label: str,
    eta: Optional[float],
    bandwidth: float,
    fallback_used: bool,
) -> Dict[str, Any]:
    rows = _top_rows(
        probabilities,
        raw_probabilities,
        kernel,
        cosine_similarity,
        tokenizer,
        top_n,
        gold_id,
    )
    return {
        "label": label,
        "eta": eta,
        "gold_prob": float(probabilities[gold_id].item()),
        "gold_rank": _rank(probabilities, gold_id),
        "top1": float(probabilities.max().item()),
        "entropy": _entropy(probabilities),
        "sum": float(probabilities.sum().item()),
        "bandwidth": float(bandwidth),
        "fallback_used": bool(fallback_used),
        "gold_kernel": float(kernel[gold_id].item()),
        "rows": rows,
    }


def _embedding_matrix(model, vocab_size: int) -> torch.Tensor:
    embeddings_model = DiffusionSemanticUnlearn._unwrap_embeddings_model(model)
    weight = embeddings_model.get_input_embeddings().weight
    if weight.shape[0] < vocab_size:
        raise ValueError(
            f"input embedding has {weight.shape[0]} rows but tokenizer has {vocab_size} tokens"
        )
    return weight[:vocab_size].detach()


def _semantic_kernels_with_fallbacks(
    probe: DiffusionSemanticUnlearn,
    center_ids: torch.Tensor,
    E: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call production kernel construction while capturing its fallback decision."""
    valid_chunks: List[torch.Tensor] = []
    production_squared_distances = DiffusionSemanticUnlearn._squared_distances

    def diagnostic_squared_distances(self, centers, embeddings):
        distances = production_squared_distances(self, centers, embeddings)
        median_d2 = distances.median(dim=-1).values
        valid_chunks.append(torch.isfinite(median_d2) & (median_d2 > 0.0))
        return distances

    probe._squared_distances = diagnostic_squared_distances.__get__(
        probe, DiffusionSemanticUnlearn
    )
    try:
        kernels, bandwidths = probe._semantic_kernels(center_ids, E)
    finally:
        del probe._squared_distances
    valid = torch.cat(valid_chunks) if valid_chunks else center_ids.new_empty(0, dtype=torch.bool)
    return kernels, bandwidths, ~valid


@torch.no_grad()
def _analyze_example(
    model,
    tokenizer,
    example: Dict[str, Any],
    right_shift_logits: bool,
    etas: Sequence[float],
    sigma_fallback: float,
    center_chunk_size: int,
    vocab_chunk_size: int,
    mask_position_values: Sequence[str],
    top_n: int,
    max_length: int,
    include_special_tokens: bool,
    hidden_terminal_ids: set,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    item = base.build_pair(tokenizer, example["question"], example["answer"], max_length)
    answer_positions = (item["labels"] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
    if answer_positions.numel() == 0:
        raise ValueError("tokenization produced no answer positions")

    answer_token_ids = item["input_ids"][answer_positions]
    print(f"[semantic-viz] answer tokens: {base._answer_token_preview(tokenizer, answer_token_ids)}", flush=True)
    selected_indices = base._parse_mask_positions(
        mask_position_values, int(answer_positions.numel())
    )
    print(
        "[semantic-viz] masking answer token positions: "
        + ", ".join(str(position) for position in selected_indices),
        flush=True,
    )

    selected_sequence_positions = answer_positions[selected_indices]
    mask_override = torch.zeros_like(item["labels"], dtype=torch.bool)
    mask_override[selected_sequence_positions] = True
    batch = {key: value.unsqueeze(0) for key, value in item.items()}
    log_probs_batch, labels_batch = base.tokenwise_vocab_logprobs_diffusion(
        model,
        batch,
        right_shift_logits=right_shift_logits,
        mask_token_id=tokenizer.mask_token_id,
        return_labels=True,
        mask_override=mask_override.unsqueeze(0),
    )
    full_log_probs = log_probs_batch[0]
    gold_ids = labels_batch[0]
    if full_log_probs.shape[0] != len(selected_indices):
        raise RuntimeError(
            "selected answer/logit alignment mismatch "
            f"({len(selected_indices)} selected vs {full_log_probs.shape[0]} distributions)"
        )

    vocab_size = len(tokenizer)
    if full_log_probs.shape[-1] < vocab_size:
        raise ValueError(
            f"model emitted {full_log_probs.shape[-1]} logits but tokenizer has {vocab_size} tokens"
        )
    # Re-normalize after dropping any padded LM-head rows, matching training exactly.
    log_p0 = full_log_probs[:, :vocab_size].float()
    log_p0 = log_p0 - torch.logsumexp(log_p0, dim=-1, keepdim=True)
    raw_probabilities = log_p0.exp()

    E = _embedding_matrix(model, vocab_size)
    probe = _build_probe(
        vocab_size,
        etas[0],
        sigma_fallback,
        center_chunk_size,
        vocab_chunk_size,
    )
    unique_ids, inverse = torch.unique(gold_ids, sorted=True, return_inverse=True)
    unique_kernels, unique_bandwidths, unique_fallback_flags = (
        _semantic_kernels_with_fallbacks(probe, unique_ids, E)
    )
    kernels = unique_kernels[inverse]
    bandwidths = unique_bandwidths[inverse]
    fallback_flags = unique_fallback_flags[inverse]
    normalized_embeddings = F.normalize(E.float(), dim=-1)
    cosine_similarities = normalized_embeddings[gold_ids] @ normalized_embeddings.T

    targets: Dict[float, Tuple[torch.Tensor, torch.Tensor]] = {}
    for eta in etas:
        probe.eta = float(eta)
        targets[eta] = probe._tilted_target(log_p0, kernels)

    eta_zero_error = 0.0
    if 0.0 in targets:
        eta_zero_error = float((targets[0.0][1] - raw_probabilities).abs().max().item())
    gold_kernel_error = float(
        (kernels.gather(1, gold_ids.unsqueeze(1)).squeeze(1) - 1.0).abs().max().item()
    )
    raw_sum_error = float((raw_probabilities.sum(-1) - 1.0).abs().max().item())
    target_sum_error = max(
        float((target[1].sum(-1) - 1.0).abs().max().item())
        for target in targets.values()
    )
    normalization_error = max(raw_sum_error, target_sum_error)

    if 0.0 in targets and eta_zero_error > ETA_ZERO_ATOL:
        raise RuntimeError(
            f"eta=0 invariant failed: max |q-p0|={eta_zero_error:.3e} > {ETA_ZERO_ATOL:.1e}"
        )
    if gold_kernel_error > GOLD_KERNEL_ATOL:
        raise RuntimeError(
            "gold self-kernel invariant failed: "
            f"max |k(gold)-1|={gold_kernel_error:.3e} > {GOLD_KERNEL_ATOL:.1e}"
        )
    if normalization_error > NORMALIZATION_ATOL:
        raise RuntimeError(
            "normalization invariant failed: "
            f"max |sum-1|={normalization_error:.3e} > {NORMALIZATION_ATOL:.1e}"
        )

    positions: List[Dict[str, Any]] = []
    for selected_offset, answer_index in enumerate(selected_indices):
        gold_id = int(gold_ids[selected_offset].item())
        is_special = gold_id in hidden_terminal_ids
        if is_special and not include_special_tokens:
            continue
        raw = raw_probabilities[selected_offset]
        kernel = kernels[selected_offset]
        cosine_similarity = cosine_similarities[selected_offset]
        bandwidth = float(bandwidths[selected_offset].item())
        fallback_used = bool(fallback_flags[selected_offset].item())
        panels = [
            _panel(
                raw,
                raw,
                kernel,
                cosine_similarity,
                tokenizer,
                top_n,
                gold_id,
                "raw",
                None,
                bandwidth,
                fallback_used,
            )
        ]
        for eta in etas:
            panels.append(
                _panel(
                    targets[eta][1][selected_offset],
                    raw,
                    kernel,
                    cosine_similarity,
                    tokenizer,
                    top_n,
                    gold_id,
                    _eta_label(eta),
                    float(eta),
                    bandwidth,
                    fallback_used,
                )
            )
        gold_surface = base._decode_token(tokenizer, gold_id)
        positions.append(
            {
                "answer_index": int(answer_index),
                "sequence_position": int(answer_positions[answer_index].item())
                + int(right_shift_logits),
                "gold": gold_surface,
                "gold_display": base._visible_token(gold_surface),
                "gold_id": gold_id,
                "is_special": is_special,
                "bandwidth": bandwidth,
                "fallback_used": fallback_used,
                "gold_kernel": float(kernel[gold_id].item()),
                "panels": panels,
            }
        )

    if not positions:
        raise ValueError(
            "no displayable answer tokens remain after hiding special tokens; "
            "pass --include-special-tokens or select a textual answer token"
        )

    record = {
        "example": 0,
        "id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "mask_positions": selected_indices,
        "n_answer_tokens": int(answer_positions.numel()),
        "n_positions_total": len(selected_indices),
        "n_rendered": len(positions),
        "positions": positions,
    }
    diagnostics = {
        "eta_zero_max_abs_error": eta_zero_error,
        "gold_kernel_max_abs_error": gold_kernel_error,
        "normalization_max_abs_error": normalization_error,
        "logit_vocab_size": int(full_log_probs.shape[-1]),
        "embedding_vocab_size": int(E.shape[0]),
    }
    return record, diagnostics


def _format_probability(value: float) -> str:
    return base._format_probability(value)


def _render_rows(rows: Sequence[Dict[str, Any]], is_raw: bool) -> str:
    max_probability = max((float(row["prob"]) for row in rows), default=0.0)
    rendered: List[str] = []
    for row in rows:
        probability = float(row["prob"])
        width = 0.0 if probability <= 0.0 or max_probability <= 0.0 else 100.0 * probability / max_probability
        if 0.0 < width < 0.35:
            width = 0.35
        classes = ["dist-row"]
        notes: List[str] = []
        if row["is_gold"]:
            classes.append("ground-truth")
            notes.append("GT")
        if row.get("below_top_n"):
            notes.append("OUTSIDE TOP-N")
        badges = "".join(f'<span class="badge">{note}</span>' for note in notes)
        display = html.escape(row["display"], quote=True)
        exact = html.escape(
            f"token id {row['id']}; shown probability {probability:.17g}; "
            f"raw p0 {row['raw_prob']:.17g}; semantic kernel {row['kernel']:.17g}; "
            f"cosine similarity {row['cosine']:.17g}",
            quote=True,
        )
        probability_label = "p0" if is_raw else "q*"
        rendered.append(
            f'<tr class="{" ".join(classes)}" tabindex="0" title="{exact}">'
            f'<th scope="row"><span class="token">{display}</span>{badges}'
            f'<span class="token-id">id {row["id"]}</span></th>'
            f'<td class="bar-cell"><span class="bar-track" aria-hidden="true">'
            f'<span class="bar-fill" style="width:{width:.4f}%"></span></span></td>'
            f'<td class="number" aria-label="{probability_label}">{_format_probability(probability)}</td>'
            f'<td class="number raw">{_format_probability(float(row["raw_prob"]))}</td>'
            f'<td class="number kernel">{float(row["kernel"]):.4f}</td>'
            f'<td class="number cosine">{float(row["cosine"]):.4f}</td></tr>'
        )
    return "".join(rendered)


def _render_panel(panel: Dict[str, Any]) -> str:
    is_raw = panel["eta"] is None
    title = "Raw teacher p0" if is_raw else f"Semantic target · η={_eta_text(panel['eta'])}"
    fallback = " · fallback" if panel["fallback_used"] else ""
    return (
        '<section class="panel">'
        f'<h4>{html.escape(title)}</h4>'
        '<div class="panel-stats">'
        f'<span>GT <b>{_format_probability(panel["gold_prob"])}</b></span>'
        f'<span>rank <b>{panel["gold_rank"]}</b></span>'
        f'<span>top-1 <b>{_format_probability(panel["top1"])}</b></span>'
        f'<span>H <b>{panel["entropy"]:.3f}</b> nats</span>'
        f'<span>Σ <b>{panel["sum"]:.6f}</b></span>'
        f'<span>h <b>{panel["bandwidth"]:.4e}</b>{fallback}</span>'
        f'<span>k(GT) <b>{panel["gold_kernel"]:.6f}</b></span></div>'
        '<table class="distribution"><thead><tr><th scope="col">token</th>'
        '<th scope="col">relative probability</th><th scope="col">shown</th>'
        '<th scope="col">p0</th><th scope="col">k</th><th scope="col">cos</th></tr></thead>'
        f'<tbody>{_render_rows(panel["rows"], is_raw)}</tbody></table></section>'
    )


def _render_example(record: Dict[str, Any]) -> str:
    positions: List[str] = []
    for position in record["positions"]:
        panels = "".join(_render_panel(panel) for panel in position["panels"])
        special = '<span class="auto-badge">AUTO/SPECIAL</span>' if position["is_special"] else ""
        fallback = '<span class="fallback-badge">FALLBACK h</span>' if position["fallback_used"] else ""
        positions.append(
            f'<section class="position{" special" if position["is_special"] else ""}" tabindex="0">'
            '<div class="position-heading">'
            f'<span>Answer token {position["answer_index"]}</span>'
            f'<strong>{html.escape(position["gold_display"], quote=True)}</strong>'
            f'<code>id {position["gold_id"]}</code>'
            f'<code>sequence pos {position["sequence_position"]}</code>{special}{fallback}</div>'
            f'<div class="panels">{panels}</div></section>'
        )
    return (
        f'<article class="example"><div class="example-heading">Example {record["example"]}'
        f'<span>id {html.escape(str(record["id"]), quote=True)}</span>'
        f'<span>{record["n_rendered"]} / {record["n_positions_total"]} selected masked positions shown</span>'
        f'<span>{record["n_answer_tokens"]} labelled answer tokens</span></div>'
        f'<div class="qa"><b>Question</b>{html.escape(record["question"], quote=True)}</div>'
        f'<div class="qa"><b>Ground-truth answer</b>{html.escape(record["answer"], quote=True)}</div>'
        f'{"".join(positions)}</article>'
    )


def _write_html(path: Path, records: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    examples_html = "".join(_render_example(record) for record in records)
    eta_text = ", ".join(_eta_text(eta) for eta in summary["etas"])
    checkpoint = html.escape(summary["checkpoint"], quote=True)
    model = html.escape(summary["model"], quote=True)
    invariant = summary["invariants"]
    document = f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semantic_Unlearn target distributions</title>
<style>
:root {{
  color-scheme:dark; --page:#0d0d0d; --surface:#1a1a19; --surface-2:#222220;
  --text:#ffffff; --secondary:#c3c2b7; --muted:#898781; --grid:#2c2c2a;
  --blue:#3987e5; --blue-soft:rgba(57,135,229,.15); --gold:#e66767;
  --gold-soft:rgba(230,103,103,.12); --orange:#d95926; --border:rgba(255,255,255,.10);
}}
:root[data-theme="light"] {{
  color-scheme:light; --page:#f9f9f7; --surface:#fcfcfb; --surface-2:#f0efec;
  --text:#0b0b0b; --secondary:#52514e; --muted:#898781; --grid:#e1e0d9;
  --blue:#2a78d6; --blue-soft:rgba(42,120,214,.12); --gold:#e34948;
  --gold-soft:rgba(227,73,72,.10); --orange:#eb6834; --border:rgba(11,11,11,.10);
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--page); color:var(--text); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
header {{ position:sticky; top:0; z-index:10; padding:14px 22px; background:var(--surface); border-bottom:1px solid var(--border); box-shadow:0 5px 20px rgba(0,0,0,.12); }}
.header-top {{ display:flex; justify-content:space-between; align-items:flex-start; gap:18px; }}
h1 {{ margin:0; font-size:19px; letter-spacing:-.01em; }}
.subtitle {{ margin-top:3px; color:var(--secondary); font-size:12px; overflow-wrap:anywhere; }}
.controls {{ flex-shrink:0; }}
button,input {{ color:var(--text); background:var(--surface-2); border:1px solid var(--border); border-radius:7px; padding:7px 10px; font:inherit; }}
button {{ cursor:pointer; }} button:hover,button:focus-visible,input:focus-visible {{ border-color:var(--blue); outline:none; }}
.stats {{ display:flex; flex-wrap:wrap; gap:9px 20px; margin-top:11px; color:var(--secondary); font-size:12px; }}
.stats b {{ color:var(--text); }}
.search-row {{ display:flex; gap:10px; align-items:center; margin-top:10px; }}
#search {{ width:min(500px,100%); }} #count {{ color:var(--muted); font-size:12px; }}
main {{ max-width:1900px; margin:0 auto; padding:18px 22px 50px; }}
.legend {{ color:var(--secondary); font-size:12px; margin:0 0 12px; }}
.legend-mark {{ display:inline-block; width:18px; height:7px; border-radius:0 4px 4px 0; vertical-align:middle; margin:0 5px; }}
.legend-mark.blue {{ background:var(--blue); }} .legend-mark.gold {{ background:var(--gold); }}
.example {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:15px; margin-bottom:16px; }}
.example-heading {{ display:flex; flex-wrap:wrap; gap:8px 16px; color:var(--muted); font-size:12px; margin-bottom:9px; }}
.qa {{ white-space:pre-wrap; overflow-wrap:anywhere; padding:6px 0; }}
.qa b {{ display:inline-block; min-width:150px; color:var(--secondary); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
.position {{ border-top:1px solid var(--grid); padding:14px 0 5px; scroll-margin-top:175px; }}
.position:focus,.position:focus-within,.position.hovered {{ outline:2px solid var(--blue); outline-offset:5px; border-radius:5px; }}
.position.special {{ opacity:.78; }}
.position-heading {{ display:flex; flex-wrap:wrap; gap:7px 10px; align-items:center; color:var(--muted); font-size:12px; margin-bottom:9px; }}
.position-heading strong {{ color:var(--text); font:600 14px ui-monospace,SFMono-Regular,Consolas,monospace; white-space:pre-wrap; }}
code {{ color:var(--secondary); background:var(--surface-2); padding:1px 5px; border-radius:4px; }}
.auto-badge,.fallback-badge,.badge {{ display:inline-block; border:1px solid var(--gold); color:var(--text); border-radius:999px; padding:0 5px; font-size:9px; font-weight:700; letter-spacing:.04em; }}
.fallback-badge {{ border-color:var(--orange); }}
.panels {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:12px; align-items:start; }}
.panel {{ min-width:0; background:var(--surface-2); border:1px solid var(--border); border-radius:9px; padding:10px; }}
.panel h4 {{ margin:0 0 4px; font-size:12px; font-weight:700; }}
.panel-stats {{ display:flex; flex-wrap:wrap; gap:3px 10px; color:var(--muted); font-size:10px; margin-bottom:7px; }}
.panel-stats b {{ color:var(--secondary); font-variant-numeric:tabular-nums; }}
table {{ border-collapse:separate; border-spacing:0 2px; width:100%; table-layout:fixed; }}
thead {{ position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); }}
.dist-row:focus-visible {{ outline:1px solid var(--blue); outline-offset:1px; }}
th,td {{ padding:1px 3px; font-weight:400; }}
th {{ width:105px; text-align:left; overflow:hidden; }}
.token {{ display:inline-block; max-width:70px; color:var(--text); font:11px ui-monospace,SFMono-Regular,Consolas,monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; vertical-align:middle; }}
.token-id {{ display:block; color:var(--muted); font-size:8px; }}
.badge {{ margin-left:3px; padding:0 3px; font-size:7px; vertical-align:middle; }}
.bar-cell {{ width:auto; }} .bar-track {{ display:block; height:9px; background:var(--grid); border-radius:0 5px 5px 0; overflow:hidden; }}
.bar-fill {{ display:block; height:100%; background:var(--blue); border-radius:0 5px 5px 0; }}
.number {{ width:70px; text-align:right; color:var(--secondary); font:9px ui-monospace,SFMono-Regular,Consolas,monospace; font-variant-numeric:tabular-nums; }}
.number.kernel {{ width:48px; }} .number.cosine {{ width:48px; }} .number.raw {{ width:70px; }}
.ground-truth {{ background:var(--gold-soft); }} .ground-truth .bar-fill {{ background:var(--gold); }} .ground-truth .token {{ font-weight:700; }}
@media (max-width:760px) {{ header {{ position:static; }} .header-top {{ display:block; }} .controls {{ margin-top:9px; }} main {{ padding:12px; }} .qa b {{ display:block; min-width:0; }} .panels {{ grid-template-columns:1fr; }} .number.raw {{ display:none; }} }}
@media print {{ header {{ position:static; }} .controls,.search-row {{ display:none; }} .example {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<header>
  <div class="header-top"><div><h1>Semantic_Unlearn target distributions</h1>
    <div class="subtitle">{model} · {checkpoint} · q*(v) ∝ p0(v) exp(-η k(v)); selected answer positions masked simultaneously</div></div>
    <div class="controls"><button id="theme" type="button" aria-label="Toggle light and dark theme">Light theme</button></div>
  </div>
  <div class="stats">
    <span><b>{summary['n_positions_rendered']}</b> shown / <b>{summary['n_positions_selected']}</b> selected positions</span>
    <span>η <b>{html.escape(eta_text)}</b></span>
    <span>vocabulary <b>{summary['vocab_size']:,}</b></span>
    <span>fallback bandwidths <b>{summary['bandwidth']['fallback_count']}</b></span>
    <span>eta-0 error <b>{invariant['eta_zero_max_abs_error']:.2e}</b></span>
    <span>normalization error <b>{invariant['normalization_max_abs_error']:.2e}</b></span>
  </div>
  <div class="search-row"><input id="search" type="search" placeholder="Filter questions, answers, tokens…" autocomplete="off"><span id="count"></span></div>
</header>
<main>
  <p class="legend"><span class="legend-mark blue"></span>distribution token
    <span class="legend-mark gold"></span>ground-truth center (also labelled GT). Bars are relative within each panel; the table gives shown probability, raw p0, semantic kernel k, and embedding cosine similarity. Use ↑/↓ or j/k to move between positions.</p>
  {examples_html}
</main>
<script>
(() => {{
  const root=document.documentElement, theme=document.getElementById('theme');
  let saved=null; try {{ saved=localStorage.getItem('dlm-viz-theme'); }} catch (_) {{}}
  if(saved==='light'||saved==='dark') root.dataset.theme=saved;
  function themeLabel() {{ theme.textContent=root.dataset.theme==='dark'?'Light theme':'Dark theme'; }}
  themeLabel();
  theme.addEventListener('click',()=>{{ root.dataset.theme=root.dataset.theme==='dark'?'light':'dark'; themeLabel(); try {{ localStorage.setItem('dlm-viz-theme',root.dataset.theme); }} catch (_) {{}} }});
  const search=document.getElementById('search'), count=document.getElementById('count');
  const examples=[...document.querySelectorAll('.example')];
  function filter() {{ const query=search.value.trim().toLowerCase(); let shown=0; for(const example of examples) {{ const hit=!query||example.textContent.toLowerCase().includes(query); example.hidden=!hit; if(hit) shown++; }} count.textContent=`${{shown}} / ${{examples.length}} examples`; }}
  search.addEventListener('input',filter); filter();
  function visiblePositions() {{ return [...document.querySelectorAll('.position')].filter(p=>!p.closest('.example').hidden); }}
  function move(delta) {{ const positions=visiblePositions(); if(!positions.length) return; let index=positions.indexOf(document.activeElement.closest?.('.position')); if(index<0) index=delta>0?-1:0; positions[Math.max(0,Math.min(positions.length-1,index+delta))].focus(); }}
  document.addEventListener('keydown',event=>{{ const tag=document.activeElement.tagName; if(tag==='INPUT'||tag==='BUTTON') {{ if(event.key==='Escape') document.activeElement.blur(); return; }} if(event.key==='ArrowDown'||event.key==='j') {{ event.preventDefault(); move(1); }} else if(event.key==='ArrowUp'||event.key==='k') {{ event.preventDefault(); move(-1); }} else if(event.key==='Home') {{ const p=visiblePositions(); if(p.length) {{ event.preventDefault(); p[0].focus(); }} }} else if(event.key==='End') {{ const p=visiblePositions(); if(p.length) {{ event.preventDefault(); p[p.length-1].focus(); }} }} else if(event.key==='/') {{ event.preventDefault(); search.focus(); }} }});
  for(const position of document.querySelectorAll('.position')) {{ position.addEventListener('mouseenter',()=>position.classList.add('hovered')); position.addEventListener('mouseleave',()=>position.classList.remove('hovered')); }}
}})();
</script>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _summarize(
    records: Sequence[Dict[str, Any]],
    diagnostics: Dict[str, float],
    args,
    checkpoint: str,
    right_shift_logits: bool,
    vocab_size: int,
) -> Dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty record set")
    labels = ["raw"] + [_eta_label(eta) for eta in args.etas]
    aggregates = {
        label: {"gold_prob": 0.0, "top1": 0.0, "entropy": 0.0, "sum": 0.0}
        for label in labels
    }
    positions = [position for record in records for position in record["positions"]]
    for position in positions:
        for panel in position["panels"]:
            for metric in aggregates[panel["label"]]:
                aggregates[panel["label"]][metric] += panel[metric]
    denominator = max(len(positions), 1)
    means = {
        metric: {
            label: aggregates[label][metric] / denominator for label in labels
        }
        for metric in ("gold_prob", "top1", "entropy", "sum")
    }
    bandwidths = [position["bandwidth"] for position in positions]
    fallback_count = sum(int(position["fallback_used"]) for position in positions)
    sorted_bandwidths = sorted(bandwidths)
    midpoint = len(sorted_bandwidths) // 2
    if len(sorted_bandwidths) % 2:
        median_bandwidth = sorted_bandwidths[midpoint]
    else:
        median_bandwidth = (sorted_bandwidths[midpoint - 1] + sorted_bandwidths[midpoint]) / 2.0
    eta_zero_checked = 0.0 in args.etas
    return {
        "model": args.model,
        "checkpoint": checkpoint,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "gpu_memory": args.gpu_memory if args.device_map == "auto" else None,
        "cpu_memory": args.cpu_memory if args.device_map == "auto" else None,
        "right_shift_logits": bool(right_shift_logits),
        "formula": "q*(v) = normalize(p0(v) * exp(-eta * k(v)))",
        "kernel": "k(v) = exp(-||E_v-E_gold||^2/h), h=median(d^2)/log(|V|)",
        "masking": "selected labelled answer positions simultaneously; unselected answer tokens visible",
        "mask_positions": records[0]["mask_positions"],
        "etas": list(args.etas),
        "sigma_fallback": args.sigma_fallback,
        "center_chunk_size": args.center_chunk_size,
        "vocab_chunk_size": args.vocab_chunk_size,
        "top_n": args.top_n,
        "max_length": args.max_length,
        "include_special_tokens": bool(args.include_special_tokens),
        "vocab_size": int(vocab_size),
        "logit_vocab_size": int(diagnostics["logit_vocab_size"]),
        "embedding_vocab_size": int(diagnostics["embedding_vocab_size"]),
        "n_examples": len(records),
        "n_answer_tokens": sum(record["n_answer_tokens"] for record in records),
        "n_positions_selected": sum(record["n_positions_total"] for record in records),
        "n_positions_rendered": len(positions),
        "mean_gold_prob": means["gold_prob"],
        "mean_top1": means["top1"],
        "mean_entropy": means["entropy"],
        "mean_probability_sum": means["sum"],
        "bandwidth": {
            "min": min(bandwidths),
            "max": max(bandwidths),
            "mean": sum(bandwidths) / len(bandwidths),
            "median": median_bandwidth,
            "fallback_count": fallback_count,
            "fallback_fraction": fallback_count / len(bandwidths),
        },
        "invariants": {
            "eta_zero_checked": eta_zero_checked,
            "eta_zero_max_abs_error": diagnostics["eta_zero_max_abs_error"],
            "eta_zero_tolerance": ETA_ZERO_ATOL,
            "eta_zero_pass": (not eta_zero_checked)
            or diagnostics["eta_zero_max_abs_error"] <= ETA_ZERO_ATOL,
            "gold_kernel_max_abs_error": diagnostics["gold_kernel_max_abs_error"],
            "gold_kernel_tolerance": GOLD_KERNEL_ATOL,
            "gold_kernel_pass": diagnostics["gold_kernel_max_abs_error"] <= GOLD_KERNEL_ATOL,
            "normalization_max_abs_error": diagnostics["normalization_max_abs_error"],
            "normalization_tolerance": NORMALIZATION_ATOL,
            "normalization_pass": diagnostics["normalization_max_abs_error"] <= NORMALIZATION_ATOL,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=sorted(base.MODEL_PRESETS), required=True)
    parser.add_argument(
        "--ckpt",
        required=True,
        help="original fine-tuned checkpoint used as frozen teacher and embedding source",
    )
    parser.add_argument("--question", required=True, help="question for the QA example")
    parser.add_argument("--answer", required=True, help="ground-truth answer for the QA example")
    parser.add_argument(
        "--mask-positions",
        nargs="+",
        default=["all"],
        metavar="INDEX",
        help="0-based tokenized-answer positions to mask together, or 'all'",
    )
    parser.add_argument(
        "--eta",
        nargs="+",
        default=["0", "1", "10"],
        metavar="ETA",
        help="non-negative target strengths; accepts space- or comma-separated values",
    )
    parser.add_argument("--sigma-fallback", type=float, default=0.05)
    parser.add_argument("--center-chunk-size", type=int, default=8)
    parser.add_argument("--vocab-chunk-size", type=int, default=4096)
    parser.add_argument("--top-n", type=int, default=12, help="vocabulary rows shown per panel")
    parser.add_argument("--max-length", type=int, default=512, help="maximum tokenized chat length")
    parser.add_argument("--dtype", default="bfloat16", help="model dtype passed to transformers")
    parser.add_argument(
        "--device-map",
        choices=("auto", "single", "cpu"),
        default="auto",
        help=(
            "checkpoint placement: auto splits across GPU/CPU, single uses the current "
            "GPU, and cpu keeps all weights on CPU"
        ),
    )
    parser.add_argument(
        "--gpu-memory",
        default="9GiB",
        help="per-GPU limit used by --device-map auto (for example 9GiB)",
    )
    parser.add_argument(
        "--cpu-memory",
        default="25GiB",
        help="CPU RAM limit used by --device-map auto (for example 25GiB)",
    )
    parser.add_argument(
        "--include-special-tokens",
        action="store_true",
        help="render selected answer terminator/EOS positions",
    )
    parser.add_argument(
        "--output",
        default="logs/qa_semantic_target_vis/semantic_target.html",
        help="self-contained HTML output; JSONL and summary JSON share its stem",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.question:
            raise ValueError("--question must not be empty")
        if not args.answer:
            raise ValueError("--answer must not be empty")
        if args.top_n < 1:
            raise ValueError("--top-n must be positive")
        if args.max_length < 1:
            raise ValueError("--max-length must be positive")
        args.etas = _parse_etas(args.eta)
        if not math.isfinite(args.sigma_fallback) or args.sigma_fallback <= 0.0:
            raise ValueError("--sigma-fallback must be finite and > 0")
        args.center_chunk_size = _positive_integer(args.center_chunk_size, "--center-chunk-size")
        args.vocab_chunk_size = _positive_integer(args.vocab_chunk_size, "--vocab-chunk-size")
        html_path, jsonl_path, summary_path = base._output_paths(args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    checkpoint = str(Path(args.ckpt).expanduser())
    if not Path(checkpoint).exists():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    if "/unlearn/" in f"/{Path(checkpoint).as_posix().strip('/')}/":
        print(
            "[semantic-viz] WARNING: checkpoint path appears to be an unlearned model. "
            "Semantic_Unlearn targets should normally use the original fine-tuned checkpoint.",
            file=sys.stderr,
            flush=True,
        )

    right_shift_logits = base.MODEL_PRESETS[args.model]["right_shift_logits"]
    print(
        f"[semantic-viz] loading {checkpoint} (model={args.model}, dtype={args.dtype}, "
        f"device_map={args.device_map}, right_shift_logits={right_shift_logits})",
        flush=True,
    )
    try:
        model, tokenizer = _load_dlm(
            checkpoint,
            args.model,
            args.dtype,
            args.device_map,
            args.gpu_memory,
            args.cpu_memory,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[semantic-viz] ERROR loading checkpoint: {exc}", file=sys.stderr, flush=True)
        return 1
    if tokenizer.mask_token_id is None:
        print("[semantic-viz] ERROR: tokenizer has no mask_token_id", file=sys.stderr, flush=True)
        return 1

    example = {"id": "cli-0", "question": args.question, "answer": args.answer}
    try:
        record, diagnostics = _analyze_example(
            model,
            tokenizer,
            example,
            right_shift_logits,
            args.etas,
            args.sigma_fallback,
            args.center_chunk_size,
            args.vocab_chunk_size,
            args.mask_positions,
            args.top_n,
            args.max_length,
            args.include_special_tokens,
            base._hidden_terminal_ids(tokenizer),
        )
    except (TypeError, ValueError) as exc:
        parser.error(f"while processing example id={example['id']!r}: {exc}")
    except RuntimeError as exc:
        print(
            f"[semantic-viz] ERROR while processing example id={example['id']!r}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    records = [record]
    summary = _summarize(
        records,
        diagnostics,
        args,
        checkpoint,
        right_shift_logits,
        len(tokenizer),
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for current in records:
            handle.write(json.dumps(current, ensure_ascii=False, allow_nan=False) + "\n")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_html(html_path, records, summary)

    print("[semantic-viz] DONE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[semantic-viz] HTML        : {html_path.resolve()}", flush=True)
    print(f"[semantic-viz] JSONL       : {jsonl_path.resolve()}", flush=True)
    print(f"[semantic-viz] summary JSON: {summary_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
