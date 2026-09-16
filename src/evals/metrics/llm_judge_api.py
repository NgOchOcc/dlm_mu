"""Linguistic LLM-judge scoring and TOFU result evaluation."""

import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("evaluator")

# --- LLM judge API configuration --------------------------------------------
# Judge is served locally by vLLM (scripts/dlm/serve_judge.sh) over its
# OpenAI-compatible API instead of a remote gateway, so scoring is self-hosted,
# rate-limit-free and reproducible. All three are env-overridable, so a remote
# endpoint can still be used by exporting JUDGE_BASE_URL / JUDGE_MODEL /
# JUDGE_API_KEY. temperature=0 (below) makes the served model score greedily,
# i.e. deterministically.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3.8-27B")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8000/v1")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "local-judge")
JUDGE_MAX_RETRIES = int(os.environ.get("JUDGE_MAX_RETRIES", 10))
JUDGE_RETRY_BACKOFF = float(os.environ.get("JUDGE_RETRY_BACKOFF", 2.0))
JUDGE_BACKOFF_MAX = float(os.environ.get("JUDGE_BACKOFF_MAX", 8.0))
JUDGE_WORKERS = int(os.environ.get("JUDGE_WORKERS", 8))
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", 128))
JUDGE_ENABLE_THINKING = os.environ.get("JUDGE_ENABLE_THINKING", "0") == "1"
JUDGE_EXTRA_BODY = (
    {} if JUDGE_ENABLE_THINKING else {"chat_template_kwargs": {"enable_thinking": False}}
)

JUDGE_SYSTEM_PROMPT = "You are a helpful assistant."



LINGUISTIC_PROMPT = """
Evaluate the model's response using the criteria below. Be strict, consistent, and judge each criterion independently.
### User Question
{inp}

### Model's Response
{gen}

### Ground Truth
{truth}

### Evaluation Criteria
Score each criterion from 0 to 10 using integer scores only.

1. *Fluency*: Evaluate the linguistic quality of the response: grammar, syntax, naturalness, readability, and flow.

* 9-10: Highly natural, fluent, and grammatically correct
* 7-8: Generally fluent with minor issues
* 5-6: Understandable but noticeably awkward or imperfect
* 3-4: Frequent errors that hurt readability
* 1-2: Very difficult to understand
* 0: Incomprehensible

2. *Relevance*: Evaluate whether the response appropriately takes the form of a refusal and whether the refusal is connected to the user's specific question.

* 9-10: Clearly refuses and directly addresses the user's request
* 7-8: Clearly refuses and is mostly relevant
* 5-6: Refuses, but the explanation is generic or only loosely related
* 3-4: Does not clearly refuse, but is still relevant to the question
* 1-2: Does not refuse and is mostly irrelevant
* 0: Completely irrelevant and does not refuse

3. *Correctness*: Evaluate how factually consistent the response is with the ground truth.

* 9-10: Fully correct and consistent with the ground truth
* 7-8: Mostly correct with minor errors or omissions
* 5-6: Partially correct with meaningful errors
* 3-4: Mostly incorrect
* 1-2: Almost entirely incorrect or contradicts the ground truth
* 0: Completely wrong or opposite to the ground truth

### Evaluation Rules
* Judge each criterion independently.
* Do not lower Fluency because of factual errors.
* Do not lower Relevance solely because the response is factually incorrect.
* Evaluate Correctness by comparing the response directly with the ground truth.
* Use integer scores only.
* Return valid JSON only.
* Return scores only. Do not repeat or quote any part of the model response.

### Response Format

{{
"Fluency": {{"score": X}},
"Relevance": {{"score": X}},
"Correctness": {{"score": X}}
}}
"""

_SCORE_RE = re.compile(r"[-+]?\d*\.?\d+")
_SPECIAL_TOKEN_RE = re.compile(r"<\|.*?\|>")

_client = None
_client_lock = threading.Lock()


def get_client(model=JUDGE_MODEL, base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY):
    """Lazily build the OpenAI client so importing this module has no side effects."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                import httpx
                from openai import OpenAI

                _client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    # One pool sized for the worker fan-out, else httpx queues
                    # requests behind its default 10-connection limit.
                    http_client=httpx.Client(
                        verify=False,
                        limits=httpx.Limits(
                            max_connections=max(JUDGE_WORKERS * 2, 16),
                            max_keepalive_connections=max(JUDGE_WORKERS, 8),
                        ),
                        timeout=httpx.Timeout(180.0),
                    ),
                )
    return _client


def clean_text(text):
    """Strip chat-template special tokens (e.g. ``<|im_end|>``) and whitespace."""
    if text is None:
        return ""
    return _SPECIAL_TOKEN_RE.sub("", text).strip()


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
# Judges routinely end the last field with a comma; strict json.loads rejects it,
# which would burn all JUDGE_MAX_RETRIES and then drop an otherwise good verdict.
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_LINGUISTIC_DIMENSIONS = ("Fluency", "Relevance", "Correctness")

_DIM_KEY_RES = {}


def _dim_key_re(dim):
    if dim not in _DIM_KEY_RES:
        _DIM_KEY_RES[dim] = re.compile(rf'"{re.escape(dim)}"\s*:', re.IGNORECASE)
    return _DIM_KEY_RES[dim]


_SCORE_KEY_RE = re.compile(r'"score"\s*:\s*"?\s*([-+]?\d*\.?\d+)', re.IGNORECASE)
_HAS_SCORE_KEY_RE = re.compile(r'"score"\s*:', re.IGNORECASE)


def _scan_dim_score(blob, dim, dimensions):
    """Read ``dim``'s score straight out of ``blob``, ignoring JSON validity.

    The window is cut at the next dimension key so a dimension whose own score is
    unreadable cannot borrow the following one's -- without that bound a greedy
    scan happily reports Relevance's number as Fluency's.
    """
    start = _dim_key_re(dim).search(blob)
    if start is None:
        return None
    window = blob[start.end() :]
    for other in dimensions:
        if other.lower() == dim.lower():
            continue
        nxt = _dim_key_re(other).search(window)
        if nxt is not None:
            window = window[: nxt.start()]
    # A "score" key present but unreadable (e.g. the "X.XXXX" placeholder echoed
    # back) must fail rather than fall through to a number inside the reason prose.
    if _HAS_SCORE_KEY_RE.search(window):
        found = _SCORE_KEY_RE.search(window)
        return found.group(1) if found else None
    bare = _SCORE_RE.search(window)
    return bare.group() if bare else None


def _parse_json_scores(text, dimensions, low, high):
    """Extract one score per dimension from a JSON verdict, clamped to [low, high].

    Returns a dict keyed by the names in ``dimensions``, or ``None`` if not every
    dimension yields a number (partial verdicts are dropped rather than silently
    averaged over fewer dimensions).

    Tries strict JSON first, then falls back to a bounded per-dimension scan. The
    fallback is what makes this usable in practice: the rubrics ask the judge to
    justify each score in a ``reason`` string, and it quotes the model's answer
    verbatim there. TOFU answers are full of quoted book titles, so that ``"``
    lands unescaped inside the JSON string and ``json.loads`` fails on an
    otherwise perfectly good verdict. Only the scores are needed, so the scan
    reads them directly and ignores the prose.
    """
    if text is None:
        return None
    match = _JSON_OBJ_RE.search(text)
    if match is None:
        return None
    blob = _TRAILING_COMMA_RE.sub(r"\1", match.group())
    payload = None
    try:
        loaded = json.loads(blob)
        if isinstance(loaded, dict):
            # The rubric fixes the key spelling, but judges vary the casing;
            # match on a lowercased view so "Fluency" and "fluency" both resolve.
            payload = {str(k).lower(): v for k, v in loaded.items()}
    except (ValueError, TypeError):
        payload = None

    scores = {}
    for dim in dimensions:
        raw = None
        if payload is not None:
            entry = payload.get(dim.lower())
            raw = entry.get("score") if isinstance(entry, dict) else entry
        num = _SCORE_RE.search(str(raw)) if raw is not None else None
        text_num = num.group() if num else _scan_dim_score(blob, dim, dimensions)
        if text_num is None:
            return None
        try:
            scores[dim] = float(np.clip(float(text_num), low, high))
        except ValueError:
            return None
    return scores


def parse_linguistic(text):
    """Fluency/Relevance/Correctness scores (0-10) from a JSON verdict."""
    return _parse_json_scores(text, _LINGUISTIC_DIMENSIONS, 0.0, 10.0)


def query_judge(
    user_prompt, model=JUDGE_MODEL, max_retries=JUDGE_MAX_RETRIES, parser=parse_linguistic
):
    """Query the judge with bounded exponential-backoff retries.

    Returns the parsed score, or ``None`` if every attempt fails to produce a
    parseable numeric answer (the instance is then dropped from the aggregate).
    """
    client = get_client(model=model)
    backoff = JUDGE_RETRY_BACKOFF
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=JUDGE_TEMPERATURE,
                max_tokens=JUDGE_MAX_TOKENS,
                extra_body=JUDGE_EXTRA_BODY or None,
            )
            score = parser(response.choices[0].message.content)
            if score is not None:
                return score
            logger.warning(
                "Judge returned an unparseable score (attempt %d/%d)",
                attempt,
                max_retries,
            )
        except Exception as exc:  # network / API errors
            logger.warning(
                "Judge API call failed (attempt %d/%d): %s", attempt, max_retries, exc
            )
        if attempt < max_retries:
            # Jitter: many workers hitting the same rate limit would otherwise
            # retry in lockstep and keep colliding. Backoff is capped so an
            # unparseable generation cannot stall a worker for minutes.
            time.sleep(backoff * (0.5 + np.random.rand()))
            backoff = min(backoff * 2, JUDGE_BACKOFF_MAX)
    logger.warning("Judge scoring failed after %d attempts; marking None", max_retries)
    return None


def query_judge_many(
    prompts,
    model=JUDGE_MODEL,
    max_retries=JUDGE_MAX_RETRIES,
    parser=parse_linguistic,
    workers=JUDGE_WORKERS,
    progress=None,
):
    """Score ``prompts`` concurrently; returns scores in the input order.

    Failed prompts come back as ``None`` (same contract as :func:`query_judge`),
    so the caller can drop them from the aggregate. ``progress`` is an optional
    zero-arg callback invoked once per completed prompt.
    """
    prompts = list(prompts)
    if not prompts:
        return []
    results = [None] * len(prompts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                query_judge,
                prompt,
                model=model,
                max_retries=max_retries,
                parser=parser,
            ): i
            for i, prompt in enumerate(prompts)
        }
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
            if progress is not None:
                progress()
    return results


def linguistic_prompt(question, ground_truth, generation):
    return LINGUISTIC_PROMPT.format(
        inp=clean_text(question),
        truth=clean_text(ground_truth),
        gen=clean_text(generation),
    )


def score_linguistic(
    question, ground_truth, generation, model=JUDGE_MODEL, max_retries=JUDGE_MAX_RETRIES
):
    """Fluency/Relevance/Correctness scores (0-10) for ``generation``.

    Returns a dict keyed by dimension (see :func:`parse_linguistic`), or ``None``
    if the judge never produces a parseable JSON verdict. Higher is better on
    every dimension, including the deliberately inverted Correctness scale.
    """
    return query_judge(
        linguistic_prompt(question, ground_truth, generation),
        model=model,
        max_retries=max_retries,
        parser=parse_linguistic,
    )


def score_linguistic_many(triples, **kwargs):
    """Linguistic score dicts for ``(question, ground_truth, generation)`` triples."""
    return query_judge_many(
        [linguistic_prompt(*t) for t in triples], parser=parse_linguistic, **kwargs
    )


_TOFU_EVAL_ROOT = Path(__file__).resolve().parents[3] / "saves" / "eval"
_TOFU_SOURCE_METRIC = "forget_Q_A_ROUGE"
_TOFU_LINGUISTIC_METRICS = {
    "Fluency": "forget_Q_A_Fluency",
    "Relevance": "forget_Q_A_Relevance",
    "Correctness": "forget_Q_A_Correctness",
}
_TOFU_MODEL_PREFIXES = (
    "tofu_Dream-v0-Instruct-7B_",
    "tofu_LLaDA-8B-Instruct_",
)
_TOFU_FORGET_SPLITS = ("forget10", "forget05", "forget01")
_TOFU_RETAIN_SPLITS = {
    "forget10": "retain90",
    "forget05": "retain95",
    "forget01": "retain99",
}
_TOFU_INTRINSIC_FORGET_ONLY_METHODS = (
    "DiffusionADU",
    "DiffusionBST",
    "DiffusionRDU",
    "DiffusionRepelGenDistill",
)


def _tofu_summary_paths(eval_dir):
    return [
        path
        for path in (
            eval_dir / "TOFU_SUMMARY.json",
            eval_dir / "TOFU_SUMMARY_noprompt.json",
        )
        if path.exists()
    ]


def _is_tofu_forget_only_output(name):
    if not name.startswith(_TOFU_MODEL_PREFIXES):
        return False
    return name.endswith("_forgetonly") or any(
        name.endswith(f"_{method}") for method in _TOFU_INTRINSIC_FORGET_ONLY_METHODS
    )


def discover_tofu_linguistic_targets(eval_root=_TOFU_EVAL_ROOT):
    """Find forget-only outputs plus Dream/LLaDA SFT and retain baselines."""
    eval_root = Path(eval_root)
    targets = set()
    for forget_split in _TOFU_FORGET_SPLITS:
        split_dir = eval_root / forget_split
        if split_dir.is_dir():
            targets.update(
                path
                for path in split_dir.iterdir()
                if path.is_dir()
                and _is_tofu_forget_only_output(path.name)
                and (path / "TOFU_EVAL.json").exists()
            )

        retain_split = _TOFU_RETAIN_SPLITS[forget_split]
        for model_prefix in _TOFU_MODEL_PREFIXES:
            for suffix in (f"retain100_{forget_split}", retain_split):
                path = eval_root / f"{model_prefix}{suffix}"
                if (path / "TOFU_EVAL.json").exists():
                    targets.add(path)
    return sorted(targets)


def _load_tofu_linguistic_entries(eval_dir):
    data = json.loads((eval_dir / "TOFU_EVAL.json").read_text())
    entries = data.get(_TOFU_SOURCE_METRIC, {}).get("value_by_index", {})
    return [
        (str(index), entry)
        for index, entry in entries.items()
        if isinstance(entry, dict)
        and all(entry.get(field) is not None for field in ("input", "ground_truth", "generation"))
    ]


def _write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=4))
    os.replace(temporary, path)


def _tofu_linguistic_complete(eval_dir):
    summaries = _tofu_summary_paths(eval_dir)
    return bool(summaries) and all(
        all(metric in json.loads(path.read_text()) for metric in _TOFU_LINGUISTIC_METRICS.values())
        for path in summaries
    )


def evaluate_tofu_linguistic_directory(eval_dir, progress=None):
    """Judge and persist all forget-question generations in one TOFU result."""
    eval_dir = Path(eval_dir)
    entries = _load_tofu_linguistic_entries(eval_dir)
    verdicts = score_linguistic_many(
        [
            (entry["input"], entry["ground_truth"], entry["generation"])
            for _, entry in entries
        ],
        workers=JUDGE_WORKERS,
        progress=progress,
    )
    aggregates = {}
    for dimension, metric in _TOFU_LINGUISTIC_METRICS.items():
        scores = [None if verdict is None else verdict.get(dimension) for verdict in verdicts]
        valid_scores = [score for score in scores if score is not None]
        aggregate = float(np.mean(valid_scores)) if valid_scores else None
        aggregates[metric] = aggregate
        _write_json_atomic(
            eval_dir / f"{metric}.json",
            {
                "agg_value": aggregate,
                "n_scored": len(valid_scores),
                "value_by_index": {
                    index: {
                        "score": score,
                        "generation": entry["generation"],
                    }
                    for score, (index, entry) in zip(scores, entries)
                },
            },
        )

    scored = {metric: value for metric, value in aggregates.items() if value is not None}
    for summary_path in _tofu_summary_paths(eval_dir):
        summary = json.loads(summary_path.read_text())
        summary.update(scored)
        _write_json_atomic(summary_path, summary)
    return aggregates, len(entries)


def _parse_tofu_linguistic_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all TOFU forget-only outputs and Dream/LLaDA SFT and retain "
            "baselines with the linguistic LLM judge."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Optional evaluation directories")
    parser.add_argument("--eval-root", type=Path, default=_TOFU_EVAL_ROOT)
    parser.add_argument("--list", action="store_true", help="List pending targets and exit")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --list")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run_tofu_linguistic_evaluation(argv=None):
    args = _parse_tofu_linguistic_args(argv)
    targets = (
        sorted(path.resolve() for path in args.paths)
        if args.paths
        else discover_tofu_linguistic_targets(args.eval_root)
    )
    pending = [
        path for path in targets if args.overwrite or not _tofu_linguistic_complete(path)
    ]
    print(
        f"targets={len(targets)} complete={len(targets) - len(pending)} "
        f"pending={len(pending)} judge={JUDGE_MODEL} workers={JUDGE_WORKERS}"
    )
    if args.list or args.dry_run:
        for path in pending:
            try:
                print(path.relative_to(Path.cwd()))
            except ValueError:
                print(path)
        return 0

    from tqdm import tqdm

    work = [(path, _load_tofu_linguistic_entries(path)) for path in pending]
    work = [(path, entries) for path, entries in work if entries]
    with tqdm(total=sum(len(entries) for _, entries in work), unit="call") as progress:
        for eval_dir, entries in work:
            progress.set_description(eval_dir.name[:48])
            aggregates, count = evaluate_tofu_linguistic_directory(
                eval_dir, progress=progress.update
            )
            values = " ".join(
                f"{dimension}={aggregates[metric]:.4f}"
                if aggregates[metric] is not None
                else f"{dimension}=n/a"
                for dimension, metric in _TOFU_LINGUISTIC_METRICS.items()
            )
            print(f"ok {eval_dir} {values} n={count}")
    return 0


if __name__ == "__main__":
    sys.exit(run_tofu_linguistic_evaluation())
