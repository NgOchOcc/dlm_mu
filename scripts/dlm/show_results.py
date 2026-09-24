"""Render TOFU evaluation summaries as comparable terminal tables."""

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_EVAL_DIR = Path("saves/ablation")
ERASING_KNOWLEDGE_COMPONENTS = {
    "forget_Q_A_Prob": "lower",
    "forget_Q_A_ROUGE": "lower",
    "forget_truth_ratio": "higher",
}
ERASING_JUDGE_COMPONENTS = {
    "forget_Q_A_Relevance": "judge_higher",
    "forget_Q_A_Correctness": "judge_lower",
}
UTILITY_COMPONENT_GROUPS = (
    {
        "retain_Q_A_Prob": "higher",
        "retain_Q_A_ROUGE": "higher",
    },
    {
        "ra_Q_A_ROUGE": "higher",
    },
    {
        "wf_Q_A_ROUGE": "higher",
    },
)
LINGUISTIC_COMPONENTS = {
    "forget_Q_A_Fluency": "judge_higher",
    "retain_Q_A_Fluency": "judge_higher",
    "ra_Q_A_Fluency": "judge_higher",
    "wf_Q_A_Fluency": "judge_higher",
}

def _hmean(values):
    if not values or any(not isinstance(value, (int, float)) for value in values):
        return None
    if min(values) <= 0.0:
        return 0.0
    return float(len(values) / sum(1.0 / value for value in values))


def _amean(values):
    if any(not isinstance(value, (int, float)) for value in values):
        return None
    return float(sum(values) / len(values))


def _score_unit(value, direction):
    if not isinstance(value, (int, float)):
        return None
    if direction == "higher":
        return max(0.0, min(1.0, value))
    if direction == "lower":
        return max(0.0, min(1.0, 1.0 - value))
    if direction == "judge_higher":
        return max(0.0, min(1.0, value / 10.0))
    if direction == "judge_lower":
        return max(0.0, min(1.0, (10.0 - value) / 10.0))
    raise ValueError(f"unknown metric direction: {direction}")


def _component_hmean(data, components, skip_missing=False):
    values = [
        _score_unit(data.get(key), direction)
        for key, direction in components.items()
    ]
    if skip_missing:
        values = [value for value in values if value is not None]
        if not values:
            return None
    return _hmean(values)


def _component_amean(data, components):
    values = [
        _score_unit(data.get(key), direction)
        for key, direction in components.items()
    ]
    return _amean(values)


GROUPS = [
    (
        "Final Scores",
        [
            ("erase_quality", "EQ"),
            ("model_utility", "MU"),
            ("linguistic_quality", "LQ"),
            ("final_average", "AVG"),
        ],
    ),
]

DETAIL_GROUPS = [
    ("Forget", [
        ("forget_Q_A_Prob", "f_P"),
        ("forget_Q_A_ROUGE", "f_R"),
        ("forget_truth_ratio", "f_TR"),
        ("forget_Q_A_Relevance", "f_Rel"),
        ("forget_Q_A_Correctness", "f_Cor"),
        ("forget_Q_A_Fluency", "f_Flu"),
    ]),
    ("Retain", [
        ("retain_Q_A_Prob", "ret_P"),
        ("retain_Q_A_ROUGE", "ret_R"),
        ("retain_Q_A_Relevance", "ret_Rel"),
        ("retain_Q_A_Correctness", "ret_Cor"),
        ("retain_Q_A_Fluency", "ret_Flu"),
    ]),
    ("Real Authors", [
        ("ra_Q_A_ROUGE", "ra_R"),
        ("ra_Q_A_Relevance", "ra_Rel"),
        ("ra_Q_A_Correctness", "ra_Cor"),
        ("ra_Q_A_Fluency", "ra_Flu"),
    ]),
    ("World Facts", [
        ("wf_Q_A_ROUGE", "wf_R"),
        ("wf_Q_A_Relevance", "wf_Rel"),
        ("wf_Q_A_Correctness", "wf_Cor"),
        ("wf_Q_A_Fluency", "wf_Flu"),
    ]),
    ("Final Scores", [
        ("erase_quality", "EQ"),
        ("model_utility", "MU"),
        ("linguistic_quality", "LQ"),
        ("final_average", "AVG"),
    ]),
]

JUDGE_METRICS = {
    "forget_Q_A_Fluency",
    "forget_Q_A_Relevance",
    "forget_Q_A_Correctness",
    "retain_Q_A_Fluency",
    "retain_Q_A_Relevance",
    "retain_Q_A_Correctness",
    "ra_Q_A_Fluency",
    "ra_Q_A_Relevance",
    "ra_Q_A_Correctness",
    "wf_Q_A_Fluency",
    "wf_Q_A_Relevance",
    "wf_Q_A_Correctness",
}
FINAL_SCORE_METRICS = {
    "erase_quality",
    "model_utility",
    "linguistic_quality",
    "final_average",
}
METHOD_ORDER = {
    "GA": 4,
    "GradAscent": 4,
    "GradDiff": 5,
    "NPO": 6,
    "SimNPO": 7,
    "WGA": 8,
}

FORGET_ONLY_SUFFIXES = ("_forgetonly", "_forget_only")
INHERENT_FORGET_ONLY = {"BST"}

DISPLAY_METHOD = {
    "GradAscent": "GA",
}


def _split_forget_only(core):
    """(is_forget_only, base_method): strip a trailing forget-only suffix, if any."""
    for sfx in FORGET_ONLY_SUFFIXES:
        if core.endswith(sfx):
            return True, core[: -len(sfx)]
    return False, core


def _is_inherent_forget_only(base_method):
    """Retain-free-by-design methods, plus their hyper-parameter sweep arms."""
    return base_method in INHERENT_FORGET_ONLY or any(
        base_method.startswith(f"{m}_") for m in INHERENT_FORGET_ONLY
    )


SIMPLE_METHODS = set(METHOD_ORDER)


def _is_simple_method(base_method):
    """True for a clean baseline method."""
    return base_method in SIMPLE_METHODS


# Optional epoch tags used by training/evaluation scripts. They are stripped
# before method categorisation but retained in the displayed row label.
_EPOCHS_RE = re.compile(r"_(?:(\d+)epochs|ep(\d+))$")


def _strip_epochs(label):
    return _EPOCHS_RE.sub("", label)


def _epochs_of(label):
    match = _EPOCHS_RE.search(label)
    return int(match.group(1) or match.group(2)) if match else 0


def _epoch_label(label):
    epoch = _epochs_of(label)
    return f"_ep{epoch:02d}" if epoch else ""


def _method_parts(label):
    return _split_forget_only(_strip_epochs(label.rsplit("/", 1)[-1]))


def _normalized_name(name):
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def _method_matches(method, selected_method):
    if selected_method is None:
        return True
    _, method_name = _method_parts(method)
    _, selected_name = _method_parts(selected_method)
    method_name = DISPLAY_METHOD.get(method_name, method_name)
    selected_name = DISPLAY_METHOD.get(selected_name, selected_name)
    return _normalized_name(method_name).startswith(_normalized_name(selected_name))


def _model_matches(model, selected_model):
    if selected_model is None:
        return True
    return _normalized_name(model).startswith(_normalized_name(selected_model))


FORGET_SPLITS = ["forget01", "forget05", "forget10"]
RETAIN_FOR = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}
EXCLUDE_METHODS = {"NullAnchor", "AMTU", "TMOPU", "FCP", "CFTMOPU", "OCPE"}
EXCLUDE_METHOD_PREFIXES = (
    "SimpleTSD",
    "TSD",
    "UniformTSD",
    "UTSD",
    "SBRU",
    "Rollout",
    "ConceptSubspace",
    "ConcepSubspace",
    "CSD",
)
EXCLUDE_SUFFIXES = ("_oldbase",)


def _is_excluded_method(base_method):
    normalized = _normalized_name(base_method)
    return base_method in EXCLUDE_METHODS or any(
        normalized.startswith(_normalized_name(prefix))
        for prefix in EXCLUDE_METHOD_PREFIXES
    )


def _is_diagnostic(name):
    return name.endswith(EXCLUDE_SUFFIXES) or any(
        f"{sfx}_" in name for sfx in EXCLUDE_SUFFIXES
    )

_COLOR = sys.stdout.isatty()
WHITE = "\033[97m" if _COLOR else ""
TOP_ONE = "\033[1;92m" if _COLOR else ""
TOP_TWO = "\033[1;93m" if _COLOR else ""
RESET = "\033[0m" if _COLOR else ""
LOWER_BETTER = {
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "forget_Q_A_Correctness",
}

def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# Which mean combines the TOFU components. "harmonic" is the original: it
# punishes the weakest component, so a single zero drives the whole score to
# zero. "arithmetic" treats the components as independent and matches how the
# RWKU scores are computed. Set by --mean; load_summary() reads it, so it must
# be assigned before any summary is loaded.
TOFU_MEAN = "harmonic"


def _combine(values):
    return _amean(values) if TOFU_MEAN == "arithmetic" else _hmean(values)


def _component_combine(data, components, skip_missing=False):
    if TOFU_MEAN == "arithmetic":
        values = [
            _score_unit(data.get(key), direction)
            for key, direction in components.items()
        ]
        if skip_missing:
            values = [value for value in values if value is not None]
            if not values:
                return None
        return _amean(values)
    return _component_hmean(data, components, skip_missing=skip_missing)


def load_summary(eval_dir):
    if not eval_dir.is_dir():
        return None
    noprompt = eval_dir / "TOFU_SUMMARY_noprompt.json"
    summary = noprompt if noprompt.exists() else eval_dir / "TOFU_SUMMARY.json"
    if not summary.exists():
        return None
    data = _read_json(summary)
    if data is None:
        return None
    erasing_knowledge = _component_combine(data, ERASING_KNOWLEDGE_COMPONENTS)
    erasing_judge = _component_amean(data, ERASING_JUDGE_COMPONENTS)
    erase_quality = _combine([erasing_knowledge, erasing_judge])
    if erase_quality is not None:
        data["erase_quality"] = erase_quality

    dataset_utilities = [
        _component_combine(data, components, skip_missing=True)
        for components in UTILITY_COMPONENT_GROUPS
    ]
    dataset_utilities = [value for value in dataset_utilities if value is not None]
    model_utility = _combine(dataset_utilities)
    if model_utility is not None:
        data["model_utility"] = model_utility

    linguistic_quality = _component_amean(data, LINGUISTIC_COMPONENTS)
    if linguistic_quality is not None:
        data["linguistic_quality"] = linguistic_quality

    if all(
        isinstance(value, (int, float))
        for value in (erase_quality, model_utility, linguistic_quality)
    ):
        data["final_average"] = float(
            (erase_quality + model_utility + linguistic_quality) / 3.0
        )

    return data


def _forget_split_for(eval_dir, eval_root):
    relative = eval_dir.relative_to(eval_root)
    for part in relative.parts[:-1]:
        if part in FORGET_SPLITS:
            return part

    config = eval_dir / ".hydra/config.yaml"
    if not config.exists():
        return None
    try:
        text = config.read_text()
    except OSError:
        return None
    match = re.search(
        r"^\s*forget_split:\s*(forget(?:01|05|10))\s*$",
        text,
        re.MULTILINE,
    )
    return match.group(1) if match else None


def discover_method_runs(eval_root):
    runs = []
    for summary in eval_root.rglob("TOFU_SUMMARY.json"):
        eval_dir = summary.parent
        match = re.match(r"tofu_(.+?)_Diffusion(.+)$", eval_dir.name)
        if not match:
            continue
        forget_split = _forget_split_for(eval_dir, eval_root)
        if forget_split is not None:
            runs.append((forget_split, match.group(1), match.group(2), eval_dir))
    return sorted(runs, key=lambda item: str(item[3]))


def _run_source_label(eval_dir, eval_root, forget_split):
    parent = eval_dir.parent.relative_to(eval_root)
    return "" if parent in {Path("."), Path(forget_split)} else f"{parent.as_posix()}/"


def discover_models(eval_root, method_runs):
    models = {model for _, model, _, _ in method_runs}
    for eval_dir in eval_root.glob("tofu_*_retain*"):
        match = re.match(r"tofu_(.+)_retain\d+", eval_dir.name)
        if match:
            models.add(match.group(1))
    return sorted(model for model in models if not _is_diagnostic(model))


def collect_rows(
    eval_root,
    method_runs,
    model,
    forget_split,
    mode="both",
    selected_method=None,
):
    rows = []
    base = load_summary(eval_root / f"tofu_{model}_retain100_{forget_split}")
    if base is not None:
        rows.append(("base", base))

    retain = RETAIN_FOR[forget_split]
    retain_data = load_summary(eval_root / f"tofu_{model}_{retain}")
    if retain_data is not None:
        rows.append((retain, retain_data))

    methods = []
    for split, run_model, method, eval_dir in method_runs:
        if split != forget_split or run_model != model:
            continue
        if not _method_matches(method, selected_method):
            continue
        is_forget_only, base_method = _method_parts(method)
        if _is_excluded_method(base_method) or _is_diagnostic(method):
            continue
        if mode == "forget_only" and not (is_forget_only or _is_inherent_forget_only(base_method)):
            continue
        # `simple`: only clean baselines, using their forget-only arms.
        if mode == "simple" and not (
            _is_simple_method(base_method)
            and (is_forget_only or _is_inherent_forget_only(base_method))
        ):
            continue
        data = load_summary(eval_dir)
        if data is not None:
            source = _run_source_label(eval_dir, eval_root, forget_split)
            display_method = (
                f"{base_method}{_epoch_label(method)}"
                if mode == "forget_only"
                else method
            )
            display_method = DISPLAY_METHOD.get(display_method, display_method)
            methods.append((f"{source}{display_method}", data))
    methods.sort(key=lambda r: _method_rank(r[0]))
    if selected_method is not None and not methods:
        return []
    rows.extend(methods)
    return rows


LABEL_W = 14
COL_W = 8
SEP = " │ "

def _group_header(groups):
    group = SEP.join(f"{name:^{COL_W * len(cols)}}" for name, cols in groups)
    metric = SEP.join("".join(f"{h:>{COL_W}}" for _, h in cols) for _, cols in groups)
    return group, metric


def _render_block(data, col_ranks, groups, judge_metrics, force_white=False):
    """Render one model's cells for a row. force_white tags the whole block white
    (used for the base/retain reference rows) instead of method ranking."""
    blocks = []
    for _, cols in groups:
        block = ""
        for key, _ in cols:
            value = data.get(key)
            if not isinstance(value, (int, float)):
                block += f"{'-':>{COL_W}}"
                continue
            if key in FINAL_SCORE_METRICS:
                display_value = value
            elif key in judge_metrics:
                display_value = value / 10.0
            else:
                display_value = value
            text = f"{display_value:{COL_W}.2f}"
            rank = col_ranks.get(key, {}).get(value)
            if not force_white and rank == 1:
                text = f"{TOP_ONE}{text}{RESET}"
            elif not force_white and rank == 2:
                text = f"{TOP_TWO}{text}{RESET}"
            block += text
        blocks.append(block)
    rendered = SEP.join(blocks)
    if force_white:
        return f"{WHITE}{rendered}{RESET}"
    return rendered


def _method_rank(label):
    is_forget_only, base_method = _method_parts(label)
    return (
        METHOD_ORDER.get(base_method, 99),
        base_method,
        is_forget_only,
        _epochs_of(label),
    )


def _label_order(label, retain_name, order_map=None):
    if label == "base":
        return (0, "", False, 0)
    if label == retain_name:
        return (1, "", False, 0)
    if order_map is not None:
        # RWKU/MUSE supply their own ordering, and their labels carry RUN_TAG
        # suffixes that _method_rank's TOFU-specific parsing does not know.
        rank, _ = _ranked_by_order(label, order_map)
        return (2 + rank, label, False, 0)
    order, base_method, is_forget_only, epochs = _method_rank(label)
    return (2 + order, base_method, is_forget_only, epochs)


def _row(segments, widths):
    cells = [f" {seg}{' ' * (w - _visible(seg))} " for seg, w in zip(segments, widths)]
    return "│" + "│".join(cells) + "│"

def _rule(left, mid, right, widths):
    return left + mid.join("─" * (w + 2) for w in widths) + right

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _visible(s):
    return len(_ANSI.sub("", s))


def print_table(title, model_rows, models, retain_name, groups, cols, judge_metrics,
                lower_better, order_map=None):
    group_one, metric_one = _group_header(groups)
    block_w = len(metric_one)
    all_labels = set().union(*(rows.keys() for rows in model_rows.values()))
    labels = sorted(all_labels, key=lambda lbl: _label_order(lbl, retain_name, order_map))
    label_w = max([LABEL_W] + [len(lbl) for lbl in labels])
    widths = [label_w] + [block_w] * len(models)
    col_ranks_per_model = {}
    for model in models:
        rows = model_rows.get(model, {})
        col_ranks = {}
        for key, _ in cols:
            values = [
                data[key]
                for label, data in rows.items()
                if label not in {"base", retain_name}
                and isinstance(data.get(key), (int, float))
            ]
            if values:
                ranked = sorted(set(values), reverse=key not in lower_better)
                col_ranks[key] = {
                    value: rank for rank, value in enumerate(ranked[:2], start=1)
                }
        col_ranks_per_model[model] = col_ranks

    print(f"\n## {title}")
    print(_rule("╭", "┬", "╮", widths))
    print(_row([""] + [f"{model:^{block_w}}" for model in models], widths))
    print(_row([""] + [group_one] * len(models), widths))
    print(_row(["method"] + [metric_one] * len(models), widths))
    print(_rule("├", "┼", "┤", widths))
    for label in labels:
        is_ref = label == "base" or label == retain_name
        if is_ref:
            display_label = f"{WHITE}{label}{RESET}"
        else:
            display_label = label
        segments = [display_label]
        for model in models:
            data = model_rows.get(model, {}).get(label)
            if data is None:
                segments.append(" " * block_w)
            else:
                segments.append(
                    _render_block(
                        data, col_ranks_per_model[model], groups, judge_metrics,
                        force_white=is_ref,
                    )
                )
        print(_row(segments, widths))
    print(_rule("╰", "┴", "╯", widths))


def run_tofu(args):
    groups = DETAIL_GROUPS if args.detail else GROUPS
    cols = [(key, header) for _, group_cols in groups for key, header in group_cols]
    detail_label = " [DETAIL]" if args.detail else ""
    splits = [args.split] if args.split else FORGET_SPLITS
    method_runs = discover_method_runs(args.eval_dir)
    models = [
        model
        for model in discover_models(args.eval_dir, method_runs)
        if _model_matches(model, args.model)
    ]

    for forget_split in splits:
        model_rows = {}
        present = []
        for model in models:
            block = dict(
                collect_rows(
                    args.eval_dir,
                    method_runs,
                    model,
                    forget_split,
                    mode=args.mode,
                    selected_method=args.method,
                )
            )
            if not block:
                continue
            present.append(model)
            model_rows[model] = block
        if model_rows:
            if args.detail:
                for model in present:
                    print_table(
                        f"{forget_split} — {model}{detail_label}",
                        {model: model_rows[model]}, [model], RETAIN_FOR[forget_split],
                        groups, cols, JUDGE_METRICS, LOWER_BETTER,
                    )
            else:
                print_table(
                    f"{forget_split}", model_rows, present,
                    RETAIN_FOR[forget_split],
                    groups, cols, JUDGE_METRICS, LOWER_BETTER,
                )

    print()


# ---------------------------------------------------------------------------
# RWKU
# ---------------------------------------------------------------------------

DEFAULT_RWKU_EVAL_DIR = Path("eval/RWKU")

RWKU_ENTITIES = [
    "Stephen_King",
    "Confucius",
    "Bruce_Lee",
    "Warren_Buffett",
    "Christina_Aguilera",
]

RWKU_FORGET_KNOWLEDGE_COMPONENTS = {
    "forget_level1_ROUGE": "lower",
    "forget_level2_ROUGE": "lower",
    "forget_level3_ROUGE": "lower",
}
RWKU_FORGET_JUDGE_COMPONENTS = {
    "forget_level1_Relevance": "judge_higher",
    "forget_level2_Relevance": "judge_higher",
    "forget_level3_Relevance": "judge_higher",
    "forget_level1_Correctness": "judge_lower",
    "forget_level2_Correctness": "judge_lower",
    "forget_level3_Correctness": "judge_lower",
}
RWKU_FORGET_LINGUISTIC_COMPONENTS = {
    "forget_level1_Fluency": "judge_higher",
    "forget_level2_Fluency": "judge_higher",
    "forget_level3_Fluency": "judge_higher",
}
# Neighbor facts should be *retained*, so knowledge/correctness are "higher".
RWKU_NEIGHBOR_KNOWLEDGE_COMPONENTS = {
    "neighbor_level1_ROUGE": "higher",
    "neighbor_level2_ROUGE": "higher",
}
RWKU_NEIGHBOR_JUDGE_COMPONENTS = {
    "neighbor_level1_Relevance": "judge_higher",
    "neighbor_level2_Relevance": "judge_higher",
    "neighbor_level1_Correctness": "judge_higher",
    "neighbor_level2_Correctness": "judge_higher",
}
RWKU_NEIGHBOR_LINGUISTIC_COMPONENTS = {
    "neighbor_level1_Fluency": "judge_higher",
    "neighbor_level2_Fluency": "judge_higher",
}

RWKU_RAW_KEYS = (
    set(RWKU_FORGET_KNOWLEDGE_COMPONENTS)
    | set(RWKU_FORGET_JUDGE_COMPONENTS)
    | set(RWKU_FORGET_LINGUISTIC_COMPONENTS)
    | set(RWKU_NEIGHBOR_KNOWLEDGE_COMPONENTS)
    | set(RWKU_NEIGHBOR_JUDGE_COMPONENTS)
    | set(RWKU_NEIGHBOR_LINGUISTIC_COMPONENTS)
    | {"mmlu"}
)

# Raw judge metrics are on a 0-10 scale; used only in --detail tables.
RWKU_JUDGE_METRICS = (
    set(RWKU_FORGET_JUDGE_COMPONENTS)
    | set(RWKU_FORGET_LINGUISTIC_COMPONENTS)
    | set(RWKU_NEIGHBOR_JUDGE_COMPONENTS)
    | set(RWKU_NEIGHBOR_LINGUISTIC_COMPONENTS)
)
# Raw components where a lower value is better; used only in --detail tables.
RWKU_LOWER_BETTER = {
    "forget_level1_ROUGE",
    "forget_level2_ROUGE",
    "forget_level3_ROUGE",
    "forget_level1_Correctness",
    "forget_level2_Correctness",
    "forget_level3_Correctness",
}

RWKU_GROUPS = [
    ("Forget", [("forget_quality", "EQ"), ("forget_linguistic", "LQ")]),
    ("Neighbor", [("neighbor_quality", "MU"), ("neighbor_linguistic", "LQ")]),
    ("MMLU", [("mmlu", "ACC")]),
]

RWKU_DETAIL_GROUPS = [
    ("Forget", [
        ("forget_level1_ROUGE", "f1_R"),
        ("forget_level2_ROUGE", "f2_R"),
        ("forget_level3_ROUGE", "f3_R"),
        ("forget_level1_Relevance", "f1_Rel"),
        ("forget_level2_Relevance", "f2_Rel"),
        ("forget_level3_Relevance", "f3_Rel"),
        ("forget_level1_Correctness", "f1_Cor"),
        ("forget_level2_Correctness", "f2_Cor"),
        ("forget_level3_Correctness", "f3_Cor"),
        ("forget_level1_Fluency", "f1_Flu"),
        ("forget_level2_Fluency", "f2_Flu"),
        ("forget_level3_Fluency", "f3_Flu"),
    ]),
    ("Neighbor", [
        ("neighbor_level1_ROUGE", "n1_R"),
        ("neighbor_level2_ROUGE", "n2_R"),
        ("neighbor_level1_Relevance", "n1_Rel"),
        ("neighbor_level2_Relevance", "n2_Rel"),
        ("neighbor_level1_Correctness", "n1_Cor"),
        ("neighbor_level2_Correctness", "n2_Cor"),
        ("neighbor_level1_Fluency", "n1_Flu"),
        ("neighbor_level2_Fluency", "n2_Flu"),
    ]),
    ("MMLU", [("mmlu", "ACC")]),
    ("Final Scores", [
        ("forget_quality", "F_EQ"),
        ("forget_linguistic", "F_LQ"),
        ("neighbor_quality", "N_MU"),
        ("neighbor_linguistic", "N_LQ"),
        ("mmlu", "ACC"),
    ]),
]

# Criteria averaged across levels; used only in --detail=criteria tables.
RWKU_CRITERIA_JUDGE_METRICS = {
    "forget_Relevance",
    "forget_Correctness",
    "forget_Fluency",
    "neighbor_Relevance",
    "neighbor_Correctness",
    "neighbor_Fluency",
}
RWKU_CRITERIA_LOWER_BETTER = {"forget_ROUGE", "forget_Correctness"}

RWKU_CRITERIA_GROUPS = [
    ("Forget", [
        ("forget_ROUGE", "ROUGE"),
        ("forget_Relevance", "Rel"),
        ("forget_Correctness", "Cor"),
        ("forget_Fluency", "Flu"),
    ]),
    ("Neighbor", [
        ("neighbor_ROUGE", "ROUGE"),
        ("neighbor_Relevance", "Rel"),
        ("neighbor_Correctness", "Cor"),
        ("neighbor_Fluency", "Flu"),
    ]),
    ("MMLU", [("mmlu", "ACC")]),
    ("Final Scores", [
        ("forget_quality", "F_EQ"),
        ("forget_linguistic", "F_LQ"),
        ("neighbor_quality", "N_MU"),
        ("neighbor_linguistic", "N_LQ"),
        ("mmlu", "ACC"),
    ]),
]

# Display order: gradient-based baselines first, then the selective methods.
# Anything unlisted sorts last, alphabetically.
RWKU_METHOD_ORDER = {
    "GA": 0, "GD": 1, "NPO": 2, "SimNPO": 3, "WGA": 4, "BST": 5,
    "SelectiveUnlearning": 6, "SDU": 7, "DiRU": 7,
}

# The trailing group captures the optional RUN_TAG that hyperparameter sweeps
# append (e.g. "..._DiffusionSDU_b10_pw0.1"); without it those runs were
# skipped entirely, so a tuning sweep produced an empty table.
_RWKU_RUN_RE = re.compile(r"^(.+?)_(base|Diffusion[A-Za-z0-9]+)((?:_[^_]+)*)$")
_RWKU_ENTITY_DIR_RE = re.compile(r"^\d+_(.+)$")


def discover_rwku_runs(eval_root):
    """[(model, method, run_dir), ...] for eval/RWKU/{model}_{method}/train_v1."""
    runs = []
    if not eval_root.is_dir():
        return runs
    for run_dir in eval_root.glob("*/train_v1"):
        match = _RWKU_RUN_RE.match(run_dir.parent.name)
        if not match:
            continue
        # group(3) is the optional RUN_TAG suffix; tolerate an older two-group
        # pattern so a partially updated checkout still runs.
        tag = match.group(3) if match.re.groups >= 3 else ""
        runs.append((match.group(1), match.group(2) + (tag or ""), run_dir))
    return sorted(runs)


def load_rwku_run(run_dir, entities=RWKU_ENTITIES):
    """Average each raw metric across `entities`' RWKU_SUMMARY/LLM_JUDGE files."""
    entity_dirs = {}
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        match = _RWKU_ENTITY_DIR_RE.match(child.name)
        if match:
            entity_dirs[match.group(1)] = child

    raw_values = {key: [] for key in RWKU_RAW_KEYS}
    for entity in entities:
        entity_dir = entity_dirs.get(entity)
        if entity_dir is None:
            continue
        summary = _read_json(entity_dir / "RWKU_SUMMARY.json") or {}
        judge = _read_json(entity_dir / "LLM_JUDGE_COMPLETE.json") or {}
        merged = dict(summary)
        if isinstance(judge.get("metrics"), dict):
            merged.update(judge["metrics"])
        for key in RWKU_RAW_KEYS:
            value = merged.get(key)
            if isinstance(value, (int, float)):
                raw_values[key].append(value)

    if not any(raw_values.values()):
        return None
    return {
        key: (sum(values) / len(values) if values else None)
        for key, values in raw_values.items()
    }


def _levels_mean(data, keys):
    values = [data.get(key) for key in keys]
    values = [value for value in values if isinstance(value, (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


# Which mean joins the classic and judge halves of each RWKU score, mirroring
# TOFU's and MUSE's --mean. "arithmetic" is the default; "harmonic" punishes
# the weaker half, so a model that erases the answer but stops responding
# coherently cannot score as well as one that stays on topic.
RWKU_MEAN = "arithmetic"


def _rwku_combine(values):
    return _hmean(values) if RWKU_MEAN == "harmonic" else _amean(values)


def compute_rwku_scores(averaged):
    data = dict(averaged)

    data["forget_ROUGE"] = _levels_mean(
        data, ["forget_level1_ROUGE", "forget_level2_ROUGE", "forget_level3_ROUGE"]
    )
    data["forget_Relevance"] = _levels_mean(
        data,
        ["forget_level1_Relevance", "forget_level2_Relevance", "forget_level3_Relevance"],
    )
    data["forget_Correctness"] = _levels_mean(
        data,
        ["forget_level1_Correctness", "forget_level2_Correctness", "forget_level3_Correctness"],
    )
    data["forget_Fluency"] = _levels_mean(
        data, ["forget_level1_Fluency", "forget_level2_Fluency", "forget_level3_Fluency"]
    )
    data["neighbor_ROUGE"] = _levels_mean(
        data, ["neighbor_level1_ROUGE", "neighbor_level2_ROUGE"]
    )
    data["neighbor_Relevance"] = _levels_mean(
        data, ["neighbor_level1_Relevance", "neighbor_level2_Relevance"]
    )
    data["neighbor_Correctness"] = _levels_mean(
        data, ["neighbor_level1_Correctness", "neighbor_level2_Correctness"]
    )
    data["neighbor_Fluency"] = _levels_mean(
        data, ["neighbor_level1_Fluency", "neighbor_level2_Fluency"]
    )

    # Forget: the classic metric and the two judge metrics are combined as two
    # halves rather than three equal terms, so ROUGE -- the only objective
    # signal -- is not outvoted 2:1 by the judge. Higher EQ = better erasure.
    #
    #   knowledge = 1 - ROUGE
    #   judge     = average(Relevance, 1 - Correctness)
    #   EQ        = combine(knowledge, judge)
    #
    # Relevance counts upward on purpose: a model that erased the fact but now
    # answers off-topic has broken rather than forgotten, and should score
    # below one that still answers the question without the memorised content.
    forget_level_scores = []
    for level in (1, 2, 3):
        rouge = _score_unit(data.get(f"forget_level{level}_ROUGE"), "lower")
        relevance = _score_unit(data.get(f"forget_level{level}_Relevance"), "judge_higher")
        correctness = _score_unit(data.get(f"forget_level{level}_Correctness"), "judge_lower")
        level_score = _rwku_combine([rouge, _amean([relevance, correctness])])
        if level_score is not None:
            forget_level_scores.append(level_score)
    forget_quality = _amean(forget_level_scores) if forget_level_scores else None
    if forget_quality is not None:
        data["forget_quality"] = forget_quality
    forget_linguistic = _component_amean(data, RWKU_FORGET_LINGUISTIC_COMPONENTS)
    if forget_linguistic is not None:
        data["forget_linguistic"] = forget_linguistic

    # Retain: same two-half split, all terms kept as-is (higher is better).
    #   MU = combine(ROUGE, average(Relevance, Correctness))
    neighbor_level_scores = []
    for level in (1, 2):
        rouge = _score_unit(data.get(f"neighbor_level{level}_ROUGE"), "higher")
        relevance = _score_unit(data.get(f"neighbor_level{level}_Relevance"), "judge_higher")
        correctness = _score_unit(data.get(f"neighbor_level{level}_Correctness"), "judge_higher")
        level_score = _rwku_combine([rouge, _amean([relevance, correctness])])
        if level_score is not None:
            neighbor_level_scores.append(level_score)
    neighbor_quality = _amean(neighbor_level_scores) if neighbor_level_scores else None
    if neighbor_quality is not None:
        data["neighbor_quality"] = neighbor_quality
    neighbor_linguistic = _component_amean(data, RWKU_NEIGHBOR_LINGUISTIC_COMPONENTS)
    if neighbor_linguistic is not None:
        data["neighbor_linguistic"] = neighbor_linguistic

    mmlu = data.get("mmlu")
    if isinstance(mmlu, (int, float)):
        data["mmlu"] = max(0.0, min(1.0, mmlu))

    return data


def _ranked_by_order(label, order):
    """Rank a method label, ignoring any RUN_TAG suffix.

    Sweep runs arrive as "SDU_b10_pw0.1"; ranking on the bare method name keeps
    every variant of a method together instead of scattering them alphabetically
    among the unlisted names.
    """
    base = label
    while base:
        if base in order:
            return (order[base], label)
        if "_" not in base:
            break
        base = base.rsplit("_", 1)[0]
    return (99, label)


def _rwku_method_rank(label):
    return _ranked_by_order(label, RWKU_METHOD_ORDER)


def collect_rwku_rows(runs, model):
    rows = []
    base_run = next(
        (run_dir for m, method, run_dir in runs if m == model and method == "base"),
        None,
    )
    if base_run is not None:
        averaged = load_rwku_run(base_run)
        if averaged is not None:
            rows.append(("base", compute_rwku_scores(averaged)))

    methods = []
    for m, method, run_dir in runs:
        if m != model or method == "base":
            continue
        averaged = load_rwku_run(run_dir)
        if averaged is None:
            continue
        label = method[len("Diffusion"):] if method.startswith("Diffusion") else method
        methods.append((label, compute_rwku_scores(averaged)))
    methods.sort(key=lambda r: _rwku_method_rank(r[0]))
    rows.extend(methods)
    return rows


def run_rwku(args):
    eval_root = args.eval_dir
    runs = discover_rwku_runs(eval_root)
    models = [
        model
        for model in sorted({m for m, _, _ in runs})
        if _model_matches(model, args.model)
    ]

    if args.detail == "criteria":
        groups = RWKU_CRITERIA_GROUPS
        judge_metrics = RWKU_CRITERIA_JUDGE_METRICS
        lower_better = RWKU_CRITERIA_LOWER_BETTER
    elif args.detail:
        groups = RWKU_DETAIL_GROUPS
        judge_metrics = RWKU_JUDGE_METRICS
        lower_better = RWKU_LOWER_BETTER
    else:
        groups = RWKU_GROUPS
        judge_metrics = set()
        lower_better = set()
    cols = [(key, header) for _, group_cols in groups for key, header in group_cols]
    detail_label = f" [DETAIL:{args.detail}]" if args.detail else ""

    model_rows = {}
    present = []
    for model in models:
        rows = collect_rwku_rows(runs, model)
        if args.method is not None:
            rows = [
                (label, data)
                for label, data in rows
                if label == "base"
                or _normalized_name(label).startswith(_normalized_name(args.method))
            ]
        block = dict(rows)
        if not block:
            continue
        present.append(model)
        model_rows[model] = block

    if not model_rows:
        print("\nNo RWKU results found.")
        return

    if args.detail:
        for model in present:
            print_table(
                f"RWKU — {model}{detail_label}",
                {model: model_rows[model]}, [model], None,
                groups, cols, judge_metrics, lower_better,
                order_map=RWKU_METHOD_ORDER,
            )
    else:
        print_table(
            f"RWKU{detail_label}", model_rows, present, None,
            groups, cols, judge_metrics, lower_better,
            order_map=RWKU_METHOD_ORDER,
        )
    print()



# ---------------------------------------------------------------- MUSE ------

DEFAULT_MUSE_EVAL_DIR = Path("saves/eval")

# saves/eval/muse_{MODEL}_{split}_{method}[_{tag}]/MUSE_SUMMARY.json
_MUSE_RUN_RE = re.compile(r"^muse_(.+?)_(News|Books)_(.+)$")

# Verbatim and knowledge memorisation of the forget set should both drop;
# retain knowledge should hold.
MUSE_FORGET_KNOWLEDGE_COMPONENTS = {
    "forget_knowmem_ROUGE": "lower",
    "forget_verbmem_ROUGE": "lower",
}
MUSE_FORGET_JUDGE_COMPONENTS = {
    "forget_knowmem_Relevance": "judge_higher",
    "forget_verbmem_Relevance": "judge_higher",
    "forget_knowmem_Correctness": "judge_lower",
    "forget_verbmem_Correctness": "judge_lower",
}
MUSE_FORGET_LINGUISTIC_COMPONENTS = {
    "forget_knowmem_Fluency": "judge_higher",
    "forget_verbmem_Fluency": "judge_higher",
}
MUSE_RETAIN_KNOWLEDGE_COMPONENTS = {"retain_knowmem_ROUGE": "higher"}
MUSE_RETAIN_JUDGE_COMPONENTS = {
    "retain_knowmem_Relevance": "judge_higher",
    "retain_knowmem_Correctness": "judge_higher",
}
MUSE_RETAIN_LINGUISTIC_COMPONENTS = {"retain_knowmem_Fluency": "judge_higher"}

MUSE_RAW_KEYS = (
    set(MUSE_FORGET_KNOWLEDGE_COMPONENTS)
    | set(MUSE_FORGET_JUDGE_COMPONENTS)
    | set(MUSE_FORGET_LINGUISTIC_COMPONENTS)
    | set(MUSE_RETAIN_KNOWLEDGE_COMPONENTS)
    | set(MUSE_RETAIN_JUDGE_COMPONENTS)
    | set(MUSE_RETAIN_LINGUISTIC_COMPONENTS)
)
MUSE_JUDGE_METRICS = (
    set(MUSE_FORGET_JUDGE_COMPONENTS)
    | set(MUSE_FORGET_LINGUISTIC_COMPONENTS)
    | set(MUSE_RETAIN_JUDGE_COMPONENTS)
    | set(MUSE_RETAIN_LINGUISTIC_COMPONENTS)
)
MUSE_LOWER_BETTER = {
    "forget_knowmem_ROUGE",
    "forget_verbmem_ROUGE",
    "forget_knowmem_Correctness",
    "forget_verbmem_Correctness",
}

MUSE_GROUPS = [
    ("Forget", [("forget_quality", "EQ"), ("forget_linguistic", "LQ")]),
    ("Retain", [("retain_quality", "MU"), ("retain_linguistic", "LQ")]),
]

MUSE_DETAIL_GROUPS = [
    ("Forget", [
        ("forget_knowmem_ROUGE", "fk_R"),
        ("forget_verbmem_ROUGE", "fv_R"),
        ("forget_knowmem_Relevance", "fk_Rel"),
        ("forget_verbmem_Relevance", "fv_Rel"),
        ("forget_knowmem_Correctness", "fk_Cor"),
        ("forget_verbmem_Correctness", "fv_Cor"),
        ("forget_knowmem_Fluency", "fk_Flu"),
        ("forget_verbmem_Fluency", "fv_Flu"),
    ]),
    ("Retain", [
        ("retain_knowmem_ROUGE", "rk_R"),
        ("retain_knowmem_Relevance", "rk_Rel"),
        ("retain_knowmem_Correctness", "rk_Cor"),
        ("retain_knowmem_Fluency", "rk_Flu"),
    ]),
]

MUSE_METHOD_ORDER = {
    "full": 0, "GA": 1, "GD": 2, "NPO": 3, "SimNPO": 4,
    "WGA": 5, "BST": 6, "SelectiveUnlearning": 7, "SDU": 8, "DiRU": 8,
}


def discover_muse_runs(eval_root):
    """[(model, split, method, run_dir), ...] for saves/eval/muse_*."""
    runs = []
    if not eval_root.is_dir():
        return runs
    for run_dir in sorted(eval_root.glob("muse_*")):
        if not run_dir.is_dir():
            continue
        match = _MUSE_RUN_RE.match(run_dir.name)
        if match:
            runs.append((match.group(1), match.group(2), match.group(3), run_dir))
    return runs


def load_muse_run(run_dir):
    """Raw metrics for one run, merging the judge scores over the summary."""
    summary = _read_json(run_dir / "MUSE_SUMMARY.json") or {}
    judge = _read_json(run_dir / "LLM_JUDGE_COMPLETE.json") or {}
    merged = dict(summary)
    if isinstance(judge.get("metrics"), dict):
        merged.update(judge["metrics"])
    values = {
        key: merged[key]
        for key in MUSE_RAW_KEYS
        if isinstance(merged.get(key), (int, float))
    }
    return values or None


# Which mean combines the MUSE components, mirroring TOFU's --mean. The
# default is "arithmetic"; "harmonic" punishes the weakest component, so a
# single zero drives the whole score to zero. Set by --mean.
MUSE_MEAN = "arithmetic"


def _muse_combine(values):
    return _hmean(values) if MUSE_MEAN == "harmonic" else _amean(values)


def compute_muse_scores(raw):
    data = dict(raw)

    # Forget: score knowmem and verbmem separately, then average the two.
    # Within each split the three signals enter the harmonic mean directly,
    # with equal weight:
    #   EQ_split = harmonic(1 - ROUGE, 1 - Correctness, Relevance)
    # _score_unit already applies the inversion for "lower is better", so
    # `rouge` is 1 - ROUGE and `correctness` is 1 - Correctness/10 here.
    # Harmonic means punish the weakest term: a single zero (e.g. the model
    # still reproduces the text verbatim) drives EQ for that split to zero.
    forget_split_scores = []
    for split in ("knowmem", "verbmem"):
        rouge = _score_unit(data.get(f"forget_{split}_ROUGE"), "lower")
        relevance = _score_unit(data.get(f"forget_{split}_Relevance"), "judge_higher")
        correctness = _score_unit(data.get(f"forget_{split}_Correctness"), "judge_lower")
        score = _hmean([rouge, correctness, relevance])
        if score is not None:
            forget_split_scores.append(score)
    if forget_split_scores:
        data["forget_quality"] = _amean(forget_split_scores)
    forget_linguistic = _component_amean(data, MUSE_FORGET_LINGUISTIC_COMPONENTS)
    if forget_linguistic is not None:
        data["forget_linguistic"] = forget_linguistic

    # Retain: same two-half split, all terms kept as-is (higher is better).
    rouge = _score_unit(data.get("retain_knowmem_ROUGE"), "higher")
    relevance = _score_unit(data.get("retain_knowmem_Relevance"), "judge_higher")
    correctness = _score_unit(data.get("retain_knowmem_Correctness"), "judge_higher")
    retain_quality = _muse_combine([rouge, _amean([relevance, correctness])])
    if retain_quality is not None:
        data["retain_quality"] = retain_quality
    retain_linguistic = _component_amean(data, MUSE_RETAIN_LINGUISTIC_COMPONENTS)
    if retain_linguistic is not None:
        data["retain_linguistic"] = retain_linguistic

    return data


def _muse_method_rank(label):
    return _ranked_by_order(label, MUSE_METHOD_ORDER)


def collect_muse_rows(runs, model, split):
    """Rows for one (model, split): the SFT reference first, then methods."""
    rows = []
    for m, sp, method, run_dir in runs:
        if m != model or sp != split or method != "full":
            continue
        raw = load_muse_run(run_dir)
        if raw is not None:
            rows.append(("sft (full)", compute_muse_scores(raw)))

    methods = []
    for m, sp, method, run_dir in runs:
        if m != model or sp != split or method == "full":
            continue
        raw = load_muse_run(run_dir)
        if raw is None:
            continue
        label = method[len("Diffusion"):] if method.startswith("Diffusion") else method
        methods.append((label, compute_muse_scores(raw)))
    methods.sort(key=lambda r: _muse_method_rank(r[0]))
    rows.extend(methods)
    return rows


def run_muse(args):
    eval_root = args.eval_dir
    runs = discover_muse_runs(eval_root)
    models = [
        model
        for model in sorted({m for m, _, _, _ in runs})
        if _model_matches(model, args.model)
    ]
    splits = sorted({sp for _, sp, _, _ in runs})
    if args.data_split:
        splits = [sp for sp in splits if sp == args.data_split]

    if args.detail:
        groups = MUSE_DETAIL_GROUPS
        judge_metrics = MUSE_JUDGE_METRICS
        lower_better = MUSE_LOWER_BETTER
    else:
        groups = MUSE_GROUPS
        judge_metrics = set()
        lower_better = set()
    cols = [(key, header) for _, group_cols in groups for key, header in group_cols]
    detail_label = " [DETAIL]" if args.detail else ""

    rendered = False
    for split in splits:
        model_rows = {}
        present = []
        for model in models:
            rows = collect_muse_rows(runs, model, split)
            if args.method is not None:
                rows = [
                    (label, data)
                    for label, data in rows
                    if label.startswith("sft")
                    or _normalized_name(label).startswith(_normalized_name(args.method))
                ]
            block = dict(rows)
            if not block:
                continue
            present.append(model)
            model_rows[model] = block

        if not model_rows:
            continue
        rendered = True
        print_table(
            f"MUSE {split}{detail_label}",
            model_rows,
            present,
            "sft (full)",
            groups,
            cols,
            judge_metrics,
            lower_better,
            order_map=MUSE_METHOD_ORDER,
        )

    if not rendered:
        print("\nNo MUSE results found.")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        choices=["tofu", "rwku", "muse"],
        default="tofu",
        help="Which evaluation results to render (default: tofu).",
    )
    ap.add_argument(
        "--mean",
        choices=["harmonic", "arithmetic"],
        default=None,
        help="How to combine score components: 'harmonic' (a "
             "single weak component drags the score down) or 'arithmetic' "
             "(components averaged independently). Defaults to harmonic for "
             "TOFU and arithmetic for RWKU/MUSE.",
    )
    ap.add_argument(
        "--data-split",
        choices=["News", "Books"],
        default=None,
        help="MUSE only. Render just this corpus (default: every split found).",
    )
    ap.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Evaluation root to scan recursively (default: saves/ablation for "
        "tofu, eval/RWKU for rwku).",
    )
    ap.add_argument(
        "--mode",
        choices=["both", "forget_only", "simple"],
        default="both",
        help="TOFU only. 'both' shows every method variant; 'forget_only' shows "
        "the _forgetonly runs plus retain-free-by-design methods (all other "
        "no-suffix runs dropped, GA included); 'simple' shows only the "
        "standard baselines. Reference rows always show.",
    )
    ap.add_argument(
        "--detail",
        nargs="?",
        const="level",
        default=None,
        choices=["level", "criteria"],
        help="Show component metrics used by the final scores in addition to "
        "the final scores themselves. 'level' (default when --detail is passed "
        "with no value) breaks each metric out per level (e.g. f1_R, f2_R, "
        "f3_R). 'criteria' (rwku only) instead averages across levels and "
        "breaks out per criteria (ROUGE, Rel, Cor, Flu).",
    )
    ap.add_argument(
        "--split",
        choices=FORGET_SPLITS,
        default=None,
        help="TOFU only. Only show this forget split (default: show forget01, "
        "forget05, and forget10).",
    )
    ap.add_argument(
        "--method",
        default=None,
        help="Only show methods matching this case-insensitive name prefix, plus "
        "their base (and, for tofu, retain) reference rows.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Only show models matching this case-insensitive name prefix.",
    )
    args = ap.parse_args()
    # Must be set before the summaries are loaded inside run_tofu()/run_muse().
    # Each benchmark keeps its own default when --mean is not given.
    if args.mean is not None:
        globals()["TOFU_MEAN"] = args.mean
        globals()["MUSE_MEAN"] = args.mean
        globals()["RWKU_MEAN"] = args.mean
    if args.eval_dir is None:
        args.eval_dir = {
            "rwku": DEFAULT_RWKU_EVAL_DIR,
            "muse": DEFAULT_MUSE_EVAL_DIR,
        }.get(args.dataset, DEFAULT_EVAL_DIR)

    if args.dataset == "rwku":
        run_rwku(args)
    elif args.dataset == "muse":
        run_muse(args)
    else:
        run_tofu(args)

if __name__ == "__main__":
    main()
