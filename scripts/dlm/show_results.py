import argparse
import json
import re
import sys
from pathlib import Path

EVAL_DIR = Path("saves/eval")
USE_SYNTHETIC_RDU_FLUENCY = False
ERASING_KNOWLEDGE_COMPONENTS = {
    "forget_Q_A_Prob": "lower",
    "forget_Q_A_ROUGE": "lower",
    "forget_truth_ratio": "higher",
}
ERASING_JUDGE_COMPONENTS = {
    "forget_Q_A_Relevance": "judge_higher",
    "forget_Q_A_Correctness": "judge_lower",
}
UTILITY_DATASETS = {
    "retain": {
        "retain_Q_A_Prob": "higher",
        "retain_Q_A_ROUGE": "higher",
    },
    "ra": {
        "ra_Q_A_ROUGE": "higher",
    },
    "wf": {
        "wf_Q_A_ROUGE": "higher",
    },
}
LINGUISTIC_COMPONENTS = {
    "forget_Q_A_Fluency": "judge_higher",
    "retain_Q_A_Fluency": "judge_higher",
    "ra_Q_A_Fluency": "judge_higher",
    "wf_Q_A_Fluency": "judge_higher",
}

def _hmean(vals):
    if not vals or any(not isinstance(v, (int, float)) for v in vals):
        return None
    return 0.0 if min(vals) <= 0.0 else float(len(vals) / sum(1.0 / v for v in vals))


def _amean(vals):
    if any(not isinstance(v, (int, float)) for v in vals):
        return None
    return float(sum(vals) / len(vals))


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
<<<<<<< HEAD
    ("Final Scores", [
        ("erase_quality", "EQ"),
        ("model_utility", "MU"),
=======
    ("Erasing Quality", [
        ("forget_Q_A_Prob", "f_P"),
        ("forget_Q_A_ROUGE", "f_R"),
        ("forget_truth_ratio", "f_TR"),
        ("erasing_quality", "EQ"),
        ("forget_Q_A_gibberish", "f_Flu"),
    ]),
    ("Retention Quality", [
        ("retain_Q_A_ROUGE", "ret_R"),
        ("ra_Q_A_ROUGE", "ra_R"),
        ("wf_Q_A_ROUGE", "wf_R"),
        ("model_utility", "RQ"),
    ]),
    ("Linguistic Quality", [
        ("forget_Q_A_Fluency", "L_Flu"),
        ("forget_Q_A_Relevance", "L_Rel"),
        ("forget_Q_A_Hallucination", "L_Hal"),
        ("forget_Q_A_Correctness", "L_Cor"),
>>>>>>> 0857b2bd3d2f6c471e8b374890a2a89321d9b766
        ("linguistic_quality", "LQ"),
        ("final_average", "AVG"),
    ]),
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
    "erase_quality",
    "model_utility",
    "linguistic_quality",
    "final_average",
}
FINAL_SCORE_METRICS = {
    "erase_quality",
    "model_utility",
    "linguistic_quality",
    "final_average",
}
COLS = [(key, header) for _, cols in GROUPS for key, header in cols]
METHOD_ORDER = {
    "GA": 4, "GradAscent": 4, "GradDiff": 5, "NPO": 6, "SimNPO": 7,
    "WGA": 8, "RDU": 1000, "RepelGenDistill": 1000,
}

FORGET_ONLY_SUFFIXES = ("_forgetonly", "_forget_only")
INHERENT_FORGET_ONLY = {"RDU", "RepelGenDistill", "BST"}

DISPLAY_METHOD = {
    "GradAscent": "GA",
    "RepelGenDistill": "RDU",
}
    
# Historical result names are retained in the aliases above.


def _split_forget_only(core):
    """(is_forget_only, base_method): strip a trailing forget-only suffix, if any."""
    for sfx in FORGET_ONLY_SUFFIXES:
        if core.endswith(sfx):
            return True, core[: -len(sfx)]
    return False, core


def _is_inherent_forget_only(base_method):
    """Retain-free-by-design methods, plus their hyper-param sweep arms.

    A sweep tags the arm onto the method name (e.g. TSD_pathwise_minimal,
    SimpleTSD_alpha0.5), so match both the bare method and any ``<method>_...``
    variant. Keeps every TSD / SimpleTSD setting in the forget_only tables.
    """
    return base_method in INHERENT_FORGET_ONLY or any(
        base_method.startswith(f"{m}_") for m in INHERENT_FORGET_ONLY
    )


# `simple` mode: only the headline TSD run and the standard baselines. Every TSD /
# SimpleTSD sweep variant (base_method carrying a "<method>_..." arm tag, e.g.
# TSD_pathwise_minimal, SimpleTSD, SimpleTSD_alpha0.5) is dropped; bare "TSD" stays.
SIMPLE_METHODS = set(METHOD_ORDER) | {"TSD"}


def _is_simple_method(base_method):
    """True for a clean baseline or the bare TSD run; False for any sweep variant."""
    return base_method in SIMPLE_METHODS

<<<<<<< HEAD
# Optional epoch tags used by training/evaluation scripts. They are stripped
# before method categorisation but retained in the displayed row label.
_EPOCHS_RE = re.compile(r"_(?:(\d+)epochs|ep(\d+))$")
=======
# Optional trailing "_<N>epochs" task-name tag (written by unlearn.sh with
# EPOCH_IN_NAME=1) recording how long a run trained, so epoch sweeps of one method
# land in distinct dirs. Stripped before a method is categorised / ranked, but
# kept in the printed row label.
_EPOCHS_RE = re.compile(r"_(\d+)epochs$")
>>>>>>> 0857b2bd3d2f6c471e8b374890a2a89321d9b766

def _strip_epochs(label):
    return _EPOCHS_RE.sub("", label)

def _epochs_of(label):
    m = _EPOCHS_RE.search(label)
    return int(m.group(1) or m.group(2)) if m else 0

def _epoch_label(label):
    epoch = _epochs_of(label)
    return f"_ep{epoch:02d}" if epoch else ""


def _method_matches(method, selected_method):
    if selected_method is None:
        return True
    _, method_name = _split_forget_only(_strip_epochs(method.rsplit("/", 1)[-1]))
    _, selected_name = _split_forget_only(
        _strip_epochs(selected_method.rsplit("/", 1)[-1])
    )
    method_name = DISPLAY_METHOD.get(method_name, method_name)
    selected_name = DISPLAY_METHOD.get(selected_name, selected_name)
    return method_name.casefold().startswith(selected_name.casefold())

FORGET_SPLITS = ["forget01", "forget05", "forget10"]
RETAIN_FOR = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}
EXCLUDE_METHODS = {"NullAnchor", "AMTU", "TMOPU", "FCP", "CFTMOPU", "OCPE"}
EXCLUDE_SUFFIXES = ("_oldbase",)

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

# ---------------------------------------------------------------------------
# MUSE benchmark (News / Books splits) — rendered as separate tables from TOFU.
# Different metrics, splits, and no linguistic-judge dimensions. All four metrics
# are ROUGE-like fractions shown x100. The three forget metrics are lower-better;
# retain knowledge-memorisation is higher-better. The reference row is the SFT
# target (saves/eval/muse_<model>_<split>_full) if it has been evaluated; every
# method dir here is forget-only by construction, so the "_forget_only" tag is
# stripped from the printed label.
# ---------------------------------------------------------------------------
MUSE_SPLITS = ["News", "Books"]
GROUPS_MUSE = [
    ("Forget", [
        ("forget_verbmem_ROUGE", "f_VM"),
        ("forget_knowmem_ROUGE", "f_KM"),
        ("extraction_strength", "f_ES"),
    ]),
    ("Retain", [
        ("retain_knowmem_ROUGE", "ret_KM"),
    ]),
]
COLS_MUSE = [(key, header) for _, cols in GROUPS_MUSE for key, header in cols]
JUDGE_METRICS_MUSE = set()
LOWER_BETTER_MUSE = {
    "forget_verbmem_ROUGE", "forget_knowmem_ROUGE", "extraction_strength",
}
MUSE_FORGET_ONLY_SUFFIX = "_forget_only"

def load_summary(eval_dir):
    if not eval_dir.is_dir():
        return None
    noprompt = eval_dir / "TOFU_SUMMARY_noprompt.json"
    summary = noprompt if noprompt.exists() else eval_dir / "TOFU_SUMMARY.json"
    if not summary.exists():
        return None
    try:
        data = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if USE_SYNTHETIC_RDU_FLUENCY and eval_dir.name.endswith(
        ("DiffusionRDU", "DiffusionRepelGenDistill")
    ):
        synthetic_path = eval_dir / "forget_Q_A_Fluency_synthetic.json"
        if synthetic_path.exists():
            try:
                synthetic = json.loads(synthetic_path.read_text())
            except (OSError, json.JSONDecodeError):
                synthetic = {}
            synthetic_value = synthetic.get("agg_value")
            if isinstance(synthetic_value, (int, float)):
                data["forget_Q_A_Fluency"] = synthetic_value
    erasing_knowledge = _component_hmean(data, ERASING_KNOWLEDGE_COMPONENTS)
    erasing_judge = _component_amean(data, ERASING_JUDGE_COMPONENTS)
    erase_quality = _hmean([erasing_knowledge, erasing_judge])
    if erase_quality is not None:
        data["erase_quality"] = erase_quality

    dataset_utilities = [
        _component_hmean(data, components, skip_missing=True)
        for components in UTILITY_DATASETS.values()
    ]
    dataset_utilities = [value for value in dataset_utilities if value is not None]
    model_utility = _hmean(dataset_utilities)
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


def _forget_split_for(eval_dir):
    relative = eval_dir.relative_to(EVAL_DIR)
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


def discover_method_runs():
    runs = []
    for summary in EVAL_DIR.rglob("TOFU_SUMMARY.json"):
        eval_dir = summary.parent
        match = re.match(r"tofu_(.+?)_Diffusion(.+)$", eval_dir.name)
        if not match:
            continue
        forget_split = _forget_split_for(eval_dir)
        if forget_split is not None:
            runs.append((forget_split, match.group(1), match.group(2), eval_dir))
    return sorted(runs, key=lambda item: str(item[3]))


def _run_source_label(eval_dir, forget_split):
    parent = eval_dir.parent.relative_to(EVAL_DIR)
    return "" if parent in {Path("."), Path(forget_split)} else f"{parent.as_posix()}/"


def discover_models():
    models = {model for _, model, _, _ in discover_method_runs()}
    for rd in EVAL_DIR.glob("tofu_*_retain*"):
        m = re.match(r"tofu_(.+)_retain\d+", rd.name)
        if m:
            models.add(m.group(1))
    return sorted(m for m in models if not _is_diagnostic(m))


def collect_rows(model, forget_split, mode="both", selected_method=None):
    rows = []
    base = load_summary(EVAL_DIR / f"tofu_{model}_retain100_{forget_split}")
    if base is not None:
        rows.append(("base", base))

    retain = RETAIN_FOR[forget_split]
    retain_data = load_summary(EVAL_DIR / f"tofu_{model}_{retain}")
    if retain_data is not None:
        rows.append((retain, retain_data))

    methods = []
    for split, run_model, method, md in discover_method_runs():
        if split != forget_split or run_model != model:
            continue
        if not _method_matches(method, selected_method):
            continue
        core = _strip_epochs(method)
        is_forget_only, base_method = _split_forget_only(core)
        if base_method in EXCLUDE_METHODS or _is_diagnostic(method):
            continue
        if mode == "forget_only" and not (is_forget_only or _is_inherent_forget_only(base_method)):
            continue
        # `simple`: only clean baselines + bare TSD, and for the baselines only their
        # forget-only arm (TSD is inherently forget-only, so it always qualifies).
        if mode == "simple" and not (
            _is_simple_method(base_method)
            and (is_forget_only or _is_inherent_forget_only(base_method))
        ):
            continue
        data = load_summary(md)
        if data is not None:
            source = _run_source_label(md, forget_split)
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


def load_summary_muse(eval_dir):
    if not eval_dir.is_dir():
        return None
    summary = eval_dir / "MUSE_SUMMARY.json"
    if not summary.exists():
        return None
    try:
        return json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def discover_models_muse():
    models = set()
    for split in MUSE_SPLITS:
        for md in EVAL_DIR.glob(f"muse_*_{split}_Diffusion*"):
            m = re.match(rf"muse_(.+)_{split}_Diffusion", md.name)
            if m:
                models.add(m.group(1))
    return sorted(models)


def collect_rows_muse(model, split, mode="both"):
    rows = []
    base = load_summary_muse(EVAL_DIR / f"muse_{model}_{split}_full")
    if base is not None:
        rows.append(("base", base))

    methods = []
    for md in EVAL_DIR.glob(f"muse_{model}_{split}_Diffusion*"):
        method = md.name.split("Diffusion")[-1]
        if method.endswith(MUSE_FORGET_ONLY_SUFFIX):
            method = method[: -len(MUSE_FORGET_ONLY_SUFFIX)]
<<<<<<< HEAD
        method = DISPLAY_METHOD.get(method, method)
=======
>>>>>>> 0857b2bd3d2f6c471e8b374890a2a89321d9b766
        _, base_method = _split_forget_only(_strip_epochs(method))
        if mode == "simple" and not _is_simple_method(base_method):
            continue
        data = load_summary_muse(md)
        if data is not None:
            methods.append((method, data))
    methods.sort(key=lambda r: _method_rank(r[0]))
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
            val = data.get(key)
            if not isinstance(val, (int, float)):
                block += f"{'-':>{COL_W}}"
                continue
            if key in FINAL_SCORE_METRICS:
                display_value = val
            elif key in judge_metrics:
                display_value = val / 10.0
            else:
                display_value = val
            text = f"{display_value:{COL_W}.2f}"
            rank = col_ranks.get(key, {}).get(val)
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
    core = _strip_epochs(label.rsplit("/", 1)[-1])
    is_forget_only, base_method = _split_forget_only(core)
    return (METHOD_ORDER.get(base_method, 99), base_method, is_forget_only, _epochs_of(label))


def _label_order(label, retain_name):
    if label == "base":
        return (0, "", False, 0)
    if label == retain_name:
        return (1, "", False, 0)
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


def print_table(title, model_rows, models, retain_name, groups, cols, judge_metrics, lower_better):
    group_one, metric_one = _group_header(groups)
    block_w = len(metric_one)
    all_labels = set().union(*(rows.keys() for rows in model_rows.values()))
    labels = sorted(all_labels, key=lambda lbl: _label_order(lbl, retain_name))
    label_w = max([LABEL_W] + [len(lbl) for lbl in labels])
    widths = [label_w] + [block_w] * len(models)
    col_ranks_per_model = {}
    for model in models:
        rows = model_rows.get(model, {})
        col_ranks = {}
        for key, _ in cols:
            vals = [
                data[key]
                for label, data in rows.items()
                if label not in {"base", retain_name}
                and isinstance(data.get(key), (int, float))
            ]
            if vals:
                ranked = sorted(set(vals), reverse=key not in lower_better)
                col_ranks[key] = {
                    value: rank for rank, value in enumerate(ranked[:2], start=1)
                }
        col_ranks_per_model[model] = col_ranks

    print(f"\n## {title}")
    print(_rule("╭", "┬", "╮", widths))
    print(_row([""] + [f"{m:^{block_w}}" for m in models], widths))
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


def main():
    global EVAL_DIR, USE_SYNTHETIC_RDU_FLUENCY
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-dir",
        type=Path,
        default=EVAL_DIR,
        help="Evaluation root to scan recursively (default: saves/eval).",
    )
    ap.add_argument(
        "--mode",
        choices=["both", "forget_only", "simple"],
        default="both",
        help="'both' shows every method variant; 'forget_only' shows the "
        "_forgetonly runs plus retain-free-by-design methods like TSD (all other "
<<<<<<< HEAD
        "no-suffix runs dropped, GA included); 'simple' shows only the "
        "headline TSD run and the standard baselines, dropping all TSD/SimpleTSD "
        "sweep variants. Reference rows always show.",
    )
    ap.add_argument(
        "--synthetic-rdu-fluency",
        action="store_true",
        help="Use clearly labeled synthetic RDU forget fluency files instead of "
        "the original judge aggregate.",
    )
    ap.add_argument(
        "--detail",
        action="store_true",
        help="Show all component metrics used by EQ, MU, and LQ in addition "
        "to the four final scores.",
    )
    ap.add_argument(
        "--split",
        choices=FORGET_SPLITS,
        default=None,
        help="Only show this forget split (default: show forget01, forget05, "
        "and forget10).",
    )
    ap.add_argument(
        "--method",
        default=None,
        help="Only show methods matching this case-insensitive name prefix, plus "
        "their base and retain reference rows.",
=======
        "no-suffix runs dropped, GradAscent included); 'simple' shows only the "
        "headline TSD run and the standard baselines, dropping all TSD/SimpleTSD "
        "sweep variants. Reference rows always show.",
>>>>>>> 0857b2bd3d2f6c471e8b374890a2a89321d9b766
    )
    args = ap.parse_args()
    EVAL_DIR = args.eval_dir
    USE_SYNTHETIC_RDU_FLUENCY = args.synthetic_rdu_fluency
    synthetic_label = " [SYNTHETIC RDU FLUENCY]" if USE_SYNTHETIC_RDU_FLUENCY else ""
    groups = DETAIL_GROUPS if args.detail else GROUPS
    cols = [(key, header) for _, group_cols in groups for key, header in group_cols]
    detail_label = " [DETAIL]" if args.detail else ""
    splits = [args.split] if args.split else FORGET_SPLITS
    for forget_split in splits:
        model_rows = {}
        present = []
        for model in discover_models():
            block = dict(
                collect_rows(
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
<<<<<<< HEAD
            if args.detail:
                for model in present:
                    print_table(
                        f"{forget_split} — {model}{synthetic_label}{detail_label}",
                        {model: model_rows[model]}, [model], RETAIN_FOR[forget_split],
                        groups, cols, JUDGE_METRICS, LOWER_BETTER,
                    )
            else:
                print_table(
                    f"{forget_split}{synthetic_label}", model_rows, present,
                    RETAIN_FOR[forget_split],
                    groups, cols, JUDGE_METRICS, LOWER_BETTER,
                )
=======
            print_table(
                forget_split, model_rows, present, RETAIN_FOR[forget_split],
                GROUPS, COLS, JUDGE_METRICS, LOWER_BETTER,
            )

    muse_models = discover_models_muse()
    for split in MUSE_SPLITS:
        model_rows = {}
        present = []
        for model in muse_models:
            block = dict(collect_rows_muse(model, split, mode=args.mode))
            if not block:
                continue
            present.append(model)
            model_rows[model] = block
        if model_rows:
            print_table(
                f"MUSE {split}  (Forget f_VM/f_KM/f_ES ↓ better, Retain ret_KM ↑ better)",
                model_rows, present, None,
                GROUPS_MUSE, COLS_MUSE, JUDGE_METRICS_MUSE, LOWER_BETTER_MUSE,
            )
>>>>>>> 0857b2bd3d2f6c471e8b374890a2a89321d9b766
    print()

if __name__ == "__main__":
    main()
