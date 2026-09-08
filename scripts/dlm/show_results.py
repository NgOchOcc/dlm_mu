import argparse
import json
import re
import sys
from pathlib import Path

import scipy as sc

EVAL_DIR = Path("saves/eval")
UTILITY_ROUGE = ["retain_Q_A_ROUGE", "ra_Q_A_ROUGE", "wf_Q_A_ROUGE"]
ERASING_PARTS = [
    ("forget_Q_A_Prob", False),
    ("forget_Q_A_ROUGE", False),
    ("forget_truth_ratio", True),
]


ERASING_FLUENCY = ("forget_Q_A_gibberish", True)
LINGUISTIC_PARTS = [
    "forget_Q_A_Fluency",
    "forget_Q_A_Relevance",
    "forget_Q_A_Hallucination",
    "forget_Q_A_Correctness",
]

LINGUISTIC_MAX = 10.0
LINGUISTIC_LQ_PARTS = [
    ("forget_Q_A_Fluency", False),        # used as stored
    ("forget_Q_A_Hallucination", False),  # used as stored
    ("forget_Q_A_Correctness", True),     # complemented: 10 - value
]

def _hmean(vals):
    if any(not isinstance(v, (int, float)) for v in vals):
        return None
    return 0.0 if min(vals) <= 0.0 else float(sc.stats.hmean(vals))


def _amean(vals):
    if any(not isinstance(v, (int, float)) for v in vals):
        return None
    return sum(vals) / len(vals)

GROUPS = [
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
        ("linguistic_quality", "LQ"),
    ]),
]

JUDGE_METRICS = set(LINGUISTIC_PARTS) | {"linguistic_quality"}
COLS = [(key, header) for _, cols in GROUPS for key, header in cols]
METHOD_ORDER = {
    "GradAscent": 4, "GradDiff": 5, "NPO": 6, "SimNPO": 7,
    "WGA": 8, "DDU": 9, "TRACEU": 11,
}

FORGET_ONLY_SUFFIXES = ("_forgetonly", "_forget_only")
INHERENT_FORGET_ONLY = {"RepelGenDistill"}
    
# INHERENT_FORGET_ONLY = {"TSD", "RepelGenDistill"}


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

# Optional trailing "_<N>epochs" task-name tag (written by unlearn.sh with
# EPOCH_IN_NAME=1) recording how long a run trained, so epoch sweeps of one method
# land in distinct dirs. Stripped before a method is categorised / ranked, but
# kept in the printed row label.
_EPOCHS_RE = re.compile(r"_(\d+)epochs$")

def _strip_epochs(label):
    return _EPOCHS_RE.sub("", label)

def _epochs_of(label):
    m = _EPOCHS_RE.search(label)
    return int(m.group(1)) if m else 0

FORGET_SPLITS = ["forget01", "forget05", "forget10"]
RETAIN_FOR = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}
EXCLUDE_METHODS = {"NullAnchor", "AMTU", "TMOPU", "FCP", "CFTMOPU", "OCPE"}
EXCLUDE_SUFFIXES = ("_oldbase",)

def _is_diagnostic(name):
    return name.endswith(EXCLUDE_SUFFIXES) or any(
        f"{sfx}_" in name for sfx in EXCLUDE_SUFFIXES
    )

_COLOR = sys.stdout.isatty()
GREEN = "\033[92m" if _COLOR else ""
WHITE = "\033[97m" if _COLOR else ""
RESET = "\033[0m" if _COLOR else ""
LOWER_BETTER = {"forget_Q_A_Prob", "forget_Q_A_ROUGE"}

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
    data = json.loads(summary.read_text())
    mu = _hmean([data.get(k) for k in UTILITY_ROUGE])
    if mu is not None:
        data["model_utility"] = mu

    eq_parts = [
        data.get(k) if higher else
        (1.0 - data[k] if isinstance(data.get(k), (int, float)) else None)
        for k, higher in ERASING_PARTS
    ]
    eq = _hmean(eq_parts)
    # Second-stage HM: balance the erasing-strength bundle against fluency 50/50 so a
    # method that forgets *fluently* scores above one that forgets via gibberish.
    fk, _ = ERASING_FLUENCY
    if eq is not None and isinstance(data.get(fk), (int, float)):
        eq = _hmean([eq, data[fk]])
    if eq is not None:
        data["erasing_quality"] = eq

    lq_parts = [
        (LINGUISTIC_MAX - data[k] if isinstance(data.get(k), (int, float)) else None)
        if comp else data.get(k)
        for k, comp in LINGUISTIC_LQ_PARTS
    ]
    lq = _amean(lq_parts)
    if lq is not None:
        data["linguistic_quality"] = lq
    return data


def discover_models():
    models = set()
    for fs in FORGET_SPLITS:
        for md in (EVAL_DIR / fs).glob("tofu_*_Diffusion*"):
            m = re.match(r"tofu_(.+)_Diffusion\w+$", md.name)
            if m:
                models.add(m.group(1))
    for rd in EVAL_DIR.glob("tofu_*_retain*"):
        m = re.match(r"tofu_(.+)_retain\d+", rd.name)
        if m:
            models.add(m.group(1))
    return sorted(m for m in models if not _is_diagnostic(m))


def collect_rows(model, forget_split, mode="both"):
    rows = []
    base = load_summary(EVAL_DIR / f"tofu_{model}_retain100_{forget_split}")
    if base is not None:
        rows.append(("base", base))

    retain = RETAIN_FOR[forget_split]
    retain_data = load_summary(EVAL_DIR / f"tofu_{model}_{retain}")
    if retain_data is not None:
        rows.append((retain, retain_data))

    methods = []
    for md in (EVAL_DIR / forget_split).glob(f"tofu_{model}_Diffusion*"):
        method = md.name.split("Diffusion")[-1]
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
            methods.append((method, data))
    methods.sort(key=lambda r: _method_rank(r[0]))
    rows.extend(methods)
    return rows


def load_summary_muse(eval_dir):
    if not eval_dir.is_dir():
        return None
    summary = eval_dir / "MUSE_SUMMARY.json"
    if not summary.exists():
        return None
    return json.loads(summary.read_text())


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
COL_W = 7
SEP = " │ "

def _group_header(groups):
    group = SEP.join(f"{name:^{COL_W * len(cols)}}" for name, cols in groups)
    metric = SEP.join("".join(f"{h:>{COL_W}}" for _, h in cols) for _, cols in groups)
    return group, metric


def _render_block(data, col_best, groups, judge_metrics, force_white=False):
    """Render one model's cells for a row. force_white tags the whole block white
    (used for the base/retain reference rows) instead of the per-column-best green."""
    blocks = []
    for _, cols in groups:
        block = ""
        for key, _ in cols:
            val = data.get(key)
            if not isinstance(val, (int, float)):
                block += f"{'-':>{COL_W}}"
                continue
            text = f"{val:{COL_W}.2f}" if key in judge_metrics else f"{val * 100:{COL_W}.2f}"
            if not force_white and val == col_best.get(key):
                text = f"{GREEN}{text}{RESET}"
            block += text
        blocks.append(block)
    rendered = SEP.join(blocks)
    return f"{WHITE}{rendered}{RESET}" if force_white else rendered


def _method_rank(label):
    core = _strip_epochs(label)
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
    col_best_per_model = {}
    for model in models:
        rows = model_rows.get(model, {})
        col_best = {}
        for key, _ in cols:
            vals = [d[key] for d in rows.values() if isinstance(d.get(key), (int, float))]
            if vals:
                col_best[key] = min(vals) if key in lower_better else max(vals)
        col_best_per_model[model] = col_best

    print(f"\n## {title}")
    print(_rule("╭", "┬", "╮", widths))
    print(_row([""] + [f"{m:^{block_w}}" for m in models], widths))
    print(_row([""] + [group_one] * len(models), widths))
    print(_row(["method"] + [metric_one] * len(models), widths))
    print(_rule("├", "┼", "┤", widths))
    for label in labels:
        is_ref = label == "base" or label == retain_name
        segments = [f"{WHITE}{label}{RESET}" if is_ref else label]
        for model in models:
            data = model_rows.get(model, {}).get(label)
            if data is None:
                segments.append(" " * block_w)
            else:
                segments.append(
                    _render_block(
                        data, col_best_per_model[model], groups, judge_metrics,
                        force_white=is_ref,
                    )
                )
        print(_row(segments, widths))
    print(_rule("╰", "┴", "╯", widths))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=["both", "forget_only", "simple"],
        default="both",
        help="'both' shows every method variant; 'forget_only' shows the "
        "_forgetonly runs plus retain-free-by-design methods like TSD (all other "
        "no-suffix runs dropped, GradAscent included); 'simple' shows only the "
        "headline TSD run and the standard baselines, dropping all TSD/SimpleTSD "
        "sweep variants. Reference rows always show.",
    )
    args = ap.parse_args()
    for forget_split in FORGET_SPLITS:
        models = discover_models()
        model_rows = {}
        present = []
        for model in models:
            block = dict(collect_rows(model, forget_split, mode=args.mode))
            if not block:
                continue
            present.append(model)
            model_rows[model] = block
        if model_rows:
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
    print()

if __name__ == "__main__":
    main()
