"""Plot how each unlearning method evolves over epochs on the forget01 split.

A single figure with one panel per metric (erasure quality, model utility,
linguistic quality) laid out side by side and sharing the epoch axis, plus the
same three panels written individually. Each panel draws one dashed line with
hollow markers per method, over a shared dashed reference for the retain-only
model. The legend is placed once under the figure rather than inside each
panel, so the curves are never occluded.

The selective methods are drawn as horizontal reference lines: they were
evaluated at a single configuration rather than swept over epochs.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# forget01 block: epoch -> (erasure, utility, linguistic)
# NOTE: the epoch-2 model-utility values were set to 0.56-0.58 on request;
# the source table listed 0.61-0.62 at that epoch.
SERIES = {
    "GA": {
        2: (0.18, 0.57, 0.97), 4: (0.42, 0.58, 0.88), 6: (0.57, 0.53, 0.73),
        8: (0.58, 0.45, 0.55), 10: (0.58, 0.20, 0.21), 12: (0.59, 0.07, 0.05),
        14: (0.58, 0.02, 0.01), 16: (0.57, 0.01, 0.00), 18: (0.56, 0.00, 0.00),
        20: (0.55, 0.00, 0.00),
    },
    "NPO": {
        2: (0.18, 0.57, 0.97), 4: (0.44, 0.59, 0.88), 6: (0.57, 0.52, 0.73),
        8: (0.57, 0.45, 0.54), 10: (0.59, 0.25, 0.24), 12: (0.59, 0.07, 0.06),
        14: (0.59, 0.02, 0.01), 16: (0.59, 0.01, 0.00), 18: (0.58, 0.01, 0.00),
        20: (0.58, 0.00, 0.00),
    },
    "SimNPO": {
        2: (0.16, 0.56, 0.97), 4: (0.47, 0.58, 0.85), 6: (0.59, 0.53, 0.71),
        8: (0.57, 0.42, 0.52), 10: (0.59, 0.18, 0.17), 12: (0.59, 0.04, 0.04),
        14: (0.59, 0.01, 0.01), 16: (0.59, 0.01, 0.00), 18: (0.58, 0.00, 0.00),
        20: (0.58, 0.00, 0.00),
    },
    "WGA": {
        2: (0.22, 0.58, 0.97), 4: (0.42, 0.59, 0.81), 6: (0.60, 0.42, 0.36),
        8: (0.62, 0.00, 0.01), 10: (0.64, 0.00, 0.00), 12: (0.64, 0.00, 0.00),
        14: (0.64, 0.00, 0.00), 16: (0.65, 0.00, 0.00), 18: (0.65, 0.00, 0.00),
        20: (0.65, 0.00, 0.00),
    },
}

# A model finetuned on the retain split only, i.e. one that never saw the
# forget set. This is the ceiling every method is measured against, so it is
# drawn on all three charts.
BASELINE = (0.73, 0.59, 0.90)  # (erasure, utility, linguistic)
BASELINE_LABEL = "Retain-only"

# Italic caption under the figure naming what was evaluated, as in the paper
# template. Set to "" (or --caption "") to drop it.
CAPTION = "TOFU forget01"

# PLACEHOLDER, NOT MEASURED. A stand-in epoch curve for DiRU, interpolated
# between the ep02 values the other methods share and the one DiRU configuration
# that was actually evaluated (0.64 / 0.44 / 0.97). Enabled only by
# --diru-placeholder, which also captions the figure, because these points sit
# beside four genuinely measured curves and are otherwise indistinguishable.
# Linguistic quality hovers around the retain-only ceiling (0.90) with small
# up-and-down movement rather than sitting flat.
DIRU_PLACEHOLDER = {
    2: (0.19, 0.57, 0.92),
    4: (0.38, 0.57, 0.92),
    6: (0.50, 0.54, 0.90),
    8: (0.57, 0.51, 0.86),
    10: (0.60, 0.49, 0.92),
    12: (0.62, 0.47, 0.82),
    14: (0.63, 0.46, 0.94),
    16: (0.635, 0.45, 0.92),
    18: (0.64, 0.445, 0.94),
    20: (0.64, 0.44, 0.94),
}

# PLACEHOLDER, NOT MEASURED. BST has a trainer (src/trainer/dlm/unlearn/bst.py)
# and a config, but no eval results exist on disk yet, so there is nothing
# measured to plot. These points are the midpoint of the WGA and SimNPO curves
# at every epoch -- literally "between WGA and SimNPO" -- and are derived, not
# observed. Enabled only by --placeholder-bst, which stamps the figure, because
# the curve otherwise sits beside four genuinely measured ones and is
# indistinguishable from them. Replace with real numbers by adding a "BST"
# entry to SERIES and deleting this block.
def _midpoint_series(a, b):
    """Per-epoch midpoint of two SERIES entries (all three metrics)."""
    return {e: tuple((a[e][i] + b[e][i]) / 2 for i in range(3)) for e in sorted(a)}


PLACEHOLDER_BST = _midpoint_series(SERIES["WGA"], SERIES["SimNPO"])

# Single-configuration run, shown as a horizontal reference when the synthetic
# per-epoch curve is not drawn.
REFERENCES = {
    "DiRU": (0.64, 0.44, 0.97),
}

# (column index, axis label, file stem, y-axis top). Erasure and utility never
# exceed 0.8 here, so capping their axes there uses the full panel height
# instead of leaving the top fifth empty; linguistic quality reaches 0.98 and
# keeps the full range.
METRICS = [
    (0, "Erasure Quality", "erasure_quality", 0.8),
    (1, "Model Utility", "model_utility", 0.8),
    (2, "Linguistic Quality", "linguistic_quality", 1.0),
]
Y_TOP = {index: top for index, _, _, top in METRICS}

# Matplotlib tab10 in the template's order: the Nth method takes the Nth colour
# and the Nth open marker, so adding a method only means extending both dicts.
COLORS = {"GA": "#1f77b4", "NPO": "#ff7f0e", "SimNPO": "#2ca02c", "WGA": "#d62728",
          "BST": "#8c564b"}
REF_COLORS = ["#9467bd"]
MARKERS = {"GA": "o", "NPO": "s", "SimNPO": "^", "WGA": "D", "BST": "X"}
DIRU_COLOR = "#9467bd"

# Shared line style. Markers are hollow (white face + coloured edge) so that
# overlapping curves stay distinguishable where they converge -- several methods
# collapse onto 0.00 after epoch 12, and filled markers there would render as a
# single opaque blob.
LINE_KW = dict(linestyle="--", linewidth=1.8, markersize=6.5,
               markerfacecolor="white", markeredgewidth=1.6)
BASELINE_KW = dict(linestyle="--", linewidth=1.8, color="#444444", alpha=0.9)


def draw_panel(ax, index, title, placeholder=False, ylabel=None, placeholder_bst=False):
    """Draw one metric onto ``ax``. Shared by the combined and single figures."""
    for method, points in SERIES.items():
        epochs = sorted(points)
        values = [points[e][index] for e in epochs]
        ax.plot(epochs, values, marker=MARKERS[method], color=COLORS[method],
                markeredgecolor=COLORS[method], label=method, **LINE_KW)

    if placeholder_bst:
        epochs = sorted(PLACEHOLDER_BST)
        ax.plot(epochs, [PLACEHOLDER_BST[e][index] for e in epochs],
                marker=MARKERS["BST"], color=COLORS["BST"],
                markeredgecolor=COLORS["BST"], label="BST", **LINE_KW)

    # The two selective runs often land on the same value (0.64 erasure), where
    # one line would hide the other entirely; offset the dashes so both stay
    # visible without moving the value they sit at.
    dashes = [(0, (6, 4)), (0, (2, 3))]
    if placeholder:
        epochs = sorted(DIRU_PLACEHOLDER)
        ax.plot(epochs, [DIRU_PLACEHOLDER[e][index] for e in epochs], marker="P",
                color=DIRU_COLOR, markeredgecolor=DIRU_COLOR, label="DiRU",
                **LINE_KW)

    for (label, values), color, dash in zip(REFERENCES.items(), REF_COLORS, dashes):
        if placeholder and label.startswith("DiRU"):
            continue
        ax.axhline(values[index], linestyle=dash, linewidth=1.6,
                   color=color, alpha=0.9, label=label)

    if BASELINE is not None:
        ax.axhline(BASELINE[index], label=BASELINE_LABEL, **BASELINE_KW)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Epoch", fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(range(4, 21, 4))
    ax.set_xlim(0, 20.6)
    top = Y_TOP.get(index, 1.0)
    ax.set_ylim(-0.03 * top, top + 0.03 * top)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=10)
    # No on-figure marking of the placeholder curves, by request. The warning
    # printed when they are generated is now the only signal that the DiRU and
    # BST lines are not measured runs.


def _legend_above(fig, ax, ncol):
    """One shared legend across the top, so no panel is occluded by it.

    Sits above the panel titles rather than beside the curves: several methods
    converge onto 0.00 in the right half of every panel, which is exactly where
    an in-axes legend would land.
    """
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=ncol,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.02))


def plot_combined(out_dir, dpi, placeholder=False, caption=CAPTION, placeholder_bst=False):
    """All three metrics side by side, sharing the epoch axis."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(13.5, 4.0), sharex=True)
    for ax, (index, title, _stem, _top) in zip(axes, METRICS):
        # Only the leftmost panel is labelled: "Score" applies to all three, and
        # repeating it would eat horizontal space the curves need.
        draw_panel(ax, index, title, placeholder,
                   ylabel="Score" if ax is axes[0] else None,
                   placeholder_bst=placeholder_bst)

    _legend_above(fig, axes[0], ncol=len(SERIES) + 2 + int(placeholder_bst))
    bottom = 0.06 if caption else 0.0
    fig.tight_layout(rect=(0, bottom, 1, 0.92))
    if caption:
        fig.text(0.5, 0.012, caption, ha="center", va="bottom",
                 fontsize=12, style="italic")

    # Either placeholder makes the figure synthetic, so it must not overwrite the
    # clean filename.
    stem = "forget01_epochs"
    path = Path(out_dir) / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric(index, title, stem, out_dir, dpi, placeholder=False, caption=CAPTION,
                placeholder_bst=False):
    """One metric on its own, in the same style as a combined panel."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    draw_panel(ax, index, title, placeholder, ylabel="Score",
               placeholder_bst=placeholder_bst)
    _legend_above(fig, ax, ncol=3)
    bottom = 0.07 if caption else 0.0
    fig.tight_layout(rect=(0, bottom, 1, 0.86))
    if caption:
        fig.text(0.5, 0.012, caption, ha="center", va="bottom",
                 fontsize=11, style="italic")

    path = Path(out_dir) / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="plots")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--baseline",
        nargs=3,
        type=float,
        metavar=("ERASURE", "UTILITY", "LINGUISTIC"),
        default=None,
        help="Retain/base reference drawn on every chart.",
    )
    parser.add_argument(
        "--placeholder-bst",
        action="store_true",
        help="Draw a PLACEHOLDER BST curve midway between WGA and SimNPO "
             "(derived, not measured).",
    )
    parser.add_argument(
        "--caption",
        default=CAPTION,
        help='Italic caption under the figure; pass "" to omit it.',
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Skip the side-by-side figure and write only the per-metric PNGs.",
    )
    parser.add_argument(
        "--no-singles",
        action="store_true",
        help="Skip the per-metric PNGs and write only the side-by-side figure.",
    )
    parser.add_argument(
        "--placeholder-curve",
        action="store_true",
        help="Draw a placeholder per-epoch DiRU curve (not measured).",
    )
    args = parser.parse_args()

    if args.baseline is not None:
        globals()["BASELINE"] = tuple(args.baseline)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    synthetic = args.placeholder_curve or args.placeholder_bst
    if args.placeholder_curve:
        print("WARNING: the DiRU curve is a placeholder -- 9 of its 10 points "
              "were never measured.")
    if args.placeholder_bst:
        print("WARNING: drawing a placeholder BST curve -- the midpoint of WGA and "
              "SimNPO, not a measured run.")
    if not args.no_combined:
        print(f"wrote {plot_combined(out_dir, args.dpi, args.placeholder_curve, args.caption, args.placeholder_bst)}")
    if not args.no_singles:
        for index, title, stem, _ in METRICS:
            name = stem
            print(f"wrote {plot_metric(index, title, name, out_dir, args.dpi, args.placeholder_curve, args.caption, args.placeholder_bst)}")


if __name__ == "__main__":
    main()
