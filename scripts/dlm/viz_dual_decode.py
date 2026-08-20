"""Render the dual-decode trace: an interactive token map plus static figures.

Consumes ``trace.json`` from ``probe_dual_decode.py`` and emits

* ``tokens.html`` — every committed token as a colored cell, switchable between
  driver confidence, the ``log p_driver - log p_scorer`` log-odds, cross-model
  Jensen-Shannon divergence, and commit step; hovering a cell shows both models'
  top-5. Under each token sits the scorer's own top-1 when it differs, so the
  substitution the oracle would have made is visible without hovering.
* ``figs/{light,dark}/*.png`` — the aggregate views.
* ``summary.json`` — the numbers worth quoting.

Usage:
    python scripts/dlm/viz_dual_decode.py \
        --trace saves/probe/dual_decode_llada_forget05/trace.json
"""

import argparse
import html
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# --------------------------------------------------------------------------- #
# palette (see the dataviz skill's reference instance; validated for both modes)
# --------------------------------------------------------------------------- #
SEQ_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_ORANGE = [
    "#fbdccd", "#f8c9b1", "#f5b596", "#f2a17c", "#ef8d62", "#ec7a4b", "#eb6834",
    "#d95926", "#c44e21", "#ae451d", "#993c19", "#833315", "#6d2a11",
]
SEQ_RED = [
    "#fbd6d5", "#f7bcba", "#f3a3a0", "#ef8a86", "#eb716d", "#e75a55", "#e34948",
    "#d03b3b", "#b93534", "#a22e2e", "#8b2827", "#742121", "#5e1b1a",
]

THEME = {
    "light": {
        "surface": "#fcfcfb", "plane": "#f9f9f7", "ink": "#0b0b0b",
        "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
        "axis": "#c3c2b7", "mid": "#f0efec",
        "s1": "#2a78d6", "s2": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19", "plane": "#0d0d0d", "ink": "#ffffff",
        "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
        "axis": "#383835", "mid": "#383835",
        "s1": "#3987e5", "s2": "#d95926",
    },
}


def seq_cmap(steps, mode):
    """One-hue sequential ramp; on dark the surface end is the dark step."""
    ordered = steps if mode == "light" else list(reversed(steps))
    return LinearSegmentedColormap.from_list("seq", ordered)


def div_cmap(mode):
    """Diverging blue <-> red with the neutral gray midpoint (never a hue at mid)."""
    blue = list(reversed(SEQ_BLUE[3:]))  # dark blue -> light blue
    red = SEQ_RED[3:]  # light red -> dark red
    return LinearSegmentedColormap.from_list(
        "div", blue + [THEME[mode]["mid"]] + red
    )


# --------------------------------------------------------------------------- #
# trace -> tidy rows
# --------------------------------------------------------------------------- #
STOP_PIECES = {"<|eot_id|>", "<|endoftext|>", "<|im_end|>"}


def answer_len(tokens):
    """Positions strictly before the first turn terminator = the answer proper.

    Masked diffusion fills the whole 128-token block, so most committed tokens
    are post-answer padding the model is trivially confident about. Aggregates
    over the full block would be dominated by that padding.
    """
    stops = [t["pos"] for t in tokens if t["piece"] in STOP_PIECES]
    return min(stops) if stops else max((t["pos"] for t in tokens), default=-1) + 1


def tidy(trace):
    """Flatten to one row per committed token, tagged with run and answer-span."""
    rows = []
    for ex in trace["examples"]:
        for run_name, run in ex["runs"].items():
            n_ans = answer_len(run["tokens"])
            for t in run["tokens"]:
                rows.append(
                    {
                        "example": ex["index"],
                        "run": run_name,
                        "driver": run["driver"],
                        "scorer": run["scorer"],
                        "in_answer": t["pos"] < n_ans,
                        "pos": t["pos"],
                        "step": t["step"],
                        "text": t["text"],
                        "piece": t["piece"],
                        "conf": math.exp(t["driver"]["logp"]),
                        "logp_d": t["driver"]["logp"],
                        "logp_s": t["scorer"]["logp"],
                        "log_odds": t["log_odds"],
                        "js": t["js"],
                        "ent_d": t["driver"]["entropy"],
                        "ent_s": t["scorer"]["entropy"],
                        "rank_s": t["scorer"]["rank"],
                        "agree": t["scorer"]["argmax_agrees"],
                    }
                )
    return rows


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _style(ax, th, xlabel=None, ylabel=None, title=None):
    ax.set_facecolor(th["surface"])
    ax.figure.patch.set_facecolor(th["plane"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(th["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=th["muted"], labelsize=9, length=3, width=1.0)
    ax.grid(True, color=th["grid"], linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=th["ink2"], fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=th["ink2"], fontsize=10)
    if title:
        ax.set_title(title, color=th["ink"], fontsize=12, loc="left", pad=12)


def _legend(ax, th):
    lg = ax.legend(frameon=False, fontsize=9, labelcolor=th["ink2"])
    return lg


def _save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def fig_token_heatmap(trace, rows, outdir, mode, tag_a, n_ex=18):
    """Per-token log-odds over the answer span, one row per example."""
    th = THEME[mode]
    run = f"{tag_a}_driven"
    by_ex = defaultdict(dict)
    for r in rows:
        if r["run"] == run and r["in_answer"]:
            by_ex[r["example"]][r["pos"]] = r["log_odds"]
    exs = sorted(by_ex)[:n_ex]
    if not exs:
        return None
    width = max(max(by_ex[e]) for e in exs) + 1
    grid = np.full((len(exs), width), np.nan)
    for i, e in enumerate(exs):
        for p, v in by_ex[e].items():
            grid[i, p] = v

    vmax = max(1e-6, np.nanpercentile(np.abs(grid), 99))
    fig, ax = plt.subplots(figsize=(min(16, 0.30 * width + 3), 0.34 * len(exs) + 2.4))
    im = ax.imshow(
        grid, aspect="auto", cmap=div_cmap(mode),
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        interpolation="nearest",
    )
    _style(
        ax,
        th,
        xlabel="token position in the answer   (blank = answer ended before here)",
        ylabel="TOFU example",
        title="Where the fact lives: per-token log-odds",
    )
    ax.grid(False)
    ax.set_yticks(range(len(exs)), [str(e) for e in exs])
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cb.set_label(f"log p({tag_a}) - log p(oracle), nats", color=th["ink2"], fontsize=9)
    cb.ax.tick_params(colors=th["muted"], labelsize=8)
    cb.outline.set_visible(False)
    return _save(fig, outdir, "fig_token_heatmap.png")


def fig_logodds_ecdf(rows, outdir, mode, tag_a, tag_b):
    """How model-specific the committed tokens are, per direction.

    The two curves land on top of each other: whichever model drives, the other
    one is equally surprised. Cross-model log-odds therefore measures
    "this trajectory is mine", not "I memorised this fact" -- read it together
    with ``fig_driver_certainty``, which does separate the two checkpoints.
    """
    th = THEME[mode]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for run, color, label in (
        (f"{tag_a}_driven", th["s1"], f"driven by {tag_a} (saw the split)"),
        (f"{tag_b}_driven", th["s2"], f"driven by {tag_b} (oracle)"),
    ):
        v = np.sort([r["log_odds"] for r in rows if r["run"] == run and r["in_answer"]])
        if not len(v):
            continue
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, lw=2, label=label)
    ax.axvline(0, color=th["axis"], lw=1)
    _style(ax, th, xlabel="log-odds of the committed token (nats)",
           ylabel="cumulative fraction of answer tokens",
           title="Symmetric: each model owns its own trajectory")
    _legend(ax, th)
    return _save(fig, outdir, "fig_logodds_ecdf.png")


def fig_driver_certainty(rows, outdir, mode, tag_a, tag_b):
    """Each model's certainty on its *own* decode -- the unconfounded comparison.

    Unlike the cross-model log-odds, this needs no second model, so neither
    direction is privileged: it is just "how sure is this checkpoint about the
    answer it produces".
    """
    th = THEME[mode]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for run, color, label in (
        (f"{tag_a}_driven", th["s1"], f"{tag_a} (saw the split)"),
        (f"{tag_b}_driven", th["s2"], f"{tag_b} (oracle)"),
    ):
        v = np.sort([r["ent_d"] for r in rows if r["run"] == run and r["in_answer"]])
        if not len(v):
            continue
        ax.plot(np.maximum(v, 1e-4), np.arange(1, len(v) + 1) / len(v),
                color=color, lw=2, label=label)
    ax.set_xscale("log")
    _style(ax, th, xlabel="predictive entropy at the committed token (nats)",
           ylabel="cumulative fraction of answer tokens",
           title="The memorising model decodes its answer without hesitation")
    _legend(ax, th)
    return _save(fig, outdir, "fig_driver_certainty.png")


def fig_rank_ecdf(rows, outdir, mode, tag_a, tag_b):
    """Rank the committed token holds under the other model."""
    th = THEME[mode]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for run, color, label in (
        (f"{tag_a}_driven", th["s1"], f"{tag_a} commits, oracle ranks"),
        (f"{tag_b}_driven", th["s2"], f"oracle commits, {tag_a} ranks"),
    ):
        v = np.sort([r["rank_s"] for r in rows if r["run"] == run and r["in_answer"]])
        if not len(v):
            continue
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                color=color, lw=2, label=label)
    ax.set_xscale("log")
    _style(ax, th, xlabel="rank of the committed token under the other model",
           ylabel="cumulative fraction of answer tokens",
           title="Where the other model would not have said this at all")
    _legend(ax, th)
    return _save(fig, outdir, "fig_rank_ecdf.png")


def fig_entropy_scatter(rows, outdir, mode, tag_a):
    """Driver vs scorer predictive entropy, colored by log-odds."""
    th = THEME[mode]
    run = f"{tag_a}_driven"
    sel = [r for r in rows if r["run"] == run and r["in_answer"]]
    if not sel:
        return None
    x = np.array([r["ent_d"] for r in sel])
    y = np.array([r["ent_s"] for r in sel])
    c = np.array([r["log_odds"] for r in sel])
    vmax = max(1e-6, np.percentile(np.abs(c), 99))
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    lim = max(1.0, float(max(x.max(), y.max())) * 1.15)
    ax.plot([0, lim], [0, lim], color=th["axis"], lw=1, zorder=1)
    sc = ax.scatter(
        x, y, c=c, cmap=div_cmap(mode),
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        s=26, linewidths=0.8, edgecolors=th["surface"], zorder=2,
    )
    # Both entropies pile up near zero on the shared scaffolding tokens; symlog
    # keeps that mass readable without hiding the high-entropy oracle tail.
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    _style(ax, th, xlabel=f"predictive entropy, {tag_a} (nats)",
           ylabel="predictive entropy, oracle (nats)",
           title="Certain where the oracle is not")
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("log-odds (nats)", color=th["ink2"], fontsize=9)
    cb.ax.tick_params(colors=th["muted"], labelsize=8)
    cb.outline.set_visible(False)
    return _save(fig, outdir, "fig_entropy_scatter.png")


def fig_decode_order(rows, outdir, mode, tag_a):
    """Which positions the sampler commits first."""
    th = THEME[mode]
    run = f"{tag_a}_driven"
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for flag, color, label in (
        (True, th["s1"], "answer tokens"),
        (False, th["s2"], "post-answer padding"),
    ):
        sel = [r for r in rows if r["run"] == run and r["in_answer"] is flag]
        if not sel:
            continue
        ax.scatter([r["step"] for r in sel], [r["pos"] for r in sel],
                   s=10, color=color, alpha=0.55, linewidths=0, label=label)
    _style(ax, th, xlabel="commit step (denoising order)",
           ylabel="token position in the block",
           title=f"Decode order under {tag_a}: the answer is committed last")
    _legend(ax, th)
    return _save(fig, outdir, "fig_decode_order.png")


def fig_conf_vs_step(trace, outdir, mode, tag_a, tag_b):
    """Mean confidence over still-masked positions, as denoising proceeds."""
    th = THEME[mode]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for run, color, label in (
        (f"{tag_a}_driven", th["s1"], tag_a),
        (f"{tag_b}_driven", th["s2"], tag_b),
    ):
        acc = defaultdict(list)
        for ex in trace["examples"]:
            for s in ex["runs"][run]["steps"]:
                acc[s["step"]].append(s["mean_conf_masked"])
        if not acc:
            continue
        steps = sorted(acc)
        mean = np.array([np.mean(acc[s]) for s in steps])
        lo = np.array([np.percentile(acc[s], 25) for s in steps])
        hi = np.array([np.percentile(acc[s], 75) for s in steps])
        ax.fill_between(steps, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(steps, mean, color=color, lw=2, label=f"{label} (driving)")
        ax.annotate(label, (steps[int(0.5 * len(steps))], mean[int(0.5 * len(steps))]),
                    color=color, fontsize=9, xytext=(6, 6), textcoords="offset points")
    _style(ax, th, xlabel="commit step",
           ylabel="mean max-prob over still-masked positions",
           title="Confidence on the remaining canvas, IQR band")
    _legend(ax, th)
    return _save(fig, outdir, "fig_conf_vs_step.png")


def fig_js_vs_step(trace, outdir, mode, tag_a, tag_b):
    """Cross-model disagreement as a function of denoising progress."""
    th = THEME[mode]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for run, color, label in (
        (f"{tag_a}_driven", th["s1"], f"{tag_a} drives"),
        (f"{tag_b}_driven", th["s2"], f"{tag_b} drives"),
    ):
        acc = defaultdict(list)
        for ex in trace["examples"]:
            for s in ex["runs"][run]["steps"]:
                acc[s["step"]].append(s["mean_js_committed"])
        if not acc:
            continue
        steps = sorted(acc)
        ax.plot(steps, [np.mean(acc[s]) for s in steps], color=color, lw=2, label=label)
    _style(ax, th, xlabel="commit step",
           ylabel="Jensen-Shannon divergence at committed positions (nats)",
           title="The two models only disagree at the very end of the decode")
    _legend(ax, th)
    return _save(fig, outdir, "fig_js_vs_step.png")


# --------------------------------------------------------------------------- #
# interactive token map
# --------------------------------------------------------------------------- #
HTML_HEAD = """<!DOCTYPE html>
<meta charset="utf-8"><title>Dual-decode token map</title>
<style>
:root{color-scheme:light dark}
body{margin:0;font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
     background:var(--plane);color:var(--ink)}
.viz-root{--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
  --muted:#898781;--grid:#e1e0d9;--mid:#f0efec;--border:rgba(11,11,11,.10)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])) .viz-root{
  --surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;
  --muted:#898781;--grid:#2c2c2a;--mid:#383835;--border:rgba(255,255,255,.10)}}
:root[data-theme=dark] .viz-root{--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;
  --ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--mid:#383835;
  --border:rgba(255,255,255,.10)}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:0 0 2px}
.sub{color:var(--ink2);margin:0 0 22px}
.controls{position:sticky;top:0;z-index:5;background:var(--plane);
  padding:12px 0;border-bottom:1px solid var(--border);
  display:flex;gap:18px;align-items:center;flex-wrap:wrap}
.controls label{color:var(--ink2);cursor:pointer;display:flex;gap:5px;align-items:center}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:16px 18px;margin:20px 0}
.q{color:var(--ink);font-weight:600} .gt{color:var(--ink2);margin:4px 0 0}
.panel{margin-top:16px;padding-top:14px;border-top:1px solid var(--grid)}
.phead{color:var(--ink2);font-size:12px;margin-bottom:8px}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;
  border:1px solid var(--border);margin-right:6px;font-variant-numeric:tabular-nums}
.toks{display:flex;flex-wrap:wrap;gap:2px}
.tok{position:relative;border-radius:4px;padding:3px 4px 2px;cursor:default;
  border:1px solid var(--border);min-width:12px;text-align:center}
.tok .t{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;
  white-space:pre}
.tok .s{display:block;font-size:9px;color:var(--muted);
  font-variant-numeric:tabular-nums}
.tok .alt{display:block;font-size:10px;font-family:ui-monospace,monospace;
  white-space:pre;opacity:.75}
.tok.pad{opacity:.42}
#tip{position:fixed;z-index:20;pointer-events:none;display:none;max-width:340px;
  background:var(--surface);color:var(--ink);border:1px solid var(--border);
  border-radius:8px;padding:10px 12px;font-size:12px;
  box-shadow:0 6px 24px rgba(0,0,0,.18)}
#tip b{font-weight:600} #tip table{border-collapse:collapse;margin-top:5px;width:100%}
#tip td{padding:1px 6px 1px 0;color:var(--ink2);font-variant-numeric:tabular-nums}
#tip .hd{color:var(--muted);font-size:11px;padding-top:6px}
.scale{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:11px}
.bar{width:150px;height:10px;border-radius:99px;border:1px solid var(--border)}
details{margin-top:12px} summary{color:var(--ink2);cursor:pointer;font-size:12px}
table.tv{border-collapse:collapse;font-size:11px;margin-top:8px;width:100%}
table.tv th,table.tv td{border-bottom:1px solid var(--grid);padding:3px 6px;
  text-align:right;color:var(--ink2);font-variant-numeric:tabular-nums}
table.tv th{color:var(--muted);font-weight:500;text-align:right}
table.tv td.l,table.tv th.l{text-align:left;font-family:ui-monospace,monospace}
</style>
<body class="viz-root"><div class="wrap">
"""

HTML_JS = """
<script>
const SEQ_BLUE=%(seq_blue)s, SEQ_ORANGE=%(seq_orange)s, SEQ_RED=%(seq_red)s;
const MID=getComputedStyle(document.body).getPropertyValue('--mid').trim()||'#f0efec';
function hex2rgb(h){return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];}
function rampAt(ramp,u){u=Math.max(0,Math.min(1,u));const x=u*(ramp.length-1),
  i=Math.floor(x),f=x-i,a=hex2rgb(ramp[i]),b=hex2rgb(ramp[Math.min(i+1,ramp.length-1)]);
  return `rgb(${a.map((v,k)=>Math.round(v+f*(b[k]-v))).join(',')})`;}
function diverge(v,vmax){const u=Math.max(-1,Math.min(1,v/vmax));
  if(Math.abs(u)<1e-6) return MID;
  // red arm = only the driver is confident, blue arm = the reverse; both arms
  // start at the neutral midpoint so near-zero never jumps to a saturated hue.
  const ramp=[MID].concat(u>0?SEQ_RED.slice(3):SEQ_BLUE.slice(3));
  return rampAt(ramp,Math.abs(u));}
// Ink that stays legible on the cell it sits on.
function inkOn(rgb){const m=rgb.match(/\\d+/g).map(Number);
  const L=(0.2126*m[0]+0.7152*m[1]+0.0722*m[2])/255;return L>0.55?'#0b0b0b':'#ffffff';}
const MODES={
  conf:{label:'driver confidence',color:d=>rampAt(SEQ_BLUE,Math.exp(d.lpd)),
        fmt:d=>'p = '+Math.exp(d.lpd).toFixed(3)},
  confs:{label:'same token under the other model',color:d=>rampAt(SEQ_BLUE,Math.exp(d.lps)),
        fmt:d=>'p (other model) = '+Math.exp(d.lps).toExponential(2)},
  odds:{label:'log-odds vs the other model',color:d=>diverge(d.lo,10),
        fmt:d=>'log-odds = '+d.lo.toFixed(2)+' nats'},
  js:{label:'Jensen-Shannon divergence',color:d=>rampAt(SEQ_ORANGE,d.js/0.6931),
        fmt:d=>'JS = '+d.js.toFixed(3)+' nats'},
  step:{label:'commit step',color:d=>rampAt(SEQ_BLUE,d.st/MAXSTEP),
        fmt:d=>'step '+d.st}
};
let MAXSTEP=1;
function paint(){
  const m=document.querySelector('input[name=mode]:checked').value, M=MODES[m];
  document.querySelectorAll('.tok').forEach(el=>{
    const d=JSON.parse(el.dataset.d);
    const c=M.color(d); el.style.background=c; el.style.color=inkOn(c);
    el.querySelector('.alt') && (el.querySelector('.alt').style.color=inkOn(c));
  });
  document.getElementById('scaleLabel').textContent=M.label;
  document.getElementById('scaleBar').style.background=
    m==='odds' ? `linear-gradient(90deg,${diverge(-10,10)},${MID},${diverge(10,10)})`
    : m==='js' ? `linear-gradient(90deg,${rampAt(SEQ_ORANGE,0)},${rampAt(SEQ_ORANGE,1)})`
    : `linear-gradient(90deg,${rampAt(SEQ_BLUE,0)},${rampAt(SEQ_BLUE,1)})`;
  document.getElementById('scaleLo').textContent = m==='odds'?'-10':'0';
  document.getElementById('scaleHi').textContent =
    m==='conf'||m==='confs'?'1':m==='odds'?'+10':m==='js'?'0.69':String(MAXSTEP);
}
const tip=document.getElementById('tip');
function rowsOf(top){return top.map(a=>
  `<tr><td class="l">${a.t}</td><td>${Math.exp(a.lp).toFixed(3)}</td></tr>`).join('');}
document.addEventListener('mouseover',e=>{
  const el=e.target.closest('.tok'); if(!el){tip.style.display='none';return;}
  const d=JSON.parse(el.dataset.d), m=document.querySelector('input[name=mode]:checked').value;
  tip.innerHTML=`<b>${d.tx.replace(/</g,'&lt;')}</b> &middot; pos ${d.p} &middot; step ${d.st}`
    +`<div>${MODES[m].fmt(d)}</div>`
    +`<table><tr><td class="hd" colspan=2>${d.dn}: p=${Math.exp(d.lpd).toFixed(3)}, H=${d.ed.toFixed(2)} nats</td></tr>`
    +rowsOf(d.td)
    +`<tr><td class="hd" colspan=2>${d.sn}: p=${Math.exp(d.lps).toExponential(2)}, rank ${d.rk}, H=${d.es.toFixed(2)} nats</td></tr>`
    +rowsOf(d.ts)+`</table>`;
  tip.style.display='block';
});
document.addEventListener('mousemove',e=>{
  if(tip.style.display!=='block')return;
  const pad=14,w=tip.offsetWidth,h=tip.offsetHeight;
  tip.style.left=Math.min(e.clientX+pad,innerWidth-w-8)+'px';
  tip.style.top=(e.clientY-h-pad<8?e.clientY+pad:e.clientY-h-pad)+'px';
});
document.querySelectorAll('input[name=mode]').forEach(r=>r.addEventListener('change',paint));
MAXSTEP=Math.max(1,...[...document.querySelectorAll('.tok')].map(
  el=>JSON.parse(el.dataset.d).st));
paint();
</script></div></body>
"""


def _tok_span(t, driver, scorer, in_answer):
    d = {
        "tx": t["text"], "p": t["pos"], "st": t["step"],
        "lpd": round(t["driver"]["logp"], 5), "lps": round(t["scorer"]["logp"], 5),
        "lo": round(t["log_odds"], 4), "js": round(t["js"], 5),
        "ed": round(t["driver"]["entropy"], 4), "es": round(t["scorer"]["entropy"], 4),
        "rk": t["scorer"]["rank"], "dn": driver, "sn": scorer,
        "td": [{"t": a["text"], "lp": round(a["logp"], 4)} for a in t["driver"]["top"]],
        "ts": [{"t": a["text"], "lp": round(a["logp"], 4)} for a in t["scorer"]["top"]],
    }
    shown = t["text"] if t["text"].strip() else t["piece"]
    alt = t["scorer"]["top"][0]["text"]
    # The oracle's own top-1, shown only where it differs -- the substitution
    # made visible without hovering.
    alt_html = (
        f'<span class="alt">{html.escape(alt) if alt.strip() else "&nbsp;"}</span>'
        if not t["scorer"]["argmax_agrees"] else ""
    )
    cls = "tok" + ("" if in_answer else " pad")
    return (
        f'<span class="{cls}" data-d=\'{html.escape(json.dumps(d), quote=True)}\'>'
        f'<span class="t">{html.escape(shown)}</span>'
        f'<span class="s">{t["step"]}</span>{alt_html}</span>'
    )


def _table_view(tokens, n_ans):
    """Accessibility fallback: the same numbers as a table."""
    head = ("<tr><th class='l'>token</th><th>pos</th><th>step</th><th>p driver</th>"
            "<th>p scorer</th><th>log-odds</th><th>JS</th><th>rank</th></tr>")
    body = "".join(
        f"<tr><td class='l'>{html.escape(t['piece'])}</td><td>{t['pos']}</td>"
        f"<td>{t['step']}</td><td>{math.exp(t['driver']['logp']):.3f}</td>"
        f"<td>{math.exp(t['scorer']['logp']):.2e}</td><td>{t['log_odds']:.2f}</td>"
        f"<td>{t['js']:.3f}</td><td>{t['scorer']['rank']}</td></tr>"
        for t in sorted(tokens, key=lambda t: t["pos"]) if t["pos"] < n_ans
    )
    return (f"<details><summary>table view (answer span)</summary>"
            f"<table class='tv'>{head}{body}</table></details>")


def write_html(trace, path, n_ex=25):
    a, b = trace["meta"]["tag_a"], trace["meta"]["tag_b"]
    out = [HTML_HEAD]
    out.append(
        f"<h1>Which tokens does memorisation actually live in?</h1>"
        f"<p class='sub'>LLaDA-8B on TOFU <b>{html.escape(trace['meta']['split'])}</b>. "
        f"One model ({html.escape(a)}) trained on this split, the other "
        f"({html.escape(b)}) never saw it. Each decode is driven by one model while "
        f"the other is scored on the identical canvas at every denoising step, so "
        f"every cell is a matched comparison. Small number = commit step; the faint "
        f"line under a cell is the other model's own top-1 where it differs. "
        f"Faded cells are post-answer padding.</p>"
    )
    out.append(
        "<div class='controls'>"
        "<span style='color:var(--ink2)'>colour by</span>"
        "<label><input type='radio' name='mode' value='odds' checked> log-odds</label>"
        "<label><input type='radio' name='mode' value='conf'> driver confidence</label>"
        "<label><input type='radio' name='mode' value='confs'> other model's p</label>"
        "<label><input type='radio' name='mode' value='js'> JS divergence</label>"
        "<label><input type='radio' name='mode' value='step'> commit step</label>"
        "<span class='scale'><span id='scaleLabel'></span>"
        "<span id='scaleLo'></span><span class='bar' id='scaleBar'></span>"
        "<span id='scaleHi'></span></span></div>"
    )
    for ex in trace["examples"][:n_ex]:
        out.append("<div class='card'>")
        out.append(f"<h2 class='q'>{html.escape(ex['question'])}</h2>")
        out.append(f"<p class='gt'>ground truth &middot; {html.escape(ex['ground_truth'])}</p>")
        for run_name, run in ex["runs"].items():
            n_ans = answer_len(run["tokens"])
            out.append("<div class='panel'>")
            out.append(
                f"<div class='phead'><span class='pill'>driver "
                f"{html.escape(run['driver'])}</span>"
                f"<span class='pill'>scorer {html.escape(run['scorer'])}</span>"
                f"<span class='pill'>ROUGE-L recall {run['rougeL_recall']:.3f}</span>"
                f"{html.escape(run['generation'])}</div>"
            )
            out.append("<div class='toks'>")
            for t in sorted(run["tokens"], key=lambda t: t["pos"]):
                out.append(_tok_span(t, run["driver"], run["scorer"], t["pos"] < n_ans))
            out.append("</div>")
            out.append(_table_view(run["tokens"], n_ans))
            out.append("</div>")
        out.append("</div>")
    out.append('<div id="tip"></div>')
    out.append(HTML_JS % {
        "seq_blue": json.dumps(SEQ_BLUE),
        "seq_orange": json.dumps(SEQ_ORANGE),
        "seq_red": json.dumps(SEQ_RED),
    })
    with open(path, "w") as f:
        f.write("".join(out))
    return path


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def summarize(trace, rows):
    a, b = trace["meta"]["tag_a"], trace["meta"]["tag_b"]
    out = {"split": trace["meta"]["split"], "runs": {}}
    for run in (f"{a}_driven", f"{b}_driven"):
        ans = [r for r in rows if r["run"] == run and r["in_answer"]]
        pad = [r for r in rows if r["run"] == run and not r["in_answer"]]
        if not ans:
            continue
        lo = np.array([r["log_odds"] for r in ans])
        out["runs"][run] = {
            "n_answer_tokens": len(ans),
            "n_padding_tokens": len(pad),
            "mean_rougeL_recall": float(np.mean([
                ex["runs"][run]["rougeL_recall"] for ex in trace["examples"]])),
            "answer_span": {
                "mean_conf_driver": float(np.mean([r["conf"] for r in ans])),
                "mean_logp_scorer": float(np.mean([r["logp_s"] for r in ans])),
                "mean_log_odds": float(lo.mean()),
                "median_log_odds": float(np.median(lo)),
                "frac_log_odds_gt_2nats": float((lo > 2).mean()),
                "frac_log_odds_gt_5nats": float((lo > 5).mean()),
                "mean_js": float(np.mean([r["js"] for r in ans])),
                "frac_scorer_argmax_agrees": float(np.mean([r["agree"] for r in ans])),
                "median_scorer_rank": float(np.median([r["rank_s"] for r in ans])),
                "mean_entropy_driver": float(np.mean([r["ent_d"] for r in ans])),
                "mean_entropy_scorer": float(np.mean([r["ent_s"] for r in ans])),
                "mean_commit_step": float(np.mean([r["step"] for r in ans])),
            },
            "padding_span": {
                "mean_log_odds": float(np.mean([r["log_odds"] for r in pad])) if pad else None,
                "mean_js": float(np.mean([r["js"] for r in pad])) if pad else None,
                "mean_commit_step": float(np.mean([r["step"] for r in pad])) if pad else None,
            },
        }
    # The tokens that carry the gap, ranked -- the "where the fact lives" list.
    ans_a = [r for r in rows if r["run"] == f"{a}_driven" and r["in_answer"]]
    top = sorted(ans_a, key=lambda r: -r["log_odds"])[:40]
    out["top_log_odds_tokens"] = [
        {"example": r["example"], "pos": r["pos"], "piece": r["piece"],
         "log_odds": round(r["log_odds"], 3), "scorer_rank": r["rank_s"],
         "conf_driver": round(r["conf"], 4)}
        for r in top
    ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", default=None, help="defaults to the trace's directory")
    ap.add_argument("--n-html", type=int, default=25)
    args = ap.parse_args()

    with open(args.trace) as f:
        trace = json.load(f)
    outdir = args.out or os.path.dirname(os.path.abspath(args.trace))
    os.makedirs(outdir, exist_ok=True)

    rows = tidy(trace)
    a, b = trace["meta"]["tag_a"], trace["meta"]["tag_b"]
    print(f"[viz] {len(rows)} committed tokens over {len(trace['examples'])} examples")

    for mode in ("light", "dark"):
        d = os.path.join(outdir, "figs", mode)
        os.makedirs(d, exist_ok=True)
        for fn in (
            lambda: fig_token_heatmap(trace, rows, d, mode, a),
            lambda: fig_logodds_ecdf(rows, d, mode, a, b),
            lambda: fig_driver_certainty(rows, d, mode, a, b),
            lambda: fig_rank_ecdf(rows, d, mode, a, b),
            lambda: fig_entropy_scatter(rows, d, mode, a),
            lambda: fig_decode_order(rows, d, mode, a),
            lambda: fig_conf_vs_step(trace, d, mode, a, b),
            lambda: fig_js_vs_step(trace, d, mode, a, b),
        ):
            p = fn()
            if p and mode == "light":
                print(f"[viz] {p}")

    hp = write_html(trace, os.path.join(outdir, "tokens.html"), args.n_html)
    print(f"[viz] {hp}")

    summary = summarize(trace, rows)
    sp = os.path.join(outdir, "summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[viz] {sp}\n")
    for run, s in summary["runs"].items():
        v = s["answer_span"]
        print(f"  {run}")
        print(f"    rougeL_recall            {s['mean_rougeL_recall']:.3f}")
        print(f"    answer tokens            {s['n_answer_tokens']}")
        print(f"    mean conf (driver)       {v['mean_conf_driver']:.4f}")
        print(f"    mean log p (scorer)      {v['mean_logp_scorer']:.3f}")
        print(f"    mean / median log-odds   {v['mean_log_odds']:.3f} / {v['median_log_odds']:.3f}")
        print(f"    frac log-odds > 2 / 5    {v['frac_log_odds_gt_2nats']:.3f} / {v['frac_log_odds_gt_5nats']:.3f}")
        print(f"    mean JS                  {v['mean_js']:.4f}")
        print(f"    scorer argmax agrees     {v['frac_scorer_argmax_agrees']:.3f}")
        print(f"    median scorer rank       {v['median_scorer_rank']:.1f}")
        print(f"    entropy driver / scorer  {v['mean_entropy_driver']:.3f} / {v['mean_entropy_scorer']:.3f}")
        print(f"    mean commit step         {v['mean_commit_step']:.1f}"
              f"   (padding {s['padding_span']['mean_commit_step']:.1f})")


if __name__ == "__main__":
    main()
