"""Render the SimpleTSD PCSS probe output (``ahat_<split>.jsonl``) as a single
self-contained HTML file: each forget example shows the question and the answer
with the selected forget set ``A_hat`` highlighted (colour intensity ~ PCSS
score, exact score on hover). Light/dark aware, a live search box, and a header
with the run summary.

    PYTHONPATH=src python scripts/dlm/tofu/simple_tsd_probe_to_html.py forget10
    # -> logs/simpletsd_probe/ahat_forget10.html
"""

import html
import json
import os
import sys

OUT_DIR = os.getenv("PROBE_OUT_DIR", "logs/simpletsd_probe")


def _load(split):
    recs = [json.loads(l) for l in open(os.path.join(OUT_DIR, f"ahat_{split}.jsonl"))]
    spath = os.path.join(OUT_DIR, f"summary_{split}.json")
    summary = json.load(open(spath)) if os.path.exists(spath) else {}
    return recs, summary


def _load_gt(split):
    """Raw ground-truth (question, answer) per row from locuslab/TOFU, indexed by
    row order -- the probe iterates the forget set with ``shuffle=False`` so example
    ``i`` is dataset row ``i``. Best-effort: returns ``[]`` if the dataset is absent."""
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    try:
        from datasets import load_dataset

        ds = load_dataset("locuslab/TOFU", split)["train"]
        return [(ds[i]["question"], ds[i]["answer"]) for i in range(len(ds))]
    except Exception as e:  # noqa: BLE001 -- probe view degrades gracefully without GT
        print(f"[html] ground truth unavailable ({e}); rendering without it")
        return []


def _score_range(recs):
    """Max |score| among A_hat tokens, for scaling the highlight opacity."""
    m = 1e-6
    for r in recs:
        for t in r["tokens"]:
            if t["in_a_hat"]:
                m = max(m, abs(float(t["score"])))
    return m


def _render_tokens(tokens, smax):
    out = []
    for t in tokens:
        surf = t["tok"]
        lead = len(surf) - len(surf.lstrip(" "))
        space = "&nbsp;" * lead
        text = html.escape(surf.strip()) or "&nbsp;"
        score = float(t["score"])
        if t["in_a_hat"]:
            op = 0.15 + 0.85 * min(1.0, abs(score) / smax)  # stronger = higher score
            style = f"background:rgba(220,38,38,{op:.3f});"
            out.append(
                f'{space}<span class="ah" style="{style}" '
                f'title="PCSS score = {score:.3f} (selected)">{text}</span>'
            )
        else:
            out.append(f'{space}<span class="tok" title="score = {score:.3f}">{text}</span>')
    return "".join(out)


def _card(r, smax, gt=None):
    q = html.escape(r["question"].replace("user", "").replace("assistant", "").strip())
    nf, na = r["n_forget_tokens"], r["n_answer_tokens"]
    frac = r["selected_frac"]
    chips = " ".join(f'<span class="chip">{html.escape(w)}</span>' for w in r["forget_tokens"])
    gt_html, gt_text = "", ""
    if gt is not None:
        gq, ga = html.escape(gt[0].strip()), html.escape(gt[1].strip())
        gt_text = (gt[0] + " " + gt[1]).lower()
        gt_html = (
            f'<div class="gt"><div class="gt-h">GROUND TRUTH</div>'
            f'<div class="gq"><b>Q</b> {gq}</div>'
            f'<div class="ga"><b>A</b> {ga}</div></div>'
        )
    return f"""
    <div class="card" data-text="{html.escape((r['question'] + ' ' + r['answer'] + ' ' + gt_text).lower())}">
      <div class="meta"><span class="idx">#{r['example']}</span>
        <span class="frac">A&#770;: {nf}/{na} = {frac:.0%}</span></div>
      {gt_html}
      <div class="q"><b>Q</b> {q}</div>
      <div class="a"><b>A</b> <span class="ans">{_render_tokens(r['tokens'], smax)}</span></div>
      <div class="chips">{chips or '<i>(A&#770; rỗng)</i>'}</div>
    </div>"""


def build(split):
    recs, summary = _load(split)
    gt = _load_gt(split)
    smax = _score_range(recs)
    cards = "\n".join(
        _card(r, smax, gt[r["example"]] if r["example"] < len(gt) else None) for r in recs
    )
    sm = summary
    head = f"""
      <div class="stat"><div class="v">{sm.get('n_examples', len(recs))}</div><div class="k">examples</div></div>
      <div class="stat"><div class="v">{sm.get('token_level_selected_frac', 0):.1%}</div><div class="k">tokens in A&#770; (token-level)</div></div>
      <div class="stat"><div class="v">{sm.get('mean_per_example_selected_frac', 0):.1%}</div><div class="k">mean A&#770; / example</div></div>
      <div class="stat"><div class="v">{sm.get('select_method', '?')}</div><div class="k">select_method</div></div>
      <div class="stat"><div class="v">{sm.get('otsu_relax', '?')}</div><div class="k">otsu_relax</div></div>
      <div class="stat"><div class="v">{sm.get('sel_mc', '?')}</div><div class="k">sel_mc</div></div>"""
    doc = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SimpleTSD A&#770; — {split}</title>
<style>
  :root {{ color-scheme: light dark;
    --bg:#fbfbfd; --fg:#1d1d1f; --muted:#6e6e73; --card:#fff; --border:#e5e5ea; --accent:#dc2626; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0d0d0f; --fg:#f5f5f7; --muted:#a1a1a6; --card:#1a1a1d; --border:#2c2c2e; }} }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  header {{ position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--border);
    padding:18px 24px; z-index:5; }}
  h1 {{ margin:0 0 4px; font-size:20px; }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:14px; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:22px; margin-bottom:12px; }}
  .stat .v {{ font-size:22px; font-weight:650; }}
  .stat .k {{ font-size:12px; color:var(--muted); }}
  #q {{ width:100%; max-width:420px; padding:8px 12px; border:1px solid var(--border);
    border-radius:9px; background:var(--card); color:var(--fg); font-size:14px; }}
  main {{ padding:20px 24px; display:grid; gap:14px;
    grid-template-columns:repeat(auto-fill,minmax(440px,1fr)); }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px 16px; }}
  .meta {{ display:flex; justify-content:space-between; font-size:12px; color:var(--muted); margin-bottom:8px; }}
  .idx {{ font-weight:600; }}
  .frac {{ font-variant-numeric:tabular-nums; }}
  .q {{ margin-bottom:8px; }} .q b, .a b {{ color:var(--muted); margin-right:6px; font-size:12px; }}
  .gt {{ background:rgba(16,185,129,.08); border:1px solid rgba(16,185,129,.28);
    border-radius:9px; padding:8px 11px; margin-bottom:10px; }}
  .gt-h {{ font-size:10.5px; letter-spacing:.08em; font-weight:700; color:#059669; margin-bottom:4px; }}
  .gq, .ga {{ font-size:13.5px; }} .gt b {{ color:#059669; margin-right:6px; font-size:12px; }}
  .ans {{ white-space:pre-wrap; word-break:break-word; }}
  .tok {{ border-radius:3px; }}
  .ah {{ border-radius:3px; color:#fff; padding:0 1px; }}
  .chips {{ margin-top:10px; display:flex; flex-wrap:wrap; gap:5px; }}
  .chip {{ font-size:12px; background:rgba(220,38,38,.14); color:var(--accent);
    border:1px solid rgba(220,38,38,.3); border-radius:20px; padding:1px 9px; }}
  .legend {{ font-size:12px; color:var(--muted); margin-top:8px; }}
  .swatch {{ display:inline-block; width:11px; height:11px; border-radius:3px;
    background:var(--accent); vertical-align:middle; margin:0 4px 0 10px; }}
</style></head>
<body>
<header>
  <h1>SimpleTSD — forget set A&#770; (PCSS) · {split}</h1>
  <div class="sub">Khối xanh = <b>ground truth</b> (Q/A gốc TOFU). Token nền đỏ <span class="swatch"></span> = được chọn vào A&#770; (đậm hơn = PCSS score cao hơn). Hover để xem score.</div>
  <div class="stats">{head}</div>
  <input id="q" placeholder="Tìm trong câu hỏi / câu trả lời…" oninput="filt(this.value)">
  <span id="cnt" class="legend"></span>
</header>
<main id="main">
{cards}
</main>
<script>
  const cards=[...document.querySelectorAll('.card')], cnt=document.getElementById('cnt');
  function filt(q){{ q=q.toLowerCase().trim(); let n=0;
    for(const c of cards){{ const hit=!q||c.dataset.text.includes(q); c.style.display=hit?'':'none'; if(hit)n++; }}
    cnt.textContent=n+' / '+cards.length+' hiển thị'; }}
  filt('');
</script>
</body></html>"""
    out = os.path.join(OUT_DIR, f"ahat_{split}.html")
    with open(out, "w") as f:
        f.write(doc)
    print(f"[html] {out}  ({len(recs)} examples)")
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "forget10")
