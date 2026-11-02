"""Mission reports: a self-contained HTML survey report and a compact PDF.

The HTML file has no external requests except optional Google Fonts (with
system fallbacks), so it can be emailed or archived as-is, and it prints to
A4 cleanly ("Print -> Save as PDF"). ``render_pdf`` builds a two-page PDF with
matplotlib (already a core dependency) for people who need a real PDF file.
"""

from __future__ import annotations

import base64
import html
import io
import textwrap
from typing import Callable, Dict, List, Optional

from PIL import Image, ImageDraw

from .brand import CLASS_LABELS, PALETTE, PRODUCT, TAGLINE, annotate, class_color, hex_to_rgb
from .mission import build_inventory, effective_detections

esc = html.escape


def _fmt_t(t: Optional[float]) -> str:
    if t is None:
        return "-"
    m, s = divmod(int(round(t)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _items_text(est: dict) -> str:
    return str(est["low"]) if est["low"] == est["high"] else f"{est['low']}–{est['high']}"


def annotated_thumbnail(mission: dict, frame: dict, img: Image.Image, max_side: int = 560) -> Image.Image:
    """Counted boxes in class colour; boxes awaiting review drawn thin in the signal colour."""
    eff = effective_detections(mission, frame)
    small = img.convert("RGB")
    small.thumbnail((max_side, max_side))
    sx, sy = small.width / float(img.width), small.height / float(img.height)
    scaled = [{**e, "x1": e["x1"] * sx, "y1": e["y1"] * sy, "x2": e["x2"] * sx, "y2": e["y2"] * sy} for e in eff]
    out = annotate(small, [e for e in scaled if e["counted"]], width=3)
    draw = ImageDraw.Draw(out)
    for e in scaled:
        if e["status"] == "pending":
            draw.rectangle([e["x1"], e["y1"], e["x2"], e["y2"]], outline=hex_to_rgb(PALETTE["signal"]), width=2)
    return out


def _data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ----------------------------------------------------------------------------
# SVG charts (inline, no JS needed; <title> gives native hover tooltips)
# ----------------------------------------------------------------------------
def _when(iso: str) -> str:
    """'2026-11-02T20:11:50Z' -> '2026-11-02 20:11 UTC'."""
    return iso.replace("T", " ")[:16] + " UTC" if iso and "T" in iso else iso


def svg_class_bars(classes: List[dict], width: int = 960) -> str:
    rows = [c for c in classes if c["counted"] or c["pending"]]
    if not rows:
        return '<p class="muted">No debris counted.</p>'
    bar_h, gap, label_w, value_w = 18, 10, 130, 90
    peak = max(c["counted"] + c["pending"] for c in rows) or 1
    plot_w = width - label_w - value_w
    h = len(rows) * (bar_h + gap)
    parts = [f'<svg class="chart" viewBox="0 0 {width} {h}" role="img" aria-label="Counted debris by class">']
    for i, c in enumerate(rows):
        y = i * (bar_h + gap)
        w_c = plot_w * c["counted"] / peak
        w_p = plot_w * c["pending"] / peak
        parts.append(f'<text x="0" y="{y + 13}" class="lbl">{esc(c["label"])}</text>')
        if w_c:
            parts.append(f'<rect x="{label_w}" y="{y}" width="{max(w_c, 2):.1f}" height="{bar_h}" rx="4" fill="{c["color"]}">'
                         f'<title>{esc(c["label"])}: {c["counted"]} counted</title></rect>')
        if w_p:
            x = label_w + w_c + (2 if w_c else 0)
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w_p, 2):.1f}" height="{bar_h}" rx="4" fill="url(#hatch)" '
                         f'stroke="{PALETTE["signal"]}" stroke-width="1"><title>{esc(c["label"])}: {c["pending"]} awaiting review</title></rect>')
        val = f'{c["counted"]}' + (f' <tspan class="pend">+{c["pending"]} ?</tspan>' if c["pending"] else "")
        parts.append(f'<text x="{label_w + w_c + w_p + 8:.1f}" y="{y + 13}" class="val">{val}</text>')
    parts.append(f'<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                 f'<rect width="6" height="6" fill="#fff"/><line x1="0" y1="0" x2="0" y2="6" stroke="{PALETTE["signal"]}" stroke-width="2"/></pattern></defs>')
    parts.append("</svg>")
    return "".join(parts)


def svg_timeline(inv: dict, classes_order: List[str], width: int = 960, height: int = 180) -> str:
    tl = inv["timeline"]
    if not tl:
        return '<p class="muted">No frames.</p>'
    left, bottom, top = 34, 26, 8
    plot_w, plot_h = width - left - 6, height - bottom - top - 14
    peak = max([t["total"] for t in tl] + [1])
    n = len(tl)
    slot = plot_w / n
    bw = max(1.0, slot - (2 if slot > 4 else 0))
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Counted debris per frame">']
    for frac in (0.5, 1.0):
        y = top + plot_h * (1 - frac)
        parts.append(f'<line x1="{left}" x2="{width - 6}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                     f'<text x="{left - 6}" y="{y + 4:.1f}" class="axis" text-anchor="end">{round(peak * frac)}</text>')
    parts.append(f'<line x1="{left}" x2="{width - 6}" y1="{top + plot_h}" y2="{top + plot_h}" class="base"/>')
    for i, t in enumerate(tl):
        x = left + i * slot
        y = top + plot_h
        tip = f'Frame {t["index"]}' + (f' @ {_fmt_t(t["t"])}' if t["t"] is not None else f' ({esc(t["source"])})')
        tip += ": " + (", ".join(f'{CLASS_LABELS.get(c, c)} {k}' for c, k in t["counts"].items()) or "no counted debris")
        if t["pending"]:
            tip += f'; {t["pending"]} awaiting review'
        parts.append(f'<g><title>{esc(tip)}</title><rect x="{x:.1f}" y="{top}" width="{slot:.1f}" height="{plot_h + 12}" fill="transparent"/>')
        for c in classes_order:
            k = t["counts"].get(c, 0)
            if not k:
                continue
            hgt = plot_h * k / peak
            y -= hgt
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(hgt - 1, 1):.1f}" fill="{class_color(c)}"/>')
        if t["pending"]:
            parts.append(f'<rect x="{x:.1f}" y="{top + plot_h + 4}" width="{bw:.1f}" height="5" fill="{PALETTE["signal"]}"/>')
        parts.append("</g>")
    # x labels: first, middle, last
    for i in sorted({0, n // 2, n - 1}):
        t = tl[i]
        lab = _fmt_t(t["t"]) if t["t"] is not None else f'#{t["index"]}'
        x = left + i * slot + bw / 2
        parts.append(f'<text x="{x:.1f}" y="{height - 4}" class="axis" text-anchor="middle">{esc(lab)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------
REPORT_CSS = f"""
:root {{ --ink:{PALETTE['ink']}; --deep:{PALETTE['deep']}; --paper:{PALETTE['paper']}; --line:{PALETTE['line']};
  --muted:{PALETTE['muted']}; --signal:{PALETTE['signal']}; --ok:{PALETTE['ok']}; --card:#ffffff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.5 "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; }}
.wrap {{ max-width:1000px; margin:0 auto; padding:0 16px 48px; }}
header.band {{ background:var(--deep); color:#fff; padding:28px 0 22px; margin-bottom:24px; }}
header.band .wrap {{ padding-bottom:0; }}
.kicker {{ font:600 12px/1 "IBM Plex Mono", ui-monospace, monospace; letter-spacing:.12em; text-transform:uppercase; color:#ffd2bf; }}
h1 {{ font-size:30px; line-height:1.15; margin:8px 0 6px; font-weight:700; }}
.meta {{ color:#c9d6dc; font-size:14px; display:flex; flex-wrap:wrap; gap:4px 18px; }}
h2 {{ font-size:13px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin:32px 0 10px; font-weight:600; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }}
.tile {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }}
.tile .v {{ font:700 28px/1.1 "IBM Plex Sans", system-ui, sans-serif; font-variant-numeric:tabular-nums; }}
.tile .k {{ color:var(--muted); font-size:13px; }}
.tile.warn .v {{ color:var(--signal); }}
.panel {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:right; padding:7px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }}
th:first-child, td:first-child {{ text-align:left; }}
th {{ font-size:12px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.06em; }}
.sw {{ display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:8px; vertical-align:-1px; }}
.chart {{ width:100%; height:auto; display:block; }}
.chart .lbl, .chart .val, .chart .axis {{ font:12px "IBM Plex Sans", system-ui, sans-serif; fill:var(--ink); }}
.chart .axis {{ fill:var(--muted); font-size:11px; }}
.chart .pend {{ fill:var(--signal); font-weight:600; }}
.chart .grid {{ stroke:var(--line); stroke-width:1; }}
.chart .base {{ stroke:var(--ink); stroke-width:1; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 16px; font-size:13px; color:var(--ink); margin-top:10px; }}
.frames {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:12px; }}
.frames figure {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; break-inside:avoid; }}
.frames img {{ width:100%; display:block; }}
.frames figcaption {{ padding:8px 10px; font-size:13px; }}
.muted {{ color:var(--muted); }}
.note {{ font-size:13.5px; color:var(--ink); }}
.note li {{ margin:4px 0; }}
.badge {{ display:inline-block; font:600 11px/1 "IBM Plex Mono", monospace; padding:4px 7px; border-radius:4px; background:#fff0e9; color:#8a2c00; }}
.badge.ok {{ background:#e4f3ec; color:#145238; }}
footer {{ margin-top:36px; font-size:12px; color:var(--muted); border-top:1px solid var(--line); padding-top:12px; }}
@media print {{ body {{ background:#fff; }} header.band {{ -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
  .panel, .tile, .frames figure {{ border-color:#bbb; }} h2 {{ break-after:avoid; }} }}
"""


def render_html(mission: dict, load_frame: Optional[Callable[[dict], Optional[Image.Image]]] = None,
                version: str = "") -> str:
    """Full survey report. ``load_frame(frame) -> PIL image`` enables hotspot thumbnails."""
    inv = build_inventory(mission)
    est, tot, rev = inv["estimate"], inv["totals"], inv["review"]
    kind = mission["source"]["kind"]
    order = [c["class_name"] for c in inv["classes"]]
    s = mission["settings"]

    meta = [f'Created {esc(_when(mission["created_at"]))}']
    if mission.get("site"):
        meta.insert(0, f'Site: {esc(mission["site"])}')
    if mission.get("operator"):
        meta.append(f'Operator: {esc(mission["operator"])}')
    meta.append(f'Mission <code>{esc(mission["id"])}</code>')

    review_badge = ('<span class="badge ok">fully reviewed</span>' if rev["reviewable"] and rev["reviewed"] == rev["reviewable"]
                    else f'<span class="badge">{rev["pct"]:.0f}% human-reviewed</span>')

    tiles = [
        ("Estimated items", _items_text(est), ""),
        ("Frames analysed", str(inv["frames"]), ""),
        ("Frames with debris", str(inv["frames_with_debris"]), ""),
        ("Awaiting review", str(tot["pending"]), "warn" if tot["pending"] else ""),
        ("Human-verified", str(tot["verified"] + tot["added"]), ""),
    ]
    if inv["duration_s"] is not None:
        tiles.insert(2, ("Footage sampled", _fmt_t(inv["duration_s"]), ""))
    tiles_html = "".join(f'<div class="tile {cls}"><div class="v">{esc(v)}</div><div class="k">{esc(k)}</div></div>'
                         for k, v, cls in tiles)

    rows = []
    for c in inv["classes"]:
        if not (c["counted"] or c["pending"] or c["rejected"]):
            continue
        mean = "-" if c["mean_conf"] is None else f'{c["mean_conf"]:.2f}'
        rows.append(
            f'<tr><td><span class="sw" style="background:{c["color"]}"></span>{esc(c["label"])} '
            f'<span class="muted">{esc(c["class_name"])}</span></td><td><b>{c["counted"]}</b></td><td>{c["verified"]}</td>'
            f'<td>{c["pending"]}</td><td>{c["rejected"]}</td><td>{c["frames_present"]}</td><td>{c["peak_in_frame"]}</td>'
            f'<td>{mean}</td></tr>')
    table = ('<table><thead><tr><th>Class</th><th>Counted</th><th>Verified</th><th>Awaiting review</th><th>Rejected</th>'
             '<th>Frames seen</th><th>Peak / frame</th><th>Mean conf.</th></tr></thead><tbody>'
             + ("".join(rows) or '<tr><td colspan="8" class="muted">No debris detected above the floor threshold.</td></tr>')
             + "</tbody></table>")

    legend = "".join(f'<span><span class="sw" style="background:{class_color(c)}"></span>{esc(CLASS_LABELS.get(c, c))}</span>'
                     for c in order if any(t["counts"].get(c) for t in inv["timeline"]))
    legend += f'<span><span class="sw" style="background:{PALETTE["signal"]}"></span>Awaiting review (below bars)</span>'

    frames_html = ""
    if load_frame is not None and inv["hotspots"]:
        figs = []
        for h in inv["hotspots"]:
            frame = mission["frames"][h["index"]]
            img = load_frame(frame)
            if img is None:
                continue
            thumb = _data_uri(annotated_thumbnail(mission, frame, img))
            when = f'@ {_fmt_t(frame["t"])}' if frame["t"] is not None else esc(frame["source"])
            counts = ", ".join(f'{CLASS_LABELS.get(c, c)} {n}' for c, n in h["counts"].items())
            figs.append(f'<figure><img src="{thumb}" alt="Frame {h["index"]} with detections"><figcaption>'
                        f'<b>Frame {h["index"]}</b> <span class="muted">{when}</span><br>{esc(counts)}</figcaption></figure>')
        if figs:
            frames_html = '<h2>Debris hotspots</h2><div class="frames">' + "".join(figs) + "</div>"

    source_line = (f'{inv["frames"]} frames sampled every {mission["source"].get("sample_every_s") or "?"} s from '
                   f'{", ".join(esc(f) for f in mission["source"]["files"]) or "video"}' if kind == "video"
                   else f'{inv["frames"]} still image(s)')

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(mission["name"])} · {PRODUCT} survey report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:wght@400;600;700&display=swap" rel="stylesheet">
<style>{REPORT_CSS}</style></head>
<body>
<header class="band"><div class="wrap">
  <div class="kicker">{PRODUCT} · Survey report</div>
  <h1>{esc(mission["name"])}</h1>
  <div class="meta">{"".join(f"<span>{m}</span>" for m in meta)}</div>
</div></header>
<main class="wrap">
  <div class="tiles">{tiles_html}</div>
  <p class="note">{review_badge} &nbsp;{esc(est["note"])}</p>

  <h2>Inventory by class</h2>
  <div class="panel">{table}</div>
  <div class="panel" style="margin-top:10px">{svg_class_bars(inv["classes"])}</div>

  <h2>Timeline</h2>
  <div class="panel">{svg_timeline(inv, order)}<div class="legend">{legend}</div></div>

  {frames_html}

  <h2>How these numbers were made</h2>
  <div class="panel note"><ul>
    <li>Source: {source_line}. Model: <code>{esc(mission.get("model") or "RT-DETR-L (weights/best.pt)")}</code>.</li>
    <li>Detections at or above confidence <b>{s["conf"]:.2f}</b> are counted automatically. Detections between
        <b>{s["floor"]:.2f}</b> and {s["conf"]:.2f} are <b>not counted</b> until a person confirms them in the review queue.
        A reviewer's confirm, relabel, reject or added box always overrides the model.</li>
    <li>{esc(est["note"])}</li>
    <li>The detector scored 92.32% mAP@50 on the SeaClear hold-out split. Sites, cameras and turbidity that differ from
        SeaClear will be harder; known failure modes (red-light loss on metal, glass caustics, burial, bio-fouling, silt
        clouds) are documented in the AquaGuard technical report. Treat unreviewed counts as estimates.</li>
    <li>Only the six classes are detected: {", ".join(esc(v) for v in CLASS_LABELS.values())}. Other litter is not counted.</li>
  </ul></div>
  <footer>{PRODUCT} {esc(version)} · {TAGLINE} · generated from mission data; re-export after review to update.</footer>
</main></body></html>"""


# ----------------------------------------------------------------------------
# PDF (matplotlib, no extra dependencies)
# ----------------------------------------------------------------------------
def _mpl(text) -> str:
    """Escape '$' so matplotlib never parses user text as mathtext (which can raise)."""
    return str(text).replace("$", "\\$")


def render_pdf(mission: dict, load_frame: Optional[Callable[[dict], Optional[Image.Image]]] = None) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    inv = build_inventory(mission)
    est, tot = inv["estimate"], inv["totals"]
    ink, muted, signal = PALETTE["ink"], PALETTE["muted"], PALETTE["signal"]
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.patch.set_facecolor("white")
        fig.add_artist(matplotlib.patches.Rectangle((0, 0.9), 1, 0.1, transform=fig.transFigure, color=PALETTE["deep"]))
        fig.text(0.06, 0.965, f"{PRODUCT.upper()} · SURVEY REPORT", color="#ffd2bf", fontsize=9, family="monospace")
        fig.text(0.06, 0.93, _mpl(mission["name"]), color="white", fontsize=18, weight="bold")
        sub = " · ".join(x for x in [mission.get("site", ""), _when(mission["created_at"]), mission["id"]] if x)
        fig.text(0.06, 0.912, _mpl(sub), color="#c9d6dc", fontsize=8.5)

        tiles = [("Estimated items", _items_text(est)), ("Frames analysed", str(inv["frames"])),
                 ("With debris", str(inv["frames_with_debris"])), ("Awaiting review", str(tot["pending"])),
                 ("Human-verified", str(tot["verified"] + tot["added"]))]
        for i, (k, v) in enumerate(tiles):
            x = 0.06 + i * 0.18
            fig.text(x, 0.855, v, fontsize=17, weight="bold", color=signal if (k == "Awaiting review" and tot["pending"]) else ink)
            fig.text(x, 0.838, k, fontsize=8, color=muted)

        rows = [c for c in inv["classes"] if c["counted"] or c["pending"] or c["rejected"]]
        ax = fig.add_axes([0.25, 0.58, 0.68, 0.22])
        if rows:
            labels = [c["label"] for c in rows][::-1]
            counted = [c["counted"] for c in rows][::-1]
            pending = [c["pending"] for c in rows][::-1]
            colors = [c["color"] for c in rows][::-1]
            ax.barh(labels, counted, color=colors, height=0.6)
            ax.barh(labels, pending, left=counted, color="white", edgecolor=signal, hatch="///", height=0.6)
            peak = max(c + p for c, p in zip(counted, pending)) or 1
            for y, (c, p) in enumerate(zip(counted, pending)):
                ax.text(c + p + peak * 0.01, y, f"{c}" + (f"  +{p} awaiting review" if p else ""), va="center", fontsize=8, color=ink)
            ax.set_xlim(0, peak * 1.45)
        else:
            ax.text(0.5, 0.5, "No debris counted", ha="center", va="center", color=muted)
        ax.set_title("Counted debris by class (hatched = awaiting review)", fontsize=9.5, loc="left", color=ink)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8)

        tl = inv["timeline"]
        ax2 = fig.add_axes([0.08, 0.36, 0.86, 0.16])
        xs = list(range(len(tl)))
        bottom = [0] * len(tl)
        for c in [c["class_name"] for c in rows]:
            vals = [t["counts"].get(c, 0) for t in tl]
            if any(vals):
                ax2.bar(xs, vals, bottom=bottom, color=class_color(c), width=0.85, label=CLASS_LABELS.get(c, c))
                bottom = [b + v for b, v in zip(bottom, vals)]
        pend = [t["pending"] for t in tl]
        if any(pend):
            ax2.scatter([x for x, p in zip(xs, pend) if p], [-0.4] * sum(1 for p in pend if p), marker="|", color=signal, s=60, label="awaiting review")
        ax2.set_title("Counted debris per frame", fontsize=9.5, loc="left", color=ink)
        ax2.set_xlabel("frame" if mission["source"]["kind"] == "stills" else "sampled frame (see HTML report for timestamps)", fontsize=8, color=muted)
        for sp in ("top", "right"):
            ax2.spines[sp].set_visible(False)
        ax2.tick_params(labelsize=7)
        if rows:
            ax2.legend(fontsize=7, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.28))
        ax2.set_ylim(-1, max(bottom + [1]) * 1.1)

        s = mission["settings"]
        notes = [
            f"Counting rule: detections >= {s['conf']:.2f} are counted; {s['floor']:.2f}-{s['conf']:.2f} wait for human review.",
            est["note"],
            "Detector: RT-DETR-L, 92.32% mAP@50 on the SeaClear hold-out split; other sites may be harder.",
            "Only six classes are detected: " + ", ".join(CLASS_LABELS.values()) + ".",
        ]
        y = 0.25
        fig.text(0.06, y, "HOW THESE NUMBERS WERE MADE", fontsize=8, color=muted, weight="bold")
        for n in notes:
            for line in textwrap.wrap(n, 110):
                y -= 0.017
                fig.text(0.06, y, _mpl(line), fontsize=8, color=ink)
            y -= 0.006
        pdf.savefig(fig)
        plt.close(fig)

        if load_frame is not None and inv["hotspots"]:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.06, 0.96, "Debris hotspots", fontsize=14, weight="bold", color=ink)
            for i, h in enumerate(inv["hotspots"][:6]):
                frame = mission["frames"][h["index"]]
                img = load_frame(frame)
                if img is None:
                    continue
                r, c = divmod(i, 2)
                ax = fig.add_axes([0.06 + c * 0.46, 0.64 - r * 0.3, 0.42, 0.27])
                ax.imshow(annotated_thumbnail(mission, frame, img, 700))
                ax.axis("off")
                when = f"@ {_fmt_t(frame['t'])}" if frame["t"] is not None else _mpl(frame["source"])
                ax.set_title(f"Frame {h['index']} {when}: " + ", ".join(f"{CLASS_LABELS.get(k, k)} {v}" for k, v in h["counts"].items()),
                             fontsize=7.5, loc="left", color=ink)
            pdf.savefig(fig)
            plt.close(fig)
    return buf.getvalue()
