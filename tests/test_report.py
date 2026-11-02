"""Report rendering: escaping, charts, thumbnails, PDF, brand drawing."""

import pytest
from PIL import Image

from aq_fakes import det, make_mission
from app.brand import annotate
from app.mission import build_inventory, pick_hotspots
from app.report import render_html, render_pdf, svg_class_bars, svg_timeline
from app.review import add_box


def mission():
    m = make_mission("video", [[det("tire_rubber", 0.9)], [det("tire_rubber", 0.9), det("can_metal", 0.3)], []])
    m["name"] = '<script>alert("x")</script> Pier'
    m["site"] = "Lokrum & reef"
    return m


def test_html_escapes_user_text_and_has_sections():
    html = render_html(mission())
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html and "Lokrum &amp; reef" in html
    for needle in ("Inventory by class", "Timeline", "How these numbers were made", "Awaiting review", "<svg"):
        assert needle in html
    assert "1–2" in html  # video estimate is a range: peak-per-frame .. all counted


def test_html_estimate_and_thumbnails():
    m = mission()
    inv = build_inventory(m)
    assert (inv["estimate"]["low"], inv["estimate"]["high"]) == (1, 2)
    html = render_html(m, load_frame=lambda f: Image.new("RGB", (640, 480), (30, 90, 110)))
    assert html.count("data:image/jpeg;base64,") == 2
    assert "Frame 1" in html and "0:02" in html


def test_added_boxes_render_without_confidence():
    m = mission()
    add_box(m, 2, {"x1": 10, "y1": 10, "x2": 200, "y2": 200}, "net_plastic")
    html = render_html(m, load_frame=lambda f: Image.new("RGB", (640, 480)))
    assert "Net / rope" in html
    out = annotate(Image.new("RGB", (320, 240)), [m["frames"][2]["added"][0]])
    assert out.size == (320, 240)


def test_empty_mission_renders():
    m = make_mission("stills", [])
    html = render_html(m)
    assert "No debris detected" in html and "No frames." in html
    assert "No debris counted" in svg_class_bars([])


def test_svg_charts_have_tooltips():
    inv = build_inventory(mission())
    tl = svg_timeline(inv, ["tire_rubber"])
    assert "<title>Frame 1 @ 0:02: Tyre 1; 1 awaiting review</title>" in tl
    bars = svg_class_bars(inv["classes"])
    assert "Tyre: 2 counted" in bars and "Metal can: 1 awaiting review" in bars


def test_hotspots_spread_across_dive():
    tl = [{"index": i, "total": 3 if i < 10 else 1} for i in range(60)]
    picks = [t["index"] for t in pick_hotspots(tl, 60)]
    assert len(picks) == 6 and picks[0] == 0
    assert all(b - a >= 5 for a, b in zip(picks, picks[1:]))


def test_pdf_is_valid():
    pytest.importorskip("matplotlib")
    pdf = render_pdf(mission(), load_frame=lambda f: Image.new("RGB", (640, 480)))
    assert pdf[:4] == b"%PDF" and len(pdf) > 2000


def test_pdf_survives_dollar_signs_in_user_text():
    pytest.importorskip("matplotlib")
    m = mission()
    m["name"] = r"Pier $\frac$ survey $5"
    m["site"] = "$$"
    assert render_pdf(m)[:4] == b"%PDF"
