"""Tests for the Gradio Space entry point (app_ui.py).

`ultralytics` is replaced by a stub before import, so no real RT-DETR weights
are loaded, and the demo is never launched. Skipped when gradio is not installed
(the CI job installs only the API dependencies).
"""

import importlib
import sys
import types

import pytest
from PIL import Image

from conftest import FakeModel

pytest.importorskip("gradio")


@pytest.fixture
def app_ui(monkeypatch):
    stub = types.ModuleType("ultralytics")
    stub.RTDETR = lambda weights: FakeModel()
    monkeypatch.setitem(sys.modules, "ultralytics", stub)
    monkeypatch.delitem(sys.modules, "app_ui", raising=False)
    module = importlib.import_module("app_ui")
    yield module
    sys.modules.pop("app_ui", None)


@pytest.fixture
def image():
    return Image.new("RGB", (320, 240), color=(20, 60, 80))


def test_import_builds_demo_with_fake_model(app_ui):
    import gradio as gr

    assert isinstance(app_ui.demo, gr.Blocks)
    assert isinstance(app_ui.model, FakeModel)


@pytest.mark.parametrize("question", ["hi", "Hi there!", "hello", "hey AquaGuard"])
def test_greetings_get_hello_reply(app_ui, question):
    assert app_ui.direct_answer(question) == app_ui.GREETING_REPLY


@pytest.mark.parametrize("question", ["I like sushi", "this is nice", "which way is north", "thinking about lunch"])
def test_hi_inside_other_words_is_not_a_greeting(app_ui, question):
    assert app_ui.direct_answer(question) == app_ui.BYPASS_REPLY


def test_capability_question(app_ui):
    assert app_ui.direct_answer("Who are you?") == app_ui.CAPABILITY_REPLY
    assert app_ui.direct_answer("what can you do") == app_ui.CAPABILITY_REPLY


def test_run_ask_sushi_bypasses_without_hello(app_ui, image):
    _, answer, _, table, raw = app_ui.run_ask(image, "I like sushi", 0.25)
    assert answer == app_ui.BYPASS_REPLY
    assert raw["used_detector"] is False
    assert table == []
    assert app_ui.model.calls == []


def test_run_ask_greeting(app_ui, image):
    _, answer, _, _, raw = app_ui.run_ask(image, "hi", 0.25)
    assert answer == app_ui.GREETING_REPLY
    assert raw["used_detector"] is False


def test_run_ask_uses_fixed_reasoning(app_ui, image, make_box):
    # Only a can is detected: "Is there a tire?" must be answered "No".
    app_ui.model.boxes = [make_box([10, 10, 60, 60], 0, 0.9)]
    _, answer, _, table, raw = app_ui.run_ask(image, "Is there a tire?", 0.25)
    assert raw["objects_count"] == 1
    assert table[0][0] == "can_metal"
    assert answer.lower().startswith("no")
    assert "tire_rubber" in answer


def test_run_ask_count_is_class_specific(app_ui, image, make_box):
    app_ui.model.boxes = [
        make_box([10, 10, 60, 60], 0, 0.9),
        make_box([100, 10, 160, 60], 0, 0.8),
        make_box([10, 100, 60, 160], 2, 0.7),
    ]
    _, answer, _, _, _ = app_ui.run_ask(image, "How many plastic bottles?", 0.25)
    assert "0 bottle_plastic" in answer


def test_run_detect_sorted_and_counted(app_ui, image, make_box):
    app_ui.model.boxes = [
        make_box([10, 10, 60, 60], 5, 0.5),
        make_box([100, 10, 160, 60], 1, 0.95),
    ]
    annotated, pills, table, raw = app_ui.run_detect(image, 0.25)
    assert annotated.size == image.size
    assert raw["objects_count"] == 2
    assert [d["class_name"] for d in raw["detections"]] == ["bottle_plastic", "tire_rubber"]
    assert "2 object(s) detected" in pills
    assert app_ui.model.calls[-1]["conf"] == 0.25


def test_run_detect_without_image(app_ui):
    assert app_ui.run_detect(None, 0.25) == (None, "", [], {})


def test_survey_tab_builds_report(app_ui, tmp_path, make_box):
    app_ui.model.boxes = [make_box([10, 10, 80, 90], 5, 0.9), make_box([100, 20, 150, 60], 2, 0.3)]
    paths = []
    for i in range(3):
        p = tmp_path / f"s{i}.jpg"
        Image.new("RGB", (320, 240), (20, 60, 80)).save(p)
        paths.append(str(p))
    (tmp_path / "bad.jpg").write_bytes(b"nope")
    tiles, rows, report = app_ui.run_survey(paths + [str(tmp_path / "bad.jpg")], "Pier <4>", 0.5)
    assert "Estimated items" in tiles and "<b>3</b>" in tiles
    assert rows == [["Tyre", 3, 0, 3, "0.90"], ["Glass bottle", 0, 3, 0, "-"]]
    html = open(report, encoding="utf-8").read()
    assert "Pier &lt;4&gt;" in html and "data:image/jpeg;base64," in html
    assert app_ui.model.calls[-1]["batch"] == 3


def test_survey_tab_without_files(app_ui):
    tiles, rows, report = app_ui.run_survey([], "x", 0.5)
    assert rows == [] and report is None


def test_class_colours_follow_brand(app_ui):
    assert app_ui.CLASS_COLORS["tire_rubber"] == (0, 114, 178)
