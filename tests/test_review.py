"""Review decisions and the retraining export."""

import io
import json
import zipfile

import pytest

from app.mission import build_inventory
from app.review import (ReviewError, add_box, apply_review, build_corrections_zip, confirm_frame,
                        correction_records, frame_fully_reviewed, remove_added, yolo_labels)
from aq_fakes import det, make_mission

NOW = lambda: "2026-11-03T01:00:00Z"  # noqa: E731


def mission():
    return make_mission(frames=[[det("tire_rubber", 0.9), det("can_metal", 0.3)], [det("bag_plastic", 0.7)], []])


def test_apply_review_actions():
    m = mission()
    assert apply_review(m, "00000-00", "confirm", reviewer=" deck-1 ", now=NOW) == {"action": "confirm", "reviewer": "deck-1", "at": NOW()}
    e = apply_review(m, "00000-01", "relabel", "bottle_glass", note="label was glare", now=NOW)
    assert e["class_name"] == "bottle_glass" and e["note"] == "label was glare"
    # relabel to the class the model already gave is just a confirmation
    assert apply_review(m, "00001-00", "relabel", "bag_plastic", now=NOW)["action"] == "confirm"
    assert apply_review(m, "00001-00", "undo") is None
    assert "00001-00" not in m["reviews"]


@pytest.mark.parametrize("args", [("nope", "confirm"), ("00000-00", "delete"), ("00000-00", "relabel", "boat")])
def test_apply_review_rejects_bad_input(args):
    with pytest.raises(ReviewError):
        apply_review(mission(), *args)


def test_confirm_frame_and_undo_clears_flag():
    m = mission()
    apply_review(m, "00000-00", "reject", now=NOW)
    assert confirm_frame(m, 0, "r", now=NOW) == 1  # only the undecided box
    assert m["reviews"]["00000-00"]["action"] == "reject"
    assert frame_fully_reviewed(m, m["frames"][0])
    confirm_frame(m, 2)  # empty frame marked clear -> negative example
    assert frame_fully_reviewed(m, m["frames"][2])
    apply_review(m, "00000-01", "undo")
    assert not m["frames"][0].get("cleared")
    assert not frame_fully_reviewed(m, m["frames"][0])
    with pytest.raises(ReviewError):
        confirm_frame(m, 9)


def test_add_box_clamps_and_counts():
    m = mission()
    a = add_box(m, 2, {"x1": 700, "y1": -5, "x2": 600, "y2": 100}, "net_plastic", "r", now=NOW)
    assert (a["x1"], a["y1"], a["x2"], a["y2"]) == (600.0, 0.0, 640.0, 100.0)
    assert a["id"] == "00002-a00" and a["confidence"] is None
    inv = build_inventory(m)
    assert inv["totals"]["added"] == 1
    assert next(c for c in inv["classes"] if c["class_name"] == "net_plastic")["counted"] == 1
    remove_added(m, "00002-a00")
    assert build_inventory(m)["totals"]["added"] == 0
    with pytest.raises(ReviewError):
        remove_added(m, "00002-a00")


@pytest.mark.parametrize("box,cls", [({"x1": 1, "y1": 1, "x2": 2, "y2": 2}, "net_plastic"),
                                     ({"x1": "a"}, "net_plastic"), ({"x1": 0, "y1": 0, "x2": 50, "y2": 50}, "kelp")])
def test_add_box_rejects_bad_boxes(box, cls):
    with pytest.raises(ReviewError):
        add_box(mission(), 0, box, cls)


def test_yolo_labels_only_for_fully_reviewed_frames():
    m = mission()
    apply_review(m, "00000-00", "confirm", now=NOW)
    assert yolo_labels(m) == {}  # frame 0 still has an undecided box
    apply_review(m, "00000-01", "relabel", "bottle_glass", now=NOW)
    labels = yolo_labels(m)
    assert list(labels) == ["00000"]
    lines = labels["00000"].strip().splitlines()
    assert [ln.split()[0] for ln in lines] == ["5", "2"]  # tire_rubber, bottle_glass
    cx, cy, w, h = map(float, lines[0].split()[1:])
    assert (round(cx, 4), round(cy, 4), round(w, 4), round(h, 4)) == (0.0469, 0.0729, 0.0625, 0.1042)


def test_rejected_boxes_are_left_out_of_labels_but_kept_in_corrections():
    m = mission()
    apply_review(m, "00001-00", "reject", reviewer="r", now=NOW)
    assert yolo_labels(m) == {"00001": ""}
    rec = correction_records(m)
    assert rec == [{
        "mission_id": m["id"], "frame_index": 1, "frame_source": "f1.jpg", "t": None, "image": "frames/00001.jpg",
        "image_size": [640, 480], "box_id": "00001-00", "bbox_xyxy": [10, 10, 50, 60], "decision": "rejected",
        "model_class": "bag_plastic", "model_confidence": 0.7, "final_class": None, "reviewer": "r", "at": NOW(),
    }]


def test_corrections_zip_layout():
    m = mission()
    confirm_frame(m, 1, now=NOW)
    add_box(m, 2, {"x1": 0, "y1": 0, "x2": 64, "y2": 64}, "tire_rubber", now=NOW)
    data = build_corrections_zip(m, read_frame=lambda rel: b"JPEG:" + rel.encode())
    z = zipfile.ZipFile(io.BytesIO(data))
    names = set(z.namelist())
    assert {"corrections.jsonl", "data.yaml", "README.txt", "labels/00001.txt", "labels/00002.txt",
            "images/00001.jpg", "images/00002.jpg"} <= names
    assert "labels/00000.txt" not in names
    lines = [json.loads(x) for x in z.read("corrections.jsonl").decode().splitlines()]
    assert {r["decision"] for r in lines} == {"confirmed", "added"}
    assert "5: tire_rubber" in z.read("data.yaml").decode()


def test_added_box_ids_never_repeat_after_removal():
    m = mission()
    a0 = add_box(m, 2, {"x1": 0, "y1": 0, "x2": 50, "y2": 50}, "net_plastic")
    a1 = add_box(m, 2, {"x1": 60, "y1": 0, "x2": 120, "y2": 50}, "net_plastic")
    remove_added(m, a0["id"])
    a2 = add_box(m, 2, {"x1": 0, "y1": 60, "x2": 50, "y2": 120}, "tire_rubber")
    ids = [a["id"] for a in m["frames"][2]["added"]]
    assert ids == [a1["id"], a2["id"]] and len(set(ids)) == 2
