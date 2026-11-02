"""Mission inventory: counting rule, review overrides, estimates, batching and video sampling."""

import pytest
from PIL import Image

from app.mission import (BATCH_SIZE, build_inventory, detections_from_result, downscale, effective_detections,
                         frame_record, new_mission, review_queue, run_frames, sample_video, summary)
from aq_fakes import FakeBox, FakeModel, FakeResult, det, make_mission


def test_new_mission_validates_inputs():
    with pytest.raises(ValueError):
        new_mission("x", "audio")
    with pytest.raises(ValueError):
        new_mission("x", "stills", conf=0.3, floor=0.4)
    m = new_mission("  ", "video")
    assert m["name"] == "Untitled survey"
    assert m["id"].startswith("m-") and m["status"] == "processing"


def test_frame_record_assigns_stable_ids():
    f = frame_record(7, "a.jpg", 3.14159, (100, 50), [det("can_metal", 0.912345), det("tire_rubber", 0.4)])
    assert [d["id"] for d in f["detections"]] == ["00007-00", "00007-01"]
    assert f["detections"][0]["confidence"] == 0.9123
    assert f["t"] == 3.14 and f["width"] == 100


def test_detections_from_result_handles_unknown_and_empty():
    r = FakeResult([FakeBox([0, 0, 5, 5], 99, 0.5), FakeBox([1, 1, 9, 9], 5, 0.9)])
    out = detections_from_result(r)
    assert [d["class_name"] for d in out] == ["tire_rubber", "99"]
    assert detections_from_result(FakeResult(None)) == []


def test_counting_rule_auto_vs_pending():
    m = make_mission(frames=[[det("tire_rubber", 0.9), det("can_metal", 0.3)]])
    eff = effective_detections(m, m["frames"][0])
    assert [(e["status"], e["counted"]) for e in eff] == [("auto", True), ("pending", False)]
    inv = build_inventory(m)
    assert inv["totals"]["counted"] == 1 and inv["totals"]["pending"] == 1
    tire = next(c for c in inv["classes"] if c["class_name"] == "tire_rubber")
    assert tire["counted"] == 1 and tire["mean_conf"] == 0.9


def test_review_overrides_model():
    m = make_mission(frames=[[det("tire_rubber", 0.9), det("can_metal", 0.3), det("bag_plastic", 0.8)]])
    m["reviews"] = {
        "00000-00": {"action": "reject", "reviewer": "a", "at": "t"},
        "00000-01": {"action": "relabel", "class_name": "bottle_glass", "reviewer": "a", "at": "t"},
        "00000-02": {"action": "confirm", "reviewer": "a", "at": "t"},
    }
    inv = build_inventory(m)
    counted = {c["class_name"]: c["counted"] for c in inv["classes"]}
    assert counted["tire_rubber"] == 0 and counted["bottle_glass"] == 1 and counted["can_metal"] == 0
    assert inv["totals"] == {"counted": 2, "auto": 0, "verified": 2, "pending": 0, "rejected": 1, "added": 0}
    assert inv["review"] == {"reviewed": 3, "reviewable": 3, "pct": 100.0}


def test_video_estimate_is_a_range_stills_is_a_sum():
    frames = [[det("tire_rubber", 0.9)], [det("tire_rubber", 0.9), det("tire_rubber", 0.8, x=200)], [det("tire_rubber", 0.9)]]
    v = build_inventory(make_mission("video", frames))
    assert v["estimate"]["method"] == "range"
    assert (v["estimate"]["low"], v["estimate"]["high"]) == (2, 4)
    assert v["duration_s"] == 4.0
    s = build_inventory(make_mission("stills", frames))
    assert (s["estimate"]["low"], s["estimate"]["high"]) == (4, 4)
    assert s["duration_s"] is None


def test_timeline_and_hotspots():
    frames = [[], [det("can_metal", 0.9), det("can_metal", 0.95, x=100)], [det("net_plastic", 0.7), det("bag_plastic", 0.3)]]
    inv = build_inventory(make_mission("video", frames))
    assert [t["total"] for t in inv["timeline"]] == [0, 2, 1]
    assert inv["timeline"][2]["pending"] == 1
    assert [h["index"] for h in inv["hotspots"]] == [1, 2]
    assert inv["frames_with_debris"] == 2


def test_review_queue_puts_most_ambiguous_first():
    m = make_mission(frames=[[det("tire_rubber", 0.95), det("can_metal", 0.48), det("bag_plastic", 0.27)], [det("net_plastic", 0.55)]])
    q = review_queue(m)
    assert [e["id"] for e in q] == ["00000-01", "00000-02", "00001-00", "00000-00"]
    m["reviews"]["00000-01"] = {"action": "confirm", "reviewer": "", "at": "t"}
    assert "00000-01" not in [e["id"] for e in review_queue(m)]


def test_run_frames_batches_at_most_eight(make_box):
    model = FakeModel([make_box([0, 0, 10, 10], 5, 0.9), make_box([0, 0, 10, 10], 0, 0.1)])
    m = new_mission("batch", "stills", floor=0.25)
    saved = []
    frames = [(f"{i}.jpg", None, Image.new("RGB", (2000, 1000))) for i in range(19)]
    progress = []
    run_frames(m, model, frames, save_frame=lambda i, img: saved.append(img.size) or f"frames/{i:05d}.jpg",
               on_progress=lambda mm: progress.append(mm["progress"]["done"]))
    assert [c["batch"] for c in model.calls] == [BATCH_SIZE, BATCH_SIZE, 3]
    assert all(c["conf"] == 0.25 for c in model.calls)
    assert progress == [8, 16, 19]
    assert saved[0] == (1280, 640)  # downscaled before inference and storage
    assert m["status"] == "ready" and len(m["frames"]) == 19
    assert all(len(f["detections"]) == 1 for f in m["frames"])  # 0.1 box dropped by the floor
    assert summary(m)["counted"] == 19


def test_downscale_keeps_small_images():
    img = Image.new("RGB", (300, 200))
    assert downscale(img).size == (300, 200)
    assert downscale(Image.new("RGB", (4000, 3000)), 1000).size == (1000, 750)


def test_sample_video_steps_by_time(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = tmp_path / "dive.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not w.isOpened():
        pytest.skip("no MJPG writer in this OpenCV build")
    for i in range(35):
        w.write(np.full((48, 64, 3), i * 7 % 255, dtype=np.uint8))
    w.release()
    out = list(sample_video(str(path), every_s=1.0, max_frames=10))
    assert [round(t, 1) for t, _ in out] == [0.0, 1.0, 2.0, 3.0]
    assert out[0][1].size == (64, 48)
    assert len(list(sample_video(str(path), every_s=0.5, max_frames=2))) == 2
    with pytest.raises(ValueError):
        list(sample_video(str(tmp_path / "missing.mp4")))
