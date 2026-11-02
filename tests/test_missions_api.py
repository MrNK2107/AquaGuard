"""Mission endpoints end-to-end against the fake detector (no weights, no GPU)."""

import io
import zipfile

import pytest
from PIL import Image


def jpg(color=(20, 60, 80), size=(320, 240)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def create(client, n=3, **form):
    files = [("files", (f"dive_{i}.jpg", io.BytesIO(jpg()), "image/jpeg")) for i in range(n)]
    data = {"name": "Harbour wall survey", "site": "Pier 4", "operator": "deck-1", **form}
    return client.post("/api/v1/missions", files=files, data=data)


@pytest.fixture
def ready_mission(client, fake_model, make_box):
    fake_model.boxes = [make_box([10, 10, 80, 90], 5, 0.91), make_box([100, 20, 150, 60], 1, 0.4)]
    r = create(client)
    assert r.status_code == 202
    return r.json()["id"]


def test_create_runs_detection_in_background(client, ready_mission, fake_model, mission_store):
    m = client.get(f"/api/v1/missions/{ready_mission}").json()
    assert m["status"] == "ready" and m["frames"] == 3 and m["kind"] == "stills"
    inv = m["inventory"]
    assert inv["totals"]["counted"] == 3 and inv["totals"]["pending"] == 3
    assert inv["estimate"] == {"method": "sum", "low": 3, "high": 3, "note": inv["estimate"]["note"]}
    assert fake_model.calls[-1]["batch"] == 3 and fake_model.calls[-1]["conf"] == 0.25
    # uploads are cleaned up, downscaled frames are kept for review
    d = mission_store.root / ready_mission
    assert not (d / "uploads").exists()
    assert sorted(p.name for p in (d / "frames").iterdir()) == ["00000.jpg", "00001.jpg", "00002.jpg"]


def test_list_and_detail_with_frames(client, ready_mission):
    lst = client.get("/api/v1/missions").json()["missions"]
    assert [m["id"] for m in lst] == [ready_mission]
    m = client.get(f"/api/v1/missions/{ready_mission}?frames=true").json()
    assert m["frames"][0]["detections"][0]["status"] == "auto"
    assert m["frames"][0]["detections"][1]["status"] == "pending"


def test_reports_and_exports(client, ready_mission):
    r = client.get(f"/api/v1/missions/{ready_mission}/report.html")
    assert r.status_code == 200 and "Harbour wall survey" in r.text and "<svg" in r.text
    assert "data:image/jpeg;base64," in r.text  # hotspot thumbnails embedded
    csv_text = client.get(f"/api/v1/missions/{ready_mission}/detections.csv").text
    assert csv_text.splitlines()[0].startswith("frame_index,frame_source")
    assert len(csv_text.strip().splitlines()) == 7
    img = client.get(f"/api/v1/missions/{ready_mission}/frames/1.jpg")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/v1/missions/{ready_mission}/frames/9.jpg").status_code == 404


def test_pdf_report(client, ready_mission):
    pytest.importorskip("matplotlib")
    r = client.get(f"/api/v1/missions/{ready_mission}/report.pdf")
    assert r.status_code == 200 and r.content[:4] == b"%PDF"
    assert "attachment" in r.headers["content-disposition"]


def test_review_flow_updates_inventory_and_export(client, ready_mission):
    base = f"/api/v1/missions/{ready_mission}"
    q = client.get(f"{base}/review/queue").json()
    assert q["remaining"] == 6
    assert q["items"][0]["status"] == "pending"
    pending_id = q["items"][0]["id"]

    r = client.post(f"{base}/review", json={"detection_id": pending_id, "action": "relabel", "class_name": "bottle_glass", "reviewer": "deck-1"})
    assert r.status_code == 200 and r.json()["inventory_totals"]["verified"] == 1
    assert client.post(f"{base}/review", json={"detection_id": pending_id, "action": "relabel", "class_name": "kelp"}).status_code == 400
    assert client.post(f"{base}/review", json={"detection_id": "zzz", "action": "confirm"}).status_code == 400

    # the relabeled box is already decided, so only frame 0's other box is confirmed
    assert client.post(f"{base}/frames/0/confirm", json={"reviewer": "deck-1"}).json()["confirmed"] == 1
    box = client.post(f"{base}/frames/2/boxes", json={"x1": 5, "y1": 5, "x2": 60, "y2": 60, "class_name": "net_plastic"})
    assert box.status_code == 201
    frame = client.get(f"{base}/frames/2").json()
    assert frame["detections"][-1]["status"] == "added"

    zdata = client.get(f"{base}/corrections.zip").content
    names = zipfile.ZipFile(io.BytesIO(zdata)).namelist()
    assert "labels/00000.txt" in names and "images/00000.jpg" in names
    jsonl = client.get(f"{base}/corrections.jsonl").text.strip().splitlines()
    assert len(jsonl) == 3  # relabel + frame confirm + added box

    assert client.delete(f"{base}/boxes/{box.json()['id']}").status_code == 204
    assert client.delete(f"{base}/boxes/{box.json()['id']}").status_code == 404


def test_upload_validation(client):
    bad = client.post("/api/v1/missions", files=[("files", ("notes.txt", io.BytesIO(b"hi"), "text/plain"))])
    assert bad.status_code == 400
    mixed = client.post("/api/v1/missions", files=[("files", ("a.mp4", io.BytesIO(b"x"), "video/mp4")),
                                                  ("files", ("b.jpg", io.BytesIO(jpg()), "image/jpeg"))])
    assert mixed.status_code == 400 and "not both" in mixed.json()["detail"]
    r = create(client, 1, conf="0.3", floor="0.4")
    assert r.status_code == 422
    assert client.get("/api/v1/missions").json()["missions"] == []


def test_unreadable_stills_fail_the_mission(client):
    files = [("files", ("broken.jpg", io.BytesIO(b"not a jpeg"), "image/jpeg"))]
    mid = client.post("/api/v1/missions", files=files).json()["id"]
    m = client.get(f"/api/v1/missions/{mid}").json()
    assert m["status"] == "failed" and "No readable frames" in m["error"]
    assert client.get(f"/api/v1/missions/{mid}/report.html").status_code == 409


def test_unknown_and_malformed_ids(client):
    assert client.get("/api/v1/missions/m-20990101-000000-abcdef").status_code == 404
    assert client.get("/api/v1/missions/..%2F..%2Fetc").status_code == 404
    assert client.delete("/api/v1/missions/m-20990101-000000-abcdef").status_code == 404


def test_delete_mission(client, ready_mission):
    assert client.delete(f"/api/v1/missions/{ready_mission}").status_code == 204
    assert client.get(f"/api/v1/missions/{ready_mission}").status_code == 404


def test_console_shell_and_meta(client):
    r = client.get("/console")
    assert r.status_code == 200 and "AquaGuard" in r.text
    meta = client.get("/console/meta").json()
    assert meta["auth_required"] is False and len(meta["classes"]) == 6


def test_duplicate_file_names_are_kept_apart(client, fake_model, make_box):
    fake_model.boxes = [make_box([10, 10, 80, 90], 5, 0.9)]
    files = [("files", ("0001.jpg", io.BytesIO(jpg((i * 60, 40, 40))), "image/jpeg")) for i in range(3)]
    mid = client.post("/api/v1/missions", files=files).json()["id"]
    m = client.get(f"/api/v1/missions/{mid}?frames=true").json()
    assert m["source"]["files"] == ["0001.jpg"] * 3
    detail = client.get(f"/api/v1/missions/{mid}/frames/2").json()
    assert detail["source"] == "0001.jpg"
    assert m["inventory"]["frames"] == 3  # three uploads, three frames (not one overwritten file)


def test_recover_interrupted_marks_stale_missions_failed(mission_store):
    from app.mission import new_mission
    from app.missions_api import recover_interrupted

    stale = new_mission("crashed", "stills")
    done = new_mission("fine", "stills")
    done["status"] = "ready"
    mission_store.save(stale)
    mission_store.save(done)
    assert recover_interrupted(mission_store) == 1
    assert mission_store.load(stale["id"])["status"] == "failed"
    assert "restarted" in mission_store.load(stale["id"])["error"]
    assert mission_store.load(done["id"])["status"] == "ready"


def test_concurrent_reviews_are_not_lost(client, ready_mission):
    from concurrent.futures import ThreadPoolExecutor

    base = f"/api/v1/missions/{ready_mission}"
    ids = [i["id"] for i in client.get(f"{base}/review/queue").json()["items"]]
    with ThreadPoolExecutor(max_workers=6) as pool:
        codes = list(pool.map(lambda d: client.post(f"{base}/review", json={"detection_id": d, "action": "confirm"}).status_code, ids))
    assert codes == [200] * len(ids)
    assert client.get(f"{base}/review/queue").json()["remaining"] == 0


def test_mutations_reject_malformed_ids(client):
    assert client.post("/api/v1/missions/not-an-id/review", json={}).status_code == 404


def test_startup_runs_recovery(fake_model, open_access, mission_store):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.mission import new_mission

    stale = new_mission("crashed", "stills")
    mission_store.save(stale)
    with TestClient(app):
        pass
    assert mission_store.load(stale["id"])["status"] == "failed"


def test_csv_neutralises_formulas_in_file_names(client, fake_model, make_box):
    fake_model.boxes = [make_box([10, 10, 80, 90], 5, 0.9)]
    files = [("files", ('=HYPERLINK("x").jpg', io.BytesIO(jpg()), "image/jpeg"))]
    mid = client.post("/api/v1/missions", files=files).json()["id"]
    row = client.get(f"/api/v1/missions/{mid}/detections.csv").text.splitlines()[1]
    assert "'=HYPERLINK" in row


def test_delete_refused_while_processing(client, mission_store):
    from app.mission import new_mission

    m = new_mission("busy", "stills")
    mission_store.save(m)
    assert client.delete(f"/api/v1/missions/{m['id']}").status_code == 409


def test_oversized_and_broken_images_are_skipped_not_fatal(client, fake_model, make_box, monkeypatch):
    from PIL import Image as PILImage

    fake_model.boxes = [make_box([10, 10, 80, 90], 5, 0.9)]
    monkeypatch.setattr(PILImage, "MAX_IMAGE_PIXELS", 200_000)  # 400x300 fine; 2000x2000 is over 2x the limit, a "bomb"
    files = [("files", ("ok.jpg", io.BytesIO(jpg(size=(400, 300))), "image/jpeg")),
             ("files", ("huge.jpg", io.BytesIO(jpg(size=(2000, 2000))), "image/jpeg")),
             ("files", ("broken.jpg", io.BytesIO(b"xx"), "image/jpeg"))]
    mid = client.post("/api/v1/missions", files=files).json()["id"]
    m = client.get(f"/api/v1/missions/{mid}").json()
    assert m["status"] == "ready" and m["frames"] == 1


def test_large_stills_are_downscaled_before_batching(client, fake_model):
    files = [("files", (f"{i}.jpg", io.BytesIO(jpg(size=(3000, 2000))), "image/jpeg")) for i in range(2)]
    client.post("/api/v1/missions", files=files)
    assert fake_model.calls[-1]["size"][0] <= 1280
