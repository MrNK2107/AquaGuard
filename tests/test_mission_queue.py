"""Missions run one at a time on a single worker thread fed by a bounded queue (probe_aq3.py)."""

import io
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image


def png():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def threaded(fake_model, open_access, mission_store, monkeypatch):
    """A real (threaded) worker with a fake, slow process_mission that records concurrency."""
    import app.missions_api as mapi

    stats = {"active": 0, "peak": 0, "done": 0}
    lk = threading.Lock()
    gate = threading.Event()

    def fake(mid, paths, every_s, mf):
        with lk:
            stats["active"] += 1
            stats["peak"] = max(stats["peak"], stats["active"])
        gate.wait(5)
        time.sleep(0.05)
        with lk:
            stats["active"] -= 1
            stats["done"] += 1

    monkeypatch.setattr(mapi, "process_mission", fake)
    worker = mapi.MissionWorker(queue_size=8, per_client=8)
    monkeypatch.setattr(mapi, "worker", worker)
    yield TestClient(mapi._main.app), worker, stats, gate
    gate.set()
    worker.stop(timeout=5)


def post(c, client_id=None):
    headers = {"X-Forwarded-For": client_id} if client_id else {}
    return c.post("/api/v1/missions", files={"files": ("a.png", io.BytesIO(png()), "image/png")}, headers=headers)


def test_four_simultaneous_uploads_are_processed_one_at_a_time(threaded):
    c, worker, stats, gate = threaded
    codes = []
    ts = [threading.Thread(target=lambda: codes.append(post(c).status_code)) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert codes == [202] * 4
    gate.set()
    assert worker.join(timeout=10)
    assert stats["done"] == 4 and stats["peak"] == 1


def test_full_queue_returns_503(fake_model, open_access, mission_store, monkeypatch):
    import app.missions_api as mapi

    gate = threading.Event()
    monkeypatch.setattr(mapi, "process_mission", lambda *a: gate.wait(5))
    worker = mapi.MissionWorker(queue_size=2, per_client=10)
    monkeypatch.setattr(mapi, "worker", worker)
    c = TestClient(mapi._main.app)
    try:
        first = post(c)
        assert first.status_code == 202
        assert worker.wait_running(timeout=5)  # first one leaves the queue and runs
        assert [post(c).status_code for _ in range(2)] == [202, 202]
        r = post(c)
        assert r.status_code == 503 and "Retry-After" in r.headers
        assert r.json()["status"] == 503
        assert len(mission_store.list()) == 3  # the rejected upload left nothing behind
    finally:
        gate.set()
        worker.stop(timeout=5)


def test_per_client_cap_returns_429(fake_model, open_access, mission_store, monkeypatch):
    import app.missions_api as mapi

    gate = threading.Event()
    monkeypatch.setattr(mapi, "process_mission", lambda *a: gate.wait(5))
    worker = mapi.MissionWorker(queue_size=10, per_client=2)
    monkeypatch.setattr(mapi, "worker", worker)
    c = TestClient(mapi._main.app)
    try:
        assert [post(c).status_code for _ in range(2)] == [202, 202]  # running + queued both count
        r = post(c)
        assert r.status_code == 429 and "Retry-After" in r.headers
        gate.set()
        assert worker.join(timeout=10)
        assert post(c).status_code == 202  # slots are released when missions finish
    finally:
        gate.set()
        worker.stop(timeout=5)


def test_worker_survives_a_crashing_job(monkeypatch):
    import app.missions_api as mapi

    ran = []

    def job(mid, *a):
        ran.append(mid)
        if mid == "boom":
            raise RuntimeError("boom")

    monkeypatch.setattr(mapi, "process_mission", job)
    w = mapi.MissionWorker(queue_size=4, per_client=4)
    try:
        for mid in ("boom", "ok"):
            w.submit(w.reserve("ip:x"), mid, [], 1.0, 1)
        assert w.join(timeout=5)
        assert ran == ["boom", "ok"] and w.outstanding("ip:x") == 0
    finally:
        w.stop(timeout=5)


def test_worker_settings_from_env():
    import app.missions_api as mapi

    w = mapi.MissionWorker.from_env({"AQUAGUARD_MISSION_QUEUE": "7", "AQUAGUARD_MISSIONS_PER_CLIENT": "3"})
    assert (w.queue_size, w.per_client) == (7, 3)
    w = mapi.MissionWorker.from_env({})
    assert (w.queue_size, w.per_client) == (4, 2)


def test_docstring_matches_behaviour():
    import app.missions_api as mapi

    doc = mapi.__doc__.lower()
    assert "background task" not in doc and "queue" in doc and "one worker thread" in doc
