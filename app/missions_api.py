"""Mission endpoints: upload a dive, get an inventory report, review detections, export corrections.

All data routes require access (API key when configured) and are rate limited.
Uploads go into a bounded queue drained by one worker thread, so exactly one mission is
processed at a time per server process (the detector is shared), which keeps memory flat
on a small vessel PC. Poll ``GET /api/v1/missions/{id}`` until ``status`` is ``ready``.

  AQUAGUARD_MISSION_QUEUE        missions that may wait behind the running one (default 4);
                                 a full queue answers 503 with Retry-After.
  AQUAGUARD_MISSIONS_PER_CLIENT  missions one API key / client IP may have queued or running
                                 (default 2); over it answers 429.
"""

from __future__ import annotations

import csv
import io
import os
import queue
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.background import BackgroundTask
from PIL import Image

from . import main as _main
from .brand import CLASS_LABELS, CLASS_NAMES
from .media import UnsupportedMedia, open_image
from .mission import (IMAGE_EXTS, MAX_SIDE, VIDEO_EXTS, build_inventory, downscale, effective_detections, new_mission,
                      review_queue, run_frames, sample_video, summary)
from .report import render_html, render_pdf
from .review import (ReviewError, add_box, apply_review, build_corrections_zip, confirm_frame,
                     correction_records, remove_added)
from .store import _ID_RE, MissionNotFound, MissionStore

router = APIRouter(prefix="/api/v1/missions", tags=["missions"])
WEB_DIR = Path(__file__).parent / "web"

MAX_STILLS = 300
MAX_STILL_BYTES = 15 * 1024 * 1024
MAX_VIDEO_BYTES = int(os.environ.get("AQUAGUARD_MAX_VIDEO_MB", "500")) * 1024 * 1024
MAX_FRAMES = int(os.environ.get("AQUAGUARD_MAX_FRAMES", "600"))
CHUNK = 1024 * 1024

store = MissionStore(os.environ.get("AQUAGUARD_MISSIONS_DIR", "missions"))
_edit_locks: dict = {}
_edit_locks_guard = threading.Lock()


def _edit_lock(mission_id: str) -> threading.Lock:
    """Serialises load-modify-save of one mission, so two reviewers can't lose each other's clicks."""
    if not _ID_RE.match(mission_id or ""):
        raise HTTPException(status_code=404, detail=f"Mission '{mission_id}' not found.")
    with _edit_locks_guard:
        return _edit_locks.setdefault(mission_id, threading.Lock())


def recover_interrupted(s: Optional[MissionStore] = None) -> int:
    """Mark missions left 'processing' by a previous server process as failed (called at startup)."""
    s = s or _store()
    n = 0
    for m in s.list():
        if m.get("status") == "processing":
            m["status"] = "failed"
            m["error"] = (f"Server restarted while processing ({len(m.get('frames', []))} frames done). "
                          "Upload the footage again.")
            s.save(m)
            s.clear_uploads(m["id"])
            n += 1
    return n


def _store() -> MissionStore:
    return store


def _load(mission_id: str) -> dict:
    try:
        return _store().load(mission_id)
    except MissionNotFound:
        raise HTTPException(status_code=404, detail=f"Mission '{mission_id}' not found.")


def _ready(mission: dict) -> dict:
    if mission["status"] != "ready":
        raise HTTPException(status_code=409, detail=f"Mission is {mission['status']}; try again when it is ready.")
    return mission


def _loader(mission_id: str):
    s = _store()
    return lambda frame: s.open_frame(mission_id, frame["image"])


def _kind_of(name: str) -> Optional[str]:
    ext = Path(name or "").suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "stills"
    return None


async def _stream_to_disk(upload: UploadFile, path: Path, limit: int) -> int:
    size = 0
    with open(path, "wb") as fh:
        while True:
            chunk = await upload.read(CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                fh.close()
                path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"'{upload.filename}' exceeds the {limit // (1024 * 1024)} MB limit.")
            fh.write(chunk)
    return size


# ----------------------------------------------------------------------------
# Processing
# ----------------------------------------------------------------------------
class _LockedModel:
    """Takes the shared inference lock per batch, so /detect and /reason interleave with a running mission."""

    def predict(self, **kwargs):
        with _main.infer_lock:
            return _main.get_model().predict(**kwargs)


def process_mission(mission_id: str, paths: List[str], every_s: float, max_frames: int) -> None:
    s = _store()
    try:
        mission = s.load(mission_id)
    except MissionNotFound:  # deleted before processing started
        return
    names = mission["source"]["files"]  # original upload names (stored copies carry an index prefix)
    try:
        if mission["source"]["kind"] == "video":
            frames = ((names[0], t, img) for t, img in sample_video(paths[0], every_s, max_frames))
            mission["progress"]["total"] = None  # unknown until decoded
        else:
            def stills():
                for name, p in zip(names, paths):
                    try:
                        with open_image(p) as im:  # allowlisted formats, pixel cap enforced
                            im.draft("RGB", (MAX_SIDE, MAX_SIDE))  # JPEG: decode at reduced size
                            img = downscale(im)  # one full-size image in memory at a time, never a batch
                    except (OSError, ValueError, UnsupportedMedia, Image.DecompressionBombError,
                            Image.DecompressionBombWarning):
                        continue  # unreadable or oversized image: skip it, keep the mission
                    yield name, None, img
            frames = stills()
            mission["progress"]["total"] = len(paths)
        s.save(mission)
        run_frames(mission, _LockedModel(), frames,
                   save_frame=lambda i, img: s.save_frame(mission_id, i, img),
                   on_progress=s.save)
        if not mission["frames"]:
            mission["status"], mission["error"] = "failed", "No readable frames in the upload."
        mission["progress"]["total"] = len(mission["frames"])
    except Exception as exc:  # surface the failure on the mission instead of losing it in a log
        _main.logger.error(f"mission {mission_id} failed: {exc}", exc_info=True)
        mission["status"], mission["error"] = "failed", str(exc)[:300]
    finally:
        s.save(mission)
        s.clear_uploads(mission_id)


class QueueFull(Exception):
    pass


class ClientBusy(Exception):
    pass


class MissionWorker:
    """One worker thread fed by a bounded queue: missions run strictly one after another.

    ``reserve`` is called before the upload is streamed to disk, so a rejected request costs no
    disk; ``submit`` hands the reserved slot to the worker; ``release`` gives it back on failure.
    Per-client accounting covers queued *and* running missions. ``sync=True`` runs jobs inline
    (tests only).
    """

    def __init__(self, queue_size: int = 4, per_client: int = 2, sync: bool = False) -> None:
        self.queue_size, self.per_client, self.sync = queue_size, per_client, sync
        self._q = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._waiting = 0            # reserved or queued, not started
        self._busy = 0               # reserved + queued + running
        self._per_client: dict = {}  # client -> reserved + queued + running
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def from_env(cls, env=os.environ) -> "MissionWorker":
        return cls(int(env.get("AQUAGUARD_MISSION_QUEUE", "4")), int(env.get("AQUAGUARD_MISSIONS_PER_CLIENT", "2")))

    def outstanding(self, client: str) -> int:
        with self._lock:
            return self._per_client.get(client, 0)

    def reserve(self, client: str) -> str:
        with self._lock:
            if self._per_client.get(client, 0) >= self.per_client:
                raise ClientBusy(client)
            if self._waiting >= self.queue_size:
                raise QueueFull()
            self._waiting += 1
            self._busy += 1
            self._per_client[client] = self._per_client.get(client, 0) + 1
        return client

    def _finish(self, client: str) -> None:  # caller holds the lock
        self._busy -= 1
        n = self._per_client.get(client, 1) - 1
        if n:
            self._per_client[client] = n
        else:
            self._per_client.pop(client, None)
        self._idle.notify_all()

    def release(self, token: str) -> None:
        """Give back a reservation that never became a job (upload failed or was rejected)."""
        with self._lock:
            self._waiting -= 1
            self._finish(token)

    def submit(self, token: str, mission_id: str, paths: List[str], every_s: float, max_frames: int) -> None:
        job = (token, mission_id, paths, every_s, max_frames)
        if self.sync:
            with self._lock:
                self._waiting -= 1
            self._run(job)
            return
        self._ensure_thread()
        self._q.put_nowait(job)  # cannot overflow: reserve() bounds the number of waiting jobs

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="aquaguard-missions", daemon=True)
                self._thread.start()

    def _run(self, job) -> None:
        token, mission_id, paths, every_s, max_frames = job
        try:
            process_mission(mission_id, paths, every_s, max_frames)  # module global: swappable in tests
        except Exception as exc:  # process_mission records its own failures; never kill the worker
            _main.logger.error(f"mission worker: {mission_id} crashed: {exc}", exc_info=True)
        finally:
            self._running.clear()
            with self._lock:
                self._finish(token)

    def _loop(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                return
            with self._lock:
                self._waiting -= 1
            self._running.set()
            self._run(job)

    def wait_running(self, timeout: float = 5.0) -> bool:
        return self._running.wait(timeout)

    def join(self, timeout: float = 10.0) -> bool:
        """Wait until nothing is reserved, queued or running."""
        with self._lock:
            return self._idle.wait_for(lambda: self._busy == 0, timeout)

    def stop(self, timeout: float = 5.0) -> None:
        t = self._thread
        if t is not None and t.is_alive():
            self._q.put(None)
            t.join(timeout)


worker = MissionWorker.from_env()


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@router.post("", status_code=202)
async def create_mission(
    files: List[UploadFile] = File(..., description="One dive video, or up to 300 still images"),
    name: str = Form("Untitled survey"),
    site: str = Form(""),
    operator: str = Form(""),
    conf: float = Form(0.50, ge=0.0, le=1.0),
    floor: float = Form(0.25, ge=0.0, le=1.0),
    every_s: float = Form(1.0, gt=0.0, le=60.0),
    max_frames: int = Form(120, ge=1),
    _client: str = Depends(_main.require_access(10.0)),
):
    """Start a survey mission. Returns 202 with the mission id; poll ``GET /api/v1/missions/{id}``.

    429 when this client already has ``AQUAGUARD_MISSIONS_PER_CLIENT`` missions queued or
    processing; 503 when the queue is full."""
    if floor > conf:
        raise HTTPException(status_code=422, detail="'floor' must be <= 'conf'.")
    kinds = {_kind_of(f.filename) for f in files}
    if None in kinds:
        raise HTTPException(status_code=400, detail="Upload images (jpg/png/webp/bmp) or one video (mp4/mov/avi/mkv/webm).")
    if kinds == {"video", "stills"} or ("video" in kinds and len(files) > 1):
        raise HTTPException(status_code=400, detail="Upload either one video or a batch of images, not both.")
    kind = kinds.pop()
    if kind == "stills" and len(files) > MAX_STILLS:
        raise HTTPException(status_code=400, detail=f"At most {MAX_STILLS} images per mission.")

    w = worker
    try:
        token = w.reserve(_client)
    except ClientBusy:
        raise HTTPException(status_code=429, headers={"Retry-After": "30"},
                            detail=f"You already have {w.per_client} mission(s) queued or processing; "
                                   "wait for one to finish.")
    except QueueFull:
        raise HTTPException(status_code=503, headers={"Retry-After": "30"},
                            detail="The mission queue is full; try again shortly.")
    s = _store()
    try:
        mission = new_mission(name, kind, conf=conf, floor=floor, site=site, operator=operator,
                              sample_every_s=every_s if kind == "video" else None,
                              model=str(_main.MODEL_PATH if _main.MODEL_PATH.exists() else _main.FALLBACK))
        mission["source"]["files"] = [Path(f.filename).name for f in files]
        s.save(mission)
    except BaseException:
        w.release(token)
        raise
    paths = []
    try:
        for i, f in enumerate(files):
            # index prefix: stills from different folders often share names like 0001.jpg
            p = s.save_upload(mission["id"], f"{i:04d}_{Path(f.filename).name}", b"")
            await _stream_to_disk(f, p, MAX_VIDEO_BYTES if kind == "video" else MAX_STILL_BYTES)
            paths.append(str(p))
    except BaseException:  # 413, client disconnect, disk full: never leave a mission stuck in "processing"
        w.release(token)
        try:
            s.delete(mission["id"])
        except (MissionNotFound, OSError):
            pass
        raise
    w.submit(token, mission["id"], paths, every_s, min(max_frames, MAX_FRAMES))
    return summary(mission)


@router.get("")
def list_missions(_client: str = Depends(_main.require_access(0.1))):
    return {"missions": [summary(m) for m in _store().list()]}


@router.get("/{mission_id}")
def get_mission(mission_id: str, frames: bool = Query(False, description="Include per-frame detections"),
                _client: str = Depends(_main.require_access(0.1))):
    m = _load(mission_id)
    out = summary(m)
    out.update({"settings": m["settings"], "source": m["source"], "model": m.get("model"),
                "operator": m.get("operator", "")})
    if m["status"] == "ready":
        out["inventory"] = build_inventory(m)
    if frames:
        out["frames"] = [{**{k: f[k] for k in ("index", "source", "t", "width", "height")},
                          "detections": effective_detections(m, f)} for f in m["frames"]]
    return out


@router.delete("/{mission_id}", status_code=204)
def delete_mission(mission_id: str, _client: str = Depends(_main.require_access(1.0))):
    with _edit_lock(mission_id):
        m = _load(mission_id)
        if m["status"] == "processing":
            raise HTTPException(status_code=409, detail="Mission is still processing; delete it when it is ready or failed.")
        try:
            _store().delete(mission_id)
        except MissionNotFound:
            raise HTTPException(status_code=404, detail=f"Mission '{mission_id}' not found.")
    return Response(status_code=204)


@router.get("/{mission_id}/report.html", response_class=HTMLResponse)
def report_html(mission_id: str, _client: str = Depends(_main.require_access(1.0))):
    m = _ready(_load(mission_id))
    return HTMLResponse(render_html(m, _loader(mission_id), version=f"v{_main.app.version}"))


@router.get("/{mission_id}/report.pdf")
def report_pdf(mission_id: str, _client: str = Depends(_main.require_access(2.0))):
    m = _ready(_load(mission_id))
    pdf = render_pdf(m, _loader(mission_id))
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{mission_id}-report.pdf"'})


def _csv_safe(value) -> str:
    """Neutralise spreadsheet formulas in uploader-controlled text (e.g. a file named '=HYPERLINK(...)')."""
    value = str(value)
    return "'" + value if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


@router.get("/{mission_id}/detections.csv")
def detections_csv(mission_id: str, _client: str = Depends(_main.require_access(0.5))):
    m = _ready(_load(mission_id))
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["frame_index", "frame_source", "t_s", "box_id", "class_name", "model_class", "confidence",
                "status", "counted", "x1", "y1", "x2", "y2"])
    for f in m["frames"]:
        for e in effective_detections(m, f):
            w.writerow([f["index"], _csv_safe(f["source"]), f["t"], e["id"], e["class_name"], e.get("model_class"),
                        e.get("confidence"), e["status"], e["counted"], e["x1"], e["y1"], e["x2"], e["y2"]])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{mission_id}-detections.csv"'})


@router.get("/{mission_id}/frames/{index}.jpg")
def frame_image(mission_id: str, index: int, _client: str = Depends(_main.require_access(0.05))):
    m = _load(mission_id)
    if not (0 <= index < len(m["frames"])):
        raise HTTPException(status_code=404, detail="Frame not found.")
    try:
        path = _store().frame_path(mission_id, m["frames"][index]["image"])
    except MissionNotFound:
        raise HTTPException(status_code=404, detail="Frame not found.")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frame image missing.")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{mission_id}/frames/{index}")
def frame_detail(mission_id: str, index: int, _client: str = Depends(_main.require_access(0.05))):
    m = _load(mission_id)
    if not (0 <= index < len(m["frames"])):
        raise HTTPException(status_code=404, detail="Frame not found.")
    f = m["frames"][index]
    return {**{k: f[k] for k in ("index", "source", "t", "width", "height")},
            "cleared": bool(f.get("cleared")), "detections": effective_detections(m, f)}


@router.get("/{mission_id}/review/queue")
def get_review_queue(mission_id: str, limit: int = Query(50, ge=1, le=1000),
                     _client: str = Depends(_main.require_access(0.1))):
    m = _ready(_load(mission_id))
    items = review_queue(m)
    inv = build_inventory(m)
    return {"mission_id": mission_id, "threshold": m["settings"]["conf"], "remaining": len(items),
            "review": inv["review"], "items": items[:limit]}


def _review_error(exc: ReviewError):
    raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{mission_id}/review")
def post_review(mission_id: str, payload: dict = Body(..., examples=[{"detection_id": "00003-01", "action": "relabel",
                                                                       "class_name": "bottle_glass", "reviewer": "deck-1"}]),
                _client: str = Depends(_main.require_access(0.1))):
    """Record a decision on one detection: ``confirm``, ``reject``, ``relabel`` (needs ``class_name``) or ``undo``."""
    with _edit_lock(mission_id):
        s = _store()
        m = _ready(_load(mission_id))
        try:
            entry = apply_review(m, str(payload.get("detection_id", "")), str(payload.get("action", "")),
                                 payload.get("class_name"), str(payload.get("reviewer", "")), str(payload.get("note", "")))
        except ReviewError as exc:
            _review_error(exc)
        s.save(m)
    return {"detection_id": payload.get("detection_id"), "review": entry, "inventory_totals": build_inventory(m)["totals"]}


@router.post("/{mission_id}/frames/{index}/confirm")
def post_confirm_frame(mission_id: str, index: int, payload: dict = Body(default={}),
                       _client: str = Depends(_main.require_access(0.1))):
    """Accept every undecided box in the frame (or mark an empty frame as clear)."""
    with _edit_lock(mission_id):
        s = _store()
        m = _ready(_load(mission_id))
        try:
            n = confirm_frame(m, index, str((payload or {}).get("reviewer", "")))
        except ReviewError as exc:
            _review_error(exc)
        s.save(m)
    return {"frame_index": index, "confirmed": n}


@router.post("/{mission_id}/frames/{index}/boxes", status_code=201)
def post_add_box(mission_id: str, index: int, payload: dict = Body(..., examples=[{"x1": 10, "y1": 20, "x2": 120, "y2": 160,
                                                                                    "class_name": "tire_rubber"}]),
                 _client: str = Depends(_main.require_access(0.1))):
    """Add a box for debris the model missed (stored-frame pixel coordinates)."""
    with _edit_lock(mission_id):
        s = _store()
        m = _ready(_load(mission_id))
        try:
            entry = add_box(m, index, payload, str(payload.get("class_name", "")), str(payload.get("reviewer", "")))
        except ReviewError as exc:
            _review_error(exc)
        s.save(m)
    return entry


@router.delete("/{mission_id}/boxes/{box_id}", status_code=204)
def delete_added_box(mission_id: str, box_id: str, _client: str = Depends(_main.require_access(0.1))):
    with _edit_lock(mission_id):
        s = _store()
        m = _ready(_load(mission_id))
        try:
            remove_added(m, box_id)
        except ReviewError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        s.save(m)
    return Response(status_code=204)


@router.get("/{mission_id}/corrections.jsonl")
def corrections_jsonl(mission_id: str, _client: str = Depends(_main.require_access(0.5))):
    import json
    m = _ready(_load(mission_id))
    body = "".join(json.dumps(r) + "\n" for r in correction_records(m))
    return Response(body, media_type="application/x-ndjson",
                    headers={"Content-Disposition": f'attachment; filename="{mission_id}-corrections.jsonl"'})


@router.get("/{mission_id}/corrections.zip")
def corrections_zip(mission_id: str, _client: str = Depends(_main.require_access(2.0))):
    """Retraining bundle: corrections.jsonl + YOLO labels/images for fully reviewed frames + data.yaml."""
    s = _store()
    m = _ready(_load(mission_id))
    # built in a temp file and streamed, so a large export never sits in memory as one bytes object
    tmp = tempfile.NamedTemporaryFile(prefix="aquaguard-", suffix=".zip", delete=False)
    try:
        with tmp:
            build_corrections_zip(m, lambda rel: s.read_frame(mission_id, rel), out=tmp)
    except BaseException:
        os.unlink(tmp.name)
        raise
    return FileResponse(tmp.name, media_type="application/zip", filename=f"{mission_id}-corrections.zip",
                        background=BackgroundTask(os.unlink, tmp.name))


# ----------------------------------------------------------------------------
# Operator console (static shell; data comes from the authenticated routes above)
# ----------------------------------------------------------------------------
console_router = APIRouter(include_in_schema=False)


@console_router.get("/console", response_class=HTMLResponse)
def console():
    return HTMLResponse((WEB_DIR / "console.html").read_text(encoding="utf-8"))


@console_router.get("/console/meta")
def console_meta():
    """Public, non-sensitive config the console needs before it has a key."""
    return {"auth_required": _main.access.enabled, "classes": CLASS_NAMES, "labels": CLASS_LABELS,
            "version": _main.app.version}
