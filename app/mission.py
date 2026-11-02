"""Missions: turn a dive's frames (a video or a batch of stills) into a debris inventory.

A mission is a plain JSON document (see ``new_mission``). Frames hold the
model's raw detections; human review decisions live next to them in
``mission["reviews"]`` so the original model output is never overwritten.
Every count in the inventory is derived from those two pieces, which keeps
the report, the review queue and the retraining export consistent.

Counting rule (stated in every report):
  * a detection at or above ``settings.conf`` is counted automatically;
  * a detection between ``settings.floor`` and ``settings.conf`` is *pending*:
    it is shown in the review queue but not counted until a human confirms it;
  * a human decision always wins (confirm / relabel / reject), and boxes a
    reviewer adds for missed debris are counted as well.

Video frames overlap, so the same tyre is usually seen in many frames. Without
object tracking the honest number is a range: the per-class *peak in a single
frame* is a lower bound on distinct items, the detection total is an upper bound.
"""

from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from PIL import Image

from .brand import CLASS_COLORS_HEX, CLASS_LABELS, CLASS_NAMES

DEFAULT_CONF = 0.50       # auto-count threshold
DEFAULT_FLOOR = 0.25      # below this a detection is dropped entirely
MAX_SIDE = 1280           # stored frames are downscaled to this (model letterboxes to 640 anyway)
BATCH_SIZE = 8

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ----------------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------------
def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_mission_id(now: Optional[float] = None) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now if now is not None else time.time()))
    return f"m-{stamp}-{uuid.uuid4().hex[:6]}"


def new_mission(name: str, kind: str, *, conf: float = DEFAULT_CONF, floor: float = DEFAULT_FLOOR,
                sample_every_s: Optional[float] = None, site: str = "", operator: str = "",
                model: str = "", mission_id: Optional[str] = None) -> dict:
    if kind not in {"video", "stills"}:
        raise ValueError("kind must be 'video' or 'stills'")
    if not (0.0 <= floor <= conf <= 1.0):
        raise ValueError("need 0 <= floor <= conf <= 1")
    return {
        "id": mission_id or new_mission_id(),
        "name": name.strip() or "Untitled survey",
        "site": site.strip(),
        "operator": operator.strip(),
        "created_at": utcnow(),
        "status": "processing",
        "error": None,
        "progress": {"done": 0, "total": 0},
        "source": {"kind": kind, "files": [], "sample_every_s": sample_every_s},
        "settings": {"conf": conf, "floor": floor},
        "model": model,
        "frames": [],
        "reviews": {},
    }


def detections_from_result(result, names: Sequence[str] = CLASS_NAMES) -> List[dict]:
    """Flatten one ultralytics ``Results`` object into dicts sorted by confidence."""
    out = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return out
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cid = int(box.cls[0].item())
        out.append({
            "x1": round(float(x1), 1), "y1": round(float(y1), 1),
            "x2": round(float(x2), 1), "y2": round(float(y2), 1),
            "class_id": cid,
            "class_name": names[cid] if 0 <= cid < len(names) else str(cid),
            "confidence": float(box.conf[0].item()),
        })
    return sorted(out, key=lambda d: d["confidence"], reverse=True)


def frame_record(index: int, source: str, t: Optional[float], size: Tuple[int, int],
                 detections: List[dict], image: str = "") -> dict:
    dets = []
    for k, d in enumerate(detections):
        d = dict(d)
        d["id"] = f"{index:05d}-{k:02d}"
        d["confidence"] = round(float(d["confidence"]), 4)
        dets.append(d)
    return {
        "index": index,
        "source": source,
        "t": None if t is None else round(float(t), 2),
        "width": size[0],
        "height": size[1],
        "image": image,
        "detections": dets,
        "added": [],
    }


# ----------------------------------------------------------------------------
# Frame sources
# ----------------------------------------------------------------------------
def downscale(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = max_side / float(max(w, h))
    if scale >= 1.0:
        return img
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)


def sample_video(path: str, every_s: float = 1.0, max_frames: int = 120) -> Iterator[Tuple[float, Image.Image]]:
    """Yield ``(timestamp_s, frame)`` every ``every_s`` seconds, at most ``max_frames``.

    Seeks by frame index, so memory stays at one decoded frame regardless of video length.
    """
    import cv2  # opencv-python is a core dependency

    if every_s <= 0:
        raise ValueError("every_s must be > 0")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {Path(path).name}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not fps or math.isnan(fps) or fps <= 0:
            fps = 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(every_s * fps)))
        emitted = 0
        idx = 0
        while emitted < max_frames and (n <= 0 or idx < n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            yield idx / fps, Image.fromarray(rgb)
            emitted += 1
            idx += step
    finally:
        cap.release()


def batched(items: Iterable, size: int = BATCH_SIZE) -> Iterator[list]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def detect_batch(model, images: List[Image.Image], floor: float) -> List[List[dict]]:
    """Run the detector on up to BATCH_SIZE images; one detection list per image."""
    results = model.predict(source=list(images), conf=floor, verbose=False)
    results = list(results)
    if len(results) != len(images):
        raise RuntimeError(f"detector returned {len(results)} results for {len(images)} images")
    return [detections_from_result(r) for r in results]


def run_frames(mission: dict, model, frames: Iterable[Tuple[str, Optional[float], Image.Image]],
               save_frame: Callable[[int, Image.Image], str],
               on_progress: Optional[Callable[[dict], None]] = None,
               batch_size: int = BATCH_SIZE) -> dict:
    """Detect debris on ``(source, t, image)`` frames and append them to ``mission``.

    ``save_frame(index, image) -> relative path`` persists the downscaled frame
    (that copy is what reviewers see, so box coordinates refer to it).
    """
    floor = mission["settings"]["floor"]
    index = len(mission["frames"])
    for batch in batched(frames, batch_size):
        imgs = [downscale(img) for _, _, img in batch]
        dets = detect_batch(model, imgs, floor)
        for (source, t, _), img, d in zip(batch, imgs, dets):
            rel = save_frame(index, img)
            mission["frames"].append(frame_record(index, source, t, img.size, d, rel))
            index += 1
        mission["progress"]["done"] = len(mission["frames"])
        if on_progress:
            on_progress(mission)
    mission["status"] = "ready"
    return mission


# ----------------------------------------------------------------------------
# Effective state (model output + human review)
# ----------------------------------------------------------------------------
def effective_detections(mission: dict, frame: dict) -> List[dict]:
    """Every box in a frame with its current status and class.

    status: auto | pending | confirmed | relabeled | rejected | added
    counted: whether it enters the inventory.
    """
    conf_t = mission["settings"]["conf"]
    reviews = mission.get("reviews", {})
    out = []
    for d in frame["detections"]:
        r = reviews.get(d["id"])
        e = dict(d)
        e["model_class"] = d["class_name"]
        if r is None:
            e["status"] = "auto" if d["confidence"] >= conf_t else "pending"
            e["counted"] = e["status"] == "auto"
        else:
            action = r["action"]
            e["status"] = {"confirm": "confirmed", "relabel": "relabeled", "reject": "rejected"}[action]
            if action == "relabel":
                e["class_name"] = r["class_name"]
            e["counted"] = action != "reject"
            e["review"] = r
        out.append(e)
    for a in frame.get("added", []):
        e = dict(a)
        e.setdefault("model_class", None)
        e["status"] = "added"
        e["counted"] = True
        out.append(e)
    return out


def review_priority(det: dict, conf_t: float) -> Tuple[int, float]:
    """Most ambiguous first: pending boxes nearest the threshold, then auto boxes nearest it."""
    band = 0 if det["status"] == "pending" else 1
    return band, abs(det["confidence"] - conf_t)


def review_queue(mission: dict, limit: Optional[int] = None) -> List[dict]:
    conf_t = mission["settings"]["conf"]
    items = []
    for frame in mission["frames"]:
        for e in effective_detections(mission, frame):
            if e["status"] in {"auto", "pending"}:
                e["frame_index"] = frame["index"]
                e["t"] = frame["t"]
                e["source"] = frame["source"]
                items.append(e)
    items.sort(key=lambda e: review_priority(e, conf_t))
    return items[:limit] if limit else items


def build_inventory(mission: dict) -> dict:
    kind = mission["source"]["kind"]
    per_class = {c: {"counted": 0, "verified": 0, "pending": 0, "rejected": 0, "frames": set(),
                     "peak": 0, "confs": []} for c in CLASS_NAMES}
    timeline = []
    totals = {"counted": 0, "auto": 0, "verified": 0, "pending": 0, "rejected": 0, "added": 0}
    reviewable = 0
    reviewed = 0

    for frame in mission["frames"]:
        eff = effective_detections(mission, frame)
        frame_counts: Dict[str, int] = {}
        pending = 0
        for e in eff:
            cls = e["class_name"]
            slot = per_class.setdefault(cls, {"counted": 0, "verified": 0, "pending": 0, "rejected": 0,
                                              "frames": set(), "peak": 0, "confs": []})
            st = e["status"]
            if st != "added":
                reviewable += 1
                if st in {"confirmed", "relabeled", "rejected"}:
                    reviewed += 1
            if st == "pending":
                slot["pending"] += 1
                totals["pending"] += 1
                pending += 1
                continue
            if st == "rejected":
                slot["rejected"] += 1
                totals["rejected"] += 1
                continue
            slot["counted"] += 1
            totals["counted"] += 1
            if st == "auto":
                totals["auto"] += 1
            elif st == "added":
                totals["added"] += 1
                slot["verified"] += 1
            else:
                totals["verified"] += 1
                slot["verified"] += 1
            if e.get("confidence") is not None and st != "added":
                slot["confs"].append(e["confidence"])
            slot["frames"].add(frame["index"])
            frame_counts[cls] = frame_counts.get(cls, 0) + 1
        for cls, n in frame_counts.items():
            per_class[cls]["peak"] = max(per_class[cls]["peak"], n)
        timeline.append({
            "index": frame["index"], "t": frame["t"], "source": frame["source"],
            "counts": frame_counts, "total": sum(frame_counts.values()), "pending": pending,
        })

    classes = []
    for cls, s in per_class.items():
        if not (s["counted"] or s["pending"] or s["rejected"]) and cls not in CLASS_NAMES:
            continue
        confs = s["confs"]
        classes.append({
            "class_name": cls,
            "label": CLASS_LABELS.get(cls, cls),
            "color": CLASS_COLORS_HEX.get(cls, "#888888"),
            "counted": s["counted"],
            "verified": s["verified"],
            "pending": s["pending"],
            "rejected": s["rejected"],
            "frames_present": len(s["frames"]),
            "peak_in_frame": s["peak"],
            "mean_conf": round(sum(confs) / len(confs), 3) if confs else None,
            "min_conf": round(min(confs), 3) if confs else None,
        })
    classes.sort(key=lambda c: (-c["counted"], CLASS_NAMES.index(c["class_name"]) if c["class_name"] in CLASS_NAMES else 99))

    if kind == "video":
        low = sum(c["peak_in_frame"] for c in classes)
        estimate = {"method": "range", "low": low, "high": totals["counted"],
                    "note": "Video frames overlap. Low = sum of per-class peaks in a single frame "
                            "(a lower bound on distinct items); high = all counted detections."}
    else:
        estimate = {"method": "sum", "low": totals["counted"], "high": totals["counted"],
                    "note": "Stills are treated as independent photos, so every counted detection is one item."}

    n_frames = len(mission["frames"])
    hotspots = pick_hotspots(timeline, n_frames)
    return {
        "frames": n_frames,
        "frames_with_debris": sum(1 for t in timeline if t["total"]),
        "duration_s": max((f["t"] or 0.0) for f in mission["frames"]) if (kind == "video" and n_frames) else None,
        "totals": totals,
        "classes": classes,
        "estimate": estimate,
        "timeline": timeline,
        "hotspots": hotspots,
        "review": {"reviewed": reviewed, "reviewable": reviewable,
                   "pct": round(100.0 * reviewed / reviewable, 1) if reviewable else 100.0},
    }


def pick_hotspots(timeline: List[dict], n_frames: int, k: int = 6) -> List[dict]:
    """Busiest frames, spread across the dive (consecutive video frames look alike)."""
    gap = max(1, n_frames // (k * 2))
    chosen: List[dict] = []
    for t in sorted((t for t in timeline if t["total"]), key=lambda t: (-t["total"], t["index"])):
        if all(abs(t["index"] - c["index"]) >= gap for c in chosen):
            chosen.append(t)
        if len(chosen) == k:
            break
    return sorted(chosen, key=lambda t: t["index"])


def summary(mission: dict) -> dict:
    """Lightweight listing view (no frames)."""
    inv = build_inventory(mission) if mission.get("status") == "ready" else None
    return {
        "id": mission["id"], "name": mission["name"], "site": mission.get("site", ""),
        "created_at": mission["created_at"], "status": mission["status"], "error": mission.get("error"),
        "progress": mission["progress"], "kind": mission["source"]["kind"],
        "frames": len(mission["frames"]),
        "counted": inv["totals"]["counted"] if inv else None,
        "pending": inv["totals"]["pending"] if inv else None,
        "review_pct": inv["review"]["pct"] if inv else None,
    }
