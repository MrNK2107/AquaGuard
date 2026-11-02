"""Human review: confirm, relabel or reject detections, add missed debris, export corrections.

Decisions are stored in ``mission["reviews"]`` keyed by detection id and never
modify the model's original output, so the export can show "model said X,
reviewer said Y" for every box. The export is what feeds the next retraining run.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Callable, Dict, List, Optional

from .brand import CLASS_NAMES
from .mission import effective_detections, utcnow

ACTIONS = {"confirm", "reject", "relabel", "undo"}


class ReviewError(ValueError):
    """Bad review input (unknown detection, action or class)."""


def _find_detection(mission: dict, det_id: str):
    for frame in mission["frames"]:
        for d in frame["detections"]:
            if d["id"] == det_id:
                return frame, d
    raise ReviewError(f"unknown detection id: {det_id}")


def _frame(mission: dict, index: int) -> dict:
    if not (0 <= index < len(mission["frames"])):
        raise ReviewError(f"unknown frame index: {index}")
    return mission["frames"][index]


def apply_review(mission: dict, det_id: str, action: str, class_name: Optional[str] = None,
                 reviewer: str = "", note: str = "", now: Callable[[], str] = utcnow) -> Optional[dict]:
    """Record one decision. Returns the stored review (``None`` for undo)."""
    if action not in ACTIONS:
        raise ReviewError(f"action must be one of {sorted(ACTIONS)}")
    frame, det = _find_detection(mission, det_id)
    reviews = mission.setdefault("reviews", {})
    if action == "undo":
        reviews.pop(det_id, None)
        frame.pop("cleared", None)
        return None
    if action == "relabel":
        if class_name not in CLASS_NAMES:
            raise ReviewError(f"class_name must be one of {CLASS_NAMES}")
        if class_name == det["class_name"]:
            action = "confirm"  # relabel to the same class is a confirmation
    entry = {"action": action, "reviewer": reviewer.strip()[:80], "at": now()}
    if action == "relabel":
        entry["class_name"] = class_name
    if note:
        entry["note"] = note.strip()[:500]
    reviews[det_id] = entry
    return entry


def confirm_frame(mission: dict, index: int, reviewer: str = "", now: Callable[[], str] = utcnow) -> int:
    """Accept every undecided box in a frame as the model labelled it. Returns how many."""
    frame = _frame(mission, index)
    n = 0
    for d in frame["detections"]:
        if d["id"] not in mission.setdefault("reviews", {}):
            apply_review(mission, d["id"], "confirm", reviewer=reviewer, now=now)
            n += 1
    frame["cleared"] = True
    return n


def add_box(mission: dict, index: int, box: Dict[str, float], class_name: str,
            reviewer: str = "", now: Callable[[], str] = utcnow) -> dict:
    """Add a box for debris the model missed (coordinates in stored-frame pixels)."""
    frame = _frame(mission, index)
    if class_name not in CLASS_NAMES:
        raise ReviewError(f"class_name must be one of {CLASS_NAMES}")
    try:
        x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        raise ReviewError("box needs numeric x1, y1, x2, y2")
    x1, x2 = sorted((max(0.0, min(x1, frame["width"])), max(0.0, min(x2, frame["width"]))))
    y1, y2 = sorted((max(0.0, min(y1, frame["height"])), max(0.0, min(y2, frame["height"]))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ReviewError("box is too small")
    added = frame.setdefault("added", [])
    # ids only ever go up, so removing a box never lets a later one reuse its id
    seq = frame.get("next_added", len(added))
    frame["next_added"] = seq + 1
    entry = {
        "id": f"{index:05d}-a{seq:02d}",
        "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
        "class_id": CLASS_NAMES.index(class_name), "class_name": class_name,
        "confidence": None, "reviewer": reviewer.strip()[:80], "at": now(),
    }
    added.append(entry)
    return entry


def remove_added(mission: dict, box_id: str) -> None:
    for frame in mission["frames"]:
        for i, a in enumerate(frame.get("added", [])):
            if a["id"] == box_id:
                del frame["added"][i]
                return
    raise ReviewError(f"unknown added box: {box_id}")


def frame_fully_reviewed(mission: dict, frame: dict) -> bool:
    """True when a human has looked at every box in the frame.

    A frame with no model boxes counts once a reviewer cleared it or added a box,
    so "nothing here" frames become negative training examples.
    """
    reviews = mission.get("reviews", {})
    dets = frame["detections"]
    if not dets:
        return bool(frame.get("cleared") or frame.get("added"))
    return all(d["id"] in reviews for d in dets)


def correction_records(mission: dict) -> List[dict]:
    """One JSON record per human decision: what the model said and what the reviewer said."""
    out = []
    for frame in mission["frames"]:
        for e in effective_detections(mission, frame):
            if e["status"] in {"auto", "pending"}:
                continue
            out.append({
                "mission_id": mission["id"],
                "frame_index": frame["index"],
                "frame_source": frame["source"],
                "t": frame["t"],
                "image": frame["image"],
                "image_size": [frame["width"], frame["height"]],
                "box_id": e["id"],
                "bbox_xyxy": [e["x1"], e["y1"], e["x2"], e["y2"]],
                "decision": e["status"],
                "model_class": e.get("model_class"),
                "model_confidence": e.get("confidence"),
                "final_class": None if e["status"] == "rejected" else e["class_name"],
                "reviewer": (e.get("review") or {}).get("reviewer") or e.get("reviewer", ""),
                "at": (e.get("review") or {}).get("at") or e.get("at"),
            })
    return out


def yolo_labels(mission: dict) -> Dict[str, str]:
    """YOLO txt labels for fully reviewed frames only (unreviewed frames are not ground truth)."""
    labels = {}
    for frame in mission["frames"]:
        if not frame_fully_reviewed(mission, frame):
            continue
        w, h = float(frame["width"]), float(frame["height"])
        lines = []
        for e in effective_detections(mission, frame):
            if not e["counted"]:
                continue
            cx = (e["x1"] + e["x2"]) / 2 / w
            cy = (e["y1"] + e["y2"]) / 2 / h
            bw = (e["x2"] - e["x1"]) / w
            bh = (e["y2"] - e["y1"]) / h
            lines.append(f"{CLASS_NAMES.index(e['class_name'])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        labels[f"{frame['index']:05d}"] = "\n".join(lines) + ("\n" if lines else "")
    return labels


def build_corrections_zip(mission: dict, read_frame: Optional[Callable[[str], bytes]] = None) -> bytes:
    """Zip with corrections.jsonl, YOLO labels/images for reviewed frames, and data.yaml."""
    buf = io.BytesIO()
    records = correction_records(mission)
    labels = yolo_labels(mission)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("corrections.jsonl", "".join(json.dumps(r) + "\n" for r in records))
        for stem, text in labels.items():
            z.writestr(f"labels/{stem}.txt", text)
            if read_frame is not None:
                frame = mission["frames"][int(stem)]
                try:
                    z.writestr(f"images/{stem}.jpg", read_frame(frame["image"]))
                except FileNotFoundError:
                    pass
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
        z.writestr("data.yaml", f"# AquaGuard review export for mission {mission['id']}\n"
                                f"path: .\ntrain: images\nval: images\nnames:\n{names}\n")
        z.writestr("README.txt",
                   f"AquaGuard corrections export\nmission: {mission['id']} ({mission['name']})\n"
                   f"exported: {utcnow()}\n\n"
                   f"corrections.jsonl  one line per human decision ({len(records)} lines)\n"
                   f"labels/, images/   YOLO labels for the {len(labels)} fully reviewed frames\n"
                   "Frames with unreviewed boxes are left out on purpose: they are not ground truth.\n")
    return buf.getvalue()
