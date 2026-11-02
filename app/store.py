"""File-backed mission storage: ``<root>/<mission_id>/mission.json`` plus ``frames/*.jpg``.

Plain files on purpose: a survey vessel often has no database server, and an
operator can copy a mission folder onto a USB stick and open it elsewhere.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional

from PIL import Image

_ID_RE = re.compile(r"^m-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")


class MissionNotFound(KeyError):
    pass


class MissionStore:
    def __init__(self, root) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # paths --------------------------------------------------------------
    def _dir(self, mission_id: str) -> Path:
        if not _ID_RE.match(mission_id or ""):
            raise MissionNotFound(mission_id)
        return self.root / mission_id

    def frame_path(self, mission_id: str, rel: str) -> Path:
        base = self._dir(mission_id).resolve()
        p = (base / rel).resolve()
        if base not in p.parents:
            raise MissionNotFound(rel)
        return p

    # CRUD ---------------------------------------------------------------
    def save(self, mission: dict) -> None:
        d = self._dir(mission["id"])
        d.mkdir(parents=True, exist_ok=True)
        data = json.dumps(mission, indent=1)
        with self._lock:
            fd, tmp = tempfile.mkstemp(prefix="mission.", suffix=".tmp", dir=d)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(data)
            for attempt in range(20):  # Windows: replace fails while a reader has the file open
                try:
                    os.replace(tmp, d / "mission.json")
                    break
                except PermissionError:
                    if attempt == 19:
                        os.unlink(tmp)
                        raise
                    time.sleep(0.05)

    def load(self, mission_id: str) -> dict:
        p = self._dir(mission_id) / "mission.json"
        if not p.exists():
            raise MissionNotFound(mission_id)
        with self._lock:
            return json.loads(p.read_text(encoding="utf-8"))

    def list(self) -> List[dict]:
        out = []
        with self._lock:
            for p in sorted(self.root.glob("m-*/mission.json"), reverse=True):
                try:
                    out.append(json.loads(p.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    continue
        return out

    def delete(self, mission_id: str) -> None:
        d = self._dir(mission_id)
        if not d.exists():
            raise MissionNotFound(mission_id)
        shutil.rmtree(d)

    # frames -------------------------------------------------------------
    def save_frame(self, mission_id: str, index: int, img: Image.Image, quality: int = 85) -> str:
        rel = f"frames/{index:05d}.jpg"
        p = self._dir(mission_id) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(p, format="JPEG", quality=quality)
        return rel

    def read_frame(self, mission_id: str, rel: str) -> bytes:
        p = self.frame_path(mission_id, rel)
        if not p.exists():
            raise FileNotFoundError(rel)
        return p.read_bytes()

    def save_upload(self, mission_id: str, name: str, data: bytes) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)[:120] or "upload.bin"
        p = self._dir(mission_id) / "uploads" / safe
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def clear_uploads(self, mission_id: str) -> None:
        shutil.rmtree(self._dir(mission_id) / "uploads", ignore_errors=True)

    def open_frame(self, mission_id: str, rel: str) -> Optional[Image.Image]:
        try:
            return Image.open(self.frame_path(mission_id, rel)).convert("RGB")
        except (OSError, MissionNotFound):
            return None
