"""Shared fakes for the tests (named uniquely so a site-packages ``tests`` package can't shadow it)."""

from app.mission import frame_record, new_mission


class _Vec:
    """Mimics a 1-row tensor: supports ``t[0].item()`` and ``t[0].tolist()``."""

    def __init__(self, values):
        self._values = values

    def __getitem__(self, idx):
        return self

    def item(self):
        return self._values[0]

    def tolist(self):
        return list(self._values)


class FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = _Vec(xyxy)
        self.cls = _Vec([cls])
        self.conf = _Vec([conf])


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Stand-in for ultralytics.RTDETR that returns canned boxes."""

    def __init__(self, boxes=None):
        self.boxes = boxes if boxes is not None else []
        self.calls = []

    def predict(self, source=None, conf=0.25, verbose=False, **kwargs):
        kept = [b for b in self.boxes if b.conf.item() >= conf]
        if isinstance(source, list):  # batched call: one result per image, like ultralytics
            self.calls.append({"conf": conf, "batch": len(source), "size": getattr(source[0], "size", None) if source else None})
            return [FakeResult(list(kept)) for _ in source]
        self.calls.append({"conf": conf, "size": getattr(source, "size", None)})
        return [FakeResult(kept)]



def det(cls, conf, x=10):
    return {"x1": x, "y1": 10, "x2": x + 40, "y2": 60, "class_id": 0, "class_name": cls, "confidence": conf}


def make_mission(kind="stills", frames=(), conf=0.5, floor=0.25):
    m = new_mission("Pier 4 north", kind, conf=conf, floor=floor, mission_id="m-20261103-010000-abcdef")
    for i, dets in enumerate(frames):
        m["frames"].append(frame_record(i, f"f{i}.jpg", i * 2.0 if kind == "video" else None, (640, 480), dets, f"frames/{i:05d}.jpg"))
    m["status"] = "ready"
    return m
