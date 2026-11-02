"""Offline survey report: run AquaGuard on a dive video or a folder of stills, no server needed.

    python scripts/mission_report.py dive.mp4 --name "Harbour wall T3" --site "Pier 4" --every 2
    python scripts/mission_report.py frames/ --name "Lokrum stills" --pdf

Writes a mission folder under --out (default ./missions, the same folder the API
and the console read), so the mission can be reviewed in the console afterwards.
Prints the paths of report.html (and report.pdf with --pdf).

Memory: frames are decoded one at a time and detected in batches of 8; on CPU the
RT-DETR-L model needs about 1.5 GB of RAM in total.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from app.mission import BATCH_SIZE, IMAGE_EXTS, VIDEO_EXTS, build_inventory, new_mission, run_frames, sample_video  # noqa: E402
from app.report import render_html, render_pdf  # noqa: E402
from app.store import MissionStore  # noqa: E402


class DeviceModel:
    """Pins every predict call to one device (CPU by default)."""

    def __init__(self, model, device: str):
        self.model, self.device = model, device

    def predict(self, **kw):
        return self.model.predict(device=self.device, **kw)


def collect(inputs):
    files = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            files += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        elif p.exists():
            files.append(p)
        else:
            raise SystemExit(f"not found: {raw}")
    videos = [f for f in files if f.suffix.lower() in VIDEO_EXTS]
    stills = [f for f in files if f.suffix.lower() in IMAGE_EXTS]
    if videos and (stills or len(videos) > 1):
        raise SystemExit("give one video, or images / folders of images, not both")
    if not files or (not videos and not stills):
        raise SystemExit("no images or video found")
    return ("video", videos) if videos else ("stills", stills)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="one video, or image files / folders")
    ap.add_argument("--name", default="Untitled survey")
    ap.add_argument("--site", default="")
    ap.add_argument("--operator", default="")
    ap.add_argument("--out", default="missions", help="missions folder (shared with the API/console)")
    ap.add_argument("--every", type=float, default=1.0, help="video: seconds between sampled frames")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--conf", type=float, default=0.50, help="auto-count threshold")
    ap.add_argument("--floor", type=float, default=0.25, help="drop detections below this")
    ap.add_argument("--weights", default=str(ROOT / "weights" / "best.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--pdf", action="store_true", help="also write report.pdf")
    a = ap.parse_args(argv)

    kind, files = collect(a.inputs)
    import torch
    torch.set_num_threads(a.threads)
    from ultralytics import RTDETR
    model = DeviceModel(RTDETR(a.weights), a.device)

    store = MissionStore(a.out)
    m = new_mission(a.name, kind, conf=a.conf, floor=a.floor, site=a.site, operator=a.operator,
                    sample_every_s=a.every if kind == "video" else None, model=Path(a.weights).name)
    m["source"]["files"] = [f.name for f in files]
    store.save(m)

    if kind == "video":
        frames = ((files[0].name, t, img) for t, img in sample_video(str(files[0]), a.every, a.max_frames))
    else:
        m["progress"]["total"] = len(files)
        frames = ((f.name, None, Image.open(f).convert("RGB")) for f in files)

    t0 = time.time()
    with torch.no_grad():
        run_frames(m, model, frames, save_frame=lambda i, img: store.save_frame(m["id"], i, img),
                   on_progress=lambda mm: print(f"  {mm['progress']['done']} frames", flush=True), batch_size=a.batch)
    m["progress"]["total"] = len(m["frames"])
    store.save(m)
    dt = time.time() - t0

    loader = lambda fr: store.open_frame(m["id"], fr["image"])  # noqa: E731
    d = store.root / m["id"]
    (d / "report.html").write_text(render_html(m, loader), encoding="utf-8")
    inv = build_inventory(m)
    print(f"mission {m['id']}: {inv['frames']} frames in {dt:.1f} s ({dt / max(1, inv['frames']):.2f} s/frame), "
          f"{inv['totals']['counted']} counted, {inv['totals']['pending']} awaiting review")
    print(d / "report.html")
    if a.pdf:
        (d / "report.pdf").write_bytes(render_pdf(m, loader))
        print(d / "report.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
