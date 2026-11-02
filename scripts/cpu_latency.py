"""
CPU latency of the detector, measured the same way /detect calls it.

Loads weights/best.pt with Ultralytics RTDETR on device="cpu" and times
model.predict(source=<PIL image>, conf=0.25) on a few images that ship with the
repo. Nothing is downloaded. Prints a Markdown table (per-image median and p95
of wall time, plus Ultralytics' preprocess/inference/postprocess split).

Usage:
    python scripts/cpu_latency.py                       # bundled memo/assets images
    python scripts/cpu_latency.py img1.jpg img2.jpg     # your own images
    python scripts/cpu_latency.py --runs 20 --threads 4
"""

import argparse
import platform
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES = [
    ROOT / "memo/assets/val_batch0_labels.jpg",
    ROOT / "memo/assets/val_batch1_labels.jpg",
    ROOT / "memo/assets/val_batch2_labels.jpg",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--weights", type=Path, default=ROOT / "weights/best.pt")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from ultralytics import RTDETR

    torch.set_num_threads(args.threads)
    images = args.images or DEFAULT_IMAGES
    model = RTDETR(str(args.weights))

    print(f"CPU: {platform.processor()} | torch {torch.__version__} | threads={args.threads} | runs={args.runs}\n")
    print("| Image | Size | Wall median (ms) | Wall p95 (ms) | Pre / Infer / Post median (ms) |")
    print("|---|---|---|---|---|")
    for path in images:
        img = Image.open(path).convert("RGB")
        for _ in range(args.warmup):
            model.predict(source=img, conf=args.conf, device="cpu", verbose=False)
        walls, pre, inf, post = [], [], [], []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            res = model.predict(source=img, conf=args.conf, device="cpu", verbose=False)
            walls.append((time.perf_counter() - t0) * 1000)
            sp = res[0].speed
            pre.append(sp["preprocess"])
            inf.append(sp["inference"])
            post.append(sp["postprocess"])
        walls.sort()
        p95 = walls[min(len(walls) - 1, int(round(0.95 * (len(walls) - 1))))]
        print(
            f"| {path.name} | {img.width}x{img.height} | {statistics.median(walls):.0f} | {p95:.0f} | "
            f"{statistics.median(pre):.1f} / {statistics.median(inf):.0f} / {statistics.median(post):.1f} |"
        )


if __name__ == "__main__":
    main()
