"""Decoding untrusted uploads: pixel caps, a format allowlist, and video probing.

Configuration (environment variables, read at import):

  AQUAGUARD_MAX_IMAGE_MP      still-image cap in megapixels (default 50).
  AQUAGUARD_MAX_VIDEO_PIXELS  per-frame video cap, width*height (default 4096*4096).
  AQUAGUARD_MAX_VIDEO_FRAMES  total frames a video may declare (default 500000, ~4.6 h at 30 fps).

Only JPEG, PNG, WebP and BMP stills are decoded (no TIFF: its many codecs are attack
surface the survey never needs). Videos are probed with ``ffprobe`` when it is installed
(container + codec allowlist, size and frame caps) before OpenCV opens them; without it the
same caps are checked from OpenCV's own properties right after open. OpenCV's FFmpeg backend
is restricted to the ``file`` protocol, so a crafted container can't make it fetch URLs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import warnings
from typing import Optional

from PIL import Image

# must be set before cv2 opens anything; OpenCV reads it on every VideoCapture open
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file"

MAX_IMAGE_PIXELS = int(float(os.environ.get("AQUAGUARD_MAX_IMAGE_MP", "50")) * 1_000_000)
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
warnings.simplefilter("error", Image.DecompressionBombWarning)

ALLOWED_IMAGE_FORMATS = ("JPEG", "PNG", "WEBP", "BMP")
MAX_VIDEO_PIXELS = int(os.environ.get("AQUAGUARD_MAX_VIDEO_PIXELS", str(4096 * 4096)))
MAX_VIDEO_FRAMES = int(os.environ.get("AQUAGUARD_MAX_VIDEO_FRAMES", "500000"))
VIDEO_CONTAINERS = {"mov", "mp4", "m4a", "3gp", "3g2", "mj2", "matroska", "webm", "avi"}
VIDEO_CODECS = {"h264", "hevc", "mpeg4", "vp8", "vp9", "av1", "mjpeg"}


class UnsupportedMedia(ValueError):
    """The upload is a format we don't decode, or exceeds a size cap."""


_OTHER_MAGIC = (b"II*\x00", b"MM\x00*", b"GIF87a", b"GIF89a", b"\x00\x00\x01\x00", b"P1", b"P2", b"P3", b"P4",
                b"P5", b"P6", b"8BPS", b"\x00\x00\x00\x0cjP")


def _known_other_format(fp) -> bool:
    """Sniff magic bytes only (no decoder runs) to tell 'wrong format' from 'garbage'."""
    try:
        if isinstance(fp, (str, os.PathLike)):
            with open(fp, "rb") as fh:
                head = fh.read(16)
        else:
            fp.seek(0)
            head = fp.read(16)
    except (OSError, AttributeError):
        return False
    return head.startswith(_OTHER_MAGIC)


def open_image(fp) -> Image.Image:
    """Open an untrusted still: allowlisted formats only, and any image over the pixel cap is an error
    (Pillow itself only warns between 1x and 2x ``MAX_IMAGE_PIXELS``)."""
    try:
        im = Image.open(fp, formats=ALLOWED_IMAGE_FORMATS)
    except Image.UnidentifiedImageError:
        if _known_other_format(fp):  # a real image, just not one we decode
            raise UnsupportedMedia("image format not accepted (use JPEG, PNG, WebP or BMP)")
        raise
    if im.width * im.height > MAX_IMAGE_PIXELS:
        im.close()
        raise Image.DecompressionBombError(
            f"image is {im.width}x{im.height}; the limit is {MAX_IMAGE_PIXELS // 1_000_000} megapixels")
    return im


def check_video_caps(width: int, height: int, frames: int) -> None:
    if width * height > MAX_VIDEO_PIXELS:
        raise UnsupportedMedia(f"video resolution {width}x{height} exceeds the limit")
    if frames > MAX_VIDEO_FRAMES:
        raise UnsupportedMedia(f"video declares {frames} frames; the limit is {MAX_VIDEO_FRAMES}")


def probe_video(path: str) -> Optional[dict]:
    """Inspect a video with ffprobe before decoding. Returns None when ffprobe isn't installed."""
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    cmd = [exe, "-v", "error", "-protocol_whitelist", "file", "-show_entries",
           "format=format_name:stream=codec_type,codec_name,width,height,nb_frames",
           "-of", "json", "file:" + os.path.abspath(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        raise UnsupportedMedia("video could not be probed")
    if out.returncode != 0:
        raise UnsupportedMedia("video could not be probed")
    try:
        info = json.loads(out.stdout or "{}")
    except ValueError:
        raise UnsupportedMedia("video could not be probed")
    fmt = (info.get("format") or {}).get("format_name", "")
    if not fmt or not set(fmt.split(",")) <= VIDEO_CONTAINERS:
        raise UnsupportedMedia(f"video container {fmt!r} not accepted")
    video = [s for s in info.get("streams") or [] if s.get("codec_type") == "video"]
    if not video:
        raise UnsupportedMedia("no video stream found")
    s = video[0]
    if s.get("codec_name") not in VIDEO_CODECS:
        raise UnsupportedMedia(f"video codec {s.get('codec_name')!r} not accepted")
    w, h = int(s.get("width") or 0), int(s.get("height") or 0)
    try:
        n = int(s.get("nb_frames") or 0)
    except ValueError:
        n = 0
    if w * h > MAX_VIDEO_PIXELS:
        raise UnsupportedMedia(f"video resolution {w}x{h} exceeds the limit")
    if n > MAX_VIDEO_FRAMES:
        raise UnsupportedMedia(f"video declares {n} frames; the limit is {MAX_VIDEO_FRAMES}")
    return {"format": fmt, "codec": s.get("codec_name"), "width": w, "height": h, "frames": n}
