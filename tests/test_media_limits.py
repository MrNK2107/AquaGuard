"""Decoding untrusted media: pixel caps, format allowlists, video probing, OpenCV protocol whitelist."""

import io
import json
import os
import subprocess
import zipfile

import pytest
from PIL import Image

from app import media


def encode(fmt, size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buf, format=fmt)
    return buf.getvalue()


def test_pillow_pixel_cap_is_set():
    assert Image.MAX_IMAGE_PIXELS == media.MAX_IMAGE_PIXELS
    assert media.MAX_IMAGE_PIXELS == 50_000_000


def test_image_over_cap_but_under_pillow_2x_is_an_error(monkeypatch):
    """Pillow only warns between 1x and 2x MAX_IMAGE_PIXELS; we refuse instead."""
    monkeypatch.setattr(media, "MAX_IMAGE_PIXELS", 64 * 48 - 1)
    with pytest.raises(Image.DecompressionBombError):
        media.open_image(io.BytesIO(encode("PNG")))


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "BMP"])
def test_allowed_formats_decode(fmt):
    with media.open_image(io.BytesIO(encode(fmt))) as im:
        assert im.size == (64, 48)


@pytest.mark.parametrize("fmt", ["TIFF", "GIF", "ICO", "PPM"])
def test_other_formats_are_rejected(fmt):
    with pytest.raises(media.UnsupportedMedia):
        media.open_image(io.BytesIO(encode(fmt)))


def test_tiff_is_not_an_accepted_upload_extension():
    from app.mission import IMAGE_EXTS

    assert ".tif" not in IMAGE_EXTS and ".tiff" not in IMAGE_EXTS


def test_detect_rejects_tiff_and_bombs(client, monkeypatch):
    r = client.post("/detect", files={"file": ("a.jpg", encode("TIFF"), "image/jpeg")})
    assert r.status_code == 415
    monkeypatch.setattr(media, "MAX_IMAGE_PIXELS", 100)
    r = client.post("/detect", files={"file": ("a.jpg", encode("JPEG"), "image/jpeg")})
    assert r.status_code == 413


def test_mission_skips_tiff_disguised_as_jpg(client, fake_model):
    files = [("files", ("ok.jpg", io.BytesIO(encode("JPEG")), "image/jpeg")),
             ("files", ("sneaky.jpg", io.BytesIO(encode("TIFF")), "image/jpeg"))]
    mid = client.post("/api/v1/missions", files=files).json()["id"]
    m = client.get(f"/api/v1/missions/{mid}").json()
    assert m["status"] == "ready" and m["frames"] == 1


def test_opencv_protocol_whitelist_is_file_only():
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "protocol_whitelist;file"


# --- video probing --------------------------------------------------------------------
def fake_ffprobe(monkeypatch, payload, rc=0):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, rc, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(media.shutil, "which", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr(media.subprocess, "run", run)
    return calls


def good_probe(**over):
    stream = {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080, "nb_frames": "900"}
    stream.update(over.pop("stream", {}))
    out = {"format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"}, "streams": [stream]}
    out.update(over)
    return out


def test_ffprobe_accepts_a_normal_dive_video(monkeypatch, tmp_path):
    calls = fake_ffprobe(monkeypatch, good_probe())
    info = media.probe_video(str(tmp_path / "dive.mp4"))
    assert info == {"format": "mov,mp4,m4a,3gp,3g2,mj2", "codec": "h264", "width": 1920, "height": 1080,
                    "frames": 900}
    cmd = calls[0]
    assert "-protocol_whitelist" in cmd and cmd[cmd.index("-protocol_whitelist") + 1] == "file"
    assert cmd[-1].startswith("file:")


@pytest.mark.parametrize("payload,msg", [
    (good_probe(format={"format_name": "hls"}), "container"),
    (good_probe(stream={"codec_name": "prores"}), "codec"),
    (good_probe(stream={"width": 20000, "height": 20000}), "resolution"),
    (good_probe(stream={"nb_frames": "99999999"}), "frames"),
    ({"format": {"format_name": "mp4"}, "streams": [{"codec_type": "audio", "codec_name": "aac"}]}, "video stream"),
])
def test_ffprobe_rejects_unexpected_media(monkeypatch, tmp_path, payload, msg):
    fake_ffprobe(monkeypatch, payload)
    with pytest.raises(media.UnsupportedMedia, match=msg):
        media.probe_video(str(tmp_path / "dive.mp4"))


def test_ffprobe_failure_is_a_rejection(monkeypatch, tmp_path):
    fake_ffprobe(monkeypatch, {}, rc=1)
    with pytest.raises(media.UnsupportedMedia):
        media.probe_video(str(tmp_path / "dive.mp4"))


def test_no_ffprobe_means_no_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    assert media.probe_video(str(tmp_path / "dive.mp4")) is None


class FakeCap:
    def __init__(self, w, h, n):
        import cv2
        self.props = {cv2.CAP_PROP_FRAME_WIDTH: w, cv2.CAP_PROP_FRAME_HEIGHT: h, cv2.CAP_PROP_FRAME_COUNT: n,
                      cv2.CAP_PROP_FPS: 25.0}
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return self.props.get(prop, 0)

    def set(self, *a):
        return True

    def read(self):
        import numpy as np
        return True, np.zeros((4, 4, 3), dtype="uint8")

    def release(self):
        self.released = True


@pytest.mark.parametrize("w,h,n", [(20000, 20000, 10), (640, 480, 99_999_999)])
def test_sample_video_checks_caps_after_open(monkeypatch, tmp_path, w, h, n):
    cv2 = pytest.importorskip("cv2")
    from app import mission

    cap = FakeCap(w, h, n)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: cap)
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    with pytest.raises(media.UnsupportedMedia):
        list(mission.sample_video(str(tmp_path / "x.mp4")))
    assert cap.released


def test_sample_video_runs_probe_before_opening(monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    from app import mission

    fake_ffprobe(monkeypatch, good_probe(format={"format_name": "hls"}))
    opened = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: opened.append(a) or FakeCap(64, 48, 10))
    with pytest.raises(media.UnsupportedMedia):
        list(mission.sample_video(str(tmp_path / "x.mp4")))
    assert opened == []


# --- corrections.zip is streamed from a temp file ----------------------------------------
def test_corrections_zip_is_a_file_response(client, fake_model, make_box, monkeypatch):
    import app.missions_api as mapi

    fake_model.boxes = [make_box([10, 10, 80, 90], 5, 0.91)]
    files = [("files", ("a.jpg", io.BytesIO(encode("JPEG", (320, 240))), "image/jpeg"))]
    mid = client.post("/api/v1/missions", files=files).json()["id"]
    made = []
    real = mapi.tempfile.NamedTemporaryFile

    def spy(*a, **k):
        f = real(*a, **k)
        made.append(f.name)
        return f

    monkeypatch.setattr(mapi.tempfile, "NamedTemporaryFile", spy)
    r = client.get(f"/api/v1/missions/{mid}/corrections.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert "attachment" in r.headers["content-disposition"]
    assert "corrections.jsonl" in zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert len(made) == 1 and not os.path.exists(made[0])  # temp file removed after sending
