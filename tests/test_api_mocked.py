"""Fast API tests with the RT-DETR detector replaced by a fake (see conftest.py)."""

import io

import pytest


def post_image(client, path, jpeg_bytes, **kwargs):
    return client.post(path, files={"file": ("x.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")}, **kwargs)


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "AquaGuard" in r.text


@pytest.mark.parametrize("path", ["/health", "/api/v1/health"])
def test_health(client, path):
    r = client.get(path)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["classes"] == ["can_metal", "bottle_plastic", "bottle_glass", "net_plastic", "bag_plastic", "tire_rubber"]
    assert "cuda" in body


def test_telemetry_headers_and_request_id_passthrough(client):
    r = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    assert float(r.headers["X-Process-Time-Ms"]) >= 0
    assert r.headers["X-Compute-Device"] in {"CPU", "CUDA"}


@pytest.mark.parametrize("path", ["/detect", "/api/v1/detect"])
def test_detect_returns_sorted_detections(client, fake_model, make_box, jpeg_bytes, path):
    fake_model.boxes = [
        make_box([1.04, 2.0, 30.0, 40.0], 5, 0.4),
        make_box([5.0, 6.0, 50.0, 60.0], 0, 0.91234),
        make_box([0.0, 0.0, 1.0, 1.0], 99, 0.6),  # unknown class id
    ]
    r = post_image(client, path, jpeg_bytes)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    names = [d["class_name"] for d in body["detections"]]
    assert names == ["can_metal", "99", "tire_rubber"]
    assert body["detections"][0]["confidence"] == 0.912
    assert body["detections"][2]["x1"] == 1.0
    assert fake_model.calls[-1] == {"conf": 0.25, "size": (64, 48)}


def test_detect_passes_conf_to_model(client, fake_model, make_box, jpeg_bytes):
    fake_model.boxes = [make_box([0, 0, 1, 1], 1, 0.5), make_box([0, 0, 1, 1], 2, 0.9)]
    r = post_image(client, "/detect?conf=0.6", jpeg_bytes)
    assert r.json()["count"] == 1
    assert fake_model.calls[-1]["conf"] == 0.6


@pytest.mark.parametrize("conf", ["-0.1", "1.5"])
def test_detect_rejects_out_of_range_conf(client, jpeg_bytes, conf):
    r = post_image(client, f"/detect?conf={conf}", jpeg_bytes)
    assert r.status_code == 422
    assert r.json()["title"] == "Payload Validation Failed"


def test_detect_rejects_non_image_content_type(client):
    r = client.post("/detect", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    body = r.json()
    assert body["status"] == 400 and body["instance"] == "/detect"


def test_detect_rejects_corrupt_image(client):
    r = client.post("/detect", files={"file": ("a.jpg", b"not really a jpeg", "image/jpeg")})
    assert r.status_code == 400
    assert "decoded" in r.json()["detail"]


def test_detect_rejects_oversized_upload(client, monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/detect", files={"file": ("a.jpg", b"x" * 11, "image/jpeg")})
    assert r.status_code == 413


def test_detect_missing_file_is_422(client):
    r = client.post("/detect")
    assert r.status_code == 422


def test_unhandled_model_error_is_rfc7807_500(fake_model, jpeg_bytes):
    from fastapi.testclient import TestClient
    from app.main import app

    def boom(**kwargs):
        raise RuntimeError("cuda exploded")

    fake_model.predict = boom
    c = TestClient(app, raise_server_exceptions=False)
    r = post_image(c, "/detect", jpeg_bytes)
    assert r.status_code == 500
    assert r.json()["title"] == "Internal Server Error"
    assert "cuda exploded" not in r.text


@pytest.mark.parametrize("question,snippet", [
    ("Hello!", "Hello! I am AquaGuard"),
    ("What can you do?", "6 fine-grained"),
    ("Tell me a joke", "does not require visual perception"),
])
def test_reason_non_visual_bypasses_detector(client, fake_model, jpeg_bytes, question, snippet):
    r = post_image(client, "/reason", jpeg_bytes, data={"question": question})
    assert r.status_code == 200
    body = r.json()
    assert body["used_detector"] is False
    assert snippet in body["answer"]
    assert fake_model.calls == []


def test_reason_hi_substring_is_not_a_greeting(client, jpeg_bytes):
    # "sushi" contains "hi" but must not trigger the greeting reply.
    r = post_image(client, "/reason", jpeg_bytes, data={"question": "Do you like sushi?"})
    assert "Hello!" not in r.json()["answer"]


def test_reason_counts_with_detector(client, fake_model, make_box, jpeg_bytes):
    fake_model.boxes = [
        make_box([0, 0, 10, 10], 5, 0.9),
        make_box([20, 20, 30, 30], 5, 0.8),
        make_box([40, 40, 50, 50], 1, 0.25),  # above detector conf, below reasoning conf
    ]
    r = post_image(client, "/api/v1/reason", jpeg_bytes, data={"question": "How many tires are there?"})
    assert r.status_code == 200
    body = r.json()
    assert body["used_detector"] is True
    assert body["count"] == 3
    assert body["insufficient"] is False
    assert body["answer"] == "There are 2 tire_rubber(s) detected."
    assert fake_model.calls[-1]["conf"] == 0.20


def test_reason_guardrail_low_confidence(client, fake_model, make_box, jpeg_bytes):
    fake_model.boxes = [make_box([0, 0, 10, 10], 0, 0.25)]
    r = post_image(client, "/reason", jpeg_bytes, data={"question": "How many cans?"})
    body = r.json()
    assert body["insufficient"] is True
    assert body["reason"].startswith("confidence guardrail")


def test_reason_conf_query_param(client, fake_model, make_box, jpeg_bytes):
    fake_model.boxes = [make_box([0, 0, 10, 10], 0, 0.25)]
    r = post_image(client, "/reason?conf=0.2", jpeg_bytes, data={"question": "How many cans?"})
    assert r.json()["answer"] == "There are 1 can_metal(s) detected."


def test_reason_unanswerable(client, jpeg_bytes):
    r = post_image(client, "/reason", jpeg_bytes, data={"question": "How deep is the water here?"})
    body = r.json()
    assert body["used_detector"] is True
    assert body["insufficient"] is True


def test_reason_blank_question_rejected(client, jpeg_bytes):
    r = post_image(client, "/reason", jpeg_bytes, data={"question": "   "})
    assert r.status_code == 400


def test_reason_rejects_non_image(client):
    r = client.post("/reason", files={"file": ("a.txt", b"hi", "text/plain")}, data={"question": "How many cans?"})
    assert r.status_code == 400


def test_reason_truncates_detections_to_ten(client, fake_model, make_box, jpeg_bytes):
    fake_model.boxes = [make_box([i, i, i + 5, i + 5], 0, 0.5 + i / 100) for i in range(15)]
    r = post_image(client, "/reason", jpeg_bytes, data={"question": "How many cans?"})
    body = r.json()
    assert body["count"] == 15
    assert len(body["detections"]) == 10
    confs = [d["confidence"] for d in body["detections"]]
    assert confs == sorted(confs, reverse=True)


def test_health_survives_broken_cuda(client, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def boom(_):
        raise AssertionError("Invalid device id")

    monkeypatch.setattr(torch.cuda, "get_device_name", boom)
    body = client.get("/health").json()
    assert body["cuda"] is False and body["device"] == "CPU"
