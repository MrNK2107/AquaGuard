"""The ASGI upload guard: auth, rate limit and size cap are decided before the body is read.

Based on the review probes (probe_aq.py / probe_aq2.py): previously a client with no key could
make the server parse a whole multipart upload before getting a 401.
"""

import io

import pytest
from fastapi.testclient import TestClient

KEY = "secret-key-123"


@pytest.fixture
def guarded(fake_model, mission_store, monkeypatch):
    import app.main as main
    from app.security import AccessControl

    monkeypatch.setattr(main, "access", AccessControl.from_env({"AQUAGUARD_API_KEYS": KEY,
                                                                "AQUAGUARD_RATE_LIMIT": "60/minute"}))
    return TestClient(main.app)


@pytest.fixture
def parse_spy(monkeypatch):
    """Records every multipart parse, so a test can prove the body was never parsed."""
    import starlette.formparsers as fp

    calls = []
    orig = fp.MultiPartParser.parse

    async def spy(self):
        calls.append(1)
        return await orig(self)

    monkeypatch.setattr(fp.MultiPartParser, "parse", spy)
    return calls


def big_video(mb=3):
    return {"files": ("a.mp4", io.BytesIO(b"\0" * (mb * 1024 * 1024)), "video/mp4")}


def test_upload_without_key_is_rejected_before_parsing(guarded, parse_spy):
    r = guarded.post("/api/v1/missions", files=big_video())
    assert r.status_code == 401
    assert r.json()["status"] == 401 and r.json()["instance"] == "/api/v1/missions"
    assert "WWW-Authenticate" in r.headers
    assert parse_spy == []


@pytest.mark.parametrize("path", ["/detect", "/api/v1/detect", "/reason", "/api/v1/reason"])
def test_single_frame_routes_reject_before_parsing(guarded, parse_spy, path):
    r = guarded.post(path, files={"file": ("a.jpg", io.BytesIO(b"x" * 1024), "image/jpeg")},
                     headers={"X-API-Key": "wrong"})
    assert r.status_code == 401
    assert parse_spy == []


def test_malformed_multipart_without_key_is_401_not_400(guarded):
    r = guarded.post("/api/v1/missions", content=b"--x\r\ngarbage",
                     headers={"content-type": "multipart/form-data; boundary=x"})
    assert r.status_code == 401


def test_rate_limit_is_charged_before_parsing(fake_model, mission_store, monkeypatch, parse_spy):
    import app.main as main
    from app.security import AccessControl

    # missions cost 10 tokens; a 10/minute bucket allows exactly one upload
    monkeypatch.setattr(main, "access", AccessControl.from_env({"AQUAGUARD_API_KEYS": KEY,
                                                                "AQUAGUARD_RATE_LIMIT": "10/minute"}))
    c = TestClient(main.app)
    first = c.post("/api/v1/missions", files={"files": ("a.txt", io.BytesIO(b"x"), "text/plain")},
                   headers={"X-API-Key": KEY})
    assert first.status_code == 400  # authorised and charged, then rejected for the file type
    n = len(parse_spy)
    second = c.post("/api/v1/missions", files=big_video(), headers={"X-API-Key": KEY})
    assert second.status_code == 429 and "Retry-After" in second.headers
    assert len(parse_spy) == n  # second body never parsed


def test_content_length_over_cap_is_413_before_parsing(guarded, parse_spy, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "MAX_REQUEST_BYTES", 1024 * 1024)
    r = guarded.post("/api/v1/missions", files=big_video(2), headers={"X-API-Key": KEY})
    assert r.status_code == 413
    assert r.json()["status"] == 413
    assert parse_spy == []


def test_single_frame_routes_have_a_smaller_cap(guarded, parse_spy):
    import app.main as main

    body = b"x" * (main.MAX_UPLOAD_BYTES + 2 * 1024 * 1024)
    r = guarded.post("/detect", files={"file": ("a.jpg", io.BytesIO(body), "image/jpeg")},
                     headers={"X-API-Key": KEY})
    assert r.status_code == 413 and parse_spy == []


def test_chunked_body_over_cap_is_413(guarded, monkeypatch):
    """No Content-Length (chunked transfer): the receive wrapper counts bytes and stops at the cap."""
    import app.main as main

    monkeypatch.setattr(main, "MAX_REQUEST_BYTES", 256 * 1024)
    boundary = "bnd"
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"a.mp4\"\r\n"
            "Content-Type: video/mp4\r\n\r\n").encode()

    def chunks():
        yield head
        for _ in range(16):
            yield b"\0" * (64 * 1024)
        yield f"\r\n--{boundary}--\r\n".encode()

    r = guarded.post("/api/v1/missions", content=chunks(),
                     headers={"content-type": f"multipart/form-data; boundary={boundary}", "X-API-Key": KEY})
    assert r.status_code == 413
    assert mission_count(main) == 0


def mission_count(main):
    import app.missions_api as mapi
    return len(mapi.store.list())


def test_guard_parse_max_upload_env():
    from app.main import _max_request_bytes

    assert _max_request_bytes({"AQUAGUARD_MAX_UPLOAD_MB": "20"}) == 20 * 1024 * 1024
    assert _max_request_bytes({}) == 512 * 1024 * 1024
    with pytest.raises(ValueError):
        _max_request_bytes({"AQUAGUARD_MAX_UPLOAD_MB": "0"})


def test_valid_key_still_works_end_to_end(guarded, jpeg_bytes):
    r = guarded.post("/detect", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")}, headers={"X-API-Key": KEY})
    assert r.status_code == 200
    assert r.headers["X-RateLimit-Limit"] == "60"
    # charged once (by the guard), not twice (guard + route dependency)
    assert r.headers["X-RateLimit-Remaining"] == "59"


def test_get_routes_still_authenticated(guarded):
    assert guarded.get("/api/v1/missions").status_code == 401
    assert guarded.get("/api/v1/missions", headers={"X-API-Key": KEY}).status_code == 200
    assert guarded.get("/health").status_code == 200
