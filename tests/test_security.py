"""API keys and rate limiting: unit tests plus the behaviour through the real routes."""

import io

import pytest

from app.security import AccessControl, RateLimiter, fingerprint, parse_keys, parse_rate


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.mark.parametrize("spec,expected", [(None, (60, 60.0)), ("10/second", (10, 1.0)), ("100/h", (100, 3600.0)),
                                           ("off", None), ("0", None), ("", None), ("0/minute", None)])
def test_parse_rate(spec, expected):
    assert parse_rate(spec) == expected


@pytest.mark.parametrize("spec", ["ten/minute", "5/fortnight", "5"])
def test_parse_rate_rejects_garbage(spec):
    with pytest.raises(ValueError):
        parse_rate(spec)


def test_parse_keys_and_fingerprint():
    assert parse_keys(" a, ,b ,") == ["a", "b"]
    assert parse_keys(None) == []
    assert len(fingerprint("secret")) == 12 and fingerprint("secret") != "secret"


def test_token_bucket_refills_evenly():
    clock = Clock()
    rl = RateLimiter(3, 60.0, clock=clock)
    assert [rl.hit("a").allowed for _ in range(4)] == [True, True, True, False]
    d = rl.hit("a")
    assert not d.allowed and d.retry_after == pytest.approx(20.0)
    assert rl.hit("b").allowed  # buckets are per identity
    clock.t += 20.0
    assert rl.hit("a").allowed
    assert not rl.hit("a").allowed


def test_token_bucket_cost():
    rl = RateLimiter(10, 10.0, clock=Clock())
    assert rl.hit("a", cost=10).allowed
    assert not rl.hit("a", cost=0.5).allowed


def test_bucket_table_is_bounded():
    clock = Clock()
    rl = RateLimiter(1, 1.0, clock=clock, max_buckets=10)
    for i in range(30):
        clock.t += 1
        rl.hit(f"ip{i}")
    assert len(rl._buckets) <= 11


def test_access_control_keys():
    ac = AccessControl(["k1", "k2"], None)
    assert ac.enabled and ac.check_key("k2") and not ac.check_key("k3") and not ac.check_key(None)
    assert AccessControl([], None).check_key(None)
    assert AccessControl.extract_key({"x-api-key": " k1 "}) == "k1"
    assert AccessControl.extract_key({"authorization": "Bearer k2"}) == "k2"
    assert AccessControl.extract_key({"authorization": "Basic abc"}) is None


def test_from_env():
    ac = AccessControl.from_env({"AQUAGUARD_API_KEYS": "x,y", "AQUAGUARD_RATE_LIMIT": "5/minute"})
    assert ac.enabled and ac.limiter.capacity == 5
    assert AccessControl.from_env({"AQUAGUARD_RATE_LIMIT": "off"}).limiter is None


# --- through the API ----------------------------------------------------------
def _post(client, jpeg_bytes, headers=None):
    return client.post("/detect", files={"file": ("x.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")}, headers=headers or {})


@pytest.fixture
def secured(monkeypatch, client):
    import app.main as main

    clock = Clock()
    ac = AccessControl(["deck-key"], RateLimiter(2, 60.0, clock=clock))
    monkeypatch.setattr(main, "access", ac)
    return clock


def test_missing_or_wrong_key_is_401(client, secured, jpeg_bytes):
    r = _post(client, jpeg_bytes)
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"].startswith("Bearer")
    assert r.json()["status"] == 401 and "X-API-Key" in r.json()["detail"]
    assert _post(client, jpeg_bytes, {"X-API-Key": "nope"}).status_code == 401


def test_valid_key_and_rate_limit_headers(client, secured, jpeg_bytes):
    r = _post(client, jpeg_bytes, {"X-API-Key": "deck-key"})
    assert r.status_code == 200
    assert r.headers["X-RateLimit-Limit"] == "2" and r.headers["X-RateLimit-Remaining"] == "1"
    assert "X-AquaGuard-Auth" not in r.headers
    assert _post(client, jpeg_bytes, {"Authorization": "Bearer deck-key"}).status_code == 200
    r = _post(client, jpeg_bytes, {"X-API-Key": "deck-key"})
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) == 30
    assert r.json()["title"] == "HTTP Exception"
    secured.t += 30
    assert _post(client, jpeg_bytes, {"X-API-Key": "deck-key"}).status_code == 200


def test_health_and_console_stay_public(client, secured):
    assert client.get("/health").status_code == 200
    assert client.get("/console").status_code == 200
    assert client.get("/console/meta").json()["auth_required"] is True
    assert client.get("/api/v1/missions").status_code == 401


def test_open_mode_is_labelled(client, monkeypatch, jpeg_bytes):
    import app.main as main

    monkeypatch.setattr(main, "access", AccessControl([], RateLimiter(100, 60.0)))
    r = _post(client, jpeg_bytes)
    assert r.status_code == 200
    assert r.headers["X-AquaGuard-Auth"] == "disabled"
    assert r.headers["X-RateLimit-Limit"] == "100"


def test_cost_above_capacity_is_clamped_not_blocked_forever():
    clock = Clock()
    rl = RateLimiter(5, 60.0, clock=clock)
    assert rl.hit("a", cost=10).allowed  # costs the whole bucket
    d = rl.hit("a", cost=10)
    assert not d.allowed and d.retry_after == pytest.approx(60.0)
    clock.t += 60
    assert rl.hit("a", cost=10).allowed


def test_failed_auth_is_throttled_per_ip(client, secured, jpeg_bytes):
    assert _post(client, jpeg_bytes, {"X-API-Key": "guess1"}).status_code == 401
    assert _post(client, jpeg_bytes, {"X-API-Key": "guess2"}).status_code == 401
    r = _post(client, jpeg_bytes, {"X-API-Key": "guess3"})
    assert r.status_code == 429 and "Retry-After" in r.headers
