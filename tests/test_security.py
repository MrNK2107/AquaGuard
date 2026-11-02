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


# --- review 2026-11-03: LRU buckets, trusted proxies, key-keyed limits, digests -------------
from app.security import client_ip, parse_trusted_proxies  # noqa: E402


def test_bucket_eviction_is_lru_not_oldest_insert():
    clock = Clock()
    rl = RateLimiter(5, 60.0, clock=clock, max_buckets=3)
    for ident in ("a", "b", "c"):
        rl.hit(ident)
    rl.hit("a")            # a is now most recently used
    rl.hit("d")            # evicts exactly one: the least recently used (b)
    assert list(rl._buckets) == ["c", "a", "d"]
    assert len(rl._buckets) == 3


def test_evicting_a_drained_bucket_does_not_happen_while_it_is_hot():
    clock = Clock()
    rl = RateLimiter(1, 3600.0, clock=clock, max_buckets=2)
    assert rl.hit("attacker").allowed and not rl.hit("attacker").allowed
    for i in range(10):    # churn other identities, but the attacker keeps hammering
        rl.hit(f"x{i}")
        assert not rl.hit("attacker").allowed


def test_parse_trusted_proxies():
    nets = parse_trusted_proxies(" 10.0.0.0/8, 172.18.0.5 ,::1")
    assert [str(n) for n in nets] == ["10.0.0.0/8", "172.18.0.5/32", "::1/128"]
    assert parse_trusted_proxies(None) == [] and parse_trusted_proxies("") == []
    with pytest.raises(ValueError):
        parse_trusted_proxies("not-an-ip")


@pytest.mark.parametrize("peer,xff,expected", [
    ("203.0.113.9", "1.2.3.4", "203.0.113.9"),                 # untrusted peer: header ignored
    ("172.18.0.5", None, "172.18.0.5"),                        # trusted peer, no header
    ("172.18.0.5", "198.51.100.7", "198.51.100.7"),            # trusted proxy forwards the client
    ("172.18.0.5", "6.6.6.6, 198.51.100.7", "198.51.100.7"),   # spoofed left hops are ignored
    ("172.18.0.5", "198.51.100.7, 10.1.2.3", "198.51.100.7"),  # skip trusted hops from the right
    ("172.18.0.5", "garbage, 198.51.100.7", "198.51.100.7"),
    ("172.18.0.5", "198.51.100.7, garbage", "172.18.0.5"),     # unparsable rightmost hop: fall back to peer
    ("172.18.0.5", "10.0.0.1, 10.0.0.2", "10.0.0.1"),          # every hop trusted: leftmost
])
def test_client_ip_rightmost_untrusted(peer, xff, expected):
    trusted = parse_trusted_proxies("172.18.0.0/16,10.0.0.0/8")
    assert client_ip(peer, xff, trusted) == expected


GONE = "/api/v1/missions/m-00000000-000000-000000"  # DELETE costs 1 token and answers 404


def _limited(main, monkeypatch, env):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, "access", AccessControl.from_env(env))
    return TestClient


def test_spoofed_xff_from_untrusted_peer_shares_one_bucket(client, monkeypatch):
    """probe_aq.py: rotating X-Forwarded-For must not give an attacker a fresh failed-auth bucket each time."""
    import app.main as main

    TC = _limited(main, monkeypatch, {"AQUAGUARD_API_KEYS": "k", "AQUAGUARD_RATE_LIMIT": "5/minute"})
    c = TC(main.app)
    codes = [c.get("/api/v1/missions", headers={"X-Forwarded-For": f"10.0.0.{i}", "X-API-Key": "bad"}).status_code
             for i in range(8)]
    assert codes == [401] * 5 + [429] * 3


def test_trusted_proxy_gives_each_real_client_its_own_bucket(client, monkeypatch):
    """probe_aq2.py: behind a proxy at 172.18.0.5 all clients used to share the proxy's bucket."""
    import app.main as main

    TC = _limited(main, monkeypatch, {"AQUAGUARD_RATE_LIMIT": "1/minute", "AQUAGUARD_TRUSTED_PROXIES": "172.18.0.0/16"})
    c = TC(main.app, client=("172.18.0.5", 5000))
    codes = [c.delete(GONE, headers={"X-Forwarded-For": f"203.0.113.{i}"}).status_code for i in range(3)]
    assert codes == [404, 404, 404]
    assert c.delete(GONE, headers={"X-Forwarded-For": "203.0.113.0"}).status_code == 429
    # a client behind the trusted proxy can't spoof its way out by prepending hops
    assert c.delete(GONE, headers={"X-Forwarded-For": "1.1.1.1, 203.0.113.0"}).status_code == 429


def test_untrusted_peer_cannot_use_xff_even_when_proxies_configured(client, monkeypatch):
    import app.main as main

    TC = _limited(main, monkeypatch, {"AQUAGUARD_RATE_LIMIT": "1/minute", "AQUAGUARD_TRUSTED_PROXIES": "172.18.0.0/16"})
    c = TC(main.app, client=("203.0.113.50", 5000))
    assert c.delete(GONE, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 404
    assert c.delete(GONE, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429


def test_with_auth_on_the_limit_follows_the_key_not_the_ip(client, monkeypatch):
    import app.main as main

    TC = _limited(main, monkeypatch, {"AQUAGUARD_API_KEYS": "k1,k2", "AQUAGUARD_RATE_LIMIT": "1/minute"})
    a = TC(main.app, client=("198.51.100.1", 1))
    b = TC(main.app, client=("198.51.100.2", 1))
    assert a.delete(GONE, headers={"X-API-Key": "k1"}).status_code == 404
    # same key from another IP: same bucket
    assert b.delete(GONE, headers={"X-API-Key": "k1"}).status_code == 429
    # another key from the same IP: its own bucket
    assert a.delete(GONE, headers={"X-API-Key": "k2"}).status_code == 404


def test_keys_are_compared_as_sha256_digests():
    import hashlib

    ac = AccessControl(["short", "a-much-longer-key"], None)
    assert all(len(d) == 32 for d in ac._digests)
    assert hashlib.sha256(b"short").digest() in ac._digests
    assert ac.check_key("short") and ac.check_key("a-much-longer-key")
    assert not ac.check_key("shor") and not ac.check_key("short ") and not ac.check_key("")
    assert not hasattr(ac, "_keys")  # raw keys are not kept around
