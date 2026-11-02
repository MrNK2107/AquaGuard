"""API keys and per-client rate limiting for the AquaGuard API.

Configuration (environment variables, read at startup):

  AQUAGUARD_API_KEYS    comma-separated keys. Unset or empty = auth disabled
                        (local development); responses then carry
                        ``X-AquaGuard-Auth: disabled`` so nobody mistakes it for a
                        secured deployment.
  AQUAGUARD_RATE_LIMIT  "<requests>/<second|minute|hour>", default "60/minute".
                        "0" or "off" disables limiting.
  AQUAGUARD_TRUSTED_PROXIES  comma-separated IPs/CIDRs of reverse proxies in front of
                        the API (e.g. "172.18.0.0/16"). Only when the TCP peer is one of
                        them is ``X-Forwarded-For`` used, and then only its rightmost hop
                        that is not itself a trusted proxy. Unset = the header is ignored.

With auth on, the rate limit is keyed by API key (a key has the same budget from any
IP); failed authentication attempts are charged to the client IP.

Clients send the key as ``X-API-Key: <key>`` or ``Authorization: Bearer <key>``.
Keys are kept only as SHA-256 digests, compared in constant time, and never logged;
logs and limiter buckets use a short SHA-256 fingerprint instead.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

IPNetwork = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]

_PERIODS = {"s": 1, "sec": 1, "second": 1, "m": 60, "min": 60, "minute": 60, "h": 3600, "hour": 3600}


def parse_rate(spec: Optional[str]) -> Optional[Tuple[int, float]]:
    """"60/minute" -> (60, 60.0). Returns None when limiting is disabled."""
    if spec is None:
        return (60, 60.0)
    spec = spec.strip().lower()
    if spec in {"", "0", "off", "none", "disabled"}:
        return None
    try:
        n, per = spec.split("/", 1)
        count = int(n)
        period = _PERIODS[per.strip()]
    except (ValueError, KeyError):
        raise ValueError(f"bad rate limit {spec!r}; use e.g. '60/minute'")
    if count <= 0:
        return None
    return count, float(period)


def fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def parse_keys(raw: Optional[str]) -> List[str]:
    return [k.strip() for k in (raw or "").split(",") if k.strip()]


def parse_trusted_proxies(raw: Optional[str]) -> List[IPNetwork]:
    """"10.0.0.0/8, 172.18.0.5" -> networks. Raises ValueError on anything unparsable."""
    nets = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            nets.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            raise ValueError(f"bad AQUAGUARD_TRUSTED_PROXIES entry {part!r}; use IPs or CIDRs")
    return nets


def _parse_ip(value: str):
    value = value.strip()
    if value.startswith("[") and "]" in value:  # [v6]:port
        value = value[1:value.index("]")]
    elif value.count(":") == 1:  # v4:port
        value = value.split(":", 1)[0]
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def _is_trusted(ip, trusted: Sequence[IPNetwork]) -> bool:
    return ip is not None and any(ip.version == n.version and ip in n for n in trusted)


def client_ip(peer: Optional[str], forwarded_for: Optional[str], trusted: Sequence[IPNetwork]) -> str:
    """The address to rate-limit on.

    The TCP peer, unless the peer is a trusted proxy: then walk ``X-Forwarded-For`` from the
    right, skipping trusted proxies, and take the first hop that is not one. Hops left of it
    were written by the client and can be anything, so they are never used.
    """
    peer = peer or "unknown"
    if not trusted or not forwarded_for or not _is_trusted(_parse_ip(peer), trusted):
        return peer
    hops = [h.strip() for h in forwarded_for.split(",") if h.strip()]
    for hop in reversed(hops):
        ip = _parse_ip(hop)
        if ip is None:
            return peer  # garbage where a trusted hop should have written an address: don't guess
        if not _is_trusted(ip, trusted):
            return str(ip)
    first = _parse_ip(hops[0]) if hops else None
    return str(first) if first is not None else peer


@dataclass
class Decision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: float


class RateLimiter:
    """Token bucket per identity: ``capacity`` requests, refilled evenly over ``period`` seconds."""

    def __init__(self, capacity: int, period: float, clock: Callable[[], float] = time.monotonic,
                 max_buckets: int = 10_000) -> None:
        self.capacity = capacity
        self.rate = capacity / period
        self.clock = clock
        self.max_buckets = max_buckets
        # LRU: every hit moves the identity to the end; eviction pops from the front, so an
        # identity that keeps hitting (e.g. a drained attacker bucket) is never the one dropped.
        self._buckets: "OrderedDict[str, Tuple[float, float]]" = OrderedDict()
        self._lock = threading.Lock()

    def hit(self, identity: str, cost: float = 1.0) -> Decision:
        # an endpoint costing more than the whole bucket would otherwise be blocked forever
        cost = min(cost, float(self.capacity))
        now = self.clock()
        with self._lock:
            tokens, last = self._buckets.get(identity, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= cost:
                tokens -= cost
                allowed, retry = True, 0.0
            else:
                allowed, retry = False, (cost - tokens) / self.rate
            self._buckets[identity] = (tokens, now)
            self._buckets.move_to_end(identity)
            while len(self._buckets) > self.max_buckets:
                self._buckets.popitem(last=False)
        return Decision(allowed, self.capacity, int(math.floor(tokens)), retry)


def _digest(key: str) -> bytes:
    return hashlib.sha256(key.encode("utf-8")).digest()


@dataclass
class Verdict:
    """Outcome of :meth:`AccessControl.authorize`: ``status`` set = reject; otherwise ``ident`` is the caller."""

    status: Optional[int] = None
    detail: str = ""
    ident: str = ""
    headers: Dict[str, str] = field(default_factory=dict)


class AccessControl:
    def __init__(self, keys: List[str], limiter: Optional[RateLimiter],
                 trusted_proxies: Sequence[IPNetwork] = ()) -> None:
        # fixed-length digests: compare_digest does the same work whatever the key length
        self._digests = [_digest(k) for k in keys]
        self.limiter = limiter
        self.trusted_proxies = list(trusted_proxies)

    @classmethod
    def from_env(cls, env=os.environ, clock: Callable[[], float] = time.monotonic) -> "AccessControl":
        rate = parse_rate(env.get("AQUAGUARD_RATE_LIMIT"))
        limiter = RateLimiter(rate[0], rate[1], clock=clock) if rate else None
        return cls(parse_keys(env.get("AQUAGUARD_API_KEYS")), limiter,
                   parse_trusted_proxies(env.get("AQUAGUARD_TRUSTED_PROXIES")))

    @property
    def enabled(self) -> bool:
        return bool(self._digests)

    @staticmethod
    def extract_key(headers) -> Optional[str]:
        key = headers.get("x-api-key")
        if key:
            return key.strip()
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return None

    def check_key(self, key: Optional[str]) -> bool:
        if not self.enabled:
            return True
        if not key:
            return False
        candidate = _digest(key)
        ok = False
        for d in self._digests:  # no early exit: constant work per request
            ok |= hmac.compare_digest(candidate, d)
        return ok

    def client_ip(self, peer: Optional[str], headers) -> str:
        return client_ip(peer, headers.get("x-forwarded-for"), self.trusted_proxies)

    def authorize(self, headers, peer: Optional[str], cost: float = 1.0) -> Verdict:
        """Check the key and charge ``cost`` tokens. Framework-free, so the ASGI upload guard
        (before the body is read) and the route dependency share it."""
        key = self.extract_key(headers)
        ip = self.client_ip(peer, headers)
        if not self.check_key(key):
            if self.limiter is not None:  # failed attempts spend the caller's IP bucket: key guessing is throttled
                d = self.limiter.hit(f"ip:{ip}", 1.0)
                if not d.allowed:
                    return Verdict(429, "Too many failed authentication attempts.",
                                   headers={"Retry-After": str(max(1, math.ceil(d.retry_after)))})
            return Verdict(401, "Missing or invalid API key. Send it as 'X-API-Key: <key>' or "
                                "'Authorization: Bearer <key>'.",
                           headers={"WWW-Authenticate": 'Bearer realm="aquaguard"'})
        out: Dict[str, str] = {}
        if not self.enabled:
            out["X-AquaGuard-Auth"] = "disabled"
        ident = f"key:{fingerprint(key)}" if (self.enabled and key) else f"ip:{ip}"
        if self.limiter is not None:
            d = self.limiter.hit(ident, cost)
            out.update({"X-RateLimit-Limit": str(d.limit), "X-RateLimit-Remaining": str(max(d.remaining, 0))})
            if not d.allowed:
                out["Retry-After"] = str(max(1, math.ceil(d.retry_after)))
                return Verdict(429, "Rate limit exceeded. Retry after the number of seconds in Retry-After.",
                               ident=ident, headers=out)
        return Verdict(None, ident=ident, headers=out)
