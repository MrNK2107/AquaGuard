"""API keys and per-client rate limiting for the AquaGuard API.

Configuration (environment variables, read at startup):

  AQUAGUARD_API_KEYS    comma-separated keys. Unset or empty = auth disabled
                        (local development); responses then carry
                        ``X-AquaGuard-Auth: disabled`` so nobody mistakes it for a
                        secured deployment.
  AQUAGUARD_RATE_LIMIT  "<requests>/<second|minute|hour>", default "60/minute".
                        "0" or "off" disables limiting.

Clients send the key as ``X-API-Key: <key>`` or ``Authorization: Bearer <key>``.
Keys are compared in constant time and never logged; logs and limiter buckets
use a short SHA-256 fingerprint instead.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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
        self._buckets: Dict[str, Tuple[float, float]] = {}
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
                self._buckets[identity] = (tokens, now)
                allowed, retry = True, 0.0
            else:
                self._buckets[identity] = (tokens, now)
                allowed, retry = False, (cost - tokens) / self.rate
            if len(self._buckets) > self.max_buckets:  # crude bound on memory
                oldest = sorted(self._buckets.items(), key=lambda kv: kv[1][1])[: len(self._buckets) // 2]
                for k, _ in oldest:
                    self._buckets.pop(k, None)
        return Decision(allowed, self.capacity, int(math.floor(tokens)), retry)


class AccessControl:
    def __init__(self, keys: List[str], limiter: Optional[RateLimiter]) -> None:
        self._keys = [k.encode("utf-8") for k in keys]
        self.limiter = limiter

    @classmethod
    def from_env(cls, env=os.environ, clock: Callable[[], float] = time.monotonic) -> "AccessControl":
        rate = parse_rate(env.get("AQUAGUARD_RATE_LIMIT"))
        limiter = RateLimiter(rate[0], rate[1], clock=clock) if rate else None
        return cls(parse_keys(env.get("AQUAGUARD_API_KEYS")), limiter)

    @property
    def enabled(self) -> bool:
        return bool(self._keys)

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
        candidate = key.encode("utf-8")
        ok = False
        for k in self._keys:  # no early exit: constant work per request
            ok |= hmac.compare_digest(candidate, k)
        return ok
