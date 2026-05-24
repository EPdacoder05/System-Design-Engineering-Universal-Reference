"""
Carrier-Aware Rate Limiter — abstract base + token-bucket + sliding-window.

Design decisions:
- Abstract base class ``BaseRateLimiter`` forces a uniform interface across
  all limiter variants (token bucket, sliding window, fixed window, …).
- ``TokenBucketRateLimiter`` is O(1) time and O(1) space per key:
  it stores (tokens, last_refill_ts) and recomputes on each call.
- ``SlidingWindowRateLimiter`` uses the "sliding window counter" algorithm:
  two integers per key → O(1) space, O(1) time, no list/deque growth.
- ``RateLimiterRegistry`` maps carrier_id → limiter config so each carrier
  can have independent limits (carrier APIs have wildly different quotas).
- Thread-safe: each limiter instance holds its own ``threading.Lock``.
- Async wrappers: ``acheck`` and ``arecord`` for FastAPI / asyncio callers.

Apply to: outbound carrier API calls, inbound claims ingestion endpoints,
          internal microservice-to-microservice traffic.

Usage:
    registry = RateLimiterRegistry()
    registry.register("progressive-001", TokenBucketRateLimiterConfig(
        capacity=100, refill_rate=10.0  # 10 tokens/s, burst up to 100
    ))
    registry.register("geico-002", SlidingWindowRateLimiterConfig(
        max_requests=50, window_seconds=60
    ))

    allowed, info = registry.check("progressive-001")
    if not allowed:
        raise TooManyRequestsError(info)
"""

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


# ===========================================================================
# Result type
# ===========================================================================


@dataclass(frozen=True)
class RateLimitResult:
    """
    Outcome of a single rate-limit check.

    Fields:
        allowed:         True if the request is within quota.
        remaining:       Tokens/requests remaining in the current window.
        reset_after:     Seconds until quota fully resets (approximate).
        limit:           The configured limit ceiling.
        retry_after:     Seconds the caller should wait before retrying
                         (0 when ``allowed`` is True).
    """

    allowed: bool
    remaining: float
    reset_after: float
    limit: float
    retry_after: float = 0.0


# ===========================================================================
# Abstract base
# ===========================================================================


class BaseRateLimiter(ABC):
    """
    Abstract base class for all rate-limiter variants.

    Subclasses must implement:
        check(key)  → (allowed, RateLimitResult)
        reset(key)  → None
        stats(key)  → dict

    Thread safety is the responsibility of each concrete implementation.
    """

    @abstractmethod
    def check(self, key: str) -> Tuple[bool, RateLimitResult]:
        """
        Check whether ``key`` is within its rate limit and consume one unit.

        Returns ``(True, result)`` if the request is allowed, or
        ``(False, result)`` if the caller should back off.

        This method must be O(1) time and O(1) space per key.
        """

    @abstractmethod
    def reset(self, key: str) -> None:
        """Clear rate-limit state for ``key`` (e.g., after a billing period)."""

    @abstractmethod
    def stats(self, key: str) -> Dict[str, Any]:
        """Return diagnostic information about the current state for ``key``."""

    # ------------------------------------------------------------------
    # Async shims — thin wrappers; no await needed since these are CPU-only
    # ------------------------------------------------------------------

    async def acheck(self, key: str) -> Tuple[bool, RateLimitResult]:
        """Async wrapper around :meth:`check` for FastAPI / asyncio callers."""
        return self.check(key)

    async def areset(self, key: str) -> None:
        """Async wrapper around :meth:`reset`."""
        self.reset(key)


# ===========================================================================
# Token-bucket limiter  —  O(1) time, O(1) space per key
# ===========================================================================


@dataclass
class TokenBucketRateLimiterConfig:
    """
    Configuration for a token-bucket rate limiter.

    Args:
        capacity:    Maximum number of tokens (burst ceiling).
        refill_rate: Tokens added per second (sustained throughput).
        cost:        Tokens consumed per request (default 1).
    """

    capacity: float = 60.0
    refill_rate: float = 10.0   # tokens/second
    cost: float = 1.0


@dataclass
class _TokenBucketState:
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)


class TokenBucketRateLimiter(BaseRateLimiter):
    """
    Token-bucket rate limiter.

    Each key gets an independent bucket.  On every :meth:`check` call the
    bucket is refilled based on elapsed time and then one token is consumed.

    Time complexity:  O(1) per check
    Space complexity: O(K) where K = number of distinct keys (one state tuple)
    """

    def __init__(self, config: Optional[TokenBucketRateLimiterConfig] = None) -> None:
        self._config = config or TokenBucketRateLimiterConfig()
        self._buckets: Dict[str, _TokenBucketState] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> Tuple[bool, RateLimitResult]:
        """O(1): refill bucket, consume one token, return result."""
        cfg = self._config
        now = time.monotonic()

        with self._lock:
            state = self._buckets.get(key)
            if state is None:
                state = _TokenBucketState(tokens=cfg.capacity, last_refill=now)
                self._buckets[key] = state

            # Refill proportional to elapsed time
            elapsed = now - state.last_refill
            state.tokens = min(cfg.capacity, state.tokens + elapsed * cfg.refill_rate)
            state.last_refill = now

            if state.tokens >= cfg.cost:
                state.tokens -= cfg.cost
                remaining = state.tokens
                reset_after = (cfg.capacity - remaining) / cfg.refill_rate
                return True, RateLimitResult(
                    allowed=True,
                    remaining=remaining,
                    reset_after=reset_after,
                    limit=cfg.capacity,
                    retry_after=0.0,
                )

            # Not enough tokens — compute how long to wait
            deficit = cfg.cost - state.tokens
            retry_after = deficit / cfg.refill_rate
            return False, RateLimitResult(
                allowed=False,
                remaining=0.0,
                reset_after=cfg.capacity / cfg.refill_rate,
                limit=cfg.capacity,
                retry_after=retry_after,
            )

    def reset(self, key: str) -> None:
        with self._lock:
            self._buckets.pop(key, None)

    def stats(self, key: str) -> Dict[str, Any]:
        with self._lock:
            state = self._buckets.get(key)
            if state is None:
                return {"tokens": self._config.capacity, "key": key}
            return {
                "key": key,
                "tokens": state.tokens,
                "capacity": self._config.capacity,
                "refill_rate": self._config.refill_rate,
            }


# ===========================================================================
# Sliding-window counter limiter  —  O(1) time, O(1) space per key
# ===========================================================================


@dataclass
class SlidingWindowRateLimiterConfig:
    """
    Configuration for the sliding-window counter limiter.

    The algorithm approximates a true sliding window using two counters
    (current window + previous window weighted by overlap).  It requires only
    two integers per key — O(1) space — and checks in O(1) time.

    Args:
        max_requests:   Maximum requests allowed in ``window_seconds``.
        window_seconds: Length of the sliding window in seconds.
    """

    max_requests: int = 100
    window_seconds: float = 60.0


@dataclass
class _SlidingWindowState:
    prev_count: int = 0
    curr_count: int = 0
    window_start: float = field(default_factory=time.monotonic)


class SlidingWindowRateLimiter(BaseRateLimiter):
    """
    Sliding-window counter rate limiter.

    Uses the "weighted two-counter" approximation:

        effective_count = prev_count * overlap_ratio + curr_count

    where ``overlap_ratio`` is the fraction of the previous window that
    overlaps the current window position.

    Time complexity:  O(1) per check
    Space complexity: O(K) — two integers per active key
    """

    def __init__(
        self, config: Optional[SlidingWindowRateLimiterConfig] = None
    ) -> None:
        self._config = config or SlidingWindowRateLimiterConfig()
        self._windows: Dict[str, _SlidingWindowState] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> Tuple[bool, RateLimitResult]:
        """O(1): compute weighted count, consume one request if allowed."""
        cfg = self._config
        now = time.monotonic()

        with self._lock:
            state = self._windows.get(key)
            if state is None:
                state = _SlidingWindowState(window_start=now)
                self._windows[key] = state

            elapsed = now - state.window_start

            # Advance windows if current window has expired
            if elapsed >= cfg.window_seconds * 2:
                # Both windows are stale — reset fully
                state.prev_count = 0
                state.curr_count = 0
                state.window_start = now
                elapsed = 0.0
            elif elapsed >= cfg.window_seconds:
                # Roll: curr becomes prev, start a new curr window
                state.prev_count = state.curr_count
                state.curr_count = 0
                state.window_start = now + (elapsed - cfg.window_seconds)
                elapsed = now - state.window_start

            # Fraction of previous window still within the sliding view
            overlap = max(0.0, 1.0 - elapsed / cfg.window_seconds)
            effective = state.prev_count * overlap + state.curr_count

            remaining = max(0.0, cfg.max_requests - effective - 1)
            reset_after = cfg.window_seconds - elapsed

            if effective < cfg.max_requests:
                state.curr_count += 1
                return True, RateLimitResult(
                    allowed=True,
                    remaining=remaining,
                    reset_after=reset_after,
                    limit=float(cfg.max_requests),
                    retry_after=0.0,
                )

            # Over limit
            retry_after = cfg.window_seconds - elapsed
            return False, RateLimitResult(
                allowed=False,
                remaining=0.0,
                reset_after=reset_after,
                limit=float(cfg.max_requests),
                retry_after=max(0.0, retry_after),
            )

    def reset(self, key: str) -> None:
        with self._lock:
            self._windows.pop(key, None)

    def stats(self, key: str) -> Dict[str, Any]:
        cfg = self._config
        with self._lock:
            state = self._windows.get(key)
            if state is None:
                return {
                    "key": key,
                    "curr_count": 0,
                    "prev_count": 0,
                    "max_requests": cfg.max_requests,
                }
            return {
                "key": key,
                "curr_count": state.curr_count,
                "prev_count": state.prev_count,
                "max_requests": cfg.max_requests,
                "window_seconds": cfg.window_seconds,
            }


# ===========================================================================
# Registry — maps carrier_id → dedicated limiter instance
# ===========================================================================


class RateLimiterRegistry:
    """
    Central registry that maps a carrier_id (or any string key) to a
    per-carrier :class:`BaseRateLimiter` instance.

    Different carriers expose wildly different API quotas; this registry lets
    you configure each independently without any hardcoded enum.

    Args:
        default_limiter: Fallback limiter used for unregistered carrier keys.
                         Defaults to a conservative token bucket (60 req/min).

    Thread safety: registry operations are O(1) dict lookups behind a lock.
    """

    def __init__(
        self, default_limiter: Optional[BaseRateLimiter] = None
    ) -> None:
        self._limiters: Dict[str, BaseRateLimiter] = {}
        self._lock = threading.Lock()
        self._default = default_limiter or TokenBucketRateLimiter(
            TokenBucketRateLimiterConfig(capacity=60, refill_rate=1.0)
        )

    def register(
        self,
        carrier_id: str,
        config: Any,
    ) -> None:
        """
        Register a rate-limit config for a carrier.

        Args:
            carrier_id: Stable carrier identifier (from the carrier catalog).
            config:     A ``TokenBucketRateLimiterConfig`` or
                        ``SlidingWindowRateLimiterConfig`` instance.
        """
        if isinstance(config, TokenBucketRateLimiterConfig):
            limiter: BaseRateLimiter = TokenBucketRateLimiter(config)
        elif isinstance(config, SlidingWindowRateLimiterConfig):
            limiter = SlidingWindowRateLimiter(config)
        else:
            raise TypeError(
                f"Unsupported config type: {type(config).__name__}.  "
                "Use TokenBucketRateLimiterConfig or SlidingWindowRateLimiterConfig."
            )

        with self._lock:
            self._limiters[carrier_id] = limiter

    def register_limiter(
        self, carrier_id: str, limiter: BaseRateLimiter
    ) -> None:
        """Register a pre-built limiter instance directly."""
        with self._lock:
            self._limiters[carrier_id] = limiter

    def check(self, carrier_id: str) -> Tuple[bool, RateLimitResult]:
        """
        Check rate limit for ``carrier_id``.

        Falls back to the default limiter if no carrier-specific limiter has
        been registered.  All lookups are O(1).
        """
        with self._lock:
            limiter = self._limiters.get(carrier_id, self._default)
        return limiter.check(carrier_id)

    async def acheck(self, carrier_id: str) -> Tuple[bool, RateLimitResult]:
        """Async wrapper for FastAPI / asyncio callers."""
        return self.check(carrier_id)

    def reset(self, carrier_id: str) -> None:
        with self._lock:
            limiter = self._limiters.get(carrier_id)
        if limiter:
            limiter.reset(carrier_id)

    def stats(self, carrier_id: str) -> Dict[str, Any]:
        with self._lock:
            limiter = self._limiters.get(carrier_id, self._default)
        return limiter.stats(carrier_id)

    @property
    def registered_carriers(self) -> frozenset:
        """Return the set of carrier_ids with custom limiters registered."""
        with self._lock:
            return frozenset(self._limiters.keys())
