"""
Idempotency Key Store — O(1) atomic check-and-set, duplicate-request prevention.

Design decisions:
- Idempotency keys are client-supplied UUIDs (or any opaque string ≤ 128 chars).
- State machine per key: PROCESSING → COMPLETED | FAILED.
- Atomic compare-and-set (CAS) via threading.Lock prevents race conditions:
  the first request acquires the key, concurrent duplicates get the stored result.
- Append-only: completed/failed outcomes are never mutated (immutable result records).
- TTL: entries expire after a configurable window (default 24 hours) to bound
  memory growth without needing a background GC thread — lazy eviction on access.
- In production, replace InMemoryIdempotencyStore with a Redis/Postgres backend.
  The abstract interface guarantees all call sites remain unchanged.
- O(1) space per key (one record); O(1) time for all operations (hash map).

Apply to: claims submission, payment processing, carrier API dispatch,
          any write endpoint that must not double-execute on retry.

Usage:
    store = InMemoryIdempotencyStore(ttl_seconds=86400)

    async def submit_claim(idempotency_key: str, payload: dict):
        async with idempotent_operation(store, idempotency_key) as ctx:
            if ctx.already_completed:
                return ctx.stored_result     # replay cached response
            result = await _do_real_work(payload)
            ctx.result = result              # persisted on context exit
        return result
"""

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_KEY_LENGTH = 128
_DEFAULT_TTL_SECONDS = 86_400  # 24 hours


# ===========================================================================
# Key state machine
# ===========================================================================


class IdempotencyState(str, Enum):
    """Lifecycle of an idempotency key."""

    PROCESSING = "processing"   # First request is in-flight
    COMPLETED = "completed"     # Successfully finished; result cached
    FAILED = "failed"           # Request failed; may be retried


# ===========================================================================
# Data types
# ===========================================================================


@dataclass
class IdempotencyRecord:
    """
    Stored state for one idempotency key.

    Immutable once state reaches COMPLETED or FAILED (no mutation after
    terminal state).
    """

    key: str
    state: IdempotencyState
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    result: Optional[Any] = None
    error: Optional[str] = None
    key_hash: str = ""          # SHA-256 of key for safe logging

    def __post_init__(self) -> None:
        if not self.key_hash:
            self.key_hash = hashlib.sha256(self.key.encode()).hexdigest()[:16]

    @property
    def is_terminal(self) -> bool:
        return self.state in (IdempotencyState.COMPLETED, IdempotencyState.FAILED)


class DuplicateRequestError(Exception):
    """
    Raised when a request with the same idempotency key is already PROCESSING.

    The caller should return HTTP 409 Conflict and ask the client to retry
    after a short backoff.
    """

    def __init__(self, key_hash: str) -> None:
        self.key_hash = key_hash
        super().__init__(
            f"Request with idempotency key [{key_hash}] is already processing. "
            "Retry after the in-flight request completes."
        )


class IdempotencyKeyError(ValueError):
    """Raised when the provided idempotency key fails validation."""


# ===========================================================================
# Abstract store interface
# ===========================================================================


class AbstractIdempotencyStore(ABC):
    """
    Abstract persistence interface for idempotency records.

    All implementations must be:
    - Thread-safe (or async-safe for async implementations).
    - O(1) time for ``get`` and ``cas`` operations.
    - Safe against SQL injection: use parameterized queries for any DB backend.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[IdempotencyRecord]:
        """
        Retrieve the record for ``key``.

        Returns None if no record exists or the record has expired.
        """

    @abstractmethod
    def cas(self, key: str) -> IdempotencyRecord:
        """
        Compare-and-set: atomically create a PROCESSING record for ``key``.

        If no record exists (or the record has expired), a new PROCESSING
        record is created and returned.

        If a record already exists in PROCESSING state, :class:`DuplicateRequestError`
        is raised — the first caller "wins"; the duplicate is rejected.

        If a terminal record exists (COMPLETED or FAILED), it is returned
        immediately so the caller can replay the cached result.
        """

    @abstractmethod
    def complete(self, key: str, result: Any) -> None:
        """Mark a PROCESSING key as COMPLETED with the given result."""

    @abstractmethod
    def fail(self, key: str, error: str) -> None:
        """Mark a PROCESSING key as FAILED with an error message."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a key (e.g., after TTL expiry or explicit cancellation)."""


# ===========================================================================
# In-memory implementation
# ===========================================================================


def _validate_key(key: str) -> str:
    """Validate and return the idempotency key, or raise IdempotencyKeyError."""
    if not isinstance(key, str) or not key.strip():
        raise IdempotencyKeyError("idempotency_key must be a non-empty string")
    stripped = key.strip()
    if len(stripped) > _MAX_KEY_LENGTH:
        raise IdempotencyKeyError(
            f"idempotency_key exceeds max length of {_MAX_KEY_LENGTH} characters"
        )
    return stripped


class InMemoryIdempotencyStore(AbstractIdempotencyStore):
    """
    In-memory idempotency store backed by a plain dict.

    Use this for:
    - Local development and unit tests
    - Single-process deployments (no horizontal scaling)

    For multi-instance deployments, replace with a Redis or Postgres backend
    that uses atomic SETNX / INSERT ... ON CONFLICT DO NOTHING semantics.

    Thread safety: a single ``threading.Lock`` protects all dict mutations.
    All public operations are O(1).

    Args:
        ttl_seconds: How long completed/failed records are retained.
                     Expired records are evicted lazily on next access.
    """

    def __init__(self, ttl_seconds: float = _DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._records: Dict[str, IdempotencyRecord] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_expired(self, record: IdempotencyRecord) -> bool:
        """True if the record's TTL has elapsed (lazy eviction)."""
        return (time.monotonic() - record.created_at) > self._ttl

    # ------------------------------------------------------------------
    # AbstractIdempotencyStore implementation
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[IdempotencyRecord]:
        """O(1) lookup; evicts expired records transparently."""
        safe_key = _validate_key(key)
        with self._lock:
            record = self._records.get(safe_key)
            if record is None:
                return None
            if self._is_expired(record):
                del self._records[safe_key]
                return None
            return record

    def cas(self, key: str) -> IdempotencyRecord:
        """
        Atomic compare-and-set.

        - No record / expired → create PROCESSING record (O(1)).
        - PROCESSING           → raise DuplicateRequestError.
        - COMPLETED / FAILED   → return existing terminal record.
        """
        safe_key = _validate_key(key)

        with self._lock:
            record = self._records.get(safe_key)

            if record is not None and not self._is_expired(record):
                if record.state == IdempotencyState.PROCESSING:
                    raise DuplicateRequestError(record.key_hash)
                # Terminal state — replay
                return record

            # Create new PROCESSING record (or overwrite expired one)
            new_record = IdempotencyRecord(
                key=safe_key,
                state=IdempotencyState.PROCESSING,
            )
            self._records[safe_key] = new_record
            return new_record

    def complete(self, key: str, result: Any) -> None:
        """O(1) state transition PROCESSING → COMPLETED."""
        safe_key = _validate_key(key)
        with self._lock:
            record = self._records.get(safe_key)
            if record is None:
                raise KeyError(f"No idempotency record found for key hash [...{safe_key[-8:]}]")
            if record.is_terminal:
                return  # idempotent — already completed
            record.state = IdempotencyState.COMPLETED
            record.result = result
            record.updated_at = time.monotonic()

    def fail(self, key: str, error: str) -> None:
        """O(1) state transition PROCESSING → FAILED."""
        safe_key = _validate_key(key)
        with self._lock:
            record = self._records.get(safe_key)
            if record is None:
                raise KeyError(f"No idempotency record found for key hash [...{safe_key[-8:]}]")
            if record.is_terminal:
                return  # idempotent — already finalized
            record.state = IdempotencyState.FAILED
            record.error = error
            record.updated_at = time.monotonic()

    def delete(self, key: str) -> None:
        """O(1) delete."""
        safe_key = _validate_key(key)
        with self._lock:
            self._records.pop(safe_key, None)

    def size(self) -> int:
        """Return the number of currently stored records (for diagnostics)."""
        with self._lock:
            return len(self._records)


# ===========================================================================
# Synchronous context-manager helper
# ===========================================================================


@dataclass
class _IdempotencyContext:
    """Mutable context handed to the caller inside :func:`idempotent_operation`."""

    already_completed: bool
    stored_result: Optional[Any]
    result: Optional[Any] = None  # caller sets this on success


@contextmanager
def idempotent_operation(
    store: AbstractIdempotencyStore,
    key: str,
) -> Generator[_IdempotencyContext, None, None]:
    """
    Synchronous context manager for idempotent write operations.

    On entry:
    - If the key has a COMPLETED record → yields context with
      ``already_completed=True`` and ``stored_result`` populated.
      No work should be performed; simply return ``stored_result``.
    - If the key is new → yields context with ``already_completed=False``.
      Perform work, then set ``ctx.result`` before exiting the block.
    - If the key is currently PROCESSING → raises :class:`DuplicateRequestError`.

    On exit (no exception):
    - Calls ``store.complete(key, ctx.result)``.

    On exit (exception):
    - Calls ``store.fail(key, str(exc))`` and re-raises.

    Example::

        with idempotent_operation(store, idempotency_key) as ctx:
            if ctx.already_completed:
                return ctx.stored_result
            result = perform_expensive_work()
            ctx.result = result
        return result
    """
    record = store.cas(key)

    if record.is_terminal:
        ctx = _IdempotencyContext(
            already_completed=True,
            stored_result=record.result,
        )
        yield ctx
        return

    ctx = _IdempotencyContext(already_completed=False, stored_result=None)
    try:
        yield ctx
        store.complete(key, ctx.result)
    except DuplicateRequestError:
        raise
    except Exception as exc:
        store.fail(key, str(exc))
        raise
