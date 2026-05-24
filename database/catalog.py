"""
Data-Driven Reference Catalogs — part_type / glass_types soft validation.

Design decisions:
- ``part_type`` stays a plain ``str`` in the API contract (max flexibility).
- A ``glass_types`` DB table acts as a reference catalog.
- Validation is SOFT: unknown types produce a warning, not a rejection.
  New types are added via a DB insert — never a code deploy.
- The in-process ``PartTypeCatalog`` caches the DB table so lookups are O(1).
- Cache is refreshed on-demand (TTL-based or explicit reload) to avoid
  stale data without a restart.
- Multi-tenant: every catalog is scoped to a ``domain`` so different product
  lines (auto glass, FNOL, medical) can share the same infrastructure while
  maintaining independent type sets.
- All DB access uses SQLAlchemy parameterized queries — no f-string SQL.

Apply to: glass replacement claims (part_type field), any domain where a
          controlled vocabulary needs to grow without code deploys.

Schema (reference only — run migrations separately):

    CREATE TABLE glass_types (
        id          SERIAL PRIMARY KEY,
        type_code   VARCHAR(64)  NOT NULL,
        label       VARCHAR(255) NOT NULL,
        domain      VARCHAR(64)  NOT NULL DEFAULT 'auto_glass',
        active      BOOLEAN      NOT NULL DEFAULT TRUE,
        created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        UNIQUE (type_code, domain)
    );

    -- Seed data
    INSERT INTO glass_types (type_code, label, domain) VALUES
        ('windshield',       'Windshield',        'auto_glass'),
        ('rear_window',      'Rear Window',        'auto_glass'),
        ('side_window',      'Side Window',        'auto_glass'),
        ('sunroof',          'Sunroof/Moonroof',   'auto_glass'),
        ('quarter_glass',    'Quarter Glass',      'auto_glass'),
        ('vent_glass',       'Vent Glass',         'auto_glass'),
        ('mirror',           'Mirror',             'auto_glass'),
        ('door_glass',       'Door Glass',         'auto_glass');

Usage:
    catalog = PartTypeCatalog(domain="auto_glass")
    catalog.load_from_records([
        PartTypeRecord(type_code="windshield", label="Windshield"),
        PartTypeRecord(type_code="sunroof", label="Sunroof/Moonroof"),
    ])

    result = catalog.validate("windshield")
    # ValidationResult(valid=True, known=True, type_code="windshield", ...)

    result = catalog.validate("panoramic_roof")
    # ValidationResult(valid=True, known=False, warning="...", ...)
    # → soft: accepted but flagged for later review / catalog enrichment
"""

import logging
import re
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_TYPE_CODE_LENGTH = 64
_MAX_DOMAIN_LENGTH = 64
# Only alphanumerics and underscores in type codes — prevents injection
_TYPE_CODE_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")
_DOMAIN_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


# ===========================================================================
# Data types
# ===========================================================================


@dataclass(frozen=True)
class PartTypeRecord:
    """
    One entry in the glass_types reference catalog.

    Fields:
        type_code: Machine-stable identifier (e.g. "windshield").
                   Only alphanumerics and underscores — safe to use as a
                   parameterized query value.
        label:     Human-readable display string.
        domain:    Product-line scope (e.g. "auto_glass", "commercial").
        active:    False means the type is deprecated; still returned for
                   historical records but flagged in new submissions.
    """

    type_code: str
    label: str
    domain: str = "auto_glass"
    active: bool = True

    def __post_init__(self) -> None:
        if not _TYPE_CODE_RE.fullmatch(self.type_code):
            raise ValueError(
                f"type_code '{self.type_code}' contains invalid characters. "
                "Only [A-Za-z0-9_] are permitted."
            )
        if not _DOMAIN_RE.fullmatch(self.domain):
            raise ValueError(
                f"domain '{self.domain}' contains invalid characters."
            )


@dataclass(frozen=True)
class ValidationResult:
    """
    Outcome of a soft ``part_type`` validation.

    The API contract never rejects an unknown type (``valid`` is always True
    for well-formed strings); it only sets ``known=False`` and populates
    ``warning`` so the caller can emit a metric / log / alert.

    Fields:
        valid:     True unless the input itself is malformed (empty, too long,
                   or contains dangerous characters).
        known:     True if type_code is present in the active catalog.
        type_code: The normalized type code that was looked up.
        label:     Human-readable label from the catalog, or None if unknown.
        active:    False if the type is present but deprecated.
        warning:   Non-empty when ``known=False`` or ``active=False``.
    """

    valid: bool
    known: bool
    type_code: str
    label: Optional[str] = None
    active: bool = True
    warning: str = ""


class PartTypeValidationError(ValueError):
    """Raised when the input is structurally invalid (not just unknown)."""


# ===========================================================================
# Catalog
# ===========================================================================


class PartTypeCatalog:
    """
    O(1) in-process cache of the glass_types reference table.

    Backed by a plain dict indexed by ``(type_code, domain)``.  Load from
    a list of :class:`PartTypeRecord` objects (populated from DB at startup)
    or via :meth:`load_from_db` with a SQLAlchemy session.

    Thread safety: a ``threading.RLock`` serialises all writes (reloads).
    Reads are lock-free after initial load because Python dict reads are
    effectively atomic for GIL-bound workloads; for absolute correctness
    they are protected by the same lock.

    Args:
        domain:       Product-line scope filter; only records matching this
                      domain are validated.
        cache_ttl:    Seconds before the cache is considered stale (used by
                      :meth:`maybe_reload`).
    """

    def __init__(
        self,
        domain: str = "auto_glass",
        cache_ttl: float = 300.0,
    ) -> None:
        if not _DOMAIN_RE.fullmatch(domain):
            raise ValueError(f"Invalid domain: '{domain}'")
        self._domain = domain
        self._cache_ttl = cache_ttl
        self._lock = threading.RLock()
        # (type_code, domain) → PartTypeRecord
        self._index: Dict[str, PartTypeRecord] = {}
        self._loaded_at: float = 0.0

    # ------------------------------------------------------------------
    # Population (data-driven, not hardcoded)
    # ------------------------------------------------------------------

    def load_from_records(self, records: List[PartTypeRecord]) -> None:
        """
        Bulk-load part type records into the cache.

        Only records whose ``domain`` matches this catalog's domain are
        indexed.  This is idempotent — calling multiple times merges.
        """
        with self._lock:
            for rec in records:
                if rec.domain == self._domain:
                    self._index[rec.type_code] = rec
            self._loaded_at = time.monotonic()
        logger.info(
            "PartTypeCatalog[%s]: loaded %d active records",
            self._domain,
            len(self._index),
        )

    def add_type(self, record: PartTypeRecord) -> None:
        """
        Add or replace a single part-type record (thread-safe).

        New glass shapes are added this way at runtime — no code deploy needed.
        """
        if record.domain != self._domain:
            raise ValueError(
                f"Record domain '{record.domain}' does not match "
                f"catalog domain '{self._domain}'"
            )
        with self._lock:
            self._index[record.type_code] = record
            self._loaded_at = time.monotonic()

    def deactivate_type(self, type_code: str) -> bool:
        """
        Mark a type as inactive (deprecated) without removing it.

        Returns True if the type was found and updated.
        """
        clean_code = _clean_type_code(type_code)
        with self._lock:
            rec = self._index.get(clean_code)
            if rec is None:
                return False
            # Replace with an inactive copy (frozen dataclass → new instance)
            self._index[clean_code] = PartTypeRecord(
                type_code=rec.type_code,
                label=rec.label,
                domain=rec.domain,
                active=False,
            )
        return True

    # ------------------------------------------------------------------
    # Validation — O(1)
    # ------------------------------------------------------------------

    def validate(self, part_type: str) -> ValidationResult:
        """
        Soft-validate a ``part_type`` value against the catalog.

        Never rejects an unknown type — returns ``known=False`` with a
        warning so downstream systems can emit metrics / alerts without
        blocking the claim.

        Security: ``part_type`` is sanitized to ``[A-Za-z0-9_]`` before any
        lookup, preventing injection through the type-code path.

        Args:
            part_type: Raw part type string from the API request.

        Returns:
            :class:`ValidationResult` — always ``valid=True`` for structurally
            correct input, ``known=True/False`` depending on catalog membership.

        Raises:
            PartTypeValidationError: if ``part_type`` is empty, too long, or
                                     contains structurally invalid characters.
        """
        clean = _clean_type_code(part_type)  # raises on invalid input

        with self._lock:
            rec = self._index.get(clean)

        if rec is None:
            logger.warning(
                "Unknown part_type '%s' for domain '%s'. "
                "Add it to the glass_types catalog via DB insert.",
                clean,
                self._domain,
            )
            return ValidationResult(
                valid=True,
                known=False,
                type_code=clean,
                warning=(
                    f"part_type '{clean}' is not in the '{self._domain}' catalog. "
                    "It has been accepted but flagged for catalog review."
                ),
            )

        if not rec.active:
            return ValidationResult(
                valid=True,
                known=True,
                type_code=clean,
                label=rec.label,
                active=False,
                warning=(
                    f"part_type '{clean}' is deprecated in the "
                    f"'{self._domain}' catalog. Consider using a current type."
                ),
            )

        return ValidationResult(
            valid=True,
            known=True,
            type_code=clean,
            label=rec.label,
            active=True,
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def maybe_reload(self, reload_fn) -> bool:  # type: ignore[type-arg]
        """
        Re-populate the cache if the TTL has elapsed.

        Args:
            reload_fn: Zero-argument callable that returns
                       ``List[PartTypeRecord]``.  Called only when stale.

        Returns True if a reload was performed.
        """
        if time.monotonic() - self._loaded_at < self._cache_ttl:
            return False
        records = reload_fn()
        self.load_from_records(records)
        return True

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def domain(self) -> str:
        return self._domain

    def known_types(self) -> FrozenSet[str]:
        """Return the set of active type codes currently in the catalog."""
        with self._lock:
            return frozenset(
                code for code, rec in self._index.items() if rec.active
            )

    def size(self) -> int:
        """Total records in catalog (active + inactive)."""
        with self._lock:
            return len(self._index)


# ===========================================================================
# Input sanitization helper
# ===========================================================================


def _clean_type_code(raw: str) -> str:
    """
    Normalize and validate a part-type code.

    Lowercases the value and validates it matches ``[a-z0-9_]{1,64}``.

    Raises:
        PartTypeValidationError: on empty, too-long, or character-invalid input.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise PartTypeValidationError("part_type must be a non-empty string")
    cleaned = raw.strip().lower()
    if len(cleaned) > _MAX_TYPE_CODE_LENGTH:
        raise PartTypeValidationError(
            f"part_type exceeds max length of {_MAX_TYPE_CODE_LENGTH}"
        )
    # After lowercasing, only [a-z0-9_] are valid
    if not re.fullmatch(r"[a-z0-9_]{1,64}", cleaned):
        raise PartTypeValidationError(
            f"part_type '{cleaned}' contains invalid characters. "
            "Only [a-z0-9_] are permitted."
        )
    return cleaned


# ===========================================================================
# Abstract DB repository interface (parameterized queries required)
# ===========================================================================


class AbstractPartTypeRepository(ABC):
    """
    Abstract interface for a persistent glass_types store.

    Implementations MUST use parameterized queries.

    Correct (SQLAlchemy ORM):
        session.execute(
            select(GlassTypeRow)
            .where(GlassTypeRow.domain == bindparam("domain"))
            .where(GlassTypeRow.active.is_(True)),
            {"domain": domain},
        )

    Wrong (SQL injection risk):
        session.execute(f"SELECT * FROM glass_types WHERE domain = '{domain}'")
    """

    @abstractmethod
    async def list_active(self, domain: str) -> List[PartTypeRecord]:
        """Return all active part types for a domain (for cache population)."""

    @abstractmethod
    async def find(self, type_code: str, domain: str) -> Optional[PartTypeRecord]:
        """Exact lookup by type_code + domain."""

    @abstractmethod
    async def insert(self, record: PartTypeRecord) -> None:
        """Insert a new part type.  This is how new types are added — not code deploys."""

    @abstractmethod
    async def deactivate(self, type_code: str, domain: str) -> bool:
        """Mark a type as inactive.  Returns True if found and updated."""
