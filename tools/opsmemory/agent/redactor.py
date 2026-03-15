"""Secret redaction module — strips credentials from text before storage."""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Tuple

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Compiled redaction patterns
# ---------------------------------------------------------------------------
REDACTION_PATTERNS: list[re.Pattern[str]] = [
    # AWS access key IDs
    re.compile(r"AKIA[0-9A-Z]{16}"),
    # AWS secret access keys (must be preceded by aws/secret context)
    re.compile(r"(?i)aws.{0,10}secret.{0,30}[=:]\s*([A-Za-z0-9/+=]{20,})"),
    # Generic API keys, tokens, secrets, passwords
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+"),
    # PEM private key blocks
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
    # GitHub personal access tokens and fine-grained tokens
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}"),
]


def redact_text(text: str) -> Tuple[str, int]:
    """Apply all redaction patterns to *text*.

    Returns a tuple of (redacted_text, count_of_replacements).
    """
    redacted = text
    total_count = 0
    for pattern in REDACTION_PATTERNS:
        redacted, n = pattern.subn("[REDACTED]", redacted)
        total_count += n
    return redacted, total_count


@dataclass
class RedactionEvent:
    """Structured record of a redaction operation."""

    correlation_id: str
    redaction_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def log_redaction_event(event: RedactionEvent) -> None:
    """Emit a structured log entry for a redaction event."""
    log.info(
        "secrets_redacted",
        correlation_id=event.correlation_id,
        redaction_count=event.redaction_count,
        timestamp=event.timestamp.isoformat(),
    )
