"""Inbox watcher — ingests files and manual payloads into OpsMemory."""

import asyncio
import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
from pydantic import BaseModel, Field

from tools.opsmemory.agent.consolidator import generate_embedding
from tools.opsmemory.agent.redactor import RedactionEvent, log_redaction_event, redact_text
from tools.opsmemory.storage.models import EvidenceItem
from tools.opsmemory.storage.repository import EmbeddingRepository, EvidenceRepository

log = structlog.get_logger(__name__)

_EXCERPT_MAX_CHARS = 500


@dataclass
class IngestorConfig:
    """Configuration for the inbox file watcher."""

    inbox_dir: str = "inbox"
    poll_interval: float = 5.0


class IngestPayload(BaseModel):
    """Schema for evidence submitted to the ingest endpoint or inbox JSON files."""

    text: str
    source_type: str = "manual"
    source_ref: str = ""
    author: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    # Fields for structured/auditable evidence ----------------------------
    # Owner/repo name (e.g. "EPdacoder05/my-repo") for GitHub sources.
    repo: Optional[str] = None
    # Native identifier in the originating source system.
    # For commits: commit SHA.  For PRs: PR number as a string.
    native_id: Optional[str] = None
    # ISO-8601 timestamp of when the original event occurred.
    occurred_at: Optional[str] = None


def compute_evidence_id(
    source_type: str,
    repo: str,
    native_id: str,
    occurred_at: str,
) -> str:
    """Compute a deterministic SHA-256 evidence ID.

    The ID is stable across re-ingestions of the same source event so that
    duplicate submissions are easy to detect.
    """
    raw = f"{source_type}:{repo}:{native_id}:{occurred_at}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def ingest_evidence(
    session,
    payload: IngestPayload,
    correlation_id: str,
) -> EvidenceItem:
    """Redact secrets, persist evidence, and upsert its embedding.

    Returns the newly created :class:`EvidenceItem`.
    """
    redacted_text, redaction_count = redact_text(payload.text)

    if redaction_count > 0:
        event = RedactionEvent(
            correlation_id=correlation_id, redaction_count=redaction_count
        )
        log_redaction_event(event)

    # Build a short, redacted excerpt for citation display.
    excerpt = redacted_text[:_EXCERPT_MAX_CHARS] if redacted_text else None

    # Compute deterministic sha256 evidence_id when enough context is available.
    evidence_id: Optional[str] = None
    if payload.repo and payload.native_id and payload.occurred_at:
        evidence_id = compute_evidence_id(
            source_type=payload.source_type,
            repo=payload.repo,
            native_id=str(payload.native_id),
            occurred_at=payload.occurred_at,
        )

    # Parse occurred_at to a datetime if provided.
    occurred_at_dt: Optional[datetime] = None
    if payload.occurred_at:
        try:
            occurred_at_dt = datetime.fromisoformat(
                payload.occurred_at.replace("Z", "+00:00")
            )
        except ValueError:
            pass

    item = await EvidenceRepository.create(
        session,
        raw_text=redacted_text,
        source_type=payload.source_type,
        source_ref=payload.source_ref,
        author=payload.author,
        occurred_at=occurred_at_dt,
        metadata=payload.metadata,
        correlation_id=correlation_id,
        evidence_id=evidence_id,
        repo=payload.repo,
        native_id=str(payload.native_id) if payload.native_id is not None else None,
        excerpt=excerpt,
    )

    vec = generate_embedding(redacted_text)
    await EmbeddingRepository.upsert(session, item.id, vec)

    log.info(
        "evidence_ingested",
        evidence_id=evidence_id or str(item.id),
        source_type=payload.source_type,
        repo=payload.repo,
        native_id=payload.native_id,
        redaction_count=redaction_count,
        correlation_id=correlation_id,
    )
    return item


async def watch_inbox(session_factory, config: IngestorConfig) -> None:
    """Poll *inbox_dir* for new ``.txt`` / ``.json`` files and ingest them.

    Processed files are moved to ``inbox_dir/processed/``.
    Errors are logged and the loop continues.
    """
    inbox = Path(config.inbox_dir)
    processed_dir = inbox / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    log.info("inbox_watcher_started", inbox_dir=str(inbox))

    while True:
        try:
            for path in sorted(inbox.glob("*.txt")) + sorted(inbox.glob("*.json")):
                correlation_id = str(uuid.uuid4())
                try:
                    if path.suffix == ".txt":
                        raw = path.read_text(encoding="utf-8")
                        payload = IngestPayload(text=raw, source_type="file", source_ref=path.name)
                    else:
                        data = json.loads(path.read_text(encoding="utf-8"))
                        payload = IngestPayload(**data)

                    async with session_factory() as session:
                        await ingest_evidence(session, payload, correlation_id)
                        await session.commit()

                    dest = processed_dir / path.name
                    shutil.move(str(path), str(dest))
                    log.info("inbox_file_processed", file=path.name, correlation_id=correlation_id)

                except Exception as file_exc:
                    log.error(
                        "inbox_file_error",
                        file=path.name,
                        error=str(file_exc),
                        correlation_id=correlation_id,
                    )
        except Exception as loop_exc:
            log.error("inbox_watch_loop_error", error=str(loop_exc))

        await asyncio.sleep(config.poll_interval)
