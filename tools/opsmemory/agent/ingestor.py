"""Inbox watcher — ingests files and manual payloads into OpsMemory."""

import asyncio
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog
from pydantic import BaseModel, Field

from tools.opsmemory.agent.consolidator import generate_embedding
from tools.opsmemory.agent.redactor import RedactionEvent, log_redaction_event, redact_text
from tools.opsmemory.storage.models import EvidenceItem
from tools.opsmemory.storage.repository import EmbeddingRepository, EvidenceRepository

log = structlog.get_logger(__name__)


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

    item = await EvidenceRepository.create(
        session,
        raw_text=redacted_text,
        source_type=payload.source_type,
        source_ref=payload.source_ref,
        author=payload.author,
        metadata=payload.metadata,
        correlation_id=correlation_id,
    )

    vec = generate_embedding(redacted_text)
    await EmbeddingRepository.upsert(session, item.id, vec)

    log.info(
        "evidence_ingested",
        evidence_id=str(item.id),
        source_type=payload.source_type,
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
