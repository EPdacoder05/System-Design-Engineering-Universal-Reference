"""Example flows for the Media Processing → OpsMemory ingestion integration.

These examples demonstrate how pipeline stages push normalised evidence into
OpsMemory:

1. Ingest a transcript result.
2. Ingest an OCR extraction result.
3. Batch-ingest multiple items.
4. Health check before ingestion.

All examples use environment variables for configuration.  No secrets or
personal identifiers are included.

Run any example::

    python -m tools.opsmemory.integrations.media_pipeline.examples
"""

from __future__ import annotations

import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _should_run_live_examples() -> bool:
    """Return True unless ``OPSMEMORY_SKIP_LIVE=true`` is set.

    Set ``OPSMEMORY_SKIP_LIVE=true`` in CI or unit-test runs to skip examples
    that require a running OpsMemory instance.
    """
    return os.environ.get("OPSMEMORY_SKIP_LIVE", "false").lower() not in (
        "1", "true", "yes"
    )


# ---------------------------------------------------------------------------
# Example 1: Ingest a transcript result
# ---------------------------------------------------------------------------


async def example_ingest_transcript() -> None:
    """Push a speech-to-text transcript result into OpsMemory."""
    from tools.opsmemory.integrations.media_pipeline.client import MediaIngestionClient
    from tools.opsmemory.integrations.media_pipeline.models import TranscriptResult

    client = MediaIngestionClient()

    result = TranscriptResult(
        text=(
            "Speaker A: The new caching layer reduced p99 latency by 40 percent. "
            "Speaker B: Great — let's include that in the release notes."
        ),
        source_ref="media://recordings/standup-2026-03-15.wav",
        native_id="standup-2026-03-15",
        occurred_at="2026-03-15T09:00:00Z",
        language="en",
        confidence=0.95,
        duration_seconds=180.0,
    )

    log.info("Ingesting transcript result...")
    response = await client.ingest(result)
    log.info("Ingested transcript. evidence_id=%s", response.get("evidence_id"))
    return response  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Example 2: Ingest an OCR extraction result
# ---------------------------------------------------------------------------


async def example_ingest_ocr() -> None:
    """Push an OCR extraction result into OpsMemory."""
    from tools.opsmemory.integrations.media_pipeline.client import MediaIngestionClient
    from tools.opsmemory.integrations.media_pipeline.models import OcrExtractionResult

    client = MediaIngestionClient()

    result = OcrExtractionResult(
        text="Change Request #CR-2026-042 approved by engineering review board on 2026-03-14.",
        source_ref="media://documents/cr-2026-042.pdf",
        native_id="cr-2026-042",
        occurred_at="2026-03-14T16:00:00Z",
        page_count=3,
        confidence=0.98,
        doc_format="application/pdf",
    )

    log.info("Ingesting OCR result...")
    response = await client.ingest(result)
    log.info("Ingested OCR. evidence_id=%s", response.get("evidence_id"))
    return response  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Example 3: Batch ingestion
# ---------------------------------------------------------------------------


async def example_batch_ingest() -> None:
    """Ingest multiple evidence items from a single pipeline run."""
    from tools.opsmemory.integrations.media_pipeline.client import MediaIngestionClient
    from tools.opsmemory.integrations.media_pipeline.models import (
        EnrichmentResult,
        MediaMetadataResult,
        TranscriptResult,
    )

    client = MediaIngestionClient()

    items = [
        TranscriptResult(
            text="The rollout completed at 14:30 UTC with zero errors.",
            source_ref="media://recordings/rollout-call.wav",
            native_id="rollout-call-001",
            occurred_at="2026-03-15T14:30:00Z",
            language="en",
        ),
        MediaMetadataResult(
            text="Video: 1920x1080, 30fps, H.264, 5m 12s, 124 MB",
            source_ref="media://videos/rollout-recording.mp4",
            native_id="rollout-recording",
            occurred_at="2026-03-15T14:30:00Z",
            asset_type="video",
            duration_seconds=312.0,
            file_size_bytes=130_023_424,
            mime_type="video/mp4",
        ),
        EnrichmentResult(
            text="Entities: rollout (EVENT), 14:30 UTC (TIME), zero errors (OUTCOME)",
            source_ref="media://recordings/rollout-call.wav",
            native_id="rollout-call-001-entities",
            occurred_at="2026-03-15T14:30:00Z",
            enrichment_type="entity_extraction",
            model="generic-ner-v1",
            labels=["EVENT", "TIME", "OUTCOME"],
        ),
    ]

    log.info("Batch-ingesting %d items...", len(items))
    results = await client.ingest_batch(items)
    log.info("Batch complete. %d/%d items ingested.", len(results), len(items))
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Example 4: Health check
# ---------------------------------------------------------------------------


async def example_health_check() -> None:
    """Check OpsMemory API availability before starting ingestion."""
    from tools.opsmemory.integrations.media_pipeline.client import MediaIngestionClient

    client = MediaIngestionClient()
    healthy = await client.health_check()
    log.info("OpsMemory health check: %s", "OK" if healthy else "UNAVAILABLE")
    return healthy  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    if not _should_run_live_examples():
        log.info("OPSMEMORY_SKIP_LIVE=true — skipping live server examples.")
        log.info(
            "Start OpsMemory (docker compose up) and unset OPSMEMORY_SKIP_LIVE "
            "to run these examples against a real server."
        )
        return

    log.info("=== Health check ===")
    try:
        await example_health_check()
    except Exception as exc:
        log.error("Health check failed: %s", exc)

    log.info("=== Example 1: Ingest transcript ===")
    try:
        await example_ingest_transcript()
    except Exception as exc:
        log.error("Example 1 failed: %s", exc)

    log.info("=== Example 2: Ingest OCR result ===")
    try:
        await example_ingest_ocr()
    except Exception as exc:
        log.error("Example 2 failed: %s", exc)

    log.info("=== Example 3: Batch ingest ===")
    try:
        await example_batch_ingest()
    except Exception as exc:
        log.error("Example 3 failed: %s", exc)


if __name__ == "__main__":
    asyncio.run(main())
