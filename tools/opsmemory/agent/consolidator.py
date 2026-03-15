"""Consolidation logic — batches evidence into memories on a timer."""

import asyncio
from dataclasses import dataclass
from typing import List

import numpy as np
import structlog

from tools.opsmemory.storage.models import ConsolidationRun
from tools.opsmemory.storage.repository import (
    ConsolidationRunRepository,
    EmbeddingRepository,
    EvidenceRepository,
    MemoryRepository,
)

log = structlog.get_logger(__name__)


@dataclass
class ConsolidationConfig:
    """Tuning parameters for the consolidation loop."""

    interval_seconds: int = 1800
    batch_size: int = 100
    embedding_dim: int = 1536


def generate_embedding(text: str, dim: int = 1536) -> List[float]:
    """Generate a deterministic mock embedding vector.

    NOTE: This is a mock implementation using seeded random numbers.
    Replace with a call to a real embedding API (e.g. OpenAI
    ``text-embedding-3-small``) before deploying to production.
    """
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.random(dim).tolist()


async def consolidate(session, config: ConsolidationConfig) -> ConsolidationRun:
    """Run a single consolidation cycle and return the audit record.

    Steps:
    1. Open a ConsolidationRun.
    2. Load up to *batch_size* unconsolidated evidence items.
    3. Upsert an embedding for each item.
    4. Group items into chunks of ~10 and create a Memory per chunk.
    5. Mark all processed evidence as consolidated.
    6. Close the ConsolidationRun.
    """
    run = await ConsolidationRunRepository.start(session)
    log.info("consolidation_started", run_id=str(run.id))

    try:
        items = await EvidenceRepository.list_unconsolidated(
            session, limit=config.batch_size
        )
        evidence_count = len(items)

        for item in items:
            vec = generate_embedding(item.raw_text, dim=config.embedding_dim)
            await EmbeddingRepository.upsert(session, item.id, vec)

        # Group into chunks of 10 and create one Memory per chunk.
        chunk_size = 10
        memory_count = 0
        for i in range(0, len(items), chunk_size):
            chunk = items[i : i + chunk_size]
            if not chunk:
                break
            content = "\n---\n".join(
                f"[{e.source_type}] {e.source_ref}\n{e.raw_text[:500]}"
                for e in chunk
            )
            summary = f"Consolidated {len(chunk)} items from sources: " + ", ".join(
                {e.source_ref for e in chunk if e.source_ref}
            )
            await MemoryRepository.create(
                session,
                content=content,
                summary=summary,
                source_ids=[e.id for e in chunk],
            )
            memory_count += 1

        await EvidenceRepository.mark_consolidated(session, [e.id for e in items])

        run = await ConsolidationRunRepository.finish(
            session,
            run.id,
            memory_count=memory_count,
            evidence_count=evidence_count,
        )
        log.info(
            "consolidation_finished",
            run_id=str(run.id),
            evidence_count=evidence_count,
            memory_count=memory_count,
        )
    except Exception as exc:
        log.error("consolidation_error", run_id=str(run.id), error=str(exc))
        run = await ConsolidationRunRepository.finish(
            session, run.id, memory_count=0, evidence_count=0, error=str(exc)
        )
        raise

    return run


async def run_consolidation_loop(session_factory, config: ConsolidationConfig) -> None:
    """Run ``consolidate`` every *interval_seconds*, forever.

    Exceptions are caught and logged so the loop never crashes the process.
    """
    log.info(
        "consolidation_loop_started", interval_seconds=config.interval_seconds
    )
    while True:
        try:
            async with session_factory() as session:
                await consolidate(session, config)
                await session.commit()
        except Exception as exc:
            log.error("consolidation_loop_error", error=str(exc))
        await asyncio.sleep(config.interval_seconds)
