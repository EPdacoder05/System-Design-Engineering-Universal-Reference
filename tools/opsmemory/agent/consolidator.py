"""Consolidation logic — batches evidence into memories on a timer."""

import asyncio
import os
from dataclasses import dataclass
from typing import List, Optional

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
    """Generate an embedding vector for *text*.

    Provider selection
    ------------------
    - When ``OPSMEMORY_EMBEDDING_PROVIDER`` is set to ``"litellm"`` (or another
      real provider), this function delegates to the configured provider and
      returns its embedding synchronously via ``asyncio.run``.
    - In all other cases (default / ``"mock"``), a deterministic seeded
      random vector is returned so that tests and local dev work without any
      API credentials.

    The *dim* parameter is honoured for the mock path.  Real providers return
    vectors at their native dimension (see ``model_registry.yaml``); in that
    case the *dim* argument is silently ignored.
    """
    provider_name = os.environ.get("OPSMEMORY_EMBEDDING_PROVIDER", "mock").lower()

    if provider_name != "mock":
        try:
            from tools.opsmemory.providers import get_embedding_provider

            provider = get_embedding_provider(provider_name=provider_name)

            async def _embed() -> List[float]:
                return await provider.embed_as_list(text)

            try:
                # If we are inside a running event loop (async context), create
                # a new thread to run the coroutine so we don't block the loop.
                import concurrent.futures

                loop = asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, _embed())
                    return future.result()
            except RuntimeError:
                # No running event loop — safe to use asyncio.run directly.
                return asyncio.run(_embed())
        except Exception as exc:
            log.warning(
                "generate_embedding_provider_failed_falling_back",
                provider=provider_name,
                error=str(exc),
            )
            # Fall through to deterministic mock on failure.

    # Deterministic mock — no external calls.
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
