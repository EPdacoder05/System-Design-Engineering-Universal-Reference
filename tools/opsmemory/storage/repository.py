"""Repository / DAO layer for OpsMemory storage models."""

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from tools.opsmemory.storage.models import (
    ConsolidationRun,
    EvidenceEmbedding,
    EvidenceItem,
    Memory,
    Source,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EvidenceRepository:
    """CRUD operations for EvidenceItem records."""

    @staticmethod
    async def create(
        session: AsyncSession,
        raw_text: str,
        source_type: str,
        source_ref: str,
        author: Optional[str] = None,
        event_ts: Optional[datetime] = None,
        metadata: Optional[dict] = None,
        correlation_id: Optional[str] = None,
    ) -> EvidenceItem:
        item = EvidenceItem(
            raw_text=raw_text,
            source_type=source_type,
            source_ref=source_ref,
            author=author,
            event_ts=event_ts,
            metadata_=metadata or {},
            correlation_id=correlation_id,
        )
        session.add(item)
        await session.flush()
        await session.refresh(item)
        return item

    @staticmethod
    async def list_unconsolidated(
        session: AsyncSession, limit: int = 100
    ) -> List[EvidenceItem]:
        result = await session.execute(
            select(EvidenceItem)
            .where(EvidenceItem.consolidated.is_(False))
            .order_by(EvidenceItem.ingested_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def mark_consolidated(
        session: AsyncSession, ids: List[uuid.UUID]
    ) -> None:
        if not ids:
            return
        items = await session.execute(
            select(EvidenceItem).where(EvidenceItem.id.in_(ids))
        )
        for item in items.scalars().all():
            item.consolidated = True
        await session.flush()

    @staticmethod
    async def get(
        session: AsyncSession, evidence_id: uuid.UUID
    ) -> Optional[EvidenceItem]:
        result = await session.execute(
            select(EvidenceItem).where(EvidenceItem.id == evidence_id)
        )
        return result.scalar_one_or_none()


class EmbeddingRepository:
    """Operations for EvidenceEmbedding records including semantic search."""

    @staticmethod
    async def upsert(
        session: AsyncSession,
        evidence_id: uuid.UUID,
        embedding: List[float],
        model_name: str = "mock",
    ) -> EvidenceEmbedding:
        # Delete existing embedding for this evidence item then insert fresh.
        await session.execute(
            delete(EvidenceEmbedding).where(
                EvidenceEmbedding.evidence_id == evidence_id
            )
        )
        record = EvidenceEmbedding(
            evidence_id=evidence_id,
            embedding=embedding,
            model_name=model_name,
        )
        session.add(record)
        await session.flush()
        await session.refresh(record)
        return record

    @staticmethod
    async def semantic_search(
        session: AsyncSession,
        query_embedding: List[float],
        limit: int = 10,
    ) -> List[Tuple[EvidenceItem, float]]:
        distance_col = EvidenceEmbedding.embedding.cosine_distance(
            query_embedding
        ).label("distance")
        result = await session.execute(
            select(EvidenceItem, distance_col)
            .join(EvidenceEmbedding, EvidenceEmbedding.evidence_id == EvidenceItem.id)
            .order_by(distance_col.asc())
            .limit(limit)
        )
        rows = result.all()
        return [(row[0], float(1.0 - row[1])) for row in rows]


class MemoryRepository:
    """CRUD operations for consolidated Memory records."""

    @staticmethod
    async def create(
        session: AsyncSession,
        content: str,
        summary: Optional[str] = None,
        source_ids: Optional[List[uuid.UUID]] = None,
        metadata: Optional[dict] = None,
    ) -> Memory:
        memory = Memory(
            content=content,
            summary=summary,
            source_ids=source_ids or [],
            metadata_=metadata or {},
        )
        session.add(memory)
        await session.flush()
        await session.refresh(memory)
        return memory

    @staticmethod
    async def list_all(session: AsyncSession, limit: int = 50) -> List[Memory]:
        result = await session.execute(
            select(Memory).order_by(Memory.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())


class ConsolidationRunRepository:
    """Audit-log operations for consolidation cycle records."""

    @staticmethod
    async def start(session: AsyncSession) -> ConsolidationRun:
        run = ConsolidationRun(status="running")
        session.add(run)
        await session.flush()
        await session.refresh(run)
        return run

    @staticmethod
    async def finish(
        session: AsyncSession,
        run_id: uuid.UUID,
        memory_count: int,
        evidence_count: int,
        error: Optional[str] = None,
    ) -> ConsolidationRun:
        result = await session.execute(
            select(ConsolidationRun).where(ConsolidationRun.id == run_id)
        )
        run = result.scalar_one()
        run.finished_at = _utcnow()
        run.memory_count = memory_count
        run.evidence_count = evidence_count
        run.status = "error" if error else "completed"
        run.error = error
        await session.flush()
        await session.refresh(run)
        return run


class SourceRepository:
    """CRUD operations for registered data sources."""

    @staticmethod
    async def upsert(
        session: AsyncSession,
        owner: str,
        repo: str,
        source_type: str = "github",
        metadata: Optional[dict] = None,
    ) -> Source:
        result = await session.execute(
            select(Source).where(
                Source.owner == owner,
                Source.repo == repo,
                Source.source_type == source_type,
            )
        )
        source = result.scalar_one_or_none()
        if source is None:
            source = Source(
                owner=owner,
                repo=repo,
                source_type=source_type,
                metadata_=metadata or {},
            )
            session.add(source)
        elif metadata is not None:
            source.metadata_ = metadata
        await session.flush()
        await session.refresh(source)
        return source

    @staticmethod
    async def list_all(session: AsyncSession) -> List[Source]:
        result = await session.execute(
            select(Source).order_by(Source.created_at.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def update_fetched(
        session: AsyncSession, source_id: uuid.UUID
    ) -> None:
        result = await session.execute(
            select(Source).where(Source.id == source_id)
        )
        source = result.scalar_one_or_none()
        if source is not None:
            source.last_fetched_at = _utcnow()
            await session.flush()
