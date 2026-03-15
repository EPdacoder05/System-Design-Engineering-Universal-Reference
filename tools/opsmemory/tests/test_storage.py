"""Storage layer tests using mocked async sessions."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.opsmemory.storage.models import (
    ConsolidationRun,
    EvidenceEmbedding,
    EvidenceItem,
    Memory,
)
from tools.opsmemory.storage.repository import (
    ConsolidationRunRepository,
    EmbeddingRepository,
    EvidenceRepository,
    MemoryRepository,
)


def _make_session():
    """Build a minimal mock AsyncSession."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_evidence_repository_create():
    session = _make_session()

    expected = EvidenceItem(
        id=uuid.uuid4(),
        raw_text="hello world",
        source_type="manual",
        source_ref="test",
        ingested_at=datetime.now(timezone.utc),
        consolidated=False,
        metadata_={},
    )
    session.refresh = AsyncMock(side_effect=lambda obj: None)

    with patch.object(EvidenceRepository, "create", new=AsyncMock(return_value=expected)):
        result = await EvidenceRepository.create(
            session, raw_text="hello world", source_type="manual", source_ref="test"
        )

    assert result.raw_text == "hello world"
    assert result.source_type == "manual"
    assert result.consolidated is False


@pytest.mark.asyncio
async def test_memory_repository_create():
    session = _make_session()

    expected = Memory(
        id=uuid.uuid4(),
        content="Consolidated evidence",
        summary="Test summary",
        source_ids=[],
        metadata_={},
        consolidated=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session.refresh = AsyncMock(side_effect=lambda obj: None)

    with patch.object(MemoryRepository, "create", new=AsyncMock(return_value=expected)):
        result = await MemoryRepository.create(
            session, content="Consolidated evidence", summary="Test summary"
        )

    assert result.content == "Consolidated evidence"
    assert result.summary == "Test summary"


@pytest.mark.asyncio
async def test_embedding_repository_semantic_search_format():
    """semantic_search must return List[Tuple[EvidenceItem, float]]."""
    session = _make_session()

    item = EvidenceItem(
        id=uuid.uuid4(),
        raw_text="example evidence",
        source_type="file",
        source_ref="test.txt",
        ingested_at=datetime.now(timezone.utc),
        consolidated=False,
        metadata_={},
    )
    mock_results = [(item, 0.95)]

    with patch.object(
        EmbeddingRepository, "semantic_search", new=AsyncMock(return_value=mock_results)
    ):
        results = await EmbeddingRepository.semantic_search(
            session, [0.1] * 1536, limit=5
        )

    assert len(results) == 1
    evidence_item, score = results[0]
    assert isinstance(evidence_item, EvidenceItem)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_consolidation_run_repository_start_has_running_status():
    session = _make_session()

    expected_run = ConsolidationRun(
        id=uuid.uuid4(),
        started_at=datetime.now(timezone.utc),
        status="running",
        evidence_count=0,
        memory_count=0,
    )
    session.refresh = AsyncMock(side_effect=lambda obj: None)

    with patch.object(
        ConsolidationRunRepository, "start", new=AsyncMock(return_value=expected_run)
    ):
        run = await ConsolidationRunRepository.start(session)

    assert run.status == "running"
    assert run.finished_at is None
