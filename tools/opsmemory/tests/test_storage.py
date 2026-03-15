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


# ---------------------------------------------------------------------------
# EvidenceRepository — delete operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evidence_delete_by_id_returns_true_when_found():
    """delete_by_id returns True when the record exists and is deleted."""
    session = _make_session()
    item_id = uuid.uuid4()
    item = EvidenceItem(id=item_id, raw_text="test", source_type="manual", source_ref="")

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = item
    session.execute = AsyncMock(return_value=mock_result)
    session.delete = AsyncMock()

    result = await EvidenceRepository.delete_by_id(session, item_id)

    assert result is True
    session.delete.assert_called_once_with(item)
    session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_evidence_delete_by_id_returns_false_when_not_found():
    """delete_by_id returns False when no record matches the given UUID."""
    session = _make_session()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=mock_result)
    session.delete = AsyncMock()

    result = await EvidenceRepository.delete_by_id(session, uuid.uuid4())

    assert result is False
    session.delete.assert_not_called()


@pytest.mark.asyncio
async def test_evidence_delete_all_returns_count():
    """delete_all returns the number of records that were present before deletion."""
    session = _make_session()

    count_result = MagicMock()
    count_result.scalar_one.return_value = 7
    # delete() call returns a mock result that is not used
    delete_result = MagicMock()
    session.execute = AsyncMock(side_effect=[count_result, delete_result])

    count = await EvidenceRepository.delete_all(session)

    assert count == 7
    session.flush.assert_called_once()


# ---------------------------------------------------------------------------
# MemoryRepository — delete operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_delete_by_id_returns_true_when_found():
    """delete_by_id returns True when the Memory record exists."""
    session = _make_session()
    mem_id = uuid.uuid4()
    memory = Memory(id=mem_id, content="test", source_ids=[], metadata_={})

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = memory
    session.execute = AsyncMock(return_value=mock_result)
    session.delete = AsyncMock()

    result = await MemoryRepository.delete_by_id(session, mem_id)

    assert result is True
    session.delete.assert_called_once_with(memory)
    session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_memory_delete_by_id_returns_false_when_not_found():
    """delete_by_id returns False when no Memory matches the UUID."""
    session = _make_session()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=mock_result)
    session.delete = AsyncMock()

    result = await MemoryRepository.delete_by_id(session, uuid.uuid4())

    assert result is False
    session.delete.assert_not_called()


@pytest.mark.asyncio
async def test_memory_delete_all_returns_count():
    """delete_all returns the number of Memory records deleted."""
    session = _make_session()

    count_result = MagicMock()
    count_result.scalar_one.return_value = 3
    delete_result = MagicMock()
    session.execute = AsyncMock(side_effect=[count_result, delete_result])

    count = await MemoryRepository.delete_all(session)

    assert count == 3
    session.flush.assert_called_once()
