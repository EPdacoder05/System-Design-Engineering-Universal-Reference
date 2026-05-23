"""Tests for OpsMemory MCP tool functions.

Validates that MCP tool functions are importable, have the correct signatures,
and behave correctly when the underlying API is mocked.  Does not require a
running OpsMemory server or database.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Server — tool registration
# ---------------------------------------------------------------------------


def test_mcp_server_imports():
    """The FastMCP server module must import without error."""
    import tools.opsmemory.mcp.server as server  # noqa: F401

    assert server.mcp is not None


def test_mcp_server_has_expected_tools():
    """All expected tool names must be registered on the FastMCP instance."""
    import asyncio

    from tools.opsmemory.mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    tool_names = {t.name for t in tools}
    expected = {
        "memory_ingest_text",
        "memory_ingest_repo",
        "memory_query",
        "memory_status",
        "memory_consolidate",
        "memory_list_sources",
        "memory_sync_github_owner",
        "memory_delete_memory",
        "memory_delete_evidence",
    }
    assert expected.issubset(tool_names), (
        f"Missing tools: {expected - tool_names}"
    )


# ---------------------------------------------------------------------------
# memory_query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_query_calls_api():
    """memory_query should GET /query and return the parsed JSON."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "query": "deployments",
        "answer": "Deployed v2.3.1",
        "citations": [],
        "memories": [],
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        from tools.opsmemory.mcp.tools.query import memory_query

        result = await memory_query("deployments", limit=5)

    assert result["query"] == "deployments"
    assert result["answer"] == "Deployed v2.3.1"


@pytest.mark.asyncio
async def test_memory_query_clamps_limit():
    """limit should be clamped to [1, 50]."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"query": "x", "answer": "", "citations": [], "memories": []}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        from tools.opsmemory.mcp.tools.query import memory_query

        await memory_query("test", limit=999)

    call_params = mock_client.get.call_args.kwargs["params"]
    assert call_params["limit"] == 50


# ---------------------------------------------------------------------------
# memory_ingest_text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_ingest_text_posts_payload():
    """memory_ingest_text should POST /ingest with the expected payload."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "evidence_id": "abc123",
        "redacted": False,
        "correlation_id": "corr-1",
        "excerpt": "test",
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        from tools.opsmemory.mcp.tools.ingest import memory_ingest_text

        result = await memory_ingest_text(
            text="Deployed v2.3.1",
            source_type="manual",
            author="alice",
        )

    assert result["evidence_id"] == "abc123"
    posted = mock_client.post.call_args.kwargs["json"]
    assert posted["text"] == "Deployed v2.3.1"
    assert posted["source_type"] == "manual"
    assert posted["author"] == "alice"


# ---------------------------------------------------------------------------
# memory_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_status_returns_counts():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "evidence_total": 42,
        "evidence_unconsolidated": 5,
        "memories": 7,
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        from tools.opsmemory.mcp.tools.admin import memory_status

        result = await memory_status()

    assert result["evidence_total"] == 42
    assert result["memories"] == 7


# ---------------------------------------------------------------------------
# memory_consolidate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_consolidate_posts_and_returns_run():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "run_id": "run-uuid",
        "memories_created": 3,
        "evidence_consolidated": 30,
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        from tools.opsmemory.mcp.tools.admin import memory_consolidate

        result = await memory_consolidate()

    assert result["run_id"] == "run-uuid"
    assert result["memories_created"] == 3


# ---------------------------------------------------------------------------
# memory_delete_memory / memory_delete_evidence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_delete_memory():
    mock_response = MagicMock()
    mock_response.json.return_value = {"deleted": "mem-uuid"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.delete = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        from tools.opsmemory.mcp.tools.admin import memory_delete_memory

        result = await memory_delete_memory("mem-uuid")

    assert result["deleted"] == "mem-uuid"
    url = mock_client.delete.call_args.args[0]
    assert "mem-uuid" in url


@pytest.mark.asyncio
async def test_memory_delete_evidence():
    mock_response = MagicMock()
    mock_response.json.return_value = {"deleted": "ev-uuid"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.delete = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        from tools.opsmemory.mcp.tools.admin import memory_delete_evidence

        result = await memory_delete_evidence("ev-uuid")

    assert result["deleted"] == "ev-uuid"


# ---------------------------------------------------------------------------
# memory_list_sources
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_list_sources():
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"id": "src-1", "owner": "alice", "repo": "my-repo", "source_type": "github_commit"}
    ]
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        from tools.opsmemory.mcp.tools.admin import memory_list_sources

        result = await memory_list_sources()

    assert len(result) == 1
    assert result[0]["owner"] == "alice"


# ---------------------------------------------------------------------------
# memory_sync_github_owner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_sync_github_owner_delegates_to_connector():
    """memory_sync_github_owner should call GitHubConnector.run_once."""
    from tools.opsmemory.mcp.tools.github import memory_sync_github_owner

    mock_stats = {"my-repo": {"commits": 5, "prs": 2}}

    with (
        patch(
            "tools.opsmemory.connectors.github_connector.GitHubConnector.resolve_owner",
            new=AsyncMock(return_value="resolved-user"),
        ),
        patch(
            "tools.opsmemory.connectors.github_connector.GitHubConnector.run_once",
            new=AsyncMock(return_value=mock_stats),
        ),
    ):
        result = await memory_sync_github_owner(owner="resolved-user")

    assert result["owner"] == "resolved-user"
    assert result["repos"] == mock_stats


# ---------------------------------------------------------------------------
# memory_ingest_repo
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_ingest_repo_delegates_to_connector():
    """memory_ingest_repo should call RepoConnector.run_once and merge stats."""
    from tools.opsmemory.mcp.tools.repo import memory_ingest_repo

    mock_stats = {"files_ingested": 12, "chunks_posted": 15, "files_skipped": 1}

    with patch(
        "tools.opsmemory.connectors.repo_connector.RepoConnector.run_once",
        new=AsyncMock(return_value=mock_stats),
    ):
        result = await memory_ingest_repo(
            repo_path="/tmp",
            repo="EPdacoder05/System-Design-Engineering-Universal-Reference",
        )

    assert result["repo"] == "EPdacoder05/System-Design-Engineering-Universal-Reference"
    assert result["files_ingested"] == 12
    assert result["chunks_posted"] == 15
    assert result["files_skipped"] == 1


# ---------------------------------------------------------------------------
# RepoConnector unit tests
# ---------------------------------------------------------------------------


def test_repo_connector_chunk_text_short():
    """Short text is returned as a single chunk."""
    from tools.opsmemory.connectors.repo_connector import RepoConnector

    text = "hello world"
    assert RepoConnector._chunk_text(text) == [text]


def test_repo_connector_chunk_text_long():
    """Text longer than _CHUNK_CHARS is split into overlapping chunks."""
    from tools.opsmemory.connectors.repo_connector import (
        RepoConnector,
        _CHUNK_CHARS,
        _CHUNK_OVERLAP_CHARS,
    )

    text = "x" * (_CHUNK_CHARS + 500)
    chunks = RepoConnector._chunk_text(text)
    assert len(chunks) > 1
    # Each chunk is at most _CHUNK_CHARS characters.
    for chunk in chunks:
        assert len(chunk) <= _CHUNK_CHARS
    # Chunks overlap: the tail of chunk n equals the head of chunk n+1.
    assert chunks[0][-_CHUNK_OVERLAP_CHARS:] == chunks[1][:_CHUNK_OVERLAP_CHARS]


def test_repo_connector_iter_files_skips_git(tmp_path):
    """_iter_files must not descend into .git or other skip-listed dirs."""
    from tools.opsmemory.connectors.repo_connector import RepoConnector, RepoConnectorConfig

    # Create a .git directory with a markdown file inside.
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "COMMIT_EDITMSG").write_text("a commit message")
    (git_dir / "README.md").write_text("should be skipped")

    # Create a normal markdown file at the root.
    (tmp_path / "README.md").write_text("# Real readme")

    config = RepoConnectorConfig(repo_path=str(tmp_path))
    connector = RepoConnector(config=config)
    found = connector._iter_files()

    paths = [p.name for p in found]
    assert "README.md" in paths
    # No file from inside .git should appear.
    for p in found:
        assert ".git" not in p.parts


def test_repo_connector_iter_files_filters_extensions(tmp_path):
    """_iter_files only returns files whose extension is in file_extensions."""
    from tools.opsmemory.connectors.repo_connector import RepoConnector, RepoConnectorConfig

    (tmp_path / "doc.md").write_text("# Doc")
    (tmp_path / "config.yaml").write_text("key: value")
    (tmp_path / "binary.exe").write_bytes(b"\x00\x01")

    config = RepoConnectorConfig(
        repo_path=str(tmp_path), file_extensions=[".md", ".yaml"]
    )
    connector = RepoConnector(config=config)
    found = [p.name for p in connector._iter_files()]

    assert "doc.md" in found
    assert "config.yaml" in found
    assert "binary.exe" not in found


@pytest.mark.asyncio
async def test_repo_connector_ingest_file_posts_payload(tmp_path):
    """ingest_file should POST a correctly shaped payload for a small file."""
    from tools.opsmemory.connectors.repo_connector import RepoConnector, RepoConnectorConfig

    doc = tmp_path / "guide.md"
    doc.write_text("# Best Practices\n\nAlways run as non-root.")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    config = RepoConnectorConfig(
        repo_path=str(tmp_path),
        repo="owner/ref-repo",
        ingest_url="http://localhost:8000/ingest",
    )
    connector = RepoConnector(config=config)

    with patch("httpx.AsyncClient", return_value=mock_client):
        n_chunks = await connector.ingest_file(doc, tmp_path)

    assert n_chunks == 1
    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["source_type"] == "reference_doc"
    assert "guide.md" in payload["native_id"]
    assert payload["repo"] == "owner/ref-repo"
    assert "Best Practices" in payload["text"]
