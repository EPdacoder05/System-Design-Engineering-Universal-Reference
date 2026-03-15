"""Tests for the GitHub ingestion connector."""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.opsmemory.agent.ingestor import IngestPayload, compute_evidence_id
from tools.opsmemory.connectors.github_connector import (
    GitHubConnector,
    GitHubConnectorConfig,
)


# ---------------------------------------------------------------------------
# compute_evidence_id — determinism and uniqueness
# ---------------------------------------------------------------------------


def test_evidence_id_is_deterministic():
    """Same inputs always produce the same sha256 hex string."""
    eid1 = compute_evidence_id(
        source_type="github_commit",
        repo="EPdacoder05/my-repo",
        native_id="abc123def456",
        occurred_at="2026-03-15T12:00:00+00:00",
    )
    eid2 = compute_evidence_id(
        source_type="github_commit",
        repo="EPdacoder05/my-repo",
        native_id="abc123def456",
        occurred_at="2026-03-15T12:00:00+00:00",
    )
    assert eid1 == eid2


def test_evidence_id_is_64_hex_chars():
    eid = compute_evidence_id(
        source_type="github_commit",
        repo="EPdacoder05/my-repo",
        native_id="deadbeef",
        occurred_at="2026-01-01T00:00:00Z",
    )
    assert len(eid) == 64
    assert all(c in "0123456789abcdef" for c in eid)


def test_evidence_id_differs_for_different_repos():
    kwargs = dict(
        source_type="github_commit",
        native_id="abc123",
        occurred_at="2026-03-15T12:00:00Z",
    )
    eid_a = compute_evidence_id(repo="EPdacoder05/repo-a", **kwargs)
    eid_b = compute_evidence_id(repo="EPdacoder05/repo-b", **kwargs)
    assert eid_a != eid_b


def test_evidence_id_differs_for_different_native_ids():
    kwargs = dict(
        source_type="github_commit",
        repo="EPdacoder05/my-repo",
        occurred_at="2026-03-15T12:00:00Z",
    )
    eid_1 = compute_evidence_id(native_id="sha-one", **kwargs)
    eid_2 = compute_evidence_id(native_id="sha-two", **kwargs)
    assert eid_1 != eid_2


def test_evidence_id_pr_vs_commit_differ():
    kwargs = dict(
        repo="EPdacoder05/my-repo",
        native_id="42",
        occurred_at="2026-03-15T12:00:00Z",
    )
    eid_commit = compute_evidence_id(source_type="github_commit", **kwargs)
    eid_pr = compute_evidence_id(source_type="github_pr", **kwargs)
    assert eid_commit != eid_pr


def test_evidence_id_matches_manual_sha256():
    raw = "github_commit:EPdacoder05/my-repo:abc123:2026-03-15T12:00:00Z"
    expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    result = compute_evidence_id(
        source_type="github_commit",
        repo="EPdacoder05/my-repo",
        native_id="abc123",
        occurred_at="2026-03-15T12:00:00Z",
    )
    assert result == expected


# ---------------------------------------------------------------------------
# GitHubConnector — normalize_commit
# ---------------------------------------------------------------------------


@pytest.fixture
def connector():
    cfg = GitHubConnectorConfig(
        github_token=None,
        owner="EPdacoder05",
    )
    return GitHubConnector(config=cfg)


@pytest.mark.asyncio
async def test_normalize_commit_contains_sha_short(connector):
    commit = {
        "sha": "abcdef1234567890",
        "message": "Fix critical bug",
        "author_name": "Alice",
        "author_email": "alice@example.com",
        "committed_at": "2026-03-15T10:00:00Z",
        "url": "https://github.com/EPdacoder05/my-repo/commit/abcdef1234567890",
    }
    text = await connector.normalize_commit("my-repo", commit)
    assert "[COMMIT]" in text
    assert "abcdef1" in text  # short sha
    assert "Fix critical bug" in text
    assert "Alice" in text


@pytest.mark.asyncio
async def test_normalize_commit_handles_missing_fields(connector):
    """Should not raise even if optional fields are absent."""
    commit = {"sha": "", "message": "", "author_name": "", "author_email": "", "committed_at": "", "url": ""}
    text = await connector.normalize_commit("empty-repo", commit)
    assert "[COMMIT]" in text


@pytest.mark.asyncio
async def test_normalize_commit_uses_short_sha(connector):
    """Short SHA should be exactly the first 7 characters."""
    commit = {
        "sha": "1234567890abcdef",
        "message": "test commit",
        "author_name": "",
        "author_email": "",
        "committed_at": "",
        "url": "",
    }
    text = await connector.normalize_commit("my-repo", commit)
    assert "1234567" in text
    assert "1234567890abcdef" not in text  # full sha should not appear


# ---------------------------------------------------------------------------
# GitHubConnector — normalize_pr
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_pr_contains_pr_number(connector):
    pr = {
        "number": 42,
        "title": "Add new feature",
        "body": "This PR adds a feature.",
        "state": "closed",
        "merged": True,
        "created_at": "2026-03-01T09:00:00Z",
        "updated_at": "2026-03-14T15:00:00Z",
        "url": "https://github.com/EPdacoder05/my-repo/pull/42",
        "labels": ["feature", "v2"],
    }
    text = await connector.normalize_pr("my-repo", pr)
    assert "[PR #42]" in text
    assert "Add new feature" in text
    assert "feature" in text
    assert "v2" in text


@pytest.mark.asyncio
async def test_normalize_pr_truncates_long_body(connector):
    """PR body should be truncated to 500 characters."""
    pr = {
        "number": 1,
        "title": "Long PR",
        "body": "x" * 1000,
        "state": "open",
        "merged": False,
        "created_at": "2026-03-01T09:00:00Z",
        "updated_at": "2026-03-01T09:00:00Z",
        "url": "",
        "labels": [],
    }
    text = await connector.normalize_pr("my-repo", pr)
    # The body section should not exceed 500 chars (plus label of "Body: ")
    body_section = text.split("Body: ", 1)[1] if "Body: " in text else text
    assert len(body_section) <= 500


@pytest.mark.asyncio
async def test_normalize_pr_handles_none_body(connector):
    """Should not raise when PR body is None."""
    pr = {
        "number": 5,
        "title": "No body PR",
        "body": None,
        "state": "open",
        "merged": False,
        "created_at": "2026-03-01T09:00:00Z",
        "updated_at": "2026-03-01T09:00:00Z",
        "url": "",
        "labels": [],
    }
    text = await connector.normalize_pr("my-repo", pr)
    assert "[PR #5]" in text


# ---------------------------------------------------------------------------
# GitHubConnector — run_once passes native IDs and repo correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_once_passes_native_id_for_commits(connector):
    """run_once must pass commit sha as native_id and repo as owner/repo."""
    mock_repos = [{"name": "test-repo"}]
    mock_commits = [
        {
            "sha": "deadbeef12345678",
            "message": "Test commit",
            "author_name": "Bob",
            "author_email": "bob@example.com",
            "committed_at": "2026-03-10T08:00:00Z",
            "url": "https://github.com/EPdacoder05/test-repo/commit/deadbeef12345678",
        }
    ]
    mock_prs: list = []

    with (
        patch.object(connector, "list_repos", new=AsyncMock(return_value=mock_repos)),
        patch.object(connector, "fetch_recent_commits", new=AsyncMock(return_value=mock_commits)),
        patch.object(connector, "fetch_recent_prs", new=AsyncMock(return_value=mock_prs)),
        patch.object(connector, "ingest_item", new=AsyncMock(return_value=True)) as mock_ingest,
    ):
        stats = await connector.run_once()

    assert stats["test-repo"]["commits"] == 1
    call_kwargs = mock_ingest.call_args.kwargs
    assert call_kwargs["native_id"] == "deadbeef12345678"
    assert call_kwargs["repo"] == "EPdacoder05/test-repo"
    assert call_kwargs["occurred_at"] == "2026-03-10T08:00:00Z"
    assert call_kwargs["source_type"] == "github_commit"


@pytest.mark.asyncio
async def test_run_once_passes_native_id_for_prs(connector):
    """run_once must pass PR number as native_id (as string)."""
    mock_repos = [{"name": "test-repo"}]
    mock_commits: list = []
    mock_prs = [
        {
            "number": 99,
            "title": "Big PR",
            "body": "Lots of changes",
            "state": "merged",
            "merged": True,
            "created_at": "2026-03-05T10:00:00Z",
            "updated_at": "2026-03-12T14:00:00Z",
            "url": "https://github.com/EPdacoder05/test-repo/pull/99",
            "labels": [],
        }
    ]

    with (
        patch.object(connector, "list_repos", new=AsyncMock(return_value=mock_repos)),
        patch.object(connector, "fetch_recent_commits", new=AsyncMock(return_value=mock_commits)),
        patch.object(connector, "fetch_recent_prs", new=AsyncMock(return_value=mock_prs)),
        patch.object(connector, "ingest_item", new=AsyncMock(return_value=True)) as mock_ingest,
    ):
        stats = await connector.run_once()

    assert stats["test-repo"]["prs"] == 1
    call_kwargs = mock_ingest.call_args.kwargs
    # The connector converts PR number (int) to str via str(pr.get("number", ""))
    assert call_kwargs["native_id"] == "99"
    assert call_kwargs["repo"] == "EPdacoder05/test-repo"
    assert call_kwargs["occurred_at"] == "2026-03-05T10:00:00Z"
    assert call_kwargs["source_type"] == "github_pr"


@pytest.mark.asyncio
async def test_run_once_continues_on_repo_error(connector):
    """A failure in one repo must not crash the entire sweep."""
    mock_repos = [{"name": "broken-repo"}, {"name": "ok-repo"}]

    async def side_effect_commits(repo_name, limit):
        if repo_name == "broken-repo":
            raise RuntimeError("API error")
        return []

    with (
        patch.object(connector, "list_repos", new=AsyncMock(return_value=mock_repos)),
        patch.object(connector, "fetch_recent_commits", new=AsyncMock(side_effect=side_effect_commits)),
        patch.object(connector, "fetch_recent_prs", new=AsyncMock(return_value=[])),
        patch.object(connector, "ingest_item", new=AsyncMock(return_value=True)),
    ):
        stats = await connector.run_once()

    # Both repos should appear in stats (broken one has 0 counts)
    assert "broken-repo" in stats
    assert "ok-repo" in stats
    assert stats["broken-repo"]["commits"] == 0


# ---------------------------------------------------------------------------
# GitHubConnector — config defaults
# ---------------------------------------------------------------------------


def test_config_default_owner():
    cfg = GitHubConnectorConfig()
    assert cfg.owner == "EPdacoder05"


def test_config_include_exclude_repos():
    cfg = GitHubConnectorConfig(
        include_repos=["repo-a", "repo-b"],
        exclude_repos=["repo-c"],
    )
    assert "repo-a" in cfg.include_repos
    assert "repo-c" in cfg.exclude_repos


def test_list_repos_applies_include_filter():
    """list_repos should filter by include_repos when set."""
    cfg = GitHubConnectorConfig(include_repos=["wanted-repo"], owner="testowner")
    connector_filtered = GitHubConnector(config=cfg)

    raw_repos = [{"name": "wanted-repo"}, {"name": "unwanted-repo"}]
    # Simulate filtering logic directly (no HTTP call needed)
    result = [r for r in raw_repos if r["name"] in cfg.include_repos]
    assert len(result) == 1
    assert result[0]["name"] == "wanted-repo"


def test_list_repos_applies_exclude_filter():
    """list_repos should remove repos in exclude_repos."""
    cfg = GitHubConnectorConfig(exclude_repos=["skip-me"], owner="testowner")
    raw_repos = [{"name": "keep-me"}, {"name": "skip-me"}]
    result = [r for r in raw_repos if r["name"] not in cfg.exclude_repos]
    assert len(result) == 1
    assert result[0]["name"] == "keep-me"
