"""Tests for jarvis_lint.

Validates:
- Docker containers referenced in wiki are checked against live state.
- Absolute paths referenced in wiki are verified for existence.
- Raw backlog count is accurate.
- Missing Docker returns drift entries (not a crash).
- Result always includes required keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_vault(tmp_path: Path, wiki_content: str = "") -> Path:
    vault = tmp_path / "vault"
    wiki = vault / "wiki"
    wiki.mkdir(parents=True)
    (vault / "raw" / "ssd_transfers").mkdir(parents=True)

    default = (
        "---\ntitle: Storage\nupdated: 2026-01-01T00:00:00Z\ntags: [storage]\n---\n\n"
        "# Storage Dashboard\n\n"
    )
    (wiki / "storage_dashboard.md").write_text(
        wiki_content or default, encoding="utf-8"
    )
    return vault


@pytest.mark.asyncio
async def test_lint_returns_required_keys(tmp_path):
    """jarvis_lint always returns timestamp, passing, drift, raw_backlog, docker_available."""
    from tools.jarvis.mcp.tools.lint import jarvis_lint

    vault = _make_vault(tmp_path)
    result = await jarvis_lint(vault_path=str(vault))

    assert "timestamp" in result
    assert "passing" in result
    assert "drift" in result
    assert "raw_backlog" in result
    assert "docker_available" in result
    assert isinstance(result["passing"], list)
    assert isinstance(result["drift"], list)
    assert isinstance(result["raw_backlog"], int)


@pytest.mark.asyncio
async def test_lint_file_path_exists(tmp_path):
    """A real file path referenced in wiki appears in passing."""
    from tools.jarvis.mcp.tools.lint import jarvis_lint

    real_file = tmp_path / "real_file.txt"
    real_file.write_text("exists")

    wiki_content = (
        "---\ntitle: Storage\nupdated: 2026-01-01T00:00:00Z\ntags: []\n---\n\n"
        f"See `{real_file}` for details.\n"
    )
    vault = _make_vault(tmp_path, wiki_content)
    result = await jarvis_lint(vault_path=str(vault))

    assert any(str(real_file) in item for item in result["passing"])


@pytest.mark.asyncio
async def test_lint_missing_file_path_is_drift(tmp_path):
    """A non-existent absolute path referenced in wiki appears in drift."""
    from tools.jarvis.mcp.tools.lint import jarvis_lint

    missing = "/nonexistent/path/that/does/not/exist.txt"
    wiki_content = (
        "---\ntitle: Storage\nupdated: 2026-01-01T00:00:00Z\ntags: []\n---\n\n"
        f"Config at `{missing}`.\n"
    )
    vault = _make_vault(tmp_path, wiki_content)
    result = await jarvis_lint(vault_path=str(vault))

    assert any(missing in item for item in result["drift"])


@pytest.mark.asyncio
async def test_lint_running_docker_container_is_passing(tmp_path):
    """A service name matching a running container appears in passing."""
    from tools.jarvis.mcp.tools.lint import jarvis_lint

    wiki_content = (
        "---\ntitle: Storage\nupdated: 2026-01-01T00:00:00Z\ntags: []\n---\n\n"
        "Service `plex` is running.\n"
    )
    vault = _make_vault(tmp_path, wiki_content)

    with patch(
        "tools.jarvis.mcp.tools.lint._running_docker_containers",
        return_value={"plex", "homeassistant"},
    ):
        result = await jarvis_lint(vault_path=str(vault))

    assert any("plex" in item and "running" in item for item in result["passing"])


@pytest.mark.asyncio
async def test_lint_stopped_docker_container_is_drift(tmp_path):
    """A service name NOT in running containers appears in drift."""
    from tools.jarvis.mcp.tools.lint import jarvis_lint

    wiki_content = (
        "---\ntitle: Storage\nupdated: 2026-01-01T00:00:00Z\ntags: []\n---\n\n"
        "Service `plex` handles media.\n"
    )
    vault = _make_vault(tmp_path, wiki_content)

    with patch(
        "tools.jarvis.mcp.tools.lint._running_docker_containers",
        return_value=set(),  # nothing running
    ), patch(
        "tools.jarvis.mcp.tools.lint._docker_is_installed",
        return_value=True,
    ):
        result = await jarvis_lint(vault_path=str(vault))

    assert any("plex" in item and "not running" in item for item in result["drift"])


@pytest.mark.asyncio
async def test_lint_raw_backlog_counts_unprocessed(tmp_path):
    """raw_backlog equals the number of JSON files not in the ingest registry."""
    from tools.jarvis.mcp.tools.lint import jarvis_lint

    vault = _make_vault(tmp_path)
    # Put 3 raw JSON files, but only register 1 of them
    for i in range(3):
        raw_file = vault / "raw" / "ssd_transfers" / f"transfer_{i:03d}.json"
        raw_file.write_text(json.dumps({"file_count": i}))

    registry_path = vault / ".ingest_registry.json"
    first_file = vault / "raw" / "ssd_transfers" / "transfer_000.json"
    mtime = int(first_file.stat().st_mtime)
    registry_path.write_text(
        json.dumps({f"transfer_000.json:{mtime}": "2026-05-24T00:00:00Z"}),
        encoding="utf-8",
    )

    result = await jarvis_lint(vault_path=str(vault))

    assert result["raw_backlog"] == 2


@pytest.mark.asyncio
async def test_lint_docker_unavailable_reports_drift(tmp_path):
    """If Docker is not installed, checked service names land in drift (not crash)."""
    from tools.jarvis.mcp.tools.lint import jarvis_lint

    wiki_content = (
        "---\ntitle: Storage\nupdated: 2026-01-01T00:00:00Z\ntags: []\n---\n\n"
        "Service `myservice` is running.\n"
    )
    vault = _make_vault(tmp_path, wiki_content)

    with patch(
        "tools.jarvis.mcp.tools.lint._running_docker_containers",
        return_value=set(),
    ), patch(
        "tools.jarvis.mcp.tools.lint._docker_is_installed",
        return_value=False,
    ):
        result = await jarvis_lint(vault_path=str(vault))

    assert any("myservice" in item for item in result["drift"])
    assert result["docker_available"] is False
