"""Tests for jarvis_log.

Validates:
- A note is appended to journal.md with correct timestamp format.
- The frontmatter updated: field is refreshed.
- Empty / whitespace-only notes are rejected.
- Missing journal.md returns an error (no crash).
- Re-calling log appends a second entry (not overwriting the first).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _make_vault_with_journal(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    wiki = vault / "wiki"
    wiki.mkdir(parents=True)

    (wiki / "journal.md").write_text(
        "---\ntitle: Journal\nupdated: 2026-01-01T00:00:00Z\ntags: [journal]\n---\n\n"
        "# Journal\n\n## Entries\n\n",
        encoding="utf-8",
    )
    return vault


@pytest.mark.asyncio
async def test_log_appends_note_to_journal(tmp_path):
    """A valid note is appended to journal.md with ISO-8601 timestamp heading."""
    from tools.jarvis.mcp.tools.log import jarvis_log

    vault = _make_vault_with_journal(tmp_path)
    result = await jarvis_log("Working on SSD offload pipeline.", vault_path=str(vault))

    assert result["logged"] is True
    assert result["timestamp"]

    content = (vault / "wiki" / "journal.md").read_text()
    assert "Working on SSD offload pipeline." in content
    # Timestamp heading present
    assert "###" in content
    # ISO-8601 format: YYYY-MM-DDTHH:MM:SSZ
    assert "T" in result["timestamp"] and result["timestamp"].endswith("Z")


@pytest.mark.asyncio
async def test_log_updates_frontmatter_updated_field(tmp_path):
    """After logging, the frontmatter updated: field reflects the new timestamp."""
    from tools.jarvis.mcp.tools.log import jarvis_log

    vault = _make_vault_with_journal(tmp_path)
    result = await jarvis_log("Architecture note.", vault_path=str(vault))

    content = (vault / "wiki" / "journal.md").read_text()
    assert f"updated: {result['timestamp']}" in content
    # Original stub timestamp should be replaced
    assert "2026-01-01T00:00:00Z" not in content


@pytest.mark.asyncio
async def test_log_empty_note_rejected(tmp_path):
    """An empty note returns logged=False with an error message."""
    from tools.jarvis.mcp.tools.log import jarvis_log

    vault = _make_vault_with_journal(tmp_path)
    result = await jarvis_log("", vault_path=str(vault))

    assert result["logged"] is False
    assert "error" in result


@pytest.mark.asyncio
async def test_log_whitespace_only_note_rejected(tmp_path):
    """A whitespace-only note is also rejected."""
    from tools.jarvis.mcp.tools.log import jarvis_log

    vault = _make_vault_with_journal(tmp_path)
    result = await jarvis_log("   \n  ", vault_path=str(vault))

    assert result["logged"] is False


@pytest.mark.asyncio
async def test_log_missing_journal_returns_error(tmp_path):
    """If journal.md does not exist, logged=False and no crash."""
    from tools.jarvis.mcp.tools.log import jarvis_log

    vault = tmp_path / "vault"
    (vault / "wiki").mkdir(parents=True)
    # journal.md deliberately not created

    result = await jarvis_log("A note.", vault_path=str(vault))

    assert result["logged"] is False
    assert "journal.md" in result.get("error", "")


@pytest.mark.asyncio
async def test_log_multiple_entries_accumulate(tmp_path):
    """Multiple log calls append distinct entries without overwriting each other."""
    from tools.jarvis.mcp.tools.log import jarvis_log

    vault = _make_vault_with_journal(tmp_path)

    await jarvis_log("First note.", vault_path=str(vault))
    await jarvis_log("Second note.", vault_path=str(vault))
    await jarvis_log("Third note.", vault_path=str(vault))

    content = (vault / "wiki" / "journal.md").read_text()
    assert "First note." in content
    assert "Second note." in content
    assert "Third note." in content
    # Three separator lines (---) beyond the frontmatter
    assert content.count("---\n") >= 4  # 1 frontmatter close + 3 entries


@pytest.mark.asyncio
async def test_log_journal_path_in_result(tmp_path):
    """Result includes the absolute journal_path regardless of success/failure."""
    from tools.jarvis.mcp.tools.log import jarvis_log

    vault = _make_vault_with_journal(tmp_path)
    result = await jarvis_log("Check path.", vault_path=str(vault))

    assert "journal_path" in result
    assert result["journal_path"].endswith("journal.md")
