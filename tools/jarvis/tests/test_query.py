"""Tests for jarvis_query.

Validates:
- Questions are matched against compiled wiki sections.
- Results are correctly cited with file + section.
- Confidence levels are computed correctly.
- Empty / missing wiki returns a "no record" response.
- limit parameter is clamped to [1, 20].
- needs_review entries downgrade confidence to low.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _make_wiki(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    wiki = vault / "wiki"
    wiki.mkdir(parents=True)
    (vault / "raw").mkdir()

    (wiki / "storage_dashboard.md").write_text(
        "---\ntitle: Storage Dashboard\nupdated: 2026-05-24T02:00:00Z\ntags: [storage]\n---\n\n"
        "# Storage Dashboard\n\n"
        "### 2026-05-24T02:00:00Z — ssd_transfers event\n\n"
        "| Field | Value |\n|-------|-------|\n"
        "| source | `raw/ssd_transfers/transfer_001.json` |\n"
        "| status | success |\n"
        "| file_count | 142 |\n"
        "| checksum_errors | 0 |\n\n",
        encoding="utf-8",
    )
    (wiki / "security_events.md").write_text(
        "---\ntitle: Security Events\nupdated: 2026-05-24T22:00:00Z\ntags: [security]\n---\n\n"
        "# Security Events\n\n"
        "### 2026-05-24T22:00:00Z — dashcam_events event\n\n"
        "| Field | Value |\n|-------|-------|\n"
        "| source | `raw/dashcam_events/motion_001.json` |\n"
        "| status | success |\n"
        "| confidence | 0.92 |\n\n",
        encoding="utf-8",
    )
    (wiki / "errors.md").write_text(
        "---\ntitle: Errors\nupdated: 2026-05-24T00:00:00Z\ntags: [errors]\n---\n\n"
        "# Errors Index\n\n## Error Log\n\n",
        encoding="utf-8",
    )
    return vault


@pytest.mark.asyncio
async def test_query_finds_ssd_transfer(tmp_path):
    """Query for SSD transfer returns a cited answer from storage_dashboard."""
    from tools.jarvis.mcp.tools.query import jarvis_query

    vault = _make_wiki(tmp_path)
    result = await jarvis_query("Did the SSD transfer finish?", vault_path=str(vault))

    assert result["confidence"] in ("high", "medium", "low")
    assert "storage_dashboard" in result["answer"].lower() or any(
        "storage_dashboard" in c["file"] for c in result["citations"]
    )
    assert result["question"] == "Did the SSD transfer finish?"


@pytest.mark.asyncio
async def test_query_finds_dashcam_event(tmp_path):
    """Query for dashcam / motion event returns a result from security_events."""
    from tools.jarvis.mcp.tools.query import jarvis_query

    vault = _make_wiki(tmp_path)
    result = await jarvis_query("Any motion detected last night?", vault_path=str(vault))

    assert result["confidence"] != "none"
    assert any("security_events" in c["file"] for c in result["citations"])


@pytest.mark.asyncio
async def test_query_no_match_returns_none_confidence(tmp_path):
    """Unrelated question returns confidence 'none' and no citations."""
    from tools.jarvis.mcp.tools.query import jarvis_query

    vault = _make_wiki(tmp_path)
    result = await jarvis_query("What is the weather in Tokyo?", vault_path=str(vault))

    assert result["confidence"] == "none"
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_query_empty_wiki_returns_no_record(tmp_path):
    """Query against an empty vault returns a helpful 'no record' message."""
    from tools.jarvis.mcp.tools.query import jarvis_query

    vault = tmp_path / "vault"
    (vault / "wiki").mkdir(parents=True)

    result = await jarvis_query("Anything?", vault_path=str(vault))

    assert result["confidence"] == "none"
    assert "ingest" in result["answer"].lower()


@pytest.mark.asyncio
async def test_query_limit_clamped(tmp_path):
    """limit is clamped to [1, 20] — no crashes on out-of-range values."""
    from tools.jarvis.mcp.tools.query import jarvis_query

    vault = _make_wiki(tmp_path)
    result_high = await jarvis_query("transfer", vault_path=str(vault), limit=999)
    result_low = await jarvis_query("transfer", vault_path=str(vault), limit=0)

    assert len(result_high["citations"]) <= 20
    assert len(result_low["citations"]) >= 1 or result_low["confidence"] == "none"


@pytest.mark.asyncio
async def test_query_needs_review_lowers_confidence(tmp_path):
    """A section marked 'needs_review' returns confidence 'low'."""
    from tools.jarvis.mcp.tools.query import jarvis_query

    vault = tmp_path / "vault"
    wiki = vault / "wiki"
    wiki.mkdir(parents=True)
    (vault / "raw").mkdir()

    (wiki / "storage_dashboard.md").write_text(
        "---\ntitle: Storage\nupdated: 2026-05-24T00:00:00Z\ntags: [storage]\n---\n\n"
        "### 2026-05-24T00:00:00Z — ssd_transfers event\n\n"
        "| status | needs_review |\n| file_count | unknown |\n\n",
        encoding="utf-8",
    )

    result = await jarvis_query("SSD transfer status", vault_path=str(vault))

    assert result["confidence"] == "low"


@pytest.mark.asyncio
async def test_query_citation_structure(tmp_path):
    """Each citation must include 'file', 'section', and 'score' keys."""
    from tools.jarvis.mcp.tools.query import jarvis_query

    vault = _make_wiki(tmp_path)
    result = await jarvis_query("SSD storage transfer file count", vault_path=str(vault))

    for citation in result["citations"]:
        assert "file" in citation
        assert "section" in citation
        assert "score" in citation
        assert citation["score"] > 0
