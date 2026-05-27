"""Tests for jarvis_ingest.

Validates:
- New JSON files in raw/ are compiled into the correct wiki dashboard.
- Processed files are skipped on re-run (idempotency).
- Files with errors are indexed in errors.md.
- Parse failures produce a needs-review entry, not a crash.
- Non-JSON / hidden files are ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vault(tmp_path: Path) -> Path:
    """Create a minimal vault skeleton under tmp_path."""
    vault = tmp_path / "vault"
    (vault / "raw" / "ssd_transfers").mkdir(parents=True)
    (vault / "raw" / "dashcam_events").mkdir(parents=True)
    (vault / "wiki").mkdir(parents=True)

    seed_files = {
        "storage_dashboard.md": (
            "---\ntitle: Storage Dashboard\nupdated: 2026-01-01T00:00:00Z\ntags: [storage]\n---\n\n"
            "# Storage Dashboard\n\n## Transfer Log\n\n"
        ),
        "security_events.md": (
            "---\ntitle: Security Events\nupdated: 2026-01-01T00:00:00Z\ntags: [security]\n---\n\n"
            "# Security Events\n\n## Event Log\n\n"
        ),
        "errors.md": (
            "---\ntitle: Errors Index\nupdated: 2026-01-01T00:00:00Z\ntags: [errors]\n---\n\n"
            "# Errors Index\n\n## Error Log\n\n"
        ),
    }
    for name, content in seed_files.items():
        (vault / "wiki" / name).write_text(content, encoding="utf-8")

    return vault


def _write_raw(vault: Path, subdir: str, filename: str, payload: dict) -> Path:
    p = vault / "raw" / subdir / filename
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_single_ssd_transfer(tmp_path):
    """A valid SSD transfer log is compiled into storage_dashboard.md."""
    from tools.jarvis.mcp.tools.ingest import jarvis_ingest

    vault = _make_vault(tmp_path)
    _write_raw(vault, "ssd_transfers", "transfer_001.json", {
        "timestamp": "2026-05-24T02:00:00Z",
        "file_count": 142,
        "checksum_errors": 0,
        "duration_s": 45,
    })

    result = await jarvis_ingest(vault_path=str(vault))

    assert result["ingested"] == 1
    assert result["skipped"] == 0
    assert result["errors"] == 0

    dashboard = (vault / "wiki" / "storage_dashboard.md").read_text()
    assert "2026-05-24T02:00:00Z" in dashboard
    assert "142" in dashboard
    assert "transfer_001.json" in dashboard


@pytest.mark.asyncio
async def test_ingest_is_idempotent(tmp_path):
    """Re-running ingest on the same file skips it — no duplicate entries."""
    from tools.jarvis.mcp.tools.ingest import jarvis_ingest

    vault = _make_vault(tmp_path)
    _write_raw(vault, "ssd_transfers", "transfer_002.json", {
        "timestamp": "2026-05-24T03:00:00Z",
        "file_count": 10,
    })

    result1 = await jarvis_ingest(vault_path=str(vault))
    assert result1["ingested"] == 1

    result2 = await jarvis_ingest(vault_path=str(vault))
    assert result2["ingested"] == 0
    assert result2["skipped"] == 1

    # Dashboard should contain exactly one occurrence of the timestamp
    dashboard = (vault / "wiki" / "storage_dashboard.md").read_text()
    assert dashboard.count("2026-05-24T03:00:00Z") == 1


@pytest.mark.asyncio
async def test_ingest_error_events_indexed_in_errors(tmp_path):
    """Files with an 'errors' field are also indexed in errors.md."""
    from tools.jarvis.mcp.tools.ingest import jarvis_ingest

    vault = _make_vault(tmp_path)
    _write_raw(vault, "ssd_transfers", "bad_transfer.json", {
        "timestamp": "2026-05-24T04:00:00Z",
        "errors": "checksum mismatch on 3 files",
        "file_count": 50,
    })

    result = await jarvis_ingest(vault_path=str(vault))

    assert result["errors"] == 1
    errors_doc = (vault / "wiki" / "errors.md").read_text()
    assert "bad_transfer.json" in errors_doc
    assert "storage_dashboard" in errors_doc


@pytest.mark.asyncio
async def test_ingest_parse_failure_graceful(tmp_path):
    """A malformed JSON file produces a _parse_error entry, not an exception."""
    from tools.jarvis.mcp.tools.ingest import jarvis_ingest

    vault = _make_vault(tmp_path)
    bad_file = vault / "raw" / "ssd_transfers" / "corrupt.json"
    bad_file.write_text("NOT VALID JSON {{{{", encoding="utf-8")

    result = await jarvis_ingest(vault_path=str(vault))

    assert result["ingested"] == 1   # file was processed (even if with error)
    assert result["errors"] == 1


@pytest.mark.asyncio
async def test_ingest_ignores_hidden_and_non_data_files(tmp_path):
    """Hidden files and unsupported extensions are silently ignored."""
    from tools.jarvis.mcp.tools.ingest import jarvis_ingest

    vault = _make_vault(tmp_path)
    (vault / "raw" / "ssd_transfers" / ".gitkeep").write_text("")
    (vault / "raw" / "ssd_transfers" / "readme.md").write_text("# readme")

    result = await jarvis_ingest(vault_path=str(vault))

    assert result["ingested"] == 0


@pytest.mark.asyncio
async def test_ingest_dashcam_events_go_to_security_dashboard(tmp_path):
    """Dashcam events are compiled into security_events.md, not storage_dashboard.md."""
    from tools.jarvis.mcp.tools.ingest import jarvis_ingest

    vault = _make_vault(tmp_path)
    _write_raw(vault, "dashcam_events", "motion_001.json", {
        "timestamp": "2026-05-24T22:00:00Z",
        "confidence": 0.92,
        "clip_path": "/mnt/dashcam/2026-05-24T22:00:00Z.mp4",
    })

    await jarvis_ingest(vault_path=str(vault))

    security = (vault / "wiki" / "security_events.md").read_text()
    assert "2026-05-24T22:00:00Z" in security

    storage = (vault / "wiki" / "storage_dashboard.md").read_text()
    # Motion event should NOT appear in storage dashboard
    assert "2026-05-24T22:00:00Z" not in storage


@pytest.mark.asyncio
async def test_ingest_missing_raw_dir_returns_gracefully(tmp_path):
    """If vault/raw/ does not exist, ingest returns a graceful result."""
    from tools.jarvis.mcp.tools.ingest import jarvis_ingest

    vault = tmp_path / "empty_vault"
    vault.mkdir()

    result = await jarvis_ingest(vault_path=str(vault))

    assert result["ingested"] == 0
    assert "not found" in result.get("detail", "")
