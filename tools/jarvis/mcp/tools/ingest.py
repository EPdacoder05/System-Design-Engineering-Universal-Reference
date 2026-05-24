"""Jarvis /ingest tool.

Scans vault/raw/ for new JSON event files, extracts key fields, and appends
a clean Markdown summary to the appropriate vault/wiki/ dashboard.

Design principles
-----------------
- Idempotent: re-running on an already-processed file produces no duplicate entries.
  A processed-files registry is kept in ``vault/.ingest_registry.json``.
- Read-only on ``raw/``: this module never modifies files under ``raw/``.
- Source-cited: every wiki entry includes the relative path to the originating file.
"""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_REGISTRY_FILENAME = ".ingest_registry.json"

# Map raw sub-directory names to wiki dashboard files
_RAW_DIR_TO_WIKI: Dict[str, str] = {
    "ssd_transfers": "storage_dashboard.md",
    "plex_webhooks": "storage_dashboard.md",
    "dashcam_events": "security_events.md",
    "iot_mqtt": "storage_dashboard.md",
}


def _vault_path() -> Path:
    env = os.environ.get("JARVIS_VAULT_PATH")
    if env:
        return Path(env)
    return Path(__file__).parent.parent.parent / "vault"


def _load_registry(vault: Path) -> Dict[str, str]:
    registry_path = vault / _REGISTRY_FILENAME
    if registry_path.exists():
        try:
            return json.loads(registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_registry(vault: Path, registry: Dict[str, str]) -> None:
    registry_path = vault / _REGISTRY_FILENAME
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8"
    )


def _native_id(path: Path) -> str:
    """Stable identity for a raw file: relative path + mtime (seconds)."""
    mtime = int(path.stat().st_mtime)
    return f"{path.name}:{mtime}"


def _parse_raw_file(path: Path) -> Dict[str, Any]:
    """Parse a raw JSON file; return raw dict or a minimal error record."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"_parse_error": str(exc), "_raw_path": str(path)}


def _build_wiki_entry(raw: Dict[str, Any], source_rel: str, domain: str) -> str:
    """Render a Markdown entry for one raw event."""
    ts = raw.get("timestamp") or raw.get("occurred_at") or datetime.now(timezone.utc).isoformat()
    status = "failed" if raw.get("errors") or raw.get("_parse_error") else "success"

    lines = [
        f"### {ts} — {domain} event",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| source | `{source_rel}` |",
        f"| status | {status} |",
    ]
    for key, value in raw.items():
        if key.startswith("_") or key in ("timestamp", "occurred_at"):
            continue
        lines.append(f"| {key} | {value} |")
    lines.append("")
    return "\n".join(lines)


def _build_error_entry(raw: Dict[str, Any], source_rel: str, wiki_page: str) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    errors = raw.get("errors") or raw.get("_parse_error", "unknown error")
    wiki_stem = wiki_page.replace(".md", "")
    return (
        f"### {ts} — error from `{source_rel}`\n\n"
        f"- errors: {errors}\n"
        f"- see: [[{wiki_stem}]]\n\n"
    )


def _append_to_wiki(wiki_file: Path, entry: str) -> None:
    content = wiki_file.read_text(encoding="utf-8")
    # Update the `updated:` frontmatter field
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    content = _update_frontmatter_updated(content, now)
    wiki_file.write_text(content + entry, encoding="utf-8")


def _update_frontmatter_updated(content: str, ts: str) -> str:
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("updated:"):
            lines[i] = f"updated: {ts}\n"
            break
    return "".join(lines)


async def jarvis_ingest(vault_path: str = "") -> Dict[str, Any]:
    """Compile new raw event files into the appropriate wiki dashboards.

    Parameters
    ----------
    vault_path:
        Absolute path to the vault root.  If empty, uses the ``JARVIS_VAULT_PATH``
        environment variable or the default ``tools/jarvis/vault/`` directory.

    Returns
    -------
    dict with ``ingested``, ``skipped``, and ``errors`` counts plus a list of
    ``processed`` file paths.
    """
    vault = Path(vault_path) if vault_path else _vault_path()
    raw_root = vault / "raw"
    wiki_root = vault / "wiki"
    errors_file = wiki_root / "errors.md"

    if not raw_root.exists():
        return {"ingested": 0, "skipped": 0, "errors": 0, "processed": [], "detail": "raw/ not found"}

    registry = _load_registry(vault)
    ingested: List[str] = []
    skipped: List[str] = []
    error_count = 0

    for raw_subdir in sorted(raw_root.iterdir()):
        if not raw_subdir.is_dir():
            continue
        domain = raw_subdir.name
        wiki_filename = _RAW_DIR_TO_WIKI.get(domain, "storage_dashboard.md")
        wiki_file = wiki_root / wiki_filename

        if not wiki_file.exists():
            continue

        for raw_file in sorted(raw_subdir.iterdir()):
            if raw_file.suffix not in (".json", ".txt", ".csv") or raw_file.name.startswith("."):
                continue

            native_id = _native_id(raw_file)
            if native_id in registry:
                skipped.append(str(raw_file))
                continue

            raw = _parse_raw_file(raw_file)
            source_rel = str(raw_file.relative_to(vault))
            entry = _build_wiki_entry(raw, source_rel, domain)
            _append_to_wiki(wiki_file, entry)

            if raw.get("errors") or raw.get("_parse_error"):
                error_count += 1
                if errors_file.exists():
                    _append_to_wiki(errors_file, _build_error_entry(raw, source_rel, wiki_filename))

            registry[native_id] = datetime.now(timezone.utc).isoformat()
            ingested.append(str(raw_file))

    _save_registry(vault, registry)
    return {
        "ingested": len(ingested),
        "skipped": len(skipped),
        "errors": error_count,
        "processed": ingested,
    }
