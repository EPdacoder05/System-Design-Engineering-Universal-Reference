"""Jarvis /log tool.

Quick-capture a timestamped note into vault/wiki/journal.md.
Intentionally lightweight — no parsing, no cross-linking.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _vault_path() -> Path:
    env = os.environ.get("JARVIS_VAULT_PATH")
    if env:
        return Path(env)
    return Path(__file__).parent.parent.parent / "vault"


def _update_frontmatter_updated(content: str, ts: str) -> str:
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("updated:"):
            lines[i] = f"updated: {ts}\n"
            break
    return "".join(lines)


async def jarvis_log(
    note: str,
    vault_path: str = "",
) -> Dict[str, Any]:
    """Append a quick-capture note to vault/wiki/journal.md.

    Parameters
    ----------
    note:
        The text to capture.  Any length is accepted.
    vault_path:
        Absolute path to the vault root.  Defaults to ``JARVIS_VAULT_PATH`` env var
        or the default ``tools/jarvis/vault/`` directory.

    Returns
    -------
    dict with ``logged`` (bool), ``timestamp``, and ``journal_path``.
    """
    if not note or not note.strip():
        return {"logged": False, "error": "note must not be empty", "timestamp": "", "journal_path": ""}

    vault = Path(vault_path) if vault_path else _vault_path()
    journal_file = vault / "wiki" / "journal.md"

    if not journal_file.exists():
        return {
            "logged": False,
            "error": f"journal.md not found at {journal_file}",
            "timestamp": "",
            "journal_path": str(journal_file),
        }

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = f"\n### {ts}\n\n{note.strip()}\n\n---\n"

    content = journal_file.read_text(encoding="utf-8")
    content = _update_frontmatter_updated(content, ts)
    journal_file.write_text(content + entry, encoding="utf-8")

    return {
        "logged": True,
        "timestamp": ts,
        "journal_path": str(journal_file),
    }
