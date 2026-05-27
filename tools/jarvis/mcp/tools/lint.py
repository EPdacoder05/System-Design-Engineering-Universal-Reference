"""Jarvis /lint tool.

Cross-checks the compiled wiki state against live system reality and reports
any drift.  All checks are read-only — this tool never modifies wiki files.

Checks
------
1. Docker containers — every inline-code service name in wiki is checked via ``docker ps``.
2. File paths — every inline-code absolute path in wiki is verified for existence.
3. Raw backlog — counts un-ingested files waiting in ``vault/raw/``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def _vault_path() -> Path:
    env = os.environ.get("JARVIS_VAULT_PATH")
    if env:
        return Path(env)
    return Path(__file__).parent.parent.parent / "vault"


def _load_wiki_text(wiki_root: Path) -> str:
    """Concatenate all wiki Markdown files into one string for pattern scanning."""
    parts: List[str] = []
    if not wiki_root.exists():
        return ""
    for md_file in sorted(wiki_root.glob("*.md")):
        try:
            parts.append(md_file.read_text(encoding="utf-8"))
        except OSError:
            pass
    return "\n".join(parts)


def _extract_backtick_tokens(text: str) -> Set[str]:
    """Return all tokens wrapped in backticks from the wiki text."""
    return set(re.findall(r"`([^`]+)`", text))


def _classify_tokens(tokens: Set[str]) -> Tuple[Set[str], Set[str]]:
    """Split tokens into probable Docker service names and absolute file paths."""
    docker_names: Set[str] = set()
    file_paths: Set[str] = set()
    for token in tokens:
        if token.startswith("/"):
            file_paths.add(token)
        elif re.match(r"^[a-z][a-z0-9_\-]*$", token):
            docker_names.add(token)
    return docker_names, file_paths


def _running_docker_containers() -> Set[str]:
    """Return the set of running Docker container names (empty if Docker is unavailable)."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return {name.strip() for name in result.stdout.splitlines() if name.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return set()


def _count_raw_backlog(raw_root: Path, registry: Dict[str, Any]) -> int:
    """Count files in vault/raw/ that have not yet been ingested."""
    count = 0
    if not raw_root.exists():
        return 0
    for raw_file in raw_root.rglob("*"):
        if raw_file.is_file() and raw_file.suffix in (".json", ".txt", ".csv") and not raw_file.name.startswith("."):
            mtime = int(raw_file.stat().st_mtime)
            native_id = f"{raw_file.name}:{mtime}"
            if native_id not in registry:
                count += 1
    return count


def _load_registry(vault: Path) -> Dict[str, Any]:
    registry_path = vault / ".ingest_registry.json"
    if registry_path.exists():
        try:
            return json.loads(registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


async def jarvis_lint(vault_path: str = "") -> Dict[str, Any]:
    """Health-check the compiled wiki state against live system reality.

    Parameters
    ----------
    vault_path:
        Absolute path to the vault root.  Defaults to ``JARVIS_VAULT_PATH`` env var
        or the default ``tools/jarvis/vault/`` directory.

    Returns
    -------
    dict with ``timestamp``, ``passing`` list, ``drift`` list, ``raw_backlog`` count,
    and ``docker_available`` boolean.
    """
    vault = Path(vault_path) if vault_path else _vault_path()
    wiki_root = vault / "wiki"
    raw_root = vault / "raw"

    wiki_text = _load_wiki_text(wiki_root)
    tokens = _extract_backtick_tokens(wiki_text)
    docker_names, file_paths = _classify_tokens(tokens)

    running_containers = _running_docker_containers()
    docker_available = bool(running_containers) or _docker_is_installed()

    passing: List[str] = []
    drift: List[str] = []

    # Docker checks
    for name in sorted(docker_names):
        if not docker_available:
            drift.append(f"docker:{name} — Docker not available; cannot verify")
        elif name in running_containers:
            passing.append(f"docker:{name} — running ✓")
        else:
            drift.append(f"docker:{name} — container not running")

    # File path checks
    for path_str in sorted(file_paths):
        p = Path(path_str)
        if p.exists():
            passing.append(f"path:{path_str} — exists ✓")
        else:
            drift.append(f"path:{path_str} — not found")

    registry = _load_registry(vault)
    backlog = _count_raw_backlog(raw_root, registry)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passing": passing,
        "drift": drift,
        "raw_backlog": backlog,
        "docker_available": docker_available,
    }


def _docker_is_installed() -> bool:
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
