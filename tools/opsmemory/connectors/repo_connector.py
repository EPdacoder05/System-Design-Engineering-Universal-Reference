"""Repository documentation ingestion connector for OpsMemory.

Walks a local git repository tree and ingests documentation and reference
files into OpsMemory, seeding the memory store with best-practice knowledge
from the reference repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import structlog

log = structlog.get_logger(__name__)

# Directories that are never interesting for documentation ingestion.
_SKIP_DIRS = {
    ".git",
    ".github",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

# Default file extensions to ingest.
_DEFAULT_EXTENSIONS = {".md", ".txt", ".yaml", ".yml", ".json"}

# Files larger than this are split into overlapping chunks so each POST stays
# within the ingest endpoint's limits.
_CHUNK_CHARS = 8_000
_CHUNK_OVERLAP_CHARS = 200


@dataclass
class RepoConnectorConfig:
    """Configuration for the repository documentation connector."""

    repo_path: str = field(
        default_factory=lambda: os.environ.get("REPO_PATH", ".")
    )
    repo: str = field(
        default_factory=lambda: os.environ.get(
            "REPO_NAME",
            "EPdacoder05/System-Design-Engineering-Universal-Reference",
        )
    )
    file_extensions: List[str] = field(
        default_factory=lambda: list(_DEFAULT_EXTENSIONS)
    )
    max_file_size_kb: int = 500
    ingest_url: str = field(
        default_factory=lambda: os.environ.get(
            "OPSMEMORY_INGEST_URL", "http://localhost:8000/ingest"
        )
    )


class RepoConnector:
    """Walks a local repository tree and ingests documentation files into OpsMemory."""

    def __init__(self, config: Optional[RepoConnectorConfig] = None) -> None:
        self.config = config or RepoConnectorConfig()

    def _iter_files(self) -> List[Path]:
        """Return all document files under *repo_path* matching the configured extensions."""
        root = Path(self.config.repo_path).resolve()
        exts = {e.lower() for e in self.config.file_extensions}
        max_bytes = self.config.max_file_size_kb * 1024
        found: List[Path] = []

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune directories in-place to skip irrelevant subtrees.
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fname in filenames:
                if Path(fname).suffix.lower() in exts:
                    fpath = Path(dirpath) / fname
                    try:
                        if fpath.stat().st_size <= max_bytes:
                            found.append(fpath)
                    except OSError:
                        continue

        return found

    @staticmethod
    def _chunk_text(text: str) -> List[str]:
        """Split *text* into overlapping chunks suitable for a single ingest POST."""
        if len(text) <= _CHUNK_CHARS:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + _CHUNK_CHARS
            chunks.append(text[start:end])
            start = end - _CHUNK_OVERLAP_CHARS
        return chunks

    async def ingest_file(self, path: Path, root: Path) -> int:
        """Ingest a single file; returns the number of chunks posted."""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.warning("repo_connector_read_error", path=str(path), error=str(exc))
            return 0

        text = content.strip()
        if not text:
            return 0

        rel_path = path.relative_to(root)
        source_ref = (
            f"https://github.com/{self.config.repo}/blob/main/{rel_path.as_posix()}"
        )
        chunks = self._chunk_text(text)

        for i, chunk in enumerate(chunks):
            native_id = (
                f"{rel_path.as_posix()}:chunk{i}" if len(chunks) > 1 else rel_path.as_posix()
            )
            payload = {
                "text": chunk,
                "source_type": "reference_doc",
                "source_ref": source_ref,
                "repo": self.config.repo,
                "native_id": native_id,
            }
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(self.config.ingest_url, json=payload)
                response.raise_for_status()

        return len(chunks)

    async def run_once(self) -> Dict[str, Any]:
        """Walk the repository and ingest all matching documentation files.

        Returns a stats dict:
        ``{"files_ingested": N, "chunks_posted": N, "files_skipped": N}``.
        """
        root = Path(self.config.repo_path).resolve()
        files = self._iter_files()
        stats: Dict[str, Any] = {
            "files_ingested": 0,
            "chunks_posted": 0,
            "files_skipped": 0,
        }

        for fpath in files:
            try:
                n_chunks = await self.ingest_file(fpath, root)
                if n_chunks > 0:
                    stats["files_ingested"] += 1
                    stats["chunks_posted"] += n_chunks
                else:
                    stats["files_skipped"] += 1
            except Exception as exc:
                log.error(
                    "repo_connector_ingest_error", path=str(fpath), error=str(exc)
                )
                stats["files_skipped"] += 1

        log.info("repo_connector_complete", **stats)
        return stats
