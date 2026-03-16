"""OpsMemory repository ingestion MCP tool — ``memory_ingest_repo``.

Seeds the OpsMemory knowledge store with documentation and best-practice
reference files from a local repository tree.
"""

from __future__ import annotations

from typing import Any, Dict

import structlog

log = structlog.get_logger(__name__)


async def memory_ingest_repo(
    repo_path: str = ".",
    repo: str = "EPdacoder05/System-Design-Engineering-Universal-Reference",
    file_extensions: str = ".md,.txt,.yaml,.yml,.json",
    max_file_size_kb: int = 500,
) -> Dict[str, Any]:
    """Ingest all documentation and reference files from a local repository.

    Walks the local *repo_path* tree, reads every file whose extension matches
    *file_extensions*, and posts each one to OpsMemory via the ingest API.
    This seeds the memory store with best-practice knowledge so that subsequent
    queries and AI agents can apply those patterns when generating new services
    or making architectural decisions.

    Parameters
    ----------
    repo_path:
        Absolute or relative path to the repository root on the local
        filesystem.  Defaults to the current working directory.
    repo:
        ``owner/repo`` name used as the ``repo`` field in each evidence
        record and to construct ``source_ref`` URLs.
    file_extensions:
        Comma-separated list of file extensions to ingest
        (e.g. ``".md,.yaml,.txt"``).  Defaults to
        ``".md,.txt,.yaml,.yml,.json"``.
    max_file_size_kb:
        Files larger than this (in kilobytes) are skipped.  Defaults to 500.

    Returns
    -------
    dict with ``repo``, ``files_ingested``, ``chunks_posted``, and
    ``files_skipped``.
    """
    from tools.opsmemory.connectors.repo_connector import (
        RepoConnector,
        RepoConnectorConfig,
    )

    ext_list = [e.strip() for e in file_extensions.split(",") if e.strip()]
    config = RepoConnectorConfig(
        repo_path=repo_path,
        repo=repo,
        file_extensions=ext_list,
        max_file_size_kb=max_file_size_kb,
    )
    connector = RepoConnector(config=config)

    log.info("mcp_memory_ingest_repo_started", repo_path=repo_path, repo=repo)
    stats = await connector.run_once()
    log.info("mcp_memory_ingest_repo_complete", repo=repo, **stats)

    return {"repo": repo, **stats}
