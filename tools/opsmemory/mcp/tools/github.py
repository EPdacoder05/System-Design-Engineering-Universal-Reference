"""OpsMemory GitHub MCP tool — ``memory_sync_github_owner``.

Triggers a single GitHub ingestion sweep for the configured owner, pulling
recent commits and PRs into OpsMemory.
"""

from __future__ import annotations

from typing import Any, Dict

import structlog

log = structlog.get_logger(__name__)


async def memory_sync_github_owner(
    owner: str = "",
    include_repos: str = "",
    exclude_repos: str = "",
    max_items_per_repo: int = 50,
) -> Dict[str, Any]:
    """Run one GitHub ingestion sweep for the resolved owner.

    Parameters
    ----------
    owner:
        GitHub username or organisation.  If empty, falls back to the
        ``GITHUB_OWNER`` env var or the authenticated token identity.
    include_repos:
        Optional comma-separated list of repo names to include exclusively.
    exclude_repos:
        Optional comma-separated list of repo names to skip.
    max_items_per_repo:
        Maximum commits/PRs to fetch per repo.

    Returns
    -------
    dict mapping repo name → ``{"commits": N, "prs": N}``.
    """
    from tools.opsmemory.connectors.github_connector import (
        GitHubConnector,
        GitHubConnectorConfig,
    )

    include_list = [r.strip() for r in include_repos.split(",") if r.strip()]
    exclude_list = [r.strip() for r in exclude_repos.split(",") if r.strip()]

    config = GitHubConnectorConfig(
        owner=owner or None,
        include_repos=include_list,
        exclude_repos=exclude_list,
        max_items_per_repo=max_items_per_repo,
    )
    connector = GitHubConnector(config=config)

    resolved = await connector.resolve_owner()
    log.info("mcp_memory_sync_github_owner_started", owner=resolved)
    stats = await connector.run_once()
    log.info("mcp_memory_sync_github_owner_complete", owner=resolved, stats=stats)
    return {"owner": resolved, "repos": stats}
