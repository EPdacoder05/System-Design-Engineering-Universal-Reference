"""GitHub ingestion connector for OpsMemory."""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)

_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in _RETRY_STATUS_CODES


def _retry_policy():
    return dict(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )


@dataclass
class GitHubConnectorConfig:
    """Configuration for the GitHub ingestion connector."""

    github_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("GITHUB_TOKEN")
    )
    owner: str = field(
        default_factory=lambda: os.environ.get("GITHUB_OWNER", "EPdacoder05")
    )
    include_repos: List[str] = field(
        default_factory=lambda: [
            r.strip()
            for r in os.environ.get("GITHUB_INCLUDE_REPOS", "").split(",")
            if r.strip()
        ]
    )
    exclude_repos: List[str] = field(
        default_factory=lambda: [
            r.strip()
            for r in os.environ.get("GITHUB_EXCLUDE_REPOS", "").split(",")
            if r.strip()
        ]
    )
    poll_interval_seconds: int = field(
        default_factory=lambda: int(os.environ.get("GITHUB_POLL_INTERVAL", "3600"))
    )
    max_items_per_repo: int = 50
    ingest_url: str = field(
        default_factory=lambda: os.environ.get(
            "OPSMEMORY_INGEST_URL", "http://localhost:8000/ingest"
        )
    )


class GitHubConnector:
    """Fetches commits and PRs from GitHub and forwards them to OpsMemory."""

    _github_api_url = "https://api.github.com"

    def __init__(self, config: Optional[GitHubConnectorConfig] = None) -> None:
        self.config = config or GitHubConnectorConfig()

    @property
    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.config.github_token:
            headers["Authorization"] = f"token {self.config.github_token}"
        return headers

    @retry(**_retry_policy())
    async def list_repos(self) -> List[dict]:
        """Return repositories owned by *owner*, applying include/exclude filters."""
        url = f"{self._github_api_url}/users/{self.config.owner}/repos"
        params = {"per_page": 100, "type": "owner"}
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            repos: List[dict] = response.json()

        if self.config.include_repos:
            repos = [r for r in repos if r["name"] in self.config.include_repos]
        if self.config.exclude_repos:
            repos = [r for r in repos if r["name"] not in self.config.exclude_repos]
        return repos

    @retry(**_retry_policy())
    async def fetch_recent_commits(
        self, repo_name: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Fetch the *limit* most recent commits for *repo_name*."""
        url = f"{self._github_api_url}/repos/{self.config.owner}/{repo_name}/commits"
        params = {"per_page": limit}
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            raw: List[dict] = response.json()

        results = []
        for c in raw:
            commit = c.get("commit", {})
            author = commit.get("author") or {}
            results.append(
                {
                    "sha": c.get("sha", ""),
                    "message": commit.get("message", ""),
                    "author_name": author.get("name", ""),
                    "author_email": author.get("email", ""),
                    "committed_at": author.get("date", ""),
                    "url": c.get("html_url", ""),
                }
            )
        return results

    @retry(**_retry_policy())
    async def fetch_recent_prs(
        self, repo_name: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Fetch the *limit* most recently updated PRs for *repo_name*."""
        url = f"{self._github_api_url}/repos/{self.config.owner}/{repo_name}/pulls"
        params = {
            "state": "all",
            "per_page": limit,
            "sort": "updated",
            "direction": "desc",
        }
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            raw: List[dict] = response.json()

        results = []
        for pr in raw:
            results.append(
                {
                    "number": pr.get("number"),
                    "title": pr.get("title", ""),
                    "body": pr.get("body") or "",
                    "state": pr.get("state", ""),
                    "merged": pr.get("merged_at") is not None,
                    "created_at": pr.get("created_at", ""),
                    "updated_at": pr.get("updated_at", ""),
                    "url": pr.get("html_url", ""),
                    "labels": [lbl["name"] for lbl in pr.get("labels", [])],
                }
            )
        return results

    async def normalize_commit(self, repo_name: str, commit: Dict[str, Any]) -> str:
        """Format a commit dict as human-readable text."""
        sha_short = commit["sha"][:7] if commit.get("sha") else "unknown"
        return (
            f"[COMMIT] {repo_name}@{sha_short}: {commit.get('message', '').strip()}\n"
            f"Author: {commit.get('author_name', '')} <{commit.get('author_email', '')}>\n"
            f"Date: {commit.get('committed_at', '')}\n"
            f"URL: {commit.get('url', '')}"
        )

    async def normalize_pr(self, repo_name: str, pr: Dict[str, Any]) -> str:
        """Format a PR dict as human-readable text."""
        labels = ", ".join(pr.get("labels", [])) or "none"
        return (
            f"[PR #{pr.get('number')}] {repo_name}: {pr.get('title', '').strip()}\n"
            f"State: {pr.get('state', '')} | Merged: {pr.get('merged', False)}\n"
            f"Labels: {labels}\n"
            f"Created: {pr.get('created_at', '')} | Updated: {pr.get('updated_at', '')}\n"
            f"URL: {pr.get('url', '')}\n"
            f"Body: {(pr.get('body') or '')[:500]}"
        )

    @retry(**_retry_policy())
    async def ingest_item(
        self,
        text: str,
        source_type: str,
        source_ref: str,
        author: str = "",
        repo: str = "",
        native_id: str = "",
        occurred_at: str = "",
    ) -> bool:
        """POST a single item to the OpsMemory ingest endpoint."""
        payload = {
            "text": text,
            "source_type": source_type,
            "source_ref": source_ref,
            "author": author,
            "repo": repo or None,
            "native_id": native_id or None,
            "occurred_at": occurred_at or None,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self.config.ingest_url, json=payload)
            response.raise_for_status()
        return True

    async def run_once(self, session=None) -> Dict[str, Dict[str, int]]:
        """Run one full ingestion sweep across all repos.

        Returns a stats dict of the form ``{repo_name: {commits: N, prs: N}}``.
        """
        stats: Dict[str, Dict[str, int]] = {}
        try:
            repos = await self.list_repos()
        except Exception as exc:
            log.error("github_list_repos_failed", error=str(exc))
            return stats

        for repo in repos:
            repo_name: str = repo["name"]
            repo_full_name = f"{self.config.owner}/{repo_name}"
            repo_stats = {"commits": 0, "prs": 0}

            try:
                commits = await self.fetch_recent_commits(
                    repo_name, self.config.max_items_per_repo
                )
                for commit in commits:
                    text = await self.normalize_commit(repo_name, commit)
                    await self.ingest_item(
                        text=text,
                        source_type="github_commit",
                        source_ref=commit.get("url", ""),
                        author=commit.get("author_name", ""),
                        repo=repo_full_name,
                        native_id=commit.get("sha", ""),
                        occurred_at=commit.get("committed_at", ""),
                    )
                    repo_stats["commits"] += 1
            except Exception as exc:
                log.error(
                    "github_commits_failed", repo=repo_name, error=str(exc)
                )

            try:
                prs = await self.fetch_recent_prs(
                    repo_name, self.config.max_items_per_repo
                )
                for pr in prs:
                    text = await self.normalize_pr(repo_name, pr)
                    await self.ingest_item(
                        text=text,
                        source_type="github_pr",
                        source_ref=pr.get("url", ""),
                        repo=repo_full_name,
                        native_id=str(pr.get("number", "")),
                        occurred_at=pr.get("created_at", ""),
                    )
                    repo_stats["prs"] += 1
            except Exception as exc:
                log.error("github_prs_failed", repo=repo_name, error=str(exc))

            stats[repo_name] = repo_stats
            log.info("github_repo_ingested", repo=repo_name, **repo_stats)

        return stats

    async def run_loop(self, session=None) -> None:
        """Run ``run_once`` on a repeating timer, logging stats each cycle."""
        log.info(
            "github_connector_loop_started",
            owner=self.config.owner,
            poll_interval_seconds=self.config.poll_interval_seconds,
        )
        while True:
            try:
                stats = await self.run_once(session=session)
                log.info("github_connector_cycle_complete", stats=stats)
            except Exception as exc:
                log.error("github_connector_loop_error", error=str(exc))
            await asyncio.sleep(self.config.poll_interval_seconds)
