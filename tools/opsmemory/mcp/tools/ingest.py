"""OpsMemory ingest MCP tools.

Exposes evidence ingestion as MCP tools:
- ``memory_ingest_text`` — ingest free-form text
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
import structlog

log = structlog.get_logger(__name__)

_DEFAULT_API_URL = "http://localhost:8000"


def _api_url() -> str:
    return os.environ.get("OPSMEMORY_API_URL", _DEFAULT_API_URL).rstrip("/")


async def memory_ingest_text(
    text: str,
    source_type: str = "manual",
    source_ref: str = "",
    author: Optional[str] = None,
    repo: Optional[str] = None,
    native_id: Optional[str] = None,
    occurred_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingest a text evidence item into OpsMemory.

    Parameters
    ----------
    text:
        The raw text to ingest.  Secrets are automatically redacted.
    source_type:
        Category label for the source (e.g. ``"manual"``, ``"github_pr"``).
    source_ref:
        URL or reference string identifying the original source.
    author:
        Optional author name or identifier.
    repo:
        Optional repository name in ``owner/repo`` format.
    native_id:
        Optional native ID in the originating system (e.g. commit SHA).
    occurred_at:
        Optional ISO-8601 timestamp for when the original event occurred.

    Returns
    -------
    dict with ``evidence_id``, ``excerpt``, ``redacted``, and ``correlation_id``.
    """
    payload: Dict[str, Any] = {
        "text": text,
        "source_type": source_type,
        "source_ref": source_ref,
        "author": author,
        "repo": repo,
        "native_id": native_id,
        "occurred_at": occurred_at,
    }
    url = f"{_api_url()}/ingest"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
    result: Dict[str, Any] = response.json()
    log.info(
        "mcp_memory_ingest_text",
        evidence_id=result.get("evidence_id"),
        source_type=source_type,
        redacted=result.get("redacted"),
    )
    return result
