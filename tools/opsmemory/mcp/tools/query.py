"""OpsMemory query MCP tool — ``memory_query``.

Exposes semantic search over the OpsMemory evidence + memory store as an
MCP tool.  Delegates to the running OpsMemory HTTP API so there is no direct
database access from the MCP layer.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
import structlog

log = structlog.get_logger(__name__)

_DEFAULT_API_URL = "http://localhost:8000"


def _api_url() -> str:
    return os.environ.get("OPSMEMORY_API_URL", _DEFAULT_API_URL).rstrip("/")


async def memory_query(
    q: str,
    limit: int = 5,
) -> Dict[str, Any]:
    """Search OpsMemory for evidence and memories relevant to *q*.

    Parameters
    ----------
    q:
        Natural language query string.
    limit:
        Maximum number of results to return (1–50).

    Returns
    -------
    dict with keys:
        - ``query`` – echo of the input query
        - ``answer`` – synthesised answer string
        - ``citations`` – list of matching evidence items
        - ``memories`` – list of consolidated memory records
    """
    limit = max(1, min(50, limit))
    url = f"{_api_url()}/query"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, params={"q": q, "limit": limit})
        response.raise_for_status()
    result: Dict[str, Any] = response.json()
    log.info("mcp_memory_query", q=q, limit=limit, citations=len(result.get("citations", [])))
    return result
