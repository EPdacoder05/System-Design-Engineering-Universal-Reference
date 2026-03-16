"""OpsMemory admin MCP tools.

Exposes administrative and status operations:
- ``memory_status``        — current store counts
- ``memory_consolidate``   — trigger a consolidation cycle
- ``memory_list_sources``  — list registered data sources
- ``memory_delete_memory`` — delete a single memory record
- ``memory_delete_evidence`` — delete a single evidence item
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import httpx
import structlog

log = structlog.get_logger(__name__)

_DEFAULT_API_URL = "http://localhost:8000"


def _api_url() -> str:
    return os.environ.get("OPSMEMORY_API_URL", _DEFAULT_API_URL).rstrip("/")


async def memory_status() -> Dict[str, Any]:
    """Return a summary of the current OpsMemory store.

    Returns
    -------
    dict with:
        - ``evidence_total`` – total evidence items ingested
        - ``evidence_unconsolidated`` – items not yet consolidated
        - ``memories`` – consolidated memory record count
    """
    url = f"{_api_url()}/status"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url)
        response.raise_for_status()
    return response.json()


async def memory_consolidate() -> Dict[str, Any]:
    """Trigger a consolidation cycle.

    Groups recent evidence items into Memory records and stores embeddings.

    Returns
    -------
    dict with ``run_id``, ``memories_created``, ``evidence_consolidated``.
    """
    url = f"{_api_url()}/consolidate"
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url)
        response.raise_for_status()
    result: Dict[str, Any] = response.json()
    log.info(
        "mcp_memory_consolidate",
        run_id=result.get("run_id"),
        memories_created=result.get("memories_created"),
        evidence_consolidated=result.get("evidence_consolidated"),
    )
    return result


async def memory_list_sources() -> List[Dict[str, Any]]:
    """List registered data sources in the memory store.

    Returns
    -------
    List of source dicts with ``id``, ``owner``, ``repo``, ``source_type``,
    ``last_fetched_at``.
    """
    url = f"{_api_url()}/sources"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url)
        response.raise_for_status()
    return response.json()


async def memory_delete_memory(memory_id: str) -> Dict[str, Any]:
    """Delete a single consolidated memory record by UUID.

    Parameters
    ----------
    memory_id:
        UUID string of the memory to delete.

    Returns
    -------
    dict with ``deleted`` key set to the deleted ID.
    """
    url = f"{_api_url()}/memories/{memory_id}"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.delete(url)
        response.raise_for_status()
    result: Dict[str, Any] = response.json()
    log.info("mcp_memory_delete_memory", memory_id=memory_id)
    return result


async def memory_delete_evidence(evidence_id: str) -> Dict[str, Any]:
    """Delete a single evidence item by UUID.

    Parameters
    ----------
    evidence_id:
        UUID string of the evidence item to delete.

    Returns
    -------
    dict with ``deleted`` key set to the deleted ID.
    """
    url = f"{_api_url()}/evidence/{evidence_id}"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.delete(url)
        response.raise_for_status()
    result: Dict[str, Any] = response.json()
    log.info("mcp_memory_delete_evidence", evidence_id=evidence_id)
    return result
