"""Generic MCP client wrapper for calling OpsMemory tools from a Jarvis-style
assistant / orchestrator.

Architecture
------------
The OpsMemory MCP server exposes tools over two transports:

* **stdio** — for direct subprocess embedding (Claude Desktop, Cursor, IDE).
* **SSE**   — for HTTP-based clients (``python -m tools.opsmemory.mcp.server
  --transport sse --port 8100``).

Because stdio transport wiring is tightly coupled to the host process launcher,
this module provides a **REST API adapter** that calls the OpsMemory FastAPI
service directly.  For native MCP transport, see the TODO markers below and
adapt using your MCP SDK of choice.

Usage
-----
::

    import asyncio
    from tools.opsmemory.integrations.jarvis.mcp_client import OpsMemoryClient
    from tools.opsmemory.integrations.jarvis.config import JarvisClientConfig

    async def main():
        cfg = JarvisClientConfig()          # reads from env vars
        client = OpsMemoryClient(cfg)

        # Query memory before answering
        context = await client.query_memory("recent deployments")

        # Ingest session outcome after task completion
        await client.ingest_session_outcome(
            text="Completed deployment of service-x v2.1 to staging.",
            session_id="task-42",
        )

    asyncio.run(main())

Security
--------
* All secrets (``OPSMEMORY_API_KEY``, ``OPSMEMORY_MCP_TOKEN``) are read from
  environment variables or a ``.env`` file — never hardcoded.
* Connections default to ``http://localhost:*`` (loopback only).  Change
  ``OPSMEMORY_API_URL`` to target a remote server over HTTPS.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from tools.opsmemory.integrations.jarvis.config import JarvisClientConfig

log = logging.getLogger(__name__)


class OpsMemoryClient:
    """Async HTTP client for calling OpsMemory tools from a Jarvis assistant.

    This client wraps the OpsMemory REST API.  For full MCP tool dispatch
    (including stdio transport), see the TODO section at the bottom of this
    file.

    All methods are coroutines — use ``await`` or run them inside an async
    event loop.
    """

    def __init__(self, config: Optional[JarvisClientConfig] = None) -> None:
        self.config = config or JarvisClientConfig()

    # ------------------------------------------------------------------
    # Memory query
    # ------------------------------------------------------------------

    async def query_memory(
        self,
        question: str,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Query OpsMemory for context relevant to *question*.

        Performs a semantic search over evidence items and consolidated
        memories.  Call this **before** generating an answer so the assistant
        has relevant historical context.

        Parameters
        ----------
        question:
            Natural-language query string.
        limit:
            Maximum number of evidence citations to return.  Defaults to
            ``JarvisClientConfig.query_limit``.

        Returns
        -------
        dict
            ``{"query": str, "answer": str, "citations": [...], "memories": [...]}``
        """
        effective_limit = min(
            limit if limit is not None else self.config.query_limit, 50
        )
        async with httpx.AsyncClient(
            headers=self.config.auth_headers(), timeout=self.config.request_timeout
        ) as client:
            response = await client.get(
                f"{self.config.api_url}/query",
                params={"q": question, "limit": effective_limit},
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Evidence ingestion
    # ------------------------------------------------------------------

    async def ingest_session_outcome(
        self,
        text: str,
        session_id: str = "",
        author: str = "",
        occurred_at: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest the outcome of a Jarvis session or task into OpsMemory.

        Call this **after** a task completes so the result is persisted for
        future queries.

        Parameters
        ----------
        text:
            Human-readable description of what happened in the session.
        session_id:
            Stable identifier for this session or task (used as ``native_id``).
        author:
            Name or identifier of the assistant / agent that produced this
            outcome.
        occurred_at:
            ISO-8601 timestamp of when the session completed.
        metadata:
            Optional additional key/value context.

        Returns
        -------
        dict
            OpsMemory ingest response, including ``evidence_id``.
        """
        payload: Dict[str, Any] = {
            "text": text,
            "source_type": self.config.session_source_type,
            "source_ref": f"jarvis://session/{session_id}" if session_id else "",
            "author": author,
            "native_id": session_id or None,
            "occurred_at": occurred_at or None,
            "metadata": metadata or {},
        }
        async with httpx.AsyncClient(
            headers=self.config.auth_headers(), timeout=self.config.request_timeout
        ) as client:
            response = await client.post(
                f"{self.config.api_url}/ingest", json=payload
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    async def ingest_text(
        self,
        text: str,
        source_type: str = "manual",
        source_ref: str = "",
        author: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest arbitrary text evidence into OpsMemory.

        A lower-level alternative to ``ingest_session_outcome`` for when you
        need full control over the source type.
        """
        payload: Dict[str, Any] = {
            "text": text,
            "source_type": source_type,
            "source_ref": source_ref,
            "author": author,
            "metadata": metadata or {},
        }
        async with httpx.AsyncClient(
            headers=self.config.auth_headers(), timeout=self.config.request_timeout
        ) as client:
            response = await client.post(
                f"{self.config.api_url}/ingest", json=payload
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Status / sources
    # ------------------------------------------------------------------

    async def get_status(self) -> Dict[str, Any]:
        """Return current OpsMemory store counts.

        Returns
        -------
        dict
            ``{"evidence_total": int, "evidence_unconsolidated": int, "memories": int}``
        """
        async with httpx.AsyncClient(
            headers=self.config.auth_headers(), timeout=self.config.request_timeout
        ) as client:
            response = await client.get(f"{self.config.api_url}/status")
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    async def list_sources(self) -> List[Dict[str, Any]]:
        """Return registered data sources from OpsMemory."""
        async with httpx.AsyncClient(
            headers=self.config.auth_headers(), timeout=self.config.request_timeout
        ) as client:
            response = await client.get(f"{self.config.api_url}/sources")
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    async def list_memories(self) -> List[Dict[str, Any]]:
        """Return consolidated memory records."""
        async with httpx.AsyncClient(
            headers=self.config.auth_headers(), timeout=self.config.request_timeout
        ) as client:
            response = await client.get(f"{self.config.api_url}/memories")
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    async def trigger_consolidation(self) -> Dict[str, Any]:
        """Trigger an OpsMemory consolidation cycle.

        Returns
        -------
        dict
            ``{"run_id": str, "memories_created": int, "evidence_consolidated": int}``
        """
        async with httpx.AsyncClient(
            headers=self.config.auth_headers(), timeout=self.config.request_timeout
        ) as client:
            response = await client.post(f"{self.config.api_url}/consolidate")
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# TODO: native MCP transport adapter
# ---------------------------------------------------------------------------
#
# If your Jarvis deployment uses a native MCP SDK (e.g. the official
# ``mcp`` Python package) and the OpsMemory MCP server is running with
# SSE transport, wire the client here instead of using the REST API.
#
# Example skeleton (adapt to your MCP SDK):
#
#   class OpsMemoryMCPTransportClient:
#       """Calls OpsMemory via MCP SSE transport."""
#
#       def __init__(self, config: JarvisClientConfig):
#           self.config = config
#           # TODO: initialise your MCP SDK client here, pointing at
#           #       self.config.mcp_url (default: http://localhost:8100)
#           raise NotImplementedError(
#               "Replace this with your MCP SDK transport wiring."
#           )
#
#       async def call_tool(self, tool_name: str, **kwargs) -> Any:
#           # TODO: dispatch tool call via MCP protocol
#           raise NotImplementedError
#
# ---------------------------------------------------------------------------
