"""FastMCP server for OpsMemory.

Exposes core OpsMemory functionality as MCP tools accessible to any MCP
client (Claude Desktop, Cursor, IDE extensions, etc.).

Tools registered
----------------
- ``memory_ingest_text``       – ingest free-form text evidence
- ``memory_ingest_repo``       – ingest all docs from a local repository tree
- ``memory_query``             – semantic search over evidence + memories
- ``memory_status``            – current store counts
- ``memory_consolidate``       – trigger a consolidation cycle
- ``memory_list_sources``      – list registered data sources
- ``memory_sync_github_owner`` – run a GitHub ingestion sweep
- ``memory_delete_memory``     – delete a consolidated memory record
- ``memory_delete_evidence``   – delete a raw evidence item

Running the server
------------------
SDK / stdio transport (default for MCP clients)::

    python -m tools.opsmemory.mcp.server

HTTP/SSE transport (useful for debugging)::

    python -m tools.opsmemory.mcp.server --transport sse --port 8100

Environment variables
---------------------
``OPSMEMORY_API_URL``
    Base URL of the running OpsMemory FastAPI service.
    Defaults to ``http://localhost:8000``.
``GITHUB_TOKEN`` / ``GITHUB_OWNER``
    Used by the ``memory_sync_github_owner`` tool.
    Set these in your environment or a ``.env`` file — never hardcode them.
"""

from __future__ import annotations

import argparse
import os

from fastmcp import FastMCP

from tools.opsmemory.mcp.tools.admin import (
    memory_consolidate,
    memory_delete_evidence,
    memory_delete_memory,
    memory_list_sources,
    memory_status,
)
from tools.opsmemory.mcp.tools.github import memory_sync_github_owner
from tools.opsmemory.mcp.tools.ingest import memory_ingest_text
from tools.opsmemory.mcp.tools.query import memory_query
from tools.opsmemory.mcp.tools.repo import memory_ingest_repo

# ---------------------------------------------------------------------------
# Server instantiation
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="OpsMemory",
    instructions=(
        "OpsMemory is an always-on AI memory platform backed by PostgreSQL + "
        "pgvector.  Use these tools to ingest, query, and manage memory records "
        "derived from GitHub activity and operational events."
    ),
)

# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

mcp.tool()(memory_ingest_text)
mcp.tool()(memory_ingest_repo)
mcp.tool()(memory_query)
mcp.tool()(memory_status)
mcp.tool()(memory_consolidate)
mcp.tool()(memory_list_sources)
mcp.tool()(memory_sync_github_owner)
mcp.tool()(memory_delete_memory)
mcp.tool()(memory_delete_evidence)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="OpsMemory FastMCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport to use (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8100,
        help="Port for SSE transport (default: 8100)",
    )
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
