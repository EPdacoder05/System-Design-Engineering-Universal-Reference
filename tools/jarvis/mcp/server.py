"""Jarvis — MCP server for homelab vault operations.

Tools registered
----------------
- ``jarvis_ingest``  — compile raw event files into wiki dashboards
- ``jarvis_query``   — answer questions from compiled wiki state
- ``jarvis_lint``    — health-check wiki state against live system reality
- ``jarvis_log``     — quick-capture a note into the journal

Running the server
------------------
SDK / stdio transport (default for MCP clients)::

    python -m tools.jarvis.mcp.server

Environment variables
---------------------
``JARVIS_VAULT_PATH``
    Absolute path to the vault root directory.
    Defaults to the ``vault/`` directory next to this package.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fastmcp import FastMCP

from tools.jarvis.mcp.tools.ingest import jarvis_ingest
from tools.jarvis.mcp.tools.lint import jarvis_lint
from tools.jarvis.mcp.tools.log import jarvis_log
from tools.jarvis.mcp.tools.query import jarvis_query

# ---------------------------------------------------------------------------
# Default vault path — two levels up from this file: tools/jarvis/vault/
# ---------------------------------------------------------------------------
_DEFAULT_VAULT = Path(__file__).parent.parent / "vault"


def _vault_path() -> Path:
    env = os.environ.get("JARVIS_VAULT_PATH")
    return Path(env) if env else _DEFAULT_VAULT


# ---------------------------------------------------------------------------
# Server instantiation
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "jarvis",
    instructions=(
        "Jarvis homelab vault assistant. "
        "Use jarvis_ingest to compile raw events, jarvis_query to answer questions "
        "from compiled wiki state, jarvis_lint to check system health, and "
        "jarvis_log to capture quick notes."
    ),
)

mcp.tool()(jarvis_ingest)
mcp.tool()(jarvis_query)
mcp.tool()(jarvis_lint)
mcp.tool()(jarvis_log)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jarvis MCP server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run()
