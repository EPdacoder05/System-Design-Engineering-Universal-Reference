"""Tests for the Jarvis MCP server registration.

Validates:
- The FastMCP server instance imports without error.
- All four expected tools are registered.
- Each tool has a non-empty description.
"""

from __future__ import annotations

import asyncio



def test_server_imports():
    """tools.jarvis.mcp.server must import without raising."""
    import tools.jarvis.mcp.server as server  # noqa: F401

    assert server.mcp is not None


def test_server_has_four_tools():
    """All four Jarvis tools must be registered on the FastMCP instance."""
    from tools.jarvis.mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    tool_names = {t.name for t in tools}
    expected = {"jarvis_ingest", "jarvis_query", "jarvis_lint", "jarvis_log"}
    assert expected.issubset(tool_names), f"Missing tools: {expected - tool_names}"


def test_all_tools_have_descriptions():
    """Every registered tool must have a non-empty description."""
    from tools.jarvis.mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    for tool in tools:
        if tool.name.startswith("jarvis_"):
            assert tool.description, f"{tool.name} has no description"
