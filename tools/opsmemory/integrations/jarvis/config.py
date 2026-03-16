"""Configuration for the Jarvis → OpsMemory MCP client integration.

All settings are read from environment variables so that no credentials or
deployment-specific values are ever hardcoded.  Copy
``tools/opsmemory/.env.jarvis.example`` to ``.env`` and fill in your values.

Environment variables
---------------------
``OPSMEMORY_API_URL``
    Base URL of the running OpsMemory FastAPI service.
    Default: ``http://localhost:8000``.

``OPSMEMORY_API_KEY``
    Bearer token for the OpsMemory API.  Required when the server is started
    with ``OPSMEMORY_REQUIRE_API_KEY=true``.  Leave empty for local-only
    development (auth disabled).

``OPSMEMORY_MCP_TOKEN``
    Optional separate bearer token for MCP tool calls.  Falls back to
    ``OPSMEMORY_API_KEY`` when unset.

``OPSMEMORY_MCP_URL``
    Base URL of the OpsMemory FastMCP server when using SSE transport.
    Default: ``http://localhost:8100``.  Only needed when ``transport=sse``.

``JARVIS_MEMORY_QUERY_LIMIT``
    Maximum number of evidence items returned by a memory query.
    Default: ``5``.

``JARVIS_SESSION_SOURCE_TYPE``
    Source type label used when ingesting Jarvis session outcomes.
    Default: ``jarvis_session``.

``OPSMEMORY_REQUEST_TIMEOUT``
    HTTP request timeout in seconds for API calls.
    Default: ``30``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class JarvisClientConfig:
    """Runtime configuration for the Jarvis OpsMemory client.

    Instantiate with no arguments to read all values from the environment::

        cfg = JarvisClientConfig()

    Or override individual fields for testing::

        cfg = JarvisClientConfig(api_url="http://test-server:8000", api_key="test-key")
    """

    # Base URL of the OpsMemory FastAPI service.
    api_url: str = field(
        default_factory=lambda: os.environ.get(
            "OPSMEMORY_API_URL", "http://localhost:8000"
        )
    )

    # Bearer token for authenticated OpsMemory API calls.
    api_key: str = field(
        default_factory=lambda: os.environ.get("OPSMEMORY_API_KEY", "")
        or os.environ.get("OPSMEMORY_MCP_TOKEN", "")
    )

    # Base URL for the OpsMemory FastMCP SSE server (only used in SSE mode).
    mcp_url: str = field(
        default_factory=lambda: os.environ.get(
            "OPSMEMORY_MCP_URL", "http://localhost:8100"
        )
    )

    # Maximum evidence items returned per query.
    query_limit: int = field(
        default_factory=lambda: int(
            os.environ.get("JARVIS_MEMORY_QUERY_LIMIT", "5")
        )
    )

    # Source type used when ingesting Jarvis session/task outcomes.
    session_source_type: str = field(
        default_factory=lambda: os.environ.get(
            "JARVIS_SESSION_SOURCE_TYPE", "jarvis_session"
        )
    )

    # HTTP request timeout in seconds.
    request_timeout: float = field(
        default_factory=lambda: float(
            os.environ.get("OPSMEMORY_REQUEST_TIMEOUT", "30")
        )
    )

    def auth_headers(self) -> dict[str, str]:
        """Return ``Authorization`` header dict if an API key is configured."""
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}
