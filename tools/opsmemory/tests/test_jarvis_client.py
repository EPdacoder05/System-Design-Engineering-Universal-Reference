"""Tests for the Jarvis OpsMemory client integration.

Validates config parsing, authentication header injection, and all client
methods with mocked HTTP transport.  No real network calls.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.opsmemory.integrations.jarvis.config import JarvisClientConfig
from tools.opsmemory.integrations.jarvis.mcp_client import OpsMemoryClient


# ---------------------------------------------------------------------------
# JarvisClientConfig
# ---------------------------------------------------------------------------


def test_config_defaults_from_env():
    env = {
        "OPSMEMORY_API_URL": "http://test-server:8000",
        "OPSMEMORY_API_KEY": "test-key",
        "JARVIS_MEMORY_QUERY_LIMIT": "10",
        "JARVIS_SESSION_SOURCE_TYPE": "custom_session",
        "OPSMEMORY_REQUEST_TIMEOUT": "15",
    }
    with patch.dict(os.environ, env):
        cfg = JarvisClientConfig()

    assert cfg.api_url == "http://test-server:8000"
    assert cfg.api_key == "test-key"
    assert cfg.query_limit == 10
    assert cfg.session_source_type == "custom_session"
    assert cfg.request_timeout == 15.0


def test_config_default_values():
    env: dict = {}
    # Ensure keys not set
    for k in ("OPSMEMORY_API_URL", "OPSMEMORY_API_KEY", "OPSMEMORY_MCP_TOKEN",
               "JARVIS_MEMORY_QUERY_LIMIT", "JARVIS_SESSION_SOURCE_TYPE",
               "OPSMEMORY_REQUEST_TIMEOUT"):
        env[k] = ""
    # Override with minimal patching
    clean_env = {k: "" for k in env}
    with patch.dict(os.environ, clean_env):
        cfg = JarvisClientConfig(
            api_url="http://localhost:8000",
            api_key="",
            query_limit=5,
            session_source_type="jarvis_session",
            request_timeout=30.0,
        )
    assert cfg.api_url == "http://localhost:8000"
    assert cfg.query_limit == 5
    assert cfg.session_source_type == "jarvis_session"
    assert cfg.request_timeout == 30.0


def test_auth_headers_with_key():
    cfg = JarvisClientConfig(api_key="my-secret")
    headers = cfg.auth_headers()
    assert headers == {"Authorization": "Bearer my-secret"}


def test_auth_headers_empty_when_no_key():
    cfg = JarvisClientConfig(api_key="")
    assert cfg.auth_headers() == {}


# ---------------------------------------------------------------------------
# OpsMemoryClient — query_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_memory_calls_get_query():
    cfg = JarvisClientConfig(api_url="http://test:8000", api_key="key")
    client = OpsMemoryClient(cfg)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "query": "deployments",
        "answer": "v2.1 deployed",
        "citations": [],
        "memories": [],
    }

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_http_client):
        result = await client.query_memory("deployments", limit=3)

    assert result["query"] == "deployments"
    call_args = mock_http_client.get.call_args
    assert "/query" in call_args.args[0]
    assert call_args.kwargs["params"]["q"] == "deployments"
    assert call_args.kwargs["params"]["limit"] == 3


@pytest.mark.asyncio
async def test_query_memory_clamps_limit_to_50():
    cfg = JarvisClientConfig(api_url="http://test:8000", api_key="")
    client = OpsMemoryClient(cfg)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"query": "x", "answer": "", "citations": [], "memories": []}

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_http_client):
        await client.query_memory("test", limit=9999)

    params = mock_http_client.get.call_args.kwargs["params"]
    assert params["limit"] == 50


@pytest.mark.asyncio
async def test_query_memory_includes_auth_header():
    cfg = JarvisClientConfig(api_url="http://test:8000", api_key="bearer-token")
    client = OpsMemoryClient(cfg)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"query": "x", "answer": "", "citations": [], "memories": []}

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.get = AsyncMock(return_value=mock_response)

    captured_headers = {}

    def _capture_client(**kwargs):
        captured_headers.update(kwargs.get("headers", {}))
        return mock_http_client

    with patch("httpx.AsyncClient", side_effect=_capture_client):
        await client.query_memory("test")

    assert captured_headers.get("Authorization") == "Bearer bearer-token"


# ---------------------------------------------------------------------------
# OpsMemoryClient — ingest_session_outcome
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_session_outcome_posts_to_ingest():
    cfg = JarvisClientConfig(api_url="http://test:8000", api_key="key")
    client = OpsMemoryClient(cfg)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"evidence_id": "ev-001"}

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_http_client):
        result = await client.ingest_session_outcome(
            text="Task completed.",
            session_id="session-001",
            author="jarvis",
            occurred_at="2026-03-15T12:00:00Z",
        )

    assert result["evidence_id"] == "ev-001"
    posted = mock_http_client.post.call_args.kwargs["json"]
    assert posted["text"] == "Task completed."
    assert posted["source_type"] == "jarvis_session"
    assert posted["native_id"] == "session-001"
    assert "jarvis://session/session-001" in posted["source_ref"]
    assert posted["author"] == "jarvis"


@pytest.mark.asyncio
async def test_ingest_session_outcome_uses_config_source_type():
    cfg = JarvisClientConfig(
        api_url="http://test:8000",
        api_key="",
        session_source_type="my_custom_session",
    )
    client = OpsMemoryClient(cfg)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"evidence_id": "ev-002"}

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_http_client):
        await client.ingest_session_outcome(text="Done.")

    posted = mock_http_client.post.call_args.kwargs["json"]
    assert posted["source_type"] == "my_custom_session"


# ---------------------------------------------------------------------------
# OpsMemoryClient — get_status / list_sources / list_memories / consolidate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_status_calls_status_endpoint():
    cfg = JarvisClientConfig(api_url="http://test:8000")
    client = OpsMemoryClient(cfg)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "evidence_total": 42,
        "evidence_unconsolidated": 5,
        "memories": 7,
    }

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_http_client):
        result = await client.get_status()

    assert result["evidence_total"] == 42
    url = mock_http_client.get.call_args.args[0]
    assert "/status" in url


@pytest.mark.asyncio
async def test_trigger_consolidation_posts_to_consolidate():
    cfg = JarvisClientConfig(api_url="http://test:8000")
    client = OpsMemoryClient(cfg)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "run_id": "run-123",
        "memories_created": 3,
        "evidence_consolidated": 30,
    }

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_http_client):
        result = await client.trigger_consolidation()

    assert result["run_id"] == "run-123"
    url = mock_http_client.post.call_args.args[0]
    assert "/consolidate" in url
