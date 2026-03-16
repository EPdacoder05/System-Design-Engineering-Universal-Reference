"""Tests for the OpsMemory local-first auth module.

Covers:
- auth_enabled / get_api_key helpers
- require_api_key dependency (enabled / disabled)
- apply_auth_middleware (enabled / disabled, exempt paths, bad token, missing key)
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tools.opsmemory import auth as auth_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_app_with_middleware(env: dict) -> TestClient:
    """Create a minimal FastAPI app with auth middleware applied under *env*."""
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        return {"status": "ready"}

    @app.get("/protected")
    async def protected():
        return {"secret": "data"}

    @app.post("/ingest")
    async def ingest():
        return {"evidence_id": "test-id"}

    with patch.dict(os.environ, env, clear=False):
        # Re-import to pick up the patched env (middleware reads env at
        # request time, not import time).
        auth_module.apply_auth_middleware(app)

    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# auth_enabled / get_api_key
# ---------------------------------------------------------------------------


def test_auth_enabled_false_by_default():
    with patch.dict(os.environ, {}, clear=True):
        # Remove OPSMEMORY_REQUIRE_API_KEY entirely
        os.environ.pop("OPSMEMORY_REQUIRE_API_KEY", None)
        assert auth_module.auth_enabled() is False


@pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "YES"])
def test_auth_enabled_truthy_values(value):
    with patch.dict(os.environ, {"OPSMEMORY_REQUIRE_API_KEY": value}):
        assert auth_module.auth_enabled() is True


@pytest.mark.parametrize("value", ["false", "False", "0", "no", ""])
def test_auth_enabled_falsy_values(value):
    with patch.dict(os.environ, {"OPSMEMORY_REQUIRE_API_KEY": value}):
        assert auth_module.auth_enabled() is False


def test_get_api_key_returns_env_value():
    with patch.dict(os.environ, {"OPSMEMORY_API_KEY": "my-secret-key"}):
        assert auth_module.get_api_key() == "my-secret-key"


def test_get_api_key_empty_when_not_set():
    env = dict(os.environ)
    env.pop("OPSMEMORY_API_KEY", None)
    with patch.dict(os.environ, env, clear=True):
        assert auth_module.get_api_key() == ""


def test_get_mcp_token_falls_back_to_api_key():
    env = {"OPSMEMORY_API_KEY": "api-key", "OPSMEMORY_MCP_TOKEN": ""}
    with patch.dict(os.environ, env):
        assert auth_module.get_mcp_token() == "api-key"


def test_get_mcp_token_uses_dedicated_token():
    env = {"OPSMEMORY_API_KEY": "api-key", "OPSMEMORY_MCP_TOKEN": "mcp-token"}
    with patch.dict(os.environ, env):
        assert auth_module.get_mcp_token() == "mcp-token"


# ---------------------------------------------------------------------------
# Middleware — auth DISABLED (default)
# ---------------------------------------------------------------------------


def test_protected_endpoint_accessible_when_auth_disabled():
    env = {"OPSMEMORY_REQUIRE_API_KEY": "false"}
    client = _fresh_app_with_middleware(env)
    resp = client.get("/protected")
    assert resp.status_code == 200


def test_health_accessible_when_auth_disabled():
    env = {"OPSMEMORY_REQUIRE_API_KEY": "false"}
    client = _fresh_app_with_middleware(env)
    assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# Middleware — auth ENABLED
# ---------------------------------------------------------------------------


def test_health_exempt_when_auth_enabled():
    """GET /health must return 200 even when auth is enabled (health probe)."""
    env = {
        "OPSMEMORY_REQUIRE_API_KEY": "true",
        "OPSMEMORY_API_KEY": "test-api-key",
    }
    client = _fresh_app_with_middleware(env)
    resp = client.get("/health")
    assert resp.status_code == 200


def test_ready_exempt_when_auth_enabled():
    """GET /ready must return 200 even when auth is enabled."""
    env = {
        "OPSMEMORY_REQUIRE_API_KEY": "true",
        "OPSMEMORY_API_KEY": "test-api-key",
    }
    client = _fresh_app_with_middleware(env)
    resp = client.get("/ready")
    assert resp.status_code == 200


def test_protected_endpoint_returns_401_without_token():
    env = {
        "OPSMEMORY_REQUIRE_API_KEY": "true",
        "OPSMEMORY_API_KEY": "test-api-key",
    }
    client = _fresh_app_with_middleware(env)
    with patch.dict(os.environ, env):
        resp = client.get("/protected")
    assert resp.status_code == 401
    assert "detail" in resp.json()


def test_protected_endpoint_returns_401_with_wrong_token():
    env = {
        "OPSMEMORY_REQUIRE_API_KEY": "true",
        "OPSMEMORY_API_KEY": "correct-key",
    }
    client = _fresh_app_with_middleware(env)
    with patch.dict(os.environ, env):
        resp = client.get("/protected", headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 401


def test_protected_endpoint_accessible_with_correct_token():
    env = {
        "OPSMEMORY_REQUIRE_API_KEY": "true",
        "OPSMEMORY_API_KEY": "correct-key",
    }
    client = _fresh_app_with_middleware(env)
    with patch.dict(os.environ, env):
        resp = client.get(
            "/protected", headers={"Authorization": "Bearer correct-key"}
        )
    assert resp.status_code == 200


def test_post_ingest_accessible_with_correct_token():
    env = {
        "OPSMEMORY_REQUIRE_API_KEY": "true",
        "OPSMEMORY_API_KEY": "correct-key",
    }
    client = _fresh_app_with_middleware(env)
    with patch.dict(os.environ, env):
        resp = client.post(
            "/ingest", headers={"Authorization": "Bearer correct-key"}
        )
    assert resp.status_code == 200


def test_returns_503_when_auth_enabled_but_key_not_set():
    """When auth is on but OPSMEMORY_API_KEY is empty, return 503."""
    env = {
        "OPSMEMORY_REQUIRE_API_KEY": "true",
        "OPSMEMORY_API_KEY": "",
    }
    client = _fresh_app_with_middleware(env)
    with patch.dict(os.environ, env):
        resp = client.get("/protected", headers={"Authorization": "Bearer anything"})
    assert resp.status_code == 503


def test_www_authenticate_header_on_401():
    """401 responses must include WWW-Authenticate: Bearer."""
    env = {
        "OPSMEMORY_REQUIRE_API_KEY": "true",
        "OPSMEMORY_API_KEY": "some-key",
    }
    client = _fresh_app_with_middleware(env)
    with patch.dict(os.environ, env):
        resp = client.get("/protected")
    assert resp.status_code == 401
    assert "www-authenticate" in {k.lower() for k in resp.headers}
