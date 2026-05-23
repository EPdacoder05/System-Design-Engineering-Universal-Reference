"""Local-first API key authentication for OpsMemory.

Provides a lightweight, cloud-IdP-free auth layer based on shared API keys
supplied via environment variables.  Suitable for self-hosted / local deployments.

Configuration
-------------
``OPSMEMORY_REQUIRE_API_KEY``
    Set to ``true`` / ``1`` / ``yes`` to enforce authentication on all protected
    endpoints.  Defaults to ``false`` — auth is *off* for local development.

``OPSMEMORY_API_KEY``
    The bearer token that clients must supply in an ``Authorization: Bearer <key>``
    header.  Required when auth is enabled.  Generate a strong random value, e.g.::

        python -c "import secrets; print(secrets.token_urlsafe(32))"

    Store it in your ``.env`` file or a secret manager — never hardcode it.

``OPSMEMORY_MCP_TOKEN``
    Optional separate token for MCP server callers.  When set, MCP tool wrappers
    that call the OpsMemory API can read this token and include it automatically.
    If unset, falls back to ``OPSMEMORY_API_KEY``.

Usage (FastAPI middleware)
--------------------------
The ``apply_auth_middleware`` helper registers an HTTP middleware on a
``FastAPI`` application instance so that every non-exempt route is
automatically protected::

    from fastapi import FastAPI
    from tools.opsmemory.auth import apply_auth_middleware

    app = FastAPI(...)
    apply_auth_middleware(app)

Usage (FastAPI Depends)
-----------------------
You can also apply ``require_api_key`` as a per-route dependency::

    from fastapi import Depends
    from tools.opsmemory.auth import require_api_key

    @router.post("/ingest", dependencies=[Depends(require_api_key)])
    async def ingest(...): ...
"""

from __future__ import annotations

import os
import secrets

import structlog
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------

#: Paths that are always accessible even when auth is enabled.
_EXEMPT_PATHS: frozenset[str] = frozenset(
    {"/health", "/ready", "/docs", "/openapi.json", "/redoc"}
)


def auth_enabled() -> bool:
    """Return ``True`` if API key auth is required (``OPSMEMORY_REQUIRE_API_KEY``)."""
    return os.environ.get("OPSMEMORY_REQUIRE_API_KEY", "false").lower() in (
        "1",
        "true",
        "yes",
    )


def get_api_key() -> str:
    """Return the configured API key from ``OPSMEMORY_API_KEY`` (may be empty)."""
    return os.environ.get("OPSMEMORY_API_KEY", "")


def get_mcp_token() -> str:
    """Return the MCP token, falling back to the API key if not set."""
    return os.environ.get("OPSMEMORY_MCP_TOKEN", "") or get_api_key()


def _extract_bearer_token(authorization: str) -> str:
    """Strip the ``Bearer`` prefix and return the raw token."""
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return authorization.strip()


def _validate_token(token: str) -> bool:
    """Constant-time comparison of *token* against the configured API key.

    Returns ``False`` if no API key is configured, preventing accidental
    open access.
    """
    configured = get_api_key()
    if not configured:
        return False
    return secrets.compare_digest(token, configured)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def require_api_key(authorization: str = Header(default="")) -> None:
    """FastAPI dependency that enforces API key auth when enabled.

    When ``OPSMEMORY_REQUIRE_API_KEY`` is ``false`` (default), this is a no-op.
    When enabled, the request must supply ``Authorization: Bearer <key>``.
    """
    if not auth_enabled():
        return

    if not get_api_key():
        log.error(
            "auth_misconfigured",
            detail="OPSMEMORY_API_KEY is not set but auth is enabled",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "API authentication is enabled but OPSMEMORY_API_KEY is not "
                "configured on the server."
            ),
        )

    token = _extract_bearer_token(authorization)
    if not token or not _validate_token(token):
        log.warning("auth_rejected", has_token=bool(token))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# FastAPI middleware helper
# ---------------------------------------------------------------------------


def apply_auth_middleware(app: FastAPI) -> None:
    """Register the auth middleware on *app*.

    The middleware protects every route except those in ``_EXEMPT_PATHS``
    (``/health``, ``/ready``, ``/docs``, etc.).  When auth is disabled via
    ``OPSMEMORY_REQUIRE_API_KEY=false`` (the default), the middleware is
    installed but is a no-op for every request.
    """

    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        if not auth_enabled():
            return await call_next(request)

        configured_key = get_api_key()
        if not configured_key:
            log.error(
                "auth_misconfigured",
                path=request.url.path,
                detail="OPSMEMORY_API_KEY is not set",
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "detail": (
                        "API authentication is enabled but OPSMEMORY_API_KEY "
                        "is not configured on the server."
                    )
                },
            )

        auth_header = request.headers.get("authorization", "")
        token = _extract_bearer_token(auth_header)
        if not token or not secrets.compare_digest(token, configured_key):
            log.warning("auth_rejected", path=request.url.path, has_token=bool(token))
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key."},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)
