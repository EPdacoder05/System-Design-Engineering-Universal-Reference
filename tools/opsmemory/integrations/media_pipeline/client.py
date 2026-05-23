"""HTTP ingestion client for the Media Processing → OpsMemory integration.

This module provides a resilient async HTTP client that media pipeline stages
can use to push normalised evidence into OpsMemory.

Design goals
------------
- **Generic** — no pipeline-specific business logic; adapts to any media
  processing framework.
- **Resilient** — automatic retry with exponential back-off on transient
  failures (429, 5xx).
- **Public-safe** — no personal identifiers, private endpoints, or sensitive
  workflow details.

Usage
-----
::

    import asyncio
    from tools.opsmemory.integrations.media_pipeline.client import MediaIngestionClient
    from tools.opsmemory.integrations.media_pipeline.models import TranscriptResult

    async def handle_transcript(result: TranscriptResult) -> None:
        client = MediaIngestionClient()          # reads config from env vars
        response = await client.ingest(result)
        print(response["evidence_id"])

    asyncio.run(handle_transcript(
        TranscriptResult(
            text="Speaker A: Deployment completed successfully.",
            source_ref="media://recordings/session-001.wav",
            native_id="session-001",
            occurred_at="2026-03-15T14:00:00Z",
        )
    ))

Environment variables
---------------------
``OPSMEMORY_API_URL``
    Base URL of the OpsMemory API. Default: ``http://localhost:8000``.

``OPSMEMORY_API_KEY``
    Bearer token for authenticated API access.

``MEDIA_INGESTION_MAX_RETRIES``
    Maximum retry attempts on transient errors. Default: ``3``.

``MEDIA_INGESTION_RETRY_BACKOFF``
    Base back-off multiplier in seconds. Default: ``2``.

``OPSMEMORY_REQUEST_TIMEOUT``
    HTTP request timeout in seconds. Default: ``30``.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx
import structlog

from tools.opsmemory.integrations.media_pipeline.models import MediaEvidenceBase

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def _should_retry(exc: BaseException) -> bool:
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code in _RETRYABLE_STATUS_CODES
    )


async def _retry_post(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    max_retries: int,
    backoff: float,
) -> httpx.Response:
    """POST *payload* to *url* with exponential back-off retry."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries and _should_retry(exc):
                wait = backoff * (2 ** attempt)
                log.warning(
                    "media_ingest_retry",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                    error=str(exc),
                )
                await asyncio.sleep(wait)
            else:
                raise
    # Should not be reached, but satisfies type checker.
    raise last_exc or RuntimeError("Unexpected retry loop exit")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MediaIngestionClient:
    """Async HTTP client for ingesting media evidence into OpsMemory.

    Instantiate once and reuse across pipeline stages::

        client = MediaIngestionClient()

    Or pass an explicit configuration::

        client = MediaIngestionClient(
            api_url="http://opsmemory:8000",
            api_key="my-secret-key",
        )
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_backoff: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ) -> None:
        self.api_url = (
            api_url
            or os.environ.get("OPSMEMORY_API_URL", "http://localhost:8000")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OPSMEMORY_API_KEY", "")
        self.max_retries = max_retries if max_retries is not None else int(
            os.environ.get("MEDIA_INGESTION_MAX_RETRIES", "3")
        )
        self.retry_backoff = retry_backoff if retry_backoff is not None else float(
            os.environ.get("MEDIA_INGESTION_RETRY_BACKOFF", "2")
        )
        self.request_timeout = request_timeout if request_timeout is not None else float(
            os.environ.get("OPSMEMORY_REQUEST_TIMEOUT", "30")
        )

    @property
    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def ingest(self, evidence: MediaEvidenceBase) -> Dict[str, Any]:
        """Ingest a single media evidence item into OpsMemory.

        Parameters
        ----------
        evidence:
            Any ``MediaEvidenceBase`` subclass (``TranscriptResult``,
            ``OcrExtractionResult``, etc.).

        Returns
        -------
        dict
            OpsMemory ingest response, including ``evidence_id``.

        Raises
        ------
        httpx.HTTPStatusError
            On non-retryable HTTP errors.
        RuntimeError
            After exhausting all retry attempts.
        """
        payload = evidence.to_ingest_payload()
        url = f"{self.api_url}/ingest"

        async with httpx.AsyncClient(
            headers=self._headers, timeout=self.request_timeout
        ) as client:
            response = await _retry_post(
                client, url, payload, self.max_retries, self.retry_backoff
            )

        result: Dict[str, Any] = response.json()
        log.info(
            "media_ingest_success",
            evidence_id=result.get("evidence_id"),
            source_type=evidence.source_type,
            source_ref=evidence.source_ref,
        )
        return result

    async def ingest_batch(
        self, items: List[MediaEvidenceBase]
    ) -> List[Dict[str, Any]]:
        """Ingest multiple evidence items sequentially.

        Failures on individual items are logged and collected.  Successfully
        ingested items are returned even if some items fail.

        Parameters
        ----------
        items:
            List of ``MediaEvidenceBase`` instances.

        Returns
        -------
        list[dict]
            List of successful ingest response dicts.
        """
        results: List[Dict[str, Any]] = []
        for item in items:
            try:
                result = await self.ingest(item)
                results.append(result)
            except Exception as exc:
                log.error(
                    "media_ingest_item_failed",
                    source_type=item.source_type,
                    source_ref=item.source_ref,
                    error=str(exc),
                )
        return results

    async def health_check(self) -> bool:
        """Return ``True`` if the OpsMemory API is reachable and healthy."""
        try:
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                response = await client.get(f"{self.api_url}/health")
                return response.status_code == 200
        except Exception as exc:
            log.warning("media_client_health_check_failed", error=str(exc))
            return False
