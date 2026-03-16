"""Tests for the Media Processing → OpsMemory ingestion client.

Validates model serialisation, client configuration, ingestion (single +
batch), retry logic, and error handling with mocked HTTP transport.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest

from tools.opsmemory.integrations.media_pipeline.client import (
    MediaIngestionClient,
    _should_retry,
)
from tools.opsmemory.integrations.media_pipeline.models import (
    EnrichmentResult,
    MediaEvidenceBase,
    MediaMetadataResult,
    OcrExtractionResult,
    TranscriptResult,
)


# ---------------------------------------------------------------------------
# Models — to_ingest_payload
# ---------------------------------------------------------------------------


def test_transcript_result_payload():
    r = TranscriptResult(
        text="Speaker A: hello.",
        source_ref="media://recordings/test.wav",
        native_id="test-001",
        occurred_at="2026-03-15T09:00:00Z",
        language="en",
        confidence=0.95,
        duration_seconds=60.0,
    )
    payload = r.to_ingest_payload()
    assert payload["text"] == "Speaker A: hello."
    assert payload["source_type"] == "media_transcript"
    assert payload["source_ref"] == "media://recordings/test.wav"
    assert payload["native_id"] == "test-001"
    assert payload["occurred_at"] == "2026-03-15T09:00:00Z"
    assert payload["metadata"]["language"] == "en"
    assert payload["metadata"]["confidence"] == 0.95
    assert payload["metadata"]["duration_seconds"] == 60.0


def test_ocr_result_payload():
    r = OcrExtractionResult(
        text="Invoice #12345",
        source_ref="media://docs/invoice.pdf",
        native_id="invoice-12345",
        page_count=2,
        confidence=0.99,
        doc_format="application/pdf",
    )
    payload = r.to_ingest_payload()
    assert payload["source_type"] == "media_ocr"
    assert payload["metadata"]["page_count"] == 2
    assert payload["metadata"]["doc_format"] == "application/pdf"


def test_media_metadata_payload():
    r = MediaMetadataResult(
        text="Video: 1080p 30fps",
        source_ref="media://videos/demo.mp4",
        asset_type="video",
        duration_seconds=120.0,
        file_size_bytes=1_000_000,
        mime_type="video/mp4",
    )
    payload = r.to_ingest_payload()
    assert payload["source_type"] == "media_metadata"
    assert payload["metadata"]["asset_type"] == "video"
    assert payload["metadata"]["mime_type"] == "video/mp4"


def test_enrichment_result_payload():
    r = EnrichmentResult(
        text="Entities: Alice (PERSON)",
        source_ref="media://recordings/test.wav",
        enrichment_type="entity_extraction",
        model="generic-ner-v1",
        labels=["PERSON"],
    )
    payload = r.to_ingest_payload()
    assert payload["source_type"] == "media_enrichment"
    assert payload["metadata"]["enrichment_type"] == "entity_extraction"
    assert "PERSON" in payload["metadata"]["labels"]


def test_base_to_ingest_payload_omits_none_fields():
    r = MediaEvidenceBase(
        text="Some text",
        source_type="media_custom",
        source_ref="",
        repo="",
        native_id="",
        occurred_at="",
    )
    payload = r.to_ingest_payload()
    assert payload["repo"] is None
    assert payload["native_id"] is None
    assert payload["occurred_at"] is None


# ---------------------------------------------------------------------------
# MediaIngestionClient — configuration
# ---------------------------------------------------------------------------


def test_client_reads_env_vars():
    env = {
        "OPSMEMORY_API_URL": "http://custom:8000",
        "OPSMEMORY_API_KEY": "env-key",
        "MEDIA_INGESTION_MAX_RETRIES": "5",
        "MEDIA_INGESTION_RETRY_BACKOFF": "3",
        "OPSMEMORY_REQUEST_TIMEOUT": "60",
    }
    with patch.dict(os.environ, env):
        client = MediaIngestionClient()

    assert client.api_url == "http://custom:8000"
    assert client.api_key == "env-key"
    assert client.max_retries == 5
    assert client.retry_backoff == 3.0
    assert client.request_timeout == 60.0


def test_client_explicit_overrides():
    client = MediaIngestionClient(
        api_url="http://override:9000",
        api_key="override-key",
        max_retries=1,
    )
    assert client.api_url == "http://override:9000"
    assert client.api_key == "override-key"
    assert client.max_retries == 1


def test_client_strips_trailing_slash():
    client = MediaIngestionClient(api_url="http://server:8000/")
    assert not client.api_url.endswith("/")


def test_auth_header_included_when_key_set():
    client = MediaIngestionClient(api_key="my-token")
    headers = client._headers
    assert headers["Authorization"] == "Bearer my-token"


def test_auth_header_absent_when_no_key():
    client = MediaIngestionClient(api_key="")
    assert "Authorization" not in client._headers


# ---------------------------------------------------------------------------
# _should_retry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
def test_should_retry_retryable_codes(status_code):
    mock_response = MagicMock()
    mock_response.status_code = status_code
    exc = httpx.HTTPStatusError(
        message="error", request=MagicMock(), response=mock_response
    )
    assert _should_retry(exc) is True


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_should_not_retry_client_errors(status_code):
    mock_response = MagicMock()
    mock_response.status_code = status_code
    exc = httpx.HTTPStatusError(
        message="error", request=MagicMock(), response=mock_response
    )
    assert _should_retry(exc) is False


def test_should_not_retry_non_http_exceptions():
    assert _should_retry(ValueError("oops")) is False
    assert _should_retry(ConnectionError("conn")) is False


# ---------------------------------------------------------------------------
# MediaIngestionClient — ingest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_posts_payload_to_ingest_endpoint():
    client = MediaIngestionClient(api_url="http://test:8000", api_key="key")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"evidence_id": "ev-001"}
    mock_response.status_code = 200

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.post = AsyncMock(return_value=mock_response)

    evidence = TranscriptResult(
        text="Test transcript.",
        source_ref="media://recordings/x.wav",
        native_id="x-001",
        occurred_at="2026-03-15T10:00:00Z",
    )

    with patch("httpx.AsyncClient", return_value=mock_http_client):
        result = await client.ingest(evidence)

    assert result["evidence_id"] == "ev-001"
    posted = mock_http_client.post.call_args.kwargs["json"]
    assert posted["text"] == "Test transcript."
    assert posted["source_type"] == "media_transcript"


@pytest.mark.asyncio
async def test_ingest_includes_auth_header():
    client = MediaIngestionClient(api_url="http://test:8000", api_key="auth-token")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"evidence_id": "ev-002"}
    mock_response.status_code = 200

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.post = AsyncMock(return_value=mock_response)

    captured_headers = {}

    def _capture(**kwargs):
        captured_headers.update(kwargs.get("headers", {}))
        return mock_http_client

    evidence = OcrExtractionResult(
        text="Some OCR text.", source_ref="media://docs/test.pdf"
    )

    with patch("httpx.AsyncClient", side_effect=_capture):
        await client.ingest(evidence)

    assert captured_headers.get("Authorization") == "Bearer auth-token"


# ---------------------------------------------------------------------------
# MediaIngestionClient — ingest_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_batch_returns_all_successful():
    client = MediaIngestionClient(api_url="http://test:8000")

    items = [
        TranscriptResult(text=f"Item {i}", source_ref=f"ref-{i}")
        for i in range(3)
    ]

    call_count = 0

    async def mock_ingest(evidence):
        nonlocal call_count
        call_count += 1
        return {"evidence_id": f"ev-{call_count}"}

    with patch.object(client, "ingest", side_effect=mock_ingest):
        results = await client.ingest_batch(items)

    assert len(results) == 3
    assert call_count == 3


@pytest.mark.asyncio
async def test_ingest_batch_skips_failed_items_and_continues():
    client = MediaIngestionClient(api_url="http://test:8000")

    items = [
        TranscriptResult(text=f"Item {i}", source_ref=f"ref-{i}")
        for i in range(3)
    ]

    call_count = 0

    async def mock_ingest(evidence):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise httpx.ConnectError("connection refused")
        return {"evidence_id": f"ev-{call_count}"}

    with patch.object(client, "ingest", side_effect=mock_ingest):
        results = await client.ingest_batch(items)

    # Item 2 failed; items 1 and 3 should succeed.
    assert len(results) == 2
    assert call_count == 3


# ---------------------------------------------------------------------------
# MediaIngestionClient — health_check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_returns_true_on_200():
    client = MediaIngestionClient(api_url="http://test:8000")

    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_http_client):
        result = await client.health_check()

    assert result is True


@pytest.mark.asyncio
async def test_health_check_returns_false_on_connection_error():
    client = MediaIngestionClient(api_url="http://test:8000")

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

    with patch("httpx.AsyncClient", return_value=mock_http_client):
        result = await client.health_check()

    assert result is False
