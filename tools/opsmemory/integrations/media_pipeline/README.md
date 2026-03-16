# Media Processing → OpsMemory Integration

Reusable producer-side scaffolding for pushing normalised evidence from a
media processing pipeline into OpsMemory.

---

## Overview

This directory provides a generic, resilient HTTP ingestion client and
normalised payload models that media pipeline stages can use to persist
evidence into OpsMemory for semantic search and consolidation.

Supported evidence types out of the box:

| Model | Source type | Use case |
|-------|-------------|---------|
| `TranscriptResult` | `media_transcript` | Speech-to-text output |
| `OcrExtractionResult` | `media_ocr` | OCR from images / PDFs |
| `MediaMetadataResult` | `media_metadata` | Asset technical metadata |
| `EnrichmentResult` | `media_enrichment` | Downstream NLP / annotation |

---

## Files

| File | Purpose |
|------|---------|
| `models.py` | Normalised payload models (`TranscriptResult`, `OcrExtractionResult`, etc.) |
| `client.py` | `MediaIngestionClient` — async HTTP client with retry / back-off |
| `examples.py` | Four canonical usage examples |

---

## Quick Start

### 1. Configure environment

```bash
cp tools/opsmemory/.env.media.example .env
# Edit .env — set OPSMEMORY_API_URL, OPSMEMORY_API_KEY
```

### 2. Ingest a transcript

```python
import asyncio
from tools.opsmemory.integrations.media_pipeline.client import MediaIngestionClient
from tools.opsmemory.integrations.media_pipeline.models import TranscriptResult

async def after_transcription(text: str, asset_id: str) -> None:
    client = MediaIngestionClient()        # reads config from env
    result = TranscriptResult(
        text=text,
        source_ref=f"media://recordings/{asset_id}.wav",
        native_id=asset_id,
        occurred_at="2026-03-15T14:00:00Z",
        language="en",
        confidence=0.96,
    )
    response = await client.ingest(result)
    print("Ingested:", response["evidence_id"])

asyncio.run(after_transcription("Deployment succeeded.", "session-001"))
```

### 3. Batch ingest

```python
results = await client.ingest_batch([transcript, ocr_result, enrichment])
```

### 4. Run examples

```bash
uvicorn tools.opsmemory.api.app:app --port 8000
python -m tools.opsmemory.integrations.media_pipeline.examples
```

---

## Payload Normalisation

All models implement `to_ingest_payload()` which serialises to the OpsMemory
`POST /ingest` schema.  Key fields:

| Field | Description |
|-------|-------------|
| `text` | Human-readable extracted or derived text (the primary searchable content) |
| `source_type` | Stable label for the evidence type (e.g. `media_transcript`) |
| `source_ref` | Stable URI / path that uniquely identifies the asset |
| `native_id` | Asset or job ID in the upstream system |
| `occurred_at` | ISO-8601 timestamp of the original event |
| `metadata` | Open dict for pipeline-specific fields |

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPSMEMORY_API_URL` | `http://localhost:8000` | OpsMemory REST API base URL |
| `OPSMEMORY_API_KEY` | _(none)_ | Bearer token (required when auth is enabled) |
| `MEDIA_INGESTION_MAX_RETRIES` | `3` | Max retry attempts on transient failures |
| `MEDIA_INGESTION_RETRY_BACKOFF` | `2` | Base back-off multiplier (seconds) |
| `OPSMEMORY_REQUEST_TIMEOUT` | `30` | HTTP request timeout (seconds) |

---

## Retry / Error Handling

The client uses exponential back-off retry on `429` and `5xx` responses:

- **Attempt 1** — immediate
- **Attempt 2** — wait `backoff × 2⁰` seconds
- **Attempt 3** — wait `backoff × 2¹` seconds
- After `max_retries` the exception is re-raised.

`ingest_batch` logs and skips individual item failures, returning results for
all successfully ingested items.

---

## Adding a New Evidence Type

Subclass `MediaEvidenceBase`:

```python
from tools.opsmemory.integrations.media_pipeline.models import MediaEvidenceBase
from dataclasses import dataclass

@dataclass
class SentimentResult(MediaEvidenceBase):
    source_type: str = "media_sentiment"
    sentiment: str = ""        # "positive", "negative", "neutral"
    score: float = 0.0

    def __post_init__(self):
        if self.sentiment:
            self.metadata.setdefault("sentiment", self.sentiment)
        if self.score:
            self.metadata.setdefault("score", self.score)
```

Then use it with the standard client:

```python
result = SentimentResult(
    text="Overall sentiment: positive (0.82)",
    source_ref="media://recordings/session-001.wav",
    sentiment="positive",
    score=0.82,
)
await client.ingest(result)
```

---

## Public-Repo Safety

- No personal identifiers, internal endpoints, or sensitive workflow data.
- All configuration via environment variables.
- Example payloads use generic, placeholder-based content only.
