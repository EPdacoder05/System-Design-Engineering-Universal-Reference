"""Payload models for the Media Processing → OpsMemory ingestion integration.

Each model represents a normalised evidence payload for a specific media
processing result type.  All models extend ``MediaEvidenceBase`` so the
ingestion client can treat them uniformly.

Design principles
-----------------
- Fields are generic and public-safe — no personal identifiers or private
  workflow details.
- ``source_type`` values follow the convention ``media_<result_type>`` to make
  evidence queryable by type.
- ``source_ref`` should be a stable, reproducible identifier (URI, file path,
  content hash) so the OpsMemory deduplication layer can avoid re-ingesting
  the same item.
- ``metadata`` is an open dict for pipeline-specific enrichment that does not
  belong in the top-level schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


@dataclass
class MediaEvidenceBase:
    """Common fields shared by all media evidence payloads."""

    #: Free-form text extracted or derived from the media asset.
    text: str

    #: OpsMemory source type label (e.g. ``"media_transcript"``).
    source_type: str

    #: Stable reference URI / path / hash that uniquely identifies this asset.
    source_ref: str = ""

    #: Optional author or system that produced this evidence.
    author: str = ""

    #: Optional repository / project association (e.g. ``"org/project"``).
    repo: str = ""

    #: Native identifier within the upstream system (e.g. asset ID, job ID).
    native_id: str = ""

    #: ISO-8601 timestamp when the media event occurred or was captured.
    occurred_at: str = ""

    #: Additional pipeline-specific fields (not stored as first-class columns).
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_ingest_payload(self) -> Dict[str, Any]:
        """Serialise to the OpsMemory ``POST /ingest`` request body."""
        return {
            "text": self.text,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "author": self.author,
            "repo": self.repo or None,
            "native_id": self.native_id or None,
            "occurred_at": self.occurred_at or None,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Transcript result
# ---------------------------------------------------------------------------


@dataclass
class TranscriptResult(MediaEvidenceBase):
    """Evidence produced by a speech-to-text / transcription pipeline stage.

    Example usage::

        result = TranscriptResult(
            text="Speaker A: We released version 2.1 today.",
            source_ref="media://recordings/session-001.wav",
            native_id="session-001",
            occurred_at="2026-03-15T14:00:00Z",
            language="en",
            confidence=0.97,
        )
    """

    source_type: str = "media_transcript"

    #: BCP-47 language tag of the transcribed content (e.g. ``"en"``).
    language: str = ""

    #: Overall confidence score returned by the transcription engine [0, 1].
    confidence: float = 0.0

    #: Duration of the media asset in seconds.
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.language:
            self.metadata.setdefault("language", self.language)
        if self.confidence:
            self.metadata.setdefault("confidence", self.confidence)
        if self.duration_seconds:
            self.metadata.setdefault("duration_seconds", self.duration_seconds)


# ---------------------------------------------------------------------------
# OCR extraction result
# ---------------------------------------------------------------------------


@dataclass
class OcrExtractionResult(MediaEvidenceBase):
    """Evidence produced by an OCR pipeline stage (images, PDFs, scanned docs).

    Example usage::

        result = OcrExtractionResult(
            text="Invoice #12345  Date: 2026-03-15  Total: $500.00",
            source_ref="media://documents/invoice-12345.pdf",
            native_id="invoice-12345",
            page_count=2,
        )
    """

    source_type: str = "media_ocr"

    #: Number of pages processed.
    page_count: int = 0

    #: Average OCR confidence score [0, 1].
    confidence: float = 0.0

    #: Document format / MIME type (e.g. ``"application/pdf"``).
    doc_format: str = ""

    def __post_init__(self) -> None:
        if self.page_count:
            self.metadata.setdefault("page_count", self.page_count)
        if self.confidence:
            self.metadata.setdefault("confidence", self.confidence)
        if self.doc_format:
            self.metadata.setdefault("doc_format", self.doc_format)


# ---------------------------------------------------------------------------
# Media metadata extraction result
# ---------------------------------------------------------------------------


@dataclass
class MediaMetadataResult(MediaEvidenceBase):
    """Evidence produced by a media metadata extraction stage.

    Captures technical properties of an asset (codec, resolution, duration,
    tags) as a text summary for semantic search.

    Example usage::

        result = MediaMetadataResult(
            text="Video asset: 1920x1080, 60fps, H.264, 2 min 30 sec",
            source_ref="media://videos/demo-v1.mp4",
            native_id="demo-v1",
            asset_type="video",
            duration_seconds=150.0,
        )
    """

    source_type: str = "media_metadata"

    #: High-level asset class: ``"video"``, ``"audio"``, ``"image"``, ``"document"``.
    asset_type: str = ""

    #: Duration of the asset in seconds (if applicable).
    duration_seconds: float = 0.0

    #: File size in bytes.
    file_size_bytes: int = 0

    #: MIME type of the asset (e.g. ``"video/mp4"``).
    mime_type: str = ""

    def __post_init__(self) -> None:
        if self.asset_type:
            self.metadata.setdefault("asset_type", self.asset_type)
        if self.duration_seconds:
            self.metadata.setdefault("duration_seconds", self.duration_seconds)
        if self.file_size_bytes:
            self.metadata.setdefault("file_size_bytes", self.file_size_bytes)
        if self.mime_type:
            self.metadata.setdefault("mime_type", self.mime_type)


# ---------------------------------------------------------------------------
# External enrichment result
# ---------------------------------------------------------------------------


@dataclass
class EnrichmentResult(MediaEvidenceBase):
    """Evidence produced by an external enrichment / annotation service.

    Captures the output of any downstream analysis stage (sentiment, entity
    extraction, classification, etc.) as structured text evidence.

    Example usage::

        result = EnrichmentResult(
            text="Entities detected: ProductX (PRODUCT), Alice (PERSON)",
            source_ref="media://recordings/session-001.wav",
            native_id="session-001-entities",
            enrichment_type="entity_extraction",
            model="generic-ner-v1",
            labels=["PRODUCT", "PERSON"],
        )
    """

    source_type: str = "media_enrichment"

    #: Type of enrichment (e.g. ``"entity_extraction"``, ``"sentiment"``, ``"classification"``).
    enrichment_type: str = ""

    #: Model or service that produced the enrichment (generic identifier).
    model: str = ""

    #: Labels or categories assigned by the enrichment stage.
    labels: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.enrichment_type:
            self.metadata.setdefault("enrichment_type", self.enrichment_type)
        if self.model:
            self.metadata.setdefault("model", self.model)
        if self.labels:
            self.metadata.setdefault("labels", self.labels)
