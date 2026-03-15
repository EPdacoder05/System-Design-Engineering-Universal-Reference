"""SQLAlchemy ORM models for OpsMemory using pgvector."""

from sqlalchemy import (
    Column,
    String,
    Text,
    Boolean,
    Integer,
    DateTime,
    Index,
    UniqueConstraint,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime, timezone


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Memory(Base):
    """Consolidated memory record synthesised from evidence batches."""

    __tablename__ = "memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    source_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=False, default=list)
    metadata_ = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )
    consolidated = Column(Boolean, nullable=False, default=False)


class EvidenceItem(Base):
    """Raw evidence ingested from any source before consolidation."""

    __tablename__ = "evidence_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    raw_text = Column(Text, nullable=False)
    source_type = Column(String(50), nullable=False)
    source_ref = Column(String(500), nullable=False, default="")
    author = Column(String(255), nullable=True)
    event_ts = Column(DateTime(timezone=True), nullable=True)
    metadata_ = Column(JSONB, nullable=False, default=dict)
    ingested_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    consolidated = Column(Boolean, nullable=False, default=False)
    correlation_id = Column(String(36), nullable=True)

    embedding = relationship(
        "EvidenceEmbedding", back_populates="evidence", uselist=False, cascade="all, delete-orphan"
    )


class EvidenceEmbedding(Base):
    """Vector embedding for an evidence item, used for semantic search."""

    __tablename__ = "evidence_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evidence_id = Column(
        UUID(as_uuid=True),
        ForeignKey("evidence_items.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    embedding = Column(Vector(1536), nullable=False)
    model_name = Column(String(100), nullable=False, default="mock")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    evidence = relationship("EvidenceItem", back_populates="embedding")

    __table_args__ = (
        # ivfflat index for approximate nearest-neighbour cosine search.
        # NOTE: In production, ivfflat requires rows to already exist before the
        # index can be built (it needs to select centroids). Run this migration
        # after loading a representative dataset, or use hnsw instead.
        Index(
            "ix_evidence_embeddings_embedding_ivfflat",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class ConsolidationRun(Base):
    """Audit record for a single consolidation cycle."""

    __tablename__ = "consolidation_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    started_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    evidence_count = Column(Integer, nullable=False, default=0)
    memory_count = Column(Integer, nullable=False, default=0)
    status = Column(String(20), nullable=False, default="running")
    error = Column(Text, nullable=True)


class Source(Base):
    """Registered data source (e.g. a GitHub repository)."""

    __tablename__ = "sources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner = Column(String(255), nullable=False)
    repo = Column(String(255), nullable=False)
    source_type = Column(String(50), nullable=False, default="github")
    last_fetched_at = Column(DateTime(timezone=True), nullable=True)
    metadata_ = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    __table_args__ = (
        UniqueConstraint("owner", "repo", "source_type", name="uq_sources_owner_repo_type"),
    )
