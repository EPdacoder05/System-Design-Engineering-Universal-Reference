"""FastAPI application for OpsMemory."""

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tools.opsmemory.agent.consolidator import ConsolidationConfig, consolidate, generate_embedding
from tools.opsmemory.agent.ingestor import IngestPayload, ingest_evidence
from tools.opsmemory.storage.db import create_engine, get_settings, init_db
from tools.opsmemory.storage.models import EvidenceItem
from tools.opsmemory.storage.repository import (
    EmbeddingRepository,
    EvidenceRepository,
    MemoryRepository,
    SourceRepository,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------
_engine = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _session_factory
    settings = get_settings()
    _engine = create_engine(settings)
    await init_db(_engine)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    log.info("opsmemory_api_started")
    yield
    if _engine:
        await _engine.dispose()
    log.info("opsmemory_api_stopped")


app = FastAPI(title="OpsMemory", version="1.0.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def correlation_and_logging_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    start = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Correlation-ID"] = correlation_id
    log.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
        correlation_id=correlation_id,
    )
    return response


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


async def get_db() -> AsyncSession:  # type: ignore[return]
    if _session_factory is None:
        raise HTTPException(status_code=503, detail="Database not initialised")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    text: str
    source_type: str = "manual"
    source_ref: str = ""
    author: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/ready")
async def ready(session: AsyncSession = Depends(get_db)):
    from sqlalchemy import text

    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB not ready: {exc}")
    return {"status": "ready"}


@app.post("/ingest")
async def ingest(
    req: IngestRequest,
    request: Request,
    session: AsyncSession = Depends(get_db),
):
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    payload = IngestPayload(
        text=req.text,
        source_type=req.source_type,
        source_ref=req.source_ref,
        author=req.author,
        metadata=req.metadata or {},
    )
    item = await ingest_evidence(session, payload, correlation_id)
    redacted = "[REDACTED]" in item.raw_text
    return {
        "evidence_id": str(item.id),
        "redacted": redacted,
        "correlation_id": correlation_id,
    }


@app.post("/consolidate")
async def trigger_consolidate(session: AsyncSession = Depends(get_db)):
    config = ConsolidationConfig()
    run = await consolidate(session, config)
    return {
        "run_id": str(run.id),
        "memories_created": run.memory_count,
        "evidence_consolidated": run.evidence_count,
    }


@app.get("/query")
async def query(
    q: str = Query(..., description="Natural language query"),
    limit: int = Query(5, ge=1, le=50),
    session: AsyncSession = Depends(get_db),
):
    query_embedding = generate_embedding(q)

    try:
        semantic_results = await EmbeddingRepository.semantic_search(
            session, query_embedding, limit=limit
        )
    except Exception:
        # Gracefully handle the case where no embeddings exist yet.
        semantic_results = []

    memories = await MemoryRepository.list_all(session, limit=limit)

    citations = [
        {
            "evidence_id": str(item.id),
            "source_ref": item.source_ref,
            "source_type": item.source_type,
            "snippet": item.raw_text[:200],
        }
        for item, _score in semantic_results
    ]

    source_refs = list({c["source_ref"] for c in citations if c["source_ref"]})
    answer = (
        f"Based on {len(citations)} evidence items and {len(memories)} memories"
        + (f": {', '.join(source_refs)}" if source_refs else ".")
    )

    return {
        "query": q,
        "answer": answer,
        "citations": citations,
        "memories": [
            {"id": str(m.id), "summary": m.summary or m.content[:200]}
            for m in memories
        ],
    }


@app.get("/memories")
async def list_memories(session: AsyncSession = Depends(get_db)):
    memories = await MemoryRepository.list_all(session, limit=50)
    return [
        {
            "id": str(m.id),
            "summary": m.summary,
            "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in memories
    ]


@app.get("/evidence")
async def list_evidence(session: AsyncSession = Depends(get_db)):
    from sqlalchemy import select

    result = await session.execute(
        select(EvidenceItem)
        .order_by(EvidenceItem.ingested_at.desc())
        .limit(100)
    )
    items = result.scalars().all()
    return [
        {
            "id": str(e.id),
            "source_type": e.source_type,
            "source_ref": e.source_ref,
            "author": e.author,
            "consolidated": e.consolidated,
            "ingested_at": e.ingested_at.isoformat() if e.ingested_at else None,
            "snippet": e.raw_text[:200],
        }
        for e in items
    ]


@app.get("/sources")
async def list_sources(session: AsyncSession = Depends(get_db)):
    sources = await SourceRepository.list_all(session)
    return [
        {
            "id": str(s.id),
            "owner": s.owner,
            "repo": s.repo,
            "source_type": s.source_type,
            "last_fetched_at": s.last_fetched_at.isoformat() if s.last_fetched_at else None,
        }
        for s in sources
    ]
