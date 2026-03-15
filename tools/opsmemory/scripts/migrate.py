"""Database migration script — creates extension and all ORM tables."""

import asyncio
import os

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from tools.opsmemory.storage.models import Base

log = structlog.get_logger(__name__)


async def main() -> None:
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://opsmemory:opsmemory@localhost:5432/opsmemory",
    )
    log.info("migration_starting", database_url=database_url)

    engine = create_async_engine(database_url, echo=True)

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        log.info("extension_created", extension="vector")

        await conn.run_sync(Base.metadata.create_all)
        log.info("tables_created")

        # ivfflat requires rows to already exist before the index can be built
        # because it selects k-means centroids from the data.  In production,
        # run this separately after loading representative data, or switch to
        # the hnsw index type which works on empty tables.
        try:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_evidence_embeddings_embedding_ivfflat "
                    "ON evidence_embeddings USING ivfflat (embedding vector_cosine_ops)"
                )
            )
            log.info("ivfflat_index_created")
        except Exception as exc:
            log.warning(
                "ivfflat_index_skipped",
                reason=str(exc),
                hint="Load data first, then re-run migrate.py to build the index.",
            )

    await engine.dispose()
    print("Migration completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
