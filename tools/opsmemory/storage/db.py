"""Database connection management for OpsMemory."""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text

from tools.opsmemory.storage.models import Base


@dataclass
class Settings:
    """Runtime configuration read from environment variables."""

    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "DATABASE_URL",
            "postgresql+asyncpg://opsmemory:opsmemory@localhost:5432/opsmemory",
        )
    )
    db_pool_size: int = field(
        default_factory=lambda: int(os.environ.get("DB_POOL_SIZE", "10"))
    )
    db_max_overflow: int = field(
        default_factory=lambda: int(os.environ.get("DB_MAX_OVERFLOW", "20"))
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


def create_engine(settings: Settings | None = None) -> AsyncEngine:
    """Create an async SQLAlchemy engine with pool configuration."""
    if settings is None:
        settings = get_settings()
    return create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        echo=False,
    )


# Module-level session factory, lazily initialised.
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        engine = create_engine()
        _session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager that yields a database session."""
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db(engine: AsyncEngine) -> None:
    """Create the pgvector extension and all ORM tables."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
