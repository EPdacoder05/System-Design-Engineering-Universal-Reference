"""
Async SQLAlchemy Connection Management

Apply to: Database connection pooling, session management, transaction handling
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import Pool


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models"""
    pass


class DatabaseManager:
    """
    Manages async database connections with pooling and health checks.
    
    Features:
    - Connection pooling with configurable size
    - Automatic session management with context managers
    - Transaction management (commit/rollback)
    - Connection health checks
    - Graceful shutdown
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_pre_ping: bool = True,
        echo: bool = False,
    ):
        """
        Initialize database manager.
        
        Args:
            database_url: Database URL (falls back to DATABASE_URL env var)
            pool_size: Number of connections to maintain in pool
            max_overflow: Max connections beyond pool_size
            pool_pre_ping: Enable connection health checks
            echo: Log all SQL statements
        """
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://user:password@localhost:5432/dbname"
        )
        self.pool_size = int(os.getenv("DB_POOL_SIZE", pool_size))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", max_overflow))
        
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        
        # Create engine
        self._engine = create_async_engine(
            self.database_url,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_pre_ping=pool_pre_ping,
            echo=echo,
            future=True,
        )
        
        # Setup connection pooling event listeners
        self._setup_pool_listeners()
        
        # Create session factory
        # expire_on_commit=False prevents automatic expiration after commit,
        # allowing access to model attributes without additional queries
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Keep objects accessible after commit
            autocommit=False,
            autoflush=False,
        )
    
    def _setup_pool_listeners(self) -> None:
        """Setup connection pool event listeners for monitoring"""
        
        @event.listens_for(Pool, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Log when a new connection is created"""
            print(f"New DB connection created: {id(dbapi_conn)}")
        
        @event.listens_for(Pool, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Log when a connection is checked out from pool"""
            pass  # Can add logging here if needed
    
    @property
    def engine(self) -> AsyncEngine:
        """Get the async engine"""
        if self._engine is None:
            raise RuntimeError("Database engine not initialized")
        return self._engine
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Context manager for database sessions with automatic transaction management.
        
        Usage:
            async with db_manager.session() as session:
                result = await session.execute(select(User))
                users = result.scalars().all()
        """
        if self._session_factory is None:
            raise RuntimeError("Session factory not initialized")
        
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Context manager for explicit transaction control.
        
        Usage:
            async with db_manager.transaction() as session:
                # Your transaction operations
                user = User(name="John")
                session.add(user)
                # Auto-commits on success, rolls back on exception
        """
        async with self.session() as session:
            async with session.begin():
                yield session
    
    async def health_check(self) -> bool:
        """
        Check database connection health.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            print(f"Database health check failed: {e}")
            return False
    
    async def create_all(self) -> None:
        """Create all tables defined in models"""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_all(self) -> None:
        """Drop all tables (use with caution!)"""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async def close(self) -> None:
        """Close all database connections"""
        if self._engine:
            await self._engine.dispose()
            print("Database connections closed")


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency injection function for FastAPI routes.
    
    Usage:
        @app.get("/users")
        async def get_users(session: AsyncSession = Depends(get_session)):
            result = await session.execute(select(User))
            return result.scalars().all()
    """
    manager = get_db_manager()
    async with manager.session() as session:
        yield session


# Example usage
if __name__ == "__main__":
    import asyncio
    from sqlalchemy import Column, Integer, String, select
    
    # Define a simple model
    class User(Base):
        __tablename__ = "users"
        
        id = Column(Integer, primary_key=True)
        name = Column(String(100), nullable=False)
        email = Column(String(100), unique=True, nullable=False)
    
    async def main():
        # Initialize database manager
        db = DatabaseManager(
            database_url="postgresql+asyncpg://user:password@localhost:5432/testdb",
            pool_size=5,
            max_overflow=10,
        )
        
        print("=== Database Connection Example ===\n")
        
        # Create tables
        print("Creating tables...")
        await db.create_all()
        
        # Health check
        print(f"Health check: {await db.health_check()}")
        
        # Insert data using session context manager
        print("\nInserting users...")
        async with db.session() as session:
            user1 = User(name="Alice", email="alice@example.com")
            user2 = User(name="Bob", email="bob@example.com")
            session.add_all([user1, user2])
        
        # Query data
        print("\nQuerying users...")
        async with db.session() as session:
            result = await session.execute(select(User))
            users = result.scalars().all()
            for user in users:
                print(f"  - {user.name} ({user.email})")
        
        # Transaction example with explicit control
        print("\nTransaction example...")
        try:
            async with db.transaction() as session:
                user3 = User(name="Charlie", email="charlie@example.com")
                session.add(user3)
                # Simulate error
                # raise Exception("Something went wrong!")
        except Exception as e:
            print(f"Transaction rolled back: {e}")
        
        # Verify final state
        print("\nFinal user count...")
        async with db.session() as session:
            result = await session.execute(select(User))
            count = len(result.scalars().all())
            print(f"Total users: {count}")
        
        # Cleanup
        print("\nCleaning up...")
        await db.drop_all()
        await db.close()
    
    # Run example
    # asyncio.run(main())
    print("\nTo run this example:")
    print("1. Set up a PostgreSQL database")
    print("2. Update the database_url in main()")
    print("3. Uncomment asyncio.run(main())")
