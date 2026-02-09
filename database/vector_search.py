"""
pgvector Semantic Search with SQLAlchemy

Apply to: Vector embeddings, semantic search, similarity search, RAG systems
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import Column, DateTime, Index, String, Text, select
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # Fallback for type checking when pgvector is not installed
    Vector = None


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models"""
    pass


class Document(Base):
    """
    Document model with vector embeddings for semantic search.
    
    Features:
    - pgvector integration for similarity search
    - Metadata fields for filtering
    - Composite indexes for efficient queries
    """
    __tablename__ = "documents"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    
    # Vector embedding (e.g., 1536 dimensions for OpenAI ada-002)
    embedding = Column(
        Vector(1536) if Vector else None,
        nullable=True,
    )
    
    # Metadata fields
    title: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    
    source: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )
    
    document_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )
    
    metadata_json: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="JSON string with additional metadata",
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    
    # Indexes for efficient queries
    __table_args__ = (
        # Composite index for filtered searches
        Index("idx_doc_type_source", "document_type", "source"),
        Index("idx_doc_created", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, title={self.title})>"


class EmbeddingCache:
    """
    In-memory cache for embeddings to avoid redundant computations.
    
    In production, consider using Redis or Memcached for distributed caching.
    """
    
    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        return self._cache.get(key)
    
    def set(self, key: str, embedding: np.ndarray) -> None:
        """Set embedding in cache with LRU eviction"""
        if len(self._cache) >= self.max_size:
            # Simple FIFO eviction (in production, use proper LRU)
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        
        self._cache[key] = embedding
    
    def clear(self) -> None:
        """Clear all cached embeddings"""
        self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self._cache)


# Global embedding cache
embedding_cache = EmbeddingCache()


async def store_document_with_embedding(
    session: AsyncSession,
    content: str,
    embedding: List[float],
    title: Optional[str] = None,
    source: Optional[str] = None,
    document_type: Optional[str] = None,
    metadata: Optional[str] = None,
) -> Document:
    """
    Store a document with its embedding.
    
    Args:
        session: Database session
        content: Document text content
        embedding: Vector embedding (e.g., from OpenAI)
        title: Document title
        source: Document source
        document_type: Type of document
        metadata: Additional metadata as JSON string
    
    Returns:
        Created document
    """
    document = Document(
        content=content,
        embedding=embedding,
        title=title,
        source=source,
        document_type=document_type,
        metadata_json=metadata,
    )
    
    session.add(document)
    await session.commit()
    await session.refresh(document)
    
    return document


async def batch_insert_documents(
    session: AsyncSession,
    documents: List[Dict],
) -> List[Document]:
    """
    Batch insert documents with embeddings.
    
    Args:
        session: Database session
        documents: List of dicts with keys: content, embedding, title, source, etc.
    
    Returns:
        List of created documents
    """
    doc_objects = []
    
    for doc_data in documents:
        doc = Document(
            content=doc_data["content"],
            embedding=doc_data["embedding"],
            title=doc_data.get("title"),
            source=doc_data.get("source"),
            document_type=doc_data.get("document_type"),
            metadata_json=doc_data.get("metadata"),
        )
        doc_objects.append(doc)
    
    session.add_all(doc_objects)
    await session.commit()
    
    return doc_objects


async def cosine_similarity_search(
    session: AsyncSession,
    query_embedding: List[float],
    limit: int = 10,
    document_type: Optional[str] = None,
    source: Optional[str] = None,
    min_similarity: float = 0.0,
) -> List[Tuple[Document, float]]:
    """
    Perform cosine similarity search on documents.
    
    Args:
        session: Database session
        query_embedding: Query vector embedding
        limit: Maximum number of results
        document_type: Filter by document type
        source: Filter by source
        min_similarity: Minimum similarity threshold
    
    Returns:
        List of (document, similarity_score) tuples
    """
    from sqlalchemy import func
    
    # Build base query with similarity calculation
    similarity = Document.embedding.cosine_distance(query_embedding).label("distance")
    
    stmt = select(Document, similarity).order_by(similarity).limit(limit)
    
    # Apply filters
    if document_type:
        stmt = stmt.where(Document.document_type == document_type)
    
    if source:
        stmt = stmt.where(Document.source == source)
    
    # Execute query
    result = await session.execute(stmt)
    rows = result.all()
    
    # Convert distance to similarity and filter
    results = []
    for doc, distance in rows:
        similarity_score = 1 - distance  # Convert distance to similarity
        if similarity_score >= min_similarity:
            results.append((doc, similarity_score))
    
    return results


async def semantic_search_with_cache(
    session: AsyncSession,
    query: str,
    query_embedding: List[float],
    limit: int = 10,
    **filters,
) -> List[Tuple[Document, float]]:
    """
    Semantic search with embedding cache support.
    
    Args:
        session: Database session
        query: Query text (used for cache key)
        query_embedding: Query vector embedding
        limit: Maximum number of results
        **filters: Additional filter arguments
    
    Returns:
        List of (document, similarity_score) tuples
    """
    # Check cache
    cache_key = f"query:{query}:limit:{limit}"
    cached_embedding = embedding_cache.get(cache_key)
    
    if cached_embedding is not None:
        query_embedding = cached_embedding.tolist()
    else:
        # Store in cache
        embedding_cache.set(cache_key, np.array(query_embedding))
    
    # Perform search
    return await cosine_similarity_search(
        session=session,
        query_embedding=query_embedding,
        limit=limit,
        **filters,
    )


async def get_similar_documents(
    session: AsyncSession,
    document_id: uuid.UUID,
    limit: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Find documents similar to a given document.
    
    Args:
        session: Database session
        document_id: ID of the reference document
        limit: Maximum number of results
    
    Returns:
        List of (document, similarity_score) tuples
    """
    # Get reference document
    stmt = select(Document).where(Document.id == document_id)
    result = await session.execute(stmt)
    ref_doc = result.scalar_one_or_none()
    
    if not ref_doc or not ref_doc.embedding:
        return []
    
    # Search for similar documents (exclude the reference document)
    from sqlalchemy import func
    
    similarity = ref_doc.embedding.cosine_distance(Document.embedding).label("distance")
    
    stmt = (
        select(Document, similarity)
        .where(Document.id != document_id)
        .order_by(similarity)
        .limit(limit)
    )
    
    result = await session.execute(stmt)
    rows = result.all()
    
    # Convert to similarity scores
    return [(doc, 1 - distance) for doc, distance in rows]


async def update_document_embedding(
    session: AsyncSession,
    document_id: uuid.UUID,
    new_embedding: List[float],
) -> Optional[Document]:
    """
    Update document embedding.
    
    Args:
        session: Database session
        document_id: Document ID
        new_embedding: New vector embedding
    
    Returns:
        Updated document or None if not found
    """
    stmt = select(Document).where(Document.id == document_id)
    result = await session.execute(stmt)
    document = result.scalar_one_or_none()
    
    if document:
        document.embedding = new_embedding
        document.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(document)
    
    return document


# Example usage
if __name__ == "__main__":
    import asyncio
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    
    def generate_mock_embedding(dimension: int = 1536) -> List[float]:
        """Generate a mock embedding vector"""
        return np.random.rand(dimension).tolist()
    
    async def main():
        # Create engine and session
        engine = create_async_engine(
            "postgresql+asyncpg://user:password@localhost:5432/testdb",
            echo=True,
        )
        
        async_session = async_sessionmaker(
            engine,
            expire_on_commit=False,
        )
        
        print("=== pgvector Semantic Search Example ===\n")
        
        # Create tables (requires pgvector extension)
        async with engine.begin() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.run_sync(Base.metadata.create_all)
        
        async with async_session() as session:
            # Insert sample documents
            print("Inserting sample documents...\n")
            
            documents = [
                {
                    "content": "Machine learning is a subset of artificial intelligence.",
                    "embedding": generate_mock_embedding(),
                    "title": "Introduction to ML",
                    "source": "docs",
                    "document_type": "tutorial",
                },
                {
                    "content": "Deep learning uses neural networks with multiple layers.",
                    "embedding": generate_mock_embedding(),
                    "title": "Deep Learning Basics",
                    "source": "docs",
                    "document_type": "tutorial",
                },
                {
                    "content": "Python is a popular programming language for data science.",
                    "embedding": generate_mock_embedding(),
                    "title": "Python for Data Science",
                    "source": "blog",
                    "document_type": "article",
                },
            ]
            
            created_docs = await batch_insert_documents(session, documents)
            print(f"Created {len(created_docs)} documents\n")
            
            # Perform semantic search
            print("Performing semantic search...\n")
            query_embedding = generate_mock_embedding()
            
            results = await cosine_similarity_search(
                session=session,
                query_embedding=query_embedding,
                limit=5,
                document_type="tutorial",
            )
            
            print(f"Found {len(results)} results:")
            for doc, score in results:
                print(f"  - {doc.title} (similarity: {score:.4f})")
                print(f"    Content: {doc.content[:50]}...")
            print()
            
            # Find similar documents
            if created_docs:
                print(f"Finding documents similar to: {created_docs[0].title}\n")
                similar = await get_similar_documents(
                    session=session,
                    document_id=created_docs[0].id,
                    limit=2,
                )
                
                print(f"Found {len(similar)} similar documents:")
                for doc, score in similar:
                    print(f"  - {doc.title} (similarity: {score:.4f})")
                print()
            
            # Cache demo
            print(f"Embedding cache size: {embedding_cache.size()}")
            embedding_cache.clear()
            print("Cache cleared\n")
        
        # Cleanup
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        
        await engine.dispose()
    
    # Run example
    # asyncio.run(main())
    print("\nTo run this example:")
    print("1. Set up a PostgreSQL database with pgvector extension")
    print("   - Install: apt-get install postgresql-15-pgvector")
    print("   - Enable: CREATE EXTENSION vector;")
    print("2. Update the database URL in main()")
    print("3. Uncomment asyncio.run(main())")
