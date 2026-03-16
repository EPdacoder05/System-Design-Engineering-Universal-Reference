"""Abstract base interface for embedding providers.

Defines the contract that all embedding provider implementations must
satisfy.  Business logic in OpsMemory depends only on this interface.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class EmbeddingResult:
    """Result from a single embedding call."""

    embedding: List[float]
    model: str
    provider: str
    usage: dict = field(default_factory=dict)


class OpsMemoryEmbeddingError(Exception):
    """Raised when an embedding provider call fails."""


class BaseEmbeddingProvider(abc.ABC):
    """Abstract interface for embedding providers.

    All implementors must provide :meth:`embed` (single text) and
    :meth:`embed_batch` (list of texts).  New providers should subclass
    this without requiring changes to OpsMemory business logic.
    """

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Embedding model identifier."""

    @property
    @abc.abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors produced by this provider."""

    @abc.abstractmethod
    async def embed(self, text: str, **kwargs: Any) -> EmbeddingResult:
        """Return an embedding for a single *text* string."""

    @abc.abstractmethod
    async def embed_batch(
        self, texts: List[str], **kwargs: Any
    ) -> List[EmbeddingResult]:
        """Return embeddings for a list of *texts*.

        Default implementation calls :meth:`embed` sequentially.  Providers
        that support native batch endpoints should override this.
        """

    async def embed_as_list(self, text: str, **kwargs: Any) -> List[float]:
        """Convenience wrapper returning just the embedding vector."""
        result = await self.embed(text, **kwargs)
        return result.embedding


# ---------------------------------------------------------------------------
# Mock / dev-mode implementation
# ---------------------------------------------------------------------------


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic mock embedding provider for local development and tests.

    Uses seeded numpy random numbers — same text always yields the same
    vector without calling any external API.
    """

    _DEFAULT_DIM = 1536

    def __init__(self, model: str = "mock", dim: int = _DEFAULT_DIM) -> None:
        self._model = model
        self._dim = dim

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def embedding_dim(self) -> int:
        return self._dim

    async def embed(self, text: str, **kwargs: Any) -> EmbeddingResult:
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        vec: List[float] = rng.random(self._dim).tolist()
        return EmbeddingResult(
            embedding=vec,
            model=self._model,
            provider="mock",
            usage={"total_tokens": len(text.split())},
        )

    async def embed_batch(
        self, texts: List[str], **kwargs: Any
    ) -> List[EmbeddingResult]:
        return [await self.embed(t, **kwargs) for t in texts]
