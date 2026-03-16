"""LiteLLM-backed embedding provider for OpsMemory.

Uses LiteLLM in **SDK mode** by default — no separate proxy process required.
Set ``OPSMEMORY_LITELLM_BASE_URL`` to route via a proxy or OpenAI-compatible
backend.

Supported embedding model families:
- OpenAI:     ``openai/text-embedding-3-small`` (1536-dim)
- Bedrock:    ``bedrock/amazon.titan-embed-text-v2:0`` (1024-dim)
- Cohere:     ``cohere/embed-english-v3.0`` (1024-dim)
- Local/compatible backends via ``OPSMEMORY_LITELLM_BASE_URL``

Environment variables
---------------------
``OPSMEMORY_EMBEDDING_MODEL``
    LiteLLM-format embedding model string.  Defaults to
    ``openai/text-embedding-3-small``.
``OPSMEMORY_LITELLM_BASE_URL``
    Optional proxy / compatible endpoint base URL.
``OPSMEMORY_LITELLM_API_KEY``
    Optional API key override.  Provider-specific keys (``OPENAI_API_KEY``,
    etc.) are the preferred approach and are respected automatically by LiteLLM.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import structlog

from tools.opsmemory.providers.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingResult,
    OpsMemoryEmbeddingError,
)

log = structlog.get_logger(__name__)

_DEFAULT_MODEL = "openai/text-embedding-3-small"
_DEFAULT_DIM = 1536

# Known dimensions for common models — used to populate embedding_dim without
# an extra round-trip.  Fall back to _DEFAULT_DIM for unlisted models.
_MODEL_DIMS: dict = {
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
    "openai/text-embedding-ada-002": 1536,
    "bedrock/amazon.titan-embed-text-v2:0": 1024,
    "cohere/embed-english-v3.0": 1024,
    "cohere/embed-multilingual-v3.0": 1024,
}


class LiteLLMEmbeddingProvider(BaseEmbeddingProvider):
    """LiteLLM-backed embedding provider."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._model = model or os.environ.get("OPSMEMORY_EMBEDDING_MODEL", _DEFAULT_MODEL)
        self._base_url = base_url or os.environ.get("OPSMEMORY_LITELLM_BASE_URL")
        self._api_key = api_key or os.environ.get("OPSMEMORY_LITELLM_API_KEY")

    @property
    def provider_name(self) -> str:
        return "litellm"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def embedding_dim(self) -> int:
        return _MODEL_DIMS.get(self._model, _DEFAULT_DIM)

    def _build_kwargs(self, **extra: Any) -> dict:
        kwargs: dict = {"model": self._model}
        if self._base_url:
            kwargs["api_base"] = self._base_url
        if self._api_key:
            kwargs["api_key"] = self._api_key
        kwargs.update(extra)
        return kwargs

    async def embed(self, text: str, **kwargs: Any) -> EmbeddingResult:
        results = await self.embed_batch([text], **kwargs)
        return results[0]

    async def embed_batch(
        self, texts: List[str], **kwargs: Any
    ) -> List[EmbeddingResult]:
        try:
            import litellm

            call_kwargs = self._build_kwargs(input=texts, **kwargs)
            response = await litellm.aembedding(**call_kwargs)
            usage_obj = getattr(response, "usage", None)
            total_tokens: int = getattr(usage_obj, "total_tokens", 0) if usage_obj else 0

            results: List[EmbeddingResult] = []
            for item in response.data:
                results.append(
                    EmbeddingResult(
                        embedding=item["embedding"],
                        model=self._model,
                        provider="litellm",
                        usage={"total_tokens": total_tokens},
                    )
                )
            log.debug(
                "litellm_embedding_call_complete",
                model=self._model,
                batch_size=len(texts),
                total_tokens=total_tokens,
            )
            return results
        except Exception as exc:
            log.error(
                "litellm_embedding_call_failed",
                model=self._model,
                error=str(exc),
            )
            raise OpsMemoryEmbeddingError(
                f"LiteLLM embedding call failed for model '{self._model}': {exc}"
            ) from exc
