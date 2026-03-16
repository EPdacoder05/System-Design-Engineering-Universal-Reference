"""Provider abstraction layer for OpsMemory.

Callers select an LLM or embedding provider via environment variables or
explicit configuration rather than direct imports, making it easy to swap
providers without touching business logic.

Usage
-----
    from tools.opsmemory.providers import get_llm_provider, get_embedding_provider

    llm = get_llm_provider()
    response = await llm.generate("Summarise this text: ...")

    embedder = get_embedding_provider()
    vec = await embedder.embed("hello world")
"""

from __future__ import annotations

import os
from typing import Optional

__all__ = [
    "get_llm_provider",
    "get_embedding_provider",
]


def get_llm_provider(
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
):
    """Return an LLM provider instance driven by environment/config.

    Parameters
    ----------
    provider_name:
        Override the provider.  Reads ``OPSMEMORY_LLM_PROVIDER`` when *None*.
        Supported values: ``"litellm"``, ``"mock"``.
    model:
        Override the model name.  Reads ``OPSMEMORY_LLM_MODEL`` when *None*.
    """
    name = (provider_name or os.environ.get("OPSMEMORY_LLM_PROVIDER", "mock")).lower()
    resolved_model = model or os.environ.get("OPSMEMORY_LLM_MODEL")

    if name == "litellm":
        from tools.opsmemory.providers.llm.litellm import LiteLLMProvider

        return LiteLLMProvider(model=resolved_model)

    if name == "mock":
        from tools.opsmemory.providers.llm.base import MockLLMProvider

        return MockLLMProvider(model=resolved_model or "mock")

    raise ValueError(
        f"Unknown LLM provider '{name}'. "
        "Set OPSMEMORY_LLM_PROVIDER to 'litellm' or 'mock'."
    )


def get_embedding_provider(
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
):
    """Return an embedding provider instance driven by environment/config.

    Parameters
    ----------
    provider_name:
        Override the provider.  Reads ``OPSMEMORY_EMBEDDING_PROVIDER`` when
        *None*.  Supported values: ``"litellm"``, ``"mock"``.
    model:
        Override the model name.  Reads ``OPSMEMORY_EMBEDDING_MODEL`` when
        *None*.
    """
    name = (
        provider_name or os.environ.get("OPSMEMORY_EMBEDDING_PROVIDER", "mock")
    ).lower()
    resolved_model = model or os.environ.get("OPSMEMORY_EMBEDDING_MODEL")

    if name == "litellm":
        from tools.opsmemory.providers.embeddings.litellm import LiteLLMEmbeddingProvider

        return LiteLLMEmbeddingProvider(model=resolved_model)

    if name == "mock":
        from tools.opsmemory.providers.embeddings.base import MockEmbeddingProvider

        return MockEmbeddingProvider(model=resolved_model or "mock")

    raise ValueError(
        f"Unknown embedding provider '{name}'. "
        "Set OPSMEMORY_EMBEDDING_PROVIDER to 'litellm' or 'mock'."
    )
