"""Tests for OpsMemory provider abstraction layer.

Covers:
- Base class / mock provider behaviour
- Factory loader (get_llm_provider / get_embedding_provider)
- LiteLLM adapter with mocked SDK calls
- Model registry loading and validation
"""

from __future__ import annotations

import os
import unittest.mock as mock
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# MockEmbeddingProvider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_embedding_provider_returns_correct_dim():
    from tools.opsmemory.providers.embeddings.base import MockEmbeddingProvider

    p = MockEmbeddingProvider(dim=1536)
    result = await p.embed("hello world")
    assert len(result.embedding) == 1536


@pytest.mark.asyncio
async def test_mock_embedding_provider_is_deterministic():
    from tools.opsmemory.providers.embeddings.base import MockEmbeddingProvider

    p = MockEmbeddingProvider()
    r1 = await p.embed("same text")
    r2 = await p.embed("same text")
    assert r1.embedding == r2.embedding


@pytest.mark.asyncio
async def test_mock_embedding_provider_different_texts_differ():
    from tools.opsmemory.providers.embeddings.base import MockEmbeddingProvider

    p = MockEmbeddingProvider()
    r1 = await p.embed("text one")
    r2 = await p.embed("text two")
    assert r1.embedding != r2.embedding


@pytest.mark.asyncio
async def test_mock_embedding_provider_batch():
    from tools.opsmemory.providers.embeddings.base import MockEmbeddingProvider

    p = MockEmbeddingProvider()
    results = await p.embed_batch(["a", "b", "c"])
    assert len(results) == 3
    assert all(len(r.embedding) == 1536 for r in results)


@pytest.mark.asyncio
async def test_mock_embedding_provider_metadata():
    from tools.opsmemory.providers.embeddings.base import MockEmbeddingProvider

    p = MockEmbeddingProvider()
    assert p.provider_name == "mock"
    assert p.embedding_dim == 1536


# ---------------------------------------------------------------------------
# MockLLMProvider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_llm_provider_generate_returns_string():
    from tools.opsmemory.providers.llm.base import MockLLMProvider

    p = MockLLMProvider()
    response = await p.generate("What is the capital of France?")
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_mock_llm_provider_chat():
    from tools.opsmemory.providers.llm.base import LLMMessage, MockLLMProvider

    p = MockLLMProvider()
    messages = [
        LLMMessage(role="system", content="You are helpful."),
        LLMMessage(role="user", content="Hello!"),
    ]
    response = await p.chat(messages)
    assert isinstance(response.content, str)
    assert response.provider == "mock"


@pytest.mark.asyncio
async def test_mock_llm_provider_summarise():
    from tools.opsmemory.providers.llm.base import MockLLMProvider

    p = MockLLMProvider()
    summary = await p.summarise("A very long piece of text about deployments.")
    assert isinstance(summary, str)


# ---------------------------------------------------------------------------
# Factory / loader
# ---------------------------------------------------------------------------


def test_get_embedding_provider_returns_mock_by_default(monkeypatch):
    monkeypatch.delenv("OPSMEMORY_EMBEDDING_PROVIDER", raising=False)
    from tools.opsmemory.providers import get_embedding_provider
    from tools.opsmemory.providers.embeddings.base import MockEmbeddingProvider

    provider = get_embedding_provider()
    assert isinstance(provider, MockEmbeddingProvider)


def test_get_llm_provider_returns_mock_by_default(monkeypatch):
    monkeypatch.delenv("OPSMEMORY_LLM_PROVIDER", raising=False)
    from tools.opsmemory.providers import get_llm_provider
    from tools.opsmemory.providers.llm.base import MockLLMProvider

    provider = get_llm_provider()
    assert isinstance(provider, MockLLMProvider)


def test_get_embedding_provider_mock_explicit(monkeypatch):
    monkeypatch.setenv("OPSMEMORY_EMBEDDING_PROVIDER", "mock")
    from tools.opsmemory.providers import get_embedding_provider
    from tools.opsmemory.providers.embeddings.base import MockEmbeddingProvider

    provider = get_embedding_provider()
    assert isinstance(provider, MockEmbeddingProvider)


def test_get_llm_provider_litellm(monkeypatch):
    monkeypatch.setenv("OPSMEMORY_LLM_PROVIDER", "litellm")
    from tools.opsmemory.providers import get_llm_provider
    from tools.opsmemory.providers.llm.litellm import LiteLLMProvider

    provider = get_llm_provider()
    assert isinstance(provider, LiteLLMProvider)


def test_get_embedding_provider_litellm(monkeypatch):
    monkeypatch.setenv("OPSMEMORY_EMBEDDING_PROVIDER", "litellm")
    from tools.opsmemory.providers import get_embedding_provider
    from tools.opsmemory.providers.embeddings.litellm import LiteLLMEmbeddingProvider

    provider = get_embedding_provider()
    assert isinstance(provider, LiteLLMEmbeddingProvider)


def test_get_llm_provider_unknown_raises():
    from tools.opsmemory.providers import get_llm_provider

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm_provider(provider_name="nonexistent")


def test_get_embedding_provider_unknown_raises():
    from tools.opsmemory.providers import get_embedding_provider

    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedding_provider(provider_name="nonexistent")


def test_factory_model_override():
    from tools.opsmemory.providers import get_embedding_provider

    provider = get_embedding_provider(provider_name="mock", model="custom-mock")
    assert provider.model_name == "custom-mock"


# ---------------------------------------------------------------------------
# LiteLLM embedding adapter — mocked SDK
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_litellm_embedding_adapter_parses_response():
    from tools.opsmemory.providers.embeddings.litellm import LiteLLMEmbeddingProvider

    provider = LiteLLMEmbeddingProvider(model="openai/text-embedding-3-small")

    mock_data_item = {"embedding": [0.1, 0.2, 0.3]}
    mock_response = MagicMock()
    mock_response.data = [mock_data_item]
    mock_response.usage = MagicMock(total_tokens=5)

    with patch("litellm.aembedding", new=AsyncMock(return_value=mock_response)):
        result = await provider.embed("hello")

    assert result.embedding == [0.1, 0.2, 0.3]
    assert result.provider == "litellm"
    assert result.model == "openai/text-embedding-3-small"


@pytest.mark.asyncio
async def test_litellm_embedding_adapter_batch():
    from tools.opsmemory.providers.embeddings.litellm import LiteLLMEmbeddingProvider

    provider = LiteLLMEmbeddingProvider(model="openai/text-embedding-3-small")

    mock_response = MagicMock()
    mock_response.data = [{"embedding": [0.1] * 10}, {"embedding": [0.2] * 10}]
    mock_response.usage = MagicMock(total_tokens=20)

    with patch("litellm.aembedding", new=AsyncMock(return_value=mock_response)):
        results = await provider.embed_batch(["a", "b"])

    assert len(results) == 2


@pytest.mark.asyncio
async def test_litellm_embedding_adapter_raises_on_failure():
    from tools.opsmemory.providers.embeddings.base import OpsMemoryEmbeddingError
    from tools.opsmemory.providers.embeddings.litellm import LiteLLMEmbeddingProvider

    provider = LiteLLMEmbeddingProvider(model="openai/text-embedding-3-small")

    with patch("litellm.aembedding", new=AsyncMock(side_effect=Exception("API error"))):
        with pytest.raises(OpsMemoryEmbeddingError, match="API error"):
            await provider.embed("fail")


# ---------------------------------------------------------------------------
# LiteLLM LLM adapter — mocked SDK
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_litellm_llm_adapter_generate():
    from tools.opsmemory.providers.llm.litellm import LiteLLMProvider

    provider = LiteLLMProvider(model="anthropic/claude-3-haiku-20240307")

    mock_choice = MagicMock()
    mock_choice.message.content = "This is a test response."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model = "claude-3-haiku-20240307"
    mock_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )

    with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
        response = await provider.generate("Summarise this.")

    assert response.content == "This is a test response."
    assert response.provider == "litellm"
    assert response.usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_litellm_llm_adapter_raises_on_failure():
    from tools.opsmemory.providers.llm.base import OpsMemoryLLMError
    from tools.opsmemory.providers.llm.litellm import LiteLLMProvider

    provider = LiteLLMProvider(model="anthropic/claude-3-haiku-20240307")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=Exception("timeout"))):
        with pytest.raises(OpsMemoryLLMError, match="timeout"):
            await provider.generate("fail")


def test_litellm_llm_provider_metadata():
    from tools.opsmemory.providers.llm.litellm import LiteLLMProvider

    provider = LiteLLMProvider(model="openai/gpt-4o")
    assert provider.provider_name == "litellm"
    assert provider.model_name == "openai/gpt-4o"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def test_model_registry_loads_without_error():
    from pathlib import Path

    import yaml

    registry_path = (
        Path(__file__).parent.parent / "providers" / "model_registry.yaml"
    )
    assert registry_path.exists(), f"Registry not found at {registry_path}"
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    assert "models" in data
    assert len(data["models"]) > 0


def test_model_registry_validation_passes():
    from tools.opsmemory.scripts.validate_model_registry import validate

    errors = validate()
    assert errors == [], f"Registry validation failed: {errors}"


def test_model_registry_has_expected_providers():
    from pathlib import Path

    import yaml

    registry_path = (
        Path(__file__).parent.parent / "providers" / "model_registry.yaml"
    )
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    providers = {m["provider"] for m in data["models"]}
    assert "anthropic" in providers
    assert "openai" in providers
    assert "bedrock" in providers
    assert "mock" in providers


def test_model_registry_mock_is_not_production_approved():
    from pathlib import Path

    import yaml

    registry_path = (
        Path(__file__).parent.parent / "providers" / "model_registry.yaml"
    )
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    mock_model = next(m for m in data["models"] if m["id"] == "mock")
    assert mock_model["production_approved"] is False


def test_model_registry_embedding_models_have_dim():
    from pathlib import Path

    import yaml

    registry_path = (
        Path(__file__).parent.parent / "providers" / "model_registry.yaml"
    )
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    for model in data["models"]:
        if model.get("supports_embeddings"):
            assert model.get("embedding_dim"), (
                f"Model {model['id']} supports_embeddings=True but has no embedding_dim"
            )


# ---------------------------------------------------------------------------
# litellm dependency — upstream source and version pin hygiene
# ---------------------------------------------------------------------------


def test_litellm_is_declared_in_requirements():
    """litellm must be explicitly declared in requirements.txt.

    This guards against accidentally dropping the dependency and ensures we
    always ship from BerriAI's PyPI package rather than any fork.
    """
    from pathlib import Path

    repo_root = Path(__file__).parent.parent.parent.parent
    req_path = repo_root / "requirements.txt"
    assert req_path.exists(), f"requirements.txt not found at {req_path}"

    content = req_path.read_text()
    # Must contain a litellm line with a version pin
    import re

    match = re.search(r"^litellm>=(\d+\.\d+[\.\d]*)", content, re.MULTILINE)
    assert match, (
        "litellm not found (or not version-pinned with >=) in requirements.txt. "
        "Add 'litellm>=<version>' sourced from BerriAI/litellm on PyPI."
    )


def test_litellm_pin_is_at_least_version_1():
    """The litellm pin in requirements.txt must be >= 1.0.0 (BerriAI stable era)."""
    from pathlib import Path
    import re

    repo_root = Path(__file__).parent.parent.parent.parent
    req_path = repo_root / "requirements.txt"
    content = req_path.read_text()

    match = re.search(r"^litellm>=(\d+\.\d+[\.\d]*)", content, re.MULTILINE)
    assert match, "litellm pin not found in requirements.txt"

    from packaging.version import Version

    pinned = Version(match.group(1))
    assert pinned >= Version("1.0.0"), (
        f"litellm pin {pinned} is below 1.0.0. "
        "Use a BerriAI 1.x release — pre-1.0 versions predate the stable API."
    )
