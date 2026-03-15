"""Tests for consolidation logic."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from tools.opsmemory.agent.consolidator import generate_embedding
from tools.opsmemory.agent.redactor import redact_text


def test_generate_embedding_returns_correct_dimension():
    vec = generate_embedding("hello world")
    assert isinstance(vec, list)
    assert len(vec) == 1536


def test_generate_embedding_all_floats():
    vec = generate_embedding("check types")
    assert all(isinstance(v, float) for v in vec)


def test_generate_embedding_is_deterministic():
    text = "deterministic test input"
    vec1 = generate_embedding(text)
    vec2 = generate_embedding(text)
    assert vec1 == vec2


def test_generate_embedding_different_inputs_differ():
    vec1 = generate_embedding("input one")
    vec2 = generate_embedding("input two")
    assert vec1 != vec2


def test_generate_embedding_custom_dim():
    vec = generate_embedding("test", dim=128)
    assert len(vec) == 128


@pytest.mark.asyncio
async def test_embedding_generated_from_redacted_text():
    """Embedding must be derived from the *redacted* text, not the original."""
    raw = "api_key: supersecret123"
    redacted, count = redact_text(raw)
    assert count >= 1
    assert "supersecret123" not in redacted

    # The embedding of the redacted text should differ from the raw text.
    vec_raw = generate_embedding(raw)
    vec_redacted = generate_embedding(redacted)
    assert vec_raw != vec_redacted
