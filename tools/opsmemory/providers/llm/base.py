"""Abstract base interface for LLM providers.

Defines the contract that all LLM provider implementations must satisfy.
Business logic in OpsMemory should depend only on this interface, not on any
specific provider implementation.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMMessage:
    """A single message in a chat-style conversation."""

    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class LLMResponse:
    """Structured response from an LLM generation call."""

    content: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Optional[Any] = field(default=None, repr=False)


class OpsMemoryLLMError(Exception):
    """Raised when an LLM provider call fails."""


class BaseLLMProvider(abc.ABC):
    """Abstract interface for LLM text-generation providers.

    Implementing classes must be importable and returned by the factory in
    :mod:`tools.opsmemory.providers`.  New providers (Bedrock, Anthropic
    direct, etc.) should subclass this without changing any caller code.
    """

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g. ``"litellm"``)."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Model identifier forwarded to the underlying SDK."""

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from a single user *prompt*.

        Parameters
        ----------
        prompt:
            The user-facing prompt text.
        system:
            Optional system prompt / instruction prefix.
        max_tokens:
            Upper token limit for the response.
        temperature:
            Sampling temperature.  Lower = more deterministic.
        """

    @abc.abstractmethod
    async def chat(
        self,
        messages: List[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from a list of *messages*.

        Provides the full chat interface for multi-turn interactions and
        structured system prompts.
        """

    async def summarise(self, text: str, *, max_tokens: int = 256) -> str:
        """Convenience wrapper: summarise *text* and return the string.

        Subclasses may override this for provider-specific optimisations.
        """
        response = await self.generate(
            f"Summarise the following text in 3-5 sentences:\n\n{text}",
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response.content


# ---------------------------------------------------------------------------
# Mock / dev-mode implementation
# ---------------------------------------------------------------------------


class MockLLMProvider(BaseLLMProvider):
    """Deterministic mock LLM provider for local development and tests.

    Does **not** call any external service.  Returns canned responses so that
    the rest of OpsMemory can be exercised without API credentials.
    """

    def __init__(self, model: str = "mock") -> None:
        self._model = model

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> LLMResponse:
        content = f"[MOCK LLM] Received prompt ({len(prompt)} chars)"
        return LLMResponse(
            content=content,
            model=self._model,
            provider="mock",
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": 10},
        )

    async def chat(
        self,
        messages: List[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> LLMResponse:
        last = messages[-1].content if messages else ""
        content = f"[MOCK LLM] Received {len(messages)} messages; last: {last[:80]}"
        prompt_tokens = sum(len(m.content.split()) for m in messages)
        return LLMResponse(
            content=content,
            model=self._model,
            provider="mock",
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": 10},
        )
