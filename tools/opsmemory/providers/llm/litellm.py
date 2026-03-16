"""LiteLLM-backed LLM provider for OpsMemory.

Uses LiteLLM in **SDK mode** by default — no separate proxy process required.
To switch to proxy mode, set ``OPSMEMORY_LITELLM_BASE_URL`` to the proxy URL
and LiteLLM will route requests there transparently.

Supported model families (configured via ``OPSMEMORY_LLM_MODEL``):
- Anthropic Claude:   ``anthropic/claude-3-5-sonnet-20241022``
- OpenAI:             ``openai/gpt-4o``
- AWS Bedrock:        ``bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0``
- OpenAI-compatible:  Set ``OPSMEMORY_LITELLM_BASE_URL`` + any model name.

Environment variables
---------------------
``OPSMEMORY_LLM_MODEL``
    LiteLLM-format model string.  Defaults to ``anthropic/claude-3-haiku-20240307``.
``OPSMEMORY_LITELLM_BASE_URL``
    Optional.  When set, LiteLLM routes all calls through this base URL (proxy
    mode or OpenAI-compatible backend).
``OPSMEMORY_LITELLM_API_KEY``
    Optional API key override forwarded to LiteLLM.  Provider-specific keys
    (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, etc.) are also respected and
    are the preferred approach — do not hardcode them here.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import structlog

from tools.opsmemory.providers.llm.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    OpsMemoryLLMError,
)

log = structlog.get_logger(__name__)

_DEFAULT_MODEL = "anthropic/claude-3-haiku-20240307"


class LiteLLMProvider(BaseLLMProvider):
    """LiteLLM-backed LLM provider.

    Supports SDK mode (default) and proxy mode (via ``OPSMEMORY_LITELLM_BASE_URL``).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._model = model or os.environ.get("OPSMEMORY_LLM_MODEL", _DEFAULT_MODEL)
        self._base_url = base_url or os.environ.get("OPSMEMORY_LITELLM_BASE_URL")
        self._api_key = api_key or os.environ.get("OPSMEMORY_LITELLM_API_KEY")

    @property
    def provider_name(self) -> str:
        return "litellm"

    @property
    def model_name(self) -> str:
        return self._model

    def _build_kwargs(self, **extra: Any) -> dict:
        kwargs: dict = {"model": self._model}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if self._api_key:
            kwargs["api_key"] = self._api_key
        kwargs.update(extra)
        return kwargs

    async def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> LLMResponse:
        messages: List[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self._call(messages, max_tokens=max_tokens, temperature=temperature, **kwargs)

    async def chat(
        self,
        messages: List[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> LLMResponse:
        dicts = [{"role": m.role, "content": m.content} for m in messages]
        return await self._call(dicts, max_tokens=max_tokens, temperature=temperature, **kwargs)

    async def _call(
        self,
        messages: List[dict],
        *,
        max_tokens: int,
        temperature: float,
        **extra: Any,
    ) -> LLMResponse:
        try:
            import litellm  # local import to avoid hard dependency at module load

            call_kwargs = self._build_kwargs(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
            )
            response = await litellm.acompletion(**call_kwargs)
            content: str = response.choices[0].message.content or ""
            usage_obj = getattr(response, "usage", None)
            usage: dict = {}
            if usage_obj:
                usage = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                    "total_tokens": getattr(usage_obj, "total_tokens", 0),
                }
            log.debug(
                "litellm_llm_call_complete",
                model=self._model,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
            return LLMResponse(
                content=content,
                model=response.model or self._model,
                provider="litellm",
                usage=usage,
                raw=response,
            )
        except Exception as exc:
            log.error("litellm_llm_call_failed", model=self._model, error=str(exc))
            raise OpsMemoryLLMError(
                f"LiteLLM call failed for model '{self._model}': {exc}"
            ) from exc
