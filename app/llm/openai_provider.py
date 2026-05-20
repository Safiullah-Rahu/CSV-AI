"""OpenAI provider using the modern ``openai`` SDK (v1+).

We deliberately avoid LangChain's ``ChatOpenAI`` wrapper here — the native
SDK is leaner, supports streaming naturally, and removes a layer of
abstraction that historically caused breaking changes.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Optional

from app.llm.base import ChatResponse, LLMProvider, Message
from app.utils.errors import ConfigurationError, ProviderError
from app.utils.logging import get_logger

logger = get_logger("llm.openai")


class OpenAIProvider(LLMProvider):
    """OpenAI Chat Completions provider (gpt-4o, gpt-4.1, gpt-3.5, ...)."""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )
        self._client = None

    # ------------------------------------------------------------------ API
    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ConfigurationError(
                "OpenAI API key is missing. Set OPENAI_API_KEY or enter it in the sidebar."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "The 'openai' package is not installed. Run: pip install openai>=1.0"
            ) from exc

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        client = self._ensure_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=self._normalize(messages),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — surface as ProviderError
            logger.exception("OpenAI completion failed")
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            raw=response,
        )

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        client = self._ensure_client()
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=self._normalize(messages),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("OpenAI streaming failed")
            raise ProviderError(f"OpenAI stream failed: {exc}") from exc

        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
            except (IndexError, AttributeError):
                delta = None
            if delta:
                yield delta
