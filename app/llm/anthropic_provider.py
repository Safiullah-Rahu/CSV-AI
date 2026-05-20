"""Anthropic Claude provider.

Anthropic splits the system prompt out of the messages array, so we extract
any ``system`` role messages and pass them as a top-level argument.
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Tuple

from app.llm.base import ChatResponse, LLMProvider, Message
from app.utils.errors import ConfigurationError, ProviderError
from app.utils.logging import get_logger

logger = get_logger("llm.anthropic")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider (claude-sonnet-4, claude-haiku-4, ...)."""

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
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

    # --------------------------------------------------------------- helpers
    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ConfigurationError(
                "Anthropic API key is missing. Set ANTHROPIC_API_KEY or enter it in the sidebar."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "The 'anthropic' package is not installed. Run: pip install anthropic"
            ) from exc

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = anthropic.Anthropic(**kwargs)
        return self._client

    @staticmethod
    def _split_system(messages: Iterable[Message]) -> Tuple[str, List[dict]]:
        system_chunks: List[str] = []
        rest: List[dict] = []
        for msg in messages:
            if msg.role == "system":
                system_chunks.append(msg.content)
            else:
                rest.append({"role": msg.role, "content": msg.content})
        return "\n\n".join(system_chunks), rest

    # ------------------------------------------------------------------ API
    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        client = self._ensure_client()
        system, chat = self._split_system(messages)
        try:
            response = client.messages.create(
                model=self.model,
                system=system or None,
                messages=chat,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Anthropic completion failed")
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        text_parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        usage = getattr(response, "usage", None)
        return ChatResponse(
            content="".join(text_parts),
            model=response.model,
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
            raw=response,
        )

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        client = self._ensure_client()
        system, chat = self._split_system(messages)
        try:
            with client.messages.stream(
                model=self.model,
                system=system or None,
                messages=chat,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        yield text
        except Exception as exc:  # noqa: BLE001
            logger.exception("Anthropic streaming failed")
            raise ProviderError(f"Anthropic stream failed: {exc}") from exc
