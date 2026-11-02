"""Anthropic Claude provider (Messages API over HTTP).

Anthropic splits the system prompt out of the messages array, so we extract
any ``system`` role messages and pass them as a top-level argument.

API version 2026-10-21: sampling parameters (``temperature``, ``top_p``,
``top_k``) are rejected on every model, so the configured temperature is not
sent; current Claude models only sample at their defaults anyway. Depth and
cost are tuned with the optional ``effort`` setting (``output_config.effort``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

from app.llm.base import ChatResponse, LLMProvider, Message
from app.utils.errors import ConfigurationError, ProviderError
from app.utils.logging import get_logger

logger = get_logger("llm.anthropic")

API_VERSION = "2026-10-21"
EFFORTS = {"low", "medium", "high", "xhigh", "max"}


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key, base_url=base_url, **kwargs)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _split_system(messages: Iterable[Message]) -> tuple[str, list[dict]]:
        system_chunks: list[str] = []
        rest: list[dict] = []
        for msg in messages:
            role = msg.role if isinstance(msg, Message) else msg["role"]
            content = msg.content if isinstance(msg, Message) else msg["content"]
            if role == "system":
                system_chunks.append(content)
            else:
                rest.append({"role": role, "content": content})
        return "\n\n".join(system_chunks), rest

    def _request(self, messages: Iterable[Message], stream: bool) -> tuple[str, dict[str, Any], dict[str, str]]:
        if not self.api_key:
            raise ConfigurationError("Anthropic API key is missing.")
        system, chat = self._split_system(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": chat,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        effort = self.options.get("effort")
        if effort:
            if effort not in EFFORTS:
                raise ConfigurationError(f"Unsupported effort '{effort}'. Use one of {sorted(EFFORTS)}.")
            payload["output_config"] = {"effort": effort}
        return f"{self._base()}/v1/messages", payload, self._headers()

    def _base(self) -> str:
        return (self.base_url or "https://api.anthropic.com").rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"x-api-key": self.api_key or "", "anthropic-version": API_VERSION, "content-type": "application/json"}
        if self.options.get("workspace_id"):
            headers["anthropic-workspace-id"] = str(self.options["workspace_id"])
        return headers

    def list_models(self) -> list[str]:
        """The Models API (``GET /v1/models``), newest first, following pagination."""
        if not self.api_key:
            raise ConfigurationError("Anthropic API key is missing.")
        ids: list[str] = []
        after: str | None = None
        for _ in range(10):
            query = "?limit=100" + (f"&after_id={after}" if after else "")
            data = self._get_json(f"{self._base()}/v1/models{query}", self._headers())
            ids += [str(m["id"]) for m in data.get("data") or [] if m.get("id")]
            after = data.get("last_id")
            if not data.get("has_more") or not after:
                break
        return ids

    # ------------------------------------------------------------------ API
    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        url, payload, headers = self._request(messages, stream=False)
        data = self._post_json(url, payload, headers)
        if data.get("stop_reason") == "refusal":
            raise ProviderError("Claude declined this request (stop_reason: refusal).")
        text_parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        usage = data.get("usage") or {}
        return ChatResponse(
            content="".join(text_parts),
            model=data.get("model", self.model),
            prompt_tokens=int(usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("output_tokens") or 0),
            raw=data,
        )

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        url, payload, headers = self._request(messages, stream=True)
        prompt_tokens = completion_tokens = 0
        self.last_usage = None
        for line in self._stream_lines(url, payload, headers):
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "message_start":
                prompt_tokens = int(((event.get("message") or {}).get("usage") or {}).get("input_tokens") or 0)
            elif kind == "content_block_delta":
                text = (event.get("delta") or {}).get("text")
                if text:
                    yield text
            elif kind == "message_delta":
                completion_tokens = int((event.get("usage") or {}).get("output_tokens") or 0)
                if (event.get("delta") or {}).get("stop_reason") == "refusal":
                    raise ProviderError("Claude declined this request (stop_reason: refusal).")
            elif kind == "error":
                raise ProviderError(f"Anthropic stream error: {event.get('error')}")
        self.last_usage = (prompt_tokens, completion_tokens)
