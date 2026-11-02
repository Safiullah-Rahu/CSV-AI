"""Local Ollama provider.

Talks to a locally-running Ollama daemon via its HTTP API. Useful for
running fully offline against models like llama3.1, mistral, qwen, phi.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from typing import Any

import httpx

from app.llm.base import ChatResponse, LLMProvider, Message
from app.utils.errors import ProviderError
from app.utils.logging import get_logger

logger = get_logger("llm.ollama")


class OllamaProvider(LLMProvider):
    """Local Ollama chat provider."""

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.1",
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        api_key: str | None = None,
        base_url: str | None = "http://localhost:11434",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url or "http://localhost:11434",
            **kwargs,
        )

    # --------------------------------------------------------------- helpers
    def _payload(self, messages: Iterable[Message], stream: bool) -> dict:
        normalized = self._normalize(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": normalized,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "repeat_penalty": float(self.options.get("repeat_penalty", 1.12)),
                # Ollama silently truncates prompts longer than its default window (2-4K
                # tokens), which drops the system prompt and question. Size it explicitly.
                "num_ctx": int(self.options.get("num_ctx") or self._context_for(normalized)),
            },
        }
        if "think" in self.options:
            payload["think"] = bool(self.options["think"])
        return payload

    def _context_for(self, messages: list[dict]) -> int:
        from app.utils.tokens import estimate_tokens

        needed = int(sum(estimate_tokens(m.get("content", "")) for m in messages) * 1.25) + self.max_tokens + 256
        size = 2048
        while size < needed and size < 32768:
            size *= 2
        return size

    def _url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/chat"

    # ------------------------------------------------------------------ API
    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        try:
            data = self._post_json(self._url(), self._payload(messages, stream=False), {})
        except ProviderError as exc:
            raise ProviderError(f"{exc}. Is `ollama serve` running at {self.base_url}?") from exc
        return ChatResponse(
            content=(data.get("message") or {}).get("content", ""),
            model=data.get("model", self.model),
            prompt_tokens=data.get("prompt_eval_count", 0) or 0,
            completion_tokens=data.get("eval_count", 0) or 0,
            raw=data,
        )

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        self.last_usage = None
        for line in self._stream_lines(self._url(), self._payload(messages, stream=True), {}):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("error"):
                raise ProviderError(f"Ollama error: {event['error']}")
            chunk = (event.get("message") or {}).get("content", "")
            if chunk:
                yield chunk
            if event.get("done"):
                self.last_usage = (int(event.get("prompt_eval_count") or 0), int(event.get("eval_count") or 0))
                break

    def health(self) -> tuple[bool, float, str | None]:
        start = time.perf_counter()
        try:
            r = httpx.get(f"{self.base_url.rstrip('/')}/api/tags", timeout=5)
            r.raise_for_status()
            names = {m.get("name") for m in r.json().get("models", [])}
            ok = self.model in names or f"{self.model}:latest" in names
            return ok, (time.perf_counter() - start) * 1000, None if ok else f"Model '{self.model}' is not pulled on this Ollama server."
        except Exception as exc:  # noqa: BLE001
            return False, (time.perf_counter() - start) * 1000, str(exc)[:300]

    def list_models(self) -> list[str]:
        return [m["name"] for m in self._get_json(f"{self.base_url.rstrip('/')}/api/tags", {}).get("models", [])]
