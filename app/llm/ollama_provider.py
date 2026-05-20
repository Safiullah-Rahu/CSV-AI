"""Local Ollama provider.

Talks to a locally-running Ollama daemon via its HTTP API. Useful for
running CSV-AI fully offline against models like llama3.1, mistral, qwen,
phi, etc.
"""

from __future__ import annotations

import json
from typing import Iterable, Iterator, Optional

from app.llm.base import ChatResponse, LLMProvider, Message
from app.utils.errors import ConfigurationError, ProviderError
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
        api_key: Optional[str] = None,
        base_url: Optional[str] = "http://localhost:11434",
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url or "http://localhost:11434",
        )

    # --------------------------------------------------------------- helpers
    def _payload(self, messages: Iterable[Message], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": self._normalize(messages),
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

    def _ensure_requests(self):
        try:
            import requests  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "The 'requests' package is required for the Ollama provider."
            ) from exc

    # ------------------------------------------------------------------ API
    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        self._ensure_requests()
        import requests

        url = f"{self.base_url.rstrip('/')}/api/chat"
        try:
            response = requests.post(url, json=self._payload(messages, stream=False), timeout=300)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ollama completion failed")
            raise ProviderError(
                f"Ollama request failed ({url}): {exc}. Is `ollama serve` running?"
            ) from exc

        data = response.json()
        return ChatResponse(
            content=(data.get("message") or {}).get("content", ""),
            model=data.get("model", self.model),
            prompt_tokens=data.get("prompt_eval_count", 0) or 0,
            completion_tokens=data.get("eval_count", 0) or 0,
            raw=data,
        )

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        self._ensure_requests()
        import requests

        url = f"{self.base_url.rstrip('/')}/api/chat"
        try:
            response = requests.post(
                url,
                json=self._payload(messages, stream=True),
                stream=True,
                timeout=300,
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ollama streaming failed")
            raise ProviderError(
                f"Ollama stream failed ({url}): {exc}. Is `ollama serve` running?"
            ) from exc

        for line in response.iter_lines():
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = (event.get("message") or {}).get("content", "")
            if chunk:
                yield chunk
            if event.get("done"):
                break
