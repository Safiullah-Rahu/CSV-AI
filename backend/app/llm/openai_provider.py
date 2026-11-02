"""OpenAI Chat Completions over HTTP.

Covers OpenAI itself, Azure OpenAI (``azure=True``: deployment URL +
``api-key`` header + ``api-version``) and every OpenAI-compatible server
(vLLM, Hugging Face TGI ``/v1``, LM Studio, LocalAI, llama.cpp, custom
gateways) through ``base_url``. Uses plain ``httpx`` — no vendor SDK.

OpenAI's own API takes ``max_completion_tokens`` (reasoning models reject
``max_tokens``); compatible servers keep ``max_tokens``. Models that only accept
the default sampling reject ``temperature``: the first such 400 is retried
without it and the model is remembered for the life of the process.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

from app.llm.base import ChatResponse, LLMProvider, Message
from app.utils.errors import ConfigurationError, NotAvailable, ProviderError
from app.utils.logging import get_logger

logger = get_logger("llm.openai")

_NON_TEXT_MODELS = ("image", "dall-e", "tts", "whisper", "audio", "realtime", "transcribe", "moderation", "sora")

# Models learned (from a 400) to accept only the default temperature.
_DEFAULT_SAMPLING_ONLY: set[str] = set()


def _rejects_temperature(exc: ProviderError) -> bool:
    msg = exc.message.lower()
    return "temperature" in msg and ("unsupported" in msg or "not supported" in msg or "does not support" in msg)


class OpenAIProvider(LLMProvider):
    """OpenAI / Azure OpenAI / OpenAI-compatible chat provider."""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        api_key: str | None = None,
        base_url: str | None = None,
        azure: bool = False,
        api_version: str = "2024-06-01",
        require_key: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key, base_url=base_url, **kwargs)
        self.azure = azure
        self.api_version = api_version
        self.require_key = require_key

    def _endpoint(self) -> tuple[str, dict[str, str]]:
        if self.require_key and not self.api_key:
            raise ConfigurationError("API key is missing for this provider.")
        if self.azure:
            if not self.base_url:
                raise ConfigurationError("Azure OpenAI needs the resource endpoint URL.")
            url = f"{self.base_url.rstrip('/')}/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}"
            return url, {"api-key": self.api_key or ""}
        base = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return f"{base}/chat/completions", headers

    def list_models(self) -> list[str]:
        """``GET /models`` on OpenAI and on compatible servers (vLLM, LM Studio, TGI, gateways)."""
        if self.azure:
            raise NotAvailable("Azure OpenAI addresses models by deployment name. Type the deployment.")
        _, headers = self._endpoint()
        base = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        items = [m for m in (self._get_json(f"{base}/models", headers).get("data") or []) if m.get("id")]
        if self._official_api():  # OpenAI also lists image, speech and moderation models; pickers need text ones
            items = [m for m in items if not any(t in str(m["id"]) for t in _NON_TEXT_MODELS)]
        items.sort(key=lambda m: m.get("created") or 0, reverse=True)  # newest first
        return [str(m["id"]) for m in items]

    def _official_api(self) -> bool:
        return not self.azure and (self.base_url is None or "api.openai.com" in self.base_url)

    def _payload(self, messages: Iterable[Message], stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": self._normalize(messages), "stream": stream}
        if self.model not in _DEFAULT_SAMPLING_ONLY:
            payload["temperature"] = self.temperature
        payload["max_completion_tokens" if self._official_api() else "max_tokens"] = self.max_tokens
        if stream and not self.options.get("no_stream_usage"):
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _learn_default_sampling(self, exc: ProviderError) -> bool:
        if self.model in _DEFAULT_SAMPLING_ONLY or not _rejects_temperature(exc):
            return False
        logger.info("model %s accepts only the default temperature; retrying without it", self.model)
        _DEFAULT_SAMPLING_ONLY.add(self.model)
        return True

    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        url, headers = self._endpoint()
        messages = list(messages)
        try:
            data = self._post_json(url, self._payload(messages, stream=False), headers)
        except ProviderError as exc:
            if not self._learn_default_sampling(exc):
                raise
            data = self._post_json(url, self._payload(messages, stream=False), headers)
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return ChatResponse(
            content=(choice.get("message") or {}).get("content") or "",
            model=data.get("model", self.model),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            raw=data,
        )

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        messages = list(messages)
        emitted = False
        try:
            for piece in self._stream_once(messages):
                emitted = True
                yield piece
        except ProviderError as exc:
            # A rejected parameter fails before any output; never replay a partial answer.
            if emitted or not self._learn_default_sampling(exc):
                raise
            yield from self._stream_once(messages)

    def _stream_once(self, messages: list[Message]) -> Iterator[str]:
        url, headers = self._endpoint()
        self.last_usage = None
        for line in self._stream_lines(url, self._payload(messages, stream=True), headers):
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                event = json.loads(body)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                u = event["usage"]
                self.last_usage = (int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0))
            for choice in event.get("choices") or []:
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    yield delta
