"""Provider-agnostic LLM interface.

Every provider implementation subclasses :class:`LLMProvider` and supports
both ``complete`` (one-shot) and ``stream`` (incremental text). Business logic
never imports a vendor SDK; it talks to this interface only.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from app.utils.errors import NotAvailable, ProviderError

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    """A single chat message in our provider-neutral schema."""

    role: Role
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """Full response returned by :meth:`LLMProvider.complete`."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: object | None = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(ABC):
    """Abstract base for all LLM providers.

    Implementations are cheap to construct (configuration only); network
    clients are created lazily on the first call.
    """

    name: str = "base"

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.options = options or {}
        # (prompt_tokens, completion_tokens) of the last streamed call, when reported.
        self.last_usage: tuple[int, int] | None = None

    # ------------------------------------------------------------------ API
    @abstractmethod
    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        """Return a single, fully-formed response."""

    @abstractmethod
    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        """Yield incremental text chunks as the model generates."""

    def health(self) -> tuple[bool, float, str | None]:
        """Cheap liveness probe: a one-token completion."""
        start = time.perf_counter()
        original = self.max_tokens
        try:
            self.max_tokens = 4
            self.complete([Message(role="user", content="ping")])
            return True, (time.perf_counter() - start) * 1000, None
        except Exception as exc:  # noqa: BLE001
            return False, (time.perf_counter() - start) * 1000, str(exc)[:300]
        finally:
            self.max_tokens = original

    def list_models(self) -> list[str]:
        """Model ids this endpoint offers right now (see ``app.llm.catalog``)."""
        raise NotAvailable(f"{self.name} does not publish a model list. Type the model id.")

    # --------------------------------------------------------------- helpers
    def _get_json(self, url: str, headers: dict[str, str]) -> dict[str, Any]:
        try:
            r = httpx.get(url, headers=headers, timeout=min(self.timeout, 20.0))
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} model list failed: {exc}") from exc
        if r.status_code >= 400:
            raise ProviderError(f"{self.name} rejected the model list request (HTTP {r.status_code}): {_brief(r)}")
        return r.json()

    @staticmethod
    def _normalize(messages: Iterable[Message]) -> list[dict]:
        return [m.to_dict() if isinstance(m, Message) else dict(m) for m in messages]

    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        """POST with retry on 429/5xx and transport errors (exponential backoff)."""
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
            except httpx.HTTPError as exc:
                last = exc
            else:
                if r.status_code < 400:
                    return r.json()
                if r.status_code not in (408, 409, 429) and r.status_code < 500:
                    raise ProviderError(f"{self.name} rejected the request (HTTP {r.status_code}): {_brief(r)}")
                last = ProviderError(f"{self.name} HTTP {r.status_code}: {_brief(r)}")
            if attempt < self.max_retries:
                time.sleep(min(10.0, 0.75 * 2**attempt))
        raise ProviderError(f"{self.name} request failed: {last}")

    def _stream_lines(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> Iterator[str]:
        try:
            with httpx.stream("POST", url, json=payload, headers=headers, timeout=self.timeout) as r:
                if r.status_code >= 400:
                    r.read()
                    raise ProviderError(f"{self.name} stream rejected (HTTP {r.status_code}): {_brief(r)}")
                yield from r.iter_lines()
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} stream failed: {exc}") from exc


def _brief(r: httpx.Response) -> str:
    try:
        data = r.json()
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            return str(err.get("message", err))[:300]
        return str(err or data)[:300]
    except Exception:  # noqa: BLE001
        return r.text[:300]
