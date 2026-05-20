"""Provider-agnostic LLM interface.

Every provider implementation must subclass :class:`LLMProvider` and
support both ``complete`` (one-shot) and ``stream`` (incremental token
generator). Keeping the surface small makes it trivial to add a new
backend without rippling changes through services or UI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Literal, Optional


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
    raw: Optional[object] = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(ABC):
    """Abstract base for all LLM providers.

    Implementations should be cheap to construct (configuration only); any
    HTTP client setup happens lazily on the first call. This keeps the
    Streamlit UI responsive when the user is just browsing settings.
    """

    name: str = "base"

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url

    # ------------------------------------------------------------------ API
    @abstractmethod
    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        """Return a single, fully-formed response."""

    @abstractmethod
    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        """Yield incremental text chunks as the model generates."""

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _normalize(messages: Iterable[Message]) -> List[dict]:
        return [m.to_dict() for m in messages]
