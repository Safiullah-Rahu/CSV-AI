"""LLM provider layer.

Exposes a unified :class:`LLMProvider` interface so the rest of the app
does not care whether the backing model is OpenAI, Anthropic, or a local
Ollama instance.
"""

from app.llm.base import LLMProvider, Message, ChatResponse
from app.llm.factory import build_provider, available_models

__all__ = [
    "LLMProvider",
    "Message",
    "ChatResponse",
    "build_provider",
    "available_models",
]
