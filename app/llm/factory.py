"""Provider factory.

Centralizes the mapping from provider name → concrete :class:`LLMProvider`
subclass, and supplies a model catalogue used to populate UI dropdowns.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from app.config import Settings, get_settings
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.openai_provider import OpenAIProvider
from app.utils.errors import ConfigurationError


_PROVIDERS: Dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


# Curated, current model lists. Users can type a custom model in the UI to
# bypass these if a new model ships.
_MODEL_CATALOGUE: Dict[str, List[str]] = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        "claude-sonnet-4-5",
        "claude-opus-4-5",
        "claude-haiku-4-5",
        "claude-3-7-sonnet-latest",
        "claude-3-5-haiku-latest",
    ],
    "ollama": [
        "llama3.1",
        "llama3.2",
        "mistral",
        "qwen2.5",
        "phi3",
        "gemma2",
    ],
}


def available_providers() -> List[str]:
    """Return the list of registered provider names."""
    return list(_PROVIDERS.keys())


def available_models(provider: str) -> List[str]:
    """Return suggested models for ``provider``. Custom names are allowed."""
    return list(_MODEL_CATALOGUE.get(provider, []))


def build_provider(
    provider: str,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> LLMProvider:
    """Construct an :class:`LLMProvider`.

    Falls back to :class:`Settings` defaults for any value not supplied,
    and resolves the API key from settings if the caller did not provide one.
    """

    if provider not in _PROVIDERS:
        raise ConfigurationError(
            f"Unknown provider '{provider}'. Available: {available_providers()}"
        )

    settings = settings or get_settings()
    cls = _PROVIDERS[provider]

    resolved_key = api_key
    resolved_base_url = base_url
    if provider == "openai":
        resolved_key = resolved_key or settings.openai_api_key
    elif provider == "anthropic":
        resolved_key = resolved_key or settings.anthropic_api_key
    elif provider == "ollama":
        resolved_base_url = resolved_base_url or settings.ollama_base_url

    return cls(
        model=model or settings.default_model,
        temperature=settings.default_temperature if temperature is None else temperature,
        max_tokens=settings.default_max_tokens if max_tokens is None else max_tokens,
        api_key=resolved_key,
        base_url=resolved_base_url,
    )
