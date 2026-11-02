"""Provider factory.

Centralizes the mapping from provider kind → concrete :class:`LLMProvider`
subclass, and supplies a model catalogue used to populate UI dropdowns
(custom model ids are always allowed).
"""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMProvider
from app.llm.cloud_providers import BedrockProvider, GeminiProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.openai_provider import OpenAIProvider
from app.utils.errors import ConfigurationError

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
    "bedrock": BedrockProvider,
    "azure_openai": OpenAIProvider,
    "vllm": OpenAIProvider,
    "openai_compatible": OpenAIProvider,
    "huggingface": OpenAIProvider,
    "custom": OpenAIProvider,
}

PROVIDER_LABELS: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
    "azure_openai": "Azure OpenAI",
    "bedrock": "AWS Bedrock",
    "ollama": "Ollama",
    "vllm": "vLLM",
    "openai_compatible": "OpenAI-compatible API",
    "huggingface": "Hugging Face (TGI / Inference Endpoints)",
    "custom": "Custom self-hosted endpoint",
    "cohere": "Cohere (embeddings & rerank)",
    "voyage": "Voyage AI (embeddings)",
}

# Kinds that need an API key; the others can run keyless on a private network.
KEYLESS_KINDS = {"ollama", "vllm", "openai_compatible", "huggingface", "custom", "bedrock"}
EMBEDDING_ONLY_KINDS = {"cohere", "voyage"}

# Curated suggestions. Users can type a custom model id if a new model ships.
_MODEL_CATALOGUE: dict[str, list[str]] = {
    "openai": ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-4.1", "gpt-4.1-mini"],
    "anthropic": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
    "ollama": ["qwen2.5:7b-instruct", "llama3.1", "llama3.2", "mistral", "qwen3:4b", "phi3", "gemma2"],
    "gemini": ["gemini-2.5-pro", "gemini-2.5-flash"],
    "bedrock": ["anthropic.claude-sonnet-4-5", "amazon.nova-pro-v1:0", "meta.llama3-1-70b-instruct-v1:0"],
    "azure_openai": [],
    "vllm": [],
    "openai_compatible": [],
    "huggingface": [],
    "custom": [],
}


def available_providers() -> list[str]:
    """Return the list of registered provider names."""
    return list(_PROVIDERS.keys())


def available_models(provider: str) -> list[str]:
    """Return suggested models for ``provider``. Custom names are allowed."""
    return list(_MODEL_CATALOGUE.get(provider, []))


def instantiate(
    kind: str,
    *,
    model: str,
    api_key: str | None,
    base_url: str | None,
    temperature: float,
    max_tokens: int,
    settings: dict[str, Any] | None = None,
) -> LLMProvider:
    if kind not in _PROVIDERS:
        raise ConfigurationError(f"Unknown provider '{kind}'. Available: {available_providers()}")
    settings = settings or {}
    common: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": api_key,
        "base_url": base_url,
        "timeout": float(settings.get("timeout_s", get_settings().llm_timeout_seconds)),
        "max_retries": int(settings.get("max_retries", 2)),
        "options": {k: v for k, v in settings.items() if k in ("think", "num_ctx", "repeat_penalty", "region", "access_key_id", "workspace_id", "no_stream_usage", "effort")},
    }
    cls = _PROVIDERS[kind]
    if kind == "azure_openai":
        return OpenAIProvider(azure=True, api_version=settings.get("api_version", "2024-06-01"), **common)
    if cls is OpenAIProvider and kind != "openai":
        provider = OpenAIProvider(require_key=False, **common)
        provider.name = kind
        return provider
    return cls(**common)


def build_provider(
    provider: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    settings: Settings | None = None,
) -> LLMProvider:
    """Construct a provider from platform settings (CSV-AI compatible entry point)."""

    if provider not in _PROVIDERS:
        raise ConfigurationError(f"Unknown provider '{provider}'. Available: {available_providers()}")

    settings = settings or get_settings()
    resolved_key = api_key
    resolved_base_url = base_url
    if provider == "openai":
        resolved_key = resolved_key or settings.openai_api_key
    elif provider == "anthropic":
        resolved_key = resolved_key or settings.anthropic_api_key
    elif provider == "ollama":
        resolved_base_url = resolved_base_url or settings.ollama_base_url

    return instantiate(
        provider,
        model=model or settings.default_model,
        api_key=resolved_key,
        base_url=resolved_base_url,
        temperature=settings.default_temperature if temperature is None else temperature,
        max_tokens=settings.default_max_tokens if max_tokens is None else max_tokens,
    )
