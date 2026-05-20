"""Tests for the LLM provider factory.

These tests don't make real API calls — they verify wiring only.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.llm import Message
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.factory import (
    available_models,
    available_providers,
    build_provider,
)
from app.llm.ollama_provider import OllamaProvider
from app.llm.openai_provider import OpenAIProvider
from app.utils.errors import ConfigurationError


def _fake_settings(**overrides) -> Settings:
    defaults = dict(
        openai_api_key="sk-test",
        anthropic_api_key="ant-test",
        ollama_base_url="http://localhost:11434",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_available_providers_includes_all() -> None:
    providers = available_providers()
    assert {"openai", "anthropic", "ollama"} <= set(providers)


def test_available_models_returns_list() -> None:
    assert available_models("openai")
    assert available_models("anthropic")
    assert available_models("ollama")


def test_build_openai_provider() -> None:
    provider = build_provider(
        "openai",
        model="gpt-4o-mini",
        settings=_fake_settings(),
    )
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-4o-mini"
    assert provider.api_key == "sk-test"


def test_build_anthropic_provider() -> None:
    provider = build_provider(
        "anthropic",
        model="claude-sonnet-4-5",
        settings=_fake_settings(),
    )
    assert isinstance(provider, AnthropicProvider)
    assert provider.api_key == "ant-test"


def test_build_ollama_provider_uses_base_url() -> None:
    provider = build_provider(
        "ollama",
        model="llama3.1",
        settings=_fake_settings(),
    )
    assert isinstance(provider, OllamaProvider)
    assert provider.base_url.startswith("http://")


def test_unknown_provider_raises() -> None:
    with pytest.raises(ConfigurationError):
        build_provider("not-a-real-provider", settings=_fake_settings())


def test_anthropic_splits_system_messages() -> None:
    provider = AnthropicProvider(model="claude-sonnet-4-5", api_key="x")
    system, rest = provider._split_system([
        Message(role="system", content="be concise"),
        Message(role="user", content="hello"),
    ])
    assert system == "be concise"
    assert rest == [{"role": "user", "content": "hello"}]
