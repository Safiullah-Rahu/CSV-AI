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


def test_anthropic_request_matches_api_2026_10_21() -> None:
    provider = AnthropicProvider(model="claude-sonnet-5-5", api_key="x", temperature=0.1, options={"effort": "low", "workspace_id": "wrkspc_test"})
    _, payload, headers = provider._request([Message(role="user", content="hi")], stream=True)
    assert headers["anthropic-version"] == "2026-10-21"
    assert headers["anthropic-workspace-id"] == "wrkspc_test"
    assert not {"temperature", "top_p", "top_k"} & set(payload)  # rejected on every model
    assert payload["output_config"] == {"effort": "low"} and payload["max_tokens"] == provider.max_tokens
    from app.llm.factory import instantiate

    configured = instantiate("anthropic", model="claude-sonnet-5-5", api_key="x", base_url=None, temperature=0.2, max_tokens=800, settings={"effort": "low", "workspace_id": "w"})
    assert configured.options == {"effort": "low", "workspace_id": "w"}  # provider settings reach the request
    with pytest.raises(ConfigurationError):
        AnthropicProvider(model="claude-sonnet-5-5", api_key="x", options={"effort": "extreme"})._request([Message(role="user", content="hi")], stream=False)


def test_openai_request_shape_for_official_and_compatible_endpoints() -> None:
    official = OpenAIProvider(model="gpt-5.4-mini", api_key="x")
    payload = official._payload([Message(role="user", content="hi")], stream=True)
    assert payload["max_completion_tokens"] == official.max_tokens and "max_tokens" not in payload
    compat = OpenAIProvider(model="llama3", api_key=None, base_url="http://vllm:8000/v1", require_key=False)
    assert "max_tokens" in compat._payload([Message(role="user", content="hi")], stream=False)


def test_openai_retries_once_without_temperature(monkeypatch) -> None:
    from app.llm import openai_provider
    from app.utils.errors import ProviderError

    provider = OpenAIProvider(model="reasoning-model-test", api_key="x")
    sent: list[dict] = []

    def fake_post(url, payload, headers):  # noqa: ANN001, ANN202
        sent.append(payload)
        if "temperature" in payload:
            raise ProviderError("openai rejected the request (HTTP 400): Unsupported value: 'temperature' does not support 0.2 with this model.")
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr(provider, "_post_json", fake_post)
    assert provider.complete([Message(role="user", content="hi")]).content == "ok"
    assert len(sent) == 2 and "temperature" not in sent[1]
    assert "reasoning-model-test" in openai_provider._DEFAULT_SAMPLING_ONLY
    provider.complete([Message(role="user", content="hi")])
    assert len(sent) == 3  # remembered: no second failed attempt


def test_anthropic_splits_system_messages() -> None:
    provider = AnthropicProvider(model="claude-sonnet-4-5", api_key="x")
    system, rest = provider._split_system([
        Message(role="system", content="be concise"),
        Message(role="user", content="hello"),
    ])
    assert system == "be concise"
    assert rest == [{"role": "user", "content": "hello"}]
