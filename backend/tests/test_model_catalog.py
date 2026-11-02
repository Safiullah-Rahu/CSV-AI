"""Live model discovery and the platform model fallback (no network: provider HTTP is stubbed)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.cloud_providers import GeminiProvider
from app.llm.openai_provider import OpenAIProvider
from app.utils.errors import NotAvailable


def test_openai_lists_newest_first(monkeypatch) -> None:
    p = OpenAIProvider(model="gpt-5.4-mini", api_key="x")
    monkeypatch.setattr(p, "_get_json", lambda url, headers: {"data": [{"id": "gpt-4.1", "created": 1}, {"id": "gpt-6-sol", "created": 9}, {"id": "gpt-image-2", "created": 8}, {"id": "gpt-5.4-mini", "created": 5}, {"id": "text-embedding-3-large", "created": 2}]})
    assert p.list_models() == ["gpt-6-sol", "gpt-5.4-mini", "text-embedding-3-large", "gpt-4.1"]
    with pytest.raises(NotAvailable):
        OpenAIProvider(model="dep", api_key="x", base_url="https://r.openai.azure.com", azure=True).list_models()


def test_anthropic_follows_pagination(monkeypatch) -> None:
    p = AnthropicProvider(model="claude-sonnet-5-5", api_key="x", options={"workspace_id": "w"})
    pages = {None: {"data": [{"id": "claude-opus-5-5"}, {"id": "claude-sonnet-5-5"}], "has_more": True, "last_id": "claude-sonnet-5-5"},
             "claude-sonnet-5-5": {"data": [{"id": "claude-haiku-5-5"}], "has_more": False, "last_id": "claude-haiku-5-5"}}
    seen: list[str] = []

    def fake(url: str, headers: dict[str, str]) -> dict:
        seen.append(url)
        assert headers["anthropic-version"] == "2026-10-21" and headers["anthropic-workspace-id"] == "w"
        return pages["claude-sonnet-5-5" if "after_id=" in url else None]

    monkeypatch.setattr(p, "_get_json", fake)
    assert p.list_models() == ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
    assert len(seen) == 2 and seen[0].endswith("/v1/models?limit=100")


def test_gemini_keeps_generation_and_embedding_models(monkeypatch) -> None:
    p = GeminiProvider(model="gemini-2.5-flash", api_key="x")
    monkeypatch.setattr(p, "_get_json", lambda url, headers: {"models": [
        {"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/aqa", "supportedGenerationMethods": ["generateAnswer"]},
    ]})
    assert p.list_models() == ["gemini-2.5-pro", "text-embedding-004"]


def test_models_endpoint_stores_live_list_as_suggestions(make_user, monkeypatch) -> None:
    owner = make_user("Model Catalog Org")
    created = owner.post("/ai/providers", json={"name": "OpenAI", "kind": "openai", "api_key": "sk-test-not-real", "default_model": "gpt-5.4-mini"})
    assert created.status_code == 201, created.text
    pid = created.json()["id"]
    monkeypatch.setattr(OpenAIProvider, "list_models", lambda self: ["gpt-6-sol", "gpt-5.4-mini"])
    r = owner.get(f"/ai/providers/{pid}/models", params={"refresh": True})
    assert r.status_code == 200 and r.json() == ["gpt-6-sol", "gpt-5.4-mini"]
    listed = next(p for p in owner.get("/ai/providers").json() if p["id"] == pid)
    assert listed["suggested_models"] == ["gpt-6-sol", "gpt-5.4-mini"] and listed["models_refreshed_at"]


def test_platform_fallback_prefers_default_provider(monkeypatch) -> None:
    from app.llm import registry

    org = SimpleNamespace(settings={})
    both = Settings(platform_llm_fallback=True, default_provider="anthropic", anthropic_api_key="a", openai_api_key="o", anthropic_workspace_id="w")
    monkeypatch.setattr(registry, "get_settings", lambda: both)
    llm = registry.platform_fallback(org)  # type: ignore[arg-type]
    assert llm.kind == "anthropic" and llm.provider.model == "claude-sonnet-5-5" and llm.provider.options["workspace_id"] == "w"

    only_openai = Settings(platform_llm_fallback=True, default_provider="anthropic", openai_api_key="o", default_model="gpt-4.1-mini")
    monkeypatch.setattr(registry, "get_settings", lambda: only_openai)
    llm = registry.platform_fallback(org)  # type: ignore[arg-type]
    assert llm.kind == "openai" and llm.provider.model == "gpt-4.1-mini"

    restricted = SimpleNamespace(settings={"ai": {"allowed_providers": ["ollama"]}})
    assert registry.platform_fallback(restricted) is None  # type: ignore[arg-type]
    monkeypatch.setattr(registry, "get_settings", lambda: Settings(openai_api_key="o"))
    assert registry.platform_fallback(org) is None  # type: ignore[arg-type]  # fallback is opt-in
