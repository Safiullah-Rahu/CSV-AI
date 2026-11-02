"""Resolve organisation-configured AI providers into live clients.

Organisations store provider configs in ``ai_providers`` (credentials in the
encrypted ``credentials`` table). This module turns them into
:class:`LLMProvider` / embedder instances, applies the organisation's
provider allow-list, rate limits and monthly token caps, and wires a
fallback provider when configured.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.ratelimit import enforce
from app.core.security import decrypt_secret
from app.llm.base import ChatResponse, LLMProvider, Message
from app.llm.factory import EMBEDDING_ONLY_KINDS, instantiate
from app.models import AIProviderConfig, Credential, Organization, UsageRecord
from app.rag.embeddings import Embedder, build_embedder
from app.utils.errors import ConfigurationError, LimitExceeded, PermissionDenied, ProviderError


def credential_value(db: Session, credential_id: uuid.UUID | None) -> str | None:
    if credential_id is None:
        return None
    cred = db.get(Credential, credential_id)
    return decrypt_secret(cred.ciphertext) if cred else None


@dataclass
class ResolvedLLM:
    provider: LLMProvider
    config: AIProviderConfig | None
    kind: str

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        if not self.config:
            return 0.0
        for m in self.config.models or []:
            if m.get("id") == self.provider.model:
                return round(tokens_in / 1000 * float(m.get("input_cost_per_1k") or 0) + tokens_out / 1000 * float(m.get("output_cost_per_1k") or 0), 6)
        return 0.0


class FallbackProvider(LLMProvider):
    """Try the primary provider; on failure (before any output) use the fallback."""

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        super().__init__(model=primary.model, temperature=primary.temperature, max_tokens=primary.max_tokens)
        self.primary, self.fallback = primary, fallback
        self.name = primary.name
        self.used: LLMProvider = primary

    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        msgs = list(messages)
        try:
            self.used = self.primary
            return self.primary.complete(msgs)
        except ProviderError:
            self.used = self.fallback
            return self.fallback.complete(msgs)

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        msgs = list(messages)
        started = False
        try:
            self.used = self.primary
            for chunk in self.primary.stream(msgs):
                started = True
                yield chunk
            self.last_usage = self.primary.last_usage
        except ProviderError:
            if started:
                raise
            self.used = self.fallback
            yield from self.fallback.stream(msgs)
            self.last_usage = self.fallback.last_usage


def _allowed(org: Organization, kind: str) -> bool:
    allowed = ((org.settings or {}).get("ai") or {}).get("allowed_providers") or []
    return not allowed or kind in allowed


def _instantiate(db: Session, cfg: AIProviderConfig, model: str | None, overrides: dict[str, Any] | None = None) -> LLMProvider:
    s = {**(cfg.settings or {}), **(overrides or {})}
    return instantiate(
        cfg.kind,
        model=model or cfg.default_model or "",
        api_key=credential_value(db, cfg.credential_id),
        base_url=cfg.base_url,
        temperature=float(s.get("temperature", 0.2)),
        max_tokens=int(s.get("max_tokens", 1024)),
        settings=s,
    )


# Recommended models for the platform fallback when DEFAULT_MODEL belongs to another provider.
PLATFORM_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-5.4-mini"}
_MODEL_PREFIXES = {"anthropic": ("claude",), "openai": ("gpt", "o1", "o3", "o4", "chatgpt")}


def platform_fallback(org: Organization) -> ResolvedLLM | None:
    """The deployment-wide model for organisations without their own provider (or ``None``)."""
    s = get_settings()
    if not s.platform_llm_fallback:
        return None
    keys = {"anthropic": s.anthropic_api_key, "openai": s.openai_api_key}
    for kind in [s.default_provider, *(k for k in keys if k != s.default_provider)]:
        key = keys.get(kind)
        if not key or not _allowed(org, kind):
            continue
        model = s.default_model if s.default_model.startswith(_MODEL_PREFIXES[kind]) else PLATFORM_MODELS[kind]
        extra = {"workspace_id": s.anthropic_workspace_id} if kind == "anthropic" and s.anthropic_workspace_id else {}
        provider = instantiate(kind, model=model, api_key=key, base_url=None, temperature=s.default_temperature, max_tokens=s.default_max_tokens, settings=extra)
        return ResolvedLLM(provider, None, kind)
    return None


def _month_start() -> datetime:
    now = datetime.now(UTC)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def resolve_llm(
    db: Session,
    org: Organization,
    *,
    provider_id: uuid.UUID | None = None,
    model: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> ResolvedLLM | None:
    """The organisation's chat model, or ``None`` (→ extractive answers)."""
    q = select(AIProviderConfig).where(AIProviderConfig.org_id == org.id, AIProviderConfig.enabled.is_(True))
    cfg: AIProviderConfig | None
    if provider_id:
        cfg = db.scalar(q.where(AIProviderConfig.id == provider_id))
        if cfg is None:
            raise ConfigurationError("The selected AI provider is not available.")
    else:
        candidates = [c for c in db.scalars(q.order_by(AIProviderConfig.is_default.desc(), AIProviderConfig.created_at)) if c.kind not in EMBEDDING_ONLY_KINDS]
        cfg = candidates[0] if candidates else None
    if cfg is None:
        return platform_fallback(org)
    if not _allowed(org, cfg.kind):
        raise PermissionDenied(f"Your organization does not allow the '{cfg.kind}' provider.")
    s = cfg.settings or {}
    if s.get("rpm_limit"):
        enforce(f"llm:{cfg.id}", int(s["rpm_limit"]))
    if s.get("monthly_token_limit"):
        used = db.scalar(
            select(func.coalesce(func.sum(UsageRecord.tokens_in + UsageRecord.tokens_out), 0)).where(
                UsageRecord.org_id == org.id, UsageRecord.provider == cfg.name, UsageRecord.created_at >= _month_start()
            )
        ) or 0
        if used >= int(s["monthly_token_limit"]):
            raise LimitExceeded(f"Monthly token limit reached for provider '{cfg.name}'.")
    provider = _instantiate(db, cfg, model, overrides)
    fallback_id = s.get("fallback_provider_id")
    if fallback_id:
        fb = db.scalar(q.where(AIProviderConfig.id == uuid.UUID(str(fallback_id))))
        if fb is not None and fb.id != cfg.id and _allowed(org, fb.kind):
            provider = FallbackProvider(provider, _instantiate(db, fb, None))
    return ResolvedLLM(provider=provider, config=cfg, kind=cfg.kind)


def resolve_embedder(db: Session, org_id: uuid.UUID, config: dict[str, Any]) -> Embedder:
    """Embedder for a knowledge base's ``embedding_config``."""
    provider_id = config.get("provider_id")
    api_key = base_url = None
    extra: dict[str, Any] = {}
    if provider_id:
        cfg = db.get(AIProviderConfig, uuid.UUID(str(provider_id)))
        if cfg is None or cfg.org_id != org_id or not cfg.enabled:
            raise ConfigurationError("The embedding provider configured for this knowledge base is unavailable.")
        api_key = credential_value(db, cfg.credential_id)
        base_url = cfg.base_url
        extra = cfg.settings or {}
    elif config.get("provider") not in (None, "hashing", "local", "ollama"):
        raise ConfigurationError(f"Embedding provider '{config.get('provider')}' needs a configured AI provider with credentials.")
    if config.get("provider") == "ollama" and not base_url:
        base_url = get_settings().ollama_base_url
    return build_embedder(config, api_key=api_key, base_url=base_url, settings=extra)
