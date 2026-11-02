"""Live model discovery.

Each configured provider is asked which models it offers right now (OpenAI and
OpenAI-compatible servers: ``/v1/models``; Anthropic: the Models API; Gemini:
``models.list``; Ollama: ``/api/tags``). Results are cached briefly, stored on
the provider so every model picker suggests current ids, and refreshed daily by
the scheduler, so new models appear without a platform release. Providers that
don't publish a list (Azure deployments, Bedrock) keep the built-in suggestions
and accept any typed id.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import system_session
from app.models import AIProviderConfig
from app.utils.errors import CSVAIError, NotAvailable
from app.utils.logging import get_logger

log = get_logger("llm.catalog")

CACHE_TTL = timedelta(minutes=10)
REFRESH_EVERY = timedelta(hours=24)
MAX_STORED = 500

_cache: dict[tuple[uuid.UUID, str, str], tuple[datetime, list[str]]] = {}
_last_refresh: datetime | None = None


def _key(cfg: AIProviderConfig) -> tuple[uuid.UUID, str, str]:
    return (cfg.id, cfg.base_url or "", str(cfg.credential_id or ""))


def discover(db: Session, cfg: AIProviderConfig, *, force: bool = False) -> list[str]:
    """Model ids the provider offers now. Raises NotAvailable when it has no list endpoint."""
    from app.llm.registry import _instantiate

    now = datetime.now(UTC)
    hit = _cache.get(_key(cfg))
    if hit and not force and now - hit[0] < CACHE_TTL:
        return hit[1]
    models = _instantiate(db, cfg, cfg.default_model or "unused").list_models()
    _cache[_key(cfg)] = (now, models)
    return models


def store(cfg: AIProviderConfig, models: list[str]) -> None:
    """Keep the discovered ids on the provider (suggestions for every model picker)."""
    cfg.models = [{"id": m} for m in models[:MAX_STORED]]
    cfg.settings = {**(cfg.settings or {}), "models_refreshed_at": datetime.now(UTC).isoformat()}


def refresh_all(force: bool = False) -> dict[str, int]:
    """Daily refresh of every enabled provider's model list (called by the scheduler)."""
    global _last_refresh
    now = datetime.now(UTC)
    if not force and _last_refresh and now - _last_refresh < REFRESH_EVERY:
        return {}
    _last_refresh = now
    totals = {"refreshed": 0, "skipped": 0, "failed": 0}
    with system_session() as db:
        for cfg in db.scalars(select(AIProviderConfig).where(AIProviderConfig.enabled.is_(True))):
            try:
                store(cfg, discover(db, cfg, force=True))
                totals["refreshed"] += 1
            except NotAvailable:
                totals["skipped"] += 1
            except CSVAIError as exc:
                totals["failed"] += 1
                log.warning("model list refresh failed for provider %s: %s", cfg.id, exc.message)
        db.commit()
    return totals
