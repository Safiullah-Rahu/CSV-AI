"""AI provider configuration and agents."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select, update

from app.agents.runner import TOOLS, run_agent
from app.api.deps import OrgContext, requires
from app.api.schemas import Ok
from app.core.netguard import check_url
from app.core.security import encrypt_secret
from app.llm.factory import EMBEDDING_ONLY_KINDS, KEYLESS_KINDS, PROVIDER_LABELS, available_models
from app.llm.registry import _instantiate, resolve_llm
from app.models import Agent, AIProviderConfig, Credential, KnowledgeBase
from app.rbac.authorizer import MANAGE, READ
from app.services.activity import emit
from app.utils.errors import NotAvailable, NotFound, PermissionDenied, ValidationFailed

router = APIRouter(tags=["ai"])


class ProviderIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    kind: str
    base_url: str | None = Field(default=None, max_length=1000)
    api_key: str | None = Field(default=None, max_length=4096)
    models: list[dict[str, Any]] = []
    default_model: str | None = None
    default_embedding_model: str | None = None
    settings: dict[str, Any] = {}
    enabled: bool = True
    is_default: bool = False


class ProviderUpdate(BaseModel):
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    models: list[dict[str, Any]] | None = None
    default_model: str | None = None
    default_embedding_model: str | None = None
    settings: dict[str, Any] | None = None
    enabled: bool | None = None
    is_default: bool | None = None


_SETTING_KEYS = {"temperature", "max_tokens", "timeout_s", "max_retries", "rpm_limit", "monthly_token_limit", "context_window", "api_version", "region", "access_key_id", "fallback_provider_id", "think", "num_ctx", "repeat_penalty", "workspace_id", "no_stream_usage", "effort"}


def _provider_out(p: AIProviderConfig, can_manage: bool) -> dict[str, Any]:
    return {
        "id": str(p.id), "name": p.name, "kind": p.kind, "kind_label": PROVIDER_LABELS.get(p.kind, p.kind), "base_url": p.base_url if can_manage else None,
        "has_credentials": p.credential_id is not None, "models": p.models, "default_model": p.default_model, "default_embedding_model": p.default_embedding_model,
        "settings": {k: v for k, v in (p.settings or {}).items() if k in _SETTING_KEYS} if can_manage else {}, "enabled": p.enabled, "is_default": p.is_default,
        "status": p.status, "last_checked_at": p.last_checked_at.isoformat() if p.last_checked_at else None, "last_error": p.last_error if can_manage else None,
        # Live ids from the provider once discovered (app.llm.catalog), else built-in suggestions.
        "suggested_models": [m["id"] for m in p.models or [] if m.get("id")] or available_models(p.kind),
        "models_refreshed_at": (p.settings or {}).get("models_refreshed_at"),
    }


@router.get("/ai/provider-kinds")
def provider_kinds(ctx: OrgContext = Depends(requires("org.view"))) -> list[dict[str, Any]]:
    return [{"kind": k, "label": v, "keyless": k in KEYLESS_KINDS, "embedding_only": k in EMBEDDING_ONLY_KINDS, "suggested_models": available_models(k)} for k, v in PROVIDER_LABELS.items()]


@router.get("/ai/providers")
def list_providers(ctx: OrgContext = Depends(requires("org.view"))) -> list[dict[str, Any]]:
    manage = ctx.authz.has("admin.ai_providers")
    return [_provider_out(p, manage) for p in ctx.db.scalars(select(AIProviderConfig).where(AIProviderConfig.org_id == ctx.org_id).order_by(AIProviderConfig.is_default.desc(), AIProviderConfig.name))]


def _validate_url(kind: str, url: str | None) -> None:
    if url:
        check_url(url, purpose="ai_endpoint")
    elif kind in ("azure_openai", "vllm", "openai_compatible", "custom"):
        raise ValidationFailed(f"{PROVIDER_LABELS[kind]} needs an endpoint URL.")


def _set_key(ctx: OrgContext, p: AIProviderConfig, key: str | None) -> None:
    if not key:
        return
    if p.credential_id:
        cred = ctx.db.get(Credential, p.credential_id)
        if cred:
            cred.ciphertext, cred.rotated_at = encrypt_secret(key), datetime.now(UTC)
            return
    cred = Credential(org_id=ctx.org_id, kind=f"ai:{p.kind}", ciphertext=encrypt_secret(key), created_by=ctx.user_id)
    ctx.db.add(cred)
    ctx.db.flush()
    p.credential_id = cred.id


@router.post("/ai/providers", status_code=201)
def create_provider(body: ProviderIn, ctx: OrgContext = Depends(requires("admin.ai_providers"))) -> dict[str, Any]:
    if body.kind not in PROVIDER_LABELS:
        raise ValidationFailed("Unknown provider kind.")
    if body.kind not in KEYLESS_KINDS and not body.api_key:
        raise ValidationFailed(f"{PROVIDER_LABELS[body.kind]} needs an API key.")
    _validate_url(body.kind, body.base_url)
    if body.is_default:
        ctx.db.execute(update(AIProviderConfig).where(AIProviderConfig.org_id == ctx.org_id).values(is_default=False))
    p = AIProviderConfig(org_id=ctx.org_id, name=body.name.strip(), kind=body.kind, base_url=body.base_url, models=body.models, default_model=body.default_model,
                         default_embedding_model=body.default_embedding_model, settings={k: v for k, v in body.settings.items() if k in _SETTING_KEYS}, enabled=body.enabled, is_default=body.is_default)
    ctx.db.add(p)
    ctx.db.flush()
    _set_key(ctx, p, body.api_key)
    ctx.audit("ai_provider.create", resource_type="ai_provider", resource_id=p.id, meta={"kind": p.kind, "name": p.name})
    emit(ctx.db, ctx.org_id, "ai_provider.created", resource_type="ai_provider", resource_id=p.id, actor_user_id=ctx.user_id, message=f"AI provider '{p.name}' added")
    ctx.db.commit()
    return _provider_out(p, True)


def _get_provider(ctx: OrgContext, pid: uuid.UUID) -> AIProviderConfig:
    p = ctx.db.get(AIProviderConfig, pid)
    if p is None or p.org_id != ctx.org_id:
        raise NotFound("Provider not found.")
    return p


@router.patch("/ai/providers/{provider_id}")
def update_provider(provider_id: uuid.UUID, body: ProviderUpdate, ctx: OrgContext = Depends(requires("admin.ai_providers"))) -> dict[str, Any]:
    p = _get_provider(ctx, provider_id)
    data = body.model_dump(exclude_unset=True)
    if "base_url" in data:
        _validate_url(p.kind, data["base_url"])
    if data.get("is_default"):
        ctx.db.execute(update(AIProviderConfig).where(AIProviderConfig.org_id == ctx.org_id).values(is_default=False))
    for key in ("name", "base_url", "models", "default_model", "default_embedding_model", "enabled", "is_default"):
        if key in data:
            setattr(p, key, data[key])
    if data.get("settings") is not None:
        merged = {**(p.settings or {}), **{k: v for k, v in data["settings"].items() if k in _SETTING_KEYS}}
        p.settings = {k: v for k, v in merged.items() if v is not None}
    _set_key(ctx, p, data.get("api_key"))
    ctx.audit("ai_provider.update", resource_type="ai_provider", resource_id=p.id, meta={k: v for k, v in data.items() if k != "api_key"})
    ctx.db.commit()
    return _provider_out(p, True)


@router.delete("/ai/providers/{provider_id}", response_model=Ok)
def delete_provider(provider_id: uuid.UUID, ctx: OrgContext = Depends(requires("admin.ai_providers"))) -> Ok:
    p = _get_provider(ctx, provider_id)
    if p.credential_id:
        cred = ctx.db.get(Credential, p.credential_id)
        if cred:
            ctx.db.delete(cred)
    ctx.db.delete(p)
    ctx.audit("ai_provider.delete", resource_type="ai_provider", resource_id=provider_id)
    ctx.db.commit()
    return Ok()


@router.post("/ai/providers/{provider_id}/health")
def provider_health(provider_id: uuid.UUID, ctx: OrgContext = Depends(requires("admin.ai_providers"))) -> dict[str, Any]:
    p = _get_provider(ctx, provider_id)
    if p.kind in EMBEDDING_ONLY_KINDS:
        from app.llm.registry import resolve_embedder

        try:
            emb = resolve_embedder(ctx.db, ctx.org_id, {"provider": p.kind, "model": p.default_embedding_model, "provider_id": str(p.id)})
            emb.embed_query("health check")
            ok, latency, err = True, 0.0, None
        except Exception as exc:  # noqa: BLE001
            ok, latency, err = False, 0.0, getattr(exc, "message", str(exc))[:300]
    else:
        ok, latency, err = _instantiate(ctx.db, p, None).health()
    p.status, p.last_checked_at, p.last_error = ("healthy" if ok else "failed"), datetime.now(UTC), err
    emit(ctx.db, ctx.org_id, "ai_provider.health", resource_type="ai_provider", resource_id=p.id, severity="success" if ok else "error", message=f"{p.name}: {'healthy' if ok else err}", payload={"latency_ms": round(latency, 1)})
    ctx.db.commit()
    return {"ok": ok, "latency_ms": round(latency, 1), "error": err}


@router.get("/ai/platform-default")
def platform_default(ctx: OrgContext = Depends(requires("org.view"))) -> dict[str, Any]:
    """Which model answers when this organisation has no enabled chat provider of its own."""
    from app.llm.registry import platform_fallback

    llm = platform_fallback(ctx.org)
    return {"available": llm is not None, "provider": llm.kind if llm else None, "model": llm.provider.model if llm else None}


@router.get("/ai/providers/{provider_id}/models")
def provider_models(provider_id: uuid.UUID, refresh: bool = False, ctx: OrgContext = Depends(requires("admin.ai_providers"))) -> list[str]:
    """Models the provider offers right now (live), stored as this provider's suggestions."""
    from app.llm import catalog

    p = _get_provider(ctx, provider_id)
    try:
        models = catalog.discover(ctx.db, p, force=refresh)
    except NotAvailable:
        return [m.get("id") for m in p.models or [] if m.get("id")] or available_models(p.kind)
    catalog.store(p, models)
    ctx.db.commit()
    return models


# ------------------------------------------------------------------------------ agents


class AgentIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    description: str = Field(default="", max_length=2000)
    instructions: str = Field(default="", max_length=8000)
    kb_ids: list[uuid.UUID] = []
    connection_ids: list[uuid.UUID] = []
    provider_id: uuid.UUID | None = None
    model: str | None = None
    tools: list[str] = ["kb_search"]
    approval_tools: list[str] = []
    retrieval_config: dict[str, Any] = {}
    limits: dict[str, Any] = {}
    enabled: bool = True
    workspace_id: uuid.UUID | None = None


def _agent_out(a: Agent, level: int) -> dict[str, Any]:
    return {"id": str(a.id), "name": a.name, "description": a.description, "instructions": a.instructions if level >= MANAGE else None, "kb_ids": [str(k) for k in a.kb_ids], "connection_ids": [str(c) for c in a.connection_ids], "provider_id": str(a.provider_id) if a.provider_id else None, "model": a.model, "tools": a.tools, "approval_tools": a.approval_tools, "retrieval_config": a.retrieval_config, "limits": a.limits, "enabled": a.enabled, "workspace_id": str(a.workspace_id) if a.workspace_id else None, "access": ["none", "read", "write", "manage"][level], "created_at": a.created_at.isoformat()}


@router.get("/agents/tools")
def agent_tools(ctx: OrgContext = Depends(requires("agents.view"))) -> list[dict[str, str]]:
    return [{"key": k, "description": v} for k, v in TOOLS.items()]


@router.get("/agents")
def list_agents(ctx: OrgContext = Depends(requires("agents.view"))) -> list[dict[str, Any]]:
    out = []
    for a in ctx.db.scalars(select(Agent).where(Agent.org_id == ctx.org_id).order_by(Agent.name)):
        lvl = ctx.authz.agent_level(a)
        if lvl >= READ:
            out.append(_agent_out(a, lvl))
    return out


def _validate_agent(ctx: OrgContext, body: AgentIn) -> None:
    unknown = set(body.tools) - set(TOOLS)
    if unknown:
        raise ValidationFailed(f"Unknown tools: {sorted(unknown)}")
    for kid in body.kb_ids:
        kb = ctx.db.get(KnowledgeBase, kid)
        if kb is None or kb.org_id != ctx.org_id or ctx.authz.kb_level(kb) < READ:
            raise ValidationFailed("You can only attach knowledge bases you can read.")
    if body.provider_id:
        _get_provider(ctx, body.provider_id)
    from app.models import DatabaseConnection

    for cid in body.connection_ids:
        conn = ctx.db.get(DatabaseConnection, cid)
        if conn is None or conn.org_id != ctx.org_id or ctx.authz.connection_level(conn) < READ:
            raise ValidationFailed("You can only attach database connections you can access.")


@router.post("/agents", status_code=201)
def create_agent(body: AgentIn, ctx: OrgContext = Depends(requires("agents.manage"))) -> dict[str, Any]:
    _validate_agent(ctx, body)
    a = Agent(org_id=ctx.org_id, created_by=ctx.user_id, **body.model_dump())
    ctx.db.add(a)
    ctx.db.flush()
    ctx.audit("agent.create", resource_type="agent", resource_id=a.id, meta={"name": a.name, "tools": a.tools})
    emit(ctx.db, ctx.org_id, "agent.created", resource_type="agent", resource_id=a.id, actor_user_id=ctx.user_id, message=f"Agent '{a.name}' created")
    ctx.db.commit()
    return _agent_out(a, MANAGE)


def _get_agent(ctx: OrgContext, agent_id: uuid.UUID, level: int = READ) -> Agent:
    a = ctx.db.get(Agent, agent_id)
    if a is None or a.org_id != ctx.org_id or ctx.authz.agent_level(a) < READ:
        raise NotFound("Agent not found.")
    if ctx.authz.agent_level(a) < level:
        raise PermissionDenied("You cannot modify this agent.")
    return a


@router.put("/agents/{agent_id}")
def update_agent(agent_id: uuid.UUID, body: AgentIn, ctx: OrgContext = Depends(requires("agents.manage"))) -> dict[str, Any]:
    a = _get_agent(ctx, agent_id, MANAGE)
    _validate_agent(ctx, body)
    for k, v in body.model_dump().items():
        setattr(a, k, v)
    ctx.audit("agent.update", resource_type="agent", resource_id=a.id)
    ctx.db.commit()
    return _agent_out(a, MANAGE)


@router.delete("/agents/{agent_id}", response_model=Ok)
def delete_agent(agent_id: uuid.UUID, ctx: OrgContext = Depends(requires("agents.manage"))) -> Ok:
    a = _get_agent(ctx, agent_id, MANAGE)
    ctx.db.delete(a)
    ctx.audit("agent.delete", resource_type="agent", resource_id=agent_id)
    ctx.db.commit()
    return Ok()


class RunIn(BaseModel):
    question: str = Field(min_length=1, max_length=8000)


@router.post("/agents/{agent_id}/run")
def run(agent_id: uuid.UUID, body: RunIn, ctx: OrgContext = Depends(requires("agents.use"))) -> dict[str, Any]:
    a = _get_agent(ctx, agent_id)
    if not a.enabled:
        raise ValidationFailed("This agent is disabled.")
    llm = resolve_llm(ctx.db, ctx.org, provider_id=a.provider_id, model=a.model)
    if llm is None:
        raise NotAvailable("Agents with tools need a configured language model. Use AI Chat with this agent's knowledge bases instead.")
    result = run_agent(ctx.db, ctx.authz, ctx.org, a, body.question, llm)
    ctx.audit("agent.run", resource_type="agent", resource_id=a.id, meta={"status": result.status, "steps": len(result.steps)})
    ctx.db.commit()
    from app.agents.runner import run_payload

    return run_payload(result)
