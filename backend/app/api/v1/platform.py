"""API keys, analytics, notifications, global search, webhooks, billing, onboarding, support access."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import Date, cast, distinct, func, or_, select

from app.api.deps import OrgContext, get_org_context, requires
from app.api.schemas import Ok
from app.core.netguard import check_url
from app.core.security import decrypt_secret, encrypt_secret, generate_api_key, new_token
from app.jobs.queue import JobContext, PermanentJobError, enqueue, handler
from app.models import (
    ApiKey,
    Conversation,
    Credential,
    DatabaseConnection,
    Document,
    Integration,
    KnowledgeBase,
    Message,
    MessageFeedback,
    Notification,
    SqlQueryLog,
    SupportAccessGrant,
    Team,
    UsageRecord,
    User,
    Webhook,
    WebhookDelivery,
    Workspace,
)
from app.rbac.catalog import ALL_PERMISSIONS
from app.services import billing
from app.utils.errors import NotFound, PermissionDenied, ValidationFailed

router = APIRouter(tags=["platform"])


def _iso(v: datetime | None) -> str | None:
    return v.isoformat() if v else None


# ----------------------------------------------------------------------------- API keys


class ApiKeyIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    scopes: list[str] = Field(min_length=1)
    expires_in_days: int | None = Field(default=90, ge=1, le=730)
    rate_limit_per_minute: int | None = Field(default=None, ge=1, le=10000)


def _key_out(k: ApiKey) -> dict[str, Any]:
    return {"id": str(k.id), "name": k.name, "prefix": k.prefix, "scopes": k.scopes, "created_by": str(k.created_by), "rate_limit_per_minute": k.rate_limit_per_minute, "expires_at": _iso(k.expires_at), "revoked_at": _iso(k.revoked_at), "last_used_at": _iso(k.last_used_at), "created_at": _iso(k.created_at), "active": k.revoked_at is None and (k.expires_at is None or k.expires_at > datetime.now(UTC))}


@router.get("/api-keys")
def list_keys(ctx: OrgContext = Depends(requires("admin.api_keys"))) -> list[dict[str, Any]]:
    stmt = select(ApiKey).where(ApiKey.org_id == ctx.org_id)
    if not ctx.authz.has("admin.audit"):
        stmt = stmt.where(ApiKey.created_by == ctx.user_id)
    return [_key_out(k) for k in ctx.db.scalars(stmt.order_by(ApiKey.created_at.desc()))]


@router.post("/api-keys", status_code=201)
def create_key(body: ApiKeyIn, ctx: OrgContext = Depends(requires("admin.api_keys"))) -> dict[str, Any]:
    if ctx.principal.api_key_id:
        raise PermissionDenied("API keys cannot create API keys.")
    unknown = set(body.scopes) - ALL_PERMISSIONS
    if unknown:
        raise ValidationFailed(f"Unknown scopes: {sorted(unknown)}")
    ctx.authz.load()
    if not set(body.scopes) <= ctx.authz.org_perms:
        raise PermissionDenied("Keys cannot carry permissions you do not hold.", details={"missing": sorted(set(body.scopes) - ctx.authz.org_perms)})
    full, prefix, digest = generate_api_key()
    k = ApiKey(org_id=ctx.org_id, name=body.name, prefix=prefix, key_hash=digest, scopes=sorted(set(body.scopes)), created_by=ctx.user_id, rate_limit_per_minute=body.rate_limit_per_minute,
               expires_at=datetime.now(UTC) + timedelta(days=body.expires_in_days) if body.expires_in_days else None)
    ctx.db.add(k)
    ctx.db.flush()
    ctx.audit("api_key.create", resource_type="api_key", resource_id=k.id, meta={"name": k.name, "scopes": k.scopes})
    ctx.db.commit()
    return {**_key_out(k), "key": full, "notice": "Copy this key now. It will not be shown again."}


@router.delete("/api-keys/{key_id}", response_model=Ok)
def revoke_key(key_id: uuid.UUID, ctx: OrgContext = Depends(requires("admin.api_keys"))) -> Ok:
    k = ctx.db.get(ApiKey, key_id)
    if k is None or k.org_id != ctx.org_id:
        raise NotFound("API key not found.")
    if k.created_by != ctx.user_id and not ctx.authz.has("org.security"):
        raise PermissionDenied("Only the creator or a security administrator can revoke this key.")
    k.revoked_at = datetime.now(UTC)
    ctx.audit("api_key.revoke", resource_type="api_key", resource_id=k.id)
    ctx.db.commit()
    return Ok()


# ---------------------------------------------------------------------------- analytics


@router.get("/analytics/overview")
def analytics(days: int = 30, ctx: OrgContext = Depends(requires("analytics.view"))) -> dict[str, Any]:
    db, org = ctx.db, ctx.org_id
    since = datetime.now(UTC) - timedelta(days=min(max(days, 1), 365))
    day = cast(Message.created_at, Date)
    q_series = dict(db.execute(select(day, func.count()).where(Message.org_id == org, Message.role == "user", Message.created_at >= since).group_by(day)).all())
    u_day = cast(UsageRecord.created_at, Date)
    tokens = {d: (int(i or 0), int(o or 0), float(c or 0)) for d, i, o, c in db.execute(select(u_day, func.sum(UsageRecord.tokens_in), func.sum(UsageRecord.tokens_out), func.sum(UsageRecord.cost_usd)).where(UsageRecord.org_id == org, UsageRecord.created_at >= since).group_by(u_day))}
    users = dict(db.execute(select(u_day, func.count(distinct(UsageRecord.user_id))).where(UsageRecord.org_id == org, UsageRecord.created_at >= since, UsageRecord.kind == "chat").group_by(u_day)).all())
    d_day = cast(Document.created_at, Date)
    docs = dict(db.execute(select(d_day, func.count()).where(Document.org_id == org, Document.created_at >= since).group_by(d_day)).all())
    series = []
    for i in range(min(days, 365), -1, -1):
        d = (datetime.now(UTC) - timedelta(days=i)).date()
        t = tokens.get(d, (0, 0, 0.0))
        series.append({"date": d.isoformat(), "questions": q_series.get(d, 0), "active_users": users.get(d, 0), "tokens_in": t[0], "tokens_out": t[1], "cost_usd": round(t[2], 4), "documents": docs.get(d, 0)})
    by_model = [{"provider": p, "model": m, "tokens": int(t or 0), "requests": int(n), "cost_usd": round(float(c or 0), 4)} for p, m, t, n, c in db.execute(select(UsageRecord.provider, UsageRecord.model, func.sum(UsageRecord.tokens_in + UsageRecord.tokens_out), func.count(), func.sum(UsageRecord.cost_usd)).where(UsageRecord.org_id == org, UsageRecord.created_at >= since, UsageRecord.kind.in_(("chat", "agent", "sql", "embedding"))).group_by(UsageRecord.provider, UsageRecord.model))]
    kb_names = dict(db.execute(select(KnowledgeBase.id, KnowledgeBase.name).where(KnowledgeBase.org_id == org)).all())
    readable = set(ctx.authz.access_scope().readable_kb_ids())
    by_kb = [{"kb_id": str(k), "name": kb_names.get(k, "deleted"), "questions": int(n)} for k, n in db.execute(select(UsageRecord.kb_id, func.count()).where(UsageRecord.org_id == org, UsageRecord.kind == "chat", UsageRecord.kb_id.is_not(None), UsageRecord.created_at >= since).group_by(UsageRecord.kb_id)) if k in readable]
    lat = db.execute(select(func.avg(Message.retrieval_ms), func.avg(Message.generation_ms), func.percentile_cont(0.95).within_group(Message.retrieval_ms)).where(Message.org_id == org, Message.role == "assistant", Message.created_at >= since)).one()
    modes = dict(db.execute(select(Message.status, func.count()).where(Message.org_id == org, Message.role == "assistant", Message.created_at >= since).group_by(Message.status)).all())
    feedback = dict(db.execute(select(MessageFeedback.rating, func.count()).where(MessageFeedback.org_id == org, MessageFeedback.created_at >= since).group_by(MessageFeedback.rating)).all())
    doc_status = dict(db.execute(select(Document.status, func.count()).where(Document.org_id == org).group_by(Document.status)).all())
    sql = dict(db.execute(select(SqlQueryLog.status, func.count()).where(SqlQueryLog.org_id == org, SqlQueryLog.created_at >= since).group_by(SqlQueryLog.status)).all())
    api_calls = db.scalar(select(func.count()).select_from(UsageRecord).where(UsageRecord.org_id == org, UsageRecord.api_key_id.is_not(None), UsageRecord.created_at >= since)) or 0
    conversations = db.scalar(select(func.count()).select_from(Conversation).where(Conversation.org_id == org, Conversation.created_at >= since)) or 0
    return {
        "range_days": days, "series": series, "by_model": by_model, "by_kb": sorted(by_kb, key=lambda x: -x["questions"]),
        "latency": {"retrieval_avg_ms": round(float(lat[0] or 0), 1), "generation_avg_ms": round(float(lat[1] or 0), 1), "retrieval_p95_ms": round(float(lat[2] or 0), 1)},
        "answers": {"complete": modes.get("complete", 0), "refused": modes.get("refused", 0), "stopped": modes.get("stopped", 0), "error": modes.get("error", 0)},
        "feedback": {"positive": feedback.get(1, 0), "negative": feedback.get(-1, 0)}, "documents": doc_status, "sql_queries": sql,
        "api_calls": api_calls, "conversations": conversations, "usage": billing.current_usage(db, org), "limits": billing.limits_for(db, org),
    }


# ------------------------------------------------------------------------ notifications


@router.get("/notifications")
def notifications(unread: bool = False, ctx: OrgContext = Depends(get_org_context)) -> dict[str, Any]:
    stmt = select(Notification).where(Notification.org_id == ctx.org_id, Notification.user_id == ctx.user_id)
    if unread:
        stmt = stmt.where(Notification.read_at.is_(None))
    rows = list(ctx.db.scalars(stmt.order_by(Notification.created_at.desc()).limit(100)))
    count = ctx.db.scalar(select(func.count()).select_from(Notification).where(Notification.org_id == ctx.org_id, Notification.user_id == ctx.user_id, Notification.read_at.is_(None))) or 0
    return {"unread": count, "items": [{"id": str(n.id), "kind": n.kind, "title": n.title, "body": n.body, "link": n.link, "severity": n.severity, "read": n.read_at is not None, "created_at": _iso(n.created_at)} for n in rows]}


@router.post("/notifications/read", response_model=Ok)
def mark_read(ids: list[uuid.UUID] | None = None, ctx: OrgContext = Depends(get_org_context)) -> Ok:
    stmt = select(Notification).where(Notification.org_id == ctx.org_id, Notification.user_id == ctx.user_id, Notification.read_at.is_(None))
    if ids:
        stmt = stmt.where(Notification.id.in_(ids))
    for n in ctx.db.scalars(stmt):
        n.read_at = datetime.now(UTC)
    ctx.db.commit()
    return Ok()


# ------------------------------------------------------------------------------- search


@router.get("/search")
def global_search(q: str, types: str = "documents,knowledge_bases,conversations,teams,workspaces,connections", mode: str = "keyword", limit: int = 8, ctx: OrgContext = Depends(get_org_context)) -> dict[str, Any]:
    q = q.strip()[:200]
    if len(q) < 2:
        return {"query": q, "results": {}}
    wanted = set(types.split(","))
    out: dict[str, list[dict[str, Any]]] = {}
    like = f"%{q}%"
    db, authz = ctx.db, ctx.authz
    if "knowledge_bases" in wanted and authz.has("kb.view"):
        kbs = list(db.scalars(select(KnowledgeBase).where(KnowledgeBase.org_id == ctx.org_id, or_(KnowledgeBase.name.ilike(like), KnowledgeBase.description.ilike(like))).limit(50)))
        levels = authz.kb_levels(kbs)
        out["knowledge_bases"] = [{"id": str(k.id), "title": k.name, "snippet": k.description[:200], "link": f"/knowledge-bases/{k.id}"} for k in kbs if levels[k.id] >= 1][:limit]
    if "documents" in wanted and authz.has("kb.view"):
        from app.rag.engine import embedding_spaces
        from app.rag.retrieval import retrieve

        scope = authz.access_scope()
        kbs = list(db.scalars(select(KnowledgeBase).where(KnowledgeBase.id.in_(scope.readable_kb_ids() or [uuid.UUID(int=0)]))))
        docs: list[dict[str, Any]] = []
        if kbs:
            res = retrieve(db, scope=scope, spaces=embedding_spaces(db, ctx.org_id, kbs), query=q, config={"mode": "hybrid" if mode == "semantic" else "keyword", "top_k": limit * 2, "mmr": False}, user_id=ctx.user_id)
            seen: set[uuid.UUID] = set()
            for it in res.items:
                if it.document_id in seen:
                    continue
                seen.add(it.document_id)
                docs.append({"id": str(it.document_id), "title": it.title, "snippet": it.content[:240], "relevance": round(it.score, 3), "locator": it.meta.get("page") or it.meta.get("sheet"), "link": f"/documents?doc={it.document_id}"})
        names = [d for d in db.scalars(select(Document).where(Document.kb_id.in_(scope.readable_kb_ids() or [uuid.UUID(int=0)]), or_(Document.title.ilike(like), Document.filename.ilike(like))).limit(limit))]
        for d in names:
            if str(d.id) not in {x["id"] for x in docs} and d.id not in scope.denied_document_ids and (d.acl_mode == "inherit" or d.id in scope.restricted_allowed_document_ids or d.kb_id in scope.kb_manage_ids):
                docs.append({"id": str(d.id), "title": d.title, "snippet": d.filename, "relevance": None, "link": f"/documents?doc={d.id}"})
        out["documents"] = docs[:limit]
    if "conversations" in wanted and authz.has("conv.view_own"):
        convs = db.scalars(select(Conversation).where(Conversation.org_id == ctx.org_id, Conversation.user_id == ctx.user_id, or_(Conversation.title.ilike(like), Conversation.id.in_(select(Message.conversation_id).where(Message.org_id == ctx.org_id, Message.content.ilike(like))))).order_by(Conversation.updated_at.desc()).limit(limit))
        out["conversations"] = [{"id": str(c.id), "title": c.title, "link": f"/chat?c={c.id}"} for c in convs]
    if "teams" in wanted:
        out["teams"] = [{"id": str(t.id), "title": t.name, "link": "/team"} for t in db.scalars(select(Team).where(Team.org_id == ctx.org_id, Team.name.ilike(like)).limit(limit))]
    if "workspaces" in wanted and authz.has("workspace.view"):
        out["workspaces"] = [{"id": str(w.id), "title": w.name, "link": "/team?tab=workspaces"} for w in db.scalars(select(Workspace).where(Workspace.org_id == ctx.org_id, Workspace.name.ilike(like)).limit(limit))]
    if "connections" in wanted:
        items = []
        if authz.has("db.view"):
            items += [{"id": str(c.id), "title": c.name, "snippet": c.provider, "link": "/database-connections"} for c in db.scalars(select(DatabaseConnection).where(DatabaseConnection.org_id == ctx.org_id, DatabaseConnection.name.ilike(like)).limit(limit)) if authz.connection_level(c) >= 1]
        if authz.has("integrations.view"):
            items += [{"id": str(i.id), "title": i.name, "snippet": i.provider, "link": "/integrations"} for i in db.scalars(select(Integration).where(Integration.org_id == ctx.org_id, Integration.name.ilike(like)).limit(limit))]
        out["connections"] = items[:limit]
    return {"query": q, "results": out}


# ----------------------------------------------------------------------------- webhooks

from app.services.activity import WEBHOOK_EVENTS  # noqa: E402


class WebhookIn(BaseModel):
    url: str = Field(max_length=2000)
    description: str = ""
    events: list[str] = Field(min_length=1)
    enabled: bool = True


@router.get("/webhooks")
def list_webhooks(ctx: OrgContext = Depends(requires("admin.platform_settings"))) -> dict[str, Any]:
    hooks = list(ctx.db.scalars(select(Webhook).where(Webhook.org_id == ctx.org_id)))
    return {"available_events": WEBHOOK_EVENTS, "items": [{"id": str(h.id), "url": h.url, "description": h.description, "events": h.events, "enabled": h.enabled, "failure_count": h.failure_count, "last_delivery_at": _iso(h.last_delivery_at), "created_at": _iso(h.created_at)} for h in hooks]}


@router.post("/webhooks", status_code=201)
def create_webhook(body: WebhookIn, ctx: OrgContext = Depends(requires("admin.platform_settings"))) -> dict[str, Any]:
    check_url(body.url, purpose="webhook", schemes=("https", "http"))
    unknown = set(body.events) - set(WEBHOOK_EVENTS)
    if unknown:
        raise ValidationFailed(f"Unknown events: {sorted(unknown)}")
    secret = "whsec_" + new_token(24)
    cred = Credential(org_id=ctx.org_id, kind="webhook_secret", ciphertext=encrypt_secret(secret), created_by=ctx.user_id)
    ctx.db.add(cred)
    ctx.db.flush()
    h = Webhook(org_id=ctx.org_id, url=body.url, description=body.description, events=body.events, enabled=body.enabled, credential_id=cred.id, created_by=ctx.user_id)
    ctx.db.add(h)
    ctx.db.flush()
    ctx.audit("webhook.create", resource_type="webhook", resource_id=h.id, meta={"url": body.url, "events": body.events})
    ctx.db.commit()
    return {"id": str(h.id), "secret": secret, "notice": "Store this signing secret now; it is not shown again. Verify X-RAG-Signature = sha256 HMAC of '<timestamp>.<body>'."}


class WebhookUpdate(BaseModel):
    url: str | None = Field(default=None, max_length=2000)
    description: str | None = None
    events: list[str] | None = None
    enabled: bool | None = None


@router.patch("/webhooks/{webhook_id}", response_model=Ok)
def update_webhook(webhook_id: uuid.UUID, body: WebhookUpdate, ctx: OrgContext = Depends(requires("admin.platform_settings"))) -> Ok:
    h = ctx.db.get(Webhook, webhook_id)
    if h is None or h.org_id != ctx.org_id:
        raise NotFound("Webhook not found.")
    if body.url is not None:
        check_url(body.url, purpose="webhook", schemes=("https", "http"))
        h.url = body.url
    if body.events is not None:
        unknown = set(body.events) - set(WEBHOOK_EVENTS)
        if unknown or not body.events:
            raise ValidationFailed(f"Invalid events: {sorted(unknown) or 'none selected'}")
        h.events = body.events
    if body.description is not None:
        h.description = body.description
    if body.enabled is not None:
        h.enabled = body.enabled
        if body.enabled:
            h.failure_count = 0  # re-enable after automatic disabling
    ctx.audit("webhook.update", resource_type="webhook", resource_id=h.id, meta=body.model_dump(exclude_none=True))
    ctx.db.commit()
    return Ok()


@router.post("/webhooks/{webhook_id}/test", response_model=Ok)
def test_webhook(webhook_id: uuid.UUID, ctx: OrgContext = Depends(requires("admin.platform_settings"))) -> Ok:
    h = ctx.db.get(Webhook, webhook_id)
    if h is None or h.org_id != ctx.org_id:
        raise NotFound("Webhook not found.")
    d = WebhookDelivery(org_id=ctx.org_id, webhook_id=h.id, event_type="webhook.test", payload={"message": "Test delivery", "webhook_id": str(h.id)})
    ctx.db.add(d)
    ctx.db.flush()
    enqueue(ctx.db, ctx.org_id, "deliver_webhook", payload={"delivery_id": str(d.id)}, resource_type="webhook", resource_id=h.id, max_attempts=1)
    ctx.db.commit()
    return Ok(detail="Test delivery queued.")


@router.delete("/webhooks/{webhook_id}", response_model=Ok)
def delete_webhook(webhook_id: uuid.UUID, ctx: OrgContext = Depends(requires("admin.platform_settings"))) -> Ok:
    h = ctx.db.get(Webhook, webhook_id)
    if h is None or h.org_id != ctx.org_id:
        raise NotFound("Webhook not found.")
    ctx.db.delete(h)
    ctx.audit("webhook.delete", resource_type="webhook", resource_id=webhook_id)
    ctx.db.commit()
    return Ok()


@router.get("/webhooks/{webhook_id}/deliveries")
def webhook_deliveries(webhook_id: uuid.UUID, ctx: OrgContext = Depends(requires("admin.platform_settings"))) -> list[dict[str, Any]]:
    rows = ctx.db.scalars(select(WebhookDelivery).where(WebhookDelivery.org_id == ctx.org_id, WebhookDelivery.webhook_id == webhook_id).order_by(WebhookDelivery.created_at.desc()).limit(100))
    return [{"id": str(d.id), "event_type": d.event_type, "status": d.status, "attempts": d.attempts, "response_status": d.response_status, "error": d.error, "created_at": _iso(d.created_at), "delivered_at": _iso(d.delivered_at)} for d in rows]


def sign(secret: str, timestamp: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()


@handler("deliver_webhook")
def deliver_webhook(ctx: JobContext) -> dict[str, Any]:
    delivery_id = uuid.UUID(ctx.payload["delivery_id"])
    with ctx.session() as db:
        d = db.get(WebhookDelivery, delivery_id)
        h = db.get(Webhook, d.webhook_id) if d else None
        if d is None or h is None or not h.enabled:
            return {"skipped": True}
        cred = db.get(Credential, h.credential_id) if h.credential_id else None
        secret = decrypt_secret(cred.ciphertext) if cred else ""
        url, payload, event_type = h.url, d.payload, d.event_type
    body = json.dumps({"id": str(delivery_id), "type": event_type, "data": payload}, default=str).encode()
    ts = str(int(time.time()))
    try:
        check_url(url, purpose="webhook", schemes=("https", "http"))
        r = httpx.post(url, content=body, timeout=10, headers={"Content-Type": "application/json", "X-RAG-Event": event_type, "X-RAG-Delivery": str(delivery_id), "X-RAG-Timestamp": ts, "X-RAG-Signature": sign(secret, ts, body)})
        status, error = r.status_code, None if r.status_code < 400 else f"HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        status, error = None, type(exc).__name__ + ": " + str(exc)[:200]
    with ctx.session() as db:
        d = db.get(WebhookDelivery, delivery_id)
        h = db.get(Webhook, d.webhook_id)
        d.attempts += 1
        d.response_status, d.error = status, error
        if error is None:
            d.status, d.delivered_at = "delivered", datetime.now(UTC)
            h.failure_count, h.last_delivery_at = 0, datetime.now(UTC)
        else:
            d.status = "failed"
            h.failure_count += 1
            if h.failure_count >= 50:
                h.enabled = False
    if error and status is not None and 400 <= status < 500 and status not in (408, 429):
        raise PermanentJobError(error)
    if error:
        raise RuntimeError(error)
    return {"status": status}


# ------------------------------------------------------------------------------ billing


@router.get("/billing")
def billing_status(ctx: OrgContext = Depends(requires("org.view"))) -> dict[str, Any]:
    from app.config import get_settings
    from app.models import Plan

    plan, sub = billing.plan_for(ctx.db, ctx.org_id)
    plans = [{"key": p.key, "name": p.name, "limits": p.limits, "features": p.features, "price_monthly_usd": float(p.price_monthly_usd) if p.price_monthly_usd is not None else None} for p in ctx.db.scalars(select(Plan).where(Plan.is_public.is_(True)).order_by(Plan.sort))]
    return {"billing_enabled": get_settings().billing_enabled, "plan": plan.key if plan else None, "plan_name": plan.name if plan else None, "status": sub.status if sub else None, "provider": sub.provider if sub else None, "limits": billing.limits_for(ctx.db, ctx.org_id), "usage": billing.current_usage(ctx.db, ctx.org_id), "plans": plans}


# ----------------------------------------------------------------------------- onboarding

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


@router.get("/onboarding/status")
def onboarding_status(ctx: OrgContext = Depends(get_org_context)) -> dict[str, Any]:
    db = ctx.db
    from app.models import AIProviderConfig, Membership

    steps = {
        "organization": True,
        "knowledge_base": bool(db.scalar(select(KnowledgeBase.id).where(KnowledgeBase.org_id == ctx.org_id).limit(1))),
        "document": bool(db.scalar(select(Document.id).where(Document.org_id == ctx.org_id, Document.status == "ready").limit(1))),
        "ai_provider": bool(db.scalar(select(AIProviderConfig.id).where(AIProviderConfig.org_id == ctx.org_id).limit(1))),
        "teammate": (db.scalar(select(func.count()).select_from(Membership).where(Membership.org_id == ctx.org_id)) or 0) > 1,
        "question": bool(db.scalar(select(Message.id).where(Message.org_id == ctx.org_id, Message.role == "user").limit(1))),
        "data_source": bool(db.scalar(select(DatabaseConnection.id).where(DatabaseConnection.org_id == ctx.org_id).limit(1)) or db.scalar(select(Integration.id).where(Integration.org_id == ctx.org_id).limit(1))),
    }
    return {"steps": steps, "complete": all(steps.values())}


@router.post("/onboarding/sample-kb", status_code=201)
def sample_kb(ctx: OrgContext = Depends(requires("kb.create", "kb.upload"))) -> dict[str, Any]:
    """Create a sample knowledge base from bundled, clearly-labelled demo documents."""
    from app.services import knowledge_service as ks

    existing = ctx.db.scalar(select(KnowledgeBase).where(KnowledgeBase.org_id == ctx.org_id, KnowledgeBase.is_sample.is_(True)))
    if existing:
        return {"id": str(existing.id), "created": False}
    kb = ks.create_kb(ctx, {"name": "Sample · Northwind Handbook", "description": "Demo content shipped with the platform (fictional company). Safe to delete.", "visibility": "org", "tags": ["sample"], "is_sample": True})
    for path in sorted(SAMPLES.glob("*")):
        if path.is_file():
            with path.open("rb") as fh:
                up = ks.spool(fh, path.name)
            try:
                ks.add_document(ctx, kb, up, tags=["sample"], source_type="sample")
            finally:
                Path(up.path).unlink(missing_ok=True)
    ctx.db.commit()
    return {"id": str(kb.id), "created": True}


# ------------------------------------------------------------------------- support access


class SupportGrantIn(BaseModel):
    platform_admin_email: EmailStr
    hours: int = Field(default=4, ge=1, le=72)
    reason: str = Field(min_length=5, max_length=1000)


@router.get("/support-access")
def list_support_access(ctx: OrgContext = Depends(requires("org.security"))) -> list[dict[str, Any]]:
    rows = list(ctx.db.scalars(select(SupportAccessGrant).where(SupportAccessGrant.org_id == ctx.org_id).order_by(SupportAccessGrant.created_at.desc())))
    emails = dict(ctx.db.execute(select(User.id, User.email).where(User.id.in_({g.platform_user_id for g in rows} | {g.granted_by for g in rows}))).all()) if rows else {}
    return [{"id": str(g.id), "platform_user_id": str(g.platform_user_id), "platform_admin_email": emails.get(g.platform_user_id), "granted_by": str(g.granted_by), "granted_by_email": emails.get(g.granted_by), "reason": g.reason, "expires_at": _iso(g.expires_at), "revoked_at": _iso(g.revoked_at), "created_at": _iso(g.created_at)} for g in rows]


@router.post("/support-access", status_code=201)
def grant_support_access(body: SupportGrantIn, ctx: OrgContext = Depends(requires("org.security", "org.billing"))) -> dict[str, Any]:
    admin = ctx.db.scalar(select(User).where(User.email == body.platform_admin_email.lower(), User.is_platform_admin.is_(True)))
    if admin is None:
        raise ValidationFailed("No platform administrator with this email.")
    g = SupportAccessGrant(org_id=ctx.org_id, platform_user_id=admin.id, granted_by=ctx.user_id, reason=body.reason, expires_at=datetime.now(UTC) + timedelta(hours=body.hours))
    ctx.db.add(g)
    ctx.audit("support.grant", resource_type="support_access", resource_id=None, meta={"admin": admin.email, "hours": body.hours, "reason": body.reason})
    ctx.db.commit()
    return {"id": str(g.id), "expires_at": _iso(g.expires_at)}


@router.delete("/support-access/{grant_id}", response_model=Ok)
def revoke_support_access(grant_id: uuid.UUID, ctx: OrgContext = Depends(requires("org.security"))) -> Ok:
    g = ctx.db.get(SupportAccessGrant, grant_id)
    if g is None or g.org_id != ctx.org_id:
        raise NotFound("Grant not found.")
    g.revoked_at = datetime.now(UTC)
    ctx.audit("support.revoke", resource_type="support_access", resource_id=grant_id)
    ctx.db.commit()
    return Ok()
