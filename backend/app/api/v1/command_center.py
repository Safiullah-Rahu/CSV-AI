"""Integrations, the Connections command center (graph, health, events) and SSE."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from app.api.deps import OrgContext, requires
from app.api.schemas import Ok, page
from app.connectors import oauth
from app.connectors.integrations import INTEGRATIONS
from app.db.session import tenant_session
from app.events.bus import get_bus
from app.models import ActivityEvent, DatabaseConnection, Integration, Job, KnowledgeBase, SyncRun
from app.rbac.authorizer import READ, WRITE, Authorizer
from app.services import graph_service
from app.services import integration_service as isvc
from app.services.activity import emit, event_to_dict
from app.utils.errors import NotAvailable, NotFound, ValidationFailed

router = APIRouter(tags=["connections"])


# ----------------------------------------------------------------------- integrations


class IntegrationIn(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    provider: str
    config: dict[str, Any] = {}
    secret: str | None = Field(default=None, max_length=4096)
    kb_id: uuid.UUID | None = None
    workspace_id: uuid.UUID | None = None
    sync_interval_minutes: int | None = Field(default=None, ge=15, le=60 * 24 * 30)


class IntegrationUpdate(BaseModel):
    name: str | None = None
    config: dict[str, Any] | None = None
    secret: str | None = None
    kb_id: uuid.UUID | None = None
    enabled: bool | None = None
    sync_interval_minutes: int | None = Field(default=None, ge=0, le=60 * 24 * 30)


def _int_out(i: Integration) -> dict[str, Any]:
    meta = INTEGRATIONS.get(i.provider, {})
    return {
        "id": str(i.id), "name": i.name, "provider": i.provider, "provider_label": meta.get("label", i.provider), "provider_status": meta.get("status"),
        "category": meta.get("category"), "config": {k: v for k, v in (i.config or {}).items() if "secret" not in k and "token" not in k and "password" not in k},
        "has_secret": i.credential_id is not None, "status": i.status, "enabled": i.enabled, "kb_id": str(i.kb_id) if i.kb_id else None,
        "sync_interval_minutes": i.sync_interval_minutes, "sync_paused": i.sync_paused, "next_sync_at": _iso(i.next_sync_at), "last_sync_at": _iso(i.last_sync_at),
        "last_success_at": _iso(i.last_success_at), "last_error": i.last_error, "created_at": _iso(i.created_at),
        "auth_mode": (i.config or {}).get("auth_mode") or (meta.get("auth_modes") or ["token"])[0], "oauth": isvc.oauth_state(i),
    }


def _iso(v: datetime | None) -> str | None:
    return v.isoformat() if v else None


@router.get("/integrations/catalog")
def integration_catalog(ctx: OrgContext = Depends(requires("integrations.view"))) -> list[dict[str, Any]]:
    """Providers with honest status; OAuth vendors report ``needs_setup`` (missing env var names)."""
    return isvc.catalog()


@router.get("/integrations")
def list_integrations(ctx: OrgContext = Depends(requires("integrations.view"))) -> list[dict[str, Any]]:
    return [_int_out(i) for i in ctx.db.scalars(select(Integration).where(Integration.org_id == ctx.org_id).order_by(Integration.name))]


def _check_kb(ctx: OrgContext, kb_id: uuid.UUID | None) -> None:
    if kb_id:
        kb = ctx.db.get(KnowledgeBase, kb_id)
        if kb is None or kb.org_id != ctx.org_id or ctx.authz.kb_level(kb) < WRITE:
            raise ValidationFailed("Target knowledge base not found or not writable.")


@router.post("/integrations", status_code=201)
def create_integration(body: IntegrationIn, ctx: OrgContext = Depends(requires("org.integrations"))) -> dict[str, Any]:
    from app.connectors.integrations import build
    from app.services.billing import check_limit

    meta = INTEGRATIONS.get(body.provider)
    if meta is None:
        raise ValidationFailed("Unknown integration provider.")
    if meta["status"] == "planned":
        raise NotAvailable(f"{meta['label']} is not available yet (needs {meta['auth']}).")
    config = isvc.normalise_config(body.provider, body.config, body.secret)
    if isvc.oauth_key(body.provider, config):
        isvc.require_oauth_app(body.provider)  # 501 naming the env vars an operator must set
        if body.secret:
            raise ValidationFailed(f"{meta['label']} tokens are obtained through “Connect with …”; do not send a secret.")
    check_limit(ctx.db, ctx.org_id, "connected_sources")
    _check_kb(ctx, body.kb_id)
    build(body.provider, config, body.secret).validate()
    i = Integration(org_id=ctx.org_id, name=body.name.strip(), provider=body.provider, config=config, kb_id=body.kb_id, workspace_id=body.workspace_id, owner_id=ctx.user_id, sync_interval_minutes=body.sync_interval_minutes)
    ctx.db.add(i)
    ctx.db.flush()
    isvc.set_secret(ctx, i, body.secret)
    isvc.schedule(i)
    ctx.audit("integration.create", resource_type="integration", resource_id=i.id, meta={"provider": i.provider, "name": i.name})
    emit(ctx.db, ctx.org_id, "connection.created", resource_type="integration", resource_id=i.id, actor_user_id=ctx.user_id, message=f"Integration '{i.name}' ({meta['label']}) added")
    ctx.db.commit()
    return _int_out(i)


@router.patch("/integrations/{integration_id}")
def update_integration(integration_id: uuid.UUID, body: IntegrationUpdate, ctx: OrgContext = Depends(requires("org.integrations"))) -> dict[str, Any]:
    i = isvc.get(ctx, integration_id)
    data = body.model_dump(exclude_unset=True)
    was_oauth = isvc.oauth_key(i.provider, i.config)
    if data.get("config") is not None:
        from app.connectors.integrations import build

        cfg = {k: v for k, v in data["config"].items() if v is not None}
        if not cfg.get("auth_mode") and (i.config or {}).get("auth_mode"):
            cfg["auth_mode"] = i.config["auth_mode"]  # a PATCH that omits the mode keeps it
        data["config"] = isvc.normalise_config(i.provider, cfg, data.get("secret"))
        build(i.provider, data["config"], data.get("secret")).validate()
    now_oauth = isvc.oauth_key(i.provider, data["config"] if data.get("config") is not None else i.config)
    if data.get("secret") and now_oauth:
        raise ValidationFailed("OAuth tokens are obtained through “Connect with …”; a secret cannot be set for this integration.")
    if was_oauth and not now_oauth and not data.get("secret"):
        raise ValidationFailed("Switching to token authentication needs the token.")
    if now_oauth and not was_oauth:
        isvc.require_oauth_app(i.provider)
    if "kb_id" in data:
        _check_kb(ctx, data["kb_id"])
    for key in ("name", "config", "kb_id", "enabled"):
        if key in data:
            setattr(i, key, data[key])
    if "sync_interval_minutes" in data:
        i.sync_interval_minutes = data["sync_interval_minutes"] or None
    if bool(was_oauth) != bool(now_oauth):
        isvc.drop_secret(ctx.db, i)  # credentials of the old auth mode never carry over
        i.cursor = {k: v for k, v in (i.cursor or {}).items() if k not in ("oauth", "oauth_pending")}
        i.status = "pending"
    if data.get("secret"):
        isvc.set_secret(ctx, i, data["secret"])
    if data.get("enabled") is False:
        i.status = "disabled"
    isvc.schedule(i)
    ctx.audit("integration.update", resource_type="integration", resource_id=i.id, meta={k: v for k, v in data.items() if k != "secret"})
    emit(ctx.db, ctx.org_id, "connection.updated", resource_type="integration", resource_id=i.id, actor_user_id=ctx.user_id, message=f"Integration '{i.name}' updated")
    ctx.db.commit()
    return _int_out(i)


@router.post("/integrations/{integration_id}/test")
def test_integration(integration_id: uuid.UUID, ctx: OrgContext = Depends(requires("org.integrations"))) -> dict[str, Any]:
    i = isvc.get(ctx, integration_id)
    result = isvc.test(ctx, i)
    ctx.db.commit()
    return result


class SyncIn(BaseModel):
    mode: str = Field(default="incremental", pattern="^(full|incremental)$")


@router.post("/integrations/{integration_id}/sync")
def sync_integration(integration_id: uuid.UUID, body: SyncIn, ctx: OrgContext = Depends(requires("org.integrations"))) -> dict[str, Any]:
    i = isvc.get(ctx, integration_id)
    if not i.kb_id:
        raise ValidationFailed("Choose a target knowledge base first.")
    if isvc.oauth_key(i.provider, i.config) and not i.credential_id:
        raise ValidationFailed("Authorize the integration first (Connect with …).")
    job_id = isvc.sync_now(ctx, i, body.mode)
    ctx.audit("integration.sync", resource_type="integration", resource_id=i.id)
    ctx.db.commit()
    return {"job_id": str(job_id)}


@router.post("/integrations/{integration_id}/oauth/start")
def oauth_start(integration_id: uuid.UUID, ctx: OrgContext = Depends(requires("org.integrations"))) -> dict[str, Any]:
    """Authorization URL for the vendor (PKCE + signed, single-use state valid for 10 minutes)."""
    i = isvc.get(ctx, integration_id)
    url = isvc.oauth_start(ctx, i)
    ctx.audit("integration.oauth_start", resource_type="integration", resource_id=i.id, meta={"provider": i.provider})
    ctx.db.commit()
    return {"authorization_url": url, "expires_in": oauth.STATE_MINUTES * 60}


@router.get("/integrations/oauth/callback", include_in_schema=False)
def oauth_callback(code: str | None = None, state: str | None = None, error: str | None = None, error_description: str | None = None) -> RedirectResponse:
    """Vendor redirect target. No bearer token: the signed state authenticates the request."""
    return RedirectResponse(isvc.oauth_callback(code=code, state=state, error=error, error_description=error_description), status_code=302, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})


@router.post("/integrations/{integration_id}/{action}", response_model=Ok)
def pause_resume_integration(integration_id: uuid.UUID, action: str, ctx: OrgContext = Depends(requires("org.integrations"))) -> Ok:
    if action not in ("pause", "resume"):
        raise NotFound("Unknown action.")
    i = isvc.get(ctx, integration_id)
    i.sync_paused = action == "pause"
    isvc.schedule(i)
    emit(ctx.db, ctx.org_id, "connection.updated", resource_type="integration", resource_id=i.id, actor_user_id=ctx.user_id, message=f"Sync {'paused' if i.sync_paused else 'resumed'}: {i.name}", payload={"sync_paused": i.sync_paused})
    ctx.db.commit()
    return Ok()


class DeleteIn(BaseModel):
    confirm_name: str
    delete_synced_documents: bool = False


@router.delete("/integrations/{integration_id}", response_model=Ok)
def delete_integration(integration_id: uuid.UUID, body: DeleteIn, ctx: OrgContext = Depends(requires("org.integrations"))) -> Ok:
    i = isvc.get(ctx, integration_id)
    if body.confirm_name != i.name:
        raise ValidationFailed("Type the integration name exactly to confirm.")
    if body.delete_synced_documents:
        from app.models import Document
        from app.services import knowledge_service as ks

        for doc in ctx.db.scalars(select(Document).where(Document.source_type == "integration", Document.source_id == i.id)):
            ks.delete_document(ctx, doc)
    if i.credential_id:
        from app.models import Credential

        cred = ctx.db.get(Credential, i.credential_id)
        if cred:
            ctx.db.delete(cred)  # credential revocation
    ctx.db.delete(i)
    ctx.audit("integration.delete", resource_type="integration", resource_id=integration_id)
    emit(ctx.db, ctx.org_id, "connection.deleted", resource_type="integration", resource_id=integration_id, actor_user_id=ctx.user_id, severity="warning", message=f"Integration '{i.name}' disconnected")
    ctx.db.commit()
    return Ok()


@router.get("/integrations/{integration_id}/runs")
def integration_runs(integration_id: uuid.UUID, ctx: OrgContext = Depends(requires("integrations.view"))) -> list[dict[str, Any]]:
    i = isvc.get(ctx, integration_id)
    rows = ctx.db.scalars(select(SyncRun).where(SyncRun.org_id == ctx.org_id, SyncRun.source_id == i.id).order_by(SyncRun.started_at.desc()).limit(50))
    return [{"id": str(r.id), "mode": r.mode, "status": r.status, "started_at": r.started_at.isoformat(), "finished_at": _iso(r.finished_at), "items_seen": r.items_seen, "items_added": r.items_added, "items_updated": r.items_updated, "items_deleted": r.items_deleted, "items_failed": r.items_failed, "error": r.error} for r in rows]


# ------------------------------------------------------------------- command center


@router.get("/connections/graph")
def connections_graph(workspace_id: uuid.UUID | None = None, kb_id: uuid.UUID | None = None, group: bool | None = None, ctx: OrgContext = Depends(requires("org.view"))) -> dict[str, Any]:
    return graph_service.build_graph(ctx, workspace_id=workspace_id, kb_id=kb_id, group=group)


@router.get("/connections/health")
def connections_health(ctx: OrgContext = Depends(requires("org.view"))) -> dict[str, Any]:
    return graph_service.health(ctx)


@router.get("/connections/node/{node_id}")
def node_details(node_id: str, ctx: OrgContext = Depends(requires("org.view"))) -> dict[str, Any]:
    """Side-panel details for a source node (masked endpoint, KBs, access, recent runs)."""
    kind, _, raw = node_id.partition(":")
    try:
        rid = uuid.UUID(raw)
    except ValueError as exc:
        raise NotFound("Unknown node.") from exc
    from app.models import ResourceGrant

    if kind == "db":
        from app.api.v1.databases import connection_out
        from app.services.connection_service import get_connection

        conn = get_connection(ctx, rid)
        out = connection_out(ctx, conn)
        rtype = "db_connection"
    elif kind == "int":
        ctx.require("integrations.view")
        out = _int_out(isvc.get(ctx, rid))
        rtype = "integration"
    else:
        raise NotFound("Details are available for data sources.")
    runs = ctx.db.scalars(select(SyncRun).where(SyncRun.org_id == ctx.org_id, SyncRun.source_id == rid).order_by(SyncRun.started_at.desc()).limit(10))
    kb = ctx.db.get(KnowledgeBase, uuid.UUID(out["kb_id"])) if out.get("kb_id") else None
    grants = list(ctx.db.scalars(select(ResourceGrant).where(ResourceGrant.org_id == ctx.org_id, ResourceGrant.resource_type == rtype, ResourceGrant.resource_id == rid)))
    events = ctx.db.scalars(select(ActivityEvent).where(ActivityEvent.org_id == ctx.org_id, ActivityEvent.resource_id == rid).order_by(ActivityEvent.id.desc()).limit(15))
    return {
        **out,
        "knowledge_base": {"id": str(kb.id), "name": kb.name} if kb and ctx.authz.kb_level(kb) >= READ else None,
        "authorized": [{"subject_type": g.subject_type, "subject_id": str(g.subject_id) if g.subject_id else None, "privilege": g.privilege, "effect": g.effect} for g in grants] if ctx.authz.has("admin.permissions") or ctx.authz.has("db.credentials") else None,
        "recent_runs": [{"status": r.status, "mode": r.mode, "started_at": r.started_at.isoformat(), "finished_at": _iso(r.finished_at), "items_added": r.items_added, "items_updated": r.items_updated, "items_deleted": r.items_deleted, "error": r.error} for r in runs],
        "recent_events": [event_to_dict(e) for e in events],
    }


class EventVisibility:
    """Per-subscriber filter: events about resources the viewer cannot see are dropped."""

    def __init__(self, authz: Authorizer) -> None:
        self.authz = authz
        self.kbs: set[str] = set()
        self.connections: dict[str, bool] = {}
        self.integrations = authz.has("integrations.view")
        self.admin = authz.has("admin.audit")
        self.refresh()

    def refresh(self) -> None:
        self.kbs = {str(k) for k in self.authz.access_scope().readable_kb_ids()}

    def allows(self, ev: dict[str, Any]) -> bool:
        rtype, rid, payload = ev.get("resource_type"), ev.get("resource_id"), ev.get("payload") or {}
        if rtype == "db_connection":
            if rid not in self.connections:
                conn = self.authz.session.get(DatabaseConnection, uuid.UUID(rid)) if rid else None
                self.connections[rid] = bool(conn and self.authz.connection_level(conn) >= READ)
                if conn is None and rid:
                    self.connections[rid] = self.authz.has("db.view")  # deleted: show to viewers of connections
            return self.connections[rid]
        if rtype == "integration":
            return self.integrations
        if rtype == "knowledge_base":
            return rid in self.kbs or (ev["type"] in ("kb.created",) and self._recheck(rid))
        if rtype == "document":
            kb = payload.get("kb_id")
            return kb in self.kbs or self._recheck(kb)
        if rtype in ("job",) or ev["type"].startswith("job."):
            return self.admin or ev.get("actor_user_id") == str(self.authz.p.user_id)
        return True

    def _recheck(self, kb_id: str | None) -> bool:
        if not kb_id:
            return False
        self.refresh()
        return kb_id in self.kbs


@router.get("/events")
def list_events(after_id: int | None = None, types: str | None = None, limit: int = 50, ctx: OrgContext = Depends(requires("org.view"))) -> dict[str, Any]:
    vis = EventVisibility(ctx.authz)
    stmt = select(ActivityEvent).where(ActivityEvent.org_id == ctx.org_id, ActivityEvent.type != "job.progress")
    if after_id:
        stmt = stmt.where(ActivityEvent.id > after_id)
    if types:
        stmt = stmt.where(ActivityEvent.type.in_(types.split(",")))
    rows = [event_to_dict(e) for e in ctx.db.scalars(stmt.order_by(ActivityEvent.id.desc()).limit(min(limit, 200) * 3))]
    items = [e for e in rows if vis.allows(e)][: min(limit, 200)]
    return page(items, len(items), limit, 0)


@router.get("/events/stream")
async def stream_events(request: Request, ctx: OrgContext = Depends(requires("org.view"))) -> StreamingResponse:
    """Server-Sent Events, organisation-scoped and filtered by the viewer's permissions.

    Resumable: send ``Last-Event-ID`` to replay missed events from the database.
    """
    principal, org_id = ctx.principal, ctx.org_id
    last_id = request.headers.get("last-event-id") or request.query_params.get("last_event_id")

    def _replay() -> tuple[list[dict[str, Any]], EventVisibility, Any]:
        db_ctx = tenant_session(org_id, principal.user_id)
        db = db_ctx.__enter__()
        authz = Authorizer(db, principal).load()
        vis = EventVisibility(authz)
        q = select(ActivityEvent).where(ActivityEvent.org_id == org_id)
        if last_id and last_id.isdigit():
            q = q.where(ActivityEvent.id > int(last_id)).order_by(ActivityEvent.id).limit(500)
            rows = list(db.scalars(q))
        else:
            rows = list(db.scalars(q.where(ActivityEvent.type != "job.progress").order_by(ActivityEvent.id.desc()).limit(25)))[::-1]
        db.commit()
        return [event_to_dict(e) for e in rows], vis, db_ctx

    backlog, vis, db_ctx = await run_in_threadpool(_replay)

    async def gen() -> AsyncIterator[bytes]:
        yield b"retry: 3000\n\n"
        try:
            for ev in backlog:
                if vis.allows(ev):
                    yield f"id: {ev['id']}\nevent: {ev['type']}\ndata: {json.dumps(ev, default=str)}\n\n".encode()
            sub = get_bus().subscribe(str(org_id)).__aiter__()
            pending = asyncio.ensure_future(sub.__anext__())
            while True:
                done, _ = await asyncio.wait({pending}, timeout=15)
                if await request.is_disconnected():
                    pending.cancel()
                    break
                if not done:
                    yield b": keep-alive\n\n"
                    continue
                ev = pending.result()
                pending = asyncio.ensure_future(sub.__anext__())
                allowed = await run_in_threadpool(vis.allows, ev)
                if allowed:
                    yield f"id: {ev['id']}\nevent: {ev['type']}\ndata: {json.dumps(ev, default=str)}\n\n".encode()
        finally:
            await run_in_threadpool(db_ctx.__exit__, None, None, None)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


@router.get("/connections/data-flow")
def data_flow(kb_id: uuid.UUID | None = None, ctx: OrgContext = Depends(requires("org.view"))) -> dict[str, Any]:
    """Live counts per pipeline stage (Source → … → AI response)."""
    from sqlalchemy import func

    from app.models import Chunk, ChunkEmbedding, Document, Message

    scope = ctx.authz.access_scope([kb_id] if kb_id else None)
    kbs = scope.readable_kb_ids() or [uuid.UUID(int=0)]
    db = ctx.db
    docs = dict(db.execute(select(Document.status, func.count()).where(Document.kb_id.in_(kbs)).group_by(Document.status)).all())
    chunks = db.scalar(select(func.count()).select_from(Chunk).where(Chunk.kb_id.in_(kbs))) or 0
    embedded = db.scalar(select(func.count()).select_from(ChunkEmbedding).where(ChunkEmbedding.kb_id.in_(kbs))) or 0
    active = list(db.scalars(select(Job).where(Job.org_id == ctx.org_id, Job.kind == "ingest_document", Job.status.in_(("queued", "running")))))
    stages_by_progress = {"parsing": 0, "chunking": 0, "embedding": 0, "indexing": 0}
    for j in active:
        stage = "parsing" if j.progress < 0.3 else ("chunking" if j.progress < 0.4 else ("embedding" if j.progress < 0.9 else "indexing"))
        stages_by_progress[stage] += 1
    answers = db.scalar(select(func.count()).select_from(Message).where(Message.org_id == ctx.org_id, Message.role == "assistant", Message.status == "complete")) or 0
    avg_ret = db.scalar(select(func.avg(Message.retrieval_ms)).where(Message.org_id == ctx.org_id, Message.retrieval_ms.is_not(None))) or 0
    avg_gen = db.scalar(select(func.avg(Message.generation_ms)).where(Message.org_id == ctx.org_id, Message.generation_ms.is_not(None))) or 0
    sources = (db.scalar(select(func.count()).select_from(DatabaseConnection).where(DatabaseConnection.org_id == ctx.org_id)) or 0) + (db.scalar(select(func.count()).select_from(Integration).where(Integration.org_id == ctx.org_id)) or 0)
    total_docs = sum(docs.values())
    return {"stages": [
        {"key": "source", "label": "Sources", "count": sources + (1 if total_docs else 0), "detail": f"{total_docs} documents"},
        {"key": "parsing", "label": "Parsing", "count": stages_by_progress["parsing"] + docs.get("queued", 0), "detail": f"{docs.get('failed', 0)} failed"},
        {"key": "normalization", "label": "Normalization", "count": stages_by_progress["parsing"], "detail": "Unicode, whitespace, injection screening"},
        {"key": "chunking", "label": "Chunking", "count": stages_by_progress["chunking"], "detail": f"{chunks} chunks"},
        {"key": "embeddings", "label": "Embeddings", "count": stages_by_progress["embedding"], "detail": f"{embedded} vectors"},
        {"key": "indexing", "label": "Indexing", "count": stages_by_progress["indexing"], "detail": "pgvector HNSW + full-text GIN"},
        {"key": "retrieval", "label": "Retrieval", "count": answers, "detail": f"avg {avg_ret:.0f} ms"},
        {"key": "response", "label": "AI response", "count": answers, "detail": f"avg {avg_gen:.0f} ms generation"},
    ], "documents": docs, "active_jobs": len(active)}
