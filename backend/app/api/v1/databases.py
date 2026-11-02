"""Database connections, datasets (structured analysis) and query approvals."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.api.deps import OrgContext, requires, requires_any_scope
from app.api.schemas import Ok, page
from app.connectors.databases import catalogue, provider_info
from app.llm.registry import resolve_llm
from app.models import Approval, DatabaseConnection, Dataset, Document, HealthCheck, KnowledgeBase, SqlQueryLog, SyncRun
from app.rbac.authorizer import MANAGE, READ, WRITE
from app.services import connection_service as cs
from app.services import dataset_service as dss
from app.services.activity import emit
from app.services.billing import check_limit
from app.text2sql import engine as sqlengine
from app.text2sql.validator import validate
from app.utils.errors import NotFound, PermissionDenied, ValidationFailed

router = APIRouter(tags=["data"])


class ConnectionIn(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    provider: str
    host: str = Field(default="", max_length=255)
    port: int | None = Field(default=None, ge=1, le=65535)
    database: str = Field(default="", max_length=255)
    username: str = Field(default="", max_length=255)
    password: str | None = Field(default=None, max_length=1024)
    ssl_mode: str = Field(default="prefer", pattern="^(disable|allow|prefer|require|verify-ca|verify-full)$")
    options: dict[str, Any] = {}
    access_mode: str = Field(default="live", pattern="^(live|ingest|hybrid)$")
    kb_id: uuid.UUID | None = None
    workspace_id: uuid.UUID | None = None
    sync_interval_minutes: int | None = Field(default=None, ge=5, le=60 * 24 * 30)
    query_limits: dict[str, Any] = {}


class ConnectionUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=160)
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    database: str | None = None
    username: str | None = None
    password: str | None = Field(default=None, max_length=1024)
    ssl_mode: str | None = Field(default=None, pattern="^(disable|allow|prefer|require|verify-ca|verify-full)$")
    access_mode: str | None = Field(default=None, pattern="^(live|ingest|hybrid)$")
    kb_id: uuid.UUID | None = None
    enabled: bool | None = None
    sync_interval_minutes: int | None = Field(default=None, ge=0, le=60 * 24 * 30)
    query_limits: dict[str, Any] | None = None


_SAFE_OPTION_KEYS = {"ssh_host", "ssh_port", "ssh_user", "warehouse", "project", "account", "http_path", "sync_cursors"}


def connection_out(ctx: OrgContext, c: DatabaseConnection) -> dict[str, Any]:
    level = ctx.authz.connection_level(c)
    manage = level >= MANAGE
    tables = (c.schema_cache or {}).get("tables", {})
    info = provider_info(c.provider)
    return {
        "id": str(c.id), "name": c.name, "provider": c.provider, "provider_label": info["label"], "provider_status": info["status"],
        "category": info.get("category"), "endpoint": cs.masked_endpoint(c), "host": c.host if manage else None, "port": c.port,
        "database": c.database, "username": c.username if manage else None, "ssl_mode": c.ssl_mode, "has_credentials": c.credential_id is not None,
        "options": {k: v for k, v in (c.options or {}).items() if k in _SAFE_OPTION_KEYS and k != "sync_cursors"} if manage else {},
        "access_mode": c.access_mode, "enabled": c.enabled, "status": c.status, "last_tested_at": _iso(c.last_tested_at),
        "last_success_at": _iso(c.last_success_at), "last_error": c.last_error, "schema_refreshed_at": _iso(c.schema_refreshed_at),
        "table_count": len(tables), "selected_tables": list((c.selection or {}).get("tables", {}).keys()),
        "selection": (c.selection or {}).get("tables", {}) if manage else None, "query_limits": c.query_limits,
        "sync_interval_minutes": c.sync_interval_minutes, "sync_paused": c.sync_paused, "next_sync_at": _iso(c.next_sync_at),
        "last_sync_at": _iso(c.last_sync_at), "kb_id": str(c.kb_id) if c.kb_id else None, "workspace_id": str(c.workspace_id) if c.workspace_id else None,
        "owner_id": str(c.owner_id) if c.owner_id else None, "access": {0: "none", 1: "read", 2: "write", 3: "manage"}[level],
        "created_at": _iso(c.created_at), "updated_at": _iso(c.updated_at),
    }


def _iso(v: datetime | None) -> str | None:
    return v.isoformat() if v else None


@router.get("/database-connections/providers")
def providers(ctx: OrgContext = Depends(requires("db.view"))) -> list[dict[str, Any]]:
    return catalogue()


@router.get("/database-connections")
def list_connections(ctx: OrgContext = Depends(requires("db.view"))) -> list[dict[str, Any]]:
    rows = ctx.db.scalars(select(DatabaseConnection).where(DatabaseConnection.org_id == ctx.org_id).order_by(DatabaseConnection.name))
    return [connection_out(ctx, c) for c in rows if ctx.authz.connection_level(c) >= READ]


def _check_kb(ctx: OrgContext, kb_id: uuid.UUID | None) -> None:
    if kb_id:
        kb = ctx.db.get(KnowledgeBase, kb_id)
        if kb is None or kb.org_id != ctx.org_id or ctx.authz.kb_level(kb) < WRITE:
            raise ValidationFailed("Target knowledge base not found or not writable.")


@router.post("/database-connections", status_code=201)
def create_connection(body: ConnectionIn, ctx: OrgContext = Depends(requires("db.create", "db.credentials"))) -> dict[str, Any]:
    info = provider_info(body.provider)
    check_limit(ctx.db, ctx.org_id, "connected_sources")
    _check_kb(ctx, body.kb_id)
    if info["status"] != "planned" and info["family"] == "sql" and body.provider != "sqlite" and not body.host:
        raise ValidationFailed("Host is required.")
    conn = DatabaseConnection(
        org_id=ctx.org_id, name=body.name.strip(), provider=body.provider, host=body.host.strip(), port=body.port, database=body.database.strip(),
        username=body.username.strip(), ssl_mode=body.ssl_mode, options={k: v for k, v in body.options.items() if k in _SAFE_OPTION_KEYS},
        access_mode=body.access_mode, kb_id=body.kb_id, workspace_id=body.workspace_id, owner_id=ctx.user_id,
        sync_interval_minutes=body.sync_interval_minutes, query_limits={"max_rows": 500, "timeout_s": 15, "max_joins": 4, **body.query_limits},
        status="unknown" if info["status"] != "planned" else "disabled", enabled=info["status"] != "planned",
    )
    ctx.db.add(conn)
    ctx.db.flush()
    cs.set_password(ctx.db, ctx.org_id, conn, body.password, ctx.user_id)
    cs.schedule(conn)
    ctx.audit("db_connection.create", resource_type="db_connection", resource_id=conn.id, meta={"name": conn.name, "provider": conn.provider})
    emit(ctx.db, ctx.org_id, "connection.created", resource_type="db_connection", resource_id=conn.id, actor_user_id=ctx.user_id, message=f"Connection '{conn.name}' ({info['label']}) added", payload={"provider": conn.provider})
    ctx.db.commit()
    return connection_out(ctx, conn)


@router.get("/database-connections/{conn_id}")
def get_connection(conn_id: uuid.UUID, ctx: OrgContext = Depends(requires("db.view"))) -> dict[str, Any]:
    return connection_out(ctx, cs.get_connection(ctx, conn_id))


@router.patch("/database-connections/{conn_id}")
def update_connection(conn_id: uuid.UUID, body: ConnectionUpdate, ctx: OrgContext = Depends(requires("db.edit"))) -> dict[str, Any]:
    conn = cs.get_connection(ctx, conn_id, MANAGE)
    data = body.model_dump(exclude_unset=True)
    if "password" in data:
        ctx.require("db.credentials")
        cs.set_password(ctx.db, ctx.org_id, conn, data.pop("password"), ctx.user_id)
        ctx.audit("db_connection.credentials_rotated", resource_type="db_connection", resource_id=conn.id)
    if "kb_id" in data:
        _check_kb(ctx, data["kb_id"])
    was_enabled = conn.enabled
    for key in ("name", "host", "port", "database", "username", "ssl_mode", "access_mode", "kb_id", "enabled"):
        if key in data:
            setattr(conn, key, data[key])
    if "sync_interval_minutes" in data:
        conn.sync_interval_minutes = data["sync_interval_minutes"] or None
    if data.get("query_limits") is not None:
        limits = {**(conn.query_limits or {}), **data["query_limits"]}
        limits["max_rows"] = max(1, min(int(limits.get("max_rows", 500)), 10_000))
        limits["timeout_s"] = max(1, min(int(limits.get("timeout_s", 15)), 300))
        conn.query_limits = limits
    if "enabled" in data and data["enabled"] != was_enabled:
        conn.status = "disabled" if not data["enabled"] else "unknown"
    cs.schedule(conn)
    ctx.audit("db_connection.update", resource_type="db_connection", resource_id=conn.id, meta={k: v for k, v in data.items() if k != "password"})
    emit(ctx.db, ctx.org_id, "connection.updated", resource_type="db_connection", resource_id=conn.id, actor_user_id=ctx.user_id, message=f"Connection '{conn.name}' updated")
    ctx.db.commit()
    return connection_out(ctx, conn)


class ConfirmIn(BaseModel):
    confirm_name: str
    delete_synced_documents: bool = False


@router.delete("/database-connections/{conn_id}", response_model=Ok)
def delete_connection(conn_id: uuid.UUID, body: ConfirmIn, ctx: OrgContext = Depends(requires("db.delete"))) -> Ok:
    conn = cs.get_connection(ctx, conn_id, MANAGE)
    if body.confirm_name != conn.name:
        raise ValidationFailed("Type the connection name exactly to confirm.")
    if body.delete_synced_documents:
        from app.services import knowledge_service as ks

        for doc in ctx.db.scalars(select(Document).where(Document.source_type == "database", Document.source_id == conn.id)):
            ks.delete_document(ctx, doc)
    from app.models import Credential

    if conn.credential_id:
        cred = ctx.db.get(Credential, conn.credential_id)
        if cred:
            ctx.db.delete(cred)  # credential revocation
    # Pending SQL approvals for this connection can no longer run; close them instead of leaving them decidable.
    from datetime import UTC, datetime

    from app.models import Approval

    stale = list(ctx.db.scalars(select(Approval).where(Approval.org_id == ctx.org_id, Approval.resource_id == conn.id, Approval.status == "pending")))
    for a in stale:
        a.status, a.decided_at, a.reason = "expired", datetime.now(UTC), "The connection was deleted."
    ctx.db.delete(conn)
    ctx.audit("db_connection.delete", resource_type="db_connection", resource_id=conn_id, meta={"name": conn.name, "expired_approvals": len(stale)})
    emit(ctx.db, ctx.org_id, "connection.deleted", resource_type="db_connection", resource_id=conn_id, actor_user_id=ctx.user_id, severity="warning", message=f"Connection '{conn.name}' disconnected")
    ctx.db.commit()
    return Ok(detail="Connection and its stored credentials were deleted.")


@router.post("/database-connections/{conn_id}/test")
def test_connection(conn_id: uuid.UUID, ctx: OrgContext = Depends(requires("db.test"))) -> dict[str, Any]:
    conn = cs.get_connection(ctx, conn_id, READ)
    result = cs.test_connection(ctx.db, conn, ctx.user_id)
    ctx.audit("db_connection.test", resource_type="db_connection", resource_id=conn.id, outcome="success" if result["ok"] else "failure")
    ctx.db.commit()
    return result


@router.post("/database-connections/{conn_id}/discover")
def discover_schema(conn_id: uuid.UUID, ctx: OrgContext = Depends(requires("db.schemas"))) -> dict[str, Any]:
    conn = cs.get_connection(ctx, conn_id, MANAGE)
    result = cs.discover(ctx.db, conn, ctx.user_id)
    ctx.audit("db_connection.discover", resource_type="db_connection", resource_id=conn.id, meta={"tables": result["tables"]})
    ctx.db.commit()
    return result


@router.get("/database-connections/{conn_id}/schema")
def get_schema(conn_id: uuid.UUID, ctx: OrgContext = Depends(requires("db.view"))) -> dict[str, Any]:
    conn = cs.get_connection(ctx, conn_id, READ)
    tables = (conn.schema_cache or {}).get("tables", {})
    if ctx.authz.connection_level(conn) < MANAGE:
        # Non-managers see only selected tables and non-sensitive columns.
        sel = (conn.selection or {}).get("tables", {})
        allowed = ctx.authz.connection_table_constraint(conn)
        tables = {k: {**v, "columns": [c for c in v["columns"] if c["name"] in set(sel[k].get("columns") or [x["name"] for x in v["columns"]]) - set(sel[k].get("sensitive") or []) - set(sel[k].get("excluded") or [])]} for k, v in tables.items() if k in sel and (allowed is None or k in allowed)}
    return {"schemas": (conn.schema_cache or {}).get("schemas", []), "tables": tables, "refreshed_at": _iso(conn.schema_refreshed_at)}


class SelectionIn(BaseModel):
    tables: dict[str, dict[str, Any]]


@router.put("/database-connections/{conn_id}/selection")
def set_selection(conn_id: uuid.UUID, body: SelectionIn, ctx: OrgContext = Depends(requires("db.schemas"))) -> dict[str, Any]:
    conn = cs.get_connection(ctx, conn_id, MANAGE)
    clean = cs.apply_selection(conn, body.tables)
    ctx.audit("db_connection.selection", resource_type="db_connection", resource_id=conn.id, meta={"tables": sorted(clean)})
    ctx.db.commit()
    return {"tables": clean}


class SyncIn(BaseModel):
    mode: str = Field(default="incremental", pattern="^(full|incremental)$")


@router.post("/database-connections/{conn_id}/sync")
def sync_connection(conn_id: uuid.UUID, body: SyncIn, ctx: OrgContext = Depends(requires("db.sync"))) -> dict[str, Any]:
    conn = cs.get_connection(ctx, conn_id, READ)
    if conn.access_mode == "live":
        raise ValidationFailed("This connection is in live-query mode. Switch to ingest or hybrid to synchronise data.")
    if not (conn.selection or {}).get("tables"):
        raise ValidationFailed("Select at least one table to synchronise.")
    job_id = cs.sync_now(ctx, conn, body.mode)
    ctx.audit("db_connection.sync", resource_type="db_connection", resource_id=conn.id, meta={"mode": body.mode})
    ctx.db.commit()
    return {"job_id": str(job_id)}


@router.post("/database-connections/{conn_id}/pause", response_model=Ok)
def pause_sync(conn_id: uuid.UUID, ctx: OrgContext = Depends(requires("db.sync"))) -> Ok:
    conn = cs.get_connection(ctx, conn_id, READ)
    conn.sync_paused = True
    cs.schedule(conn)
    ctx.audit("db_connection.pause", resource_type="db_connection", resource_id=conn.id)
    emit(ctx.db, ctx.org_id, "connection.updated", resource_type="db_connection", resource_id=conn.id, actor_user_id=ctx.user_id, message=f"Sync paused: {conn.name}", payload={"sync_paused": True})
    ctx.db.commit()
    return Ok()


@router.post("/database-connections/{conn_id}/resume", response_model=Ok)
def resume_sync(conn_id: uuid.UUID, ctx: OrgContext = Depends(requires("db.sync"))) -> Ok:
    conn = cs.get_connection(ctx, conn_id, READ)
    conn.sync_paused = False
    cs.schedule(conn)
    ctx.audit("db_connection.resume", resource_type="db_connection", resource_id=conn.id)
    emit(ctx.db, ctx.org_id, "connection.updated", resource_type="db_connection", resource_id=conn.id, actor_user_id=ctx.user_id, message=f"Sync resumed: {conn.name}", payload={"sync_paused": False})
    ctx.db.commit()
    return Ok()


@router.get("/database-connections/{conn_id}/health")
def connection_health(conn_id: uuid.UUID, ctx: OrgContext = Depends(requires("db.view"))) -> list[dict[str, Any]]:
    conn = cs.get_connection(ctx, conn_id, READ)
    rows = ctx.db.scalars(select(HealthCheck).where(HealthCheck.org_id == ctx.org_id, HealthCheck.source_id == conn.id).order_by(HealthCheck.checked_at.desc()).limit(100))
    return [{"ok": h.ok, "latency_ms": h.latency_ms, "error": h.error, "checked_at": h.checked_at.isoformat()} for h in rows]


@router.get("/database-connections/{conn_id}/runs")
def connection_runs(conn_id: uuid.UUID, ctx: OrgContext = Depends(requires("db.view"))) -> list[dict[str, Any]]:
    conn = cs.get_connection(ctx, conn_id, READ)
    rows = ctx.db.scalars(select(SyncRun).where(SyncRun.org_id == ctx.org_id, SyncRun.source_id == conn.id).order_by(SyncRun.started_at.desc()).limit(50))
    return [{"id": str(r.id), "mode": r.mode, "status": r.status, "started_at": r.started_at.isoformat(), "finished_at": _iso(r.finished_at), "items_seen": r.items_seen, "items_added": r.items_added, "items_updated": r.items_updated, "items_deleted": r.items_deleted, "error": r.error} for r in rows]


class QueryIn(BaseModel):
    question: str | None = Field(default=None, max_length=4000)
    sql: str | None = Field(default=None, max_length=20000)


@router.post("/database-connections/{conn_id}/query")
def query_connection(conn_id: uuid.UUID, body: QueryIn, ctx: OrgContext = Depends(requires("db.query"))) -> dict[str, Any]:
    conn = cs.get_connection(ctx, conn_id, READ)
    if not conn.enabled:
        raise ValidationFailed("Connection is disabled.")
    if conn.access_mode == "ingest":
        raise ValidationFailed("Live querying is disabled for this connection (ingest-only mode).")
    source = dss.connection_source(ctx.db, ctx.authz, conn)
    if not source.tables:
        raise ValidationFailed("No tables are available to you on this connection.")
    if body.sql:
        outcome = sqlengine.run_sql(ctx.db, ctx.org_id, ctx.user_id, source, body.sql, question=body.question or "")
    elif body.question:
        outcome = sqlengine.run_question(ctx.db, ctx.org_id, ctx.user_id, source, body.question, resolve_llm(ctx.db, ctx.org))
    else:
        raise ValidationFailed("Provide a question or SQL.")
    ctx.audit("db_connection.query", resource_type="db_connection", resource_id=conn.id, outcome="success" if outcome.status == "executed" else "denied" if outcome.status == "rejected" else "failure", meta={"status": outcome.status, "rows": len(outcome.rows)})
    ctx.db.commit()
    return outcome.as_dict()


@router.get("/query-logs")
def query_logs(source_id: uuid.UUID | None = None, status: str | None = None, limit: int = 100, offset: int = 0, ctx: OrgContext = Depends(requires("db.view"))) -> dict[str, Any]:
    stmt = select(SqlQueryLog).where(SqlQueryLog.org_id == ctx.org_id)
    if not (ctx.authz.has("db.edit") or ctx.authz.has("admin.audit")):
        stmt = stmt.where(SqlQueryLog.user_id == ctx.user_id)
    if source_id:
        stmt = stmt.where(SqlQueryLog.source_id == source_id)
    if status:
        stmt = stmt.where(SqlQueryLog.status == status)
    total = ctx.db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = ctx.db.scalars(stmt.order_by(SqlQueryLog.created_at.desc()).limit(min(limit, 500)).offset(offset))
    return page([{"id": str(r.id), "source_type": r.source_type, "source_id": str(r.source_id), "user_id": str(r.user_id) if r.user_id else None, "question": r.question, "generated_sql": r.generated_sql, "executed_sql": r.executed_sql, "status": r.status, "row_count": r.row_count, "duration_ms": r.duration_ms, "error": r.error, "reasons": r.reasons, "created_at": r.created_at.isoformat()} for r in rows], total, limit, offset)


# --------------------------------------------------------------------------- approvals


def _approval_out(ctx: OrgContext, a: Approval) -> dict[str, Any]:
    # The run-resume state is internal; results go only to the requester and approvers.
    visible = a.requested_by == ctx.user_id or ctx.authz.has("db.approve")
    return {"id": str(a.id), "kind": a.kind, "status": a.status, "summary": a.summary, "payload": {k: v for k, v in a.payload.items() if k not in ("result", "resume")},
            "result": a.payload.get("result") if visible else None, "requested_by": str(a.requested_by), "mine": a.requested_by == ctx.user_id,
            "resource_type": a.resource_type, "resource_id": str(a.resource_id) if a.resource_id else None, "decided_by": str(a.decided_by) if a.decided_by else None,
            "decided_at": a.decided_at.isoformat() if a.decided_at else None, "reason": a.reason, "created_at": a.created_at.isoformat()}


@router.get("/approvals")
def list_approvals(status: str = "pending", ctx: OrgContext = Depends(requires("org.view"))) -> list[dict[str, Any]]:
    stmt = select(Approval).where(Approval.org_id == ctx.org_id, Approval.status == status)
    if not ctx.authz.has("db.approve"):
        stmt = stmt.where(Approval.requested_by == ctx.user_id)
    rows = ctx.db.scalars(stmt.order_by(Approval.created_at.desc()).limit(200))
    return [_approval_out(ctx, a) for a in rows]


def _visible_approval(ctx: OrgContext, approval_id: uuid.UUID) -> Approval:
    a = ctx.db.get(Approval, approval_id)
    if a is None or a.org_id != ctx.org_id or not (a.requested_by == ctx.user_id or ctx.authz.has("db.approve")):
        raise NotFound("Approval not found.")
    return a


@router.get("/approvals/{approval_id}")
def get_approval(approval_id: uuid.UUID, ctx: OrgContext = Depends(requires("org.view"))) -> dict[str, Any]:
    return _approval_out(ctx, _visible_approval(ctx, approval_id))


class WithdrawIn(BaseModel):
    reason: str = Field(default="", max_length=2000)


@router.post("/approvals/{approval_id}/withdraw")
def withdraw_approval(approval_id: uuid.UUID, body: WithdrawIn, ctx: OrgContext = Depends(requires("org.view"))) -> dict[str, Any]:
    """The requester takes back a request nobody has decided yet."""
    a = _visible_approval(ctx, approval_id)
    if a.requested_by != ctx.user_id:
        raise PermissionDenied("Only the person who made the request can withdraw it.")
    if a.status != "pending":
        raise ValidationFailed("This request was already decided.")
    a.status, a.decided_by, a.decided_at, a.reason = "withdrawn", ctx.user_id, datetime.now(UTC), body.reason or "Withdrawn by the requester."
    log_id = a.payload.get("log_id")
    if log_id and (log := ctx.db.get(SqlQueryLog, uuid.UUID(log_id))):
        log.status = "withdrawn"
    ctx.audit("approval.withdraw", resource_type="approval", resource_id=a.id, meta={"kind": a.kind})
    ctx.db.commit()
    return _approval_out(ctx, a)


class DecisionIn(BaseModel):
    approve: bool
    reason: str = Field(default="", max_length=2000)


@router.post("/approvals/{approval_id}/decide")
def decide_approval(approval_id: uuid.UUID, body: DecisionIn, ctx: OrgContext = Depends(requires("db.approve"))) -> dict[str, Any]:
    a = ctx.db.get(Approval, approval_id)
    if a is None or a.org_id != ctx.org_id:
        raise NotFound("Approval not found.")
    if a.status != "pending":
        raise ValidationFailed("This request was already decided.")
    if a.requested_by == ctx.user_id:
        raise PermissionDenied("You cannot approve your own request.")
    a.status = "approved" if body.approve else "denied"
    a.decided_by, a.decided_at, a.reason = ctx.user_id, datetime.now(UTC), body.reason
    result: dict[str, Any] | None = None
    if body.approve and a.kind == "sql_query":
        payload = a.payload
        source_id = uuid.UUID(payload["source_id"])
        if payload["source_kind"] == "database":
            conn = ctx.db.get(DatabaseConnection, source_id)
            if conn is None:
                raise NotFound("Connection no longer exists.")
            from app.models import User
            from app.rbac.authorizer import Authorizer, Principal

            requester = ctx.db.get(User, a.requested_by)
            # Execute with the REQUESTER's permissions, not the approver's.
            req_authz = Authorizer(ctx.db, Principal(user_id=a.requested_by, email=requester.email if requester else "", org_id=ctx.org_id)).load()
            source = dss.connection_source(ctx.db, req_authz, conn)
            validation = validate(payload["sql"], source.policy())
            if not validation.ok:
                raise ValidationFailed("The query is no longer valid: " + "; ".join(validation.errors))
            outcome = sqlengine.execute_validated(ctx.db, ctx.org_id, a.requested_by, source, sqlengine.SqlOutcome(status="approved", question=payload.get("question", ""), generated_sql=payload["sql"]), validation, approved=True)
            result = outcome.as_dict()
            result["rows"] = result["rows"][:200]
            a.payload = {**payload, "result": result}
    if a.kind == "agent_tool":
        from app.jobs.queue import enqueue
        from app.models import Notification

        if body.approve:
            # The paused run continues in the background with the requester's permissions.
            a.payload = {**a.payload, "result": {"status": "resuming"}}
            enqueue(ctx.db, ctx.org_id, "resume_agent_run", payload={"approval_id": str(a.id)}, created_by=a.requested_by, resource_type="approval", resource_id=a.id, idempotency_key=f"resume-agent:{a.id}")
        else:
            ctx.db.add(Notification(org_id=ctx.org_id, user_id=a.requested_by, kind="approval_denied", title=f"Request denied: {a.summary}"[:300], body=body.reason[:500], severity="warning", link="/agents"))
    log_id = a.payload.get("log_id")
    if log_id:
        log = ctx.db.get(SqlQueryLog, uuid.UUID(log_id))
        if log:
            log.status = "approved" if body.approve else "denied"
            log.decided_by, log.decided_at = ctx.user_id, datetime.now(UTC)
    ctx.audit("approval.decide", resource_type="approval", resource_id=a.id, meta={"approve": body.approve, "kind": a.kind})
    ctx.db.commit()
    return {"status": a.status, "result": result}


# ----------------------------------------------------------------------------- datasets


def _get_dataset(ctx: OrgContext, dataset_id: uuid.UUID) -> Dataset:
    ds = ctx.db.get(Dataset, dataset_id)
    if ds is None or ds.org_id != ctx.org_id:
        raise NotFound("Dataset not found.")
    doc = ctx.db.get(Document, ds.document_id)
    scope = ctx.authz.access_scope([ds.kb_id])
    if doc is None or not dss.dataset_visible(scope, doc, ctx.user_id):
        raise NotFound("Dataset not found.")
    return ds


def _dataset_out(ds: Dataset, doc: Document | None = None, kb_name: str = "") -> dict[str, Any]:
    return {"id": str(ds.id), "name": ds.name, "document_id": str(ds.document_id), "kb_id": str(ds.kb_id), "kb_name": kb_name, "table_name": ds.table_name, "tables": [{"name": t["name"], "rows": t["rows"], "columns": t["columns"]} for t in (ds.profile or {}).get("tables", [])], "row_count": ds.row_count, "column_count": ds.column_count, "source_type": doc.source_type if doc else None, "filename": doc.filename if doc else None, "updated_at": ds.updated_at.isoformat()}


@router.get("/datasets")
def list_datasets(kb_id: uuid.UUID | None = None, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> list[dict[str, Any]]:
    rows = dss.visible_datasets(ctx.db, ctx.authz, [kb_id] if kb_id else None)
    names = dict(ctx.db.execute(select(KnowledgeBase.id, KnowledgeBase.name).where(KnowledgeBase.org_id == ctx.org_id)).all())
    return [_dataset_out(ds, doc, names.get(ds.kb_id, "")) for ds, doc in rows]


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: uuid.UUID, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> dict[str, Any]:
    ds = _get_dataset(ctx, dataset_id)
    return {**_dataset_out(ds, ctx.db.get(Document, ds.document_id), dss.kb_name(ctx.db, ds.kb_id)), "profile": ds.profile}


@router.get("/datasets/{dataset_id}/preview")
def preview_dataset(dataset_id: uuid.UUID, table: str | None = None, limit: int = 50, offset: int = 0, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> dict[str, Any]:
    ds = _get_dataset(ctx, dataset_id)
    df = dss.load_frame(ds, table)
    window = df.iloc[offset : offset + min(limit, 500)]
    return {"columns": [str(c) for c in df.columns], "rows": [[None if (v is None or v != v) else (v if isinstance(v, (int, float, str, bool)) else str(v)) for v in row] for row in window.itertuples(index=False, name=None)], "total": int(len(df)), "offset": offset}


@router.get("/datasets/{dataset_id}/analysis")
def analyze_dataset(dataset_id: uuid.UUID, table: str | None = None, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> dict[str, Any]:
    return dss.analysis(_get_dataset(ctx, dataset_id), table)


@router.post("/datasets/{dataset_id}/summary")
def summarize_dataset(dataset_id: uuid.UUID, table: str | None = None, ctx: OrgContext = Depends(requires_any_scope("kb.chat"))) -> dict[str, Any]:
    ds = _get_dataset(ctx, dataset_id)
    return dss.summarize(ds, resolve_llm(ctx.db, ctx.org), table)


@router.post("/datasets/{dataset_id}/query")
def query_dataset(dataset_id: uuid.UUID, body: QueryIn, ctx: OrgContext = Depends(requires_any_scope("kb.chat"))) -> dict[str, Any]:
    ds = _get_dataset(ctx, dataset_id)
    with dss.dataset_source(ds) as source:
        if body.sql:
            outcome = sqlengine.run_sql(ctx.db, ctx.org_id, ctx.user_id, source, body.sql, question=body.question or "")
        elif body.question:
            outcome = sqlengine.run_question(ctx.db, ctx.org_id, ctx.user_id, source, body.question, resolve_llm(ctx.db, ctx.org))
        else:
            raise ValidationFailed("Provide a question or SQL.")
    ctx.db.commit()
    return outcome.as_dict()
