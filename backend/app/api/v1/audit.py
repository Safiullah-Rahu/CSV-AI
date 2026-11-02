"""Audit log browsing and export."""

from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from app.api.deps import OrgContext, requires
from app.api.schemas import page
from app.models import AuditEvent

router = APIRouter(tags=["audit"])


def _query(ctx: OrgContext, action: str | None, actor: uuid.UUID | None, outcome: str | None, resource_type: str | None, since: datetime | None, until: datetime | None, q: str | None):
    stmt = select(AuditEvent).where(AuditEvent.org_id == ctx.org_id)
    if action:
        stmt = stmt.where(AuditEvent.action.like(f"{action}%"))
    if actor:
        stmt = stmt.where(AuditEvent.actor_user_id == actor)
    if outcome:
        stmt = stmt.where(AuditEvent.outcome == outcome)
    if resource_type:
        stmt = stmt.where(AuditEvent.resource_type == resource_type)
    if since:
        stmt = stmt.where(AuditEvent.created_at >= since)
    if until:
        stmt = stmt.where(AuditEvent.created_at <= until)
    if q:
        stmt = stmt.where(func.lower(AuditEvent.actor_email).like(f"%{q.lower()}%") | AuditEvent.resource_id.like(f"%{q}%"))
    return stmt


def _row(e: AuditEvent) -> dict[str, Any]:
    return {
        "id": e.id, "action": e.action, "outcome": e.outcome, "actor_user_id": str(e.actor_user_id) if e.actor_user_id else None,
        "actor_email": e.actor_email, "actor_api_key_id": str(e.actor_api_key_id) if e.actor_api_key_id else None,
        "resource_type": e.resource_type, "resource_id": e.resource_id, "ip": e.ip, "user_agent": e.user_agent,
        "request_id": e.request_id, "metadata": e.meta, "created_at": e.created_at.isoformat(),
    }


@router.get("/audit")
def list_audit(
    action: str | None = None, actor: uuid.UUID | None = None, outcome: str | None = None, resource_type: str | None = None,
    since: datetime | None = None, until: datetime | None = None, q: str | None = None, limit: int = 100, offset: int = 0,
    ctx: OrgContext = Depends(requires("admin.audit")),
) -> dict[str, Any]:
    stmt = _query(ctx, action, actor, outcome, resource_type, since, until, q)
    total = ctx.db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = ctx.db.scalars(stmt.order_by(AuditEvent.id.desc()).limit(min(limit, 500)).offset(offset))
    return page([_row(e) for e in rows], total, limit, offset)


@router.get("/audit/export")
def export_audit(
    action: str | None = None, actor: uuid.UUID | None = None, outcome: str | None = None, resource_type: str | None = None,
    since: datetime | None = None, until: datetime | None = None, q: str | None = None,
    ctx: OrgContext = Depends(requires("admin.audit")),
) -> StreamingResponse:
    stmt = _query(ctx, action, actor, outcome, resource_type, since, until, q).order_by(AuditEvent.id.desc()).limit(50_000)
    rows = [_row(e) for e in ctx.db.scalars(stmt)]
    ctx.audit("audit.export", meta={"rows": len(rows)})
    ctx.db.commit()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["id", "created_at", "action", "outcome", "actor_email", "resource_type", "resource_id", "ip", "request_id"], extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=audit.csv"})
