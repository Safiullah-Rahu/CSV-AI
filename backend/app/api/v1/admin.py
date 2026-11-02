"""Platform super-administration.

Platform administrators manage tenants, users, plans and system health. They
do NOT get access to customer content: these endpoints return metadata and
counts only. Reading an organisation's knowledge requires an owner-approved,
time-boxed support grant (see ``/support-access``), and every such access is
audited.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.api.deps import platform_admin
from app.api.schemas import Ok, page
from app.db.session import system_session
from app.models import (
    AIProviderConfig,
    AuditEvent,
    Chunk,
    Document,
    Job,
    KnowledgeBase,
    Membership,
    Organization,
    Plan,
    Subscription,
    UsageRecord,
    User,
)
from app.rbac.authorizer import Principal
from app.services.activity import audit
from app.services.auth_service import revoke_all_sessions
from app.utils.errors import NotFound, ValidationFailed

router = APIRouter(prefix="/admin", tags=["platform admin"])


def _iso(v: datetime | None) -> str | None:
    return v.isoformat() if v else None


@router.get("/organizations")
def organizations(q: str | None = None, limit: int = 50, offset: int = 0, admin: tuple[Principal, Session] = Depends(platform_admin)) -> dict[str, Any]:
    with system_session() as db:
        stmt = select(Organization)
        if q:
            stmt = stmt.where(or_(Organization.name.ilike(f"%{q}%"), Organization.slug.ilike(f"%{q}%")))
        total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        orgs = list(db.scalars(stmt.order_by(Organization.created_at.desc()).limit(min(limit, 200)).offset(offset)))
        ids = [o.id for o in orgs]
        members = dict(db.execute(select(Membership.org_id, func.count()).where(Membership.org_id.in_(ids)).group_by(Membership.org_id)).all())
        kbs = dict(db.execute(select(KnowledgeBase.org_id, func.count()).where(KnowledgeBase.org_id.in_(ids)).group_by(KnowledgeBase.org_id)).all())
        docs = {o: (int(n), int(s)) for o, n, s in db.execute(select(Document.org_id, func.count(), func.coalesce(func.sum(Document.size_bytes), 0)).where(Document.org_id.in_(ids)).group_by(Document.org_id))}
        subs = {s.org_id: s for s in db.scalars(select(Subscription).where(Subscription.org_id.in_(ids)))}
        items = [{"id": str(o.id), "name": o.name, "slug": o.slug, "status": o.status, "created_at": _iso(o.created_at), "members": members.get(o.id, 0), "knowledge_bases": kbs.get(o.id, 0), "documents": docs.get(o.id, (0, 0))[0], "storage_bytes": docs.get(o.id, (0, 0))[1], "plan": subs[o.id].plan_key if o.id in subs else None} for o in orgs]
    return page(items, total, limit, offset)


class OrgAdminUpdate(BaseModel):
    status: str | None = Field(default=None, pattern="^(active|suspended|deactivated)$")
    plan: str | None = None
    limit_overrides: dict[str, Any] | None = None


@router.patch("/organizations/{org_id}", response_model=Ok)
def update_organization(org_id: uuid.UUID, body: OrgAdminUpdate, admin: tuple[Principal, Session] = Depends(platform_admin)) -> Ok:
    principal = admin[0]
    with system_session() as db:
        org = db.get(Organization, org_id)
        if org is None:
            raise NotFound("Organization not found.")
        if body.status:
            org.status = body.status
        if body.plan or body.limit_overrides is not None:
            sub = db.scalar(select(Subscription).where(Subscription.org_id == org_id))
            if sub is None:
                sub = Subscription(org_id=org_id, plan_key=body.plan or "enterprise")
                db.add(sub)
            if body.plan:
                if db.get(Plan, body.plan) is None:
                    raise ValidationFailed("Unknown plan.")
                sub.plan_key = body.plan
            if body.limit_overrides is not None:
                sub.limit_overrides = body.limit_overrides
        audit(db, "platform.org_update", org_id=None, actor_user_id=principal.user_id, actor_email=principal.email, resource_type="organization", resource_id=org_id, meta=body.model_dump(exclude_none=True))
    return Ok()


@router.get("/users")
def users(q: str | None = None, limit: int = 50, offset: int = 0, admin: tuple[Principal, Session] = Depends(platform_admin)) -> dict[str, Any]:
    with system_session() as db:
        stmt = select(User)
        if q:
            stmt = stmt.where(or_(User.email.ilike(f"%{q}%"), User.full_name.ilike(f"%{q}%")))
        total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        rows = list(db.scalars(stmt.order_by(User.created_at.desc()).limit(min(limit, 200)).offset(offset)))
        counts = dict(db.execute(select(Membership.user_id, func.count()).where(Membership.user_id.in_([u.id for u in rows])).group_by(Membership.user_id)).all())
        items = [{"id": str(u.id), "email": u.email, "full_name": u.full_name, "status": u.status, "is_platform_admin": u.is_platform_admin, "mfa_enabled": u.mfa_enabled, "email_verified": bool(u.email_verified_at), "organizations": counts.get(u.id, 0), "last_login_at": _iso(u.last_login_at), "created_at": _iso(u.created_at)} for u in rows]
    return page(items, total, limit, offset)


class UserAdminUpdate(BaseModel):
    status: str | None = Field(default=None, pattern="^(active|suspended|deactivated)$")
    is_platform_admin: bool | None = None


@router.patch("/users/{user_id}", response_model=Ok)
def update_user(user_id: uuid.UUID, body: UserAdminUpdate, admin: tuple[Principal, Session] = Depends(platform_admin)) -> Ok:
    principal = admin[0]
    if user_id == principal.user_id:
        raise ValidationFailed("You cannot change your own platform account here.")
    with system_session() as db:
        user = db.get(User, user_id)
        if user is None:
            raise NotFound("User not found.")
        if body.status:
            user.status = body.status
            if body.status != "active":
                revoke_all_sessions(db, user.id, reason=f"platform_{body.status}")
        if body.is_platform_admin is not None:
            user.is_platform_admin = body.is_platform_admin
        audit(db, "platform.user_update", actor_user_id=principal.user_id, actor_email=principal.email, resource_type="user", resource_id=user_id, meta=body.model_dump(exclude_none=True))
    return Ok()


@router.get("/system")
def system(admin: tuple[Principal, Session] = Depends(platform_admin)) -> dict[str, Any]:
    from app.api.v1.health import readiness
    from app.config import get_settings
    from app.storage import get_storage

    ok, checks = readiness()
    with system_session() as db:
        jobs = dict(db.execute(select(Job.status, func.count()).group_by(Job.status)).all())
        dead_by_kind = dict(db.execute(select(Job.kind, func.count()).where(Job.status == "dead").group_by(Job.kind)).all())
        oldest = db.scalar(select(func.min(Job.created_at)).where(Job.status == "queued"))
        providers = dict(db.execute(select(AIProviderConfig.status, func.count()).group_by(AIProviderConfig.status)).all())
        totals = {
            "organizations": db.scalar(select(func.count()).select_from(Organization)) or 0,
            "users": db.scalar(select(func.count()).select_from(User)) or 0,
            "documents": db.scalar(select(func.count()).select_from(Document)) or 0,
            "chunks": db.scalar(select(func.count()).select_from(Chunk)) or 0,
            "storage_bytes": int(db.scalar(select(func.coalesce(func.sum(Document.size_bytes), 0))) or 0),
        }
        since = datetime.now(UTC) - timedelta(days=30)
        usage = db.execute(select(func.coalesce(func.sum(UsageRecord.tokens_in + UsageRecord.tokens_out), 0), func.coalesce(func.sum(UsageRecord.cost_usd), 0), func.count()).where(UsageRecord.created_at >= since)).one()
    settings = get_settings()
    try:
        disk = get_storage().usage_bytes("")
    except Exception:  # noqa: BLE001
        disk = None
    return {"ready": ok, "checks": checks, "jobs": {"by_status": jobs, "dead_by_kind": dead_by_kind, "oldest_queued": _iso(oldest)}, "ai_providers": providers, "totals": totals, "storage_backend": settings.storage_backend, "storage_bytes_on_disk": disk, "usage_30d": {"tokens": int(usage[0]), "cost_usd": float(usage[1]), "events": int(usage[2])}, "settings": {"environment": settings.environment, "job_backend": settings.job_backend, "billing_enabled": settings.billing_enabled, "event_backend": settings.event_backend}}


@router.get("/jobs")
def jobs(status: str = "dead", limit: int = 100, admin: tuple[Principal, Session] = Depends(platform_admin)) -> list[dict[str, Any]]:
    with system_session() as db:
        rows = db.scalars(select(Job).where(Job.status == status).order_by(Job.created_at.desc()).limit(min(limit, 500)))
        # Errors are technical messages; job payloads may reference content ids only.
        return [{"id": str(j.id), "org_id": str(j.org_id), "kind": j.kind, "status": j.status, "attempts": j.attempts, "error": (j.error or "")[:300], "created_at": _iso(j.created_at), "finished_at": _iso(j.finished_at)} for j in rows]


@router.post("/jobs/{job_id}/retry", response_model=Ok)
def retry(job_id: uuid.UUID, admin: tuple[Principal, Session] = Depends(platform_admin)) -> Ok:
    with system_session() as db:
        job = db.get(Job, job_id)
        if job is None:
            raise NotFound("Job not found.")
        job.status, job.attempts, job.next_run_at, job.error = "queued", 0, None, None
        audit(db, "platform.job_retry", actor_user_id=admin[0].user_id, resource_type="job", resource_id=job_id)
    return Ok()


@router.get("/audit")
def platform_audit(limit: int = 100, admin: tuple[Principal, Session] = Depends(platform_admin)) -> list[dict[str, Any]]:
    with system_session() as db:
        rows = db.scalars(select(AuditEvent).where(or_(AuditEvent.org_id.is_(None), AuditEvent.action.like("platform.%"), AuditEvent.action.like("support.%"))).order_by(AuditEvent.id.desc()).limit(min(limit, 500)))
        return [{"id": e.id, "org_id": str(e.org_id) if e.org_id else None, "action": e.action, "outcome": e.outcome, "actor_email": e.actor_email, "resource_type": e.resource_type, "resource_id": e.resource_id, "ip": e.ip, "created_at": _iso(e.created_at)} for e in rows]


@router.get("/plans")
def plans(admin: tuple[Principal, Session] = Depends(platform_admin)) -> list[dict[str, Any]]:
    with system_session() as db:
        return [{"key": p.key, "name": p.name, "limits": p.limits, "features": p.features, "price_monthly_usd": float(p.price_monthly_usd) if p.price_monthly_usd is not None else None} for p in db.scalars(select(Plan).order_by(Plan.sort))]
