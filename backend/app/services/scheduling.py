"""Scheduler tick: enqueue due synchronisations and maintenance jobs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.db.session import system_session
from app.jobs.queue import enqueue
from app.models import DatabaseConnection, Integration
from app.utils.logging import get_logger

log = get_logger("scheduler")


def run_due_schedules() -> int:
    """Enqueue sync jobs for connections/integrations whose ``next_sync_at`` passed."""
    now = datetime.now(UTC)
    queued = 0
    with system_session() as db:
        for model, kind, rtype in (
            (DatabaseConnection, "sync_database", "db_connection"),
            (Integration, "sync_integration", "integration"),
        ):
            due = db.scalars(
                select(model)
                .where(
                    model.enabled.is_(True),
                    model.sync_paused.is_(False),
                    model.sync_interval_minutes.is_not(None),
                    model.next_sync_at.is_not(None),
                    model.next_sync_at <= now,
                )
                .with_for_update(skip_locked=True)
            )
            for src in due:
                slot = src.next_sync_at.strftime("%Y%m%d%H%M")
                enqueue(
                    db, src.org_id, kind, payload={"source_id": str(src.id), "mode": "scheduled"},
                    resource_type=rtype, resource_id=src.id, idempotency_key=f"{kind}:{src.id}:{slot}",
                )
                src.next_sync_at = now + timedelta(minutes=src.sync_interval_minutes or 60)
                queued += 1
    if queued:
        log.info("scheduled %s sync job(s)", queued)
    return queued


_last_retention: datetime | None = None


def run_retention(force: bool = False) -> dict[str, int]:
    """Apply per-organisation retention (at most hourly): old conversations and audit events.

    Settings live in ``organizations.settings["retention"]``: ``conversations_days`` and
    ``audit_days`` (``null`` = keep forever). Deleting conversations cascades to messages,
    citations and feedback.
    """
    global _last_retention
    now = datetime.now(UTC)
    if not force and _last_retention and now - _last_retention < timedelta(hours=1):
        return {}
    _last_retention = now
    from sqlalchemy import delete

    from app.models import AuditEvent, Conversation, Organization
    from app.services.activity import audit

    totals = {"conversations": 0, "audit_events": 0}
    with system_session() as db:
        for org in db.scalars(select(Organization)):
            retention = (org.settings or {}).get("retention") or {}
            conv_days = retention.get("conversations_days")
            audit_days = retention.get("audit_days")
            if conv_days:
                res = db.execute(delete(Conversation).where(Conversation.org_id == org.id, Conversation.updated_at < now - timedelta(days=int(conv_days))))
                totals["conversations"] += res.rowcount or 0
            if audit_days:
                res = db.execute(delete(AuditEvent).where(AuditEvent.org_id == org.id, AuditEvent.created_at < now - timedelta(days=int(audit_days))))
                totals["audit_events"] += res.rowcount or 0
            if conv_days or audit_days:
                audit(db, "retention.applied", org_id=org.id, meta={"conversations_days": conv_days, "audit_days": audit_days})
    if any(totals.values()):
        log.info("retention removed %s", totals)
    return totals
