"""Record activity events and audit entries.

``emit`` writes an :class:`ActivityEvent` in the caller's transaction and
queues it for publication once that transaction commits. ``audit`` writes an
append-only :class:`AuditEvent` with request metadata from context vars.
Metadata is sanitised so secrets never land in either table.
"""

from __future__ import annotations

import contextvars
import uuid
from typing import Any

from sqlalchemy import event, insert
from sqlalchemy.orm import Session

from app.db.base import utcnow
from app.events.bus import publish
from app.models import ActivityEvent, AuditEvent
from app.utils.logging import get_logger, redact, request_id_var

log = get_logger("activity")

client_ip_var: contextvars.ContextVar[str] = contextvars.ContextVar("client_ip", default="")
user_agent_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_agent", default="")

_SENSITIVE_KEYS = ("password", "secret", "token", "api_key", "apikey", "credential", "private_key")

# Event types that may be delivered to customer webhooks.
WEBHOOK_EVENTS = (
    "connection.created", "connection.updated", "connection.failed", "connection.restored", "sync.started", "sync.completed",
    "sync.failed", "document.ingested", "document.failed", "document.deleted", "index.updated", "pipeline.started",
    "pipeline.completed", "kb.created", "kb.deleted", "schema.changed", "job.dead",
)


def sanitize(meta: dict[str, Any] | None) -> dict[str, Any]:
    if not meta:
        return {}
    clean: dict[str, Any] = {}
    for key, value in meta.items():
        if any(s in key.lower() for s in _SENSITIVE_KEYS):
            clean[key] = "***"
        elif isinstance(value, dict):
            clean[key] = sanitize(value)
        elif isinstance(value, str):
            clean[key] = redact(value)[:2000]
        elif isinstance(value, uuid.UUID):
            clean[key] = str(value)
        else:
            clean[key] = value
    return clean


def emit(
    session: Session,
    org_id: uuid.UUID,
    type_: str,
    *,
    resource_type: str | None = None,
    resource_id: uuid.UUID | None = None,
    actor_user_id: uuid.UUID | None = None,
    message: str = "",
    severity: str = "info",
    payload: dict[str, Any] | None = None,
) -> ActivityEvent:
    ev = ActivityEvent(
        org_id=org_id,
        type=type_,
        resource_type=resource_type,
        resource_id=resource_id,
        actor_user_id=actor_user_id,
        message=message[:1000],
        severity=severity,
        payload=sanitize(payload),
        created_at=utcnow(),
    )
    session.add(ev)
    session.info.setdefault("pending_events", []).append(ev)
    if type_ in WEBHOOK_EVENTS:
        _queue_webhooks(session, org_id, type_, ev)
    return ev


def _queue_webhooks(session: Session, org_id: uuid.UUID, type_: str, ev: ActivityEvent) -> None:
    from sqlalchemy import select

    from app.jobs.queue import enqueue
    from app.models import Webhook, WebhookDelivery

    cache_key = f"webhooks:{org_id}"
    hooks = session.info.get(cache_key)
    if hooks is None:
        hooks = [(h.id, set(h.events)) for h in session.scalars(select(Webhook).where(Webhook.org_id == org_id, Webhook.enabled.is_(True)))]
        session.info[cache_key] = hooks
    for hook_id, events in hooks:
        if type_ not in events:
            continue
        delivery = WebhookDelivery(org_id=org_id, webhook_id=hook_id, event_type=type_, payload={"resource_type": ev.resource_type, "resource_id": str(ev.resource_id) if ev.resource_id else None, "message": ev.message, "severity": ev.severity, "payload": ev.payload})
        session.add(delivery)
        session.flush()
        enqueue(session, org_id, "deliver_webhook", payload={"delivery_id": str(delivery.id)}, resource_type="webhook", resource_id=hook_id, max_attempts=6)


def event_to_dict(ev: ActivityEvent) -> dict[str, Any]:
    return {
        "id": ev.id,
        "type": ev.type,
        "resource_type": ev.resource_type,
        "resource_id": str(ev.resource_id) if ev.resource_id else None,
        "actor_user_id": str(ev.actor_user_id) if ev.actor_user_id else None,
        "severity": ev.severity,
        "message": ev.message,
        "payload": ev.payload,
        "created_at": ev.created_at.isoformat() if ev.created_at else None,
    }


@event.listens_for(Session, "after_commit")
def _publish_pending(session: Session) -> None:
    pending: list[ActivityEvent] = session.info.pop("pending_events", [])
    for ev in pending:
        try:
            publish(ev.org_id, event_to_dict(ev))
        except Exception:  # noqa: BLE001
            log.exception("event publish failed")


@event.listens_for(Session, "after_rollback")
def _drop_pending(session: Session) -> None:
    session.info.pop("pending_events", None)


def audit(
    session: Session,
    action: str,
    *,
    org_id: uuid.UUID | None = None,
    actor_user_id: uuid.UUID | None = None,
    actor_email: str | None = None,
    actor_api_key_id: uuid.UUID | None = None,
    resource_type: str | None = None,
    resource_id: Any = None,
    outcome: str = "success",
    meta: dict[str, Any] | None = None,
) -> None:
    # Core INSERT without RETURNING: platform-level rows (org_id NULL) are write-only for
    # request contexts under RLS, and RETURNING would require them to be readable.
    session.execute(
        insert(AuditEvent.__table__).values(
            org_id=org_id,
            actor_user_id=actor_user_id,
            actor_email=actor_email,
            actor_api_key_id=actor_api_key_id,
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id is not None else None,
            outcome=outcome,
            ip=client_ip_var.get() or None,
            user_agent=(user_agent_var.get() or "")[:400] or None,
            request_id=request_id_var.get() if request_id_var.get() != "-" else None,
            metadata=sanitize(meta),
        )
    )
