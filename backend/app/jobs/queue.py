"""PostgreSQL-backed job queue.

Jobs are rows in ``jobs``. Enqueueing is transactional with the caller's
work (a document row and its ingestion job commit together or not at all)
and issues ``NOTIFY rag_jobs`` so idle workers wake immediately.

Workers claim jobs with ``FOR UPDATE SKIP LOCKED`` (safe with many workers),
run handlers inside a *tenant* session for the job's organisation (RLS stays
on in workers), retry failures with exponential backoff and jitter, and move
jobs that exhaust their attempts to ``dead`` (dead-letter) with a
notification. Handlers must be idempotent: a job may run more than once.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import event, select, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import system_session, tenant_session
from app.models import Job
from app.services.activity import emit
from app.utils.logging import get_logger

log = get_logger("jobs")


class PermanentJobError(Exception):
    """Raise from a handler to fail without retrying (bad input, deleted resource)."""


@dataclass
class JobContext:
    job_id: uuid.UUID
    org_id: uuid.UUID
    kind: str
    payload: dict[str, Any]
    created_by: uuid.UUID | None
    attempt: int
    _last_progress: float = field(default=-1.0)

    def progress(self, fraction: float, message: str = "") -> None:
        """Persist progress in a short separate transaction and emit an event."""
        fraction = max(0.0, min(1.0, fraction))
        if abs(fraction - self._last_progress) < 0.02 and fraction < 1.0:
            return
        self._last_progress = fraction
        with tenant_session(self.org_id) as db:
            job = db.get(Job, self.job_id)
            if job is None:
                return
            job.progress = fraction
            job.message = message[:500]
            emit(db, self.org_id, "job.progress", resource_type=job.resource_type, resource_id=job.resource_id,
                 payload={"job_id": str(job.id), "kind": job.kind, "progress": fraction, "message": message})

    def session(self) -> Any:
        return tenant_session(self.org_id, self.created_by)


Handler = Callable[[JobContext], dict[str, Any] | None]
HANDLERS: dict[str, Handler] = {}


def handler(kind: str) -> Callable[[Handler], Handler]:
    def deco(fn: Handler) -> Handler:
        HANDLERS[kind] = fn
        return fn

    return deco


def enqueue(
    db: Session,
    org_id: uuid.UUID,
    kind: str,
    *,
    payload: dict[str, Any] | None = None,
    created_by: uuid.UUID | None = None,
    resource_type: str | None = None,
    resource_id: uuid.UUID | None = None,
    idempotency_key: str | None = None,
    delay_seconds: float = 0,
    max_attempts: int | None = None,
) -> Job:
    if idempotency_key:
        existing = db.scalar(select(Job).where(Job.idempotency_key == idempotency_key))
        if existing is not None:
            return existing
    job = Job(
        org_id=org_id,
        kind=kind,
        payload=_jsonable(payload or {}),
        created_by=created_by,
        resource_type=resource_type,
        resource_id=resource_id,
        idempotency_key=idempotency_key,
        max_attempts=max_attempts or get_settings().job_max_attempts,
        next_run_at=datetime.now(UTC) + timedelta(seconds=delay_seconds) if delay_seconds else None,
    )
    db.add(job)
    db.flush()
    db.execute(text("NOTIFY rag_jobs"))
    db.info.setdefault("jobs_to_dispatch", []).append(job.id)
    return job


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


@event.listens_for(Session, "after_commit")
def _dispatch_after_commit(session: Session) -> None:
    ids: list[uuid.UUID] = session.info.pop("jobs_to_dispatch", [])
    if not ids:
        return
    if get_settings().job_backend == "eager":
        for job_id in ids:
            run_job(job_id, worker_id="eager")


@event.listens_for(Session, "after_rollback")
def _drop_dispatch(session: Session) -> None:
    session.info.pop("jobs_to_dispatch", None)


def _backoff(attempt: int) -> timedelta:
    base = get_settings().job_backoff_base_seconds
    delay = base * (2 ** max(0, attempt - 1))
    return timedelta(seconds=min(delay, 3600) * (0.8 + random.random() * 0.4))


def claim(job_id: uuid.UUID | None, worker_id: str, limit: int = 1) -> list[tuple[uuid.UUID, uuid.UUID, str, dict[str, Any], uuid.UUID | None, int]]:
    """Atomically claim runnable jobs (a specific one, or the oldest ``limit``)."""
    now = datetime.now(UTC)
    claimed = []
    with system_session() as db:
        q = select(Job).with_for_update(skip_locked=True)
        if job_id is not None:
            q = q.where(Job.id == job_id, Job.status.in_(("queued", "failed")))
        else:
            q = (
                q.where(Job.status.in_(("queued", "failed")))
                .where((Job.next_run_at.is_(None)) | (Job.next_run_at <= now))
                .order_by(Job.created_at)
                .limit(limit)
            )
        for job in db.scalars(q):
            job.status = "running"
            job.attempts += 1
            job.started_at = now
            job.locked_by = worker_id
            job.error = None
            claimed.append((job.id, job.org_id, job.kind, dict(job.payload), job.created_by, job.attempts))
    return claimed


def run_job(job_id: uuid.UUID, worker_id: str = "worker") -> None:
    for claimed in claim(job_id, worker_id):
        _execute(*claimed)


def _execute(job_id: uuid.UUID, org_id: uuid.UUID, kind: str, payload: dict[str, Any], created_by: uuid.UUID | None, attempt: int) -> None:
    ctx = JobContext(job_id=job_id, org_id=org_id, kind=kind, payload=payload, created_by=created_by, attempt=attempt)
    fn = HANDLERS.get(kind)
    try:
        if fn is None:
            raise PermanentJobError(f"No handler registered for job kind '{kind}'.")
        result = fn(ctx) or {}
    except Exception as exc:  # noqa: BLE001 — every failure is recorded on the job row
        permanent = isinstance(exc, PermanentJobError)
        log.warning("job %s (%s) attempt %s failed: %s", job_id, kind, attempt, exc, exc_info=not permanent)
        with tenant_session(org_id) as db:
            job = db.get(Job, job_id)
            if job is None:
                return
            job.error = f"{type(exc).__name__}: {exc}"[:4000]
            job.finished_at = datetime.now(UTC)
            job.locked_by = None
            if not permanent and job.attempts < job.max_attempts:
                job.status = "failed"
                job.next_run_at = datetime.now(UTC) + _backoff(job.attempts)
                emit(db, org_id, "job.retry", resource_type=job.resource_type, resource_id=job.resource_id, severity="warning",
                     message=f"{kind} failed (attempt {job.attempts}/{job.max_attempts}); retrying", payload={"job_id": str(job.id), "error": job.error})
            else:
                job.status = "dead"
                emit(db, org_id, "job.dead", resource_type=job.resource_type, resource_id=job.resource_id, severity="error",
                     message=f"{kind} failed permanently", payload={"job_id": str(job.id), "error": job.error})
                _notify_failure(db, job)
        _on_dead_hook(kind, payload, org_id, job_id)
        return
    with tenant_session(org_id) as db:
        job = db.get(Job, job_id)
        if job is None:
            return
        job.status = "succeeded"
        job.progress = 1.0
        job.result = _jsonable(result)
        job.finished_at = datetime.now(UTC)
        job.locked_by = None


DEAD_HOOKS: dict[str, Callable[[dict[str, Any], uuid.UUID, uuid.UUID], None]] = {}


def _on_dead_hook(kind: str, payload: dict[str, Any], org_id: uuid.UUID, job_id: uuid.UUID) -> None:
    hook = DEAD_HOOKS.get(kind)
    if hook is None:
        return
    with tenant_session(org_id) as db:
        job = db.get(Job, job_id)
        if job is not None and job.status == "dead":
            try:
                hook(payload, org_id, job_id)
            except Exception:  # noqa: BLE001
                log.exception("dead-letter hook failed for %s", kind)


def _notify_failure(db: Session, job: Job) -> None:
    from app.models import Notification

    if job.created_by:
        db.add(
            Notification(
                org_id=job.org_id,
                user_id=job.created_by,
                kind="job_failed",
                title=f"Background job failed: {job.kind.replace('_', ' ')}",
                body=(job.error or "")[:500],
                severity="error",
                link="/documents" if job.kind == "ingest_document" else None,
            )
        )


def requeue_stale(lease_minutes: int | None = None) -> int:
    """Return orphaned 'running' jobs (worker crashed) to the queue."""
    lease = timedelta(minutes=lease_minutes or get_settings().job_lease_minutes)
    cutoff = datetime.now(UTC) - lease
    with system_session() as db:
        stale = list(db.scalars(select(Job).where(Job.status == "running", Job.started_at < cutoff).with_for_update(skip_locked=True)))
        for job in stale:
            job.status = "failed" if job.attempts < job.max_attempts else "dead"
            job.error = "Worker lease expired"
            job.locked_by = None
            job.next_run_at = datetime.now(UTC)
        return len(stale)


def cancel(db: Session, job: Job) -> None:
    if job.status in ("queued", "failed"):
        job.status = "cancelled"
        job.finished_at = datetime.now(UTC)
