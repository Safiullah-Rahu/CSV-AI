"""Credentials, database connections, integrations, sync runs, jobs, events."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TenantScoped, Timestamps, UUIDPk


class Credential(UUIDPk, TenantScoped, Base):
    """Encrypted secret material (Fernet). Never serialised to API responses."""

    __tablename__ = "credentials"

    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    ciphertext: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    rotated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DatabaseConnection(UUIDPk, TenantScoped, Timestamps, Base):
    __tablename__ = "database_connections"

    name: Mapped[str] = mapped_column(String(160), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    host: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    port: Mapped[int | None] = mapped_column(Integer)
    database: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    username: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    credential_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("credentials.id", ondelete="SET NULL")
    )
    ssl_mode: Mapped[str] = mapped_column(String(16), default="prefer", nullable=False)
    # Non-secret options (ssh tunnel host/user, warehouse, project id, ...).
    options: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    # live | ingest | hybrid
    access_mode: Mapped[str] = mapped_column(String(16), default="live", nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # unknown | healthy | failed | disabled | syncing
    status: Mapped[str] = mapped_column(String(16), default="unknown", nullable=False)
    last_tested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    schema_cache: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    schema_refreshed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # {"tables": {"schema.table": {"columns": [...], "excluded": [...], "sensitive": [...],
    #   "row_filter": "...", "incremental_column": "...", "primary_key": [...]}}}
    selection: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    # {"max_rows": 500, "timeout_s": 15, "max_joins": 4, "approval_tables": [...]}
    query_limits: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    sync_interval_minutes: Mapped[int | None] = mapped_column(Integer)
    sync_paused: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    next_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    kb_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="SET NULL")
    )
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="SET NULL")
    )
    owner_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))


class Integration(UUIDPk, TenantScoped, Timestamps, Base):
    __tablename__ = "integrations"

    name: Mapped[str] = mapped_column(String(160), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    credential_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("credentials.id", ondelete="SET NULL")
    )
    # pending | connected | failed | disabled | syncing
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    kb_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="SET NULL")
    )
    sync_interval_minutes: Mapped[int | None] = mapped_column(Integer)
    sync_paused: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    next_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    cursor: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="SET NULL")
    )
    owner_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))


class SyncRun(UUIDPk, TenantScoped, Base):
    __tablename__ = "sync_runs"
    __table_args__ = (Index("ix_sync_runs_source", "org_id", "source_type", "source_id"),)

    source_type: Mapped[str] = mapped_column(String(16), nullable=False)  # database | integration
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    mode: Mapped[str] = mapped_column(String(16), default="full", nullable=False)
    # running | succeeded | failed
    status: Mapped[str] = mapped_column(String(16), default="running", nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    items_seen: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    items_added: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    items_updated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    items_deleted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    items_failed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)


class HealthCheck(UUIDPk, TenantScoped, Base):
    __tablename__ = "health_checks"
    __table_args__ = (Index("ix_health_checks_source", "org_id", "source_type", "source_id"),)

    source_type: Mapped[str] = mapped_column(String(16), nullable=False)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    ok: Mapped[bool] = mapped_column(Boolean, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Job(UUIDPk, TenantScoped, Base):
    """Persisted background job with retry / backoff / dead-letter semantics."""

    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_status_next", "status", "next_run_at"),
        Index("ix_jobs_resource", "org_id", "resource_type", "resource_id"),
    )

    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    # queued | running | succeeded | failed (will retry) | dead | cancelled
    status: Mapped[str] = mapped_column(String(16), default="queued", nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=4, nullable=False)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    idempotency_key: Mapped[str | None] = mapped_column(String(200), unique=True)
    resource_type: Mapped[str | None] = mapped_column(String(32))
    resource_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    locked_by: Mapped[str | None] = mapped_column(String(120))


class ActivityEvent(TenantScoped, Base):
    """Organisation-scoped event stream (Connections map feed, notifications)."""

    __tablename__ = "activity_events"
    __table_args__ = (Index("ix_activity_events_org_id_id", "org_id", "id"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    type: Mapped[str] = mapped_column(String(48), nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(32))
    resource_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    severity: Mapped[str] = mapped_column(String(12), default="info", nullable=False)
    message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SqlQueryLog(UUIDPk, TenantScoped, Base):
    """Every Text-to-SQL / SQL request, including rejected and pending ones."""

    __tablename__ = "sql_query_logs"

    source_type: Mapped[str] = mapped_column(String(16), nullable=False)  # database | dataset
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    question: Mapped[str] = mapped_column(Text, default="", nullable=False)
    generated_sql: Mapped[str] = mapped_column(Text, default="", nullable=False)
    executed_sql: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # executed | rejected | pending_approval | approved | denied | failed
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    row_count: Mapped[int | None] = mapped_column(Integer)
    duration_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    reasons: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    decided_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
