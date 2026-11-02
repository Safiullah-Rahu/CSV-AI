"""API keys, audit, usage, plans, subscriptions, webhooks, notifications, approvals."""

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
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TenantScoped, UUIDPk


class ApiKey(UUIDPk, TenantScoped, Base):
    __tablename__ = "api_keys"

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    prefix: Mapped[str] = mapped_column(String(24), unique=True, nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    scopes: Mapped[list[str]] = mapped_column(ARRAY(String(64)), default=list, nullable=False)
    created_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    rate_limit_per_minute: Mapped[int | None] = mapped_column(Integer)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditEvent(Base):
    """Append-only audit trail. ``org_id`` is NULL for platform-level events.

    A trigger created by the migrations rejects UPDATE/DELETE unless the
    retention purge runs with ``app.rls_bypass = on``.
    """

    __tablename__ = "audit_events"
    __tenant_scoped__ = True
    __table_args__ = (
        Index("ix_audit_events_org_created", "org_id", "created_at"),
        # Inserts must not use RETURNING (see app.services.activity.audit).
        {"implicit_returning": False},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    actor_api_key_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    actor_email: Mapped[str | None] = mapped_column(String(320))
    action: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    resource_type: Mapped[str | None] = mapped_column(String(40))
    resource_id: Mapped[str | None] = mapped_column(String(80))
    # success | denied | failure
    outcome: Mapped[str] = mapped_column(String(12), default="success", nullable=False)
    ip: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(400))
    request_id: Mapped[str | None] = mapped_column(String(64))
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UsageRecord(UUIDPk, TenantScoped, Base):
    __tablename__ = "usage_records"
    __table_args__ = (Index("ix_usage_records_org_created", "org_id", "created_at"),)

    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    api_key_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    # chat | embedding | retrieval | sql | ingest | api | agent
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    provider: Mapped[str | None] = mapped_column(String(40))
    model: Mapped[str | None] = mapped_column(String(160))
    kb_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    tokens_in: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Numeric(12, 6), default=0, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Plan(Base):
    __tablename__ = "plans"

    key: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    # {"users": 5, "knowledge_bases": 3, "storage_mb": 1024, "documents": 500,
    #  "monthly_tokens": 1000000, "api_calls_per_day": 1000, "connected_sources": 2}
    limits: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    features: Mapped[list[str]] = mapped_column(ARRAY(String(48)), default=list, nullable=False)
    price_monthly_usd: Mapped[float | None] = mapped_column(Numeric(10, 2))
    is_public: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    sort: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class Subscription(UUIDPk, TenantScoped, Base):
    __tablename__ = "subscriptions"

    plan_key: Mapped[str] = mapped_column(String(32), ForeignKey("plans.key"), nullable=False)
    # trialing | active | past_due | cancelled
    status: Mapped[str] = mapped_column(String(16), default="active", nullable=False)
    provider: Mapped[str] = mapped_column(String(16), default="none", nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(120))
    trial_ends_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    limit_overrides: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Webhook(UUIDPk, TenantScoped, Base):
    __tablename__ = "webhooks"

    url: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    credential_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("credentials.id", ondelete="SET NULL")
    )
    events: Mapped[list[str]] = mapped_column(ARRAY(String(48)), default=list, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    failure_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_delivery_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class WebhookDelivery(UUIDPk, TenantScoped, Base):
    __tablename__ = "webhook_deliveries"

    webhook_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("webhooks.id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(48), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    # pending | delivered | failed
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    response_status: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Notification(UUIDPk, TenantScoped, Base):
    __tablename__ = "notifications"
    __table_args__ = (Index("ix_notifications_user", "org_id", "user_id", "read_at"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    body: Mapped[str] = mapped_column(Text, default="", nullable=False)
    link: Mapped[str | None] = mapped_column(String(500))
    severity: Mapped[str] = mapped_column(String(12), default="info", nullable=False)
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Approval(UUIDPk, TenantScoped, Base):
    """Human-in-the-loop approval (sensitive SQL, agent tools)."""

    __tablename__ = "approvals"

    kind: Mapped[str] = mapped_column(String(24), nullable=False)  # sql_query | agent_tool
    requested_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    # pending | approved | denied | expired
    status: Mapped[str] = mapped_column(String(12), default="pending", nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(32))
    resource_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    decided_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reason: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
