"""AI provider configurations and agents."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TenantScoped, Timestamps, UUIDPk


class AIProviderConfig(UUIDPk, TenantScoped, Timestamps, Base):
    __tablename__ = "ai_providers"
    __table_args__ = (UniqueConstraint("org_id", "name"),)

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    base_url: Mapped[str | None] = mapped_column(Text)
    credential_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("credentials.id", ondelete="SET NULL")
    )
    # [{"id": "...", "type": "chat|embedding", "context_window": 128000,
    #   "input_cost_per_1k": 0.00015, "output_cost_per_1k": 0.0006, "dim": 1536}]
    models: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    default_model: Mapped[str | None] = mapped_column(String(160))
    default_embedding_model: Mapped[str | None] = mapped_column(String(160))
    # temperature, max_tokens, timeout_s, max_retries, rpm_limit, monthly_token_limit, region, ...
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # unknown | healthy | failed
    status: Mapped[str] = mapped_column(String(16), default="unknown", nullable=False)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)


class Agent(UUIDPk, TenantScoped, Timestamps, Base):
    __tablename__ = "agents"
    __table_args__ = (UniqueConstraint("org_id", "name"),)

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    instructions: Mapped[str] = mapped_column(Text, default="", nullable=False)
    kb_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), default=list, nullable=False
    )
    connection_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), default=list, nullable=False
    )
    provider_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ai_providers.id", ondelete="SET NULL")
    )
    model: Mapped[str | None] = mapped_column(String(160))
    tools: Mapped[list[str]] = mapped_column(ARRAY(String(40)), default=list, nullable=False)
    approval_tools: Mapped[list[str]] = mapped_column(ARRAY(String(40)), default=list, nullable=False)
    retrieval_config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    # {"max_steps": 6, "max_tokens_per_day": 200000}
    limits: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="SET NULL")
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
