"""Roles, permissions, role assignments and resource-level grants."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TenantScoped, Timestamps, UUIDPk


class Role(UUIDPk, TenantScoped, Timestamps, Base):
    __tablename__ = "roles"
    __table_args__ = (UniqueConstraint("org_id", "key"),)

    key: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # System roles are seeded per organisation and cannot be edited or deleted.
    is_system: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # Rank used to prevent privilege escalation (you may only assign roles at or
    # below your own highest rank). Custom roles get the creator's rank cap.
    rank: Mapped[int] = mapped_column(default=10, nullable=False)


class RolePermission(UUIDPk, TenantScoped, Base):
    __tablename__ = "role_permissions"
    __table_args__ = (UniqueConstraint("role_id", "permission"),)

    role_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("roles.id", ondelete="CASCADE"), index=True
    )
    permission: Mapped[str] = mapped_column(String(64), nullable=False)


class RoleAssignment(UUIDPk, TenantScoped, Base):
    """Assigns a role to a user or a team at organisation or workspace scope."""

    __tablename__ = "role_assignments"
    __table_args__ = (
        CheckConstraint("(user_id IS NULL) <> (team_id IS NULL)", name="one_subject"),
        CheckConstraint(
            "(scope_type = 'org' AND scope_id IS NULL) OR (scope_type = 'workspace' AND scope_id IS NOT NULL)",
            name="scope_shape",
        ),
        Index("ix_role_assignments_subject", "org_id", "user_id", "team_id"),
    )

    role_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("roles.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    team_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE")
    )
    scope_type: Mapped[str] = mapped_column(String(16), default="org", nullable=False)
    scope_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE")
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    granted_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ResourceGrant(UUIDPk, TenantScoped, Base):
    """Resource-level ACL entry (knowledge base, document, connection, agent).

    ``subject_type`` is one of user | team | department | role | workspace | org.
    ``effect=deny`` always wins over any allow for the same resource.
    """

    __tablename__ = "resource_grants"
    __table_args__ = (
        Index("ix_resource_grants_resource", "org_id", "resource_type", "resource_id"),
        Index("ix_resource_grants_subject", "org_id", "subject_type", "subject_id"),
        CheckConstraint("privilege IN ('read','write','manage')", name="privilege_values"),
        CheckConstraint("effect IN ('allow','deny')", name="effect_values"),
    )

    resource_type: Mapped[str] = mapped_column(String(32), nullable=False)
    resource_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    subject_type: Mapped[str] = mapped_column(String(16), nullable=False)
    subject_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    privilege: Mapped[str] = mapped_column(String(16), default="read", nullable=False)
    effect: Mapped[str] = mapped_column(String(8), default="allow", nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Optional narrowing, e.g. {"tables": ["public.orders"]} for database grants.
    constraints: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SupportAccessGrant(UUIDPk, TenantScoped, Base):
    """Time-boxed, owner-approved read access for a platform administrator."""

    __tablename__ = "support_access_grants"

    platform_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    granted_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    reason: Mapped[str] = mapped_column(Text, default="", nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
