"""Permission catalogue, roles, role assignments and resource grants."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.api.deps import OrgContext, get_org_context, requires
from app.api.schemas import ORM, Ok
from app.models import (
    Agent,
    DatabaseConnection,
    Document,
    Integration,
    KnowledgeBase,
    ResourceGrant,
    Role,
    RoleAssignment,
)
from app.rbac.authorizer import MANAGE
from app.rbac.catalog import PERMISSIONS
from app.services import org_service
from app.utils.errors import NotFound, PermissionDenied, ValidationFailed

router = APIRouter(tags=["access control"])


@router.get("/permissions")
def permission_catalog(ctx: OrgContext = Depends(get_org_context)) -> list[dict[str, str]]:
    return [{"key": k, "category": c, "description": d} for k, (c, d) in PERMISSIONS.items()]


@router.get("/me/permissions")
def my_permissions(ctx: OrgContext = Depends(get_org_context)) -> dict[str, Any]:
    return ctx.authz.effective_permissions()


class RoleOut(ORM):
    id: uuid.UUID
    key: str
    name: str
    description: str
    is_system: bool
    rank: int
    permissions: list[str] = []
    assignment_count: int = 0


@router.get("/roles", response_model=list[RoleOut])
def list_roles(ctx: OrgContext = Depends(requires("org.view"))) -> list[RoleOut]:
    roles = list(ctx.db.scalars(select(Role).where(Role.org_id == ctx.org_id).order_by(Role.rank.desc(), Role.name)))
    perms = org_service.role_permissions(ctx.db, [r.id for r in roles])
    counts = dict(ctx.db.execute(select(RoleAssignment.role_id, func.count()).where(RoleAssignment.org_id == ctx.org_id).group_by(RoleAssignment.role_id)).all())
    out = []
    for r in roles:
        item = RoleOut.model_validate(r)
        item.permissions = sorted(perms.get(r.id, []))
        item.assignment_count = counts.get(r.id, 0)
        out.append(item)
    return out


class RoleIn(BaseModel):
    key: str = Field(default="", max_length=64)
    name: str = Field(min_length=2, max_length=120)
    description: str = Field(default="", max_length=2000)
    permissions: list[str]


@router.post("/roles", response_model=RoleOut, status_code=201)
def create_role(body: RoleIn, ctx: OrgContext = Depends(requires("admin.roles"))) -> RoleOut:
    role = org_service.create_custom_role(ctx.db, ctx.authz, key=body.key, name=body.name, description=body.description, permissions=body.permissions)
    ctx.db.commit()
    item = RoleOut.model_validate(role)
    item.permissions = sorted(body.permissions)
    return item


class RoleUpdateIn(BaseModel):
    name: str | None = None
    description: str | None = None
    permissions: list[str] | None = None


@router.patch("/roles/{role_id}", response_model=RoleOut)
def update_role(role_id: uuid.UUID, body: RoleUpdateIn, ctx: OrgContext = Depends(requires("admin.roles"))) -> RoleOut:
    role = org_service.get_role(ctx.db, ctx.org_id, role_id)
    org_service.update_custom_role(ctx.db, ctx.authz, role, name=body.name, description=body.description, permissions=body.permissions)
    ctx.db.commit()
    item = RoleOut.model_validate(role)
    item.permissions = sorted(org_service.role_permissions(ctx.db, [role.id])[role.id])
    return item


@router.delete("/roles/{role_id}", response_model=Ok)
def delete_role(role_id: uuid.UUID, ctx: OrgContext = Depends(requires("admin.roles"))) -> Ok:
    role = org_service.get_role(ctx.db, ctx.org_id, role_id)
    if role.is_system:
        raise PermissionDenied("System roles cannot be deleted.")
    in_use = ctx.db.scalar(select(func.count()).select_from(RoleAssignment).where(RoleAssignment.role_id == role.id)) or 0
    if in_use:
        raise ValidationFailed(f"Role is assigned {in_use} time(s). Remove the assignments first.")
    ctx.db.delete(role)
    ctx.audit("role.delete", resource_type="role", resource_id=role_id)
    ctx.db.commit()
    return Ok()


class AssignmentIn(BaseModel):
    role_id: uuid.UUID
    user_id: uuid.UUID | None = None
    team_id: uuid.UUID | None = None
    workspace_id: uuid.UUID | None = None
    expires_at: datetime | None = None


class AssignmentOut(ORM):
    id: uuid.UUID
    role_id: uuid.UUID
    user_id: uuid.UUID | None
    team_id: uuid.UUID | None
    scope_type: str
    scope_id: uuid.UUID | None
    expires_at: datetime | None
    created_at: datetime


@router.get("/role-assignments", response_model=list[AssignmentOut])
def list_assignments(user_id: uuid.UUID | None = None, ctx: OrgContext = Depends(requires("org.view"))) -> list[AssignmentOut]:
    q = select(RoleAssignment).where(RoleAssignment.org_id == ctx.org_id)
    if user_id:
        q = q.where(RoleAssignment.user_id == user_id)
    return [AssignmentOut.model_validate(a) for a in ctx.db.scalars(q.order_by(RoleAssignment.created_at))]


@router.post("/role-assignments", response_model=AssignmentOut, status_code=201)
def create_assignment(body: AssignmentIn, ctx: OrgContext = Depends(requires("admin.permissions"))) -> AssignmentOut:
    role = org_service.get_role(ctx.db, ctx.org_id, body.role_id)
    a = org_service.assign_role(ctx.db, ctx.authz, role=role, user_id=body.user_id, team_id=body.team_id, workspace_id=body.workspace_id, expires_at=body.expires_at)
    ctx.db.commit()
    return AssignmentOut.model_validate(a)


@router.delete("/role-assignments/{assignment_id}", response_model=Ok)
def delete_assignment(assignment_id: uuid.UUID, ctx: OrgContext = Depends(requires("admin.permissions"))) -> Ok:
    a = ctx.db.get(RoleAssignment, assignment_id)
    if a is None or a.org_id != ctx.org_id:
        raise NotFound("Assignment not found.")
    role = org_service.get_role(ctx.db, ctx.org_id, a.role_id)
    if not ctx.authz.can_assign_role(role):
        raise PermissionDenied("You cannot remove a role above your own.")
    if role.key == "owner" and a.user_id and org_service.owner_count(ctx.db, ctx.org_id) <= 1:
        raise ValidationFailed("Cannot remove the last owner.")
    ctx.db.delete(a)
    ctx.audit("role.unassign", resource_type="role_assignment", resource_id=assignment_id, meta={"role": role.key, "user_id": a.user_id, "team_id": a.team_id})
    ctx.db.commit()
    return Ok()


# ---------------------------------------------------------------------------- grants

_RESOURCE_MODELS = {
    "knowledge_base": KnowledgeBase,
    "document": Document,
    "db_connection": DatabaseConnection,
    "integration": Integration,
    "agent": Agent,
}


def _authorize_grant_management(ctx: OrgContext, resource_type: str, resource_id: uuid.UUID) -> None:
    model = _RESOURCE_MODELS.get(resource_type)
    if model is None:
        raise ValidationFailed("Unsupported resource type.")
    obj = ctx.db.get(model, resource_id)
    if obj is None or obj.org_id != ctx.org_id:
        raise NotFound("Resource not found.")
    authz = ctx.authz
    if resource_type == "knowledge_base":
        if not (authz.has("kb.permissions", obj.workspace_id) and authz.kb_level(obj) >= MANAGE) and not authz.has("admin.permissions"):
            raise PermissionDenied("You cannot manage permissions on this knowledge base.")
    elif resource_type == "document":
        kb = ctx.db.get(KnowledgeBase, obj.kb_id)
        if not (authz.has("kb.permissions", kb.workspace_id if kb else None) and kb and authz.kb_level(kb) >= MANAGE) and not authz.has("admin.permissions"):
            raise PermissionDenied("You cannot manage permissions on this document.")
    elif resource_type == "db_connection":
        if not (authz.has("db.credentials") or authz.has("admin.permissions")):
            raise PermissionDenied("You cannot manage permissions on this connection.")
    elif resource_type == "integration":
        ctx.require("org.integrations")
    elif resource_type == "agent":
        ctx.require("agents.manage")


class GrantIn(BaseModel):
    resource_type: str
    resource_id: uuid.UUID
    subject_type: str
    subject_id: uuid.UUID | None = None
    privilege: str = Field(default="read", pattern="^(read|write|manage)$")
    effect: str = Field(default="allow", pattern="^(allow|deny)$")
    expires_at: datetime | None = None
    constraints: dict[str, Any] = {}
    note: str = Field(default="", max_length=1000)


class GrantOut(ORM):
    id: uuid.UUID
    resource_type: str
    resource_id: uuid.UUID
    subject_type: str
    subject_id: uuid.UUID | None
    privilege: str
    effect: str
    expires_at: datetime | None
    constraints: dict[str, Any]
    note: str
    created_at: datetime


@router.get("/grants", response_model=list[GrantOut])
def list_grants(resource_type: str, resource_id: uuid.UUID, ctx: OrgContext = Depends(get_org_context)) -> list[GrantOut]:
    _authorize_grant_management(ctx, resource_type, resource_id)
    rows = ctx.db.scalars(
        select(ResourceGrant).where(ResourceGrant.org_id == ctx.org_id, ResourceGrant.resource_type == resource_type, ResourceGrant.resource_id == resource_id).order_by(ResourceGrant.created_at)
    )
    return [GrantOut.model_validate(g) for g in rows]


@router.post("/grants", response_model=GrantOut, status_code=201)
def create_grant(body: GrantIn, ctx: OrgContext = Depends(get_org_context)) -> GrantOut:
    _authorize_grant_management(ctx, body.resource_type, body.resource_id)
    g = org_service.create_grant(
        ctx.db, ctx.authz, resource_type=body.resource_type, resource_id=body.resource_id, subject_type=body.subject_type,
        subject_id=body.subject_id, privilege=body.privilege, effect=body.effect, expires_at=body.expires_at,
        constraints=body.constraints, note=body.note,
    )
    ctx.db.commit()
    return GrantOut.model_validate(g)


@router.delete("/grants/{grant_id}", response_model=Ok)
def delete_grant(grant_id: uuid.UUID, ctx: OrgContext = Depends(get_org_context)) -> Ok:
    g = ctx.db.get(ResourceGrant, grant_id)
    if g is None or g.org_id != ctx.org_id:
        raise NotFound("Grant not found.")
    _authorize_grant_management(ctx, g.resource_type, g.resource_id)
    ctx.db.delete(g)
    ctx.audit("grant.delete", resource_type=g.resource_type, resource_id=g.resource_id, meta={"subject_type": g.subject_type, "subject_id": g.subject_id, "effect": g.effect})
    ctx.db.commit()
    return Ok()
