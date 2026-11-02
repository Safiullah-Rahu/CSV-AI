"""Workspaces, departments and teams."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.api.deps import OrgContext, get_org_context, requires, requires_any_scope
from app.api.schemas import ORM, Ok
from app.models import (
    Department,
    KnowledgeBase,
    Membership,
    Team,
    TeamMember,
    Workspace,
    WorkspaceMember,
)
from app.utils.errors import Conflict, NotFound, ValidationFailed

router = APIRouter(tags=["structure"])


class NamedIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)


class WorkspaceOut(ORM):
    id: uuid.UUID
    name: str
    description: str
    created_at: datetime
    member_count: int = 0
    kb_count: int = 0
    is_member: bool = False


def _get(ctx: OrgContext, model: Any, obj_id: uuid.UUID) -> Any:
    obj = ctx.db.get(model, obj_id)
    if obj is None or obj.org_id != ctx.org_id:
        raise NotFound(f"{model.__name__} not found.")
    return obj


def _commit_unique(ctx: OrgContext, what: str) -> None:
    try:
        ctx.db.commit()
    except IntegrityError as exc:
        ctx.db.rollback()
        raise Conflict(f"A {what} with this name already exists.") from exc


def _flush_unique(ctx: OrgContext, what: str) -> None:
    try:
        ctx.db.flush()
    except IntegrityError as exc:
        ctx.db.rollback()
        raise Conflict(f"A {what} with this name already exists.") from exc


@router.get("/workspaces", response_model=list[WorkspaceOut])
def list_workspaces(ctx: OrgContext = Depends(requires_any_scope("workspace.view"))) -> list[WorkspaceOut]:
    db = ctx.db
    members = dict(db.execute(select(WorkspaceMember.workspace_id, func.count()).where(WorkspaceMember.org_id == ctx.org_id).group_by(WorkspaceMember.workspace_id)).all())
    kbs = dict(db.execute(select(KnowledgeBase.workspace_id, func.count()).where(KnowledgeBase.org_id == ctx.org_id).group_by(KnowledgeBase.workspace_id)).all())
    out = []
    for ws in db.scalars(select(Workspace).where(Workspace.org_id == ctx.org_id).order_by(Workspace.name)):
        item = WorkspaceOut.model_validate(ws)
        item.member_count = members.get(ws.id, 0)
        item.kb_count = kbs.get(ws.id, 0)
        item.is_member = ws.id in ctx.authz.workspace_ids
        out.append(item)
    return out


@router.post("/workspaces", response_model=WorkspaceOut, status_code=201)
def create_workspace(body: NamedIn, ctx: OrgContext = Depends(requires("workspace.create"))) -> WorkspaceOut:
    ws = Workspace(org_id=ctx.org_id, name=body.name.strip(), description=body.description, created_by=ctx.user_id)
    ctx.db.add(ws)
    _flush_unique(ctx, "workspace")
    ctx.db.add(WorkspaceMember(org_id=ctx.org_id, workspace_id=ws.id, user_id=ctx.user_id))
    ctx.audit("workspace.create", resource_type="workspace", resource_id=ws.id, meta={"name": ws.name})
    _commit_unique(ctx, "workspace")
    return WorkspaceOut.model_validate(ws)


@router.patch("/workspaces/{workspace_id}", response_model=WorkspaceOut)
def update_workspace(workspace_id: uuid.UUID, body: NamedIn, ctx: OrgContext = Depends(get_org_context)) -> WorkspaceOut:
    ctx.require("workspace.edit", workspace_id)
    ws = _get(ctx, Workspace, workspace_id)
    ws.name, ws.description = body.name.strip(), body.description
    ctx.audit("workspace.update", resource_type="workspace", resource_id=ws.id, meta=body.model_dump())
    _commit_unique(ctx, "workspace")
    return WorkspaceOut.model_validate(ws)


@router.delete("/workspaces/{workspace_id}", response_model=Ok)
def delete_workspace(workspace_id: uuid.UUID, ctx: OrgContext = Depends(get_org_context)) -> Ok:
    ctx.require("workspace.delete", workspace_id)
    ws = _get(ctx, Workspace, workspace_id)
    in_use = ctx.db.scalar(select(func.count()).select_from(KnowledgeBase).where(KnowledgeBase.workspace_id == ws.id)) or 0
    if in_use:
        raise ValidationFailed(f"Move or delete the {in_use} knowledge base(s) in this workspace first.")
    ctx.db.delete(ws)
    ctx.audit("workspace.delete", resource_type="workspace", resource_id=workspace_id)
    ctx.db.commit()
    return Ok()


class MemberIdsIn(BaseModel):
    user_ids: list[uuid.UUID]


def _assert_members(ctx: OrgContext, user_ids: list[uuid.UUID]) -> None:
    found = set(ctx.db.scalars(select(Membership.user_id).where(Membership.org_id == ctx.org_id, Membership.user_id.in_(user_ids))))
    missing = set(user_ids) - found
    if missing:
        raise ValidationFailed("Some users are not members of this organization.", details={"user_ids": [str(m) for m in missing]})


@router.get("/workspaces/{workspace_id}/members")
def workspace_members(workspace_id: uuid.UUID, ctx: OrgContext = Depends(requires("workspace.view"))) -> list[str]:
    _get(ctx, Workspace, workspace_id)
    return [str(u) for u in ctx.db.scalars(select(WorkspaceMember.user_id).where(WorkspaceMember.workspace_id == workspace_id))]


@router.post("/workspaces/{workspace_id}/members", response_model=Ok)
def add_workspace_members(workspace_id: uuid.UUID, body: MemberIdsIn, ctx: OrgContext = Depends(get_org_context)) -> Ok:
    ctx.require("workspace.members", workspace_id)
    _get(ctx, Workspace, workspace_id)
    _assert_members(ctx, body.user_ids)
    existing = set(ctx.db.scalars(select(WorkspaceMember.user_id).where(WorkspaceMember.workspace_id == workspace_id)))
    for uid in set(body.user_ids) - existing:
        ctx.db.add(WorkspaceMember(org_id=ctx.org_id, workspace_id=workspace_id, user_id=uid))
    ctx.audit("workspace.members_add", resource_type="workspace", resource_id=workspace_id, meta={"user_ids": [str(u) for u in body.user_ids]})
    ctx.db.commit()
    return Ok()


@router.delete("/workspaces/{workspace_id}/members/{user_id}", response_model=Ok)
def remove_workspace_member(workspace_id: uuid.UUID, user_id: uuid.UUID, ctx: OrgContext = Depends(get_org_context)) -> Ok:
    ctx.require("workspace.members", workspace_id)
    row = ctx.db.scalar(select(WorkspaceMember).where(WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == user_id, WorkspaceMember.org_id == ctx.org_id))
    if row:
        ctx.db.delete(row)
        ctx.audit("workspace.members_remove", resource_type="workspace", resource_id=workspace_id, meta={"user_id": str(user_id)})
        ctx.db.commit()
    return Ok()


# ------------------------------------------------------------------------ departments


class DepartmentOut(ORM):
    id: uuid.UUID
    name: str
    description: str
    created_at: datetime
    member_count: int = 0


@router.get("/departments", response_model=list[DepartmentOut])
def list_departments(ctx: OrgContext = Depends(requires("org.view"))) -> list[DepartmentOut]:
    counts = dict(ctx.db.execute(select(Membership.department_id, func.count()).where(Membership.org_id == ctx.org_id).group_by(Membership.department_id)).all())
    out = []
    for d in ctx.db.scalars(select(Department).where(Department.org_id == ctx.org_id).order_by(Department.name)):
        item = DepartmentOut.model_validate(d)
        item.member_count = counts.get(d.id, 0)
        out.append(item)
    return out


@router.post("/departments", response_model=DepartmentOut, status_code=201)
def create_department(body: NamedIn, ctx: OrgContext = Depends(requires("org.users"))) -> DepartmentOut:
    d = Department(org_id=ctx.org_id, name=body.name.strip(), description=body.description)
    ctx.db.add(d)
    _flush_unique(ctx, "department")
    ctx.audit("department.create", resource_type="department", resource_id=d.id, meta={"name": d.name})
    _commit_unique(ctx, "department")
    return DepartmentOut.model_validate(d)


@router.patch("/departments/{department_id}", response_model=DepartmentOut)
def update_department(department_id: uuid.UUID, body: NamedIn, ctx: OrgContext = Depends(requires("org.users"))) -> DepartmentOut:
    d = _get(ctx, Department, department_id)
    d.name, d.description = body.name.strip(), body.description
    ctx.audit("department.update", resource_type="department", resource_id=d.id)
    _commit_unique(ctx, "department")
    return DepartmentOut.model_validate(d)


@router.delete("/departments/{department_id}", response_model=Ok)
def delete_department(department_id: uuid.UUID, ctx: OrgContext = Depends(requires("org.users"))) -> Ok:
    d = _get(ctx, Department, department_id)
    ctx.db.delete(d)
    ctx.audit("department.delete", resource_type="department", resource_id=department_id)
    ctx.db.commit()
    return Ok()


# ------------------------------------------------------------------------------- teams


class TeamIn(NamedIn):
    department_id: uuid.UUID | None = None
    workspace_id: uuid.UUID | None = None


class TeamOut(ORM):
    id: uuid.UUID
    name: str
    description: str
    department_id: uuid.UUID | None
    workspace_id: uuid.UUID | None
    created_at: datetime
    member_ids: list[uuid.UUID] = []


@router.get("/teams", response_model=list[TeamOut])
def list_teams(ctx: OrgContext = Depends(requires("org.view"))) -> list[TeamOut]:
    members: dict[uuid.UUID, list[uuid.UUID]] = {}
    for tm in ctx.db.scalars(select(TeamMember).where(TeamMember.org_id == ctx.org_id)):
        members.setdefault(tm.team_id, []).append(tm.user_id)
    out = []
    for t in ctx.db.scalars(select(Team).where(Team.org_id == ctx.org_id).order_by(Team.name)):
        item = TeamOut.model_validate(t)
        item.member_ids = members.get(t.id, [])
        out.append(item)
    return out


@router.post("/teams", response_model=TeamOut, status_code=201)
def create_team(body: TeamIn, ctx: OrgContext = Depends(requires("org.users"))) -> TeamOut:
    if body.department_id:
        _get(ctx, Department, body.department_id)
    if body.workspace_id:
        _get(ctx, Workspace, body.workspace_id)
    t = Team(org_id=ctx.org_id, name=body.name.strip(), description=body.description, department_id=body.department_id, workspace_id=body.workspace_id)
    ctx.db.add(t)
    _flush_unique(ctx, "team")
    ctx.audit("team.create", resource_type="team", resource_id=t.id, meta={"name": t.name})
    _commit_unique(ctx, "team")
    return TeamOut.model_validate(t)


@router.patch("/teams/{team_id}", response_model=TeamOut)
def update_team(team_id: uuid.UUID, body: TeamIn, ctx: OrgContext = Depends(requires("org.users"))) -> TeamOut:
    t = _get(ctx, Team, team_id)
    t.name, t.description, t.department_id, t.workspace_id = body.name.strip(), body.description, body.department_id, body.workspace_id
    ctx.audit("team.update", resource_type="team", resource_id=t.id)
    _commit_unique(ctx, "team")
    return TeamOut.model_validate(t)


@router.delete("/teams/{team_id}", response_model=Ok)
def delete_team(team_id: uuid.UUID, ctx: OrgContext = Depends(requires("org.users"))) -> Ok:
    t = _get(ctx, Team, team_id)
    ctx.db.delete(t)
    ctx.audit("team.delete", resource_type="team", resource_id=team_id)
    ctx.db.commit()
    return Ok()


@router.put("/teams/{team_id}/members", response_model=Ok)
def set_team_members(team_id: uuid.UUID, body: MemberIdsIn, ctx: OrgContext = Depends(requires("org.users"))) -> Ok:
    _get(ctx, Team, team_id)
    _assert_members(ctx, body.user_ids)
    current = {tm.user_id: tm for tm in ctx.db.scalars(select(TeamMember).where(TeamMember.team_id == team_id))}
    wanted = set(body.user_ids)
    for uid, tm in current.items():
        if uid not in wanted:
            ctx.db.delete(tm)
    for uid in wanted - set(current):
        ctx.db.add(TeamMember(org_id=ctx.org_id, team_id=team_id, user_id=uid))
    ctx.audit("team.members_set", resource_type="team", resource_id=team_id, meta={"user_ids": [str(u) for u in wanted]})
    ctx.db.commit()
    return Ok()
