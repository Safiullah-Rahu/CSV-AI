"""Organisations, members and invitations."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.api.deps import OrgContext, client_meta, get_db, get_org_context, get_principal, requires
from app.api.schemas import ORM, Ok, page
from app.api.v1.auth import TokenOut, _token_out
from app.db.session import system_session
from app.models import (
    Department,
    Invitation,
    Membership,
    Organization,
    Role,
    RoleAssignment,
    Subscription,
    TeamMember,
    User,
)
from app.rbac.authorizer import Principal
from app.services import auth_service, org_service
from app.utils.errors import AuthenticationRequired, NotFound, PermissionDenied, ValidationFailed

router = APIRouter(tags=["organizations"])


class OrgOut(ORM):
    id: uuid.UUID
    name: str
    slug: str
    status: str
    settings: dict[str, Any]
    created_at: datetime


class OrgCreateIn(BaseModel):
    name: str = Field(min_length=2, max_length=200)


class CurrentOrgOut(BaseModel):
    organization: OrgOut
    permissions: dict[str, Any]
    plan: str | None
    member_count: int


@router.post("/orgs", response_model=OrgOut, status_code=201)
def create_org(body: OrgCreateIn, principal: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> OrgOut:
    user = db.get(User, principal.user_id)
    assert user is not None
    org = org_service.create_organization(db, name=body.name, owner=user)
    db.commit()
    return OrgOut.model_validate(org)


@router.get("/orgs/current", response_model=CurrentOrgOut)
def current_org(ctx: OrgContext = Depends(get_org_context)) -> CurrentOrgOut:
    sub = ctx.db.scalar(select(Subscription).where(Subscription.org_id == ctx.org_id))
    count = ctx.db.scalar(select(func.count()).select_from(Membership).where(Membership.org_id == ctx.org_id, Membership.status == "active")) or 0
    return CurrentOrgOut(
        organization=OrgOut.model_validate(ctx.org),
        permissions=ctx.authz.effective_permissions(),
        plan=sub.plan_key if sub else None,
        member_count=count,
    )


class OrgUpdateIn(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=200)


@router.patch("/orgs/current", response_model=OrgOut)
def update_org(body: OrgUpdateIn, ctx: OrgContext = Depends(requires("org.edit"))) -> OrgOut:
    if body.name:
        ctx.org.name = body.name.strip()
    ctx.audit("org.update", resource_type="organization", resource_id=ctx.org_id, meta=body.model_dump(exclude_none=True))
    ctx.db.commit()
    return OrgOut.model_validate(ctx.org)


_SETTING_SECTIONS = {
    "security": "org.security",
    "retention": "org.settings",
    "ai": "admin.ai_providers",
    "branding": "org.settings",
}

_SETTING_KEYS = {
    "security": {"mfa_required", "session_ttl_days"},
    "retention": {"conversations_days", "audit_days"},
    "ai": {"grounding_policy", "allowed_providers"},
    "branding": {"accent", "logo_url"},
}
_HEX_COLOUR = re.compile(r"^#[0-9a-fA-F]{6}$")


def _check_int(values: dict[str, Any], key: str, lo: int, hi: int) -> None:
    v = values.get(key)
    if key in values and v is not None and (isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi):
        raise ValidationFailed(f"{key} must be a whole number from {lo} to {hi}.", details={"field": key})


def _validate_settings(ctx: OrgContext, section: str, values: dict[str, Any]) -> None:
    unknown = set(values) - _SETTING_KEYS[section]
    if unknown:
        raise ValidationFailed(f"Unknown {section} settings: {sorted(unknown)}.")
    if section == "security":
        if "mfa_required" in values:
            if not isinstance(values["mfa_required"], bool):
                raise ValidationFailed("mfa_required must be true or false.", details={"field": "mfa_required"})
            if values["mfa_required"] and not ctx.principal.mfa_enabled:
                # Otherwise the admin locks themselves out of the organisation on the next request.
                raise ValidationFailed(
                    "Turn on two-factor authentication for your own account before requiring it for everyone.",
                    details={"field": "mfa_required"},
                )
        _check_int(values, "session_ttl_days", 1, 365)
    elif section == "retention":
        _check_int(values, "conversations_days", 1, 3650)
        _check_int(values, "audit_days", 30, 3650)
    elif section == "ai":
        if "grounding_policy" in values and values["grounding_policy"] not in ("strict", "balanced", "open"):
            raise ValidationFailed("grounding_policy must be strict, balanced or open.", details={"field": "grounding_policy"})
        kinds = values.get("allowed_providers")
        if "allowed_providers" in values and (not isinstance(kinds, list) or not all(isinstance(k, str) for k in kinds)):
            raise ValidationFailed("allowed_providers must be a list of provider types.", details={"field": "allowed_providers"})
    elif section == "branding":
        accent = values.get("accent")
        if accent is not None and (not isinstance(accent, str) or not _HEX_COLOUR.match(accent)):
            raise ValidationFailed("accent must be a 6-digit hex colour such as #2563eb.", details={"field": "accent"})
        logo = values.get("logo_url")
        if logo is not None:
            parsed = urlparse(logo) if isinstance(logo, str) else None
            if parsed is None or parsed.scheme != "https" or not parsed.netloc or len(logo) > 1000:
                raise ValidationFailed("logo_url must be an https:// URL.", details={"field": "logo_url"})


class OrgSettingsIn(BaseModel):
    section: str
    values: dict[str, Any]


@router.patch("/orgs/current/settings", response_model=OrgOut)
def update_org_settings(body: OrgSettingsIn, ctx: OrgContext = Depends(get_org_context)) -> OrgOut:
    perm = _SETTING_SECTIONS.get(body.section)
    if perm is None:
        raise ValidationFailed(f"Unknown settings section. Use one of {sorted(_SETTING_SECTIONS)}.")
    ctx.require(perm)
    _validate_settings(ctx, body.section, body.values)
    settings = dict(ctx.org.settings or {})
    settings[body.section] = {**settings.get(body.section, {}), **body.values}
    ctx.org.settings = settings
    ctx.audit("org.settings", resource_type="organization", resource_id=ctx.org_id, meta={"section": body.section, "values": body.values})
    ctx.db.commit()
    return OrgOut.model_validate(ctx.org)


class OrgDeleteIn(BaseModel):
    confirm_name: str


@router.delete("/orgs/current", response_model=Ok)
def delete_org(body: OrgDeleteIn, ctx: OrgContext = Depends(requires("org.billing"))) -> Ok:
    """Deactivate the organisation and schedule a full purge of its data."""
    if body.confirm_name != ctx.org.name:
        raise ValidationFailed("Type the organization name exactly to confirm.")
    from app.jobs.queue import enqueue

    ctx.org.status = "deactivated"
    ctx.audit("org.delete_requested", resource_type="organization", resource_id=ctx.org_id)
    enqueue(ctx.db, ctx.org_id, "purge_organization", payload={}, created_by=ctx.user_id, resource_type="organization", resource_id=ctx.org_id, idempotency_key=f"purge-org:{ctx.org_id}")
    ctx.db.commit()
    return Ok(detail="Organization deactivated; data purge scheduled.")


# --------------------------------------------------------------------------- members


class MemberOut(BaseModel):
    user_id: uuid.UUID
    email: str
    full_name: str
    status: str
    user_status: str
    title: str
    department_id: uuid.UUID | None
    department: str | None
    roles: list[dict[str, Any]]
    teams: list[uuid.UUID]
    joined_at: datetime
    last_login_at: datetime | None
    mfa_enabled: bool


def _member_rows(ctx: OrgContext, q: str | None, limit: int, offset: int) -> tuple[list[MemberOut], int]:
    db = ctx.db
    base = select(Membership, User).join(User, User.id == Membership.user_id).where(Membership.org_id == ctx.org_id)
    if q:
        like = f"%{q.lower()}%"
        base = base.where(or_(func.lower(User.email).like(like), func.lower(User.full_name).like(like)))
    total = db.scalar(select(func.count()).select_from(base.subquery())) or 0
    rows = db.execute(base.order_by(User.full_name, User.email).limit(limit).offset(offset)).all()
    user_ids = [u.id for _, u in rows]
    roles: dict[uuid.UUID, list[dict[str, Any]]] = {u: [] for u in user_ids}
    teams: dict[uuid.UUID, list[uuid.UUID]] = {u: [] for u in user_ids}
    depts = {d.id: d.name for d in db.scalars(select(Department).where(Department.org_id == ctx.org_id))}
    if user_ids:
        for a, role in db.execute(
            select(RoleAssignment, Role).join(Role, Role.id == RoleAssignment.role_id).where(RoleAssignment.org_id == ctx.org_id, RoleAssignment.user_id.in_(user_ids))
        ):
            roles[a.user_id].append({
                "assignment_id": str(a.id), "role_id": str(role.id), "key": role.key, "name": role.name,
                "scope_type": a.scope_type, "scope_id": str(a.scope_id) if a.scope_id else None,
                "expires_at": a.expires_at.isoformat() if a.expires_at else None,
            })
        for tm in db.scalars(select(TeamMember).where(TeamMember.org_id == ctx.org_id, TeamMember.user_id.in_(user_ids))):
            teams[tm.user_id].append(tm.team_id)
    out = [
        MemberOut(
            user_id=u.id, email=u.email, full_name=u.full_name, status=m.status, user_status=u.status,
            title=m.title, department_id=m.department_id, department=depts.get(m.department_id) if m.department_id else None,
            roles=roles[u.id], teams=teams[u.id], joined_at=m.joined_at, last_login_at=u.last_login_at, mfa_enabled=u.mfa_enabled,
        )
        for m, u in rows
    ]
    return out, total


@router.get("/members")
def list_members(q: str | None = None, limit: int = 100, offset: int = 0, ctx: OrgContext = Depends(requires("org.view"))) -> dict[str, Any]:
    items, total = _member_rows(ctx, q, min(limit, 500), offset)
    return page([i.model_dump(mode="json") for i in items], total, limit, offset)


class MemberUpdateIn(BaseModel):
    status: str | None = Field(default=None, pattern="^(active|suspended)$")
    title: str | None = Field(default=None, max_length=200)
    department_id: uuid.UUID | None = None
    clear_department: bool = False


@router.patch("/members/{user_id}", response_model=Ok)
def update_member(user_id: uuid.UUID, body: MemberUpdateIn, ctx: OrgContext = Depends(requires("org.users"))) -> Ok:
    m = ctx.db.scalar(select(Membership).where(Membership.org_id == ctx.org_id, Membership.user_id == user_id))
    if m is None:
        raise NotFound("Member not found.")
    if body.status and user_id == ctx.user_id:
        raise ValidationFailed("You cannot change your own membership status.")
    if body.status:
        if body.status == "suspended" and org_service.owner_count(ctx.db, ctx.org_id) <= 1 and _is_owner(ctx, user_id):
            raise ValidationFailed("Cannot suspend the last owner.")
        m.status = body.status
    if body.title is not None:
        m.title = body.title
    if body.department_id:
        if ctx.db.get(Department, body.department_id) is None:
            raise NotFound("Department not found.")
        m.department_id = body.department_id
    if body.clear_department:
        m.department_id = None
    ctx.audit("member.update", resource_type="membership", resource_id=m.id, meta=body.model_dump(exclude_none=True, mode="json"))
    ctx.db.commit()
    return Ok()


def _is_owner(ctx: OrgContext, user_id: uuid.UUID) -> bool:
    owner = org_service.role_by_key(ctx.db, ctx.org_id, "owner")
    return bool(ctx.db.scalar(select(RoleAssignment).where(RoleAssignment.role_id == owner.id, RoleAssignment.user_id == user_id)))


@router.delete("/members/{user_id}", response_model=Ok)
def remove_member(user_id: uuid.UUID, ctx: OrgContext = Depends(requires("org.users"))) -> Ok:
    m = ctx.db.scalar(select(Membership).where(Membership.org_id == ctx.org_id, Membership.user_id == user_id))
    if m is None:
        raise NotFound("Member not found.")
    if _is_owner(ctx, user_id) and org_service.owner_count(ctx.db, ctx.org_id) <= 1:
        raise ValidationFailed("Cannot remove the last owner.")
    if _is_owner(ctx, user_id) and not ctx.authz.has("org.billing"):
        raise PermissionDenied("Only an owner can remove another owner.")
    for model in (RoleAssignment, TeamMember):
        for row in ctx.db.scalars(select(model).where(model.org_id == ctx.org_id, model.user_id == user_id)):
            ctx.db.delete(row)
    from app.models import ResourceGrant, WorkspaceMember

    for row in ctx.db.scalars(select(WorkspaceMember).where(WorkspaceMember.org_id == ctx.org_id, WorkspaceMember.user_id == user_id)):
        ctx.db.delete(row)
    for row in ctx.db.scalars(select(ResourceGrant).where(ResourceGrant.org_id == ctx.org_id, ResourceGrant.subject_type == "user", ResourceGrant.subject_id == user_id)):
        ctx.db.delete(row)
    ctx.db.delete(m)
    ctx.audit("member.remove", resource_type="user", resource_id=user_id)
    ctx.db.commit()
    return Ok()


# ------------------------------------------------------------------------ invitations


class InviteIn(BaseModel):
    email: EmailStr
    role_id: uuid.UUID | None = None
    team_id: uuid.UUID | None = None
    workspace_id: uuid.UUID | None = None


class InvitationOut(ORM):
    id: uuid.UUID
    email: str
    role_id: uuid.UUID | None
    team_id: uuid.UUID | None
    workspace_id: uuid.UUID | None
    status: str
    expires_at: datetime
    created_at: datetime


@router.get("/invitations", response_model=list[InvitationOut])
def list_invitations(ctx: OrgContext = Depends(requires("org.users"))) -> list[InvitationOut]:
    rows = ctx.db.scalars(select(Invitation).where(Invitation.org_id == ctx.org_id).order_by(Invitation.created_at.desc()).limit(500))
    return [InvitationOut.model_validate(r) for r in rows]


@router.post("/invitations", status_code=201)
def create_invitation(body: InviteIn, ctx: OrgContext = Depends(requires("org.users"))) -> dict[str, Any]:
    from app.services.billing import check_limit

    check_limit(ctx.db, ctx.org_id, "users", add=1)
    inviter = ctx.db.get(User, ctx.user_id)
    assert inviter is not None
    inv, _raw = org_service.invite(ctx.db, ctx.authz, email=body.email, role_id=body.role_id, team_id=body.team_id, workspace_id=body.workspace_id, inviter=inviter)
    ctx.db.commit()
    return {"invitation": InvitationOut.model_validate(inv).model_dump(mode="json"), "email_sent": True}


@router.delete("/invitations/{invitation_id}", response_model=Ok)
def revoke_invitation(invitation_id: uuid.UUID, ctx: OrgContext = Depends(requires("org.users"))) -> Ok:
    inv = ctx.db.get(Invitation, invitation_id)
    if inv is None or inv.org_id != ctx.org_id:
        raise NotFound("Invitation not found.")
    inv.status = "revoked"
    ctx.audit("member.invite_revoked", resource_type="invitation", resource_id=inv.id)
    ctx.db.commit()
    return Ok()


class InvitationLookupOut(BaseModel):
    organization_name: str
    email: str
    account_exists: bool
    expires_at: datetime


@router.get("/invitations/lookup", response_model=InvitationLookupOut)
def lookup_invitation(token: str) -> InvitationLookupOut:
    with system_session() as sys_db:
        inv = org_service.find_invitation(sys_db, token)
        org = sys_db.get(Organization, inv.org_id)
        exists = sys_db.scalar(select(User.id).where(User.email == inv.email)) is not None
        return InvitationLookupOut(organization_name=org.name if org else "", email=inv.email, account_exists=exists, expires_at=inv.expires_at)


class AcceptIn(BaseModel):
    token: str = Field(max_length=200)
    password: str | None = Field(default=None, max_length=256)
    full_name: str | None = Field(default=None, max_length=200)


@router.post("/invitations/accept", response_model=TokenOut)
def accept_invitation(body: AcceptIn, request: Request, response: Response) -> TokenOut:
    """Accept an invitation. Existing accounts must provide their password; new
    accounts are created (the invitation proves control of the email address).

    Runs as one narrow system transaction because the caller has no tenant yet;
    the invitation token is the authorization and every step is audited.
    """
    ua, ip = client_meta(request)
    with system_session() as sdb:
        inv = org_service.find_invitation(sdb, body.token)
        user = sdb.scalar(select(User).where(User.email == inv.email))
        if user is None:
            if not body.password:
                raise ValidationFailed("Choose a password to create your account.")
            user = auth_service.register_user(sdb, email=inv.email, password=body.password, full_name=body.full_name or "")
        else:
            result = auth_service.authenticate(sdb, email=inv.email, password=body.password or "", user_agent=ua, ip=ip)
            if result.mfa_required:
                raise AuthenticationRequired("Sign in with two-factor authentication first, then open the invitation again.")
        org_service.accept_invitation(sdb, inv, user)
        tokens = auth_service.create_session(sdb, user, user_agent=ua, ip=ip)
    return _token_out(response, tokens, tokens.user_id, "cookie")
