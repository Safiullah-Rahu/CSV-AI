"""Organisations, members, invitations, roles and grants."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core import security
from app.db.session import bind_tenant, system_session
from app.models import (
    Invitation,
    Membership,
    Organization,
    ResourceGrant,
    Role,
    RoleAssignment,
    RolePermission,
    Subscription,
    TeamMember,
    User,
    Workspace,
    WorkspaceMember,
)
from app.rbac.authorizer import Authorizer
from app.rbac.catalog import ALL_PERMISSIONS, RESOURCE_TYPES, ROLE_TEMPLATES, SUBJECT_TYPES
from app.services import email as mailer
from app.services.activity import audit
from app.utils.errors import Conflict, NotFound, PermissionDenied, ValidationFailed


def _now() -> datetime:
    return datetime.now(UTC)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:60] or "org"


def unique_slug(name: str) -> str:
    """Slugs are unique across tenants, so the check needs a cross-tenant (system) read."""
    base = slugify(name)
    slug, n = base, 1
    with system_session() as sys_db:
        while sys_db.scalar(select(func.count()).select_from(Organization).where(Organization.slug == slug)):
            n += 1
            slug = f"{base}-{n}"
    return slug


def seed_roles(db: Session, org_id: uuid.UUID) -> dict[str, Role]:
    roles: dict[str, Role] = {}
    for template in ROLE_TEMPLATES:
        role = Role(
            org_id=org_id,
            key=template.key,
            name=template.name,
            description=template.description,
            is_system=True,
            rank=template.rank,
        )
        db.add(role)
        db.flush()
        for perm in sorted(template.permissions):
            db.add(RolePermission(org_id=org_id, role_id=role.id, permission=perm))
        roles[template.key] = role
    return roles


def create_organization(db: Session, *, name: str, owner: User) -> Organization:
    settings = get_settings()
    name = name.strip()
    if not 2 <= len(name) <= 200:
        raise ValidationFailed("Organization name must be 2–200 characters.")
    org_id = uuid.uuid4()
    # The new organisation becomes the tenant context for the rest of this transaction.
    bind_tenant(db, org_id=org_id, user_id=owner.id)
    org = Organization(
        id=org_id,
        name=name,
        slug=unique_slug(name),
        created_by=owner.id,
        settings={
            "security": {"mfa_required": False, "session_ttl_days": settings.refresh_token_ttl_days},
            "retention": {"conversations_days": None, "audit_days": 365, "deleted_documents_days": 0},
            "ai": {"allowed_providers": [], "grounding_policy": "strict"},
            "branding": {"accent": None, "logo_url": None},
        },
    )
    db.add(org)
    db.flush()
    roles = seed_roles(db, org.id)
    db.add(Membership(org_id=org.id, user_id=owner.id))
    db.add(RoleAssignment(org_id=org.id, role_id=roles["owner"].id, user_id=owner.id, scope_type="org", granted_by=owner.id))
    db.add(Subscription(org_id=org.id, plan_key=settings.default_plan if not settings.billing_enabled else "free"))
    general = Workspace(org_id=org.id, name="General", description="Default workspace", created_by=owner.id)
    db.add(general)
    db.flush()
    db.add(WorkspaceMember(org_id=org.id, workspace_id=general.id, user_id=owner.id))
    audit(db, "org.create", org_id=org.id, actor_user_id=owner.id, resource_type="organization", resource_id=org.id, meta={"name": name})
    return org


def get_role(db: Session, org_id: uuid.UUID, role_id: uuid.UUID) -> Role:
    role = db.get(Role, role_id)
    if role is None or role.org_id != org_id:
        raise NotFound("Role not found.")
    return role


def role_by_key(db: Session, org_id: uuid.UUID, key: str) -> Role:
    role = db.scalar(select(Role).where(Role.org_id == org_id, Role.key == key))
    if role is None:
        raise NotFound(f"Role '{key}' not found.")
    return role


# --------------------------------------------------------------------- invitations


def invite(
    db: Session,
    authz: Authorizer,
    *,
    email: str,
    role_id: uuid.UUID | None,
    team_id: uuid.UUID | None = None,
    workspace_id: uuid.UUID | None = None,
    inviter: User,
) -> tuple[Invitation, str]:
    org_id = authz.p.org_id
    assert org_id is not None
    email = email.strip().lower()
    role = get_role(db, org_id, role_id) if role_id else role_by_key(db, org_id, "member")
    authz.load()
    if (role.key == "owner" and not authz.can_assign_role(role)) or role.rank > authz.max_rank:
        raise PermissionDenied("You cannot invite users with a role above your own.")
    existing_user = db.scalar(select(User).where(User.email == email))
    if existing_user and db.scalar(
        select(Membership).where(Membership.org_id == org_id, Membership.user_id == existing_user.id)
    ):
        raise Conflict("This user is already a member.")
    db.execute(
        delete(Invitation).where(
            Invitation.org_id == org_id, Invitation.email == email, Invitation.status == "pending"
        )
    )
    raw = security.new_token()
    inv = Invitation(
        org_id=org_id,
        email=email,
        role_id=role.id,
        team_id=team_id,
        workspace_id=workspace_id,
        token_hash=security.sha256_hex(raw),
        invited_by=inviter.id,
        expires_at=_now() + timedelta(days=7),
    )
    db.add(inv)
    db.flush()
    org = db.get(Organization, org_id)
    url = mailer.link(f"/invite?token={raw}")
    mailer.send_email(
        email,
        f"You're invited to {org.name if org else 'an organization'}",
        f"{inviter.full_name or inviter.email} invited you to join {org.name if org else ''}.\n\nAccept (valid 7 days): {url}\n",
        link=url,
    )
    audit(db, "member.invite", org_id=org_id, actor_user_id=inviter.id, resource_type="invitation", resource_id=inv.id, meta={"email": email, "role": role.key})
    return inv, raw


def find_invitation(db: Session, raw: str) -> Invitation:
    """Look up an invitation by token. Caller must use a system session (no tenant yet)."""
    inv = db.scalar(select(Invitation).where(Invitation.token_hash == security.sha256_hex(raw)))
    if inv is None or inv.status != "pending" or inv.expires_at <= _now():
        raise ValidationFailed("This invitation is invalid or has expired.")
    return inv


def accept_invitation(db: Session, inv: Invitation, user: User) -> Membership:
    if user.email != inv.email:
        raise PermissionDenied("This invitation was sent to a different email address.")
    bind_tenant(db, org_id=inv.org_id, user_id=user.id)
    membership = db.scalar(select(Membership).where(Membership.org_id == inv.org_id, Membership.user_id == user.id))
    if membership is None:
        membership = Membership(org_id=inv.org_id, user_id=user.id)
        db.add(membership)
    role_id = inv.role_id or role_by_key(db, inv.org_id, "member").id
    db.add(RoleAssignment(org_id=inv.org_id, role_id=role_id, user_id=user.id, scope_type="org", granted_by=inv.invited_by))
    if inv.team_id:
        db.add(TeamMember(org_id=inv.org_id, team_id=inv.team_id, user_id=user.id))
    if inv.workspace_id:
        db.add(WorkspaceMember(org_id=inv.org_id, workspace_id=inv.workspace_id, user_id=user.id))
    else:
        general = db.scalar(select(Workspace).where(Workspace.org_id == inv.org_id, Workspace.name == "General"))
        if general:
            db.add(WorkspaceMember(org_id=inv.org_id, workspace_id=general.id, user_id=user.id))
    inv.status = "accepted"
    inv.accepted_at = _now()
    user.email_verified_at = user.email_verified_at or _now()
    db.flush()
    audit(db, "member.join", org_id=inv.org_id, actor_user_id=user.id, resource_type="membership", resource_id=membership.id)
    return membership


# ---------------------------------------------------------------------------- roles


def create_custom_role(
    db: Session, authz: Authorizer, *, key: str, name: str, description: str, permissions: list[str]
) -> Role:
    org_id = authz.p.org_id
    assert org_id is not None
    perms = set(permissions)
    unknown = perms - ALL_PERMISSIONS
    if unknown:
        raise ValidationFailed(f"Unknown permissions: {sorted(unknown)}")
    authz.load()
    if not perms <= authz.org_perms:
        raise PermissionDenied("Custom roles cannot grant permissions you do not hold.", details={"missing": sorted(perms - authz.org_perms)})
    key = slugify(key or name).replace("-", "_")
    if db.scalar(select(Role).where(Role.org_id == org_id, Role.key == key)):
        raise Conflict("A role with this key already exists.")
    role = Role(org_id=org_id, key=key, name=name.strip(), description=description, is_system=False, rank=min(authz.max_rank, 85))
    db.add(role)
    db.flush()
    for perm in sorted(perms):
        db.add(RolePermission(org_id=org_id, role_id=role.id, permission=perm))
    audit(db, "role.create", org_id=org_id, actor_user_id=authz.p.user_id, resource_type="role", resource_id=role.id, meta={"permissions": sorted(perms)})
    return role


def update_custom_role(db: Session, authz: Authorizer, role: Role, *, name: str | None, description: str | None, permissions: list[str] | None) -> Role:
    if role.is_system:
        raise PermissionDenied("System roles cannot be modified. Create a custom role instead.")
    if role.rank > authz.max_rank:
        raise PermissionDenied("You cannot modify a role above your own.")
    if name is not None:
        role.name = name.strip()
    if description is not None:
        role.description = description
    if permissions is not None:
        perms = set(permissions)
        unknown = perms - ALL_PERMISSIONS
        if unknown:
            raise ValidationFailed(f"Unknown permissions: {sorted(unknown)}")
        authz.load()
        if not perms <= authz.org_perms:
            raise PermissionDenied("Custom roles cannot grant permissions you do not hold.")
        db.execute(delete(RolePermission).where(RolePermission.role_id == role.id))
        for perm in sorted(perms):
            db.add(RolePermission(org_id=role.org_id, role_id=role.id, permission=perm))
    audit(db, "role.update", org_id=role.org_id, actor_user_id=authz.p.user_id, resource_type="role", resource_id=role.id, meta={"permissions": permissions})
    return role


def role_permissions(db: Session, role_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[str]]:
    out: dict[uuid.UUID, list[str]] = {r: [] for r in role_ids}
    if role_ids:
        for rid, perm in db.execute(select(RolePermission.role_id, RolePermission.permission).where(RolePermission.role_id.in_(role_ids))):
            out[rid].append(perm)
    return out


def assign_role(
    db: Session,
    authz: Authorizer,
    *,
    role: Role,
    user_id: uuid.UUID | None,
    team_id: uuid.UUID | None,
    workspace_id: uuid.UUID | None,
    expires_at: datetime | None,
) -> RoleAssignment:
    if not authz.can_assign_role(role):
        raise PermissionDenied("You cannot assign a role above your own.")
    if (user_id is None) == (team_id is None):
        raise ValidationFailed("Assign to exactly one of user or team.")
    a = RoleAssignment(
        org_id=role.org_id,
        role_id=role.id,
        user_id=user_id,
        team_id=team_id,
        scope_type="workspace" if workspace_id else "org",
        scope_id=workspace_id,
        expires_at=expires_at,
        granted_by=authz.p.user_id,
    )
    db.add(a)
    db.flush()
    audit(db, "role.assign", org_id=role.org_id, actor_user_id=authz.p.user_id, resource_type="role_assignment", resource_id=a.id, meta={"role": role.key, "user_id": user_id, "team_id": team_id, "workspace_id": workspace_id})
    return a


def owner_count(db: Session, org_id: uuid.UUID) -> int:
    owner = role_by_key(db, org_id, "owner")
    return db.scalar(
        select(func.count()).select_from(RoleAssignment).where(RoleAssignment.role_id == owner.id, RoleAssignment.user_id.is_not(None))
    ) or 0


# --------------------------------------------------------------------------- grants


def create_grant(
    db: Session,
    authz: Authorizer,
    *,
    resource_type: str,
    resource_id: uuid.UUID,
    subject_type: str,
    subject_id: uuid.UUID | None,
    privilege: str,
    effect: str,
    expires_at: datetime | None,
    constraints: dict | None = None,
    note: str = "",
) -> ResourceGrant:
    if resource_type not in RESOURCE_TYPES:
        raise ValidationFailed(f"resource_type must be one of {RESOURCE_TYPES}")
    if subject_type not in SUBJECT_TYPES:
        raise ValidationFailed(f"subject_type must be one of {SUBJECT_TYPES}")
    if subject_type == "org":
        subject_id = None
    elif subject_id is None:
        raise ValidationFailed("subject_id is required for this subject type.")
    grant = ResourceGrant(
        org_id=authz.p.org_id,
        resource_type=resource_type,
        resource_id=resource_id,
        subject_type=subject_type,
        subject_id=subject_id,
        privilege=privilege,
        effect=effect,
        expires_at=expires_at,
        constraints=constraints or {},
        note=note,
        created_by=authz.p.user_id,
    )
    db.add(grant)
    db.flush()
    audit(db, "grant.create", org_id=authz.p.org_id, actor_user_id=authz.p.user_id, resource_type=resource_type, resource_id=resource_id, meta={"subject_type": subject_type, "subject_id": subject_id, "privilege": privilege, "effect": effect})
    return grant
