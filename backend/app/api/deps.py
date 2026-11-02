"""Request dependencies: DB session, principal, organisation context.

Handlers commit explicitly (``ctx.db.commit()``); the dependency only rolls
back and closes. Tenant context is bound to the session *before* any tenant
table is touched, so RLS applies to every query a handler makes.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core import security
from app.core.ratelimit import enforce
from app.db.session import bind_tenant, new_session, system_session
from app.models import ApiKey, Organization, User, UserSession
from app.rbac.authorizer import Authorizer, Principal
from app.services.activity import audit
from app.services.auth_service import validate_access_session
from app.utils.errors import AuthenticationRequired, NotFound, PermissionDenied, ValidationFailed
from app.utils.logging import org_id_var, user_id_var


def get_db() -> Iterator[Session]:
    session = new_session()
    try:
        yield session
    finally:
        if session.in_transaction():
            session.rollback()
        session.close()


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return None


def _api_key_principal(raw: str) -> Principal:
    prefix = raw.split(".", 1)[0]
    # API keys are looked up before any tenant is known: one narrow system read.
    with system_session() as sys_db:
        key = sys_db.scalar(select(ApiKey).where(ApiKey.prefix == prefix))
        if key is None or not security.constant_time_equals(key.key_hash, security.sha256_hex(raw)):
            raise AuthenticationRequired("Invalid API key.")
        now = datetime.now(UTC)
        if key.revoked_at is not None or (key.expires_at and key.expires_at <= now):
            raise AuthenticationRequired("API key revoked or expired.")
        user = sys_db.get(User, key.created_by)
        if user is None or user.status != "active":
            raise AuthenticationRequired("API key owner is not active.")
        limit = key.rate_limit_per_minute or get_settings().api_rate_limit_per_minute
        enforce(f"apikey:{key.id}", limit)
        key.last_used_at = now
        return Principal(
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            org_id=key.org_id,
            is_platform_admin=False,  # keys never carry platform privileges
            api_key_id=key.id,
            api_scopes=frozenset(key.scopes),
            mfa_enabled=user.mfa_enabled,
        )


def get_principal(request: Request, db: Session = Depends(get_db)) -> Principal:
    token = _bearer(request)
    if token is None:
        raise AuthenticationRequired("Authentication required.")
    if token.startswith("rk_"):
        principal = _api_key_principal(token)
    else:
        try:
            payload = security.decode_token(token)
        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationRequired("Access token expired.", details={"reason": "token_expired"}) from exc
        except jwt.InvalidTokenError as exc:
            raise AuthenticationRequired("Invalid access token.") from exc
        user_id = uuid.UUID(payload["sub"])
        session = validate_access_session(db, uuid.UUID(payload["sid"]), user_id)
        user = db.get(User, user_id)
        if user is None or user.status != "active":
            raise AuthenticationRequired("Account is not active.")
        principal = Principal(
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            is_platform_admin=user.is_platform_admin,
            session_id=session.id,
            mfa_enabled=user.mfa_enabled,
            session_family_id=session.family_id,
        )
    bind_tenant(db, user_id=principal.user_id)
    user_id_var.set(str(principal.user_id))
    request.state.principal = principal
    return principal


@dataclass
class OrgContext:
    db: Session
    principal: Principal
    authz: Authorizer
    org: Organization

    @property
    def org_id(self) -> uuid.UUID:
        return self.org.id

    @property
    def user_id(self) -> uuid.UUID:
        return self.principal.user_id

    def require(self, perm: str, workspace_id: uuid.UUID | None = None) -> None:
        try:
            self.authz.require(perm, workspace_id)
        except PermissionDenied:
            audit(
                self.db,
                "authz.denied",
                org_id=self.org_id,
                actor_user_id=self.user_id,
                actor_api_key_id=self.principal.api_key_id,
                outcome="denied",
                meta={"permission": perm},
            )
            self.db.commit()
            raise

    def audit(self, action: str, **kwargs: object) -> None:
        audit(
            self.db,
            action,
            org_id=self.org_id,
            actor_user_id=self.user_id,
            actor_email=self.principal.email,
            actor_api_key_id=self.principal.api_key_id,
            **kwargs,  # type: ignore[arg-type]
        )


def enforce_org_policy(db: Session, org: Organization, principal: Principal) -> None:
    """Organisation security policy: required two-factor and maximum sign-in age.

    Applies to members (and API keys, whose owner must satisfy the MFA rule).
    Support-access sessions are governed by their own time-boxed grant instead.
    """
    policy = (org.settings or {}).get("security") or {}
    if policy.get("mfa_required") and not principal.mfa_enabled:
        raise PermissionDenied(
            "This organization requires two-factor authentication. Turn it on in Settings → Security, then continue.",
            details={"reason": "mfa_required"},
        )
    ttl = policy.get("session_ttl_days")
    if ttl and principal.session_family_id is not None:
        signed_in = db.scalar(select(func.min(UserSession.created_at)).where(UserSession.family_id == principal.session_family_id))
        if signed_in is not None and signed_in < datetime.now(UTC) - timedelta(days=int(ttl)):
            raise PermissionDenied(
                f"This organization requires signing in again every {int(ttl)} day(s).",
                details={"reason": "reauth_required", "session_ttl_days": int(ttl)},
            )


def get_org_context(
    request: Request,
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> OrgContext:
    raw = request.headers.get("x-org-id") or request.query_params.get("org_id")
    if principal.api_key_id is not None:
        if raw and raw != str(principal.org_id):
            raise PermissionDenied("API key is not valid for this organization.")
        org_id = principal.org_id
    else:
        if not raw:
            raise ValidationFailed("Select an organization (X-Org-Id header).")
        try:
            org_id = uuid.UUID(raw)
        except ValueError as exc:
            raise ValidationFailed("Invalid organization id.") from exc
    assert org_id is not None
    principal.org_id = org_id
    bind_tenant(db, org_id=org_id, user_id=principal.user_id)
    org = db.get(Organization, org_id)
    if org is None:
        raise NotFound("Organization not found.")
    authz = Authorizer(db, principal).load()
    if not (authz.is_member or authz.via_support):
        raise PermissionDenied("You are not a member of this organization.")
    if org.status != "active" and not principal.is_platform_admin:
        raise PermissionDenied(f"This organization is {org.status}.")
    if authz.is_member:
        enforce_org_policy(db, org, principal)
    if authz.via_support:
        audit(db, "support.access", org_id=org_id, actor_user_id=principal.user_id, meta={"path": request.url.path})
        db.commit()
    org_id_var.set(str(org_id))
    return OrgContext(db=db, principal=principal, authz=authz, org=org)


def requires(*perms: str) -> Callable[[OrgContext], OrgContext]:
    """Dependency factory: organisation context + required org-level capabilities."""

    def _dep(ctx: OrgContext = Depends(get_org_context)) -> OrgContext:
        for perm in perms:
            ctx.require(perm)
        return ctx

    return _dep


def requires_any_scope(*perms: str) -> Callable[[OrgContext], OrgContext]:
    """Capability held at organisation scope OR in at least one workspace.

    For endpoints whose results are filtered per resource (each knowledge base
    is re-checked against the caller's workspace-scoped permissions).
    """

    def _dep(ctx: OrgContext = Depends(get_org_context)) -> OrgContext:
        for perm in perms:
            if not ctx.authz.has_anywhere(perm):
                ctx.require(perm)  # audits the denial and raises
        return ctx

    return _dep


def platform_admin(principal: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> tuple[Principal, Session]:
    if not principal.is_platform_admin or principal.api_key_id is not None:
        raise PermissionDenied("Platform administrator access required.")
    return principal, db


def client_meta(request: Request) -> tuple[str, str]:
    ip = request.client.host if request.client else ""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded and get_settings().is_production:
        ip = forwarded.split(",")[0].strip()
    return request.headers.get("user-agent", ""), ip
