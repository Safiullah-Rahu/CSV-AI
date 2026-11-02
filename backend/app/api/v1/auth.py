"""Authentication endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import client_meta, get_db, get_principal
from app.api.schemas import ORM, Ok
from app.config import get_settings
from app.core.ratelimit import enforce
from app.db.session import system_session
from app.models import Membership, Organization, Role, RoleAssignment, User
from app.rbac.authorizer import Principal
from app.services import auth_service as auth
from app.services import org_service
from app.utils.errors import AuthenticationRequired, ValidationFailed

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)
    full_name: str = Field(default="", max_length=200)
    organization_name: str | None = Field(default=None, max_length=200)


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(max_length=256)


class MfaVerifyIn(BaseModel):
    mfa_token: str
    code: str = Field(max_length=12)


class TokenOut(BaseModel):
    access_token: str | None = None
    token_type: str = "bearer"
    expires_in: int | None = None
    refresh_token: str | None = None
    mfa_required: bool = False
    mfa_token: str | None = None
    user_id: uuid.UUID | None = None


class RefreshIn(BaseModel):
    refresh_token: str | None = None


class UserOut(ORM):
    id: uuid.UUID
    email: str
    full_name: str
    status: str
    email_verified_at: datetime | None
    is_platform_admin: bool
    mfa_enabled: bool
    last_login_at: datetime | None
    locale: str
    preferences: dict
    created_at: datetime


class OrgMembershipOut(BaseModel):
    id: uuid.UUID
    name: str
    slug: str
    status: str
    roles: list[str]


class MeOut(BaseModel):
    user: UserOut
    organizations: list[OrgMembershipOut]


class SessionOut(ORM):
    id: uuid.UUID
    user_agent: str
    ip: str
    created_at: datetime
    last_used_at: datetime
    expires_at: datetime
    current: bool = False


def _set_refresh_cookie(response: Response, raw: str) -> None:
    settings = get_settings()
    response.set_cookie(
        settings.refresh_cookie_name,
        raw,
        max_age=settings.refresh_token_ttl_days * 86400,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
        path=f"{settings.api_prefix}/auth",
    )


def _clear_refresh_cookie(response: Response) -> None:
    settings = get_settings()
    response.delete_cookie(settings.refresh_cookie_name, path=f"{settings.api_prefix}/auth")


def _token_out(response: Response, tokens: auth.TokenPair, user_id: uuid.UUID, transport: str) -> TokenOut:
    out = TokenOut(access_token=tokens.access_token, expires_in=tokens.expires_in, user_id=user_id)
    if transport == "body":
        out.refresh_token = tokens.refresh_token
    else:
        _set_refresh_cookie(response, tokens.refresh_token)
    return out


def _require_csrf_header(request: Request) -> None:
    # Cookie-authenticated endpoints require a custom header: browsers will not attach
    # it cross-site without a CORS preflight, which our CORS policy rejects.
    if request.headers.get("x-requested-with", "").lower() != "fetch":
        raise ValidationFailed("Missing X-Requested-With header.")


@router.post("/register", response_model=TokenOut, status_code=201)
def register(body: RegisterIn, request: Request, response: Response, transport: str = "cookie", db: Session = Depends(get_db)) -> TokenOut:
    ua, ip = client_meta(request)
    enforce(f"register:{ip}", get_settings().register_rate_limit_per_hour, 3600)
    user = auth.register_user(db, email=body.email, password=body.password, full_name=body.full_name)
    if body.organization_name:
        org_service.create_organization(db, name=body.organization_name, owner=user)
    settings = get_settings()
    if settings.email_verification_required:
        db.commit()
        return TokenOut(user_id=user.id)
    tokens = auth.create_session(db, user, user_agent=ua, ip=ip)
    db.commit()
    return _token_out(response, tokens, user.id, transport)


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, request: Request, response: Response, transport: str = "cookie", db: Session = Depends(get_db)) -> TokenOut:
    ua, ip = client_meta(request)
    settings = get_settings()
    enforce(f"login:{ip}", settings.login_rate_limit_per_minute * 3)
    enforce(f"login:{ip}:{body.email.lower()}", settings.login_rate_limit_per_minute)
    try:
        result = auth.authenticate(db, email=body.email, password=body.password, user_agent=ua, ip=ip)
    except Exception:
        db.commit()  # persist failed-login counters and audit rows
        raise
    db.commit()
    if result.mfa_required:
        return TokenOut(mfa_required=True, mfa_token=result.mfa_token, user_id=result.user.id)
    assert result.tokens
    return _token_out(response, result.tokens, result.user.id, transport)


@router.post("/mfa/verify", response_model=TokenOut)
def mfa_verify(body: MfaVerifyIn, request: Request, response: Response, transport: str = "cookie", db: Session = Depends(get_db)) -> TokenOut:
    ua, ip = client_meta(request)
    enforce(f"mfa:{ip}", 20)
    try:
        result = auth.complete_mfa_login(db, mfa_token=body.mfa_token, code=body.code, user_agent=ua, ip=ip)
    except Exception:
        db.commit()
        raise
    db.commit()
    assert result.tokens
    return _token_out(response, result.tokens, result.user.id, transport)


@router.post("/refresh", response_model=TokenOut)
def refresh(request: Request, response: Response, body: RefreshIn | None = None, db: Session = Depends(get_db)) -> TokenOut:
    ua, ip = client_meta(request)
    enforce(f"refresh:{ip}", 120)
    raw = body.refresh_token if body and body.refresh_token else None
    transport = "body" if raw else "cookie"
    if raw is None:
        _require_csrf_header(request)
        raw = request.cookies.get(get_settings().refresh_cookie_name)
    if not raw:
        raise AuthenticationRequired("No refresh token.")
    try:
        tokens = auth.refresh(db, raw, user_agent=ua, ip=ip)
    except AuthenticationRequired:
        db.commit()  # keep family revocation on reuse detection
        _clear_refresh_cookie(response)
        raise
    db.commit()
    return _token_out(response, tokens, tokens.user_id, transport)


@router.post("/logout", response_model=Ok)
def logout(request: Request, response: Response, principal: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> Ok:
    if principal.session_id:
        auth.revoke_session(db, principal.session_id, principal.user_id)
    db.commit()
    _clear_refresh_cookie(response)
    return Ok()


@router.post("/logout-all", response_model=Ok)
def logout_all(response: Response, principal: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> Ok:
    auth.revoke_all_sessions(db, principal.user_id, reason="logout_all")
    db.commit()
    _clear_refresh_cookie(response)
    return Ok()


def _memberships(db: Session, user_id: uuid.UUID) -> list[OrgMembershipOut]:
    # A user's memberships span tenants; RLS allows reading one's own membership rows.
    rows = db.execute(
        select(Organization, Membership)
        .join(Membership, Membership.org_id == Organization.id)
        .where(Membership.user_id == user_id, Membership.status == "active", Organization.status != "deactivated")
        .order_by(Organization.name)
    ).all()
    out = []
    for org, _m in rows:
        out.append(OrgMembershipOut(id=org.id, name=org.name, slug=org.slug, status=org.status, roles=[]))
    if out:
        # Role names per org need tenant context per org; read them via a narrow system query.
        with system_session() as sys_db:
            role_rows = sys_db.execute(
                select(RoleAssignment.org_id, Role.name)
                .join(Role, Role.id == RoleAssignment.role_id)
                .where(RoleAssignment.user_id == user_id, RoleAssignment.scope_type == "org")
            ).all()
        by_org: dict[uuid.UUID, list[str]] = {}
        for org_id, name in role_rows:
            by_org.setdefault(org_id, []).append(name)
        for o in out:
            o.roles = sorted(by_org.get(o.id, []))
    user = db.get(User, user_id)
    if user is not None and user.is_platform_admin:
        from datetime import UTC, datetime

        from app.models import SupportAccessGrant

        with system_session() as sys_db:
            rows = sys_db.execute(
                select(Organization)
                .join(SupportAccessGrant, SupportAccessGrant.org_id == Organization.id)
                .where(SupportAccessGrant.platform_user_id == user_id, SupportAccessGrant.revoked_at.is_(None), SupportAccessGrant.expires_at > datetime.now(UTC))
            ).scalars().all()
            known = {o.id for o in out}
            out += [OrgMembershipOut(id=o.id, name=o.name, slug=o.slug, status=o.status, roles=["Support access (read-only)"]) for o in rows if o.id not in known]
    return out


@router.get("/me", response_model=MeOut)
def me(principal: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> MeOut:
    user = db.get(User, principal.user_id)
    assert user is not None
    return MeOut(user=UserOut.model_validate(user), organizations=_memberships(db, user.id))


class MeUpdateIn(BaseModel):
    full_name: str | None = Field(default=None, max_length=200)
    locale: str | None = Field(default=None, max_length=16)
    preferences: dict | None = None


@router.patch("/me", response_model=UserOut)
def update_me(body: MeUpdateIn, principal: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> UserOut:
    user = db.get(User, principal.user_id)
    assert user is not None
    if body.full_name is not None:
        user.full_name = body.full_name.strip()
    if body.locale is not None:
        user.locale = body.locale
    if body.preferences is not None:
        user.preferences = {**user.preferences, **body.preferences}
    db.commit()
    return UserOut.model_validate(user)


class TokenIn(BaseModel):
    token: str = Field(max_length=200)


@router.post("/verify-email", response_model=Ok)
def verify_email(body: TokenIn, db: Session = Depends(get_db)) -> Ok:
    auth.verify_email(db, body.token)
    db.commit()
    return Ok(detail="Email verified.")


@router.post("/resend-verification", response_model=Ok)
def resend_verification(principal: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> Ok:
    enforce(f"verify:{principal.user_id}", 5, 3600)
    user = db.get(User, principal.user_id)
    assert user is not None
    auth.send_verification(db, user)
    db.commit()
    return Ok(detail="Verification email sent." if not user.email_verified_at else "Already verified.")


class ForgotIn(BaseModel):
    email: EmailStr


@router.post("/password/forgot", response_model=Ok, status_code=202)
def forgot_password(body: ForgotIn, request: Request, db: Session = Depends(get_db)) -> Ok:
    _, ip = client_meta(request)
    enforce(f"forgot:{ip}", 10, 3600)
    auth.request_password_reset(db, body.email)
    db.commit()
    return Ok(detail="If the account exists, a reset link has been sent.")


class ResetIn(BaseModel):
    token: str = Field(max_length=200)
    password: str = Field(max_length=256)


@router.post("/password/reset", response_model=Ok)
def reset_password(body: ResetIn, db: Session = Depends(get_db)) -> Ok:
    auth.reset_password(db, body.token, body.password)
    db.commit()
    return Ok(detail="Password updated. Sign in with your new password.")


class ChangePasswordIn(BaseModel):
    current_password: str = Field(max_length=256)
    new_password: str = Field(max_length=256)


@router.post("/password/change", response_model=Ok)
def change_password(body: ChangePasswordIn, principal: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> Ok:
    user = db.get(User, principal.user_id)
    assert user is not None
    family = None
    if principal.session_id:
        from app.models import UserSession

        current = db.get(UserSession, principal.session_id)
        family = current.family_id if current else None
    auth.change_password(db, user, body.current_password, body.new_password, keep_family=family)
    db.commit()
    return Ok(detail="Password changed. Other sessions were signed out.")


@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(principal: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> list[SessionOut]:
    out = []
    for s in auth.active_sessions(db, principal.user_id):
        item = SessionOut.model_validate(s)
        item.current = s.id == principal.session_id
        out.append(item)
    return out


@router.delete("/sessions/{session_id}", response_model=Ok)
def revoke_session(session_id: uuid.UUID, principal: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> Ok:
    auth.revoke_session(db, session_id, principal.user_id, reason="revoked_by_user")
    db.commit()
    return Ok()


class MfaSetupOut(BaseModel):
    secret: str
    otpauth_uri: str


@router.post("/mfa/setup", response_model=MfaSetupOut)
def mfa_setup(principal: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> MfaSetupOut:
    user = db.get(User, principal.user_id)
    assert user is not None
    secret, uri = auth.mfa_begin(db, user)
    db.commit()
    return MfaSetupOut(secret=secret, otpauth_uri=uri)


class CodeIn(BaseModel):
    code: str = Field(max_length=12)


@router.post("/mfa/confirm", response_model=Ok)
def mfa_confirm(body: CodeIn, principal: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> Ok:
    user = db.get(User, principal.user_id)
    assert user is not None
    auth.mfa_confirm(db, user, body.code)
    db.commit()
    return Ok(detail="Two-factor authentication enabled.")


class MfaDisableIn(BaseModel):
    password: str = Field(max_length=256)
    code: str = Field(max_length=12)


@router.post("/mfa/disable", response_model=Ok)
def mfa_disable(body: MfaDisableIn, principal: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> Ok:
    user = db.get(User, principal.user_id)
    assert user is not None
    auth.mfa_disable(db, user, body.password, body.code)
    db.commit()
    return Ok(detail="Two-factor authentication disabled.")


class DeactivateIn(BaseModel):
    password: str = Field(max_length=256)


@router.post("/deactivate", response_model=Ok)
def deactivate(body: DeactivateIn, response: Response, principal: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> Ok:
    from app.core.security import verify_password
    from app.services.activity import audit

    user = db.get(User, principal.user_id)
    assert user is not None
    if not verify_password(body.password, user.password_hash):
        raise ValidationFailed("Password is incorrect.", details={"field": "password"})
    user.status = "deactivated"
    auth.revoke_all_sessions(db, user.id, reason="deactivated")
    audit(db, "auth.deactivate", actor_user_id=user.id)
    db.commit()
    _clear_refresh_cookie(response)
    return Ok(detail="Account deactivated.")
