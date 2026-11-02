"""Authentication: registration, login, sessions, verification, reset, MFA.

Sessions use short-lived JWT access tokens plus opaque refresh tokens that
rotate on every use. Only SHA-256 hashes of refresh tokens are stored.
Presenting an already-rotated refresh token revokes the whole session
family (token theft detection).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core import security
from app.models import EmailToken, User, UserSession
from app.services import email as mailer
from app.services.activity import audit
from app.utils.errors import (
    AuthenticationRequired,
    Conflict,
    PermissionDenied,
    ValidationFailed,
)

MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 15


def _now() -> datetime:
    return datetime.now(UTC)


def normalize_email(email: str) -> str:
    return email.strip().lower()


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    session_id: uuid.UUID
    expires_in: int
    user_id: uuid.UUID


@dataclass
class LoginResult:
    user: User
    tokens: TokenPair | None = None
    mfa_required: bool = False
    mfa_token: str | None = None


# ------------------------------------------------------------------ registration


def register_user(db: Session, *, email: str, password: str, full_name: str) -> User:
    email = normalize_email(email)
    problems = security.validate_password_strength(password)
    if problems:
        raise ValidationFailed(" ".join(problems), details={"field": "password"})
    if db.scalar(select(User).where(User.email == email)):
        raise Conflict("An account with this email already exists.")
    user = User(email=email, password_hash=security.hash_password(password), full_name=full_name.strip())
    db.add(user)
    db.flush()
    send_verification(db, user)
    audit(db, "auth.register", actor_user_id=user.id, actor_email=email, resource_type="user", resource_id=user.id)
    return user


def _issue_email_token(db: Session, user: User, purpose: str, ttl: timedelta) -> str:
    raw = security.new_token()
    db.add(
        EmailToken(
            user_id=user.id,
            purpose=purpose,
            token_hash=security.sha256_hex(raw),
            expires_at=_now() + ttl,
        )
    )
    return raw


def send_verification(db: Session, user: User) -> None:
    if user.email_verified_at:
        return
    raw = _issue_email_token(db, user, "verify", timedelta(hours=24))
    url = mailer.link(f"/verify-email?token={raw}")
    mailer.send_email(
        user.email,
        "Verify your email address",
        f"Confirm your email address by opening this link (valid 24 hours):\n\n{url}\n",
        link=url,
    )


def _consume_email_token(db: Session, raw: str, purpose: str) -> EmailToken:
    token = db.scalar(
        select(EmailToken).where(
            EmailToken.token_hash == security.sha256_hex(raw), EmailToken.purpose == purpose
        )
    )
    if token is None or token.used_at is not None or token.expires_at <= _now():
        raise ValidationFailed("This link is invalid or has expired.")
    token.used_at = _now()
    return token


def verify_email(db: Session, raw: str) -> User:
    token = _consume_email_token(db, raw, "verify")
    user = db.get(User, token.user_id)
    assert user is not None
    user.email_verified_at = _now()
    audit(db, "auth.email_verified", actor_user_id=user.id, resource_type="user", resource_id=user.id)
    return user


# ------------------------------------------------------------------------- login


def create_session(db: Session, user: User, *, user_agent: str, ip: str, family_id: uuid.UUID | None = None) -> TokenPair:
    settings = get_settings()
    raw = security.new_token(48)
    session = UserSession(
        user_id=user.id,
        refresh_hash=security.sha256_hex(raw),
        family_id=family_id or uuid.uuid4(),
        user_agent=user_agent[:400],
        ip=ip[:64],
        expires_at=_now() + timedelta(days=settings.refresh_token_ttl_days),
    )
    db.add(session)
    db.flush()
    access = security.create_access_token(user_id=user.id, session_id=session.id)
    return TokenPair(access, raw, session.id, settings.access_token_ttl_minutes * 60, user.id)


def authenticate(db: Session, *, email: str, password: str, user_agent: str, ip: str) -> LoginResult:
    settings = get_settings()
    email = normalize_email(email)
    user = db.scalar(select(User).where(User.email == email))
    generic = AuthenticationRequired("Invalid email or password.")
    if user is None:
        security.verify_password(password, None)  # equalise timing
        audit(db, "auth.login", actor_email=email, outcome="failure", meta={"reason": "unknown_email"})
        raise generic
    if user.locked_until and user.locked_until > _now():
        audit(db, "auth.login", actor_user_id=user.id, actor_email=email, outcome="denied", meta={"reason": "locked"})
        raise AuthenticationRequired("Account temporarily locked after repeated failures. Try again later.")
    if not security.verify_password(password, user.password_hash):
        user.failed_logins += 1
        if user.failed_logins >= MAX_FAILED_LOGINS:
            user.locked_until = _now() + timedelta(minutes=LOCKOUT_MINUTES)
            user.failed_logins = 0
        audit(db, "auth.login", actor_user_id=user.id, actor_email=email, outcome="failure", meta={"reason": "bad_password"})
        raise generic
    if user.status != "active":
        audit(db, "auth.login", actor_user_id=user.id, actor_email=email, outcome="denied", meta={"reason": user.status})
        raise PermissionDenied("This account is not active. Contact your administrator.")
    if settings.email_verification_required and not user.email_verified_at:
        raise PermissionDenied("Please verify your email address before signing in.", details={"reason": "email_not_verified"})
    user.failed_logins = 0
    user.locked_until = None
    if user.password_hash and security.password_needs_rehash(user.password_hash):
        user.password_hash = security.hash_password(password)
    if user.mfa_enabled:
        return LoginResult(user=user, mfa_required=True, mfa_token=security.create_purpose_token(user_id=user.id, purpose="mfa"))
    return LoginResult(user=user, tokens=_finish_login(db, user, user_agent=user_agent, ip=ip))


def _finish_login(db: Session, user: User, *, user_agent: str, ip: str) -> TokenPair:
    user.last_login_at = _now()
    tokens = create_session(db, user, user_agent=user_agent, ip=ip)
    audit(db, "auth.login", actor_user_id=user.id, actor_email=user.email, meta={"session_id": tokens.session_id})
    return tokens


def complete_mfa_login(db: Session, *, mfa_token: str, code: str, user_agent: str, ip: str) -> LoginResult:
    try:
        payload = security.decode_token(mfa_token, expected_type="mfa")
    except Exception as exc:  # noqa: BLE001
        raise AuthenticationRequired("MFA challenge expired. Sign in again.") from exc
    user = db.get(User, uuid.UUID(payload["sub"]))
    if user is None or not user.mfa_enabled or not user.mfa_secret_enc or user.status != "active":
        raise AuthenticationRequired("MFA challenge is no longer valid.")
    if not security.verify_totp(security.decrypt_secret(user.mfa_secret_enc), code):
        audit(db, "auth.mfa", actor_user_id=user.id, outcome="failure")
        raise AuthenticationRequired("Invalid authentication code.")
    return LoginResult(user=user, tokens=_finish_login(db, user, user_agent=user_agent, ip=ip))


def refresh(db: Session, raw_refresh: str, *, user_agent: str, ip: str) -> TokenPair:
    current = db.scalar(select(UserSession).where(UserSession.refresh_hash == security.sha256_hex(raw_refresh)))
    if current is None:
        raise AuthenticationRequired("Session not found. Please sign in again.")
    if current.revoked_at is not None:
        if current.revoke_reason == "rotated":
            # A rotated token was replayed: assume theft and kill the whole family.
            revoke_family(db, current.family_id, reason="reuse_detected")
            audit(db, "auth.refresh_reuse", actor_user_id=current.user_id, outcome="denied", meta={"family": current.family_id})
        raise AuthenticationRequired("Session expired. Please sign in again.")
    if current.expires_at <= _now():
        raise AuthenticationRequired("Session expired. Please sign in again.")
    user = db.get(User, current.user_id)
    if user is None or user.status != "active":
        raise AuthenticationRequired("Account is not active.")
    tokens = create_session(db, user, user_agent=user_agent or current.user_agent, ip=ip or current.ip, family_id=current.family_id)
    current.revoked_at = _now()
    current.revoke_reason = "rotated"
    current.replaced_by = tokens.session_id
    return tokens


def revoke_family(db: Session, family_id: uuid.UUID, *, reason: str) -> None:
    db.execute(
        update(UserSession)
        .where(UserSession.family_id == family_id, UserSession.revoked_at.is_(None))
        .values(revoked_at=_now(), revoke_reason=reason)
    )


def revoke_session(db: Session, session_id: uuid.UUID, user_id: uuid.UUID, *, reason: str = "logout") -> None:
    session = db.get(UserSession, session_id)
    if session is None or session.user_id != user_id:
        return
    revoke_family(db, session.family_id, reason=reason)


def revoke_all_sessions(db: Session, user_id: uuid.UUID, *, reason: str, except_family: uuid.UUID | None = None) -> None:
    stmt = update(UserSession).where(UserSession.user_id == user_id, UserSession.revoked_at.is_(None))
    if except_family:
        stmt = stmt.where(UserSession.family_id != except_family)
    db.execute(stmt.values(revoked_at=_now(), revoke_reason=reason))


def active_sessions(db: Session, user_id: uuid.UUID) -> list[UserSession]:
    return list(
        db.scalars(
            select(UserSession)
            .where(UserSession.user_id == user_id, UserSession.revoked_at.is_(None), UserSession.expires_at > _now())
            .order_by(UserSession.created_at.desc())
        )
    )


def validate_access_session(db: Session, session_id: uuid.UUID, user_id: uuid.UUID) -> UserSession:
    session = db.get(UserSession, session_id)
    if session is None or session.user_id != user_id:
        raise AuthenticationRequired("Session not found.")
    # Access tokens of a rotated session stay valid until they expire (≤ TTL); explicit
    # revocations (logout, password change, admin action) take effect immediately.
    if session.revoked_at is not None and session.revoke_reason != "rotated":
        raise AuthenticationRequired("Session has been revoked.")
    if session.expires_at <= _now():
        raise AuthenticationRequired("Session expired.")
    return session


# --------------------------------------------------------------- password reset


def request_password_reset(db: Session, email: str) -> None:
    user = db.scalar(select(User).where(User.email == normalize_email(email)))
    if user is None or user.status != "active":
        return  # do not reveal whether the account exists
    raw = _issue_email_token(db, user, "reset", timedelta(hours=1))
    url = mailer.link(f"/reset-password?token={raw}")
    mailer.send_email(user.email, "Reset your password", f"Reset your password (valid 1 hour):\n\n{url}\n", link=url)
    audit(db, "auth.password_reset_requested", actor_user_id=user.id)


def reset_password(db: Session, raw: str, new_password: str) -> None:
    problems = security.validate_password_strength(new_password)
    if problems:
        raise ValidationFailed(" ".join(problems), details={"field": "password"})
    token = _consume_email_token(db, raw, "reset")
    user = db.get(User, token.user_id)
    assert user is not None
    user.password_hash = security.hash_password(new_password)
    user.email_verified_at = user.email_verified_at or _now()
    user.failed_logins = 0
    user.locked_until = None
    revoke_all_sessions(db, user.id, reason="password_reset")
    audit(db, "auth.password_reset", actor_user_id=user.id)


def change_password(db: Session, user: User, current: str, new_password: str, keep_family: uuid.UUID | None) -> None:
    if not security.verify_password(current, user.password_hash):
        raise ValidationFailed("Current password is incorrect.", details={"field": "current_password"})
    problems = security.validate_password_strength(new_password)
    if problems:
        raise ValidationFailed(" ".join(problems), details={"field": "password"})
    user.password_hash = security.hash_password(new_password)
    revoke_all_sessions(db, user.id, reason="password_changed", except_family=keep_family)
    audit(db, "auth.password_changed", actor_user_id=user.id)


# ------------------------------------------------------------------------------ MFA


def mfa_begin(db: Session, user: User) -> tuple[str, str]:
    if user.mfa_enabled:
        raise Conflict("MFA is already enabled.")
    secret = security.new_totp_secret()
    user.mfa_secret_enc = security.encrypt_secret(secret)
    return secret, security.totp_uri(secret, user.email, get_settings().app_name)


def mfa_confirm(db: Session, user: User, code: str) -> None:
    if not user.mfa_secret_enc or not security.verify_totp(security.decrypt_secret(user.mfa_secret_enc), code):
        raise ValidationFailed("Invalid authentication code.")
    user.mfa_enabled = True
    audit(db, "auth.mfa_enabled", actor_user_id=user.id)


def mfa_disable(db: Session, user: User, password: str, code: str) -> None:
    if not security.verify_password(password, user.password_hash):
        raise ValidationFailed("Password is incorrect.", details={"field": "password"})
    if not user.mfa_secret_enc or not security.verify_totp(security.decrypt_secret(user.mfa_secret_enc), code):
        raise ValidationFailed("Invalid authentication code.")
    user.mfa_enabled = False
    user.mfa_secret_enc = None
    audit(db, "auth.mfa_disabled", actor_user_id=user.id)
