"""Password hashing, tokens, JWT, secret encryption and TOTP."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

import jwt
import pyotp
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from app.config import get_settings
from app.utils.errors import ConfigurationError

_hasher = PasswordHasher()

# Pre-computed hash used to equalise timing when the account does not exist.
_DUMMY_HASH = _hasher.hash("timing-equaliser-not-a-password")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and bool(password_hash)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def password_needs_rehash(password_hash: str) -> bool:
    return _hasher.check_needs_rehash(password_hash)


def validate_password_strength(password: str) -> list[str]:
    settings = get_settings()
    problems = []
    if len(password) < settings.password_min_length:
        problems.append(f"Password must be at least {settings.password_min_length} characters.")
    classes = sum(
        [
            any(c.islower() for c in password),
            any(c.isupper() for c in password),
            any(c.isdigit() for c in password),
            any(not c.isalnum() for c in password),
        ]
    )
    if classes < 3:
        problems.append("Use at least three of: lowercase, uppercase, digits, symbols.")
    return problems


# --- opaque tokens -------------------------------------------------------------


def new_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


# --- JWT access tokens -----------------------------------------------------------


def create_access_token(*, user_id: uuid.UUID, session_id: uuid.UUID, extra: dict[str, Any] | None = None) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "sid": str(session_id),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.access_token_ttl_minutes)).timestamp()),
        "typ": "access",
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_purpose_token(*, user_id: uuid.UUID, purpose: str, minutes: int = 5, extra: dict[str, Any] | None = None) -> str:
    """Short-lived JWT for intermediate steps (e.g. MFA challenge, OAuth state).

    ``extra`` claims are signed, not encrypted: encrypt anything secret first.
    """
    settings = get_settings()
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        **(extra or {}),
        "sub": str(user_id),
        "typ": purpose,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=minutes)).timestamp()),
        "jti": new_token(8),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str, *, expected_type: str = "access") -> dict[str, Any]:
    settings = get_settings()
    payload = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
        options={"require": ["exp", "sub", "typ"]},
    )
    if payload.get("typ") != expected_type:
        raise jwt.InvalidTokenError("unexpected token type")
    return payload


# --- secret encryption (credentials at rest) ----------------------------------------


@lru_cache(maxsize=1)
def _fernet() -> MultiFernet:
    settings = get_settings()
    keys = list(settings.encryption_keys)
    if not keys:
        if settings.is_production:
            raise ConfigurationError("ENCRYPTION_KEYS is required in production.")
        # Deterministic dev/test key derived from the JWT secret. Never used in prod.
        derived = base64.urlsafe_b64encode(hashlib.sha256(settings.jwt_secret.encode()).digest())
        keys = [derived.decode()]
    return MultiFernet([Fernet(k.encode() if isinstance(k, str) else k) for k in keys])


def encrypt_secret(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_secret(ciphertext: str) -> str:
    try:
        return _fernet().decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise ConfigurationError("Stored credential could not be decrypted (key rotated?).") from exc


def rotate_secret(ciphertext: str) -> str:
    """Re-encrypt with the primary key (after adding a new key to ENCRYPTION_KEYS)."""
    return _fernet().rotate(ciphertext.encode("ascii")).decode("ascii")


def reset_crypto_cache() -> None:
    _fernet.cache_clear()


# --- TOTP (MFA) ----------------------------------------------------------------------


def new_totp_secret() -> str:
    return pyotp.random_base32()


def totp_uri(secret: str, email: str, issuer: str) -> str:
    return pyotp.TOTP(secret).provisioning_uri(name=email, issuer_name=issuer)


def verify_totp(secret: str, code: str) -> bool:
    code = (code or "").strip().replace(" ", "")
    if not code.isdigit():
        return False
    return pyotp.TOTP(secret).verify(code, valid_window=1)


def generate_api_key() -> tuple[str, str, str]:
    """Return (full_key, prefix, sha256 hash). Only the hash is stored."""
    prefix = "rk_" + secrets.token_hex(6)
    body = secrets.token_urlsafe(32)
    full = f"{prefix}.{body}"
    return full, prefix, sha256_hex(full)
