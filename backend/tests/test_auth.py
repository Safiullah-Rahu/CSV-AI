"""Authentication: registration, login, refresh rotation, reuse detection, reset, MFA."""

from __future__ import annotations

import pyotp

from tests.conftest import PASSWORD, unique_email


def test_register_login_and_me(client, make_user):
    api = make_user("Globex")
    me = api.get("/auth/me").json()
    assert me["user"]["email"] == api.email
    assert me["organizations"][0]["name"] == "Globex"
    assert "Organization Owner" in me["organizations"][0]["roles"]


def test_weak_password_rejected(client):
    r = client.post("/api/v1/auth/register", json={"email": unique_email(), "password": "short"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_duplicate_registration_conflict(client, make_user):
    api = make_user()
    r = client.post("/api/v1/auth/register", json={"email": api.email, "password": PASSWORD})
    assert r.status_code == 409


def test_bad_password_and_lockout(client, make_user):
    api = make_user()
    for _ in range(5):
        r = client.post("/api/v1/auth/login", json={"email": api.email, "password": "Wrong-Password-1"})
        assert r.status_code == 401
    r = client.post("/api/v1/auth/login", json={"email": api.email, "password": PASSWORD})
    assert r.status_code == 401 and "locked" in r.json()["error"]["message"]


def test_unknown_email_same_error(client):
    r = client.post("/api/v1/auth/login", json={"email": unique_email(), "password": PASSWORD})
    assert r.status_code == 401
    assert r.json()["error"]["message"] == "Invalid email or password."


def test_refresh_rotation_and_reuse_detection(client, make_user):
    api = make_user()
    r = client.post("/api/v1/auth/login?transport=body", json={"email": api.email, "password": PASSWORD})
    first = r.json()["refresh_token"]
    r2 = client.post("/api/v1/auth/refresh", json={"refresh_token": first})
    assert r2.status_code == 200
    second = r2.json()["refresh_token"]
    assert second != first
    # Replaying the rotated token revokes the whole family …
    assert client.post("/api/v1/auth/refresh", json={"refresh_token": first}).status_code == 401
    # … so the newest token is dead too.
    assert client.post("/api/v1/auth/refresh", json={"refresh_token": second}).status_code == 401


def test_cookie_refresh_requires_csrf_header(client, make_user):
    api = make_user()
    r = client.post("/api/v1/auth/login", json={"email": api.email, "password": PASSWORD})
    assert r.status_code == 200 and "rag_refresh" in r.cookies
    assert client.post("/api/v1/auth/refresh").status_code == 422
    r = client.post("/api/v1/auth/refresh", headers={"X-Requested-With": "fetch"})
    assert r.status_code == 200 and r.json()["access_token"]


def test_logout_revokes_access_token(client, make_user):
    api = make_user()
    assert api.post("/auth/logout").status_code == 200
    assert api.get("/auth/me").status_code == 401


def test_password_reset_flow_revokes_sessions(client, make_user):
    from app.services.email import OUTBOX

    api = make_user()
    assert client.post("/api/v1/auth/password/forgot", json={"email": api.email}).status_code == 202
    token = OUTBOX[-1].link.split("token=")[1]
    new_pw = "Another-Strong-Pass-42"
    assert client.post("/api/v1/auth/password/reset", json={"token": token, "password": new_pw}).status_code == 200
    assert api.get("/auth/me").status_code == 401  # old session revoked
    assert client.post("/api/v1/auth/password/reset", json={"token": token, "password": new_pw}).status_code == 422  # single use
    assert client.post("/api/v1/auth/login", json={"email": api.email, "password": new_pw}).status_code == 200


def test_forgot_password_does_not_reveal_accounts(client):
    from app.services.email import OUTBOX

    r = client.post("/api/v1/auth/password/forgot", json={"email": unique_email()})
    assert r.status_code == 202 and not OUTBOX


def test_email_verification(client, make_user):
    from app.services.email import OUTBOX

    api = make_user()
    link = next(m.link for m in OUTBOX if m.to == api.email and "verify" in m.link)
    r = client.post("/api/v1/auth/verify-email", json={"token": link.split("token=")[1]})
    assert r.status_code == 200
    assert api.get("/auth/me").json()["user"]["email_verified_at"]


def test_mfa_enrolment_and_login(client, make_user):
    api = make_user()
    setup = api.post("/auth/mfa/setup").json()
    totp = pyotp.TOTP(setup["secret"])
    assert api.post("/auth/mfa/confirm", json={"code": "000000"}).status_code == 422
    assert api.post("/auth/mfa/confirm", json={"code": totp.now()}).status_code == 200
    r = client.post("/api/v1/auth/login", json={"email": api.email, "password": PASSWORD})
    body = r.json()
    assert body["mfa_required"] and not body["access_token"]
    bad = client.post("/api/v1/auth/mfa/verify", json={"mfa_token": body["mfa_token"], "code": "123456"})
    assert bad.status_code == 401
    ok = client.post("/api/v1/auth/mfa/verify", json={"mfa_token": body["mfa_token"], "code": totp.now()})
    assert ok.status_code == 200 and ok.json()["access_token"]


def test_sessions_list_and_revoke(client, make_user):
    api = make_user()
    client.post("/api/v1/auth/login", json={"email": api.email, "password": PASSWORD})
    sessions = api.get("/auth/sessions").json()
    assert len(sessions) == 2 and sum(s["current"] for s in sessions) == 1
    other = next(s for s in sessions if not s["current"])
    assert api.delete(f"/auth/sessions/{other['id']}").status_code == 200
    assert len(api.get("/auth/sessions").json()) == 1


def test_security_headers_and_request_id(client):
    r = client.get("/api/v1/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert len(r.headers["x-request-id"]) >= 8
