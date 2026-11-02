"""Organisation security policy (required MFA, sign-in age) and settings validation."""

from __future__ import annotations

import uuid
from datetime import timedelta

import pyotp
from sqlalchemy import update

from tests.conftest import PASSWORD, Api


def _enable_mfa(api: Api) -> None:
    secret = api.post("/auth/mfa/setup").json()["secret"]
    assert api.post("/auth/mfa/confirm", json={"code": pyotp.TOTP(secret).now()}).status_code == 200


def _security(api: Api, **values: object):  # noqa: ANN202
    return api.patch("/orgs/current/settings", json={"section": "security", "values": values})


def test_mfa_requirement_blocks_members_until_enrolled(client, make_user, invite_member):
    owner = make_user("MFA Policy Org")
    member = invite_member(owner)

    # An admin without MFA cannot require it (they would lock themselves out).
    r = _security(owner, mfa_required=True)
    assert r.status_code == 422 and r.json()["error"]["details"]["field"] == "mfa_required"
    _enable_mfa(owner)
    assert _security(owner, mfa_required=True).status_code == 200

    blocked = member.get("/orgs/current")
    assert blocked.status_code == 403
    assert blocked.json()["error"]["details"]["reason"] == "mfa_required"
    assert member.get("/knowledge-bases").status_code == 403
    # Account-level endpoints stay reachable so the member can enrol.
    assert member.get("/auth/me").status_code == 200
    _enable_mfa(member)
    assert member.get("/orgs/current").status_code == 200
    assert owner.get("/orgs/current").status_code == 200


def test_mfa_requirement_applies_to_api_keys(client, make_user, invite_member):
    owner = make_user("MFA Key Org")
    admin = invite_member(owner, "admin")

    def key_for(api: Api) -> Api:
        r = api.post("/api-keys", json={"name": "ci", "scopes": ["org.view", "kb.view"]})
        assert r.status_code == 201, r.text
        return Api(client, r.json()["key"], owner.org_id, api.email, api.user_id)

    owner_key, admin_key = key_for(owner), key_for(admin)
    _enable_mfa(owner)
    assert _security(owner, mfa_required=True).status_code == 200
    assert owner_key.get("/knowledge-bases").status_code == 200
    # A key acts as its owner, so a key minted by a member without MFA stops working.
    r = admin_key.get("/knowledge-bases")
    assert r.status_code == 403 and r.json()["error"]["details"]["reason"] == "mfa_required"


def test_session_lifetime_forces_sign_in_again(client, make_user):
    from app.db.session import system_session
    from app.models import UserSession

    owner = make_user("TTL Policy Org")
    assert _security(owner, session_ttl_days=1).status_code == 200
    assert owner.get("/orgs/current").status_code == 200

    with system_session() as db:
        db.execute(
            update(UserSession)
            .where(UserSession.user_id == uuid.UUID(owner.user_id))
            .values(created_at=UserSession.created_at - timedelta(days=2))
        )
        db.commit()

    r = owner.get("/orgs/current")
    assert r.status_code == 403 and r.json()["error"]["details"]["reason"] == "reauth_required"
    assert owner.get("/auth/me").status_code == 200  # account itself is still signed in

    fresh = client.post("/api/v1/auth/login", json={"email": owner.email, "password": PASSWORD}).json()["access_token"]
    assert Api(client, fresh, owner.org_id, owner.email, owner.user_id).get("/orgs/current").status_code == 200


def test_org_settings_are_validated(client, make_user):
    owner = make_user("Settings Validation Org")

    def patch(section: str, **values: object) -> int:
        return owner.patch("/orgs/current/settings", json={"section": section, "values": values}).status_code

    assert patch("security", session_ttl_days=0) == 422
    assert patch("security", session_ttl_days="7") == 422
    assert patch("security", surprise=True) == 422
    assert patch("retention", audit_days=5) == 422
    assert patch("ai", grounding_policy="anything-goes") == 422
    assert patch("branding", accent="blue") == 422
    assert patch("branding", logo_url="http://cdn.example.com/logo.png") == 422
    assert patch("branding", logo_url="javascript:alert(1)") == 422
    assert patch("locale", language="en") == 422  # unknown section

    assert patch("branding", accent="#0f766e", logo_url="https://cdn.example.com/logo.svg") == 200
    assert patch("retention", conversations_days=None, audit_days=400) == 200
    branding = owner.get("/orgs/current").json()["organization"]["settings"]["branding"]
    assert branding == {"accent": "#0f766e", "logo_url": "https://cdn.example.com/logo.svg"}
    assert patch("branding", accent=None, logo_url=None) == 200
