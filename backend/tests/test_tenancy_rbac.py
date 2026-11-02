"""Tenant isolation (API + PostgreSQL RLS) and RBAC escalation rules."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select, text

from app.db.base import tenant_tables


def test_cross_tenant_org_access_denied(client, make_user):
    a = make_user("Tenant A")
    b = make_user("Tenant B")
    # B tries to act inside A's org.
    r = b.in_org(a.org_id).get("/members")
    assert r.status_code == 403
    r = b.in_org(a.org_id).get("/orgs/current")
    assert r.status_code == 403


def test_rls_hides_other_tenants_rows(client, make_user):
    """Even a query with NO org filter returns only the bound tenant's rows."""
    from app.db.session import tenant_session
    from app.models import Membership, Role, Workspace

    a = make_user("RLS A")
    b = make_user("RLS B")
    with tenant_session(uuid.UUID(a.org_id)) as db:
        orgs_seen = {r.org_id for r in db.scalars(select(Role))}
        assert orgs_seen == {uuid.UUID(a.org_id)}
        ws = {w.org_id for w in db.scalars(select(Workspace))}
        assert ws == {uuid.UUID(a.org_id)}
        assert all(m.org_id == uuid.UUID(a.org_id) for m in db.scalars(select(Membership)))
    with tenant_session(None) as db:
        # No tenant bound: nothing is visible.
        assert db.scalar(select(Role).limit(1)) is None
    assert b.org_id != a.org_id


def test_rls_blocks_cross_tenant_writes(client, make_user):
    from app.db.session import tenant_session
    from app.models import Workspace

    a = make_user("Write A")
    b = make_user("Write B")
    with pytest.raises(Exception, match="row-level security"), tenant_session(uuid.UUID(a.org_id)) as db:
        db.add(Workspace(org_id=uuid.UUID(b.org_id), name="smuggled"))
        db.flush()


def test_every_tenant_table_has_forced_rls(client):
    from app.db.session import get_engine

    with get_engine().connect() as conn:
        rows = conn.execute(text("SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class WHERE relkind='r' AND relnamespace='public'::regnamespace")).all()
    flags = {r.relname: (r.relrowsecurity, r.relforcerowsecurity) for r in rows}
    for table in tenant_tables() + ["organizations"]:
        assert flags[table] == (True, True), table


def test_app_role_is_not_superuser(client):
    from app.db.session import get_engine

    with get_engine().connect() as conn:
        row = conn.execute(text("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname=current_user")).one()
    assert not row.rolsuper and not row.rolbypassrls


def test_audit_log_is_append_only(client, make_user):
    from app.db.session import tenant_session

    a = make_user("Audit Org")
    with pytest.raises(Exception, match="append-only"), tenant_session(uuid.UUID(a.org_id)) as db:
        db.execute(text("UPDATE audit_events SET action='tampered' WHERE org_id = :o"), {"o": a.org_id})


def test_member_cannot_manage_users(client, make_user, invite_member):
    owner = make_user("Perm Org")
    member = invite_member(owner, "member")
    r = member.post("/invitations", json={"email": "x@example.com"})
    assert r.status_code == 403
    assert member.get("/audit").status_code == 403
    # Denials are themselves audited.
    actions = [e["action"] for e in owner.get("/audit?action=authz.denied").json()["items"]]
    assert "authz.denied" in actions


def test_admin_cannot_escalate_to_owner(client, make_user, invite_member):
    owner = make_user("Escalation Org")
    admin = invite_member(owner, "admin")
    roles = {r["key"]: r["id"] for r in owner.get("/roles").json()}
    r = admin.post("/role-assignments", json={"role_id": roles["owner"], "user_id": admin.user_id})
    assert r.status_code == 403
    r = admin.post("/invitations", json={"email": "boss@example.com", "role_id": roles["owner"]})
    assert r.status_code == 403


def test_custom_role_cannot_exceed_creator(client, make_user, invite_member):
    owner = make_user("Custom Role Org")
    sec = invite_member(owner, "security_admin")
    r = sec.post("/roles", json={"name": "Sneaky", "permissions": ["kb.access_all"]})
    assert r.status_code == 403
    r = sec.post("/roles", json={"name": "Auditor", "permissions": ["org.view", "admin.audit"]})
    assert r.status_code == 201, r.text
    assert set(r.json()["permissions"]) == {"org.view", "admin.audit"}


def test_custom_role_assignment_grants_capability(client, make_user, invite_member):
    owner = make_user("Custom Assign Org")
    member = invite_member(owner, "read_only")
    assert member.get("/audit").status_code == 403
    role = owner.post("/roles", json={"name": "Audit viewer", "permissions": ["org.view", "admin.audit"]}).json()
    a = owner.post("/role-assignments", json={"role_id": role["id"], "user_id": member.user_id})
    assert a.status_code == 201
    assert member.get("/audit").status_code == 200
    owner.delete(f"/role-assignments/{a.json()['id']}")
    assert member.get("/audit").status_code == 403


def test_expired_role_assignment_is_ignored(client, make_user, invite_member):
    owner = make_user("Expiry Org")
    member = invite_member(owner, "read_only")
    roles = {r["key"]: r["id"] for r in owner.get("/roles").json()}
    owner.post("/role-assignments", json={"role_id": roles["security_admin"], "user_id": member.user_id, "expires_at": "2020-01-01T00:00:00Z"})
    assert member.get("/audit").status_code == 403


def test_last_owner_protected(client, make_user):
    owner = make_user("Owner Org")
    assignments = owner.get(f"/role-assignments?user_id={owner.user_id}").json()
    r = owner.delete(f"/role-assignments/{assignments[0]['id']}")
    assert r.status_code == 422


def test_suspended_member_loses_access(client, make_user, invite_member):
    owner = make_user("Suspend Org")
    member = invite_member(owner, "member")
    assert member.get("/orgs/current").status_code == 200
    owner.patch(f"/members/{member.user_id}", json={"status": "suspended"})
    assert member.get("/orgs/current").status_code == 403


def test_multi_org_membership_and_switching(client, make_user, invite_member):
    a = make_user("Multi A")
    b = make_user("Multi B")
    joined = invite_member(a, "member", email=None)
    # The invited user creates their own org too and can switch between both.
    r = joined.client.post("/api/v1/orgs", json={"name": "Side Project"}, headers={"Authorization": f"Bearer {joined.token}"})
    assert r.status_code == 201
    orgs = joined.get("/auth/me").json()["organizations"]
    assert {o["name"] for o in orgs} == {"Multi A", "Side Project"}
    assert joined.in_org(r.json()["id"]).get("/orgs/current").status_code == 200
    assert joined.in_org(b.org_id).get("/orgs/current").status_code == 403


def test_workspaces_teams_departments(client, make_user, invite_member):
    owner = make_user("Structure Org")
    member = invite_member(owner, "member")
    ws = owner.post("/workspaces", json={"name": "Finance"}).json()
    dept = owner.post("/departments", json={"name": "HR"}).json()
    team = owner.post("/teams", json={"name": "Payroll", "department_id": dept["id"]}).json()
    assert owner.put(f"/teams/{team['id']}/members", json={"user_ids": [member.user_id]}).status_code == 200
    assert owner.post(f"/workspaces/{ws['id']}/members", json={"user_ids": [member.user_id]}).status_code == 200
    names = {w["name"] for w in owner.get("/workspaces").json()}
    assert {"General", "Finance"} <= names
    assert owner.post("/workspaces", json={"name": "Finance"}).status_code == 409
    assert member.post("/workspaces", json={"name": "Nope"}).status_code == 403
    teams = owner.get("/teams").json()
    assert teams[0]["member_ids"] == [member.user_id]
