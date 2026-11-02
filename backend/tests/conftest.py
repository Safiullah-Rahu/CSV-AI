"""Shared pytest fixtures.

Integration tests run against a real PostgreSQL 16 + pgvector (embedded via
``pgserver``; or ``TEST_DATABASE_URL`` if set, e.g. in CI with Docker). Each
test session gets a freshly migrated database and connects as a
NON-superuser role so Row-Level Security is actually enforced.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_TMP = Path(tempfile.mkdtemp(prefix="rag_test_"))

# Environment must be set before the app is imported.
os.environ.update(
    {
        "ENVIRONMENT": "test",
        "EMAIL_BACKEND": "memory",
        "JOB_BACKEND": "eager",
        "SCHEDULER_ENABLED": "false",
        "STORAGE_LOCAL_ROOT": str(_TMP / "storage"),
        "JWT_SECRET": "test-secret-test-secret-test-secret-0123",
        "LOGIN_RATE_LIMIT_PER_MINUTE": "1000",
        "LOG_LEVEL": "WARNING",
        "ALLOW_PRIVATE_WEB_INGESTION": "false",
        "SSRF_ALLOWLIST": "127.0.0.1,localhost",
        "SQLITE_ALLOWED_DIRS": str(_TMP / "sqlite"),
    }
)
(_TMP / "sqlite").mkdir(parents=True, exist_ok=True)
SQLITE_DIR = _TMP / "sqlite"


def _provision_database() -> str:
    if os.environ.get("TEST_DATABASE_URL"):
        return os.environ["TEST_DATABASE_URL"]
    sys.path.insert(0, str(ROOT / "scripts"))
    import devdb  # type: ignore[import-not-found]

    name = f"rag_test_{os.getpid()}"
    devdb.drop(name)
    return devdb.ensure(name)


DB_URL = _provision_database()
os.environ["DATABASE_URL"] = DB_URL


def _migrate() -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", DB_URL)
    command.upgrade(cfg, "head")


_migrate()


@pytest.fixture(scope="session")
def customer_pg() -> dict[str, Any]:
    """A separate PostgreSQL server playing the customer's database (reached over TCP).

    Uses ``TEST_CUSTOMER_PG_ADMIN_URL`` when set (CI service container); otherwise an
    embedded pgserver, which listens on TCP on Windows only.
    """
    import psycopg
    from psycopg.conninfo import conninfo_to_dict

    admin = os.environ.get("TEST_CUSTOMER_PG_ADMIN_URL")
    if not admin:
        import pgserver

        admin = pgserver.get_server(str(_TMP / "customer_pg"), cleanup_mode="stop").get_uri()
    info = conninfo_to_dict(admin)
    host, port = str(info.get("host") or "localhost"), str(info.get("port") or 5432)
    if host.startswith("/"):
        pytest.skip("The customer database must be reachable over TCP: set TEST_CUSTOMER_PG_ADMIN_URL.")
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute("CREATE ROLE shop_reader LOGIN PASSWORD 'Reader-Pass-1' NOSUPERUSER")
        conn.execute("CREATE DATABASE shop OWNER shop_reader")
    host_port = f"{host}:{port}"
    with psycopg.connect(f"postgresql://shop_reader:Reader-Pass-1@{host_port}/shop", autocommit=True) as conn:
        conn.execute("CREATE TABLE customers (id int primary key, name text, country text, ssn text)")
        conn.execute("CREATE TABLE orders (id int primary key, customer_id int references customers(id), amount numeric(10,2), region text, updated_at timestamptz default now())")
        conn.execute("INSERT INTO customers VALUES (1,'Acme','DE','111-22-3333'),(2,'Globex','US','444-55-6666'),(3,'Initech','FR','777-88-9999')")
        conn.execute("INSERT INTO orders (id, customer_id, amount, region, updated_at) VALUES (1,1,1200.50,'EMEA','2026-01-05'),(2,2,980.00,'AMER','2026-01-20'),(3,1,450.25,'EMEA','2026-02-11'),(4,3,2300.00,'EMEA','2026-03-02')")
    return {"host": host, "port": int(port), "database": "shop", "username": "shop_reader", "password": "Reader-Pass-1", "dsn": f"postgresql://shop_reader:Reader-Pass-1@{host_port}/shop"}


@pytest.fixture(scope="session")
def app() -> Any:
    from app.main import app as fastapi_app

    return fastapi_app


@pytest.fixture()
def client(app: Any) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from app.core.ratelimit import get_limiter
    from app.services.email import OUTBOX

    get_limiter().reset()
    OUTBOX.clear()
    with TestClient(app) as c:
        yield c


class Api:
    """Thin authenticated client bound to one user and (optionally) an org."""

    def __init__(self, client: Any, token: str, org_id: str | None, email: str, user_id: str) -> None:
        self.client, self.token, self.org_id, self.email, self.user_id = client, token, org_id, email, user_id

    def headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.token}"}
        if self.org_id:
            h["X-Org-Id"] = self.org_id
        h.update(extra or {})
        return h

    def get(self, path: str, **kw: Any) -> Any:
        return self.client.get(f"/api/v1{path}", headers=self.headers(kw.pop("headers", None)), **kw)

    def post(self, path: str, **kw: Any) -> Any:
        return self.client.post(f"/api/v1{path}", headers=self.headers(kw.pop("headers", None)), **kw)

    def patch(self, path: str, **kw: Any) -> Any:
        return self.client.patch(f"/api/v1{path}", headers=self.headers(kw.pop("headers", None)), **kw)

    def put(self, path: str, **kw: Any) -> Any:
        return self.client.put(f"/api/v1{path}", headers=self.headers(kw.pop("headers", None)), **kw)

    def delete(self, path: str, **kw: Any) -> Any:
        return self.client.request("DELETE", f"/api/v1{path}", headers=self.headers(kw.pop("headers", None)), **kw)

    def in_org(self, org_id: str) -> Api:
        return Api(self.client, self.token, org_id, self.email, self.user_id)


PASSWORD = "Correct-Horse-9-Battery"


def unique_email(tag: str = "user") -> str:
    return f"{tag}-{uuid.uuid4().hex[:10]}@example.com"


@pytest.fixture()
def make_user(client: Any):  # noqa: ANN201
    def _make(org_name: str | None = "Acme Corp", email: str | None = None) -> Api:
        email = email or unique_email()
        r = client.post(
            "/api/v1/auth/register?transport=body",
            json={"email": email, "password": PASSWORD, "full_name": email.split("@")[0], "organization_name": org_name},
        )
        assert r.status_code == 201, r.text
        token = r.json()["access_token"]
        me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
        org_id = me["organizations"][0]["id"] if me["organizations"] else None
        return Api(client, token, org_id, email, me["user"]["id"])

    return _make


@pytest.fixture()
def invite_member(client: Any):  # noqa: ANN201
    """Invite a new user into ``admin``'s org with a role key; returns their Api."""

    def _invite(admin: Api, role_key: str = "member", email: str | None = None) -> Api:
        from app.services.email import OUTBOX

        roles = {r["key"]: r["id"] for r in admin.get("/roles").json()}
        email = email or unique_email(role_key)
        r = admin.post("/invitations", json={"email": email, "role_id": roles[role_key]})
        assert r.status_code == 201, r.text
        link = next(m.link for m in reversed(OUTBOX) if m.to == email)
        token = link.split("token=")[1]
        r = client.post("/api/v1/invitations/accept", json={"token": token, "password": PASSWORD, "full_name": role_key})
        assert r.status_code == 200, r.text
        access = r.json()["access_token"]
        me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access}"}).json()
        return Api(client, access, admin.org_id, email, me["user"]["id"])

    return _invite


# ---------------------------------------------------------------- CSV-AI fixtures


@pytest.fixture()
def sample_csv_bytes() -> bytes:
    """A tiny but realistic CSV with mixed dtypes and a missing value."""
    return (
        b"id,name,score,category\n"
        b"1,Alice,9.5,A\n"
        b"2,Bob,7.0,B\n"
        b"3,Charlie,,A\n"
        b"4,Dana,8.1,B\n"
    )


@pytest.fixture()
def sample_df(sample_csv_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(sample_csv_bytes))
