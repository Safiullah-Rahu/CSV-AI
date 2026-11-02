"""Database connections (live PostgreSQL + SQLite), Text-to-SQL, sync, approvals, datasets."""

from __future__ import annotations

import io
import re
import sqlite3
from collections.abc import Iterable, Iterator

import psycopg
import pytest

from app.llm.base import ChatResponse, LLMProvider, Message
from tests.conftest import SQLITE_DIR


class ScriptedSQL(LLMProvider):
    """Deterministic 'model': maps question keywords to SQL (records prompts)."""

    name = "scripted"
    prompts: list[str] = []

    def __init__(self, answers: dict[str, str]) -> None:
        super().__init__(model="scripted")
        self.answers = answers

    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        msgs = list(messages)
        ScriptedSQL.prompts.append(msgs[0].content)
        question = msgs[1].content.lower()
        for key, sql in self.answers.items():
            if key in question:
                return ChatResponse(content=f"```sql\n{sql}\n```", model="scripted")
        return ChatResponse(content="CANNOT_ANSWER", model="scripted")

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        yield self.complete(messages).content


def use_llm(monkeypatch: pytest.MonkeyPatch, answers: dict[str, str]) -> None:
    from app.api.v1 import databases
    from app.llm import registry

    monkeypatch.setattr(databases, "resolve_llm", lambda *a, **k: registry.ResolvedLLM(ScriptedSQL(answers), None, "scripted"))


@pytest.fixture()
def pg_conn(make_user, customer_pg):  # noqa: ANN001, ANN201
    owner = make_user("Data Co")
    kb = owner.post("/knowledge-bases", json={"name": "Sales KB"}).json()
    r = owner.post("/database-connections", json={"name": "Shop DB", "provider": "postgresql", "host": customer_pg["host"], "port": customer_pg["port"], "database": "shop", "username": "shop_reader", "password": customer_pg["password"], "ssl_mode": "disable", "access_mode": "hybrid", "kb_id": kb["id"]})
    assert r.status_code == 201, r.text
    return owner, r.json(), kb


def test_connection_lifecycle_and_no_secret_leak(pg_conn, customer_pg):
    owner, conn, _ = pg_conn
    assert "password" not in conn and conn["has_credentials"] is True
    assert customer_pg["password"] not in str(owner.get("/database-connections").json())
    t = owner.post(f"/database-connections/{conn['id']}/test").json()
    assert t["ok"] is True and t["latency_ms"] > 0
    d = owner.post(f"/database-connections/{conn['id']}/discover").json()
    assert d["tables"] == 2
    schema = owner.get(f"/database-connections/{conn['id']}/schema").json()
    orders = schema["tables"]["public.orders"]
    assert orders["primary_key"] == ["id"] and orders["foreign_keys"][0]["references"] == "public.customers"
    events = [e["type"] for e in owner.get("/events?limit=50").json()["items"]] if owner.get("/events").status_code == 200 else []
    assert not events or "connection.tested" in events


def test_wrong_password_reports_clean_error(pg_conn):
    owner, conn, _ = pg_conn
    # The embedded test server trusts local logins, so force a failure with an unknown role.
    owner.patch(f"/database-connections/{conn['id']}", json={"password": "wrong-password-xyz", "username": "no_such_role"})
    t = owner.post(f"/database-connections/{conn['id']}/test").json()
    assert t["ok"] is False and "wrong-password-xyz" not in t["error"]
    assert owner.get(f"/database-connections/{conn['id']}").json()["status"] == "failed"


def test_ssrf_guard_blocks_platform_db_and_metadata(make_user):
    from sqlalchemy.engine import make_url

    from app.config import get_settings

    owner = make_user("SSRF Co")
    platform = make_url(get_settings().database_url)
    for host, port in ((platform.host, platform.port), ("169.254.169.254", 5432)):
        c = owner.post("/database-connections", json={"name": f"bad {host}", "provider": "postgresql", "host": host, "port": port, "database": "rag", "username": "rag", "password": "rag"}).json()
        t = owner.post(f"/database-connections/{c['id']}/test").json()
        assert t["ok"] is False and t["code"] == "unsafe_request", t


def _select(owner, conn_id):  # noqa: ANN001, ANN202
    return owner.put(f"/database-connections/{conn_id}/selection", json={"tables": {
        "public.orders": {"columns": ["id", "customer_id", "amount", "region", "updated_at"], "incremental_column": "updated_at"},
        "public.customers": {"columns": ["id", "name", "country", "ssn"], "sensitive": ["ssn"]},
    }})


def test_live_text_to_sql_with_permissions(pg_conn, invite_member, monkeypatch):
    owner, conn, _ = pg_conn
    owner.post(f"/database-connections/{conn['id']}/discover")
    assert _select(owner, conn["id"]).status_code == 200
    use_llm(monkeypatch, {
        "revenue by region": "SELECT region, SUM(amount) AS total FROM public.orders GROUP BY region ORDER BY total DESC",
        "ssn": "SELECT name, ssn FROM public.customers",
        "delete": "DELETE FROM public.orders",
    })
    r = owner.post(f"/database-connections/{conn['id']}/query", json={"question": "Show revenue by region"}).json()
    assert r["status"] == "executed", r
    assert r["columns"] == ["region", "total"] and float(r["rows"][0][1]) == pytest.approx(3950.75)
    assert r["chart"]["type"] == "bar"
    assert "ssn" not in ScriptedSQL.prompts[-1]  # sensitive columns never reach the model
    ssn_attempt = owner.post(f"/database-connections/{conn['id']}/query", json={"question": "list ssn"}).json()
    assert ssn_attempt["status"] == "rejected", ssn_attempt
    assert owner.post(f"/database-connections/{conn['id']}/query", json={"question": "delete everything"}).json()["status"] == "rejected"
    assert owner.post(f"/database-connections/{conn['id']}/query", json={"sql": "SELECT * FROM public.customers"}).json()["columns"] == ["id", "name", "country"]
    # An analyst without a grant on the connection cannot query it.
    analyst = invite_member(owner, "analyst")
    assert analyst.post(f"/database-connections/{conn['id']}/query", json={"sql": "SELECT 1 FROM public.orders"}).status_code == 404
    # Grant limited to one table.
    owner.post("/grants", json={"resource_type": "db_connection", "resource_id": conn["id"], "subject_type": "user", "subject_id": analyst.user_id, "privilege": "read", "constraints": {"tables": ["public.orders"]}})
    ok = analyst.post(f"/database-connections/{conn['id']}/query", json={"sql": "SELECT region FROM public.orders"}).json()
    assert ok["status"] == "executed"
    denied = analyst.post(f"/database-connections/{conn['id']}/query", json={"sql": "SELECT name FROM public.customers"}).json()
    assert denied["status"] == "rejected"
    logs = owner.get("/query-logs").json()["items"]
    assert {"executed", "rejected"} <= {entry["status"] for entry in logs}


def test_row_filter_enforced_at_source(pg_conn):
    owner, conn, _ = pg_conn
    owner.post(f"/database-connections/{conn['id']}/discover")
    owner.put(f"/database-connections/{conn['id']}/selection", json={"tables": {"public.orders": {"columns": ["id", "amount", "region"], "row_filter": "region = 'EMEA'"}}})
    r = owner.post(f"/database-connections/{conn['id']}/query", json={"sql": "SELECT count(*) AS n FROM public.orders"}).json()
    assert r["rows"][0][0] == 3
    bad = owner.put(f"/database-connections/{conn['id']}/selection", json={"tables": {"public.orders": {"row_filter": "pg_sleep(5) IS NULL"}}})
    assert bad.status_code == 422


def test_sensitive_table_requires_approval(pg_conn, invite_member):
    owner, conn, _ = pg_conn
    owner.post(f"/database-connections/{conn['id']}/discover")
    owner.put(f"/database-connections/{conn['id']}/selection", json={"tables": {"public.customers": {"columns": ["id", "name", "country"], "requires_approval": True}}})
    pending = owner.post(f"/database-connections/{conn['id']}/query", json={"sql": "SELECT name FROM public.customers"}).json()
    assert pending["status"] == "pending_approval" and pending["approval_id"]
    # Self-approval is not allowed; a second admin approves.
    assert owner.post(f"/approvals/{pending['approval_id']}/decide", json={"approve": True}).status_code == 403
    admin = invite_member(owner, "admin")
    decision = admin.post(f"/approvals/{pending['approval_id']}/decide", json={"approve": True}).json()
    assert decision["status"] == "approved" and len(decision["result"]["rows"]) == 3
    # A request still pending when its connection is deleted expires instead of staying decidable.
    again = owner.post(f"/database-connections/{conn['id']}/query", json={"sql": "SELECT country FROM public.customers"}).json()
    assert again["status"] == "pending_approval"
    assert owner.delete(f"/database-connections/{conn['id']}", json={"confirm_name": conn["name"]}).status_code == 200
    assert admin.post(f"/approvals/{again['approval_id']}/decide", json={"approve": True}).status_code in (404, 409, 422)


def test_sync_full_incremental_and_deletion(pg_conn, customer_pg):
    owner, conn, kb = pg_conn
    owner.post(f"/database-connections/{conn['id']}/discover")
    owner.put(f"/database-connections/{conn['id']}/selection", json={"tables": {"public.orders": {"columns": ["id", "customer_id", "amount", "region", "updated_at"], "incremental_column": "updated_at"}}})
    job = owner.post(f"/database-connections/{conn['id']}/sync", json={"mode": "full"}).json()
    assert owner.get(f"/jobs/{job['job_id']}").json()["status"] == "succeeded"
    docs = owner.get(f"/documents?kb_id={kb['id']}").json()["items"]
    assert len(docs) == 1 and docs[0]["source_type"] == "database" and docs[0]["status"] == "ready"
    ans = owner.post("/chat/ask", json={"question": "What was the amount of order 4?", "kb_ids": [kb["id"]]}).json()
    assert "2300" in ans["content"]
    with psycopg.connect(customer_pg["dsn"], autocommit=True) as c:
        c.execute("INSERT INTO orders (id, customer_id, amount, region, updated_at) VALUES (5, 2, 777.00, 'AMER', '2026-04-01')")
        c.execute("UPDATE orders SET amount = 999.99, updated_at = '2026-04-02' WHERE id = 1")
        c.execute("DELETE FROM orders WHERE id = 2")
    job = owner.post(f"/database-connections/{conn['id']}/sync", json={"mode": "incremental"}).json()
    j = owner.get(f"/jobs/{job['job_id']}").json()
    assert j["status"] == "succeeded", j
    assert j["result"]["added"] == 1 and j["result"]["updated"] == 1 and j["result"]["deleted"] == 1
    runs = owner.get(f"/database-connections/{conn['id']}/runs").json()
    assert runs[0]["items_deleted"] == 1
    ans = owner.post("/chat/ask", json={"question": "What was the amount of order 1?", "kb_ids": [kb["id"]]}).json()
    assert "999.99" in ans["content"]


def test_sqlite_connection(make_user):
    path = SQLITE_DIR / "inventory.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE IF NOT EXISTS items (sku TEXT PRIMARY KEY, stock INT)")
    con.execute("INSERT OR REPLACE INTO items VALUES ('A1', 40), ('B2', 7)")
    con.commit()
    con.close()
    owner = make_user("SQLite Co")
    c = owner.post("/database-connections", json={"name": "Inventory", "provider": "sqlite", "database": str(path)}).json()
    assert owner.post(f"/database-connections/{c['id']}/test").json()["ok"]
    assert owner.post(f"/database-connections/{c['id']}/discover").json()["tables"] == 1
    owner.put(f"/database-connections/{c['id']}/selection", json={"tables": {"items": {}}})
    r = owner.post(f"/database-connections/{c['id']}/query", json={"sql": "SELECT sku FROM items WHERE stock < 10"}).json()
    assert r["rows"] == [["B2"]]
    # Files outside the approved directory are refused (no arbitrary file reads).
    bad = owner.post("/database-connections", json={"name": "Etc", "provider": "sqlite", "database": "C:/Windows/win.ini"}).json()
    assert owner.post(f"/database-connections/{bad['id']}/test").json()["code"] == "unsafe_request"


def test_planned_provider_is_not_faked(make_user):
    owner = make_user("Planned Co")
    providers = {p["key"]: p for p in owner.get("/database-connections/providers").json()}
    assert providers["postgresql"]["status"] == "tested" and providers["snowflake"]["status"] == "planned"
    c = owner.post("/database-connections", json={"name": "Snow", "provider": "snowflake", "host": "acct.snowflakecomputing.com"}).json()
    assert c["status"] == "disabled"
    t = owner.post(f"/database-connections/{c['id']}/test").json()
    assert t["ok"] is False and t["code"] == "not_available"


def test_csv_dataset_profile_analysis_and_sql(make_user, monkeypatch):
    owner = make_user("Dataset Co")
    kb = owner.post("/knowledge-bases", json={"name": "Finance data"}).json()
    csv = b"order_id,customer,amount,month\n1,Acme,1200.50,2026-01\n2,Globex,980.00,2026-01\n3,Acme,450.25,2026-02\n4,Initech,2300.00,2026-03\n"
    owner.post(f"/knowledge-bases/{kb['id']}/documents", files=[("files", ("orders.csv", io.BytesIO(csv), "text/csv"))])
    datasets = owner.get("/datasets").json()
    assert len(datasets) == 1 and datasets[0]["row_count"] == 4
    ds_id = datasets[0]["id"]
    detail = owner.get(f"/datasets/{ds_id}").json()
    cols = {c["name"]: c for c in detail["profile"]["tables"][0]["profile"]["columns"]}
    assert cols["amount"]["mean"] == pytest.approx(1232.6875)
    analysis = owner.get(f"/datasets/{ds_id}/analysis").json()
    assert analysis["rows"] == 4 and analysis["top_values"]["customer"][0]["value"] == "Acme"
    summary = owner.post(f"/datasets/{ds_id}/summary").json()
    assert summary["mode"] == "deterministic" and "4 rows" in summary["summary"]
    r = owner.post(f"/datasets/{ds_id}/query", json={"sql": "SELECT customer, SUM(amount) AS total FROM orders GROUP BY customer ORDER BY total DESC"}).json()
    assert r["status"] == "executed" and r["rows"][0][0] == "Initech"
    blocked = owner.post(f"/datasets/{ds_id}/query", json={"sql": "SELECT * FROM read_csv('C:/Windows/win.ini')"}).json()
    assert blocked["status"] == "rejected"
    use_llm(monkeypatch, {"per month": "SELECT month, SUM(amount) AS revenue FROM orders GROUP BY month ORDER BY month"})
    nl = owner.post(f"/datasets/{ds_id}/query", json={"question": "Total revenue per month?"}).json()
    assert nl["status"] == "executed" and nl["chart"]["type"] == "line" and len(nl["rows"]) == 3


def test_structured_answers_in_chat(make_user, monkeypatch):
    from app.llm import registry
    from app.rag import engine

    owner = make_user("Structured Chat Co")
    kb = owner.post("/knowledge-bases", json={"name": "Orders"}).json()
    owner.post(f"/knowledge-bases/{kb['id']}/documents", files=[("files", ("orders.csv", io.BytesIO(b"order_id,customer,amount\n1,Acme,10\n2,Acme,15\n3,Globex,7\n"), "text/csv"))])

    class Model(LLMProvider):
        name = "model"

        def complete(self, messages: Iterable[Message]) -> ChatResponse:
            return ChatResponse(content="```sql\nSELECT customer, SUM(amount) AS total FROM orders GROUP BY customer ORDER BY total DESC\n```", model="m")

        def stream(self, messages: Iterable[Message]) -> Iterator[str]:
            text = list(messages)[-1].content
            total = re.search(r"\| Acme \| (\d+(?:\.\d+)?) \|", text)
            yield f"Acme has the highest total, {total.group(1) if total else '?'} [S1]."

    monkeypatch.setattr(engine, "resolve_llm", lambda *a, **k: registry.ResolvedLLM(Model(model="m"), None, "model"))
    ans = owner.post("/chat/ask", json={"question": "What is the total amount by customer?", "kb_ids": [kb["id"]]}).json()
    assert "25" in ans["content"], ans
    first = ans["citations"][0]
    assert first["source_type"] == "dataset" and first["cited"] and "SUM(amount)" in first["locator"]["sql"]
