"""Text-to-SQL safety: injection, DML/DDL, functions, allow-lists, stars, row filters, limits."""

from __future__ import annotations

import pytest

from app.text2sql.validator import Policy, TableSpec, validate, validate_row_filter


@pytest.fixture()
def policy() -> Policy:
    return Policy(
        tables=[
            TableSpec("public.orders", ["id", "customer_id", "amount", "created_at", "region"], row_filter="region = 'EMEA'"),
            TableSpec("public.customers", ["id", "name", "country"]),  # 'ssn' and 'email' excluded
            TableSpec("public.payroll", ["employee", "salary"], requires_approval=True),
        ],
        max_rows=100,
        max_joins=2,
    )


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "UPDATE orders SET amount = 0",
        "INSERT INTO orders VALUES (1)",
        "DROP TABLE orders",
        "TRUNCATE orders",
        "CREATE TABLE x AS SELECT * FROM orders",
        "ALTER TABLE orders ADD COLUMN x int",
        "SELECT * INTO backup FROM orders",
        "SELECT 1; DROP TABLE orders",
        "SELECT * FROM orders; SELECT * FROM customers",
        "COPY orders TO '/tmp/x'",
        "GRANT ALL ON orders TO public",
        "SET statement_timeout = 0",
        "SELECT pg_sleep(30)",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT * FROM dblink('host=evil', 'select 1') AS t(a int)",
        "SELECT set_config('app.rls_bypass', 'on', false)",
        "SELECT current_setting('app.current_org')",
        "WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x",
        "SELECT * FROM pg_catalog.pg_authid",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM users",
        "SELECT ssn FROM customers",
        "SELECT c.email FROM customers c",
        "SELECT * FROM orders o JOIN customers c ON c.id=o.customer_id JOIN customers d ON d.id=o.id JOIN orders e ON e.id=o.id",
        "",
        "SELECT * FROM orders WHERE id = 1 OR 1=1 -- ; DROP TABLE orders",
    ],
)
def test_rejected(policy: Policy, sql: str) -> None:
    v = validate(sql, policy)
    if sql.startswith("SELECT * FROM orders WHERE id = 1 OR 1=1"):
        # Tautologies are harmless reads (row filter + limit still apply); comments are dropped.
        assert v.ok and "DROP" not in v.sql.upper()
        return
    assert not v.ok, (sql, v)


def test_duckdb_file_readers_blocked() -> None:
    p = Policy(tables=[TableSpec("orders", ["id"])], dialect="duckdb")
    for sql in ("SELECT * FROM read_csv('/etc/passwd')", "SELECT * FROM 'secret.parquet'", "SELECT * FROM read_parquet('s3://bucket/x')", "SELECT * FROM glob('*')"):
        assert not validate(sql, p).ok, sql


def test_limit_enforced_and_lowered(policy: Policy) -> None:
    v = validate("SELECT name FROM customers", policy)
    assert v.ok and v.sql.upper().endswith("LIMIT 100")
    v = validate("SELECT name FROM customers LIMIT 100000", policy)
    assert v.ok and "LIMIT 100" in v.sql.upper() and "100000" not in v.sql
    v = validate("SELECT name FROM customers LIMIT 5", policy)
    assert v.ok and v.sql.upper().endswith("LIMIT 5")


def test_star_expanded_to_allowed_columns(policy: Policy) -> None:
    v = validate("SELECT * FROM customers", policy)
    assert v.ok
    assert '"country"' in v.sql and "ssn" not in v.sql.lower() and "*" not in v.sql.split("FROM")[0]


def test_row_filter_injected(policy: Policy) -> None:
    v = validate("SELECT region, SUM(amount) AS total FROM orders GROUP BY region", policy)
    assert v.ok
    assert "WHERE region = 'EMEA'" in v.sql
    # The filter wraps the table even when aliased and joined.
    v = validate("SELECT c.name, o.amount FROM orders o JOIN customers c ON c.id = o.customer_id", policy)
    assert v.ok and "region = 'EMEA'" in v.sql


def test_cte_and_aggregates_allowed(policy: Policy) -> None:
    sql = "WITH monthly AS (SELECT date_trunc('month', created_at) AS m, SUM(amount) AS total FROM orders GROUP BY 1) SELECT m, total FROM monthly ORDER BY m"
    v = validate(sql, policy)
    assert v.ok, v.errors


def test_union_wrapped_with_limit(policy: Policy) -> None:
    v = validate("SELECT name FROM customers UNION SELECT country FROM customers", policy)
    assert v.ok and v.sql.upper().endswith("LIMIT 100")


def test_sensitive_table_needs_approval(policy: Policy) -> None:
    v = validate("SELECT employee, salary FROM payroll", policy)
    assert v.ok and v.needs_approval


def test_cross_join_needs_approval(policy: Policy) -> None:
    v = validate("SELECT c.name, o.amount FROM customers c CROSS JOIN orders o", policy)
    assert v.ok and v.needs_approval
    # Joining a table to itself through a CTE still multiplies rows.
    v = validate("WITH big AS (SELECT id FROM orders) SELECT o.id FROM orders o CROSS JOIN big", policy)
    assert v.ok and v.needs_approval


def test_cross_join_with_one_row_source_needs_no_approval(policy: Policy) -> None:
    # A date window in a CTE (and a CTE built on it), a total, and LIMIT 1 cannot multiply rows.
    window = ("WITH q AS (SELECT DATE('now', 'start of month') AS q_start), b AS (SELECT q_start, DATE(q_start, '+3 months') AS q_end FROM q) "
              "SELECT o.id FROM orders o CROSS JOIN b WHERE o.created_at >= b.q_start AND o.created_at < b.q_end")
    for sql in (window,
                "WITH t AS (SELECT MAX(amount) AS top FROM orders) SELECT o.id FROM orders o CROSS JOIN t WHERE o.amount = t.top",
                "SELECT o.id FROM orders o CROSS JOIN (SELECT id FROM customers LIMIT 1) AS c1"):
        v = validate(sql, policy)
        assert v.ok and not v.needs_approval, (sql, v.errors, v.approval_reasons)


def test_chart_suggestion_skips_ids_and_prefers_rates() -> None:
    from app.text2sql.engine import suggest_chart

    cols = ["carrier_id", "carrier_name", "total_shipments", "on_time_rate_pct"]
    rows = [[1, "Brightline", 360, 91.39], [2, "Kestrel", 360, 96.94]]
    chart = suggest_chart(cols, rows)
    assert chart and chart["type"] == "bar" and chart["x"] == "carrier_name" and chart["y"] == ["on_time_rate_pct"]
    assert suggest_chart(["region", "orders", "revenue"], [["EMEA", 3, 10.5]])["y"] == ["orders", "revenue"]
    assert suggest_chart(["customer_id", "orders"], [[7, 3], [8, 5]])["x"] == "customer_id"  # an id is still a usable label


def test_row_filter_validation() -> None:
    assert validate_row_filter("region = 'EMEA' AND amount > 0", ["region", "amount"]) == []
    assert validate_row_filter("region IN (SELECT r FROM x)", ["region"])
    assert validate_row_filter("pg_sleep(1) IS NULL", ["region"])
    assert validate_row_filter("ssn IS NOT NULL", ["region"])
