"""Natural language → validated SQL → read-only execution → explanation + chart.

Sources are either uploaded datasets (queried with an in-memory DuckDB
sandbox: Arrow tables registered in memory, external file/network access
disabled and the configuration locked) or live database connections
(read-only transaction, server-side statement timeout, row cap).

Generated SQL is never executed unless :func:`validator.validate` accepts it
under a policy built from the administrator's table selection *and* the
requesting user's grant constraints. Sensitive or expensive queries are held
for human approval. Every attempt is written to ``sql_query_logs``.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.connectors.databases import QueryResult
from app.llm.base import Message as LLMMessage
from app.llm.registry import ResolvedLLM
from app.models import Approval, SqlQueryLog, UsageRecord
from app.text2sql.validator import Policy, TableSpec, Validation, validate
from app.utils.errors import ConnectorError, CSVAIError, NotAvailable, ValidationFailed

_FENCE = re.compile(r"```(?:sql)?\s*(.+?)```", re.S | re.I)


@dataclass
class SqlSource:
    kind: str                        # dataset | database
    id: uuid.UUID
    name: str
    dialect: str
    tables: list[TableSpec]
    schema_text: str
    execute: Callable[[str, int, int], QueryResult]
    max_rows: int = 500
    timeout_s: int = 15
    max_joins: int = 4
    extra: dict[str, Any] = field(default_factory=dict)

    def policy(self) -> Policy:
        return Policy(tables=self.tables, dialect=self.dialect, max_rows=self.max_rows, max_joins=self.max_joins)


@dataclass
class SqlOutcome:
    status: str                      # executed | rejected | pending_approval | failed | cannot_answer
    question: str
    generated_sql: str = ""
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    truncated: bool = False
    duration_ms: float | None = None
    errors: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    explanation: str = ""
    chart: dict[str, Any] | None = None
    log_id: uuid.UUID | None = None
    approval_id: uuid.UUID | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "question": self.question, "generated_sql": self.generated_sql, "sql": self.sql,
            "columns": self.columns, "rows": self.rows, "row_count": len(self.rows), "truncated": self.truncated,
            "duration_ms": round(self.duration_ms or 0, 1), "errors": self.errors, "approval_reasons": self.reasons,
            "explanation": self.explanation, "chart": self.chart, "log_id": str(self.log_id) if self.log_id else None,
            "approval_id": str(self.approval_id) if self.approval_id else None,
        }


# ---------------------------------------------------------------------- executors


def duckdb_executor(tables: list[tuple[str, str]]) -> Callable[[str, int, int], QueryResult]:
    """Executor over Parquet files registered as in-memory Arrow tables."""

    def run(sql: str, max_rows: int, timeout_s: int) -> QueryResult:
        import duckdb
        import pyarrow.parquet as pq

        start = time.perf_counter()
        con = duckdb.connect(":memory:")
        try:
            for name, path in tables:
                con.register(name, pq.read_table(path))
            con.execute("SET enable_external_access = false")
            con.execute("SET lock_configuration = true")
            timer = threading.Timer(timeout_s, con.interrupt)
            timer.start()
            try:
                cur = con.execute(sql)
                columns = [d[0] for d in cur.description or []]
                rows = cur.fetchmany(max_rows + 1)
            finally:
                timer.cancel()
        except duckdb.InterruptException as exc:
            raise ConnectorError(f"Query exceeded the {timeout_s}s time limit.") from exc
        except duckdb.Error as exc:
            raise ConnectorError(str(exc).splitlines()[0][:400]) from exc
        finally:
            con.close()
        safe = [[v if isinstance(v, (int, float, str, bool)) or v is None else str(v) for v in r] for r in rows[:max_rows]]
        return QueryResult(columns=columns, rows=safe, truncated=len(rows) > max_rows, duration_ms=(time.perf_counter() - start) * 1000)

    return run


# --------------------------------------------------------------------- generation


SQL_SYSTEM = """You translate business questions into ONE {dialect} SQL query.
Rules:
- Read-only: a single SELECT (CTEs allowed). Never modify data or schema.
- Use only the tables and columns listed below; quote identifiers that contain spaces or capitals with double quotes.
- Prefer aggregation; always include LIMIT {limit} or less.
- Reply with the SQL inside a ```sql code block and nothing else.
- If the question cannot be answered from these tables, reply exactly: CANNOT_ANSWER
Schema:
{schema}"""


def extract_sql(text: str) -> str | None:
    if "CANNOT_ANSWER" in text and "select" not in text.lower():
        return None
    m = _FENCE.search(text)
    sql = (m.group(1) if m else text).strip()
    # Return any statement-looking output: the validator decides (and logs) rejections.
    return sql if re.match(r"^\(?\s*[A-Za-z]+", sql) else None


def generate_sql(llm: ResolvedLLM, question: str, source: SqlSource, previous: tuple[str, str] | None = None) -> str | None:
    messages = [
        LLMMessage("system", SQL_SYSTEM.format(dialect=source.dialect, limit=source.max_rows, schema=source.schema_text)),
        LLMMessage("user", question),
    ]
    if previous:
        messages += [LLMMessage("assistant", f"```sql\n{previous[0]}\n```"), LLMMessage("user", f"That query was rejected or failed: {previous[1]}. Return a corrected query.")]
    return extract_sql(llm.provider.complete(messages).content)


def suggest_chart(columns: list[str], rows: list[list[Any]]) -> dict[str, Any] | None:
    if not rows or len(columns) < 2 or len(rows) > 200:
        return None
    numeric = [i for i in range(len(columns)) if all(isinstance(r[i], (int, float)) and not isinstance(r[i], bool) for r in rows if r[i] is not None)]
    dims = [i for i in range(len(columns)) if i not in numeric]
    if not numeric or not dims:
        return None
    x = dims[0]
    xname = columns[x].lower()
    timeish = any(k in xname for k in ("date", "month", "year", "week", "day", "time", "period", "quarter")) or all(isinstance(r[x], str) and re.match(r"^\d{4}-\d{2}", r[x] or "") for r in rows)
    return {"type": "line" if timeish else "bar", "x": columns[x], "y": [columns[i] for i in numeric[:3]], "data": [{columns[i]: r[i] for i in [x] + numeric[:3]} for r in rows]}


def explain(outcome: SqlOutcome, source: SqlSource) -> str:
    if outcome.status != "executed":
        return ""
    tables = ", ".join(sorted({t.name for t in source.tables if t.name.lower() in outcome.sql.lower() or t.bare in outcome.sql.lower()})) or source.name
    more = " (row limit reached; results are truncated)" if outcome.truncated else ""
    return f"Ran a read-only query on {tables} in {source.name}; {len(outcome.rows)} row(s) returned in {outcome.duration_ms or 0:.0f} ms{more}."


# ---------------------------------------------------------------------- execution


def _log(db: Session, org_id: uuid.UUID, user_id: uuid.UUID | None, source: SqlSource, outcome: SqlOutcome) -> None:
    entry = SqlQueryLog(
        org_id=org_id, source_type=source.kind, source_id=source.id, user_id=user_id, question=outcome.question[:4000],
        generated_sql=outcome.generated_sql[:20000], executed_sql=outcome.sql[:20000], status=outcome.status,
        row_count=len(outcome.rows) if outcome.status == "executed" else None, duration_ms=outcome.duration_ms,
        error="; ".join(outcome.errors)[:4000] or None, reasons=outcome.reasons,
    )
    db.add(entry)
    db.flush()
    outcome.log_id = entry.id


def execute_validated(db: Session, org_id: uuid.UUID, user_id: uuid.UUID | None, source: SqlSource, outcome: SqlOutcome, validation: Validation, *, approved: bool = False) -> SqlOutcome:
    outcome.sql = validation.sql
    if validation.needs_approval and not approved:
        outcome.status = "pending_approval"
        outcome.reasons = validation.approval_reasons
        _log(db, org_id, user_id, source, outcome)
        approval = Approval(
            org_id=org_id, kind="sql_query", requested_by=user_id or uuid.UUID(int=0), resource_type=source.kind, resource_id=source.id,
            summary=f"{source.name}: {outcome.question[:200] or validation.sql[:200]}",
            payload={"sql": validation.sql, "question": outcome.question, "source_kind": source.kind, "source_id": str(source.id), "log_id": str(outcome.log_id), "reasons": validation.approval_reasons},
        )
        db.add(approval)
        db.flush()
        outcome.approval_id = approval.id
        return outcome
    try:
        result = source.execute(validation.sql, source.max_rows, source.timeout_s)
    except (ConnectorError, NotAvailable) as exc:
        outcome.status = "failed"
        outcome.errors = [exc.message]
        _log(db, org_id, user_id, source, outcome)
        return outcome
    outcome.status = "executed"
    outcome.columns, outcome.rows, outcome.truncated, outcome.duration_ms = result.columns, result.rows, result.truncated, result.duration_ms
    outcome.chart = suggest_chart(result.columns, result.rows)
    outcome.explanation = explain(outcome, source)
    _log(db, org_id, user_id, source, outcome)
    return outcome


def run_sql(db: Session, org_id: uuid.UUID, user_id: uuid.UUID | None, source: SqlSource, sql: str, *, question: str = "") -> SqlOutcome:
    outcome = SqlOutcome(status="rejected", question=question, generated_sql=sql)
    validation = validate(sql, source.policy())
    if not validation.ok:
        outcome.errors = validation.errors
        _log(db, org_id, user_id, source, outcome)
        return outcome
    return execute_validated(db, org_id, user_id, source, outcome, validation)


def run_question(db: Session, org_id: uuid.UUID, user_id: uuid.UUID | None, source: SqlSource, question: str, llm: ResolvedLLM | None) -> SqlOutcome:
    if llm is None:
        raise NotAvailable("Natural-language queries need a configured language model. You can still run SQL directly.")
    start = time.perf_counter()
    try:
        sql = generate_sql(llm, question, source)
    except CSVAIError as exc:
        raise ValidationFailed(f"The model could not generate SQL: {exc.message}") from exc
    if sql is None:
        outcome = SqlOutcome(status="cannot_answer", question=question, errors=["The available tables cannot answer this question."])
        _log(db, org_id, user_id, source, outcome)
        return outcome
    validation = validate(sql, source.policy())
    attempt_error = "; ".join(validation.errors) if not validation.ok else None
    outcome = SqlOutcome(status="rejected", question=question, generated_sql=sql)
    if validation.ok:
        outcome = execute_validated(db, org_id, user_id, source, outcome, validation)
        if outcome.status != "failed":
            _usage(db, org_id, user_id, llm, start)
            return outcome
        attempt_error = "; ".join(outcome.errors)
    # One self-correction round with the validator/database error.
    retry = generate_sql(llm, question, source, previous=(sql, attempt_error or "invalid"))
    outcome = SqlOutcome(status="rejected", question=question, generated_sql=retry or "")
    if retry:
        validation = validate(retry, source.policy())
        if validation.ok:
            outcome = execute_validated(db, org_id, user_id, source, outcome, validation)
            _usage(db, org_id, user_id, llm, start)
            return outcome
        outcome.errors = validation.errors
    else:
        outcome.errors = [attempt_error or "No valid SQL produced."]
    _log(db, org_id, user_id, source, outcome)
    _usage(db, org_id, user_id, llm, start)
    return outcome


def _usage(db: Session, org_id: uuid.UUID, user_id: uuid.UUID | None, llm: ResolvedLLM, start: float) -> None:
    db.add(UsageRecord(org_id=org_id, user_id=user_id, kind="sql", provider=llm.kind, model=llm.provider.model, latency_ms=(time.perf_counter() - start) * 1000))


def result_markdown(columns: list[str], rows: list[list[Any]], max_rows: int = 30) -> str:
    def cell(v: Any) -> str:
        return "" if v is None else str(v).replace("|", "/").replace("\n", " ")[:80]

    head = "| " + " | ".join(columns) + " |\n|" + " --- |" * len(columns)
    body = "\n".join("| " + " | ".join(cell(v) for v in r) + " |" for r in rows[:max_rows])
    more = f"\n… {len(rows) - max_rows} more rows" if len(rows) > max_rows else ""
    return f"{head}\n{body}{more}"


def now() -> datetime:
    return datetime.now(UTC)
