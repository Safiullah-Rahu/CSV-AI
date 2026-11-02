"""Datasets (tabular documents) and structured retrieval for chat.

Keeps CSV-AI's analysis capabilities — profiling, deterministic statistics,
LLM summaries and natural-language questions — now per tenant, permission
checked, and answered with validated SQL over a DuckDB sandbox.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.connectors.databases import ConnectionSpec, build_connector
from app.ingestion.normalize import STOPWORDS, tokenize
from app.llm.registry import ResolvedLLM, credential_value
from app.models import DatabaseConnection, Dataset, Document, KnowledgeBase
from app.rag.retrieval import Retrieved
from app.rbac.authorizer import READ, AccessScope, Authorizer
from app.storage import get_storage
from app.text2sql import engine as sqlengine
from app.text2sql.validator import TableSpec

_DUCK_TYPES = {"int64": "BIGINT", "Int64": "BIGINT", "float64": "DOUBLE", "Float64": "DOUBLE", "bool": "BOOLEAN", "boolean": "BOOLEAN", "object": "VARCHAR", "string": "VARCHAR"}


def dataset_visible(scope: AccessScope, doc: Document, user_id: uuid.UUID) -> bool:
    if doc.kb_id not in scope.kb_levels or doc.id in scope.denied_document_ids:
        return False
    if doc.acl_mode == "restricted":
        return doc.id in scope.restricted_allowed_document_ids or doc.kb_id in scope.kb_manage_ids or doc.owner_id == user_id
    return True


def visible_datasets(db: Session, authz: Authorizer, kb_ids: list[uuid.UUID] | None = None) -> list[tuple[Dataset, Document]]:
    scope = authz.access_scope(kb_ids)
    if not scope.kb_levels:
        return []
    rows = db.execute(select(Dataset, Document).join(Document, Document.id == Dataset.document_id).where(Dataset.kb_id.in_(scope.readable_kb_ids()), Document.status == "ready")).all()
    return [(ds, doc) for ds, doc in rows if dataset_visible(scope, doc, authz.p.user_id)]


def _schema_text(ds: Dataset) -> str:
    lines = []
    for t in (ds.profile or {}).get("tables", []):
        cols = []
        for c in t.get("profile", {}).get("columns", []):
            typ = _DUCK_TYPES.get(c.get("dtype", "object"), "VARCHAR")
            sample = ", ".join(str(s) for s in (c.get("sample") or [])[:3])
            cols.append(f'  "{c["name"]}" {typ}' + (f"  -- e.g. {sample}" if sample else ""))
        lines.append(f'TABLE "{t["name"]}" ({t.get("rows", "?")} rows)\n' + "\n".join(cols))
    return "\n".join(lines)


@contextlib.contextmanager
def dataset_source(ds: Dataset) -> Iterator[sqlengine.SqlSource]:
    storage = get_storage()
    with contextlib.ExitStack() as stack:
        tables: list[tuple[str, str]] = []
        specs: list[TableSpec] = []
        for t in (ds.profile or {}).get("tables", []):
            path = stack.enter_context(storage.local_path(t["storage_key"]))
            tables.append((t["name"], path))
            specs.append(TableSpec(t["name"], list(t.get("columns") or [])))
        yield sqlengine.SqlSource(
            kind="dataset", id=ds.id, name=ds.name, dialect="duckdb", tables=specs, schema_text=_schema_text(ds),
            execute=sqlengine.duckdb_executor(tables), max_rows=1000, timeout_s=20, max_joins=4,
        )


def connection_source(db: Session, authz: Authorizer, conn: DatabaseConnection) -> sqlengine.SqlSource:
    """Live-query source; tables = admin selection ∩ the user's grant constraints."""
    selection = (conn.selection or {}).get("tables", {})
    allowed_by_grant = authz.connection_table_constraint(conn)
    limits = conn.query_limits or {}
    approval_tables = set(limits.get("approval_tables") or [])
    specs: list[TableSpec] = []
    schema_lines = []
    discovered = (conn.schema_cache or {}).get("tables", {})
    for key, sel in selection.items():
        if allowed_by_grant is not None and key not in allowed_by_grant:
            continue
        info = discovered.get(key, {})
        all_cols = [c["name"] for c in info.get("columns", [])]
        chosen = sel.get("columns") or all_cols
        blocked = set(sel.get("excluded") or []) | set(sel.get("sensitive") or [])
        cols = [c for c in chosen if c not in blocked]
        if not cols:
            continue
        specs.append(TableSpec(key, cols, row_filter=sel.get("row_filter") or None, requires_approval=key in approval_tables or bool(sel.get("requires_approval"))))
        types = {c["name"]: c["type"] for c in info.get("columns", [])}
        fks = "; ".join(f"{','.join(f['columns'])} → {f['references']}({','.join(f['referred_columns'])})" for f in info.get("foreign_keys", []))
        schema_lines.append(f"TABLE {key}\n" + "\n".join(f"  {c} {types.get(c, '')}" for c in cols) + (f"\n  -- foreign keys: {fks}" if fks else ""))
    from app.connectors.databases import provider_info

    info = provider_info(conn.provider)
    spec = ConnectionSpec(provider=conn.provider, host=conn.host, port=conn.port, database=conn.database, username=conn.username, password=credential_value(db, conn.credential_id), ssl_mode=conn.ssl_mode, options=conn.options or {})
    connector = build_connector(spec)
    return sqlengine.SqlSource(
        kind="database", id=conn.id, name=conn.name, dialect=info.get("dialect", "postgres"), tables=specs, schema_text="\n".join(schema_lines),
        execute=lambda sql, max_rows, timeout: connector.execute(sql, max_rows=max_rows, timeout_s=timeout),
        max_rows=int(limits.get("max_rows", 500)), timeout_s=int(limits.get("timeout_s", 15)), max_joins=int(limits.get("max_joins", 4)),
    )


def _relevance(question: str, names: list[str]) -> float:
    q = {t for t in tokenize(question) if t not in STOPWORDS}
    hay = set()
    for n in names:
        hay |= set(tokenize(n.replace("_", " ")))
    return len(q & hay) / max(len(q), 1)


def structured_evidence(db: Session, authz: Authorizer, scope: AccessScope, question: str, llm: ResolvedLLM, *, kb_ids: list[uuid.UUID]) -> list[Retrieved]:
    """Answer analytic questions with SQL over the most relevant authorized table source."""
    candidates: list[tuple[float, str, Any]] = []
    for ds, doc in visible_datasets(db, authz, kb_ids):
        names = [ds.name, ds.table_name] + [c for t in (ds.profile or {}).get("tables", []) for c in t.get("columns", [])]
        candidates.append((_relevance(question, names), "dataset", (ds, doc)))
    for conn in db.scalars(select(DatabaseConnection).where(DatabaseConnection.org_id == authz.p.org_id, DatabaseConnection.kb_id.in_(kb_ids), DatabaseConnection.enabled.is_(True), DatabaseConnection.access_mode.in_(("live", "hybrid")))):
        if authz.connection_level(conn) < READ or not authz.has("db.query"):
            continue
        names = [conn.name] + list((conn.selection or {}).get("tables", {}).keys())
        for sel in (conn.selection or {}).get("tables", {}).values():
            names += sel.get("columns") or []
        candidates.append((_relevance(question, names), "database", conn))
    candidates = [c for c in candidates if c[0] > 0]
    if not candidates:
        return []
    candidates.sort(key=lambda c: -c[0])
    score, kind, obj = candidates[0]
    if kind == "dataset":
        ds, doc = obj
        with dataset_source(ds) as source:
            outcome = sqlengine.run_question(db, authz.p.org_id, authz.p.user_id, source, question, llm)
        title, kb_id, doc_id, locator = f"{ds.name} (dataset)", ds.kb_id, doc.id, {"dataset_id": str(ds.id), "table": ds.table_name}
    else:
        source = connection_source(db, authz, obj)
        if not source.tables:
            return []
        outcome = sqlengine.run_question(db, authz.p.org_id, authz.p.user_id, source, question, llm)
        title, kb_id, doc_id, locator = f"{obj.name} (live database)", obj.kb_id, None, {"connection_id": str(obj.id)}
    if outcome.status != "executed" or not outcome.rows:
        return []
    content = f"Result of a read-only SQL query on {title}:\n```sql\n{outcome.sql}\n```\n{sqlengine.result_markdown(outcome.columns, outcome.rows)}"
    return [Retrieved(
        chunk_id=uuid.uuid4(), document_id=doc_id or uuid.UUID(int=0), kb_id=kb_id or uuid.UUID(int=0), content=content,
        meta={**locator, "sql": outcome.sql, "rows": len(outcome.rows), "kind": "sql_result", "section": [title]},
        title=title, filename=title, ext="sql", source_url=None, source_type="sql_dataset" if kind == "dataset" else "sql_live",
        scores={"rerank": 1.0, "dense": 1.0, "fused": 1.0}, coverage=1.0,
    )]


# ---------------------------------------------------------------------- analysis


def load_frame(ds: Dataset, table: str | None = None, max_rows: int = 200_000) -> pd.DataFrame:
    tables = (ds.profile or {}).get("tables", [])
    t = next((x for x in tables if x["name"] == (table or ds.table_name)), tables[0] if tables else None)
    if t is None:
        return pd.DataFrame()
    with get_storage().local_path(t["storage_key"]) as path:
        df = pd.read_parquet(path)
    return df.head(max_rows)


def analysis(ds: Dataset, table: str | None = None) -> dict[str, Any]:
    """Deterministic statistics (CSV-AI AnalysisService, without the LLM part)."""
    df = load_frame(ds, table)
    missing = df.isna().sum()
    numeric = df.select_dtypes(include="number")
    describe = numeric.describe().T.round(4).reset_index().rename(columns={"index": "column"}) if not numeric.empty else pd.DataFrame()
    top = {}
    for col in df.select_dtypes(include=["object", "category", "bool", "string"]).columns[:20]:
        vc = df[col].value_counts(dropna=False).head(8)
        top[str(col)] = [{"value": None if pd.isna(k) else str(k), "count": int(v)} for k, v in vc.items()]
    corr = numeric.corr().round(3) if numeric.shape[1] >= 2 else None
    return {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "missing": [{"column": str(k), "missing": int(v), "pct": round(float(v) / max(len(df), 1) * 100, 2)} for k, v in missing.items()],
        "numeric": describe.to_dict(orient="records") if not describe.empty else [],
        "top_values": top,
        "correlation": {"columns": [str(c) for c in corr.columns], "matrix": corr.fillna(0).values.tolist()} if corr is not None else None,
        "duplicates": int(df.duplicated().sum()),
    }


def summarize(ds: Dataset, llm: ResolvedLLM | None, table: str | None = None) -> dict[str, Any]:
    df = load_frame(ds, table, max_rows=50_000)
    if llm is None:
        profile = (ds.profile or {}).get("tables", [{}])[0].get("profile", {})
        cols = profile.get("columns", [])
        lines = [f"**{ds.name}** has {profile.get('rows', len(df))} rows and {len(cols)} columns."]
        for c in cols[:12]:
            stat = f"range {c['min']}–{c['max']}, mean {round(c['mean'], 2)}" if c.get("mean") is not None else f"{c.get('unique')} distinct values"
            lines.append(f"- `{c['name']}` ({c['dtype']}): {stat}; {c.get('missing_pct', 0)}% missing")
        return {"summary": "\n".join(lines), "mode": "deterministic"}
    from app.analysis import SummaryService

    text = SummaryService(provider=llm.provider, df=df).generate()
    return {"summary": text, "mode": "llm", "model": llm.provider.model}


def kb_name(db: Session, kb_id: uuid.UUID) -> str:
    kb = db.get(KnowledgeBase, kb_id)
    return kb.name if kb else ""
