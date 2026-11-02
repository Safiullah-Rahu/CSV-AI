"""Database connections: lifecycle, health, schema discovery, selection and sync.

Credentials are encrypted (Fernet) in ``credentials`` and never serialised.
Sync jobs extract the selected tables (selected columns, row filters
applied), keep one document per table ("db:<schema.table>") and support:

* ``full``        — re-extract everything;
* ``incremental`` — fetch rows whose ``incremental_column`` exceeds the stored
  cursor, merge by primary key, and drop rows whose keys disappeared upstream
  (deletion propagation) before re-indexing.

Scheduled syncs are enqueued by the scheduler from ``next_sync_at``.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import OrgContext
from app.connectors.databases import ConnectionSpec, build_connector, provider_info
from app.core.security import encrypt_secret
from app.jobs.queue import JobContext, enqueue, handler
from app.llm.registry import credential_value
from app.models import Credential, DatabaseConnection, Document, DocumentVersion, HealthCheck, KnowledgeBase, SyncRun
from app.services.activity import emit
from app.storage import get_storage, make_key
from app.text2sql.validator import validate_row_filter
from app.utils.errors import ConnectorError, CSVAIError, NotFound, ValidationFailed

SYNC_ROW_LIMIT = 200_000


def _now() -> datetime:
    return datetime.now(UTC)


def spec_for(db: Session, conn: DatabaseConnection) -> ConnectionSpec:
    return ConnectionSpec(
        provider=conn.provider, host=conn.host, port=conn.port, database=conn.database, username=conn.username,
        password=credential_value(db, conn.credential_id), ssl_mode=conn.ssl_mode, options=conn.options or {},
    )


def masked_endpoint(conn: DatabaseConnection) -> str:
    if conn.provider == "sqlite":
        return f"sqlite:///…/{os.path.basename(conn.database or conn.host)}"
    host = conn.host
    if len(host) > 6:
        host = host[:3] + "•••" + host[-3:]
    return f"{conn.provider}://{conn.username + '@' if conn.username else ''}{host}:{conn.port or provider_info(conn.provider).get('port') or ''}/{conn.database}"


def set_password(db: Session, ctx_org: uuid.UUID, conn: DatabaseConnection, password: str | None, user_id: uuid.UUID | None) -> None:
    if password is None:
        return
    if conn.credential_id:
        cred = db.get(Credential, conn.credential_id)
        if cred:
            cred.ciphertext = encrypt_secret(password)
            cred.rotated_at = _now()
            return
    cred = Credential(org_id=ctx_org, kind="db_password", ciphertext=encrypt_secret(password), created_by=user_id)
    db.add(cred)
    db.flush()
    conn.credential_id = cred.id


def get_connection(ctx: OrgContext, conn_id: uuid.UUID, level: int = 1) -> DatabaseConnection:
    conn = ctx.db.get(DatabaseConnection, conn_id)
    if conn is None or conn.org_id != ctx.org_id or ctx.authz.connection_level(conn) < 1:
        raise NotFound("Connection not found.")
    if ctx.authz.connection_level(conn) < level:
        from app.utils.errors import PermissionDenied

        raise PermissionDenied("You do not have sufficient access to this connection.")
    return conn


def record_health(db: Session, conn: DatabaseConnection, ok: bool, latency: float | None, error: str | None, actor: uuid.UUID | None) -> None:
    previous = conn.status
    conn.last_tested_at = _now()
    db.add(HealthCheck(org_id=conn.org_id, source_type="database", source_id=conn.id, ok=ok, latency_ms=latency, error=error))
    if ok:
        conn.status = "healthy" if conn.enabled else "disabled"
        conn.last_success_at = _now()
        conn.last_error = None
        if previous == "failed":
            emit(db, conn.org_id, "connection.restored", resource_type="db_connection", resource_id=conn.id, actor_user_id=actor, severity="success", message=f"{conn.name} is reachable again", payload={"latency_ms": round(latency or 0, 1)})
        else:
            emit(db, conn.org_id, "connection.tested", resource_type="db_connection", resource_id=conn.id, actor_user_id=actor, severity="success", message=f"{conn.name} connection OK ({latency or 0:.0f} ms)", payload={"latency_ms": round(latency or 0, 1)})
    else:
        conn.status = "failed"
        conn.last_error = error
        emit(db, conn.org_id, "connection.failed", resource_type="db_connection", resource_id=conn.id, actor_user_id=actor, severity="error", message=f"{conn.name}: {error}", payload={"error": error})


def test_connection(db: Session, conn: DatabaseConnection, actor: uuid.UUID | None) -> dict[str, Any]:
    try:
        latency = build_connector(spec_for(db, conn)).test()
        record_health(db, conn, True, latency, None, actor)
        return {"ok": True, "latency_ms": round(latency, 1)}
    except CSVAIError as exc:
        record_health(db, conn, False, None, exc.message, actor)
        return {"ok": False, "error": exc.message, "code": exc.code}


def discover(db: Session, conn: DatabaseConnection, actor: uuid.UUID | None) -> dict[str, Any]:
    schema = build_connector(spec_for(db, conn)).discover()
    old = (conn.schema_cache or {}).get("tables", {})
    new = schema["tables"]
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(t for t in set(new) & set(old) if [c["name"] for c in new[t]["columns"]] != [c["name"] for c in old[t]["columns"]])
    conn.schema_cache = schema
    conn.schema_refreshed_at = _now()
    if old and (added or removed or changed):
        emit(db, conn.org_id, "schema.changed", resource_type="db_connection", resource_id=conn.id, actor_user_id=actor, severity="warning",
             message=f"{conn.name}: schema changed (+{len(added)} / −{len(removed)} / ~{len(changed)})", payload={"added": added[:50], "removed": removed[:50], "changed": changed[:50]})
    # Drop selections for tables that no longer exist.
    sel = dict((conn.selection or {}).get("tables", {}))
    for t in removed:
        sel.pop(t, None)
    conn.selection = {**(conn.selection or {}), "tables": sel}
    return {"tables": len(new), "added": added, "removed": removed, "changed": changed}


def apply_selection(conn: DatabaseConnection, tables: dict[str, dict[str, Any]]) -> dict[str, Any]:
    discovered = (conn.schema_cache or {}).get("tables", {})
    if not discovered:
        raise ValidationFailed("Discover the schema before selecting tables.")
    dialect = provider_info(conn.provider).get("dialect", "postgres")
    clean: dict[str, Any] = {}
    for key, sel in tables.items():
        if key not in discovered:
            raise ValidationFailed(f"Unknown table '{key}'.")
        names = [c["name"] for c in discovered[key]["columns"]]
        cols = sel.get("columns") or names
        unknown = (set(cols) | set(sel.get("excluded") or []) | set(sel.get("sensitive") or [])) - set(names)
        if unknown:
            raise ValidationFailed(f"Unknown columns in {key}: {sorted(unknown)}")
        if sel.get("row_filter"):
            problems = validate_row_filter(sel["row_filter"], names, dialect)
            if problems:
                raise ValidationFailed(f"Row filter for {key}: {'; '.join(problems)}")
        inc = sel.get("incremental_column")
        if inc and inc not in names:
            raise ValidationFailed(f"Incremental column '{inc}' not found in {key}.")
        pk = sel.get("primary_key") or discovered[key].get("primary_key") or []
        clean[key] = {
            "columns": cols, "excluded": sel.get("excluded") or [], "sensitive": sel.get("sensitive") or [],
            "row_filter": sel.get("row_filter") or None, "incremental_column": inc, "primary_key": pk,
            "requires_approval": bool(sel.get("requires_approval")), "ingest": bool(sel.get("ingest", True)),
        }
    conn.selection = {**(conn.selection or {}), "tables": clean}
    return clean


def schedule(conn: DatabaseConnection) -> None:
    if conn.sync_interval_minutes and not conn.sync_paused and conn.enabled:
        conn.next_sync_at = _now() + timedelta(minutes=conn.sync_interval_minutes)
    else:
        conn.next_sync_at = None


# ------------------------------------------------------------------------------- sync


def _table_doc(db: Session, conn: DatabaseConnection, kb: KnowledgeBase, table: str) -> Document | None:
    return db.scalar(select(Document).where(Document.kb_id == kb.id, Document.source_type == "database", Document.source_id == conn.id, Document.source_ref == table))


@handler("sync_database")
def sync_database(ctx: JobContext) -> dict[str, Any]:
    conn_id = uuid.UUID(ctx.payload["source_id"])
    requested_mode = ctx.payload.get("mode", "incremental")
    with ctx.session() as db:
        conn = db.get(DatabaseConnection, conn_id)
        if conn is None or not conn.enabled:
            return {"skipped": "connection missing or disabled"}
        if conn.kb_id is None:
            raise ConnectorError("Choose a target knowledge base before synchronising.")
        kb = db.get(KnowledgeBase, conn.kb_id)
        spec = spec_for(db, conn)
        selection = {k: v for k, v in ((conn.selection or {}).get("tables") or {}).items() if v.get("ingest", True)}
        cursors = dict((conn.options or {}).get("sync_cursors") or {})
        run = SyncRun(org_id=ctx.org_id, source_type="database", source_id=conn.id, job_id=ctx.job_id, mode=requested_mode)
        db.add(run)
        conn.status = "syncing"
        db.flush()
        run_id = run.id
        emit(db, ctx.org_id, "sync.started", resource_type="db_connection", resource_id=conn.id, actor_user_id=ctx.created_by, message=f"Sync started: {conn.name} ({requested_mode})", payload={"run_id": str(run_id), "tables": len(selection)})
        kb_id, conn_name = kb.id if kb else None, conn.name
    connector = build_connector(spec)
    stats = {"seen": 0, "added": 0, "updated": 0, "deleted": 0, "failed": 0}
    new_cursors = dict(cursors)
    try:
        for t_index, (table, sel) in enumerate(selection.items()):
            cols = [c for c in (sel.get("columns") or []) if c not in set(sel.get("excluded") or []) | set(sel.get("sensitive") or [])]
            if not cols:
                continue
            pk = [c for c in sel.get("primary_key") or [] if c in cols]
            inc = sel.get("incremental_column")
            incremental = requested_mode != "full" and inc and inc in cols and table in cursors
            where = sel.get("row_filter")
            params: dict[str, Any] = {}
            if incremental:
                quoted_inc = connector.quote(inc)  # type: ignore[attr-defined]
                where = f"({where}) AND {quoted_inc} > :sync_cursor" if where else f"{quoted_inc} > :sync_cursor"
                params["sync_cursor"] = cursors[table]
            frames = [pd.DataFrame(batch) for batch in connector.iter_rows(table, cols, where=where, params=params, batch=5000, limit=SYNC_ROW_LIMIT)]
            fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
            stats["seen"] += len(fresh)
            with ctx.session() as db:
                kb = db.get(KnowledgeBase, kb_id)
                conn = db.get(DatabaseConnection, conn_id)
                doc = _table_doc(db, conn, kb, table)
                previous = None
                if doc is not None and incremental and doc.storage_key:
                    with get_storage().local_path(doc.storage_key) as p:
                        previous = pd.read_parquet(p)
            if previous is not None and pk:
                merged = pd.concat([previous[~previous.set_index(pk).index.isin(fresh.set_index(pk).index)], fresh], ignore_index=True) if len(fresh) else previous
                stats["updated"] += int(previous.set_index(pk).index.isin(fresh.set_index(pk).index).sum()) if len(fresh) else 0
                stats["added"] += len(fresh) - (int(previous.set_index(pk).index.isin(fresh.set_index(pk).index).sum()) if len(fresh) else 0)
                # Deletion propagation: keep only keys that still exist upstream.
                live_keys = pd.concat([pd.DataFrame(b) for b in connector.iter_rows(table, pk, where=sel.get("row_filter"), batch=20000, limit=SYNC_ROW_LIMIT)], ignore_index=True)
                before = len(merged)
                merged = merged[merged.set_index(pk).index.isin(live_keys.set_index(pk).index)]
                stats["deleted"] += before - len(merged)
            else:
                merged = fresh
                stats["added"] += len(fresh)
            if inc and inc in merged.columns and len(merged):
                new_cursors[table] = str(merged[inc].max())
            if previous is not None and len(fresh) == 0 and stats["deleted"] == 0:
                ctx.progress((t_index + 1) / max(len(selection), 1), f"{table}: no changes")
                continue
            fd, path = tempfile.mkstemp(suffix=".parquet")
            os.close(fd)
            try:
                merged.to_parquet(path, index=False)
                key = make_key(ctx.org_id, "documents", f"{table}.parquet")
                size = get_storage().put_file(key, path)
            finally:
                os.unlink(path)
            with ctx.session() as db:
                kb = db.get(KnowledgeBase, kb_id)
                conn = db.get(DatabaseConnection, conn_id)
                doc = _table_doc(db, conn, kb, table)
                if doc is None:
                    doc = Document(org_id=ctx.org_id, kb_id=kb_id, title=f"{conn_name} · {table}", filename=f"{table}.parquet", ext="parquet", mime_type="application/vnd.apache.parquet", owner_id=ctx.created_by, source_type="database", source_id=conn_id, source_ref=table, size_bytes=size, storage_key=key, sha256=f"sync-{run_id}-{table}"[:64])
                    db.add(doc)
                    db.flush()
                else:
                    doc.current_version += 1
                    doc.storage_key, doc.size_bytes, doc.sha256 = key, size, f"sync-{run_id}-{table}"[:64]
                doc.status = "queued"
                doc.meta = {**(doc.meta or {}), "database": {"connection": conn_name, "table": table, "rows": len(merged), "columns": cols}}
                db.add(DocumentVersion(org_id=ctx.org_id, document_id=doc.id, version=doc.current_version, sha256=doc.sha256, size_bytes=size, storage_key=key, filename=doc.filename, created_by=ctx.created_by, note=f"sync {requested_mode}"))
                enqueue(db, ctx.org_id, "ingest_document", payload={"document_id": str(doc.id)}, created_by=ctx.created_by, resource_type="document", resource_id=doc.id, idempotency_key=f"ingest:{doc.id}:v{doc.current_version}")
                emit(db, ctx.org_id, "sync.progress", resource_type="db_connection", resource_id=conn_id, message=f"{table}: {len(merged)} rows", payload={"run_id": str(run_id), "table": table, "rows": len(merged), "progress": (t_index + 1) / max(len(selection), 1)})
            ctx.progress((t_index + 1) / max(len(selection), 1), f"Synced {table}")
    except Exception as exc:
        message = getattr(exc, "message", None) or str(exc)
        with ctx.session() as db:
            run = db.get(SyncRun, run_id)
            conn = db.get(DatabaseConnection, conn_id)
            if run:
                run.status, run.finished_at, run.error = "failed", _now(), message[:2000]
                run.items_seen, run.items_added, run.items_updated, run.items_deleted = stats["seen"], stats["added"], stats["updated"], stats["deleted"]
            if conn:
                conn.status, conn.last_error = "failed", message[:2000]
                schedule(conn)
            emit(db, ctx.org_id, "sync.failed", resource_type="db_connection", resource_id=conn_id, severity="error", message=f"Sync failed: {conn_name}: {message[:200]}", payload={"run_id": str(run_id)})
        raise
    with ctx.session() as db:
        run = db.get(SyncRun, run_id)
        conn = db.get(DatabaseConnection, conn_id)
        run.status, run.finished_at = "succeeded", _now()
        run.items_seen, run.items_added, run.items_updated, run.items_deleted = stats["seen"], stats["added"], stats["updated"], stats["deleted"]
        conn.status = "healthy"
        conn.last_sync_at = conn.last_success_at = _now()
        conn.last_error = None
        conn.options = {**(conn.options or {}), "sync_cursors": new_cursors}
        schedule(conn)
        emit(db, ctx.org_id, "sync.completed", resource_type="db_connection", resource_id=conn_id, severity="success", message=f"Sync completed: {conn_name} (+{stats['added']} ~{stats['updated']} −{stats['deleted']})", payload={"run_id": str(run_id), **stats})
    return stats


def sync_now(ctx: OrgContext, conn: DatabaseConnection, mode: str) -> uuid.UUID:
    job = enqueue(ctx.db, ctx.org_id, "sync_database", payload={"source_id": str(conn.id), "mode": mode}, created_by=ctx.user_id, resource_type="db_connection", resource_id=conn.id, idempotency_key=f"sync:{conn.id}:{int(time.time() * 1000)}")
    return job.id
