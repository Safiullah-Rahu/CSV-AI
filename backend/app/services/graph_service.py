"""Connections graph: the organisation's data ecosystem as nodes and edges.

Everything comes from real rows the viewer may see — no simulated nodes.
Topology (left → right):

    organization → workspace → knowledge base
    source (database / integration / uploads) → ingestion pipeline → knowledge base
    knowledge base → embedding model → vector index (pgvector) → RAG engine → AI model → agent
    live-query database → RAG engine

Node ``status`` is one of healthy | syncing | warning | failed | paused | disabled | idle.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from app.api.deps import OrgContext
from app.connectors.databases import provider_info
from app.connectors.integrations import INTEGRATIONS
from app.llm.registry import platform_fallback
from app.models import (
    Agent,
    AIProviderConfig,
    Chunk,
    DatabaseConnection,
    Dataset,
    Document,
    Integration,
    Job,
    KnowledgeBase,
    Workspace,
)
from app.rbac.authorizer import READ

GROUP_THRESHOLD = 40


def _iso(v: datetime | None) -> str | None:
    return v.isoformat() if v else None


def _conn_status(c: DatabaseConnection, running: set[uuid.UUID]) -> str:
    if not c.enabled:
        return "disabled"
    if c.id in running or c.status == "syncing":
        return "syncing"
    if c.status == "failed":
        return "failed"
    if c.sync_paused:
        return "paused"
    if c.status == "healthy":
        return "healthy"
    return "idle"


def build_graph(ctx: OrgContext, *, workspace_id: uuid.UUID | None = None, kb_id: uuid.UUID | None = None, group: bool | None = None) -> dict[str, Any]:
    db, authz, org = ctx.db, ctx.authz, ctx.org
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def node(id_: str, type_: str, label: str, status: str = "healthy", **data: Any) -> None:
        nodes.append({"id": id_, "type": type_, "label": label, "status": status, "data": data})

    def edge(src: str, dst: str, kind: str, status: str = "healthy", **data: Any) -> None:
        edges.append({"id": f"{src}->{dst}", "source": src, "target": dst, "kind": kind, "status": status, "data": data})

    org_node = f"org:{org.id}"
    node(org_node, "organization", org.name, data_slug=org.slug)

    # Knowledge bases the viewer can read.
    scope = authz.access_scope([kb_id] if kb_id else None)
    kbs = [k for k in db.scalars(select(KnowledgeBase).where(KnowledgeBase.id.in_(scope.readable_kb_ids() or [uuid.UUID(int=0)]))) if not workspace_id or k.workspace_id == workspace_id]
    kb_ids = [k.id for k in kbs]
    doc_stats: dict[uuid.UUID, dict[str, Any]] = {k: {"documents": 0, "failed": 0, "processing": 0, "chunks": 0, "bytes": 0} for k in kb_ids}
    by_source: dict[tuple[str, uuid.UUID | None], set[uuid.UUID]] = {}
    if kb_ids:
        for kid, status, stype, sid, n, chunks, size in db.execute(
            select(Document.kb_id, Document.status, Document.source_type, Document.source_id, func.count(), func.coalesce(func.sum(Document.chunk_count), 0), func.coalesce(func.sum(Document.size_bytes), 0))
            .where(Document.kb_id.in_(kb_ids)).group_by(Document.kb_id, Document.status, Document.source_type, Document.source_id)
        ):
            s = doc_stats[kid]
            s["documents"] += n
            s["chunks"] += int(chunks)
            s["bytes"] += int(size)
            if status == "failed":
                s["failed"] += n
            if status in ("queued", "processing"):
                s["processing"] += n
            by_source.setdefault((stype, sid), set()).add(kid)
    records = dict(db.execute(select(Dataset.kb_id, func.coalesce(func.sum(Dataset.row_count), 0)).where(Dataset.kb_id.in_(kb_ids or [uuid.UUID(int=0)])).group_by(Dataset.kb_id)).all())
    running_jobs = list(db.scalars(select(Job).where(Job.org_id == org.id, Job.status.in_(("queued", "running")))))
    running_sources = {j.resource_id for j in running_jobs if j.resource_type in ("db_connection", "integration")}

    # Workspaces.
    if authz.has("workspace.view"):
        ws_q = select(Workspace).where(Workspace.org_id == org.id)
        if workspace_id:
            ws_q = ws_q.where(Workspace.id == workspace_id)
        for ws in db.scalars(ws_q):
            node(f"workspace:{ws.id}", "workspace", ws.name, kb_count=sum(1 for k in kbs if k.workspace_id == ws.id))
            edge(org_node, f"workspace:{ws.id}", "contains")
    ws_nodes = {n["id"] for n in nodes if n["type"] == "workspace"}

    for kb in kbs:
        s = doc_stats[kb.id]
        status = "syncing" if s["processing"] else ("warning" if s["failed"] else ("healthy" if s["documents"] else "idle"))
        node(f"kb:{kb.id}", "knowledge_base", kb.name, status, documents=s["documents"], chunks=s["chunks"], failed=s["failed"], processing=s["processing"],
             storage_bytes=s["bytes"], records=int(records.get(kb.id, 0)), visibility=kb.visibility, workspace_id=str(kb.workspace_id) if kb.workspace_id else None,
             embedding=(kb.embedding_config or {}).get("model"), access=["none", "read", "write", "manage"][scope.kb_levels.get(kb.id, 0)])
        parent = f"workspace:{kb.workspace_id}" if kb.workspace_id and f"workspace:{kb.workspace_id}" in ws_nodes else org_node
        edge(parent, f"kb:{kb.id}", "contains")

    pipeline = "pipeline:ingestion"
    failed_24h = db.scalar(select(func.count()).select_from(Job).where(Job.org_id == org.id, Job.kind == "ingest_document", Job.status == "dead", Job.finished_at >= datetime.now(UTC) - timedelta(hours=24))) or 0
    active_ingest = sum(1 for j in running_jobs if j.kind == "ingest_document")
    node(pipeline, "pipeline", "Ingestion pipeline", "syncing" if active_ingest else ("warning" if failed_24h else "healthy"), active_jobs=active_ingest, failed_24h=failed_24h,
         stages=["parse", "normalize", "chunk", "embed", "index"])
    for kb in kbs:
        if doc_stats[kb.id]["documents"]:
            edge(pipeline, f"kb:{kb.id}", "indexes", "active" if doc_stats[kb.id]["processing"] else "healthy")

    # Upload source.
    upload_kbs = by_source.get(("upload", None), set()) | by_source.get(("sample", None), set())
    if upload_kbs:
        node("source:uploads", "document_source", "File uploads", "healthy", provider="upload", kbs=len(upload_kbs), documents=sum(doc_stats[k]["documents"] for k in upload_kbs))
        edge(org_node, "source:uploads", "owns")
        edge("source:uploads", pipeline, "feeds", data_kbs=[str(k) for k in upload_kbs])

    sources: list[tuple[str, dict[str, Any]]] = []
    # Database connections.
    if authz.has("db.view"):
        for c in db.scalars(select(DatabaseConnection).where(DatabaseConnection.org_id == org.id)):
            if authz.connection_level(c) < READ or (workspace_id and c.workspace_id not in (None, workspace_id)) or (kb_id and c.kb_id != kb_id):
                continue
            info = provider_info(c.provider)
            category = info.get("category", "database")
            ntype = {"warehouse": "warehouse", "vector": "vector_db", "nosql": "database"}.get(category, "database")
            status = _conn_status(c, running_sources)
            fed = by_source.get(("database", c.id), set())
            sources.append((f"db:{c.id}", {
                "type": ntype, "label": c.name, "status": status, "provider": c.provider, "provider_label": info["label"], "category": category,
                "access_mode": c.access_mode, "last_sync_at": _iso(c.last_sync_at), "last_success_at": _iso(c.last_success_at), "last_error": c.last_error,
                "tables": len((c.selection or {}).get("tables", {})), "kb_id": str(c.kb_id) if c.kb_id else None, "documents": sum(doc_stats[k]["documents"] for k in fed),
                "resource_id": str(c.id), "resource_kind": "db_connection",
            }))
            if c.access_mode in ("ingest", "hybrid"):
                edge(f"db:{c.id}", pipeline, "feeds", "failed" if status == "failed" else ("paused" if status == "paused" else ("disabled" if status == "disabled" else ("active" if status == "syncing" else "healthy"))))
            if c.access_mode in ("live", "hybrid"):
                edge(f"db:{c.id}", "rag:engine", "live_query", "failed" if status == "failed" else ("disabled" if status == "disabled" else "healthy"))
    # Integrations.
    if authz.has("integrations.view"):
        for i in db.scalars(select(Integration).where(Integration.org_id == org.id)):
            if (workspace_id and i.workspace_id not in (None, workspace_id)) or (kb_id and i.kb_id != kb_id):
                continue
            meta = INTEGRATIONS.get(i.provider, {})
            status = "disabled" if not i.enabled else ("syncing" if i.id in running_sources or i.status == "syncing" else ("failed" if i.status == "failed" else ("paused" if i.sync_paused else ("healthy" if i.status == "connected" else "idle"))))
            fed = by_source.get(("integration", i.id), set())
            sources.append((f"int:{i.id}", {
                "type": "integration", "label": i.name, "status": status, "provider": i.provider, "provider_label": meta.get("label", i.provider),
                "category": meta.get("category", "integration"), "last_sync_at": _iso(i.last_sync_at), "last_error": i.last_error,
                "kb_id": str(i.kb_id) if i.kb_id else None, "documents": sum(doc_stats[k]["documents"] for k in fed), "resource_id": str(i.id), "resource_kind": "integration",
                "connection_status": i.status,
            }))
            if i.kb_id:
                edge(f"int:{i.id}", pipeline, "feeds", {"failed": "failed", "paused": "paused", "disabled": "disabled", "syncing": "active"}.get(status, "healthy"))
    grouped = (group if group is not None else len(sources) > GROUP_THRESHOLD)
    if grouped:
        clusters: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for sid, d in sources:
            clusters.setdefault(d["provider"], []).append((sid, d))
        member_map: dict[str, str] = {}
        for provider, members in clusters.items():
            cid = f"group:{provider}"
            statuses = [m[1]["status"] for m in members]
            agg = "failed" if "failed" in statuses else ("syncing" if "syncing" in statuses else "healthy")
            node(cid, "group", f"{members[0][1]['provider_label']} ×{len(members)}", agg, provider=provider, members=[m[0] for m in members], count=len(members),
                 failed=statuses.count("failed"), syncing=statuses.count("syncing"))
            edge(org_node, cid, "owns")
            for sid, _ in members:
                member_map[sid] = cid
        remapped = {}
        for e in edges:
            src = member_map.get(e["source"], e["source"])
            key = f"{src}->{e['target']}"
            if key not in remapped:
                remapped[key] = {**e, "id": key, "source": src}
        edges = list(remapped.values())
    else:
        for sid, d in sources:
            node(sid, d.pop("type"), d.pop("label"), d.pop("status"), **d)
            edge(org_node, sid, "owns")

    # Embedding models → vector index → RAG → models → agents.
    models: dict[str, list[uuid.UUID]] = {}
    for kb in kbs:
        cfg = kb.embedding_config or {}
        key = f"{cfg.get('provider', 'hashing')}:{cfg.get('model', 'hashing-768')}"
        models.setdefault(key, []).append(kb.id)
    total_chunks = sum(s["chunks"] for s in doc_stats.values())
    node("vector:pgvector", "vector_store", "pgvector index", "healthy" if total_chunks else "idle", chunks=total_chunks, engine="PostgreSQL + pgvector (HNSW)")
    for key, members in models.items():
        provider, model = key.split(":", 1)
        nid = f"embedding:{key}"
        node(nid, "embedding", model, "healthy", provider=provider, knowledge_bases=len(members), dim=next(((k.embedding_config or {}).get("dim") for k in kbs if k.id in members), None))
        for k in members:
            edge(f"kb:{k}", nid, "embeds", "active" if doc_stats[k]["processing"] else "healthy")
        edge(nid, "vector:pgvector", "stores")
    node("rag:engine", "rag", "RAG engine", "healthy", retrieval="hybrid (dense + BM25) · rerank · MMR", grounding="citations enforced")
    edge("vector:pgvector", "rag:engine", "retrieves")
    providers = [p for p in db.scalars(select(AIProviderConfig).where(AIProviderConfig.org_id == org.id)) if p.kind not in ("cohere", "voyage")]
    for p in providers:
        status = "disabled" if not p.enabled else {"failed": "failed", "healthy": "healthy"}.get(p.status, "idle")
        node(f"model:{p.id}", "ai_model", p.name, status, kind=p.kind, model=p.default_model, is_default=p.is_default, last_checked_at=_iso(p.last_checked_at), last_error=p.last_error if authz.has("admin.ai_providers") else None)
        edge("rag:engine", f"model:{p.id}", "generates", "disabled" if not p.enabled else ("failed" if status == "failed" else "healthy"))
    # Without an enabled provider of its own, the organisation answers with the deployment's platform model (if any).
    fallback = None if any(p.enabled for p in providers) else platform_fallback(org)
    if fallback is not None:
        node("model:platform", "ai_model", "Platform model", "healthy", kind=fallback.kind, model=fallback.provider.model, is_default=True, note="Provided by this deployment")
        edge("rag:engine", "model:platform", "generates")
    elif not providers:
        node("model:extractive", "ai_model", "Extractive answers", "idle", kind="none", note="No language model configured")
        edge("rag:engine", "model:extractive", "generates", "idle")
    if authz.has("agents.view"):
        for a in db.scalars(select(Agent).where(Agent.org_id == org.id)):
            if authz.agent_level(a) < READ:
                continue
            node(f"agent:{a.id}", "agent", a.name, "healthy" if a.enabled else "disabled", tools=a.tools, knowledge_bases=len(a.kb_ids))
            target = f"model:{a.provider_id}" if a.provider_id and any(n["id"] == f"model:{a.provider_id}" for n in nodes) else "rag:engine"
            edge(target, f"agent:{a.id}", "powers", "healthy" if a.enabled else "disabled")
    node_ids = {n["id"] for n in nodes}
    edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]
    return {"nodes": nodes, "edges": edges, "grouped": grouped, "summary": summary(ctx, nodes, doc_stats, records, running_jobs), "generated_at": datetime.now(UTC).isoformat()}


def summary(ctx: OrgContext, nodes: list[dict[str, Any]], doc_stats: dict[uuid.UUID, dict[str, Any]], records: dict[Any, Any], running_jobs: list[Job]) -> dict[str, Any]:
    source_nodes = [n for n in nodes if n["type"] in ("database", "warehouse", "vector_db", "integration", "document_source")]
    for g in (n for n in nodes if n["type"] == "group"):
        source_nodes.extend({"status": "failed" if i < g["data"]["failed"] else ("syncing" if i < g["data"]["failed"] + g["data"]["syncing"] else "healthy"), "data": {}} for i in range(g["data"]["count"]))
    statuses = [n["status"] for n in source_nodes]
    last_sync = max((n["data"].get("last_sync_at") for n in source_nodes if n["data"].get("last_sync_at")), default=None)
    return {
        "total_sources": len(source_nodes),
        "active": sum(1 for s in statuses if s in ("healthy", "syncing")),
        "disconnected": sum(1 for s in statuses if s in ("disabled", "idle", "paused")),
        "failed": statuses.count("failed"),
        "syncing": statuses.count("syncing"),
        "documents": sum(s["documents"] for s in doc_stats.values()),
        "records": int(sum(records.values())) if records else 0,
        "chunks": sum(s["chunks"] for s in doc_stats.values()),
        "active_jobs": len(running_jobs),
        "storage_bytes": sum(s["bytes"] for s in doc_stats.values()),
        "last_successful_sync": last_sync,
    }


def health(ctx: OrgContext) -> dict[str, Any]:
    """Infrastructure and connection health for the side panel (no security 'threat' labels)."""
    from app.api.v1.health import readiness

    ok, checks = readiness()
    db = ctx.db
    warnings: list[dict[str, Any]] = []
    for c in db.scalars(select(DatabaseConnection).where(DatabaseConnection.org_id == ctx.org_id, DatabaseConnection.status == "failed")):
        if ctx.authz.connection_level(c) >= READ:
            warnings.append({"severity": "error", "resource": c.name, "kind": "db_connection", "id": str(c.id), "message": c.last_error or "Connection failed"})
    for i in db.scalars(select(Integration).where(Integration.org_id == ctx.org_id, Integration.status == "failed")):
        if ctx.authz.has("integrations.view"):
            warnings.append({"severity": "error", "resource": i.name, "kind": "integration", "id": str(i.id), "message": i.last_error or "Sync failed"})
    dead = db.scalar(select(func.count()).select_from(Job).where(Job.org_id == ctx.org_id, Job.status == "dead", Job.finished_at >= datetime.now(UTC) - timedelta(days=1))) or 0
    if dead:
        warnings.append({"severity": "warning", "resource": "Background jobs", "kind": "jobs", "message": f"{dead} job(s) failed permanently in the last 24 h"})
    backlog = db.scalar(select(func.count()).select_from(Job).where(Job.org_id == ctx.org_id, Job.status.in_(("queued", "failed")))) or 0
    if backlog > 100:
        warnings.append({"severity": "warning", "resource": "Job queue", "kind": "jobs", "message": f"{backlog} jobs waiting"})
    for p in db.scalars(select(AIProviderConfig).where(AIProviderConfig.org_id == ctx.org_id, AIProviderConfig.enabled.is_(True), AIProviderConfig.status == "failed")):
        warnings.append({"severity": "warning", "resource": p.name, "kind": "ai_provider", "id": str(p.id), "message": "Last health check failed"})
    unindexed = db.scalar(select(func.count()).select_from(Chunk).where(Chunk.org_id == ctx.org_id)) or 0
    return {
        "status": "operational" if ok and not any(w["severity"] == "error" for w in warnings) else ("degraded" if ok else "outage"),
        "infrastructure": {k: {"ok": v.get("ok"), **{kk: vv for kk, vv in v.items() if kk in ("latency_ms", "backend", "migration")}} for k, v in checks.items()},
        "warnings": warnings,
        "chunks_indexed": unindexed,
    }
