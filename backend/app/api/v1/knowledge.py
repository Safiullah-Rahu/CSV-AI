"""Knowledge bases, documents, datasets and ingestion jobs."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select

from app.api.deps import OrgContext, get_org_context, requires_any_scope
from app.api.schemas import ORM, Ok, page
from app.config import get_settings
from app.ingestion.parsers import format_support
from app.jobs.queue import cancel
from app.models import Dataset, Document, DocumentVersion, Job, KBConfigRevision, KnowledgeBase, ResourceGrant
from app.rag.chunking import DEFAULT_CHUNKING, STRATEGIES
from app.rag.embeddings import EMBEDDING_PROVIDERS
from app.rag.retrieval import DEFAULT_RETRIEVAL
from app.rbac.authorizer import MANAGE, READ, WRITE
from app.rbac.catalog import LEVEL_NAME
from app.services import knowledge_service as ks
from app.storage import get_storage
from app.utils.errors import NotFound, PermissionDenied, ValidationFailed

router = APIRouter(tags=["knowledge"])


class KBIn(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    description: str = Field(default="", max_length=4000)
    workspace_id: uuid.UUID | None = None
    department_id: uuid.UUID | None = None
    tags: list[str] = []
    category: str = Field(default="", max_length=80)
    visibility: str = Field(default="restricted", pattern="^(restricted|workspace|org)$")
    embedding_config: dict[str, Any] | None = None
    chunking_config: dict[str, Any] | None = None
    retrieval_config: dict[str, Any] | None = None
    generation_config: dict[str, Any] | None = None


class KBUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=160)
    description: str | None = None
    workspace_id: uuid.UUID | None = None
    department_id: uuid.UUID | None = None
    tags: list[str] | None = None
    category: str | None = None
    visibility: str | None = Field(default=None, pattern="^(restricted|workspace|org)$")
    embedding_config: dict[str, Any] | None = None
    chunking_config: dict[str, Any] | None = None
    retrieval_config: dict[str, Any] | None = None
    generation_config: dict[str, Any] | None = None
    reindex: bool = False


class KBOut(ORM):
    id: uuid.UUID
    name: str
    description: str
    workspace_id: uuid.UUID | None
    department_id: uuid.UUID | None
    owner_id: uuid.UUID | None
    tags: list[str]
    category: str
    visibility: str
    embedding_config: dict[str, Any]
    chunking_config: dict[str, Any]
    retrieval_config: dict[str, Any]
    generation_config: dict[str, Any]
    status: str
    config_version: int
    is_sample: bool
    created_at: datetime
    updated_at: datetime
    access: str = "none"
    stats: dict[str, Any] = {}


def _kb_out(kb: KnowledgeBase, level: int, stats: dict[str, Any] | None = None) -> KBOut:
    out = KBOut.model_validate(kb)
    out.access = LEVEL_NAME.get(level, "none")
    out.stats = stats or {}
    return out


@router.get("/knowledge-bases/options")
def kb_options(ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> dict[str, Any]:
    return {
        "chunking_strategies": list(STRATEGIES),
        "chunking_defaults": DEFAULT_CHUNKING,
        "retrieval_defaults": DEFAULT_RETRIEVAL,
        "embedding_providers": [{"key": k, "label": v} for k, v in EMBEDDING_PROVIDERS.items()],
        "formats": format_support(),
        "max_upload_mb": get_settings().max_upload_mb,
        "max_batch_files": get_settings().max_batch_files,
    }


@router.get("/knowledge-bases", response_model=list[KBOut])
def list_kbs(workspace_id: uuid.UUID | None = None, department_id: uuid.UUID | None = None, q: str | None = None, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> list[KBOut]:
    stmt = select(KnowledgeBase).where(KnowledgeBase.org_id == ctx.org_id)
    if workspace_id:
        stmt = stmt.where(KnowledgeBase.workspace_id == workspace_id)
    if department_id:
        stmt = stmt.where(KnowledgeBase.department_id == department_id)
    if q:
        stmt = stmt.where(or_(KnowledgeBase.name.ilike(f"%{q}%"), KnowledgeBase.description.ilike(f"%{q}%")))
    kbs = list(ctx.db.scalars(stmt.order_by(KnowledgeBase.name)))
    levels = ctx.authz.kb_levels(kbs)
    visible = [k for k in kbs if levels[k.id] >= READ]
    stats = ks.kb_stats(ctx.db, [k.id for k in visible])
    return [_kb_out(k, levels[k.id], stats[k.id]) for k in visible]


@router.post("/knowledge-bases", response_model=KBOut, status_code=201)
def create_kb(body: KBIn, ctx: OrgContext = Depends(get_org_context)) -> KBOut:
    kb = ks.create_kb(ctx, body.model_dump())
    ctx.db.commit()
    return _kb_out(kb, MANAGE, ks.kb_stats(ctx.db, [kb.id])[kb.id])


@router.get("/knowledge-bases/{kb_id}", response_model=KBOut)
def get_kb(kb_id: uuid.UUID, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> KBOut:
    kb = ks.get_kb(ctx, kb_id)
    return _kb_out(kb, ctx.authz.kb_level(kb), ks.kb_stats(ctx.db, [kb.id])[kb.id])


@router.patch("/knowledge-bases/{kb_id}")
def update_kb(kb_id: uuid.UUID, body: KBUpdate, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> dict[str, Any]:
    kb = ks.get_kb(ctx, kb_id, WRITE)
    kb, reindex = ks.update_kb(ctx, kb, body.model_dump(exclude_unset=True))
    ctx.db.commit()
    return {"knowledge_base": _kb_out(kb, ctx.authz.kb_level(kb)).model_dump(mode="json"), "reindex_scheduled": reindex}


class DeleteKBIn(BaseModel):
    confirm_name: str


@router.delete("/knowledge-bases/{kb_id}", response_model=Ok)
def delete_kb(kb_id: uuid.UUID, body: DeleteKBIn, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> Ok:
    kb = ks.get_kb(ctx, kb_id, MANAGE, "kb.delete")
    if body.confirm_name != kb.name:
        raise ValidationFailed("Type the knowledge base name exactly to confirm deletion.")
    n = ks.delete_kb(ctx, kb)
    ctx.db.commit()
    return Ok(detail=f"Deleted knowledge base and {n} document(s), including chunks, embeddings and stored files.")


@router.get("/knowledge-bases/{kb_id}/revisions")
def kb_revisions(kb_id: uuid.UUID, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> list[dict[str, Any]]:
    ks.get_kb(ctx, kb_id)
    rows = ctx.db.scalars(select(KBConfigRevision).where(KBConfigRevision.kb_id == kb_id).order_by(KBConfigRevision.version.desc()).limit(100))
    return [{"version": r.version, "config": r.config, "changed_by": str(r.changed_by) if r.changed_by else None, "created_at": r.created_at.isoformat()} for r in rows]


@router.get("/knowledge-bases/{kb_id}/access")
def kb_access_summary(kb_id: uuid.UUID, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> dict[str, Any]:
    """Who can reach this knowledge base (for the dashboard's 'authorized users')."""
    kb = ks.get_kb(ctx, kb_id)
    grants = list(ctx.db.scalars(select(ResourceGrant).where(ResourceGrant.org_id == ctx.org_id, ResourceGrant.resource_type == "knowledge_base", ResourceGrant.resource_id == kb_id)))
    return {
        "visibility": kb.visibility,
        "owner_id": str(kb.owner_id) if kb.owner_id else None,
        "grants": [{"id": str(g.id), "subject_type": g.subject_type, "subject_id": str(g.subject_id) if g.subject_id else None, "privilege": g.privilege, "effect": g.effect, "expires_at": g.expires_at.isoformat() if g.expires_at else None} for g in grants],
        "your_access": LEVEL_NAME[ctx.authz.kb_level(kb)],
    }


# ---------------------------------------------------------------------------- documents


class DocumentOut(ORM):
    id: uuid.UUID
    kb_id: uuid.UUID
    title: str
    filename: str
    ext: str
    mime_type: str
    size_bytes: int
    owner_id: uuid.UUID | None
    status: str
    parser: str
    page_count: int | None
    char_count: int
    chunk_count: int
    token_count: int
    language: str | None
    tags: list[str]
    meta: dict[str, Any]
    error: str | None
    current_version: int
    last_indexed_at: datetime | None
    embedding_status: str
    source_type: str
    source_url: str | None
    acl_mode: str
    created_at: datetime
    updated_at: datetime
    kb_name: str | None = None


_SORTS = {"created_at": Document.created_at, "title": Document.title, "size": Document.size_bytes, "status": Document.status, "chunks": Document.chunk_count, "updated_at": Document.updated_at}


@router.get("/documents")
def list_documents(
    kb_id: uuid.UUID | None = None, status: str | None = None, ext: str | None = None, q: str | None = None, tag: str | None = None,
    source_type: str | None = None, sort: str = "created_at", order: str = "desc", limit: int = 50, offset: int = 0,
    ctx: OrgContext = Depends(requires_any_scope("kb.view")),
) -> dict[str, Any]:
    scope = ctx.authz.access_scope([kb_id] if kb_id else None)
    kb_ids = scope.readable_kb_ids()
    if not kb_ids:
        return page([], 0, limit, offset)
    stmt = select(Document).where(Document.org_id == ctx.org_id, Document.kb_id.in_(kb_ids))
    if scope.denied_document_ids:
        stmt = stmt.where(Document.id.not_in(scope.denied_document_ids))
    stmt = stmt.where(or_(Document.acl_mode == "inherit", Document.id.in_(scope.restricted_allowed_document_ids or [uuid.UUID(int=0)]), Document.kb_id.in_(scope.kb_manage_ids or [uuid.UUID(int=0)]), Document.owner_id == ctx.user_id))
    if status:
        stmt = stmt.where(Document.status == status)
    if ext:
        stmt = stmt.where(Document.ext == ext.lower().lstrip("."))
    if source_type:
        stmt = stmt.where(Document.source_type == source_type)
    if tag:
        stmt = stmt.where(Document.tags.any(tag))
    if q:
        stmt = stmt.where(or_(Document.title.ilike(f"%{q}%"), Document.filename.ilike(f"%{q}%")))
    total = ctx.db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    col = _SORTS.get(sort, Document.created_at)
    rows = list(ctx.db.scalars(stmt.order_by(col.asc() if order == "asc" else col.desc()).limit(min(limit, 200)).offset(offset)))
    names = dict(ctx.db.execute(select(KnowledgeBase.id, KnowledgeBase.name).where(KnowledgeBase.id.in_(kb_ids))).all())
    items = []
    for d in rows:
        out = DocumentOut.model_validate(d)
        out.kb_name = names.get(d.kb_id)
        items.append(out.model_dump(mode="json"))
    return page(items, total, limit, offset)


@router.post("/knowledge-bases/{kb_id}/documents", status_code=201)
def upload_documents(
    kb_id: uuid.UUID,
    files: list[UploadFile] = File(...),
    tags: str = Form(default=""),
    replace_document_id: uuid.UUID | None = Form(default=None),
    allow_duplicate: bool = Form(default=False),
    acl_mode: str = Form(default="inherit"),
    relative_paths: str = Form(default=""),
    ctx: OrgContext = Depends(requires_any_scope("kb.view")),
) -> dict[str, Any]:
    kb = ks.get_kb(ctx, kb_id, WRITE, "kb.upload")
    settings = get_settings()
    if len(files) > settings.max_batch_files:
        raise ValidationFailed(f"Upload at most {settings.max_batch_files} files per request.")
    if acl_mode not in ("inherit", "restricted"):
        raise ValidationFailed("acl_mode must be inherit or restricted.")
    if acl_mode == "restricted":
        ctx.require("kb.permissions", kb.workspace_id)
    folder_paths = json.loads(relative_paths) if relative_paths else []
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    results = []
    for i, f in enumerate(files):
        name = os.path.basename((f.filename or "upload").replace("\\", "/"))
        try:
            up = ks.spool(f.file, name)
        except Exception as exc:  # noqa: BLE001
            results.append({"filename": name, "outcome": "rejected", "error": getattr(exc, "message", str(exc))})
            continue
        try:
            folder_tags = [p for p in os.path.dirname(folder_paths[i]).replace("\\", "/").split("/") if p] if i < len(folder_paths) else []
            doc, outcome = ks.add_document(ctx, kb, up, tags=tag_list + folder_tags[:3], replace_document_id=replace_document_id if len(files) == 1 else None, allow_duplicate=allow_duplicate, acl_mode=acl_mode)
            ctx.db.commit()
            results.append({"filename": name, "outcome": outcome, "document": DocumentOut.model_validate(doc).model_dump(mode="json")})
        except Exception as exc:  # noqa: BLE001 — per-file errors are reported, the batch continues
            ctx.db.rollback()
            results.append({"filename": name, "outcome": "rejected", "error": getattr(exc, "message", str(exc))})
        finally:
            os.unlink(up.path)
    return {"results": results, "accepted": sum(1 for r in results if r["outcome"] in ("created", "new_version")), "rejected": sum(1 for r in results if r["outcome"] == "rejected")}


@router.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: uuid.UUID, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> DocumentOut:
    doc, kb = ks.get_document(ctx, doc_id)
    out = DocumentOut.model_validate(doc)
    out.kb_name = kb.name
    return out


class DocumentUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=500)
    tags: list[str] | None = None
    acl_mode: str | None = Field(default=None, pattern="^(inherit|restricted)$")
    kb_id: uuid.UUID | None = None
    metadata: dict[str, Any] | None = None


@router.patch("/documents/{doc_id}", response_model=DocumentOut)
def update_document(doc_id: uuid.UUID, body: DocumentUpdate, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> DocumentOut:
    doc, kb = ks.get_document(ctx, doc_id, WRITE)
    if body.title is not None:
        doc.title = body.title.strip() or doc.title
    if body.tags is not None:
        doc.tags = [t.strip() for t in body.tags if t.strip()][:30]
    if body.metadata is not None:
        doc.meta = {**(doc.meta or {}), "custom": body.metadata}
    if body.acl_mode is not None:
        ctx.require("kb.permissions", kb.workspace_id)
        doc.acl_mode = body.acl_mode
    if body.kb_id and body.kb_id != doc.kb_id:
        target = ks.get_kb(ctx, body.kb_id, WRITE, "kb.upload")
        from sqlalchemy import update as sql_update

        from app.models import Chunk, ChunkEmbedding

        if (target.embedding_config or {}).get("model") != (kb.embedding_config or {}).get("model"):
            ks.reprocess(ctx, doc)
        ctx.db.execute(sql_update(Chunk).where(Chunk.document_id == doc.id).values(kb_id=target.id))
        ctx.db.execute(sql_update(ChunkEmbedding).where(ChunkEmbedding.chunk_id.in_(select(Chunk.id).where(Chunk.document_id == doc.id))).values(kb_id=target.id))
        ctx.db.execute(sql_update(Dataset).where(Dataset.document_id == doc.id).values(kb_id=target.id))
        doc.kb_id = target.id
    ctx.audit("document.update", resource_type="document", resource_id=doc.id, meta=body.model_dump(exclude_none=True, mode="json"))
    ctx.db.commit()
    return DocumentOut.model_validate(doc)


@router.delete("/documents/{doc_id}", response_model=Ok)
def delete_document(doc_id: uuid.UUID, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> Ok:
    doc, _ = ks.get_document(ctx, doc_id, WRITE)
    ks.delete_document(ctx, doc)
    ctx.db.commit()
    return Ok(detail="Document, versions, chunks, embeddings and stored files deleted.")


class BulkIn(BaseModel):
    document_ids: list[uuid.UUID] = Field(min_length=1, max_length=500)
    action: str = Field(pattern="^(delete|reprocess|tag)$")
    tags: list[str] = []


@router.post("/documents/bulk")
def bulk_documents(body: BulkIn, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> dict[str, Any]:
    done, failed = 0, []
    for doc_id in body.document_ids:
        try:
            doc, _ = ks.get_document(ctx, doc_id, WRITE)
            if body.action == "delete":
                ks.delete_document(ctx, doc)
            elif body.action == "reprocess":
                ks.reprocess(ctx, doc)
            else:
                doc.tags = sorted(set(doc.tags) | {t.strip() for t in body.tags if t.strip()})
            ctx.db.commit()
            done += 1
        except Exception as exc:  # noqa: BLE001
            ctx.db.rollback()
            failed.append({"document_id": str(doc_id), "error": getattr(exc, "message", str(exc))})
    return {"succeeded": done, "failed": failed}


@router.post("/documents/{doc_id}/reprocess", response_model=Ok)
def reprocess_document(doc_id: uuid.UUID, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> Ok:
    doc, _ = ks.get_document(ctx, doc_id, WRITE)
    ks.reprocess(ctx, doc)
    ctx.db.commit()
    return Ok(detail="Reprocessing queued.")


@router.get("/documents/{doc_id}/chunks")
def document_chunks(doc_id: uuid.UUID, limit: int = 50, offset: int = 0, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> dict[str, Any]:
    doc, _ = ks.get_document(ctx, doc_id)
    rows, total = ks.document_chunks(ctx.db, doc.id, min(limit, 200), offset)
    items = [{"id": str(c.id), "ordinal": c.ordinal, "content": c.content, "token_count": c.token_count, "metadata": c.meta, "parent_id": str(c.parent_id) if c.parent_id else None} for c in rows]
    return page(items, total, limit, offset)


@router.get("/documents/{doc_id}/versions")
def document_versions(doc_id: uuid.UUID, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> list[dict[str, Any]]:
    doc, _ = ks.get_document(ctx, doc_id)
    rows = ctx.db.scalars(select(DocumentVersion).where(DocumentVersion.document_id == doc.id).order_by(DocumentVersion.version.desc()))
    return [{"version": v.version, "filename": v.filename, "size_bytes": v.size_bytes, "sha256": v.sha256, "created_by": str(v.created_by) if v.created_by else None, "created_at": v.created_at.isoformat(), "current": v.version == doc.current_version} for v in rows]


@router.get("/documents/{doc_id}/activity")
def document_activity(doc_id: uuid.UUID, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> list[dict[str, Any]]:
    from app.models import ActivityEvent
    from app.services.activity import event_to_dict

    doc, _ = ks.get_document(ctx, doc_id)
    jobs = ctx.db.scalars(select(Job).where(Job.org_id == ctx.org_id, Job.resource_type == "document", Job.resource_id == doc.id).order_by(Job.created_at.desc()).limit(20))
    events = ctx.db.scalars(select(ActivityEvent).where(ActivityEvent.org_id == ctx.org_id, ActivityEvent.resource_id == doc.id).order_by(ActivityEvent.id.desc()).limit(50))
    return [{"kind": "event", **event_to_dict(e)} for e in events] + [{"kind": "job", "id": str(j.id), "type": j.kind, "status": j.status, "attempts": j.attempts, "error": j.error, "created_at": j.created_at.isoformat(), "finished_at": j.finished_at.isoformat() if j.finished_at else None} for j in jobs]


@router.get("/documents/{doc_id}/download")
def download_document(doc_id: uuid.UUID, version: int | None = None, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> StreamingResponse:
    doc, kb = ks.get_document(ctx, doc_id)
    if not ctx.authz.has("kb.export", kb.workspace_id) and doc.owner_id != ctx.user_id:
        raise PermissionDenied("Downloading originals requires the export permission.")
    key = doc.storage_key
    filename = doc.filename
    if version:
        v = ctx.db.scalar(select(DocumentVersion).where(DocumentVersion.document_id == doc.id, DocumentVersion.version == version))
        if v is None:
            raise NotFound("Version not found.")
        key, filename = v.storage_key, v.filename
    if not key:
        raise NotFound("No stored file for this document.")
    ctx.audit("document.download", resource_type="document", resource_id=doc.id, meta={"version": version or doc.current_version})
    ctx.db.commit()
    fh = get_storage().open(key)
    safe = filename.replace('"', "")
    return StreamingResponse(iter(lambda: fh.read(1024 * 256), b""), media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{safe}"', "X-Content-Type-Options": "nosniff"})


# ------------------------------------------------------------------------------- jobs


def _job_dict(j: Job) -> dict[str, Any]:
    return {
        "id": str(j.id), "kind": j.kind, "status": j.status, "progress": j.progress, "message": j.message, "attempts": j.attempts,
        "max_attempts": j.max_attempts, "error": j.error, "resource_type": j.resource_type, "resource_id": str(j.resource_id) if j.resource_id else None,
        "created_at": j.created_at.isoformat(), "started_at": j.started_at.isoformat() if j.started_at else None,
        "finished_at": j.finished_at.isoformat() if j.finished_at else None, "next_run_at": j.next_run_at.isoformat() if j.next_run_at else None,
        "result": j.result,
    }


@router.get("/jobs")
def list_jobs(status: str | None = None, kind: str | None = None, limit: int = 50, offset: int = 0, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> dict[str, Any]:
    stmt = select(Job).where(Job.org_id == ctx.org_id)
    if not (ctx.authz.has("db.view") or ctx.authz.has("admin.audit")):
        stmt = stmt.where(Job.created_by == ctx.user_id)
    if status:
        stmt = stmt.where(Job.status == status)
    if kind:
        stmt = stmt.where(Job.kind == kind)
    total = ctx.db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = ctx.db.scalars(stmt.order_by(Job.created_at.desc()).limit(min(limit, 200)).offset(offset))
    return page([_job_dict(j) for j in rows], total, limit, offset)


@router.get("/jobs/{job_id}")
def get_job(job_id: uuid.UUID, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> dict[str, Any]:
    job = ctx.db.get(Job, job_id)
    if job is None or job.org_id != ctx.org_id:
        raise NotFound("Job not found.")
    return _job_dict(job)


@router.post("/jobs/{job_id}/cancel", response_model=Ok)
def cancel_job(job_id: uuid.UUID, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> Ok:
    job = ctx.db.get(Job, job_id)
    if job is None or job.org_id != ctx.org_id:
        raise NotFound("Job not found.")
    if job.created_by != ctx.user_id and not ctx.authz.has("admin.audit"):
        raise PermissionDenied("You can only cancel your own jobs.")
    cancel(ctx.db, job)
    ctx.db.commit()
    return Ok()


@router.post("/jobs/{job_id}/retry", response_model=Ok)
def retry_job(job_id: uuid.UUID, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> Ok:
    from sqlalchemy import text

    job = ctx.db.get(Job, job_id)
    if job is None or job.org_id != ctx.org_id:
        raise NotFound("Job not found.")
    if job.status not in ("dead", "failed", "cancelled"):
        raise ValidationFailed("Only failed, dead or cancelled jobs can be retried.")
    job.status, job.attempts, job.next_run_at, job.error = "queued", 0, None, None
    ctx.db.execute(text("NOTIFY rag_jobs"))
    ctx.db.info.setdefault("jobs_to_dispatch", []).append(job.id)
    ctx.audit("job.retry", resource_type="job", resource_id=job.id)
    ctx.db.commit()
    return Ok()

