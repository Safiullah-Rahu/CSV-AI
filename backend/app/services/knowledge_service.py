"""Knowledge bases and documents: creation, configuration, uploads, versions, deletion."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, BinaryIO

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.api.deps import OrgContext
from app.config import get_settings
from app.ingestion.detect import check_name, detect, extension_of
from app.ingestion.types import InvalidDocument, UnsupportedFormat
from app.jobs.queue import enqueue
from app.models import (
    AIProviderConfig,
    Chunk,
    ChunkEmbedding,
    Citation,
    Dataset,
    Department,
    Document,
    DocumentVersion,
    KBConfigRevision,
    KnowledgeBase,
    Workspace,
)
from app.rag.chunking import STRATEGIES, merge_config
from app.rag.embeddings import EMBEDDING_PROVIDERS
from app.rag.retrieval import merge_retrieval_config
from app.rbac.authorizer import MANAGE, READ, WRITE
from app.services.activity import emit
from app.services.billing import check_limit
from app.storage import get_storage, make_key
from app.utils.errors import Conflict, NotFound, PermissionDenied, UnsafeRequest, ValidationFailed

DEFAULT_EMBEDDING = {"provider": "hashing", "model": "hashing-768", "dim": 768}


# --------------------------------------------------------------------- knowledge bases


def get_kb(ctx: OrgContext, kb_id: uuid.UUID, level: int = READ, perm: str | None = None) -> KnowledgeBase:
    kb = ctx.db.get(KnowledgeBase, kb_id)
    if kb is None or kb.org_id != ctx.org_id:
        raise NotFound("Knowledge base not found.")
    if ctx.authz.kb_level(kb) < READ:
        raise NotFound("Knowledge base not found.")  # do not reveal existence
    ctx.authz.require_kb(kb, level, perm)
    return kb


def validate_embedding_config(ctx: OrgContext, cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = {**DEFAULT_EMBEDDING, **(cfg or {})} if not cfg or cfg.get("provider") in (None, "hashing") else dict(cfg)
    provider = cfg.get("provider")
    if provider not in EMBEDDING_PROVIDERS:
        raise ValidationFailed(f"Unknown embedding provider '{provider}'.")
    if provider == "hashing":
        cfg["dim"] = int(cfg.get("dim") or 768)
        cfg["model"] = f"hashing-{cfg['dim']}"
    if cfg.get("provider_id"):
        p = ctx.db.get(AIProviderConfig, uuid.UUID(str(cfg["provider_id"])))
        if p is None or p.org_id != ctx.org_id:
            raise ValidationFailed("Embedding provider configuration not found.")
    elif provider not in ("hashing", "local", "ollama"):
        raise ValidationFailed(f"'{provider}' embeddings need an AI provider with credentials (provider_id).")
    if cfg.get("dim") is not None and not 8 <= int(cfg["dim"]) <= 8192:
        raise ValidationFailed("Embedding dimension must be between 8 and 8192.")
    return cfg


def validate_chunking(cfg: dict[str, Any] | None) -> dict[str, Any]:
    try:
        merged = merge_config(cfg)
    except ValueError as exc:
        raise ValidationFailed(str(exc)) from exc
    return {k: merged[k] for k in merged if cfg is None or k in (cfg or {}) or k == "strategy"}


def validate_retrieval(cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = cfg or {}
    merged = merge_retrieval_config(cfg)
    if merged["mode"] not in ("hybrid", "dense", "sparse", "keyword"):
        raise ValidationFailed("retrieval mode must be hybrid, dense, sparse or keyword.")
    if merged["rerank"] not in ("lexical", "none", "cohere", "cross_encoder"):
        raise ValidationFailed("rerank must be lexical, none, cohere or cross_encoder.")
    return cfg


def _snapshot(kb: KnowledgeBase) -> dict[str, Any]:
    return {
        "name": kb.name, "description": kb.description, "visibility": kb.visibility, "tags": kb.tags, "category": kb.category,
        "embedding_config": kb.embedding_config, "chunking_config": kb.chunking_config,
        "retrieval_config": kb.retrieval_config, "generation_config": kb.generation_config,
        "workspace_id": str(kb.workspace_id) if kb.workspace_id else None,
        "department_id": str(kb.department_id) if kb.department_id else None,
    }


def create_kb(ctx: OrgContext, data: dict[str, Any]) -> KnowledgeBase:
    workspace_id = data.get("workspace_id")
    ctx.require("kb.create", workspace_id)
    check_limit(ctx.db, ctx.org_id, "knowledge_bases")
    if workspace_id:
        ws = ctx.db.get(Workspace, workspace_id)
        if ws is None or ws.org_id != ctx.org_id:
            raise ValidationFailed("Workspace not found.")
    if data.get("department_id"):
        dep = ctx.db.get(Department, data["department_id"])
        if dep is None or dep.org_id != ctx.org_id:
            raise ValidationFailed("Department not found.")
    if ctx.db.scalar(select(KnowledgeBase.id).where(KnowledgeBase.org_id == ctx.org_id, KnowledgeBase.name == data["name"].strip())):
        raise Conflict("A knowledge base with this name already exists.")
    kb = KnowledgeBase(
        org_id=ctx.org_id,
        name=data["name"].strip(),
        description=data.get("description", ""),
        workspace_id=workspace_id,
        department_id=data.get("department_id"),
        owner_id=ctx.user_id,
        tags=data.get("tags") or [],
        category=data.get("category", ""),
        visibility=data.get("visibility", "restricted"),
        embedding_config=validate_embedding_config(ctx, data.get("embedding_config")),
        chunking_config=validate_chunking(data.get("chunking_config")),
        retrieval_config=validate_retrieval(data.get("retrieval_config")),
        generation_config=data.get("generation_config") or {},
        is_sample=bool(data.get("is_sample")),
    )
    ctx.db.add(kb)
    ctx.db.flush()
    ctx.db.add(KBConfigRevision(org_id=ctx.org_id, kb_id=kb.id, version=1, config=_snapshot(kb), changed_by=ctx.user_id))
    ctx.audit("kb.create", resource_type="knowledge_base", resource_id=kb.id, meta={"name": kb.name, "visibility": kb.visibility})
    emit(ctx.db, ctx.org_id, "kb.created", resource_type="knowledge_base", resource_id=kb.id, actor_user_id=ctx.user_id, message=f"Knowledge base '{kb.name}' created", severity="success")
    return kb


def update_kb(ctx: OrgContext, kb: KnowledgeBase, data: dict[str, Any]) -> tuple[KnowledgeBase, bool]:
    reindex = False
    if any(k in data for k in ("name", "description", "tags", "category", "visibility", "workspace_id", "department_id")):
        ctx.authz.require_kb(kb, WRITE, "kb.edit")
    for key in ("name", "description", "tags", "category", "visibility"):
        if key in data and data[key] is not None:
            setattr(kb, key, data[key].strip() if isinstance(data[key], str) and key == "name" else data[key])
    # Explicit null clears the department/workspace assignment.
    if "department_id" in data:
        kb.department_id = data["department_id"]
    if "workspace_id" in data:
        if data["workspace_id"] is not None:
            ctx.require("kb.create", data["workspace_id"])
        kb.workspace_id = data["workspace_id"]
    if "visibility" in data and data["visibility"] not in ("restricted", "workspace", "org"):
        raise ValidationFailed("visibility must be restricted, workspace or org.")
    if "embedding_config" in data and data["embedding_config"] is not None:
        ctx.authz.require_kb(kb, MANAGE, "kb.configure_ingestion")
        new = validate_embedding_config(ctx, data["embedding_config"])
        old = kb.embedding_config or {}
        if (new.get("provider"), new.get("model"), new.get("dim")) != (old.get("provider"), old.get("model"), old.get("dim")):
            reindex = True
        kb.embedding_config = new
    if "chunking_config" in data and data["chunking_config"] is not None:
        ctx.authz.require_kb(kb, MANAGE, "kb.configure_ingestion")
        if data["chunking_config"].get("strategy") and data["chunking_config"]["strategy"] not in STRATEGIES:
            raise ValidationFailed(f"Unknown chunking strategy. Use one of {STRATEGIES}.")
        kb.chunking_config = validate_chunking(data["chunking_config"])
        reindex = reindex or bool(data.get("reindex"))
    if "retrieval_config" in data and data["retrieval_config"] is not None:
        ctx.authz.require_kb(kb, MANAGE, "kb.configure_retrieval")
        kb.retrieval_config = validate_retrieval(data["retrieval_config"])
    if "generation_config" in data and data["generation_config"] is not None:
        ctx.authz.require_kb(kb, MANAGE, "kb.configure_retrieval")
        policy = data["generation_config"].get("grounding_policy")
        if policy and policy not in ("strict", "balanced", "open"):
            raise ValidationFailed("grounding_policy must be strict, balanced or open.")
        kb.generation_config = data["generation_config"]
    kb.config_version += 1
    ctx.db.add(KBConfigRevision(org_id=ctx.org_id, kb_id=kb.id, version=kb.config_version, config=_snapshot(kb), changed_by=ctx.user_id))
    ctx.audit("kb.update", resource_type="knowledge_base", resource_id=kb.id, meta={"fields": sorted(k for k in data if data[k] is not None), "reindex": reindex})
    reindex = reindex and has_chunks(ctx.db, kb.id)
    if reindex:
        enqueue(ctx.db, ctx.org_id, "reindex_kb", payload={"kb_id": str(kb.id)}, created_by=ctx.user_id, resource_type="knowledge_base", resource_id=kb.id, idempotency_key=f"reindex:{kb.id}:{kb.config_version}")
    emit(ctx.db, ctx.org_id, "kb.updated", resource_type="knowledge_base", resource_id=kb.id, actor_user_id=ctx.user_id, message=f"Knowledge base '{kb.name}' updated")
    return kb, reindex


def has_chunks(db: Session, kb_id: uuid.UUID) -> bool:
    return db.scalar(select(Chunk.id).where(Chunk.kb_id == kb_id).limit(1)) is not None


def kb_stats(db: Session, kb_ids: list[uuid.UUID]) -> dict[uuid.UUID, dict[str, Any]]:
    stats: dict[uuid.UUID, dict[str, Any]] = {k: {"documents": 0, "ready": 0, "failed": 0, "processing": 0, "chunks": 0, "storage_bytes": 0, "records": 0, "last_indexed_at": None} for k in kb_ids}
    if not kb_ids:
        return stats
    for kb_id, status, n, size, chunks, last in db.execute(
        select(Document.kb_id, Document.status, func.count(), func.coalesce(func.sum(Document.size_bytes), 0), func.coalesce(func.sum(Document.chunk_count), 0), func.max(Document.last_indexed_at))
        .where(Document.kb_id.in_(kb_ids))
        .group_by(Document.kb_id, Document.status)
    ):
        s = stats[kb_id]
        s["documents"] += n
        s["storage_bytes"] += int(size)
        s["chunks"] += int(chunks)
        if status == "ready":
            s["ready"] += n
        elif status == "failed":
            s["failed"] += n
        elif status in ("queued", "processing"):
            s["processing"] += n
        if last and (s["last_indexed_at"] is None or last > s["last_indexed_at"]):
            s["last_indexed_at"] = last
    for kb_id, rows in db.execute(select(Dataset.kb_id, func.coalesce(func.sum(Dataset.row_count), 0)).where(Dataset.kb_id.in_(kb_ids)).group_by(Dataset.kb_id)):
        stats[kb_id]["records"] = int(rows)
    return stats


def delete_kb(ctx: OrgContext, kb: KnowledgeBase) -> int:
    ctx.authz.require_kb(kb, MANAGE, "kb.delete")
    keys = _storage_keys_for(ctx.db, kb_id=kb.id)
    doc_ids = list(ctx.db.scalars(select(Document.id).where(Document.kb_id == kb.id)))
    _scrub_citations(ctx.db, doc_ids)
    ctx.db.delete(kb)
    ctx.audit("kb.delete", resource_type="knowledge_base", resource_id=kb.id, meta={"name": kb.name, "documents": len(doc_ids)})
    emit(ctx.db, ctx.org_id, "kb.deleted", resource_type="knowledge_base", resource_id=kb.id, actor_user_id=ctx.user_id, message=f"Knowledge base '{kb.name}' deleted", severity="warning")
    ctx.db.flush()
    _delete_objects(keys)
    return len(doc_ids)


# ---------------------------------------------------------------------------- documents


def get_document(ctx: OrgContext, doc_id: uuid.UUID, level: int = READ) -> tuple[Document, KnowledgeBase]:
    doc = ctx.db.get(Document, doc_id)
    if doc is None or doc.org_id != ctx.org_id:
        raise NotFound("Document not found.")
    kb = ctx.db.get(KnowledgeBase, doc.kb_id)
    assert kb is not None
    kb_level = ctx.authz.kb_level(kb)
    if ctx.authz.document_level(doc, kb_level) < READ:
        raise NotFound("Document not found.")
    if ctx.authz.document_level(doc, kb_level) < level:
        raise PermissionDenied("You do not have sufficient access to this document.")
    return doc, kb


@dataclass
class SpooledUpload:
    path: str
    size: int
    sha256: str
    filename: str


def spool(upload: BinaryIO, filename: str) -> SpooledUpload:
    """Stream an upload to a temp file, enforcing the size limit and hashing on the fly."""
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    digest = hashlib.sha256()
    fd, path = tempfile.mkstemp(prefix="upload_")
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise UnsafeRequest(f"File exceeds the {get_settings().max_upload_mb} MB upload limit.")
                digest.update(chunk)
                out.write(chunk)
    except Exception:
        os.unlink(path)
        raise
    return SpooledUpload(path=path, size=size, sha256=digest.hexdigest(), filename=filename)


def add_document(
    ctx: OrgContext,
    kb: KnowledgeBase,
    up: SpooledUpload,
    *,
    title: str | None = None,
    tags: list[str] | None = None,
    replace_document_id: uuid.UUID | None = None,
    allow_duplicate: bool = False,
    acl_mode: str = "inherit",
    source_type: str = "upload",
) -> tuple[Document, str]:
    """Store an upload and queue ingestion. Returns (document, outcome)."""
    ctx.authz.require_kb(kb, WRITE, "kb.upload")
    try:
        check_name(up.filename)
        detected = detect(up.path, up.filename)
    except (InvalidDocument, UnsupportedFormat) as exc:
        raise ValidationFailed(str(exc), details={"filename": up.filename}) from exc
    if not allow_duplicate and replace_document_id is None:
        dup = ctx.db.scalar(select(Document).where(Document.kb_id == kb.id, Document.sha256 == up.sha256, Document.status != "deleting"))
        if dup is not None:
            return dup, "duplicate"
    storage = get_storage()
    key = make_key(ctx.org_id, "documents", up.filename)
    if replace_document_id:
        doc, _ = get_document(ctx, replace_document_id, WRITE)
        if doc.kb_id != kb.id:
            raise ValidationFailed("The document to replace belongs to another knowledge base.")
        if doc.sha256 == up.sha256:
            return doc, "unchanged"
        check_limit(ctx.db, ctx.org_id, "storage_mb", add=max(1, up.size // (1024 * 1024)))
        storage.put_file(key, up.path)
        doc.current_version += 1
        doc.storage_key, doc.sha256, doc.size_bytes, doc.filename = key, up.sha256, up.size, up.filename
        doc.ext, doc.mime_type = detected.ext, detected.mime
        doc.status, doc.error, doc.embedding_status = "queued", None, "pending"
        outcome = "new_version"
    else:
        check_limit(ctx.db, ctx.org_id, "documents")
        check_limit(ctx.db, ctx.org_id, "storage_mb", add=max(0, up.size // (1024 * 1024)))
        storage.put_file(key, up.path)
        doc = Document(
            org_id=ctx.org_id, kb_id=kb.id, title=(title or os.path.splitext(up.filename)[0])[:500], filename=up.filename,
            ext=detected.ext, mime_type=detected.mime, size_bytes=up.size, sha256=up.sha256, storage_key=key, owner_id=ctx.user_id,
            status="queued", tags=tags or [], acl_mode=acl_mode, source_type=source_type,
        )
        ctx.db.add(doc)
        ctx.db.flush()
        outcome = "created"
    ctx.db.add(DocumentVersion(org_id=ctx.org_id, document_id=doc.id, version=doc.current_version, sha256=up.sha256, size_bytes=up.size, storage_key=key, filename=up.filename, created_by=ctx.user_id))
    job = enqueue(ctx.db, ctx.org_id, "ingest_document", payload={"document_id": str(doc.id)}, created_by=ctx.user_id, resource_type="document", resource_id=doc.id, idempotency_key=f"ingest:{doc.id}:v{doc.current_version}")
    doc.meta = {**(doc.meta or {}), "last_job_id": str(job.id)}
    ctx.audit("document.upload", resource_type="document", resource_id=doc.id, meta={"kb_id": str(kb.id), "filename": up.filename, "size": up.size, "outcome": outcome})
    emit(ctx.db, ctx.org_id, "document.uploaded", resource_type="document", resource_id=doc.id, actor_user_id=ctx.user_id, message=f"{up.filename} uploaded to {kb.name}", payload={"kb_id": str(kb.id), "version": doc.current_version})
    return doc, outcome


def reprocess(ctx: OrgContext, doc: Document) -> None:
    kb = ctx.db.get(KnowledgeBase, doc.kb_id)
    assert kb is not None
    ctx.authz.require_kb(kb, WRITE, "kb.reprocess")
    doc.status, doc.error = "queued", None
    enqueue(ctx.db, ctx.org_id, "ingest_document", payload={"document_id": str(doc.id)}, created_by=ctx.user_id, resource_type="document", resource_id=doc.id, idempotency_key=f"ingest:{doc.id}:v{doc.current_version}:r{int(time.time() * 1000)}")
    ctx.audit("document.reprocess", resource_type="document", resource_id=doc.id)


def delete_document(ctx: OrgContext, doc: Document) -> None:
    kb = ctx.db.get(KnowledgeBase, doc.kb_id)
    assert kb is not None
    if not (ctx.authz.has("kb.delete_documents", kb.workspace_id) and ctx.authz.document_level(doc) >= WRITE):
        raise PermissionDenied("You cannot delete this document.")
    keys = _storage_keys_for(ctx.db, document_id=doc.id)
    _scrub_citations(ctx.db, [doc.id])
    ctx.db.delete(doc)  # cascades: versions, chunks, embeddings, datasets
    ctx.audit("document.delete", resource_type="document", resource_id=doc.id, meta={"filename": doc.filename, "kb_id": str(doc.kb_id)})
    emit(ctx.db, ctx.org_id, "document.deleted", resource_type="document", resource_id=doc.id, actor_user_id=ctx.user_id, message=f"{doc.filename} deleted", payload={"kb_id": str(doc.kb_id)})
    ctx.db.flush()
    _delete_objects(keys)


def _storage_keys_for(db: Session, *, kb_id: uuid.UUID | None = None, document_id: uuid.UUID | None = None) -> list[str]:
    q_versions = select(DocumentVersion.storage_key).join(Document, Document.id == DocumentVersion.document_id)
    q_ds = select(Dataset.profile)
    if kb_id:
        q_versions = q_versions.where(Document.kb_id == kb_id)
        q_ds = q_ds.where(Dataset.kb_id == kb_id)
    else:
        q_versions = q_versions.where(Document.id == document_id)
        q_ds = q_ds.where(Dataset.document_id == document_id)
    keys = [k for k in db.scalars(q_versions) if k]
    for profile in db.scalars(q_ds):
        keys += [t["storage_key"] for t in (profile or {}).get("tables", []) if t.get("storage_key")]
    return keys


def _delete_objects(keys: list[str]) -> None:
    storage = get_storage()
    for key in set(keys):
        try:
            storage.delete(key)
        except Exception:  # noqa: BLE001 — orphaned objects are reported by the storage audit
            pass


def _scrub_citations(db: Session, doc_ids: list[uuid.UUID]) -> None:
    """Deletion propagation: past answers keep their text but lose deleted-source snippets."""
    if doc_ids:
        db.execute(update(Citation).where(Citation.document_id.in_(doc_ids)).values(snippet="", title="[deleted document]", chunk_id=None))


def document_chunks(db: Session, doc_id: uuid.UUID, limit: int, offset: int) -> tuple[list[Chunk], int]:
    total = db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc_id, Chunk.is_retrievable.is_(True))) or 0
    rows = list(db.scalars(select(Chunk).where(Chunk.document_id == doc_id, Chunk.is_retrievable.is_(True)).order_by(Chunk.ordinal).limit(limit).offset(offset)))
    return rows, total


def embedding_count(db: Session, doc_id: uuid.UUID) -> int:
    return db.scalar(select(func.count()).select_from(ChunkEmbedding).join(Chunk, Chunk.id == ChunkEmbedding.chunk_id).where(Chunk.document_id == doc_id)) or 0


def file_ext(filename: str) -> str:
    return extension_of(filename)
