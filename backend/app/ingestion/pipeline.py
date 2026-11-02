"""Ingestion pipeline (background job ``ingest_document``).

detect → parse → normalise → language/tags/injection screening → chunk →
embed (batched, with progress) → atomically replace the document's chunks
and embeddings → derive datasets for tabular content → emit events.

The job is idempotent: re-running it rebuilds the same document version.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import select

from app.data.profiler import profile_dataframe
from app.ingestion.detect import detect
from app.ingestion.normalize import auto_tags, detect_language, injection_findings, normalize_document
from app.ingestion.parsers import parse
from app.ingestion.types import InvalidDocument, ParsedDocument, UnsupportedFormat
from app.jobs.queue import DEAD_HOOKS, JobContext, PermanentJobError, handler
from app.llm.registry import resolve_embedder
from app.models import ChunkEmbedding, Dataset, Document, KnowledgeBase, UsageRecord
from app.rag import index
from app.rag.chunking import chunk_blocks, merge_config
from app.services.activity import emit
from app.storage import get_storage, make_key
from app.utils.logging import get_logger
from app.utils.tokens import estimate_tokens

log = get_logger("ingestion")

DEFAULT_EMBEDDING = {"provider": "hashing", "model": "hashing-768", "dim": 768}
EMBED_BATCH = 64


def _now() -> datetime:
    return datetime.now(UTC)


def _profile_table(path: str) -> dict[str, Any]:
    df = pd.read_parquet(path)
    sample = df.head(50_000)
    prof = profile_dataframe(sample)
    cols = []
    for c in prof.columns:
        cols.append({
            "name": c.name, "dtype": c.dtype, "missing_pct": round(c.pct_missing, 2), "unique": c.n_unique,
            "sample": c.sample_values[:3], "mean": c.mean, "std": c.std, "min": c.min, "max": c.max, "median": c.median,
        })
    return {"rows": int(len(df)), "columns": cols, "duplicate_rows": prof.duplicate_rows, "memory_mb": round(prof.memory_mb, 3)}


@handler("ingest_document")
def ingest_document(ctx: JobContext) -> dict[str, Any]:
    doc_id = uuid.UUID(ctx.payload["document_id"])
    started = time.perf_counter()
    with ctx.session() as db:
        doc = db.get(Document, doc_id)
        if doc is None or doc.status == "deleting":
            return {"skipped": "document deleted"}
        kb = db.get(KnowledgeBase, doc.kb_id)
        if kb is None:
            raise PermanentJobError("Knowledge base no longer exists.")
        doc.status = "processing"
        doc.error = None
        filename, storage_key, title, version = doc.filename, doc.storage_key, doc.title, doc.current_version
        kb_id, chunk_cfg = kb.id, merge_config(kb.chunking_config)
        emb_cfg = dict(kb.embedding_config) if kb.embedding_config else dict(DEFAULT_EMBEDDING)
        emit(db, ctx.org_id, "pipeline.started", resource_type="document", resource_id=doc_id, message=f"Processing {filename}", payload={"kb_id": str(kb_id), "stage": "parsing"})
    if not storage_key:
        raise PermanentJobError("Document has no stored file.")
    ctx.progress(0.05, "Detecting format")
    try:
        with get_storage().local_path(storage_key) as path:
            detected = detect(path, filename)
            ctx.progress(0.12, f"Parsing {detected.ext.upper()}")
            parsed: ParsedDocument = parse(path, detected.ext)
            tables = []
            storage = get_storage()
            for t in parsed.tables:
                key = make_key(ctx.org_id, "datasets", f"{t.name}.parquet")
                storage.put_file(key, t.path)
                tables.append({"name": t.name, "storage_key": key, "rows": t.rows, "columns": t.columns, "profile": _profile_table(t.path)})
                if t.path != path and os.path.exists(t.path):
                    os.unlink(t.path)
    except (InvalidDocument, UnsupportedFormat) as exc:
        _fail(ctx, doc_id, str(exc))
        raise PermanentJobError(str(exc)) from exc
    normalize_document(parsed)
    full_text = parsed.text
    if not full_text.strip():
        message = "No extractable text found." + (" " + " ".join(parsed.warnings) if parsed.warnings else "")
        _fail(ctx, doc_id, message)
        raise PermanentJobError(message)
    language = detect_language(full_text)
    tags = auto_tags(full_text)
    findings = injection_findings(full_text)
    ctx.progress(0.3, "Chunking")
    with ctx.session() as db:
        embedder = resolve_embedder(db, ctx.org_id, emb_cfg)
    semantic = chunk_cfg["strategy"] == "semantic"
    drafts = chunk_blocks(parsed.blocks, chunk_cfg, title=parsed.title or title, embed=embedder.embed_documents if semantic else None)
    if not drafts:
        _fail(ctx, doc_id, "Document produced no chunks.")
        raise PermanentJobError("Document produced no chunks.")
    flagged = 0
    for d in drafts:
        if language:
            d.meta.setdefault("language", language)
        hits = injection_findings(d.content) if findings else []
        if hits:
            d.meta["injection_suspect"] = True
            d.meta["injection_findings"] = hits[:3]
            flagged += 1
    retrievable = [i for i, d in enumerate(drafts) if d.retrievable]
    vectors: list[list[float] | None] = [None] * len(drafts)
    ctx.progress(0.4, f"Embedding {len(retrievable)} chunks")
    for start in range(0, len(retrievable), EMBED_BATCH):
        batch = retrievable[start : start + EMBED_BATCH]
        embedded = embedder.embed_documents([drafts[i].content for i in batch])
        for i, v in zip(batch, embedded):
            vectors[i] = v
        ctx.progress(0.4 + 0.5 * min(1.0, (start + len(batch)) / max(len(retrievable), 1)), f"Embedded {start + len(batch)}/{len(retrievable)} chunks")
    dim = embedder.dim
    model = embedder.model
    with ctx.session() as db:
        doc = db.get(Document, doc_id)
        kb = db.get(KnowledgeBase, kb_id)
        if doc is None or kb is None or doc.status == "deleting":
            return {"skipped": "document deleted during processing"}
        if doc.current_version != version:
            return {"skipped": "superseded by a newer version"}
        existing = db.scalar(select(ChunkEmbedding.model).where(ChunkEmbedding.kb_id == kb_id, ChunkEmbedding.model != model).limit(1))
        if existing:
            raise PermanentJobError(f"Knowledge base already contains '{existing}' embeddings; reindex the knowledge base to switch to '{model}'.")
        if not kb.embedding_config or kb.embedding_config.get("dim") != dim or kb.embedding_config.get("model") != model:
            kb.embedding_config = {**emb_cfg, "model": model, "dim": dim}
        index.ensure_vector_index(db, dim)
        count = index.replace_document_chunks(db, org_id=ctx.org_id, kb_id=kb_id, document_id=doc_id, version=version, drafts=drafts, vectors=vectors, model=model, dim=dim)
        doc.status = "ready"
        doc.embedding_status = "complete"
        doc.parser = parsed.parser
        doc.page_count = parsed.page_count
        doc.char_count = len(full_text)
        doc.token_count = sum(d.tokens for d in drafts if d.retrievable)
        doc.chunk_count = count
        doc.language = language
        doc.last_indexed_at = _now()
        doc.title = doc.title if doc.title and doc.title != doc.filename else (parsed.title or doc.title)
        doc.meta = {
            **(doc.meta or {}),
            "auto_tags": tags,
            "warnings": parsed.warnings[:50],
            "source_metadata": {k: v for k, v in parsed.metadata.items() if isinstance(v, (str, int, float, bool, list))},
            "security": {"injection_suspect": bool(findings), "flagged_chunks": flagged, "findings": findings[:5]},
            "embedding": {"model": model, "dim": dim},
            "chunking": {"strategy": chunk_cfg["strategy"], "chunk_size": chunk_cfg["chunk_size"]},
            "processing_ms": round((time.perf_counter() - started) * 1000),
        }
        if tables:
            ds = db.scalar(select(Dataset).where(Dataset.document_id == doc_id))
            first = tables[0]
            if ds is None:
                ds = Dataset(org_id=ctx.org_id, document_id=doc_id, kb_id=kb_id, name=doc.title, table_name=first["name"])
                db.add(ds)
            else:
                for old in (ds.profile or {}).get("tables", []):
                    if old.get("storage_key") not in {t["storage_key"] for t in tables}:
                        get_storage().delete(old["storage_key"])
            ds.name, ds.table_name, ds.storage_key = doc.title, first["name"], first["storage_key"]
            ds.row_count = sum(t["rows"] for t in tables)
            ds.column_count = len(first["columns"])
            ds.sheets = [t["name"] for t in tables]
            ds.profile = {"tables": tables}
        db.add(UsageRecord(org_id=ctx.org_id, user_id=ctx.created_by, kind="embedding", provider=embedder.provider, model=model, kb_id=kb_id, tokens_in=sum(estimate_tokens(drafts[i].content) for i in retrievable), quantity=len(retrievable)))
        emit(db, ctx.org_id, "document.ingested", resource_type="document", resource_id=doc_id, actor_user_id=ctx.created_by,
             message=f"Indexed {doc.filename}: {count} chunks", severity="warning" if findings else "success",
             payload={"kb_id": str(kb_id), "chunks": count, "pages": parsed.page_count, "parser": parsed.parser, "injection_suspect": bool(findings)})
        emit(db, ctx.org_id, "index.updated", resource_type="knowledge_base", resource_id=kb_id, payload={"document_id": str(doc_id), "chunks": count, "model": model})
        emit(db, ctx.org_id, "pipeline.completed", resource_type="document", resource_id=doc_id, payload={"kb_id": str(kb_id), "ms": doc.meta["processing_ms"]})
    return {"chunks": count, "parser": parsed.parser, "pages": parsed.page_count, "tables": len(tables), "flagged_chunks": flagged}


def _fail(ctx: JobContext, doc_id: uuid.UUID, message: str) -> None:
    with ctx.session() as db:
        doc = db.get(Document, doc_id)
        if doc is None:
            return
        doc.status = "failed"
        doc.error = message[:2000]
        emit(db, ctx.org_id, "document.failed", resource_type="document", resource_id=doc_id, severity="error", message=f"{doc.filename}: {message[:300]}", payload={"kb_id": str(doc.kb_id)})


def _on_dead(payload: dict[str, Any], org_id: uuid.UUID, job_id: uuid.UUID) -> None:
    from app.db.session import tenant_session
    from app.models import Job

    with tenant_session(org_id) as db:
        doc = db.get(Document, uuid.UUID(payload["document_id"]))
        job = db.get(Job, job_id)
        if doc is not None and doc.status != "failed":
            doc.status = "failed"
            doc.error = (job.error if job else "Processing failed") or "Processing failed"
            emit(db, org_id, "document.failed", resource_type="document", resource_id=doc.id, severity="error", message=f"{doc.filename}: {doc.error[:300]}", payload={"kb_id": str(doc.kb_id)})


DEAD_HOOKS["ingest_document"] = _on_dead


@handler("reindex_kb")
def reindex_kb(ctx: JobContext) -> dict[str, Any]:
    """Re-embed every document of a knowledge base (after an embedding/chunking change)."""
    from app.jobs.queue import enqueue

    kb_id = uuid.UUID(ctx.payload["kb_id"])
    with ctx.session() as db:
        db.query(ChunkEmbedding).filter(ChunkEmbedding.kb_id == kb_id).delete(synchronize_session=False)
        docs = list(db.scalars(select(Document).where(Document.kb_id == kb_id, Document.status.in_(("ready", "failed")))))
        stamp = int(time.time())
        for d in docs:
            d.status = "queued"
            d.embedding_status = "pending"
            enqueue(db, ctx.org_id, "ingest_document", payload={"document_id": str(d.id)}, created_by=ctx.created_by, resource_type="document", resource_id=d.id, idempotency_key=f"ingest:{d.id}:reindex:{stamp}")
        emit(db, ctx.org_id, "index.rebuild", resource_type="knowledge_base", resource_id=kb_id, message=f"Re-indexing {len(docs)} documents", payload={"documents": len(docs)})
    return {"documents": len(docs)}
