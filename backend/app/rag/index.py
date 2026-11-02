"""Index storage and low-level search (pgvector dense + BM25 sparse).

Authorization is part of every query: callers pass an :class:`AccessScope`
and the SQL itself restricts rows to readable knowledge bases, removes
denied documents and enforces restricted-document grants. Unauthorized
chunks are never fetched, scored or returned.
"""

from __future__ import annotations

import json
import math
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, insert, text
from sqlalchemy.orm import Session

from app.ingestion.normalize import STOPWORDS, tokenize
from app.models import Chunk, ChunkEmbedding
from app.rag.chunking import ChunkDraft
from app.rbac.authorizer import AccessScope

MAX_INDEXED_DIM = 2000  # pgvector HNSW limit for the vector type


def vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def ensure_vector_index(db: Session, dim: int) -> None:
    if dim > MAX_INDEXED_DIM:
        return  # exact scan; still correct, just slower on huge knowledge bases
    db.execute(
        text(
            f"CREATE INDEX IF NOT EXISTS ix_chunk_embeddings_hnsw_{int(dim)} ON chunk_embeddings "
            f"USING hnsw ((embedding::vector({int(dim)})) vector_cosine_ops) WHERE dim = {int(dim)}"
        )
    )


def replace_document_chunks(
    db: Session,
    *,
    org_id: uuid.UUID,
    kb_id: uuid.UUID,
    document_id: uuid.UUID,
    version: int,
    drafts: list[ChunkDraft],
    vectors: list[list[float] | None],
    model: str,
    dim: int,
) -> int:
    """Atomically swap a document's chunks/embeddings (incremental re-processing)."""
    db.execute(delete(Chunk).where(Chunk.document_id == document_id))
    ids = [uuid.uuid4() for _ in drafts]
    rows = []
    for i, d in enumerate(drafts):
        rows.append(
            {
                "id": ids[i], "org_id": org_id, "kb_id": kb_id, "document_id": document_id, "version": version,
                "ordinal": i, "parent_id": ids[d.parent] if d.parent is not None else None,
                "is_retrievable": d.retrievable, "content": d.content, "token_count": d.tokens, "metadata": d.meta,
            }
        )
    # Parents first so self-references resolve.
    parents = [r for r in rows if r["parent_id"] is None]
    children = [r for r in rows if r["parent_id"] is not None]
    for batch in (parents, children):
        for i in range(0, len(batch), 500):
            db.execute(insert(Chunk.__table__), batch[i : i + 500])
    emb_rows = [
        {"chunk_id": ids[i], "org_id": org_id, "kb_id": kb_id, "model": model, "dim": dim, "embedding": v}
        for i, v in enumerate(vectors)
        if v is not None
    ]
    for i in range(0, len(emb_rows), 500):
        db.execute(insert(ChunkEmbedding.__table__), emb_rows[i : i + 500])
    return len(rows)


# ------------------------------------------------------------------------- filters


@dataclass
class SearchFilters:
    document_ids: list[uuid.UUID] | None = None
    exts: list[str] | None = None
    tags: list[str] | None = None
    source_types: list[str] | None = None
    created_from: Any = None
    created_to: Any = None
    metadata: dict[str, Any] | None = None
    exclude_flagged: bool = False


@dataclass
class ScopeSQL:
    where: str
    params: dict[str, Any] = field(default_factory=dict)


def scope_sql(scope: AccessScope, kb_ids: list[uuid.UUID], filters: SearchFilters | None, user_id: uuid.UUID | None) -> ScopeSQL:
    f = filters or SearchFilters()
    clauses = ["c.kb_id = ANY(:kb_ids)", "c.is_retrievable", "d.status = 'ready'"]
    params: dict[str, Any] = {"kb_ids": kb_ids}
    if scope.denied_document_ids:
        clauses.append("NOT (d.id = ANY(:denied_docs))")
        params["denied_docs"] = list(scope.denied_document_ids)
    acl = ["d.acl_mode = 'inherit'"]
    if scope.restricted_allowed_document_ids:
        acl.append("d.id = ANY(:restricted_ok)")
        params["restricted_ok"] = list(scope.restricted_allowed_document_ids)
    manage = [k for k in kb_ids if k in scope.kb_manage_ids]
    if manage:
        acl.append("d.kb_id = ANY(:manage_kbs)")
        params["manage_kbs"] = manage
    if user_id:
        acl.append("d.owner_id = :uid")
        params["uid"] = user_id
    clauses.append("(" + " OR ".join(acl) + ")")
    if f.document_ids:
        clauses.append("d.id = ANY(:f_docs)")
        params["f_docs"] = f.document_ids
    if f.exts:
        clauses.append("d.ext = ANY(:f_exts)")
        params["f_exts"] = [e.lower().lstrip(".") for e in f.exts]
    if f.tags:
        clauses.append("d.tags && CAST(:f_tags AS text[])")
        params["f_tags"] = f.tags
    if f.source_types:
        clauses.append("d.source_type = ANY(:f_src)")
        params["f_src"] = f.source_types
    if f.created_from:
        clauses.append("d.created_at >= :f_from")
        params["f_from"] = f.created_from
    if f.created_to:
        clauses.append("d.created_at <= :f_to")
        params["f_to"] = f.created_to
    if f.metadata:
        clauses.append("c.metadata @> CAST(:f_meta AS jsonb)")
        params["f_meta"] = json.dumps(f.metadata)
    if f.exclude_flagged:
        clauses.append("NOT coalesce((c.metadata->>'injection_suspect')::boolean, false)")
    return ScopeSQL(" AND ".join(clauses), params)


# --------------------------------------------------------------------------- dense


def dense_search(db: Session, sql: ScopeSQL, *, query_vec: list[float], model: str, dim: int, k: int) -> list[tuple[uuid.UUID, float]]:
    d = int(dim)
    if k > 40:
        try:
            db.execute(text("SET LOCAL hnsw.ef_search = :ef"), {"ef": min(1000, k * 4)})
        except Exception:  # noqa: BLE001 — older pgvector without the GUC
            pass
    rows = db.execute(
        text(
            f"""
            SELECT c.id, 1 - (e.embedding::vector({d}) <=> CAST(:q AS vector({d}))) AS score
            FROM chunk_embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE e.dim = {d} AND e.model = :model AND {sql.where}
            ORDER BY e.embedding::vector({d}) <=> CAST(:q AS vector({d}))
            LIMIT :k
            """
        ),
        {**sql.params, "q": vector_literal(query_vec), "model": model, "k": k},
    ).all()
    return [(r[0], float(r[1])) for r in rows]


def embeddings_for(db: Session, chunk_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[float]]:
    if not chunk_ids:
        return {}
    rows = db.execute(text("SELECT chunk_id, embedding::text FROM chunk_embeddings WHERE chunk_id = ANY(:ids)"), {"ids": chunk_ids}).all()
    return {r[0]: json.loads(r[1]) for r in rows}


# -------------------------------------------------------------------------- sparse


def query_terms(query: str) -> list[str]:
    terms = [t for t in tokenize(query) if t not in STOPWORDS and len(t) > 1]
    return list(dict.fromkeys(terms))[:24]


PREFIX_MIN = 4  # terms this long also match longer word forms ("order" -> "orders")


def ts_term(term: str) -> str:
    return f"{term}:*" if len(term) >= PREFIX_MIN else term


def term_frequency(term: str, tf: Counter[str]) -> int:
    if len(term) < PREFIX_MIN:
        return tf.get(term, 0)
    return sum(c for tok, c in tf.items() if tok.startswith(term))


@dataclass
class TermStats:
    n: int
    avgdl: float
    df: dict[str, int]

    @property
    def idf(self) -> dict[str, float]:
        return {t: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for t, d in self.df.items()}


def term_stats(db: Session, sql: ScopeSQL, terms: list[str]) -> TermStats:
    """Exact corpus size, average length and document frequencies over the authorized scope."""
    terms = [t for t in terms if t.replace("_", "").isalnum()]
    if not terms:
        return TermStats(0, 1.0, {})
    cols = ", ".join(f"count(*) FILTER (WHERE c.tsv @@ to_tsquery('simple', :t{i}))" for i in range(len(terms)))
    row = db.execute(
        text(f"SELECT count(*), coalesce(avg(greatest(c.token_count, 1)), 1), {cols} FROM chunks c JOIN documents d ON d.id = c.document_id WHERE {sql.where}"),
        {**sql.params, **{f"t{i}": ts_term(t) for i, t in enumerate(terms)}},
    ).one()
    return TermStats(int(row[0]), float(row[1]), {t: int(row[2 + i]) for i, t in enumerate(terms)})


def sparse_search(
    db: Session,
    sql: ScopeSQL,
    *,
    terms: list[str],
    k: int,
    weights: dict[str, float] | None = None,
    k1: float = 1.2,
    b: float = 0.75,
    candidates: int = 400,
) -> list[tuple[uuid.UUID, float, str]]:
    """Okapi BM25 over the authorized corpus.

    Full-text (GIN) selects candidates; exact document frequencies, corpus size
    and average length are computed over the *same authorized scope*, so scores
    never depend on (or leak) documents the caller cannot read.
    """
    terms = [t for t in terms if t.replace("_", "").isalnum()]
    if not terms:
        return []
    weights = weights or {}
    tsq = " | ".join(ts_term(t) for t in terms)
    stats = term_stats(db, sql, terms)
    if stats.n == 0:
        return []
    idf, avgdl = stats.idf, stats.avgdl
    rows = db.execute(
        text(
            f"""
            SELECT c.id, c.content, c.token_count
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE {sql.where} AND c.tsv @@ to_tsquery('simple', :tsq)
            ORDER BY ts_rank_cd(c.tsv, to_tsquery('simple', :tsq)) DESC
            LIMIT :cand
            """
        ),
        {**sql.params, "tsq": tsq, "cand": candidates},
    ).all()
    scored: list[tuple[uuid.UUID, float, str]] = []
    for chunk_id, content, token_count in rows:
        tf = Counter(tokenize(content))
        dl = max(token_count or len(tf), 1)
        score = 0.0
        for t in terms:
            f = term_frequency(t, tf)
            if f:
                score += weights.get(t, 1.0) * idf[t] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        if score > 0:
            scored.append((chunk_id, score, content))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def keyword_search(db: Session, sql: ScopeSQL, *, query: str, k: int) -> list[tuple[uuid.UUID, float]]:
    """All-terms keyword match (websearch syntax: quotes, OR, -exclusions)."""
    rows = db.execute(
        text(
            f"""
            SELECT c.id, ts_rank_cd(c.tsv, websearch_to_tsquery('simple', :q)) AS score
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE {sql.where} AND c.tsv @@ websearch_to_tsquery('simple', :q)
            ORDER BY score DESC LIMIT :k
            """
        ),
        {**sql.params, "q": query[:500], "k": k},
    ).all()
    return [(r[0], float(r[1])) for r in rows]


def hydrate(db: Session, chunk_ids: list[uuid.UUID]) -> dict[uuid.UUID, dict[str, Any]]:
    if not chunk_ids:
        return {}
    rows = db.execute(
        text(
            """
            SELECT c.id, c.document_id, c.kb_id, c.content, c.metadata, c.parent_id, c.ordinal,
                   d.title, d.filename, d.ext, d.source_url, d.source_type, p.content AS parent_content,
                   d.created_at, d.tags
            FROM chunks c JOIN documents d ON d.id = c.document_id
            LEFT JOIN chunks p ON p.id = c.parent_id
            WHERE c.id = ANY(:ids)
            """
        ),
        {"ids": chunk_ids},
    ).mappings().all()
    return {r["id"]: dict(r) for r in rows}
