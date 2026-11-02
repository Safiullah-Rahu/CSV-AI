"""Retrieval pipeline: query → candidates → fusion → rerank → diversify → context units.

Stages (each recorded in the trace for retrieval transparency):

1. Optional query expansion (pseudo-relevance feedback offline, or LLM
   paraphrases for multi-query when a model is configured).
2. Dense (pgvector) and sparse (BM25) candidate generation per embedding
   space, inside the caller's authorized scope.
3. Fusion: reciprocal-rank fusion (default) or weighted score fusion.
4. Reranking: ``lexical`` feature reranker (dense + BM25 + coverage +
   proximity), or ``cohere`` / ``cross_encoder`` / ``none``.
5. Similarity threshold, MMR diversification, parent-child expansion and
   optional contextual compression.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
from sqlalchemy.orm import Session

from app.ingestion.normalize import STOPWORDS, tokenize
from app.rag import index
from app.rag.chunking import split_sentences
from app.rag.embeddings import Embedder
from app.rbac.authorizer import AccessScope

DEFAULT_RETRIEVAL: dict[str, Any] = {
    "mode": "hybrid",            # hybrid | dense | sparse | keyword
    "top_k": 6,
    "candidate_k": 40,
    "fusion": "rrf",             # rrf | weighted
    "rrf_k": 60,
    "alpha": 0.5,                # weighted fusion: dense share
    "rerank": "lexical",         # lexical | none | cohere | cross_encoder
    "rerank_top_n": 30,
    "min_score": 0.0,            # drop reranked items below this
    "min_evidence": 0.2,         # below → "insufficient evidence"
    "mmr": True,
    "mmr_lambda": 0.75,
    "query_expansion": "prf",    # prf | none
    "multi_query": 0,            # LLM paraphrases (needs a model)
    "compression": False,
    "max_context_tokens": 6000,
    "quarantine_suspicious": False,
}


@dataclass
class Retrieved:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    kb_id: uuid.UUID
    content: str
    meta: dict[str, Any]
    title: str
    filename: str
    ext: str
    source_url: str | None
    source_type: str
    parent_content: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    coverage: float = 0.0

    @property
    def score(self) -> float:
        return self.scores.get("rerank", self.scores.get("fused", 0.0))

    @property
    def flagged(self) -> bool:
        return bool(self.meta.get("injection_suspect"))

    @property
    def context_text(self) -> str:
        return self.parent_content or self.content


@dataclass
class RetrievalResult:
    items: list[Retrieved]
    trace: dict[str, Any]
    evidence: float
    sufficient: bool


@dataclass
class EmbeddingSpace:
    """Knowledge bases sharing one embedding model (searched together)."""

    embedder: Embedder
    kb_ids: list[uuid.UUID]


def merge_retrieval_config(*configs: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULT_RETRIEVAL)
    for c in configs:
        if c:
            out.update({k: v for k, v in c.items() if v is not None})
    out["top_k"] = max(1, min(int(out["top_k"]), 50))
    out["candidate_k"] = max(out["top_k"], min(int(out["candidate_k"]), 400))
    return out


# --------------------------------------------------------------------------- scoring


def coverage(terms: list[str], text: str, idf: dict[str, float] | None = None) -> float:
    """Share of query terms present in text; IDF-weighted when idf is given, so
    rare, informative terms ("parental") matter more than ubiquitous ones ("policy")."""
    if not terms:
        return 0.0
    tokens = set(tokenize(text))

    def present(t: str) -> bool:
        return t in tokens or (len(t) >= index.PREFIX_MIN and any(tok.startswith(t) for tok in tokens))

    if idf:
        total = sum(idf.get(t, 1.0) for t in terms)
        return sum(idf.get(t, 1.0) for t in terms if present(t)) / total if total else 0.0
    return sum(1 for t in terms if present(t)) / len(terms)


def proximity(terms: list[str], text: str) -> float:
    tokens = tokenize(text)
    long_terms = [t for t in terms if len(t) >= index.PREFIX_MIN]
    exact = set(terms)

    def match(tok: str) -> str | None:
        if tok in exact:
            return tok
        return next((t for t in long_terms if tok.startswith(t)), None)

    positions = [(i, m) for i, tok in enumerate(tokens) if (m := match(tok)) is not None]
    present = {t for _, t in positions}
    if len(present) < 2:
        return 1.0 if present else 0.0
    best = len(tokens)
    counts: dict[str, int] = defaultdict(int)
    left = 0
    have = 0
    for pos_r, term_r in positions:
        counts[term_r] += 1
        if counts[term_r] == 1:
            have += 1
        while have == len(present):
            best = min(best, pos_r - positions[left][0] + 1)
            term_l = positions[left][1]
            counts[term_l] -= 1
            if counts[term_l] == 0:
                have -= 1
            left += 1
    return 1.0 / (1.0 + max(0, best - len(present)) / 8.0)


def rrf(rankings: list[list[uuid.UUID]], k: int = 60) -> dict[uuid.UUID, float]:
    scores: dict[uuid.UUID, float] = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] += 1.0 / (k + rank)
    return dict(scores)


def _norm(values: dict[uuid.UUID, float]) -> dict[uuid.UUID, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def prf_expansion(terms: list[str], top_texts: list[str], n_terms: int = 5) -> dict[str, float]:
    """Pseudo-relevance feedback: frequent co-occurring terms from top results (weight 0.35)."""
    counts: dict[str, int] = defaultdict(int)
    for text in top_texts[:5]:
        for tok in set(tokenize(text)):
            if tok not in STOPWORDS and tok not in terms and len(tok) > 3 and not tok.isdigit():
                counts[tok] += 1
    expansions = [t for t, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if c >= 2][:n_terms]
    return {t: 0.35 for t in expansions}


def lexical_rerank(items: list[Retrieved], terms: list[str], dense_sim: dict[uuid.UUID, float], idf: dict[str, float] | None = None) -> None:
    bm = _norm({i.chunk_id: i.scores.get("sparse", 0.0) for i in items})
    for it in items:
        cov = coverage(terms, it.content, idf)
        prox = proximity(terms, it.content)
        dense = max(0.0, min(1.0, dense_sim.get(it.chunk_id, it.scores.get("dense", 0.0))))
        it.coverage = cov
        it.scores["dense"] = dense
        it.scores["rerank"] = round(0.35 * dense + 0.25 * bm.get(it.chunk_id, 0.0) + 0.30 * cov + 0.10 * prox, 6)


def cohere_rerank(items: list[Retrieved], query: str, api_key: str, model: str = "rerank-v3.5") -> None:
    r = httpx.post(
        "https://api.cohere.com/v2/rerank",
        json={"model": model, "query": query, "documents": [i.content[:4000] for i in items], "top_n": len(items)},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    r.raise_for_status()
    for res in r.json()["results"]:
        items[res["index"]].scores["rerank"] = float(res["relevance_score"])


def cross_encoder_rerank(items: list[Retrieved], query: str, model: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    scores = list(TextCrossEncoder(model_name=model).rerank(query, [i.content for i in items]))
    for it, s in zip(items, scores):
        it.scores["rerank"] = float(1 / (1 + np.exp(-s)))


def mmr(items: list[Retrieved], vectors: dict[uuid.UUID, list[float]], k: int, lam: float) -> list[Retrieved]:
    if len(items) <= k or not vectors:
        return items[:k]
    vecs = {cid: np.asarray(v, dtype=np.float32) / max(np.linalg.norm(v), 1e-9) for cid, v in vectors.items()}
    selected: list[Retrieved] = []
    pool = list(items)
    while pool and len(selected) < k:
        best, best_val = None, -1e9
        for it in pool:
            v = vecs.get(it.chunk_id)
            redundancy = max((float(v @ vecs[s.chunk_id]) for s in selected if v is not None and s.chunk_id in vecs), default=0.0)
            val = lam * it.score - (1 - lam) * redundancy
            if val > best_val:
                best, best_val = it, val
        assert best is not None
        selected.append(best)
        pool.remove(best)
    return selected


def compress(text: str, terms: list[str], max_chars: int = 1200) -> str:
    sentences = split_sentences(text)
    if len(text) <= max_chars or len(sentences) <= 2:
        return text
    scored = [(coverage(terms, s), i, s) for i, s in enumerate(sentences)]
    keep: set[int] = set()
    total = 0
    for score, i, s in sorted(scored, key=lambda x: (-x[0], x[1])):
        if score == 0 and keep:
            break
        if total + len(s) > max_chars and keep:
            continue
        keep.add(i)
        total += len(s)
    parts, last = [], -2
    for i in sorted(keep):
        if i != last + 1 and parts:
            parts.append("…")
        parts.append(sentences[i])
        last = i
    return " ".join(parts)


# ------------------------------------------------------------------------- pipeline


def retrieve(
    db: Session,
    *,
    scope: AccessScope,
    spaces: list[EmbeddingSpace],
    query: str,
    config: dict[str, Any],
    filters: index.SearchFilters | None = None,
    user_id: uuid.UUID | None = None,
    paraphrase: Callable[[str, int], list[str]] | None = None,
    rerank_credentials: dict[str, str] | None = None,
) -> RetrievalResult:
    cfg = merge_retrieval_config(config)
    trace: dict[str, Any] = {"mode": cfg["mode"], "stages": []}
    t0 = time.perf_counter()
    terms = index.query_terms(query)
    if cfg["quarantine_suspicious"]:
        filters = filters or index.SearchFilters()
        filters.exclude_flagged = True
    queries = [query]
    if cfg["multi_query"] and paraphrase is not None:
        try:
            extra = [q for q in paraphrase(query, int(cfg["multi_query"])) if q.strip() and q.strip() != query][: int(cfg["multi_query"])]
            queries += extra
            trace["multi_query"] = extra
        except Exception as exc:  # noqa: BLE001 — retrieval proceeds with the original query
            trace["multi_query_error"] = type(exc).__name__
    rankings: list[list[uuid.UUID]] = []
    dense_scores: dict[uuid.UUID, float] = {}
    sparse_scores: dict[uuid.UUID, float] = {}
    query_vectors: dict[int, list[float]] = {}
    corpus_n = 0
    corpus_df: dict[str, int] = {}
    k = cfg["candidate_k"]
    for s_index, space in enumerate(spaces):
        allowed = [kb for kb in space.kb_ids if kb in scope.kb_levels]
        if not allowed:
            continue
        sql = index.scope_sql(scope, allowed, filters, user_id)
        stats = index.term_stats(db, sql, terms)
        corpus_n += stats.n
        for t, d in stats.df.items():
            corpus_df[t] = corpus_df.get(t, 0) + d
        for q in queries:
            if cfg["mode"] in ("hybrid", "dense"):
                qv = space.embedder.embed_query(q)
                if q == query:
                    query_vectors[s_index] = qv
                hits = index.dense_search(db, sql, query_vec=qv, model=space.embedder.model, dim=space.embedder.dim, k=k)
                rankings.append([h[0] for h in hits])
                for cid, sc in hits:
                    dense_scores[cid] = max(dense_scores.get(cid, 0.0), sc)
            if cfg["mode"] in ("hybrid", "sparse"):
                q_terms = index.query_terms(q)
                hits = index.sparse_search(db, sql, terms=q_terms, k=k)
                if cfg["query_expansion"] == "prf" and q == query and hits:
                    expansion = prf_expansion(q_terms, [h[2] for h in hits])
                    if expansion:
                        trace["prf_terms"] = list(expansion)
                        weights = {t: 1.0 for t in q_terms} | expansion
                        hits = index.sparse_search(db, sql, terms=q_terms + list(expansion), k=k, weights=weights)
                rankings.append([h[0] for h in hits])
                for cid, sc, _ in hits:
                    sparse_scores[cid] = max(sparse_scores.get(cid, 0.0), sc)
            if cfg["mode"] == "keyword":
                hits2 = index.keyword_search(db, sql, query=q, k=k)
                rankings.append([h[0] for h in hits2])
                for cid, sc in hits2:
                    sparse_scores[cid] = max(sparse_scores.get(cid, 0.0), sc)
    trace["stages"].append({"stage": "candidates", "dense": len(dense_scores), "sparse": len(sparse_scores), "ms": round((time.perf_counter() - t0) * 1000, 1)})
    if cfg["fusion"] == "weighted" and cfg["mode"] == "hybrid":
        dn, sn = _norm(dense_scores), _norm(sparse_scores)
        alpha = float(cfg["alpha"])
        fused = {cid: alpha * dn.get(cid, 0.0) + (1 - alpha) * sn.get(cid, 0.0) for cid in set(dn) | set(sn)}
    else:
        fused = rrf(rankings, int(cfg["rrf_k"]))
    ordered = sorted(fused, key=lambda c: fused[c], reverse=True)[: int(cfg["rerank_top_n"])]
    rows = index.hydrate(db, ordered)
    items: list[Retrieved] = []
    for cid in ordered:
        r = rows.get(cid)
        if r is None:
            continue
        items.append(
            Retrieved(
                chunk_id=cid, document_id=r["document_id"], kb_id=r["kb_id"], content=r["content"], meta=r["metadata"] or {},
                title=r["title"], filename=r["filename"], ext=r["ext"], source_url=r["source_url"], source_type=r["source_type"],
                parent_content=r["parent_content"],
                scores={"fused": round(fused[cid], 6), "dense": round(dense_scores.get(cid, 0.0), 6), "sparse": round(sparse_scores.get(cid, 0.0), 6)},
            )
        )
    t1 = time.perf_counter()
    # Dense similarity for every candidate (also sparse-only ones) to feed the reranker and MMR.
    vectors = index.embeddings_for(db, [i.chunk_id for i in items])
    dense_sim: dict[uuid.UUID, float] = {}
    if query_vectors:
        space_of = {kb: i for i, sp in enumerate(spaces) for kb in sp.kb_ids}
        for it in items:
            v = vectors.get(it.chunk_id)
            qv = query_vectors.get(space_of.get(it.kb_id, -1))
            if v is not None and qv is not None and len(v) == len(qv):
                a, b = np.asarray(v), np.asarray(qv)
                dense_sim[it.chunk_id] = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))
    idf = index.TermStats(corpus_n, 1.0, corpus_df).idf if corpus_n else None
    reranker = cfg["rerank"]
    try:
        if reranker == "cohere" and rerank_credentials and rerank_credentials.get("cohere"):
            lexical_rerank(items, terms, dense_sim, idf)
            cohere_rerank(items, query, rerank_credentials["cohere"])
        elif reranker == "cross_encoder":
            lexical_rerank(items, terms, dense_sim, idf)
            cross_encoder_rerank(items, query)
        elif reranker == "none":
            for it in items:
                it.coverage = coverage(terms, it.content, idf)
                it.scores["rerank"] = it.scores["fused"]
        else:
            lexical_rerank(items, terms, dense_sim, idf)
    except Exception as exc:  # noqa: BLE001 — fall back to the lexical reranker
        trace["rerank_error"] = f"{reranker}: {type(exc).__name__}"
        lexical_rerank(items, terms, dense_sim, idf)
        reranker = "lexical"
    items.sort(key=lambda i: i.score, reverse=True)
    items = [i for i in items if i.score >= float(cfg["min_score"])]
    trace["stages"].append({"stage": "rerank", "reranker": reranker, "candidates": len(items), "ms": round((time.perf_counter() - t1) * 1000, 1)})
    # Parent-child: keep the best child per parent; the parent text becomes the context.
    seen_parents: set[str] = set()
    deduped: list[Retrieved] = []
    for it in items:
        key = (it.parent_content or "")[:200] + str(it.document_id) if it.parent_content else str(it.chunk_id)
        if key in seen_parents:
            continue
        seen_parents.add(key)
        deduped.append(it)
    final = mmr(deduped, vectors, cfg["top_k"], float(cfg["mmr_lambda"])) if cfg["mmr"] else deduped[: cfg["top_k"]]
    if cfg["compression"]:
        for it in final:
            it.parent_content = compress(it.context_text, terms)
    floor = min((float(getattr(sp.embedder, "similarity_floor", 0.0)) for sp in spaces), default=0.0)
    evidence = max((0.6 * i.coverage + 0.4 * max(0.0, (i.scores.get("dense", 0.0) - floor) / (1 - floor)) for i in final), default=0.0)
    sufficient = bool(final) and evidence >= float(cfg["min_evidence"])
    trace.update({"terms": terms, "returned": len(final), "evidence": round(evidence, 4), "sufficient": sufficient, "total_ms": round((time.perf_counter() - t0) * 1000, 1)})
    return RetrievalResult(items=final, trace=trace, evidence=evidence, sufficient=sufficient)
