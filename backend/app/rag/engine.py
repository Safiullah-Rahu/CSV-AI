"""Answer engine: permission-aware conversational RAG with streaming.

``answer_stream`` yields protocol events (``status``, ``sources``, ``token``,
``done``, ``error``). It runs in its own tenant session (RLS on), recomputes
the caller's access scope at answer time, and persists the assistant
message, citations and usage — also when the client stops generation.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import tenant_session
from app.ingestion.normalize import STOPWORDS, tokenize
from app.llm.base import Message as LLMMessage
from app.llm.registry import ResolvedLLM, resolve_embedder, resolve_llm
from app.models import Agent, Citation, Conversation, KnowledgeBase, Message, Organization, UsageRecord
from app.rag import grounding
from app.rag.index import SearchFilters
from app.rag.retrieval import EmbeddingSpace, Retrieved, merge_retrieval_config, retrieve
from app.rbac.authorizer import AccessScope, Authorizer, Principal
from app.utils.errors import CSVAIError
from app.utils.logging import get_logger
from app.utils.tokens import estimate_tokens

log = get_logger("rag.engine")

ANALYTIC = re.compile(r"\b(total|sum|average|avg|mean|median|count|how many|number of|max(imum)?|min(imum)?|top \d+|highest|lowest|by (month|year|quarter|region|department|customer)|per (month|year)|trend|breakdown|group(ed)? by|percentage|ratio)\b", re.I)


@dataclass
class AskOptions:
    kb_ids: list[uuid.UUID] | None = None
    filters: SearchFilters = field(default_factory=SearchFilters)
    retrieval: dict[str, Any] = field(default_factory=dict)
    provider_id: uuid.UUID | None = None
    model: str | None = None
    agent_id: uuid.UUID | None = None
    policy: str | None = None
    use_structured: bool = True


def embedding_spaces(db: Session, org_id: uuid.UUID, kbs: list[KnowledgeBase]) -> list[EmbeddingSpace]:
    groups: dict[tuple[Any, ...], list[KnowledgeBase]] = {}
    for kb in kbs:
        cfg = kb.embedding_config or {"provider": "hashing", "model": "hashing-768", "dim": 768}
        key = (cfg.get("provider"), cfg.get("model"), cfg.get("dim"), str(cfg.get("provider_id")))
        groups.setdefault(key, []).append(kb)
    spaces = []
    for members in groups.values():
        cfg = members[0].embedding_config or {"provider": "hashing", "model": "hashing-768", "dim": 768}
        try:
            embedder = resolve_embedder(db, org_id, cfg)
        except CSVAIError as exc:
            log.warning("embedding space unavailable: %s", exc)
            continue
        spaces.append(EmbeddingSpace(embedder=embedder, kb_ids=[k.id for k in members]))
    return spaces


def condense_question(question: str, history: list[dict[str, str]], llm: ResolvedLLM | None) -> tuple[str, str]:
    """Standalone retrieval query for a follow-up question. Returns (query, method)."""
    previous = [m["content"] for m in history if m["role"] == "user"]
    if not previous:
        return question, "none"
    if llm is not None:
        prompt = [
            LLMMessage("system", "Rewrite the user's follow-up as one standalone search question. Output only the question."),
            LLMMessage("user", "Conversation so far:\n" + "\n".join(f"- {p[:300]}" for p in previous[-3:]) + f"\n\nFollow-up: {question}"),
        ]
        try:
            rewritten = llm.provider.complete(prompt).content.strip().strip('"')
            if 3 <= len(rewritten) <= 500:
                return rewritten, "llm"
        except CSVAIError:
            pass
    content = [t for t in tokenize(question) if t not in STOPWORDS]
    if len(content) < 5:
        carry = [t for t in tokenize(previous[-1]) if t not in STOPWORDS][:8]
        return f"{question} {' '.join(carry)}", "carry-over"
    return question, "none"


def _paraphraser(llm: ResolvedLLM | None):  # noqa: ANN202
    if llm is None:
        return None

    def paraphrase(q: str, n: int) -> list[str]:
        out = llm.provider.complete([
            LLMMessage("system", f"Write {n} alternative phrasings of the search question, one per line, no numbering."),
            LLMMessage("user", q),
        ]).content
        return [line.strip("-• ").strip() for line in out.splitlines() if line.strip()]

    return paraphrase


def _citation_type(it: Retrieved) -> str:
    return {"sql_dataset": "dataset", "sql_live": "database"}.get(it.source_type, "document")


def _citation_payload(src: grounding.Source, cited: bool) -> dict[str, Any]:
    it = src.item
    return {
        "label": src.label, "ordinal": int(src.label[1:]), "source_type": _citation_type(it),
        "title": it.title, "filename": it.filename,
        "document_id": None if it.source_type == "sql_live" else str(it.document_id), "kb_id": str(it.kb_id),
        "chunk_id": None if it.source_type.startswith("sql_") else str(it.chunk_id), "locator": src.locator(), "locator_text": src.locator_text(),
        "snippet": it.content[:1200], "score": round(it.score, 4), "scores": it.scores, "flagged": it.flagged, "cited": cited,
    }


def answer_stream(
    *,
    principal: Principal,
    org_id: uuid.UUID,
    conversation_id: uuid.UUID,
    assistant_message_id: uuid.UUID,
    question: str,
    options: AskOptions,
) -> Iterator[dict[str, Any]]:
    t_start = time.perf_counter()
    state: dict[str, Any] = {"content": "", "status": "streaming", "persisted": False}
    db_ctx = tenant_session(org_id, principal.user_id)
    db = db_ctx.__enter__()
    try:
        authz = Authorizer(db, principal).load()
        org = db.get(Organization, org_id)
        conv = db.get(Conversation, conversation_id)
        assert org is not None and conv is not None
        agent = db.get(Agent, options.agent_id) if options.agent_id else None
        history = [
            {"role": m.role, "content": m.content}
            for m in db.scalars(
                select(Message).where(Message.conversation_id == conv.id, Message.id != assistant_message_id, Message.status.in_(("complete", "refused")))
                .order_by(Message.created_at.desc()).limit(9)
            )
        ][::-1][:-1]  # drop the just-added user question
        requested = options.kb_ids if options.kb_ids is not None else (conv.kb_ids or None)
        if agent is not None:
            # Agents narrow scope; they can never widen it beyond the user's own access.
            requested = [k for k in (requested or agent.kb_ids) if k in set(agent.kb_ids)] if agent.kb_ids else requested
        scope: AccessScope = authz.access_scope(requested)
        kb_rows = list(db.scalars(select(KnowledgeBase).where(KnowledgeBase.id.in_(scope.readable_kb_ids())))) if scope.kb_levels else []
        # kb.chat may be granted per workspace: keep only knowledge bases where it applies.
        kb_rows = [k for k in kb_rows if authz.has("kb.chat", k.workspace_id)]
        yield {"type": "status", "stage": "retrieving", "knowledge_bases": [{"id": str(k.id), "name": k.name} for k in kb_rows]}
        llm: ResolvedLLM | None = None
        llm_error = None
        try:
            llm = resolve_llm(db, org, provider_id=options.provider_id or (agent.provider_id if agent else None), model=options.model or (agent.model if agent else None))
        except CSVAIError as exc:
            llm_error = exc.message
        if not kb_rows:
            text = "You don't have access to any knowledge base in this scope, so I can't answer from organizational sources."
            yield from _finish(db, state, conv, assistant_message_id, text, mode="refused", sources=[], evidence=0.0, trace={"reason": "no_authorized_kbs"}, t_start=t_start, llm=None, principal=principal, verification=None)
            return
        kb_cfg = merge_retrieval_config(*(k.retrieval_config for k in kb_rows[:1]), agent.retrieval_config if agent else None, options.retrieval)
        policy = options.policy or (kb_rows[0].generation_config or {}).get("grounding_policy") or ((org.settings or {}).get("ai") or {}).get("grounding_policy") or "strict"
        query, condense_method = condense_question(question, history, llm)
        spaces = embedding_spaces(db, org_id, kb_rows)
        result = retrieve(
            db, scope=scope, spaces=spaces, query=query, config=kb_cfg, filters=options.filters, user_id=principal.user_id,
            paraphrase=_paraphraser(llm) if kb_cfg.get("multi_query") else None,
        )
        result.trace["condensed_query"] = query if condense_method != "none" else None
        result.trace["condense_method"] = condense_method
        retrieval_ms = (time.perf_counter() - t_start) * 1000
        structured: list[Retrieved] = []
        if options.use_structured and llm is not None and ANALYTIC.search(question):
            from app.services.dataset_service import structured_evidence

            try:
                structured = structured_evidence(db, authz, scope, question, llm, kb_ids=[k.id for k in kb_rows])
                if structured:
                    result.trace["structured"] = [s.meta.get("sql") for s in structured]
            except CSVAIError as exc:
                result.trace["structured_error"] = exc.message
        items = structured + result.items
        budget = int(kb_cfg["max_context_tokens"])
        window = int(((llm.config.settings or {}).get("context_window") or 0) if llm and llm.config else 0)
        if window:
            # Leave room for instructions, history and the answer.
            budget = max(800, min(budget, window - llm.provider.max_tokens - 1200))
        sources = grounding.assemble_context(items, budget)
        yield {"type": "sources", "sources": [_citation_payload(s, False) for s in sources], "retrieval_ms": round(retrieval_ms, 1), "trace": result.trace}
        if not result.sufficient and not structured:
            yield from _finish(db, state, conv, assistant_message_id, grounding.REFUSAL, mode="refused", sources=sources, evidence=result.evidence, trace=result.trace, t_start=t_start, llm=None, principal=principal, verification=None, retrieval_ms=retrieval_ms)
            return
        if llm is None:
            text = grounding.extractive_answer(question, sources)
            notice = "\n\n_Extractive answer — no language model is configured for this organization." + (f" ({llm_error})" if llm_error else "") + "_"
            verification = grounding.verify(text, sources, policy)
            for piece in re.findall(r"\S+\s*", verification.answer + notice):
                state["content"] += piece
                yield {"type": "token", "text": piece}
            yield from _finish(db, state, conv, assistant_message_id, verification.answer + notice, mode="extractive", sources=sources, evidence=result.evidence, trace=result.trace, t_start=t_start, llm=None, principal=principal, verification=verification, retrieval_ms=retrieval_ms)
            return
        messages = grounding.build_messages(question, sources, history=history, org_name=org.name, policy=policy, assistant=agent.name if agent else "the assistant", instructions=agent.instructions if agent else "")
        yield {"type": "status", "stage": "generating", "model": llm.provider.model, "provider": llm.kind}
        t_gen = time.perf_counter()
        try:
            for piece in llm.provider.stream([LLMMessage(m["role"], m["content"]) for m in messages]):
                state["content"] += piece
                yield {"type": "token", "text": piece}
        except CSVAIError as exc:
            if state["content"].strip():
                raise
            # The model failed before producing output: answer extractively instead of failing.
            log.warning("generation failed, falling back to extractive answer: %s", exc.message)
            text = grounding.extractive_answer(question, sources)
            notice = f"\n\n_The language model failed ({exc.message[:120]}); this is an extractive answer from the sources._"
            verification = grounding.verify(text, sources, policy)
            for piece in re.findall(r"\S+\s*", verification.answer + notice):
                state["content"] += piece
                yield {"type": "token", "text": piece}
            result.trace["generation_error"] = exc.message[:300]
            yield from _finish(db, state, conv, assistant_message_id, verification.answer + notice, mode="extractive", sources=sources, evidence=result.evidence, trace=result.trace, t_start=t_start, llm=None, principal=principal, verification=verification, retrieval_ms=retrieval_ms)
            return
        gen_ms = (time.perf_counter() - t_gen) * 1000
        yield {"type": "status", "stage": "verifying"}
        verification = grounding.verify(state["content"], sources, policy)
        state["prompt_tokens"] = sum(estimate_tokens(m["content"]) for m in messages)
        refused = grounding.is_refusal(verification.answer)
        if policy == "strict" and not verification.cited_labels and not refused:
            # Citation enforcement: an uncited model answer is withheld; the verified source
            # passages are shown instead and the rejected text is kept for transparency.
            result.trace["rejected_model_answer"] = state["content"][:2000]
            text = grounding.extractive_answer(question, sources)
            notice = "\n\n_The model's answer cited no sources, so it was withheld under the strict grounding policy. These are the most relevant passages from your sources._"
            verification = grounding.verify(text, sources, policy)
            yield from _finish(db, state, conv, assistant_message_id, verification.answer + notice, mode="extractive", sources=sources, evidence=result.evidence, trace=result.trace, t_start=t_start, llm=llm, principal=principal, verification=verification, retrieval_ms=retrieval_ms, gen_ms=gen_ms)
            return
        if not verification.grounded and policy == "strict":
            verification.answer += "\n\n> ⚠ This answer could not be fully verified against the sources. Check the citations before relying on it."
        yield from _finish(db, state, conv, assistant_message_id, verification.answer, mode="llm", sources=sources, evidence=result.evidence, trace=result.trace, t_start=t_start, llm=llm, principal=principal, verification=verification, retrieval_ms=retrieval_ms, gen_ms=gen_ms)
    except GeneratorExit:
        state["status"] = "stopped"
        raise
    except Exception as exc:  # noqa: BLE001 — reported to the client, logged with request id
        log.exception("answer failed")
        state["status"] = "error"
        state["error"] = getattr(exc, "message", None) or "The answer could not be generated."
        yield {"type": "error", "code": getattr(exc, "code", "internal_error"), "message": state["error"]}
    finally:
        if not state["persisted"]:
            _persist_partial(db, assistant_message_id, state)
        try:
            db_ctx.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            log.exception("session close failed")


def _persist_partial(db: Session, message_id: uuid.UUID, state: dict[str, Any]) -> None:
    try:
        db.rollback()
        msg = db.get(Message, message_id)
        if msg is not None:
            msg.content = state["content"] or ("" if state["status"] != "error" else "")
            msg.status = "stopped" if state["status"] in ("stopped", "streaming") else "error"
            msg.meta = {**(msg.meta or {}), "error": state.get("error")}
            db.commit()
    except Exception:  # noqa: BLE001
        log.exception("could not persist partial message")


def _finish(
    db: Session,
    state: dict[str, Any],
    conv: Conversation,
    message_id: uuid.UUID,
    text: str,
    *,
    mode: str,
    sources: list[grounding.Source],
    evidence: float,
    trace: dict[str, Any],
    t_start: float,
    llm: ResolvedLLM | None,
    principal: Principal,
    verification: grounding.Verification | None,
    retrieval_ms: float | None = None,
    gen_ms: float | None = None,
) -> Iterator[dict[str, Any]]:
    cited = set(verification.cited_labels) if verification else set()
    conf = grounding.confidence(evidence, verification) if mode != "refused" else round(evidence, 3)
    msg = db.get(Message, message_id)
    assert msg is not None
    prompt_tokens, completion_tokens = (llm.provider.last_usage if llm and llm.provider.last_usage else (state.get("prompt_tokens", 0), estimate_tokens(text)))
    msg.content = text
    msg.status = "refused" if mode == "refused" else "complete"
    msg.provider = llm.kind if llm else ("extractive" if mode == "extractive" else None)
    msg.model = llm.provider.model if llm else None
    msg.tokens_in, msg.tokens_out = int(prompt_tokens or 0), int(completion_tokens or 0)
    msg.retrieval_ms = retrieval_ms
    msg.generation_ms = gen_ms
    msg.confidence = conf
    msg.grounded = verification.grounded if verification else mode == "refused"
    follow = grounding.follow_ups(trace.get("terms") and " ".join(trace["terms"]) or "", [s for s in sources if s.label in cited] or sources[:3])
    msg.meta = {
        "mode": mode, "trace": trace, "follow_ups": follow,
        "unsupported_sentences": verification.unsupported_sentences if verification else [],
        "invalid_citations": verification.invalid_labels if verification else [],
        "support_ratio": verification.support_ratio if verification else None,
    }
    for s in sources:
        it = s.item
        is_sql = it.source_type.startswith("sql_")
        db.add(Citation(
            org_id=conv.org_id, message_id=message_id, ordinal=int(s.label[1:]), source_type=_citation_type(it),
            kb_id=it.kb_id if it.kb_id.int else None, document_id=None if it.source_type == "sql_live" else it.document_id,
            chunk_id=None if is_sql else it.chunk_id, title=it.title[:500], locator=s.locator() | ({"sql": it.meta.get("sql"), **{k: it.meta[k] for k in ("connection_id", "dataset_id", "table") if k in it.meta}} if is_sql else {}),
            snippet=it.content[:2000], score=it.score, cited=s.label in cited,
        ))
    if conv.title == "New conversation":
        first = next((m for m in db.scalars(select(Message).where(Message.conversation_id == conv.id, Message.role == "user").order_by(Message.created_at).limit(1))), None)
        if first:
            conv.title = (first.content[:80] + ("…" if len(first.content) > 80 else "")).strip()
    total_ms = (time.perf_counter() - t_start) * 1000
    db.add(UsageRecord(
        org_id=conv.org_id, user_id=principal.user_id, api_key_id=principal.api_key_id, kind="chat", provider=msg.provider, model=msg.model,
        tokens_in=msg.tokens_in, tokens_out=msg.tokens_out, cost_usd=llm.cost(msg.tokens_in, msg.tokens_out) if llm else 0, latency_ms=total_ms,
        kb_id=sources[0].item.kb_id if sources else None,
    ))
    db.add(UsageRecord(org_id=conv.org_id, user_id=principal.user_id, api_key_id=principal.api_key_id, kind="retrieval", latency_ms=retrieval_ms, quantity=len(sources)))
    db.commit()
    state["persisted"] = True
    yield {
        "type": "done", "message_id": str(message_id), "content": text, "mode": mode, "confidence": conf, "grounded": msg.grounded,
        "citations": [_citation_payload(s, s.label in cited) for s in sources], "follow_ups": follow,
        "unsupported_sentences": msg.meta["unsupported_sentences"], "invalid_citations": msg.meta["invalid_citations"],
        "usage": {"prompt_tokens": msg.tokens_in, "completion_tokens": msg.tokens_out, "provider": msg.provider, "model": msg.model},
        "timings": {"retrieval_ms": round(retrieval_ms or 0, 1), "generation_ms": round(gen_ms or 0, 1), "total_ms": round(total_ms, 1)},
    }
