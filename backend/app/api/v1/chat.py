"""Chat (streaming RAG), conversations, sharing, export, feedback, citation previews."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select

from app.api.deps import OrgContext, requires, requires_any_scope
from app.api.schemas import Ok, page
from app.models import (
    Agent,
    Chunk,
    Citation,
    Conversation,
    ConversationShare,
    DatabaseConnection,
    Document,
    Message,
    MessageFeedback,
    TeamMember,
)
from app.rag.engine import AskOptions, answer_stream
from app.rag.index import SearchFilters
from app.rbac.authorizer import READ, AccessScope
from app.utils.errors import NotFound, PermissionDenied, ValidationFailed

router = APIRouter(tags=["chat"])


class FiltersIn(BaseModel):
    document_ids: list[uuid.UUID] | None = None
    file_types: list[str] | None = None
    tags: list[str] | None = None
    source_types: list[str] | None = None
    created_from: datetime | None = None
    created_to: datetime | None = None
    metadata: dict[str, Any] | None = None


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    conversation_id: uuid.UUID | None = None
    kb_ids: list[uuid.UUID] | None = None
    filters: FiltersIn = FiltersIn()
    retrieval: dict[str, Any] = {}
    provider_id: uuid.UUID | None = None
    model: str | None = None
    agent_id: uuid.UUID | None = None
    regenerate_message_id: uuid.UUID | None = None
    policy: str | None = Field(default=None, pattern="^(strict|balanced|open)$")


_ALLOWED_RETRIEVAL_OVERRIDES = {"mode", "top_k", "rerank", "mmr", "query_expansion", "multi_query", "compression", "fusion", "alpha", "min_score"}


def _prepare(body: AskIn, ctx: OrgContext) -> tuple[Conversation, Message, Message, AskOptions, str]:
    if not ctx.authz.has_anywhere("kb.chat"):
        ctx.require("kb.chat")
    ctx.require("conv.create")
    db = ctx.db
    question = body.question.strip()
    if body.agent_id:
        agent = db.get(Agent, body.agent_id)
        if agent is None or agent.org_id != ctx.org_id or ctx.authz.agent_level(agent) < READ:
            raise NotFound("Agent not found.")
    if body.conversation_id:
        conv = db.get(Conversation, body.conversation_id)
        if conv is None or conv.org_id != ctx.org_id or conv.user_id != ctx.user_id:
            raise NotFound("Conversation not found.")
        if body.kb_ids is not None:
            conv.kb_ids = body.kb_ids
    else:
        conv = Conversation(org_id=ctx.org_id, user_id=ctx.user_id, kb_ids=body.kb_ids or [], agent_id=body.agent_id)
        db.add(conv)
        db.flush()
    replaces = None
    if body.regenerate_message_id:
        old = db.get(Message, body.regenerate_message_id)
        if old is None or old.conversation_id != conv.id or old.role != "assistant":
            raise NotFound("Message to regenerate not found.")
        prev_user = db.scalar(select(Message).where(Message.conversation_id == conv.id, Message.role == "user", Message.created_at <= old.created_at).order_by(Message.created_at.desc()).limit(1))
        if prev_user is None:
            raise ValidationFailed("Nothing to regenerate.")
        question = prev_user.content
        user_msg = prev_user
        replaces = old.id
        old.status = "superseded" if old.status == "complete" else old.status
    else:
        user_msg = Message(org_id=ctx.org_id, conversation_id=conv.id, role="user", content=question)
        db.add(user_msg)
        db.flush()
    assistant = Message(org_id=ctx.org_id, conversation_id=conv.id, role="assistant", content="", status="streaming", replaces_id=replaces)
    db.add(assistant)
    conv.updated_at = datetime.now(UTC)
    db.commit()
    f = body.filters
    options = AskOptions(
        kb_ids=body.kb_ids,
        filters=SearchFilters(document_ids=f.document_ids, exts=f.file_types, tags=f.tags, source_types=f.source_types, created_from=f.created_from, created_to=f.created_to, metadata=f.metadata),
        retrieval={k: v for k, v in body.retrieval.items() if k in _ALLOWED_RETRIEVAL_OVERRIDES},
        provider_id=body.provider_id,
        model=body.model,
        agent_id=body.agent_id or conv.agent_id,
        policy=body.policy,
    )
    return conv, user_msg, assistant, options, question


def _sse(events: Iterator[dict[str, Any]], head: dict[str, Any]) -> Iterator[bytes]:
    yield f"event: meta\ndata: {json.dumps(head)}\n\n".encode()
    for ev in events:
        yield f"event: {ev['type']}\ndata: {json.dumps(ev, default=str)}\n\n".encode()


@router.post("/chat/stream")
def chat_stream(body: AskIn, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> StreamingResponse:
    conv, user_msg, assistant, options, question = _prepare(body, ctx)
    head = {"conversation_id": str(conv.id), "user_message_id": str(user_msg.id), "assistant_message_id": str(assistant.id)}
    events = answer_stream(principal=ctx.principal, org_id=ctx.org_id, conversation_id=conv.id, assistant_message_id=assistant.id, question=question, options=options)
    return StreamingResponse(_sse(events, head), media_type="text/event-stream", headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


@router.post("/chat/ask")
def chat_ask(body: AskIn, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> dict[str, Any]:
    """Non-streaming variant for API clients."""
    conv, user_msg, assistant, options, question = _prepare(body, ctx)
    result: dict[str, Any] = {"conversation_id": str(conv.id), "user_message_id": str(user_msg.id), "assistant_message_id": str(assistant.id)}
    for ev in answer_stream(principal=ctx.principal, org_id=ctx.org_id, conversation_id=conv.id, assistant_message_id=assistant.id, question=question, options=options):
        if ev["type"] == "sources":
            result["retrieval_trace"] = ev["trace"]
        elif ev["type"] == "done":
            result.update({k: v for k, v in ev.items() if k != "type"})
        elif ev["type"] == "error":
            result["error"] = {"code": ev["code"], "message": ev["message"]}
    return result


# ------------------------------------------------------------------------ conversations


def _teammates(ctx: OrgContext) -> set[uuid.UUID]:
    if not ctx.authz.team_ids:
        return set()
    return set(ctx.db.scalars(select(TeamMember.user_id).where(TeamMember.team_id.in_(ctx.authz.team_ids))))


def _shared_ids(ctx: OrgContext) -> set[uuid.UUID]:
    clauses = [(ConversationShare.subject_type == "user") & (ConversationShare.subject_id == ctx.user_id), ConversationShare.subject_type == "org"]
    if ctx.authz.team_ids:
        clauses.append((ConversationShare.subject_type == "team") & (ConversationShare.subject_id.in_(ctx.authz.team_ids)))
    return set(ctx.db.scalars(select(ConversationShare.conversation_id).where(ConversationShare.org_id == ctx.org_id, or_(*clauses))))


def can_view(ctx: OrgContext, conv: Conversation) -> bool:
    if conv.org_id != ctx.org_id:
        return False
    if conv.user_id == ctx.user_id:
        return ctx.authz.has("conv.view_own")
    if ctx.authz.has("conv.view_authorized"):
        return True
    if ctx.authz.has("conv.view_team") and conv.user_id in _teammates(ctx):
        return True
    return conv.id in _shared_ids(ctx)


def _get_conv(ctx: OrgContext, conv_id: uuid.UUID, *, owner: bool = False) -> Conversation:
    conv = ctx.db.get(Conversation, conv_id)
    if conv is None or not can_view(ctx, conv):
        raise NotFound("Conversation not found.")
    if owner and conv.user_id != ctx.user_id:
        raise PermissionDenied("Only the conversation owner can do this.")
    return conv


@router.get("/conversations")
def list_conversations(q: str | None = None, scope: str = "own", archived: bool = False, limit: int = 50, offset: int = 0, ctx: OrgContext = Depends(requires("conv.view_own"))) -> dict[str, Any]:
    stmt = select(Conversation).where(Conversation.org_id == ctx.org_id, Conversation.archived.is_(archived))
    if scope == "own":
        stmt = stmt.where(Conversation.user_id == ctx.user_id)
    elif scope == "shared":
        stmt = stmt.where(Conversation.id.in_(_shared_ids(ctx) or {uuid.UUID(int=0)}))
    elif scope == "team":
        ctx.require("conv.view_team")
        stmt = stmt.where(Conversation.user_id.in_(_teammates(ctx) or {ctx.user_id}))
    elif scope == "all":
        ctx.require("conv.view_authorized")
    else:
        raise ValidationFailed("scope must be own, shared, team or all.")
    if q:
        matching = select(Message.conversation_id).where(Message.org_id == ctx.org_id, Message.content.ilike(f"%{q}%"))
        stmt = stmt.where(or_(Conversation.title.ilike(f"%{q}%"), Conversation.id.in_(matching)))
    total = ctx.db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = list(ctx.db.scalars(stmt.order_by(Conversation.updated_at.desc()).limit(min(limit, 200)).offset(offset)))
    counts = dict(ctx.db.execute(select(Message.conversation_id, func.count()).where(Message.conversation_id.in_([c.id for c in rows])).group_by(Message.conversation_id)).all()) if rows else {}
    items = [{"id": str(c.id), "title": c.title, "user_id": str(c.user_id), "kb_ids": [str(k) for k in c.kb_ids], "agent_id": str(c.agent_id) if c.agent_id else None, "archived": c.archived, "message_count": counts.get(c.id, 0), "created_at": c.created_at.isoformat(), "updated_at": c.updated_at.isoformat(), "own": c.user_id == ctx.user_id} for c in rows]
    return page(items, total, limit, offset)


class ConversationIn(BaseModel):
    title: str | None = Field(default=None, max_length=300)
    kb_ids: list[uuid.UUID] = []
    agent_id: uuid.UUID | None = None


@router.post("/conversations", status_code=201)
def create_conversation(body: ConversationIn, ctx: OrgContext = Depends(requires("conv.create"))) -> dict[str, Any]:
    conv = Conversation(org_id=ctx.org_id, user_id=ctx.user_id, title=body.title or "New conversation", kb_ids=body.kb_ids, agent_id=body.agent_id)
    ctx.db.add(conv)
    ctx.db.commit()
    return {"id": str(conv.id), "title": conv.title}


def _visible_docs(ctx: OrgContext, doc_ids: set[uuid.UUID]) -> set[uuid.UUID]:
    """Documents (from past citations) the viewer may still read right now."""
    if not doc_ids:
        return set()
    scope: AccessScope = ctx.authz.access_scope()
    docs = ctx.db.scalars(select(Document).where(Document.id.in_(doc_ids)))
    visible = set()
    for d in docs:
        if d.kb_id not in scope.kb_levels or d.id in scope.denied_document_ids:
            continue
        if d.acl_mode == "restricted" and not (d.id in scope.restricted_allowed_document_ids or d.kb_id in scope.kb_manage_ids or d.owner_id == ctx.user_id):
            continue
        visible.add(d.id)
    return visible


def _citations_for(ctx: OrgContext, message_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[dict[str, Any]]]:
    rows = list(ctx.db.scalars(select(Citation).where(Citation.message_id.in_(message_ids)).order_by(Citation.ordinal))) if message_ids else []
    visible = _visible_docs(ctx, {c.document_id for c in rows if c.document_id})
    conn_ok: dict[str, bool] = {}
    out: dict[uuid.UUID, list[dict[str, Any]]] = {m: [] for m in message_ids}
    for c in rows:
        if c.source_type == "database":
            cid = (c.locator or {}).get("connection_id")
            if cid not in conn_ok:
                conn = ctx.db.get(DatabaseConnection, uuid.UUID(cid)) if cid else None
                conn_ok[cid] = bool(conn and conn.org_id == ctx.org_id and ctx.authz.connection_level(conn) >= READ and ctx.authz.has("db.query"))
            accessible = conn_ok[cid]
        else:
            accessible = c.document_id is not None and c.document_id in visible
        out[c.message_id].append({
            "id": str(c.id), "label": f"S{c.ordinal}", "ordinal": c.ordinal, "source_type": c.source_type, "cited": c.cited,
            "title": c.title if accessible else "[source no longer accessible]", "document_id": str(c.document_id) if c.document_id and accessible else None,
            "kb_id": str(c.kb_id) if c.kb_id and accessible else None, "locator": c.locator if accessible else {},
            "snippet": c.snippet if accessible else "", "score": c.score, "accessible": accessible,
        })
    return out


def _message_dict(m: Message, citations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": str(m.id), "role": m.role, "content": m.content, "status": m.status, "provider": m.provider, "model": m.model,
        "tokens_in": m.tokens_in, "tokens_out": m.tokens_out, "retrieval_ms": m.retrieval_ms, "generation_ms": m.generation_ms,
        "confidence": m.confidence, "grounded": m.grounded, "metadata": m.meta, "replaces_id": str(m.replaces_id) if m.replaces_id else None,
        "created_at": m.created_at.isoformat(), "citations": citations,
    }


@router.get("/conversations/{conv_id}")
def get_conversation(conv_id: uuid.UUID, ctx: OrgContext = Depends(requires("conv.view_own"))) -> dict[str, Any]:
    conv = _get_conv(ctx, conv_id)
    messages = list(ctx.db.scalars(select(Message).where(Message.conversation_id == conv.id).order_by(Message.created_at)))
    cites = _citations_for(ctx, [m.id for m in messages if m.role == "assistant"])
    shares = list(ctx.db.scalars(select(ConversationShare).where(ConversationShare.conversation_id == conv.id))) if conv.user_id == ctx.user_id else []
    return {
        "id": str(conv.id), "title": conv.title, "user_id": str(conv.user_id), "kb_ids": [str(k) for k in conv.kb_ids],
        "agent_id": str(conv.agent_id) if conv.agent_id else None, "archived": conv.archived, "own": conv.user_id == ctx.user_id,
        "created_at": conv.created_at.isoformat(), "updated_at": conv.updated_at.isoformat(),
        "messages": [_message_dict(m, cites.get(m.id, [])) for m in messages],
        "shares": [{"id": str(s.id), "subject_type": s.subject_type, "subject_id": str(s.subject_id) if s.subject_id else None} for s in shares],
    }


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    kb_ids: list[uuid.UUID] | None = None
    archived: bool | None = None


@router.patch("/conversations/{conv_id}", response_model=Ok)
def update_conversation(conv_id: uuid.UUID, body: ConversationUpdate, ctx: OrgContext = Depends(requires("conv.view_own"))) -> Ok:
    conv = _get_conv(ctx, conv_id, owner=True)
    if body.title is not None:
        conv.title = body.title.strip()
    if body.kb_ids is not None:
        conv.kb_ids = body.kb_ids
    if body.archived is not None:
        conv.archived = body.archived
    ctx.db.commit()
    return Ok()


@router.delete("/conversations/{conv_id}", response_model=Ok)
def delete_conversation(conv_id: uuid.UUID, ctx: OrgContext = Depends(requires("conv.delete"))) -> Ok:
    conv = _get_conv(ctx, conv_id)
    if conv.user_id != ctx.user_id and not ctx.authz.has("conv.view_authorized"):
        raise PermissionDenied("Only the owner can delete this conversation.")
    ctx.db.delete(conv)
    ctx.audit("conversation.delete", resource_type="conversation", resource_id=conv_id)
    ctx.db.commit()
    return Ok()


class ShareIn(BaseModel):
    subject_type: str = Field(pattern="^(user|team|org)$")
    subject_id: uuid.UUID | None = None


@router.post("/conversations/{conv_id}/shares", status_code=201)
def share_conversation(conv_id: uuid.UUID, body: ShareIn, ctx: OrgContext = Depends(requires("conv.share"))) -> dict[str, Any]:
    conv = _get_conv(ctx, conv_id, owner=True)
    if body.subject_type != "org" and body.subject_id is None:
        raise ValidationFailed("subject_id is required.")
    share = ConversationShare(org_id=ctx.org_id, conversation_id=conv.id, subject_type=body.subject_type, subject_id=body.subject_id if body.subject_type != "org" else None, created_by=ctx.user_id)
    ctx.db.add(share)
    ctx.audit("conversation.share", resource_type="conversation", resource_id=conv.id, meta=body.model_dump(mode="json"))
    ctx.db.commit()
    return {"id": str(share.id), "note": "Recipients see the conversation, but cited sources stay hidden unless they can access them."}


@router.delete("/conversations/{conv_id}/shares/{share_id}", response_model=Ok)
def unshare_conversation(conv_id: uuid.UUID, share_id: uuid.UUID, ctx: OrgContext = Depends(requires("conv.share"))) -> Ok:
    _get_conv(ctx, conv_id, owner=True)
    share = ctx.db.get(ConversationShare, share_id)
    if share and share.conversation_id == conv_id:
        ctx.db.delete(share)
        ctx.db.commit()
    return Ok()


@router.get("/conversations/{conv_id}/export")
def export_conversation(conv_id: uuid.UUID, format: str = "md", ctx: OrgContext = Depends(requires("conv.export"))) -> Any:
    data = get_conversation(conv_id, ctx)
    ctx.audit("conversation.export", resource_type="conversation", resource_id=conv_id, meta={"format": format})
    ctx.db.commit()
    if format == "json":
        return data
    lines = [f"# {data['title']}", "", f"_Exported {datetime.now(UTC).isoformat(timespec='seconds')}_", ""]
    for m in data["messages"]:
        if m["status"] == "superseded":
            continue
        lines += [f"## {'You' if m['role'] == 'user' else 'Assistant'}", "", m["content"], ""]
        cited = [c for c in m["citations"] if c["cited"]]
        if cited:
            lines.append("**Sources**")
            for c in cited:
                loc = ", ".join(f"{k} {v}" for k, v in c["locator"].items() if k in ("page", "sheet", "row_start", "slide", "url"))
                lines.append(f"- [{c['label']}] {c['title']}" + (f" ({loc})" if loc else ""))
            lines.append("")
    return PlainTextResponse("\n".join(lines), media_type="text/markdown", headers={"Content-Disposition": f'attachment; filename="conversation-{conv_id}.md"'})


class FeedbackIn(BaseModel):
    rating: int = Field(ge=-1, le=1)
    comment: str = Field(default="", max_length=2000)


@router.post("/messages/{message_id}/feedback", response_model=Ok)
def message_feedback(message_id: uuid.UUID, body: FeedbackIn, ctx: OrgContext = Depends(requires("conv.view_own"))) -> Ok:
    msg = ctx.db.get(Message, message_id)
    if msg is None or msg.org_id != ctx.org_id:
        raise NotFound("Message not found.")
    _get_conv(ctx, msg.conversation_id)
    existing = ctx.db.scalar(select(MessageFeedback).where(MessageFeedback.message_id == message_id, MessageFeedback.user_id == ctx.user_id))
    if existing:
        existing.rating, existing.comment = body.rating, body.comment
    else:
        ctx.db.add(MessageFeedback(org_id=ctx.org_id, message_id=message_id, user_id=ctx.user_id, rating=body.rating, comment=body.comment))
    ctx.db.commit()
    return Ok()


@router.get("/citations/{citation_id}/preview")
def citation_preview(citation_id: uuid.UUID, ctx: OrgContext = Depends(requires_any_scope("kb.view"))) -> dict[str, Any]:
    """Source content around a cited chunk — re-authorized at view time."""
    c = ctx.db.get(Citation, citation_id)
    if c is None or c.org_id != ctx.org_id:
        raise NotFound("Citation not found.")
    msg = ctx.db.get(Message, c.message_id)
    assert msg is not None
    _get_conv(ctx, msg.conversation_id)
    if c.document_id is None or c.document_id not in _visible_docs(ctx, {c.document_id}):
        raise PermissionDenied("You no longer have access to this source.")
    chunk = ctx.db.get(Chunk, c.chunk_id) if c.chunk_id else None
    neighbours: list[dict[str, Any]] = []
    if chunk is not None:
        for n in ctx.db.scalars(select(Chunk).where(Chunk.document_id == chunk.document_id, Chunk.is_retrievable.is_(True), Chunk.ordinal.between(chunk.ordinal - 1, chunk.ordinal + 1)).order_by(Chunk.ordinal)):
            neighbours.append({"id": str(n.id), "ordinal": n.ordinal, "content": n.content, "metadata": n.meta, "is_cited": n.id == chunk.id})
    doc = ctx.db.get(Document, c.document_id)
    return {"citation": {"title": c.title, "locator": c.locator, "snippet": c.snippet}, "document": {"id": str(doc.id), "title": doc.title, "filename": doc.filename, "ext": doc.ext, "source_url": doc.source_url} if doc else None, "context": neighbours}
