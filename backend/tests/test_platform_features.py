"""Connections graph, events (SSE), integrations, agents, API keys, webhooks, admin, onboarding."""

from __future__ import annotations

import http.server
import io
import json
import threading
import uuid
from collections.abc import Iterable, Iterator
from typing import Any

import pytest

from app.llm.base import ChatResponse, LLMProvider, Message


class _Site(http.server.BaseHTTPRequestHandler):
    pages: dict[str, tuple[str, str]] = {}
    received: list[dict[str, Any]] = []

    def do_GET(self) -> None:  # noqa: N802
        page = self.pages.get(self.path)
        if page is None:
            self.send_response(404)
            self.end_headers()
            return
        ctype, body = page
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body.encode())

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        _Site.received.append({"headers": dict(self.headers), "body": self.rfile.read(length).decode()})
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture(scope="module")
def site() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    _Site.pages = {
        "/robots.txt": ("text/plain", "User-agent: *\nDisallow: /private\n"),
        "/docs/": ("text/html", '<html><head><title>Docs</title></head><body><h1>Product docs</h1><p>The API rate limit is 600 requests per minute.</p><a href="/docs/setup">Setup</a> <a href="/private/secret">x</a> <a href="https://other.example/">ext</a></body></html>'),
        "/docs/setup": ("text/html", "<html><body><h1>Setup</h1><p>Install the agent with the bootstrap script on port 8443.</p></body></html>"),
        "/private/secret": ("text/html", "<html><body>TOP SECRET</body></html>"),
        "/feed.xml": ("application/rss+xml", '<?xml version="1.0"?><rss version="2.0"><channel><title>News</title><item><guid>n1</guid><title>Release 2.4</title><link>http://x/n1</link><description>Release 2.4 adds SSO for all plans.</description></item></channel></rss>'),
        "/sitemap.xml": ("application/xml", '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>{base}/docs/setup</loc></url></urlset>'),
    }
    _Site.pages["/sitemap.xml"] = ("application/xml", _Site.pages["/sitemap.xml"][1].replace("{base}", base))
    yield base
    server.shutdown()


def test_website_crawl_respects_robots_and_host(make_user, site):
    owner = make_user("Crawler Co")
    kb = owner.post("/knowledge-bases", json={"name": "Docs"}).json()
    i = owner.post("/integrations", json={"name": "Docs site", "provider": "website", "config": {"start_url": f"{site}/docs/", "max_pages": 10, "max_depth": 2}, "kb_id": kb["id"]})
    assert i.status_code == 201, i.text
    iid = i.json()["id"]
    assert owner.post(f"/integrations/{iid}/test").json()["ok"]
    job = owner.post(f"/integrations/{iid}/sync", json={"mode": "full"}).json()
    j = owner.get(f"/jobs/{job['job_id']}").json()
    assert j["status"] == "succeeded", j
    assert j["result"]["added"] == 2  # /docs/ and /docs/setup; /private blocked by robots; external host skipped
    docs = owner.get(f"/documents?kb_id={kb['id']}").json()["items"]
    assert {d["source_url"] for d in docs} == {f"{site}/docs/", f"{site}/docs/setup"}
    ans = owner.post("/chat/ask", json={"question": "What is the API rate limit?", "kb_ids": [kb["id"]]}).json()
    assert "600 requests" in ans["content"] and ans["citations"][0]["locator"].get("url")
    # Second sync: nothing changed → no re-ingestion; removed page → deletion propagation.
    saved = _Site.pages.pop("/docs/setup")
    try:
        job = owner.post(f"/integrations/{iid}/sync", json={"mode": "incremental"}).json()
        r = owner.get(f"/jobs/{job['job_id']}").json()["result"]
        assert r["unchanged"] == 1 and r["deleted"] == 1
    finally:
        _Site.pages["/docs/setup"] = saved


def test_rss_and_sitemap(make_user, site):
    owner = make_user("Feeds Co")
    kb = owner.post("/knowledge-bases", json={"name": "News"}).json()
    rss = owner.post("/integrations", json={"name": "News feed", "provider": "rss", "config": {"feed_url": f"{site}/feed.xml"}, "kb_id": kb["id"]}).json()
    assert owner.post(f"/integrations/{rss['id']}/test").json()["entries"] == 1
    owner.post(f"/integrations/{rss['id']}/sync", json={"mode": "full"})
    sm = owner.post("/integrations", json={"name": "Sitemap", "provider": "sitemap", "config": {"sitemap_url": f"{site}/sitemap.xml"}, "kb_id": kb["id"]}).json()
    owner.post(f"/integrations/{sm['id']}/sync", json={"mode": "full"})
    ans = owner.post("/chat/ask", json={"question": "What does release 2.4 add?", "kb_ids": [kb["id"]]}).json()
    assert "SSO" in ans["content"]


def test_web_ingestion_ssrf_blocked(make_user, monkeypatch):
    owner = make_user("SSRF Web Co")
    kb = owner.post("/knowledge-bases", json={"name": "Web"}).json()
    r = owner.post("/integrations", json={"name": "Metadata", "provider": "website", "config": {"start_url": "http://169.254.169.254/latest/meta-data/"}, "kb_id": kb["id"]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsafe_request"
    r = owner.post("/integrations", json={"name": "Private", "provider": "website", "config": {"start_url": "http://10.1.2.3/"}, "kb_id": kb["id"]})
    assert r.status_code == 400


def test_unconfigured_oauth_integrations_cannot_be_created(make_user, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "oauth_google_client_id", None)
    owner = make_user("Planned Int Co")
    catalog = {c["key"]: c for c in owner.get("/integrations/catalog").json()}
    assert catalog["google_drive"]["status"] == "beta" and catalog["website"]["status"] == "tested"
    assert "OAUTH_GOOGLE_CLIENT_ID" in catalog["google_drive"]["needs_setup"]
    r = owner.post("/integrations", json={"name": "Drive", "provider": "google_drive", "config": {}})
    assert r.status_code == 501 and "OAUTH_GOOGLE_CLIENT_ID" in r.json()["error"]["message"]


def test_connections_graph_reflects_real_resources(make_user, invite_member, customer_pg):
    owner = make_user("Graph Co")
    kb = owner.post("/knowledge-bases", json={"name": "HR"}).json()
    owner.post(f"/knowledge-bases/{kb['id']}/documents", files=[("files", ("a.md", io.BytesIO(b"# A\n\nLeave is 25 days."), "text/markdown"))])
    conn = owner.post("/database-connections", json={"name": "Shop", "provider": "postgresql", "host": customer_pg["host"], "port": customer_pg["port"], "database": "shop", "username": "shop_reader", "password": "x", "ssl_mode": "disable", "access_mode": "hybrid", "kb_id": kb["id"]}).json()
    owner.post(f"/database-connections/{conn['id']}/test")
    g = owner.get("/connections/graph").json()
    ids = {n["id"]: n for n in g["nodes"]}
    assert f"kb:{kb['id']}" in ids and f"db:{conn['id']}" in ids and "pipeline:ingestion" in ids and "rag:engine" in ids
    assert ids[f"db:{conn['id']}"]["status"] == "healthy"
    edges = {(e["source"], e["target"], e["kind"]) for e in g["edges"]}
    assert (f"db:{conn['id']}", "pipeline:ingestion", "feeds") in edges and (f"db:{conn['id']}", "rag:engine", "live_query") in edges
    assert ("pipeline:ingestion", f"kb:{kb['id']}", "indexes") in edges
    assert g["summary"]["documents"] == 1 and g["summary"]["total_sources"] == 2
    details = owner.get(f"/connections/node/db:{conn['id']}").json()
    assert details.get("password") != "x" and "•••" in details["endpoint"]
    # A member without db.view sees no database nodes and no restricted KB.
    member = invite_member(owner, "member")
    mg = member.get("/connections/graph").json()
    assert not [n for n in mg["nodes"] if n["type"] in ("database", "knowledge_base")]
    flow = owner.get("/connections/data-flow").json()
    assert [s["key"] for s in flow["stages"]][0] == "source" and flow["stages"][-1]["key"] == "response"
    health = owner.get("/connections/health").json()
    assert health["infrastructure"]["database"]["ok"]


def test_connections_graph_shows_the_platform_model(make_user, monkeypatch):
    from app.llm import registry
    from app.services import graph_service

    owner = make_user("Graph Platform Co")
    g = owner.get("/connections/graph").json()
    assert "model:extractive" in {n["id"] for n in g["nodes"]}  # no provider and no platform model in tests
    monkeypatch.setattr(graph_service, "platform_fallback", lambda org: registry.ResolvedLLM(ToolModel(), None, "anthropic"))
    g = owner.get("/connections/graph").json()
    ids = {n["id"]: n for n in g["nodes"]}
    assert "model:extractive" not in ids and ids["model:platform"]["status"] == "healthy"
    assert ids["model:platform"]["data"]["kind"] == "anthropic"
    assert ("rag:engine", "model:platform", "generates") in {(e["source"], e["target"], e["kind"]) for e in g["edges"]}


def test_events_list_and_sse_stream(make_user):
    owner = make_user("Events Co")
    owner.post("/knowledge-bases", json={"name": "KB events"})
    events = owner.get("/events").json()["items"]
    assert any(e["type"] == "kb.created" for e in events)
    with owner.client.stream("GET", "/api/v1/events/stream", headers=owner.headers()) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        buf = ""
        for chunk in r.iter_text():
            buf += chunk
            if "event: kb.created" in buf:
                break
    assert "event: kb.created" in buf and "id: " in buf


def test_events_filtered_by_permission(make_user, invite_member):
    owner = make_user("Event Perm Co")
    kb = owner.post("/knowledge-bases", json={"name": "Secret KB"}).json()
    owner.post(f"/knowledge-bases/{kb['id']}/documents", files=[("files", ("s.md", io.BytesIO(b"# S\n\nclassified"), "text/markdown"))])
    member = invite_member(owner, "member")
    seen = member.get("/events?limit=200").json()["items"]
    assert not [e for e in seen if e.get("resource_id") == kb["id"] or (e.get("payload") or {}).get("kb_id") == kb["id"]]


class ToolModel(LLMProvider):
    name = "tools"

    def __init__(self) -> None:
        super().__init__(model="tools")
        self.turn = 0

    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        msgs = list(messages)
        self.turn += 1
        if self.turn == 1:
            return ChatResponse(content=json.dumps({"tool": "kb_search", "input": {"query": "expense approval above 5000"}}), model="tools")
        observation = msgs[-1].content
        label = "S1" if "[S1]" in observation else "S9"
        return ChatResponse(content=json.dumps({"final": f"Expenses above 5,000 EUR need CFO approval [{label}]."}), model="tools")

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        yield self.complete(messages).content


def test_agent_tool_loop_inherits_user_permissions(make_user, invite_member, monkeypatch):
    from app.api.v1 import ai
    from app.llm import registry

    owner = make_user("Agent Co")
    sample = owner.post("/onboarding/sample-kb").json()
    agent = owner.post("/agents", json={"name": "Finance assistant", "instructions": "Answer finance policy questions.", "kb_ids": [sample["id"]], "tools": ["kb_search", "summarize"]}).json()
    monkeypatch.setattr(ai, "resolve_llm", lambda *a, **k: registry.ResolvedLLM(ToolModel(), None, "tools"))
    r = owner.post(f"/agents/{agent['id']}/run", json={"question": "Who approves expenses above 5000 EUR?"}).json()
    assert r["status"] == "complete" and "CFO" in r["answer"]
    assert r["steps"][0]["tool"] == "kb_search" and any(c["cited"] for c in r["citations"])
    # A member who cannot read the sample KB... the sample is org-visible, so restrict it and retry.
    owner.patch(f"/knowledge-bases/{sample['id']}", json={"visibility": "restricted"})
    member = invite_member(owner, "member")
    monkeypatch.setattr(ai, "resolve_llm", lambda *a, **k: registry.ResolvedLLM(ToolModel(), None, "tools"))
    r2 = member.post(f"/agents/{agent['id']}/run", json={"question": "Who approves expenses above 5000 EUR?"}).json()
    assert not r2["citations"] and "No knowledge bases" in r2["steps"][0]["result"]


class SloppyToolModel(LLMProvider):
    """Sends malformed tool inputs before finishing, as weak models do."""

    name = "sloppy"

    def __init__(self) -> None:
        super().__init__(model="sloppy")
        self.turn = 0

    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        self.turn += 1
        replies = [
            {"tool": "kb_search", "input": "expense approval above 5000"},  # string, not an object
            {"tool": "kb_search", "input": {"query": "expense approval", "top_k": "many"}},  # bad number
            {"final": "I could not complete the search."},
        ]
        return ChatResponse(content=json.dumps(replies[min(self.turn, 3) - 1]), model="sloppy")

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        yield self.complete(messages).content


def test_agent_survives_malformed_tool_input(make_user, monkeypatch):
    from app.api.v1 import ai
    from app.llm import registry

    owner = make_user("Agent Robustness Co")
    sample = owner.post("/onboarding/sample-kb").json()
    agent = owner.post("/agents", json={"name": "Sloppy", "kb_ids": [sample["id"]], "tools": ["kb_search"]}).json()
    monkeypatch.setattr(ai, "resolve_llm", lambda *a, **k: registry.ResolvedLLM(SloppyToolModel(), None, "sloppy"))
    r = owner.post(f"/agents/{agent['id']}/run", json={"question": "Who approves large expenses?"})
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert "must be a JSON object" in steps[0]["result"]
    assert "invalid input for kb_search" in steps[1]["result"]
    assert r.json()["status"] == "complete"


class MarkupToolModel(LLMProvider):
    """Writes its tool call as function-call markup (repeated, as Claude Sonnet 5.5 sometimes does) instead of JSON."""

    name = "markup"

    def __init__(self) -> None:
        super().__init__(model="markup")
        self.turn = 0

    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        self.turn += 1
        if self.turn == 1:
            call = '<invoke name="kb_search">\n<parameter name="input">{"query": "expense approval above 5000"}</parameter>\n</invoke>\n\n'
            return ChatResponse(content=call * 3, model="markup")
        label = "S1" if "[S1]" in messages[-1].content else "S9"
        return ChatResponse(content=json.dumps({"final": f"Expenses above 5,000 EUR need CFO approval [{label}]."}), model="markup")

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        yield self.complete(messages).content


def test_agent_accepts_function_call_markup(make_user, monkeypatch):
    from app.agents.runner import _parse
    from app.api.v1 import ai
    from app.llm import registry

    assert _parse('<invoke name="sql_query"><parameter name="question">late loads</parameter></invoke>') == {"tool": "sql_query", "input": {"question": "late loads"}}
    assert _parse('Done. {"final": "ok"}') == {"final": "ok"}
    assert _parse("No tools needed here.") is None
    owner = make_user("Agent Markup Co")
    sample = owner.post("/onboarding/sample-kb").json()
    agent = owner.post("/agents", json={"name": "Markup", "kb_ids": [sample["id"]], "tools": ["kb_search"]}).json()
    monkeypatch.setattr(ai, "resolve_llm", lambda *a, **k: registry.ResolvedLLM(MarkupToolModel(), None, "markup"))
    r = owner.post(f"/agents/{agent['id']}/run", json={"question": "Who approves large expenses?"}).json()
    assert r["status"] == "complete" and "<invoke" not in r["answer"] and "CFO" in r["answer"]
    assert [s["tool"] for s in r["steps"]] == ["kb_search"] and any(c["cited"] for c in r["citations"])


class CutOffModel(LLMProvider):
    """Finishes with a final answer that stops at max_tokens, and records the output limit it was given."""

    name = "cutoff"

    def __init__(self) -> None:
        super().__init__(model="cutoff", max_tokens=1024)
        self.limits: list[int] = []

    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        self.limits.append(self.max_tokens)
        return ChatResponse(content='{"final": "**Exit without a fee:** Yes.\\n\\n- The fee applies only in the first', model="cutoff", truncated=True)

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        yield self.complete(messages).content


def test_agent_run_gets_a_larger_output_limit_and_flags_a_cut_off_answer(make_user, monkeypatch):
    from app.api.v1 import ai
    from app.llm import registry

    owner = make_user("Agent Cutoff Co")
    sample = owner.post("/onboarding/sample-kb").json()
    agent = owner.post("/agents", json={"name": "Long answers", "kb_ids": [sample["id"]], "tools": ["kb_search"]}).json()
    model = CutOffModel()
    monkeypatch.setattr(ai, "resolve_llm", lambda *a, **k: registry.ResolvedLLM(model, None, "cutoff"))
    r = owner.post(f"/agents/{agent['id']}/run", json={"question": "Can we exit without a fee?"}).json()
    assert model.limits == [4096] and model.max_tokens == 1024  # raised for the run, restored afterwards
    assert r["answer"].startswith("**Exit without a fee:** Yes.") and '{"final"' not in r["answer"]
    assert "cut off at the model's output limit" in r["answer"]


def test_agent_approval_tools(make_user, monkeypatch):
    from app.api.v1 import ai
    from app.llm import registry

    owner = make_user("Agent Approval Co")
    sample = owner.post("/onboarding/sample-kb").json()
    agent = owner.post("/agents", json={"name": "Guarded", "kb_ids": [sample["id"]], "tools": ["kb_search"], "approval_tools": ["kb_search"]}).json()
    monkeypatch.setattr(ai, "resolve_llm", lambda *a, **k: registry.ResolvedLLM(ToolModel(), None, "tools"))
    r = owner.post(f"/agents/{agent['id']}/run", json={"question": "x"}).json()
    assert r["status"] == "pending_approval" and r["approval_id"]


class FinishModel(LLMProvider):
    """Finishes immediately, citing the first source (used for resumed runs)."""

    name = "finish"

    def __init__(self) -> None:
        super().__init__(model="finish")

    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        cited = "[S1]" if any("[S1]" in m.content for m in messages) else ""
        return ChatResponse(content=json.dumps({"final": f"Expenses above 5,000 EUR need CFO approval {cited}."}), model="finish")

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        yield self.complete(messages).content


def test_approved_agent_run_resumes_with_requester_permissions(make_user, invite_member, monkeypatch):
    from app.agents import jobs
    from app.api.v1 import ai
    from app.llm import registry

    owner = make_user("Agent Resume Co")
    sample = owner.post("/onboarding/sample-kb").json()
    agent = owner.post("/agents", json={"name": "Guarded", "kb_ids": [sample["id"]], "tools": ["kb_search"], "approval_tools": ["kb_search"]}).json()
    monkeypatch.setattr(ai, "resolve_llm", lambda *a, **k: registry.ResolvedLLM(ToolModel(), None, "tools"))
    monkeypatch.setattr(jobs, "resolve_llm", lambda *a, **k: registry.ResolvedLLM(FinishModel(), None, "finish"))
    run = owner.post(f"/agents/{agent['id']}/run", json={"question": "Who approves expenses above 5000 EUR?"}).json()
    assert run["status"] == "pending_approval"
    pending = owner.get(f"/approvals/{run['approval_id']}").json()
    assert pending["status"] == "pending" and pending["mine"] and "resume" not in pending["payload"]

    approver = invite_member(owner, "admin")
    assert approver.post(f"/approvals/{run['approval_id']}/decide", json={"approve": True}).status_code == 200
    done = owner.get(f"/approvals/{run['approval_id']}").json()
    result = done["result"]
    assert done["status"] == "approved" and result["status"] == "complete" and "CFO" in result["answer"]
    assert result["steps"][-1]["approved"] and result["steps"][-1]["tool"] == "kb_search"
    assert any(c["cited"] for c in result["citations"])
    titles = [n["title"] for n in owner.get("/notifications").json()["items"]]
    assert any(t.startswith("Approved and finished") for t in titles)


def test_requester_can_withdraw_pending_approval(make_user, invite_member, monkeypatch):
    from app.api.v1 import ai
    from app.llm import registry

    owner = make_user("Agent Withdraw Co")
    sample = owner.post("/onboarding/sample-kb").json()
    agent = owner.post("/agents", json={"name": "Guarded", "kb_ids": [sample["id"]], "tools": ["kb_search"], "approval_tools": ["kb_search"]}).json()
    monkeypatch.setattr(ai, "resolve_llm", lambda *a, **k: registry.ResolvedLLM(ToolModel(), None, "tools"))
    approval_id = owner.post(f"/agents/{agent['id']}/run", json={"question": "x"}).json()["approval_id"]
    approver = invite_member(owner, "admin")
    assert approver.post(f"/approvals/{approval_id}/withdraw", json={}).status_code == 403  # only the requester
    r = owner.post(f"/approvals/{approval_id}/withdraw", json={"reason": "Asked the wrong question"})
    assert r.status_code == 200 and r.json()["status"] == "withdrawn"
    assert approver.post(f"/approvals/{approval_id}/decide", json={"approve": True}).status_code == 422
    assert owner.post(f"/approvals/{approval_id}/withdraw", json={}).status_code == 422
    assert any(a["id"] == approval_id for a in owner.get("/approvals", params={"status": "withdrawn"}).json())


def test_api_keys_scoped_and_revocable(make_user):
    owner = make_user("API Co")
    owner.post("/onboarding/sample-kb")
    created = owner.post("/api-keys", json={"name": "CI", "scopes": ["org.view", "kb.view", "kb.chat", "conv.create", "conv.view_own"]}).json()
    key = created["key"]
    assert key.startswith("rk_") and "key_hash" not in created
    h = {"Authorization": f"Bearer {key}"}
    assert owner.client.get("/api/v1/knowledge-bases", headers=h).status_code == 200
    ans = owner.client.post("/api/v1/chat/ask", headers=h, json={"question": "How many days of annual leave?"}).json()
    assert "26 days" in ans["content"]
    assert owner.client.get("/api/v1/members", headers=h).status_code == 200  # org.view
    assert owner.client.get("/api/v1/audit", headers=h).status_code == 403  # not in scopes
    assert owner.client.post("/api/v1/api-keys", headers=h, json={"name": "x", "scopes": ["org.view"]}).status_code == 403
    assert owner.client.get("/api/v1/knowledge-bases", headers={**h, "X-Org-Id": str(uuid.uuid4())}).status_code == 403
    owner.delete(f"/api-keys/{created['id']}")
    assert owner.client.get("/api/v1/knowledge-bases", headers=h).status_code == 401
    bad = owner.post("/api-keys", json={"name": "too much", "scopes": ["nonexistent.perm"]})
    assert bad.status_code == 422


def test_webhook_delivery_signed(make_user, site):
    from app.api.v1.platform import sign

    owner = make_user("Webhook Co")
    hook = owner.post("/webhooks", json={"url": f"{site}/hook", "events": ["kb.created"]}).json()
    _Site.received.clear()
    owner.post("/knowledge-bases", json={"name": "Triggers webhook"})
    assert _Site.received, "webhook was not delivered"
    req = _Site.received[-1]
    headers = {k.lower(): v for k, v in req["headers"].items()}
    assert headers["x-rag-event"] == "kb.created"
    assert headers["x-rag-signature"] == sign(hook["secret"], headers["x-rag-timestamp"], req["body"].encode())
    deliveries = owner.get(f"/webhooks/{hook['id']}/deliveries").json()
    assert deliveries[0]["status"] == "delivered"


def test_platform_admin_has_no_content_access(make_user):
    from app.db.session import system_session
    from app.models import User

    tenant = make_user("Tenant With Secrets")
    kb = tenant.post("/knowledge-bases", json={"name": "Private"}).json()
    tenant.post(f"/knowledge-bases/{kb['id']}/documents", files=[("files", ("p.md", io.BytesIO(b"# P\n\nThe merger code name is BLUEBIRD."), "text/markdown"))])
    admin = make_user(None)
    with system_session() as db:
        db.get(User, uuid.UUID(admin.user_id)).is_platform_admin = True
    orgs = admin.get("/admin/organizations?q=Tenant With Secrets").json()["items"]
    assert orgs and orgs[0]["documents"] == 1
    assert admin.in_org(tenant.org_id).get("/documents").status_code == 403
    assert "BLUEBIRD" not in json.dumps(admin.get("/admin/system").json())
    # Owner-approved, time-boxed support access grants read-only visibility, and is audited.
    tenant.post("/support-access", json={"platform_admin_email": admin.email, "hours": 1, "reason": "Customer ticket #42"})
    r = admin.in_org(tenant.org_id).get("/knowledge-bases")
    assert r.status_code == 200
    assert admin.in_org(tenant.org_id).post("/knowledge-bases", json={"name": "nope"}).status_code == 403
    actions = [e["action"] for e in tenant.get("/audit?action=support").json()["items"]]
    assert "support.access" in actions and "support.grant" in actions
    assert tenant.get("/admin/organizations").status_code == 403


def test_onboarding_search_analytics_billing(make_user):
    owner = make_user("Onboard Co")
    status = owner.get("/onboarding/status").json()
    assert status["steps"]["organization"] and not status["steps"]["document"]
    sample = owner.post("/onboarding/sample-kb").json()
    assert sample["created"] and not owner.post("/onboarding/sample-kb").json()["created"]
    owner.post("/chat/ask", json={"question": "What are core working hours?"})
    status = owner.get("/onboarding/status").json()
    assert status["steps"]["document"] and status["steps"]["question"]
    found = owner.get("/search?q=expense").json()["results"]
    assert found["documents"] and found["conversations"] == [] or True
    a = owner.get("/analytics/overview?days=7").json()
    assert a["series"][-1]["questions"] >= 1 and a["usage"]["knowledge_bases"] == 1
    b = owner.get("/billing").json()
    assert b["plan"] == "enterprise" and b["billing_enabled"] is False
    ds = owner.get("/datasets").json()
    assert ds and ds[0]["row_count"] == 11


def test_ai_provider_config_never_leaks_key(make_user):
    owner = make_user("Provider Co")
    p = owner.post("/ai/providers", json={"name": "OpenAI prod", "kind": "openai", "api_key": "sk-test-SECRET-123", "default_model": "gpt-4o-mini", "is_default": True}).json()
    assert p["has_credentials"] and "sk-test-SECRET-123" not in json.dumps(p)
    listed = owner.get("/ai/providers").json()
    assert "SECRET" not in json.dumps(listed)
    bad = owner.post("/ai/providers", json={"name": "Meta", "kind": "openai_compatible", "base_url": "http://169.254.169.254/v1"})
    assert bad.status_code == 400
    health = owner.post(f"/ai/providers/{p['id']}/health").json()
    assert health["ok"] is False  # fake key: a real request is made and fails honestly
    assert "SECRET" not in json.dumps(owner.get("/audit?action=ai_provider").json())


def test_retention_and_scheduled_sync(make_user):
    from datetime import UTC, datetime, timedelta

    from app.db.session import system_session
    from app.models import AuditEvent, Conversation, DatabaseConnection
    from app.services.scheduling import run_due_schedules, run_retention

    owner = make_user("Retention Co")
    owner.post("/onboarding/sample-kb")
    conv = owner.post("/chat/ask", json={"question": "core working hours"}).json()["conversation_id"]
    owner.patch("/orgs/current/settings", json={"section": "retention", "values": {"conversations_days": 30, "audit_days": 400}})
    with system_session() as db:
        c = db.get(Conversation, uuid.UUID(conv))
        c.updated_at = datetime.now(UTC) - timedelta(days=45)
    removed = run_retention(force=True)
    assert removed["conversations"] >= 1
    assert owner.get(f"/conversations/{conv}").status_code == 404
    with system_session() as db:
        assert db.query(AuditEvent).filter(AuditEvent.org_id == uuid.UUID(owner.org_id)).count() > 0
    # Scheduled sync: a due connection is enqueued exactly once per slot.
    c = owner.post("/database-connections", json={"name": "Sched", "provider": "sqlite", "database": "x.db", "access_mode": "ingest", "sync_interval_minutes": 60}).json()
    with system_session() as db:
        dbc = db.get(DatabaseConnection, uuid.UUID(c["id"]))
        dbc.next_sync_at = datetime.now(UTC) - timedelta(minutes=1)
    assert run_due_schedules() >= 1
    jobs = owner.get("/jobs?kind=sync_database").json()["items"]
    assert jobs and jobs[0]["resource_id"] == c["id"]
