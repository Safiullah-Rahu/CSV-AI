"""SaaS integrations: OAuth (PKCE + signed single-use state), token refresh, and the ten connectors.

Vendor APIs are stubbed with ``httpx.MockTransport`` (``vendor_http.TRANSPORT``) and a
fake IMAP server; nothing here talks to a real vendor. (The GitLab connector was also
checked once by hand against a public gitlab.com project; see the catalog's ``verified``.)
"""

from __future__ import annotations

import base64
import email.utils
import gzip
import hashlib
import imaplib
import io
import ipaddress
import json
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import jwt
import pytest

from app.config import get_settings
from app.connectors import integrations as ic
from app.connectors import oauth, saas, vendor_http
from app.connectors.integrations import build
from app.core.security import decrypt_secret, encrypt_secret
from app.utils.errors import ConnectorError, UnsafeRequest, ValidationFailed

CALLBACK = "/api/v1/integrations/oauth/callback"
FOLDER = "application/vnd.google-apps.folder"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
VENDORS = ("google", "microsoft", "dropbox", "notion", "atlassian", "slack")


# ---------------------------------------------------------------------------- fixtures


class Vendor:
    """Route table for the mock transport: (METHOD, host + raw path) → handler(request)."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Callable[[httpx.Request], Any]] = {}
        self.calls: list[httpx.Request] = []
        self.sleeps: list[float] = []

    def on(self, method: str, url: str, fn: Callable[[httpx.Request], Any]) -> None:
        self.routes[(method, url)] = fn

    def called(self, url: str) -> list[httpx.Request]:
        return [r for r in self.calls if self.key(r)[1] == url]

    @staticmethod
    def key(request: httpx.Request) -> tuple[str, str]:
        return request.method, request.url.host + request.url.raw_path.decode().split("?")[0]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        fn = self.routes.get(self.key(request)) or self.routes.get((request.method, request.url.host + unquote(request.url.raw_path.decode().split("?")[0])))
        if fn is None:
            return httpx.Response(404, json={"error": {"message": f"no stub for {self.key(request)}"}})
        out = fn(request)
        return out if isinstance(out, httpx.Response) else httpx.Response(200, json=out)


@pytest.fixture()
def vendor(monkeypatch: pytest.MonkeyPatch) -> Vendor:
    v = Vendor()
    monkeypatch.setattr(vendor_http, "TRANSPORT", httpx.MockTransport(v.handle))
    monkeypatch.setattr(vendor_http, "_sleep", v.sleeps.append)
    return v


@pytest.fixture()
def fake_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vendor and *.test hosts resolve to a public address; '*.internal' to a private one."""
    from app.core import netguard

    def resolve(host: str) -> list[Any]:
        try:
            return [ipaddress.ip_address(host)]
        except ValueError:
            pass
        if host == "localhost":
            return [ipaddress.ip_address("127.0.0.1")]
        return [ipaddress.ip_address("10.9.8.7" if host.endswith(".internal") else "93.184.216.34")]

    monkeypatch.setattr(netguard, "_resolve", resolve)


@pytest.fixture()
def oauth_apps(monkeypatch: pytest.MonkeyPatch) -> Any:
    s = get_settings()
    for v in VENDORS:
        monkeypatch.setattr(s, f"oauth_{v}_client_id", f"{v}-client-id")
        monkeypatch.setattr(s, f"oauth_{v}_client_secret", f"{v}-client-secret-XYZ")
    monkeypatch.setattr(s, "oauth_microsoft_tenant", "common")
    return s


@pytest.fixture()
def no_oauth_apps(monkeypatch: pytest.MonkeyPatch) -> Any:
    s = get_settings()
    for v in VENDORS:
        monkeypatch.setattr(s, f"oauth_{v}_client_id", None)
        monkeypatch.setattr(s, f"oauth_{v}_client_secret", None)
    return s


def _query(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


def _form(request: httpx.Request) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(request.content.decode()).items()}


def _callback(api: Any, **params: str) -> dict[str, str]:
    r = api.client.get(CALLBACK, params=params, follow_redirects=False)
    assert r.status_code == 302, r.text
    location = r.headers["location"]
    assert location.startswith(get_settings().public_app_url.rstrip("/") + "/integrations?")
    return _query(location)


def _ms_tokens(**extra: Any) -> str:
    return oauth.dump({"type": "oauth", "provider": "microsoft", "access_token": "MS-ACCESS", "refresh_token": "MS-REFRESH", "expires_at": int(time.time()) + 3000, "requested_scopes": ["offline_access", "User.Read", "Files.Read.All"], "extras": {"tenant": "common"}, **extra})


def _docx(text: str) -> bytes:
    import docx

    d = docx.Document()
    d.add_heading("Leave policy", 1)
    d.add_paragraph(text)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


# --------------------------------------------------------------- OAuth: Google Drive end to end


def test_google_drive_oauth_flow_sync_and_refresh(make_user, vendor, oauth_apps, fake_dns):
    from app.db.session import system_session
    from app.models import Credential, Integration

    public = get_settings().public_app_url.rstrip("/")
    owner = make_user("OAuth Drive Co")
    kb = owner.post("/knowledge-bases", json={"name": "Drive KB"}).json()
    created = owner.post("/integrations", json={"name": "Team drive", "provider": "google_drive", "config": {"folder_id": "FOLDER_root1"}, "kb_id": kb["id"]})
    assert created.status_code == 201, created.text
    i = created.json()
    assert i["auth_mode"] == "oauth" and i["oauth"]["connected"] is False and not i["has_secret"] and i["status"] == "pending"
    # Never a fake connection: without tokens, test and sync refuse.
    assert owner.post(f"/integrations/{i['id']}/test").json()["code"] == "not_authorized"
    assert owner.post(f"/integrations/{i['id']}/sync", json={"mode": "full"}).status_code == 422
    assert owner.patch(f"/integrations/{i['id']}", json={"secret": '{"type":"oauth","access_token":"forged"}'}).status_code == 422

    start = owner.post(f"/integrations/{i['id']}/oauth/start")
    assert start.status_code == 200, start.text
    url = start.json()["authorization_url"]
    q = _query(url)
    assert urlparse(url).netloc == "accounts.google.com" and q["response_type"] == "code"
    assert q["client_id"] == "google-client-id" and q["redirect_uri"] == f"{public}/api/v1/integrations/oauth/callback"
    assert q["access_type"] == "offline" and q["prompt"] == "consent" and q["scope"] == "https://www.googleapis.com/auth/drive.readonly"
    assert q["code_challenge_method"] == "S256" and "client-secret" not in url
    claims = jwt.decode(q["state"], options={"verify_signature": False})
    verifier = decrypt_secret(claims["pkv"])  # the PKCE verifier travels encrypted
    assert verifier not in q["state"] and claims["typ"] == "oauth_state" and claims["iid"] == i["id"] and claims["org"] == owner.org_id
    assert claims["exp"] - claims["iat"] == 600
    assert base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode() == q["code_challenge"]

    exchanged: dict[str, Any] = {}

    def token(req: httpx.Request) -> Any:
        form = _form(req)
        if form["grant_type"] == "authorization_code":
            exchanged.update(form)
            return {"access_token": "ya29.ACCESS-ONE", "refresh_token": "1//REFRESH-ONE", "expires_in": 3599, "scope": "https://www.googleapis.com/auth/drive.readonly", "token_type": "Bearer"}
        exchanged["refresh"] = form
        return {"access_token": "ya29.ACCESS-TWO", "expires_in": 3599, "token_type": "Bearer"}  # no new refresh token: keep the old one

    vendor.on("POST", "oauth2.googleapis.com/token", token)
    vendor.on("GET", "www.googleapis.com/drive/v3/about", lambda r: {"user": {"displayName": "Ana", "emailAddress": "ana@acme.test"}})
    vendor.on("GET", "www.googleapis.com/drive/v3/files/FOLDER_root1", lambda r: {"id": "FOLDER_root1", "name": "Handbook", "mimeType": FOLDER})

    back = _callback(owner, code="auth-code-1", state=q["state"])
    assert back == {"oauth": "connected", "integration": i["id"]}
    assert exchanged["code"] == "auth-code-1" and exchanged["code_verifier"] == verifier and exchanged["redirect_uri"] == q["redirect_uri"]
    assert exchanged["client_id"] == "google-client-id" and exchanged["client_secret"] == "google-client-secret-XYZ"
    # Single use: the same state cannot be replayed.
    replay = _callback(owner, code="auth-code-1", state=q["state"])
    assert replay["oauth"] == "error" and "already used" in replay["message"]

    listed = owner.get("/integrations").json()
    row = next(x for x in listed if x["id"] == i["id"])
    assert row["status"] == "connected" and row["has_secret"] and row["oauth"]["connected"] and row["oauth"]["account"] == "ana@acme.test"
    exposed = json.dumps(listed) + json.dumps(owner.get(f"/connections/node/int:{i['id']}").json()) + json.dumps(owner.get("/audit?limit=200").json()) + json.dumps(owner.get("/events?limit=200").json())
    for secret in ("ACCESS-ONE", "REFRESH-ONE", verifier, "client-secret", claims["pkv"]):
        assert secret not in exposed
    with system_session() as db:
        integ = db.get(Integration, uuid.UUID(i["id"]))
        cred = db.get(Credential, integ.credential_id)
        assert "ACCESS-ONE" not in cred.ciphertext and "oauth_pending" not in integ.cursor
        stored = json.loads(decrypt_secret(cred.ciphertext))
        assert stored["access_token"] == "ya29.ACCESS-ONE" and stored["refresh_token"] == "1//REFRESH-ONE" and stored["provider"] == "google"
        stored["expires_at"] = int(time.time()) - 10  # force a refresh on the next sync
        cred.ciphertext = encrypt_secret(json.dumps(stored))

    def files(req: httpx.Request) -> Any:
        assert req.headers["authorization"] == "Bearer ya29.ACCESS-TWO"
        assert req.url.params["supportsAllDrives"] == "true"
        q, page = req.url.params["q"], req.url.params.get("pageToken")
        if "'FOLDER_root1' in parents" in q and not page:
            return {"files": [{"id": "DOC1", "name": "Leave policy", "mimeType": "application/vnd.google-apps.document", "version": "7", "webViewLink": "https://docs.google.com/document/d/DOC1"}, {"id": "SUB1", "name": "Archive", "mimeType": FOLDER}], "nextPageToken": "p2"}
        if "'FOLDER_root1' in parents" in q:
            return {"files": [{"id": "NOTES1", "name": "notes.md", "mimeType": "text/markdown", "md5Checksum": "md5-notes", "size": "60"}, {"id": "FORM1", "name": "Survey", "mimeType": "application/vnd.google-apps.form"}, {"id": "EXE1", "name": "tool.exe", "mimeType": "application/octet-stream", "size": "10"}]}
        if "'SUB1' in parents" in q:
            return {"files": [{"id": "OLD1", "name": "old.txt", "mimeType": "text/plain", "md5Checksum": "md5-old", "size": "20"}]}
        return {"files": []}

    vendor.on("GET", "www.googleapis.com/drive/v3/files", files)
    vendor.on("GET", "www.googleapis.com/drive/v3/files/DOC1/export", lambda r: httpx.Response(200, content=_docx("Annual leave is 27 days per year.")) if r.url.params["mimeType"] == DOCX else httpx.Response(400))
    vendor.on("GET", "www.googleapis.com/drive/v3/files/NOTES1", lambda r: httpx.Response(200, content=b"# Notes\n\nParking is free on Fridays.\n") if r.url.params.get("alt") == "media" else httpx.Response(400))
    vendor.on("GET", "www.googleapis.com/drive/v3/files/OLD1", lambda r: httpx.Response(200, content=b"Archived travel rules.\n") if r.url.params.get("alt") == "media" else httpx.Response(400))

    job = owner.post(f"/integrations/{i['id']}/sync", json={"mode": "full"}).json()
    j = owner.get(f"/jobs/{job['job_id']}").json()
    assert j["status"] == "succeeded", j
    assert j["result"]["added"] == 3 and j["result"]["seen"] == 3  # form and .exe skipped
    assert exchanged["refresh"]["grant_type"] == "refresh_token" and exchanged["refresh"]["refresh_token"] == "1//REFRESH-ONE"
    with system_session() as db:
        integ = db.get(Integration, uuid.UUID(i["id"]))
        rotated = json.loads(decrypt_secret(db.get(Credential, integ.credential_id).ciphertext))
    assert rotated["access_token"] == "ya29.ACCESS-TWO" and rotated["refresh_token"] == "1//REFRESH-ONE" and rotated["expires_at"] > time.time()
    docs = {d["filename"] for d in owner.get(f"/documents?kb_id={kb['id']}").json()["items"]}
    assert docs == {"Leave policy.docx", "notes.md", "old.txt"}
    ans = owner.post("/chat/ask", json={"question": "How many days of annual leave per year?", "kb_ids": [kb["id"]]}).json()
    assert "27 days" in ans["content"]
    exports = len(vendor.called("www.googleapis.com/drive/v3/files/DOC1/export"))
    job = owner.post(f"/integrations/{i['id']}/sync", json={"mode": "incremental"}).json()
    r = owner.get(f"/jobs/{job['job_id']}").json()["result"]
    assert r["unchanged"] == 3 and r["added"] == 0 and len(vendor.called("www.googleapis.com/drive/v3/files/DOC1/export")) == exports


# ----------------------------------------------------------- OAuth: state, permissions, quirks


def test_oauth_callback_rejects_forged_expired_replayed_and_unauthorized(make_user, invite_member, vendor, oauth_apps, fake_dns):
    from sqlalchemy import select

    from app.db.session import system_session
    from app.models import Membership

    owner = make_user("OAuth State Co")
    kb = owner.post("/knowledge-bases", json={"name": "KB"}).json()
    i = owner.post("/integrations", json={"name": "Box", "provider": "dropbox", "config": {"path": "/Team"}, "kb_id": kb["id"]}).json()
    first = _query(owner.post(f"/integrations/{i['id']}/oauth/start").json()["authorization_url"])
    assert first["token_access_type"] == "offline" and first["scope"] == "account_info.read files.metadata.read files.content.read"
    claims = jwt.decode(first["state"], options={"verify_signature": False})
    s = get_settings()
    forged = jwt.encode({**claims, "iid": i["id"]}, "not-the-platform-secret-not-the-platform", algorithm=s.jwt_algorithm)
    assert "state is invalid" in _callback(owner, code="c", state=forged)["message"]
    expired = jwt.encode({**claims, "exp": int(time.time()) - 5}, s.jwt_secret, algorithm=s.jwt_algorithm)
    assert "expired" in _callback(owner, code="c", state=expired)["message"]
    wrong_type = jwt.encode({**claims, "typ": "mfa"}, s.jwt_secret, algorithm=s.jwt_algorithm)
    assert "state is invalid" in _callback(owner, code="c", state=wrong_type)["message"]
    assert _callback(owner, code="c")["oauth"] == "error"
    # A newer start supersedes the older link.
    second = _query(owner.post(f"/integrations/{i['id']}/oauth/start").json()["authorization_url"])
    assert "replaced" in _callback(owner, code="c", state=first["state"])["message"]
    # Cancelling at the vendor consumes the state; nothing is exchanged.
    cancelled = _callback(owner, error="access_denied", state=second["state"])
    assert cancelled["oauth"] == "error" and "cancelled at Dropbox" in cancelled["message"]
    assert "already used" in _callback(owner, code="c", state=second["state"])["message"]
    assert not vendor.called("api.dropboxapi.com/oauth2/token")
    # The user's permission is re-checked at the callback.
    admin = invite_member(owner, "admin")
    third = _query(admin.post(f"/integrations/{i['id']}/oauth/start").json()["authorization_url"])
    with system_session() as db:
        m = db.scalar(select(Membership).where(Membership.user_id == uuid.UUID(admin.user_id), Membership.org_id == uuid.UUID(owner.org_id)))
        m.status = "suspended"
    assert "no longer have permission" in _callback(owner, code="c", state=third["state"])["message"]
    assert not vendor.called("api.dropboxapi.com/oauth2/token")
    assert owner.get("/integrations").json()[0]["has_secret"] is False


def test_oauth_vendor_quirks_notion_basic_atlassian_cloud_slack_errors(make_user, vendor, oauth_apps, fake_dns):
    owner = make_user("OAuth Quirks Co")
    kb = owner.post("/knowledge-bases", json={"name": "KB"}).json()

    # Notion: token exchange uses HTTP Basic (client id/secret) and a JSON body.
    notion = owner.post("/integrations", json={"name": "Wiki", "provider": "notion", "config": {}, "kb_id": kb["id"]}).json()
    assert notion["auth_mode"] == "oauth"
    nq = _query(owner.post(f"/integrations/{notion['id']}/oauth/start").json()["authorization_url"])
    assert nq["owner"] == "user" and "scope" not in nq

    def notion_token(req: httpx.Request) -> Any:
        assert req.headers["authorization"] == "Basic " + base64.b64encode(b"notion-client-id:notion-client-secret-XYZ").decode()
        body = json.loads(req.content)
        assert body["grant_type"] == "authorization_code" and body["code"] == "n-code" and "client_secret" not in body
        return {"access_token": "ntn_OAUTH_ACCESS", "token_type": "bearer", "workspace_name": "Acme HQ", "workspace_id": "w1", "bot_id": "b1"}

    vendor.on("POST", "api.notion.com/v1/oauth/token", notion_token)
    vendor.on("GET", "api.notion.com/v1/users/me", lambda r: {"bot": {"workspace_name": "Acme HQ"}})
    vendor.on("POST", "api.notion.com/v1/search", lambda r: {"results": [], "has_more": False})
    assert _callback(owner, code="n-code", state=nq["state"])["oauth"] == "connected"
    row = next(x for x in owner.get("/integrations").json() if x["id"] == notion["id"])
    assert row["oauth"]["account"] == "Acme HQ" and row["status"] == "connected"

    # Atlassian 3LO: audience + JSON exchange, then accessible-resources picks the cloud id.
    conf = owner.post("/integrations", json={"name": "Confluence", "provider": "confluence", "config": {"space_keys": ["ENG"], "site_url": "https://acme.atlassian.net"}, "kb_id": kb["id"]}).json()
    cq = _query(owner.post(f"/integrations/{conf['id']}/oauth/start").json()["authorization_url"])
    assert cq["audience"] == "api.atlassian.com" and cq["scope"] == "read:page:confluence read:space:confluence offline_access"

    def atl_token(req: httpx.Request) -> Any:
        body = json.loads(req.content)
        assert body["client_id"] == "atlassian-client-id" and body["code_verifier"]
        return {"access_token": "ATL-ACCESS", "refresh_token": "ATL-REFRESH", "expires_in": 3600, "scope": "read:page:confluence offline_access"}

    vendor.on("POST", "auth.atlassian.com/oauth/token", atl_token)
    vendor.on("GET", "api.atlassian.com/oauth/token/accessible-resources", lambda r: [{"id": "other-cloud", "url": "https://other.atlassian.net", "name": "Other", "scopes": ["read:page:confluence"]}, {"id": "cloud-123", "url": "https://acme.atlassian.net", "name": "Acme", "scopes": ["read:page:confluence"]}])
    vendor.on("GET", "api.atlassian.com/ex/confluence/cloud-123/wiki/api/v2/spaces", lambda r: {"results": [{"id": "77", "key": "ENG", "name": "Engineering"}]})
    assert _callback(owner, code="a-code", state=cq["state"])["oauth"] == "connected"
    row = next(x for x in owner.get("/integrations").json() if x["id"] == conf["id"])
    assert row["oauth"]["account"] == "Acme" and row["status"] == "connected"
    assert vendor.called("api.atlassian.com/ex/confluence/cloud-123/wiki/api/v2/spaces")[0].headers["authorization"] == "Bearer ATL-ACCESS"

    # Slack: comma-separated bot scopes; a token error arrives as HTTP 200 with ok=false.
    slack = owner.post("/integrations", json={"name": "Chat", "provider": "slack", "config": {"auth_mode": "oauth"}, "kb_id": kb["id"]}).json()
    sq = _query(owner.post(f"/integrations/{slack['id']}/oauth/start").json()["authorization_url"])
    assert sq["scope"] == "channels:read,channels:history,groups:read,groups:history,users:read"
    vendor.on("POST", "slack.com/api/oauth.v2.access", lambda r: {"ok": False, "error": "invalid_code"})
    back = _callback(owner, code="bad", state=sq["state"])
    assert back["oauth"] == "error" and "invalid_code" in back["message"]
    row = next(x for x in owner.get("/integrations").json() if x["id"] == slack["id"])
    assert row["status"] == "failed" and not row["has_secret"]


def test_catalog_needs_setup_and_refusals(make_user, no_oauth_apps, fake_dns):
    owner = make_user("Needs Setup Co")
    kb = owner.post("/knowledge-bases", json={"name": "KB"}).json()
    cat = {c["key"]: c for c in owner.get("/integrations/catalog").json()}
    saas_keys = ["google_drive", "onedrive", "sharepoint", "dropbox", "notion", "confluence", "slack", "teams", "email"]
    assert all(cat[k]["status"] == "beta" for k in saas_keys) and cat["gitlab"]["status"] == "tested" and cat["gitlab"]["verified"]
    assert not [k for k, c in cat.items() if c["status"] == "planned"]
    assert cat["dropbox"]["needs_setup"] == ["OAUTH_DROPBOX_CLIENT_ID", "OAUTH_DROPBOX_CLIENT_SECRET"] and cat["dropbox"]["oauth"]["configured"] is False
    assert cat["teams"]["oauth"]["redirect_uri"].endswith("/api/v1/integrations/oauth/callback") and "OAUTH_MICROSOFT_TENANT (optional, default common)" in cat["teams"]["oauth"]["env"]
    assert "needs_setup" not in cat["gitlab"] and "needs_setup" not in cat["email"]
    r = owner.post("/integrations", json={"name": "Box", "provider": "dropbox", "config": {}, "kb_id": kb["id"]})
    assert r.status_code == 501 and "OAUTH_DROPBOX_CLIENT_ID" in r.json()["error"]["message"]
    # Providers with a token alternative still connect without an OAuth app.
    notion = owner.post("/integrations", json={"name": "Wiki", "provider": "notion", "config": {}, "secret": "ntn_abc123", "kb_id": kb["id"]})
    assert notion.status_code == 201 and notion.json()["auth_mode"] == "token" and notion.json()["oauth"] is None
    assert owner.post(f"/integrations/{notion.json()['id']}/oauth/start").status_code == 422
    assert owner.post("/integrations", json={"name": "Bad token", "provider": "notion", "config": {"auth_mode": "token"}, "secret": "not-a-notion-token", "kb_id": kb["id"]}).status_code == 422
    assert owner.post("/integrations", json={"name": "Slack OAuth", "provider": "slack", "config": {"auth_mode": "oauth"}, "kb_id": kb["id"]}).status_code == 501
    s = get_settings()
    s.oauth_slack_client_id, s.oauth_slack_client_secret = "id", "secret"  # restored by the fixture's monkeypatch
    slack = owner.post("/integrations", json={"name": "Slack OAuth", "provider": "slack", "config": {"auth_mode": "oauth"}, "kb_id": kb["id"]}).json()
    s.oauth_slack_client_id = None
    r = owner.post(f"/integrations/{slack['id']}/oauth/start")
    assert r.status_code == 501 and "OAUTH_SLACK_CLIENT_ID" in r.json()["error"]["message"]


def test_ssrf_rejects_private_hosts(make_user, fake_dns):
    owner = make_user("SSRF SaaS Co")
    kb = owner.post("/knowledge-bases", json={"name": "KB"}).json()

    def create(provider: str, config: dict[str, Any], secret: str | None = None) -> Any:
        return owner.post("/integrations", json={"name": f"{provider} x", "provider": provider, "config": config, "secret": secret, "kb_id": kb["id"]})

    for base in ("http://10.1.2.3", "http://169.254.169.254", "https://gitlab.corp.internal"):
        r = create("gitlab", {"project": "a/b", "base_url": base})
        assert r.status_code == 400 and r.json()["error"]["code"] == "unsafe_request", base
    r = create("confluence", {"auth_mode": "token", "site_url": "https://10.0.0.5", "email": "a@b.c", "space_keys": ["ENG"]}, "tok")
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsafe_request"
    r = create("email", {"host": "mail.corp.internal", "username": "a", "port": 993}, "pw")
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsafe_request"
    assert create("email", {"host": "imap.acme.test", "username": "a", "security": "none"}, "pw").status_code == 422
    assert create("gitlab", {"project": "group/sub/project"}).status_code == 201


# ------------------------------------------------------------------------- HTTP plumbing


def test_vendor_http_caps_retry_after_and_gzip(vendor, monkeypatch):
    vendor.on("GET", "api.vendor.test/big", lambda r: httpx.Response(200, content=b"x" * 5000))
    monkeypatch.setattr(ic, "MAX_BYTES", 1000)
    with pytest.raises(ConnectorError, match="exceeds"):
        vendor_http.VendorHTTP().request("GET", "https://api.vendor.test/big")
    vendor.on("GET", "api.vendor.test/slow", lambda r: httpx.Response(429, headers={"Retry-After": "3600"}))
    with pytest.raises(ConnectorError, match="rate limiting"):
        vendor_http.VendorHTTP().request("GET", "https://api.vendor.test/slow")
    hits = {"n": 0}

    def flaky(r: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "2"}) if hits["n"] == 1 else httpx.Response(200, json={"ok": True})

    vendor.on("GET", "api.vendor.test/flaky", flaky)
    assert vendor_http.VendorHTTP().request("GET", "https://api.vendor.test/flaky").json() == {"ok": True} and vendor.sleeps == [2.0]
    vendor.on("GET", "api.vendor.test/gz", lambda r: httpx.Response(200, content=gzip.compress(b'{"a": 1}'), headers={"content-encoding": "gzip"}))
    assert vendor_http.VendorHTTP().request("GET", "https://api.vendor.test/gz").json() == {"a": 1}
    assert vendor_http.retry_after_seconds(email.utils.formatdate(time.time() + 30, usegmt=True)) > 20


# ---------------------------------------------------------------------------- connectors


def test_onedrive_recursion_401_refresh_persisted_and_presigned_download(vendor, oauth_apps, fake_dns):
    saved: list[str] = []
    c = build("onedrive", {"auth_mode": "oauth", "path": "/Handbook", "max_items": 10}, _ms_tokens(access_token="STALE", extras={"tenant": "contoso.onmicrosoft.com"}), save_secret=saved.append)

    def token(req: httpx.Request) -> Any:
        form = _form(req)
        assert form["grant_type"] == "refresh_token" and form["refresh_token"] == "MS-REFRESH" and form["scope"] == "offline_access User.Read Files.Read.All"
        return {"access_token": "MS-ACCESS-2", "refresh_token": "MS-REFRESH-2", "expires_in": 3600}

    def auth(fn: Callable[[httpx.Request], Any]) -> Callable[[httpx.Request], Any]:
        return lambda r: fn(r) if r.headers.get("authorization") == "Bearer MS-ACCESS-2" else httpx.Response(401, json={"error": {"code": "InvalidAuthenticationToken"}})

    g = "graph.microsoft.com/v1.0"
    vendor.on("POST", "login.microsoftonline.com/contoso.onmicrosoft.com/oauth2/v2.0/token", token)
    vendor.on("GET", f"{g}/me/drive", auth(lambda r: {"id": "drv1", "driveType": "business"}))
    vendor.on("GET", f"{g}/drives/drv1/root:/Handbook", auth(lambda r: {"id": "ROOT", "name": "Handbook", "folder": {}}))
    vendor.on("GET", f"{g}/drives/drv1/items/ROOT/children", auth(lambda r: {"value": [{"id": "F2", "name": "setup.exe", "file": {}, "size": 5}, {"id": "F4", "name": "big.pdf", "file": {}, "size": 10**9}]} if r.url.params.get("$skiptoken") else {"value": [{"id": "F1", "name": "policy.md", "file": {}, "size": 30, "cTag": "c1", "eTag": "e1", "webUrl": "https://contoso-my.sharepoint.com/policy.md"}, {"id": "SUB", "name": "HR", "folder": {"childCount": 1}}], "@odata.nextLink": f"https://{g}/drives/drv1/items/ROOT/children?$skiptoken=2"}))
    vendor.on("GET", f"{g}/drives/drv1/items/SUB/children", auth(lambda r: {"value": [{"id": "F3", "name": "leave.txt", "file": {}, "size": 20, "eTag": "e3"}]}))
    vendor.on("GET", f"{g}/drives/drv1/items/F1/content", auth(lambda r: httpx.Response(302, headers={"location": "https://contoso-my.sharepoint.com/download.aspx?tempauth=signed"})))
    vendor.on("GET", "contoso-my.sharepoint.com/download.aspx", lambda r: httpx.Response(400) if "authorization" in r.headers else httpx.Response(200, content=b"# Policy\n"))
    vendor.on("GET", f"{g}/drives/drv1/items/F3/content", auth(lambda r: httpx.Response(302, headers={"location": "https://10.0.0.9/steal"})))

    items = list(c.items())
    assert [(it.external_id, it.version, it.title) for it in items] == [("drv1:F1", "c1", "policy.md"), ("drv1:F3", "e3", "HR/leave.txt")]
    assert c.fetch(items[0]).content == b"# Policy\n"  # bearer token not sent to the pre-signed host
    with pytest.raises(UnsafeRequest):
        c.fetch(items[1])  # redirect into a private network is refused
    assert len(saved) == 1 and json.loads(saved[0])["refresh_token"] == "MS-REFRESH-2" and json.loads(saved[0])["access_token"] == "MS-ACCESS-2"
    with pytest.raises(ValidationFailed):
        build("onedrive", {"auth_mode": "oauth", "path": "/../etc"}, None).validate()
    with pytest.raises(ValidationFailed):
        build("onedrive", {"auth_mode": "oauth", "tenant": "evil.com/../x"}, None).validate()


def test_sharepoint_site_library_items(vendor, oauth_apps, fake_dns):
    g = "graph.microsoft.com/v1.0"
    c = build("sharepoint", {"auth_mode": "oauth", "hostname": "contoso.sharepoint.com", "site_path": "/sites/HR", "library": "Policies"}, _ms_tokens())
    vendor.on("GET", f"{g}/sites/contoso.sharepoint.com:/sites/HR", lambda r: {"id": "contoso.sharepoint.com,g1,g2", "displayName": "HR"})
    vendor.on("GET", f"{g}/sites/contoso.sharepoint.com,g1,g2/drives", lambda r: {"value": [{"id": "d1", "name": "Documents", "driveType": "documentLibrary"}, {"id": "d2", "name": "Policies", "driveType": "documentLibrary"}]})
    vendor.on("GET", f"{g}/drives/d2/root", lambda r: {"id": "R2", "folder": {}})
    vendor.on("GET", f"{g}/drives/d2/items/R2/children", lambda r: {"value": [{"id": "P1", "name": "travel.pdf", "file": {}, "size": 100, "cTag": "ct1", "webUrl": "https://contoso.sharepoint.com/sites/HR/Policies/travel.pdf"}]})
    vendor.on("GET", f"{g}/me", lambda r: {"userPrincipalName": "ana@contoso.com"})
    assert c.test() == {"ok": True, "account": "ana@contoso.com", "site": "HR", "libraries": ["Policies"]}
    assert [(it.external_id, it.version) for it in c.items()] == [("d2:P1", "ct1")]
    assert not vendor.called(f"{g}/drives/d1/root")
    with pytest.raises(ValidationFailed):
        build("sharepoint", {"auth_mode": "oauth", "hostname": "contoso sharepoint"}, None).validate()


def test_dropbox_list_continue_cap_and_download(vendor, oauth_apps):
    tokens = oauth.dump({"type": "oauth", "provider": "dropbox", "access_token": "DBX", "refresh_token": "DBX-R", "expires_at": int(time.time()) + 3000, "extras": {}})
    c = build("dropbox", {"path": "Team", "max_items": 3}, tokens)

    def list_folder(req: httpx.Request) -> Any:
        body = json.loads(req.content)
        assert body["path"] == "/Team" and body["recursive"] is True
        return {"entries": [{".tag": "folder", "name": "HR", "id": "id:F"}, {".tag": "file", "name": "a.md", "id": "id:A", "path_display": "/Team/a.md", "content_hash": "h1", "rev": "r1", "size": 10}, {".tag": "file", "name": "b.bin", "id": "id:B", "size": 10}, {".tag": "file", "name": "big.pdf", "id": "id:BIG", "size": 10**9}], "cursor": "c1", "has_more": True}

    vendor.on("POST", "api.dropboxapi.com/2/files/list_folder", list_folder)
    vendor.on("POST", "api.dropboxapi.com/2/files/list_folder/continue", lambda r: {"entries": [{".tag": "file", "name": "c.txt", "id": "id:C", "path_display": "/Team/HR/c.txt", "rev": "r3", "size": 5}, {".tag": "file", "name": "d.md", "id": "id:D", "content_hash": "h4", "size": 5}, {".tag": "file", "name": "e.md", "id": "id:E", "content_hash": "h5", "size": 5}], "cursor": "c2", "has_more": False} if json.loads(r.content)["cursor"] == "c1" else httpx.Response(409))
    vendor.on("POST", "content.dropboxapi.com/2/files/download", lambda r: httpx.Response(200, content=b"alpha") if json.loads(r.headers["dropbox-api-arg"]) == {"path": "id:A"} and r.headers["authorization"] == "Bearer DBX" else httpx.Response(409))
    items = list(c.items())
    assert [(it.external_id, it.version) for it in items] == [("id:A", "h1"), ("id:C", "r3"), ("id:D", "h4")]
    assert items[1].title == "Team/HR/c.txt" and items[0].url.startswith("https://www.dropbox.com/home/Team?preview=a.md")
    assert c.fetch(items[0]).content == b"alpha"


def test_notion_search_pagination_and_markdown_rendering(vendor):
    c = build("notion", {"auth_mode": "token"}, "ntn_test_secret")
    pages = {
        None: {"results": [{"id": "page-1111-aaaa", "url": "https://notion.so/p1", "last_edited_time": "2026-10-30T10:00:00.000Z", "properties": {"Name": {"type": "title", "title": [{"plain_text": "Onboarding"}]}}}], "has_more": True, "next_cursor": "n2"},
        "n2": {"results": [{"id": "page-2", "archived": True, "properties": {}}, {"id": "page-3333-bbbb", "url": "https://notion.so/p3", "last_edited_time": "2026-10-29T10:00:00.000Z", "properties": {"title": {"type": "title", "title": [{"plain_text": "Expenses"}]}}}], "has_more": False},
    }

    def search(req: httpx.Request) -> Any:
        assert req.headers["authorization"] == "Bearer ntn_test_secret" and req.headers["notion-version"] == "2022-06-28"
        return pages[json.loads(req.content).get("start_cursor")]

    def rt(text: str, **ann: bool) -> list[dict[str, Any]]:
        return [{"plain_text": text, "annotations": ann}]

    blocks = {
        "page-1111-aaaa": [
            {"id": "b1", "type": "heading_1", "heading_1": {"rich_text": rt("Welcome")}},
            {"id": "b2", "type": "paragraph", "paragraph": {"rich_text": rt("Read this ") + rt("first", bold=True)}},
            {"id": "b3", "type": "bulleted_list_item", "has_children": True, "bulleted_list_item": {"rich_text": rt("Accounts")}},
            {"id": "b4", "type": "to_do", "to_do": {"rich_text": rt("Laptop"), "checked": True}},
            {"id": "b5", "type": "code", "code": {"language": "python", "rich_text": rt("print('hi')")}},
            {"id": "b6", "type": "table", "has_children": True, "table": {"has_column_header": True}},
        ],
        "b3": [{"id": "b3a", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": rt("Email")}}],
        "b6": [{"id": "r1", "type": "table_row", "table_row": {"cells": [rt("Item"), rt("Owner")]}}, {"id": "r2", "type": "table_row", "table_row": {"cells": [rt("VPN"), rt("IT")]}}],
    }
    vendor.on("POST", "api.notion.com/v1/search", search)
    for bid, children in blocks.items():
        vendor.on("GET", f"api.notion.com/v1/blocks/{bid}/children", lambda r, children=children: {"results": children, "has_more": False})
    items = list(c.items())
    assert [(it.external_id, it.title, it.version) for it in items] == [("page-1111-aaaa", "Onboarding", "2026-10-30T10:00:00.000Z"), ("page-3333-bbbb", "Expenses", "2026-10-29T10:00:00.000Z")]
    assert items[0].filename == "Onboarding-page1111.md"
    md = c.fetch(items[0]).content.decode()
    assert md.startswith("# Onboarding") and "## Welcome" in md and "Read this **first**" in md
    assert "- Accounts" in md and "  - Email" in md and "- [x] Laptop" in md and "```python\nprint('hi')\n```" in md
    assert "| Item | Owner |\n| --- | --- |" in md and "| VPN | IT |" in md
    with pytest.raises(ValidationFailed):
        build("notion", {"auth_mode": "token"}, "plain-token").validate()


def test_confluence_token_mode_basic_auth_pagination_and_fetch(vendor, fake_dns):
    c = build("confluence", {"auth_mode": "token", "site_url": "https://acme.atlassian.net/", "email": "bot@acme.test", "space_keys": "ENG"}, "atl-api-token")
    basic = "Basic " + base64.b64encode(b"bot@acme.test:atl-api-token").decode()
    root = "acme.atlassian.net/wiki/api/v2"

    def spaces(req: httpx.Request) -> Any:
        assert req.headers["authorization"] == basic and req.url.params["keys"] == "ENG"
        return {"results": [{"id": "77", "key": "ENG", "name": "Engineering"}]}

    def pages(req: httpx.Request) -> Any:
        if req.url.params.get("cursor") == "abc":
            return {"results": [{"id": "102", "title": "Runbook", "version": {"number": 1}, "_links": {"webui": "/spaces/ENG/pages/102"}}], "_links": {}}
        assert req.url.params["status"] == "current"
        return {"results": [{"id": "101", "title": "Deploys", "version": {"number": 3}, "_links": {"webui": "/spaces/ENG/pages/101/Deploys"}}], "_links": {"next": "/wiki/api/v2/spaces/77/pages?cursor=abc", "base": "https://acme.atlassian.net/wiki"}}

    vendor.on("GET", f"{root}/spaces", spaces)
    vendor.on("GET", f"{root}/spaces/77/pages", pages)
    vendor.on("GET", f"{root}/pages/101", lambda r: {"id": "101", "title": "Deploys <Prod>", "body": {"storage": {"value": "<p>Deploys happen on Tuesdays.</p>"}}} if r.url.params["body-format"] == "storage" else httpx.Response(400))
    assert c.test() == {"ok": True, "site": "https://acme.atlassian.net", "spaces": ["Engineering"]}
    items = list(c.items())
    assert [(it.external_id, it.version) for it in items] == [("101", "3"), ("102", "1")]
    assert items[0].url == "https://acme.atlassian.net/wiki/spaces/ENG/pages/101/Deploys" and items[0].filename.endswith(".html")
    html = c.fetch(items[0]).content.decode()
    assert "<title>Deploys &lt;Prod&gt;</title>" in html and "Deploys happen on Tuesdays." in html
    vendor.on("GET", f"{root}/spaces", lambda r: {"results": []})
    with pytest.raises(ConnectorError, match="ENG"):
        c.test()
    with pytest.raises(ValidationFailed):
        build("confluence", {"auth_mode": "token", "site_url": "https://acme.atlassian.net", "space_keys": ["ENG"]}, "t").validate()  # email missing


def test_slack_channels_history_day_documents_and_rate_limit(vendor):
    c = build("slack", {"auth_mode": "token", "channels": ["#general"], "lookback_days": 7}, "xoxb-test-token")
    today = datetime.now(UTC).replace(hour=10, minute=0, second=0, microsecond=0)
    d1, d2 = today - timedelta(days=1), today - timedelta(days=2)
    state = {"list_calls": 0, "edited": "1"}

    def conv_list(req: httpx.Request) -> Any:
        state["list_calls"] += 1
        if state["list_calls"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        assert req.headers["authorization"] == "Bearer xoxb-test-token"
        if req.url.params.get("cursor") == "x":
            return {"ok": True, "channels": [{"id": "C2", "name": "random", "is_member": True}], "response_metadata": {"next_cursor": ""}}
        return {"ok": True, "channels": [{"id": "C1", "name": "general", "is_member": True}], "response_metadata": {"next_cursor": "x"}}

    def history(req: httpx.Request) -> Any:
        assert req.url.params["channel"] == "C1" and float(req.url.params["oldest"]) < d2.timestamp()
        if req.url.params.get("cursor") == "h2":
            return {"ok": True, "messages": [{"ts": f"{d2.timestamp():.6f}", "user": "U1", "text": "Old plan <https://wiki.acme.test/p|here>", "edited": {"ts": state["edited"]}}], "has_more": False}
        return {"ok": True, "messages": [{"ts": f"{d1.timestamp() + 60:.6f}", "user": "U1", "text": "<@U2> deploy at 5 &amp; verify", "reply_count": 2}, {"ts": f"{d1.timestamp():.6f}", "subtype": "channel_join", "user": "U3", "text": "joined"}], "has_more": True, "response_metadata": {"next_cursor": "h2"}}

    users = {"U1": {"real_name": "Ana Lee", "profile": {"display_name": ""}}, "U2": {"real_name": "Bea", "profile": {"display_name": "bea"}}}
    vendor.on("GET", "slack.com/api/conversations.list", conv_list)
    vendor.on("GET", "slack.com/api/conversations.history", history)
    vendor.on("GET", "slack.com/api/users.info", lambda r: {"ok": True, "user": users[r.url.params["user"]]})
    items = list(c.items())
    assert vendor.sleeps == [3.0]
    assert [it.external_id for it in items] == [f"C1:{d1.date().isoformat()}", f"C1:{d2.date().isoformat()}"]
    day1 = c.fetch(items[0]).content.decode()
    assert day1.startswith(f"# #general — {d1.date().isoformat()}") and "**Ana Lee**: @bea deploy at 5 & verify (2 replies in thread)" in day1 and "joined" not in day1
    assert "[here](https://wiki.acme.test/p)" in c.fetch(items[1]).content.decode()
    state["edited"] = "2"
    again = {it.external_id: it.version for it in build("slack", {"auth_mode": "token", "channels": ["C1"], "lookback_days": 7}, "xoxb-test-token").items()}
    assert again[items[0].external_id] == items[0].version and again[items[1].external_id] != items[1].version
    vendor.on("GET", "slack.com/api/auth.test", lambda r: {"ok": False, "error": "invalid_auth"})
    with pytest.raises(ConnectorError, match="invalid_auth"):
        c.test()
    with pytest.raises(ValidationFailed):
        build("slack", {"auth_mode": "token"}, "xoxp-user-token").validate()
    missing = build("slack", {"auth_mode": "token", "channels": ["#nope"]}, "xoxb-test-token")
    with pytest.raises(ConnectorError, match="nope"):
        list(missing.items())


def test_teams_messages_with_replies_and_lookback(vendor, oauth_apps):
    team = "11111111-2222-3333-4444-555555555555"
    g = "graph.microsoft.com/v1.0"
    c = build("teams", {"team_id": team, "lookback_days": 3}, _ms_tokens())
    now = datetime.now(UTC).replace(hour=9, minute=0, second=0, microsecond=0)
    recent, old = (now - timedelta(days=1)).isoformat().replace("+00:00", "Z"), (now - timedelta(days=10)).isoformat().replace("+00:00", "Z")
    vendor.on("GET", f"{g}/teams/{team}/channels", lambda r: {"value": [{"id": "19:abc@thread.tacv2", "displayName": "General", "webUrl": "https://teams.microsoft.com/l/channel/x"}]})
    msgs = f"{g}/teams/{team}/channels/19%3Aabc%40thread.tacv2/messages"

    def messages(req: httpx.Request) -> Any:
        if req.url.params.get("$skiptoken") == "2":
            return {"value": [{"id": "M9", "messageType": "message", "createdDateTime": old, "lastModifiedDateTime": old, "body": {"contentType": "text", "content": "ancient"}}], "@odata.nextLink": f"https://{msgs}?$skiptoken=3"}
        assert req.url.params["$expand"] == "replies"
        return {"value": [
            {"id": "M1", "messageType": "message", "createdDateTime": recent, "lastModifiedDateTime": recent, "subject": "Release", "from": {"user": {"displayName": "Ana"}}, "body": {"contentType": "html", "content": "<p>Ship <b>v2</b> today</p>"}, "webUrl": "https://teams.microsoft.com/l/message/M1",
             "replies": [{"id": "R1", "messageType": "message", "createdDateTime": recent, "lastModifiedDateTime": recent, "from": {"user": {"displayName": "Bea"}}, "body": {"contentType": "text", "content": "Done"}}]},
            {"id": "M2", "messageType": "systemEventMessage", "createdDateTime": recent, "body": {"content": "<systemEventMessage/>"}},
            {"id": "M3", "messageType": "message", "createdDateTime": recent, "deletedDateTime": recent, "body": {"content": "gone"}},
        ], "@odata.nextLink": f"https://{msgs}?$skiptoken=2"}

    vendor.on("GET", msgs, messages)
    items = list(c.items())
    assert len(items) == 1 and items[0].external_id == f"19:abc@thread.tacv2:{(now - timedelta(days=1)).date().isoformat()}"
    md = c.fetch(items[0]).content.decode()
    assert "**Ana**: **Release** — Ship v2 today" in md and "reply" in md and "**Bea**: Done" in md and "gone" not in md and "ancient" not in md
    assert len(vendor.called(msgs)) == 2  # the all-old page stops paging; skiptoken=3 is never requested
    with pytest.raises(ValidationFailed):
        build("teams", {"team_id": "not-a-guid"}, None).validate()
    with pytest.raises(ValidationFailed):
        build("teams", {"team_id": team, "channel_ids": ["general"]}, None).validate()


def test_gitlab_self_managed_token_pagination_and_raw_fetch(vendor, fake_dns):
    c = build("gitlab", {"project": "docs/handbook", "base_url": "https://gitlab.acme.test/", "path_prefix": "docs/", "extensions": ["md"]}, "glpat-SECRET")
    api = "gitlab.acme.test/api/v4/projects/docs%2Fhandbook"

    def tree(req: httpx.Request) -> Any:
        assert req.headers["private-token"] == "glpat-SECRET" and req.url.params["ref"] == "main" and req.url.params["path"] == "docs"
        if req.url.params["page"] == "2":
            return httpx.Response(200, json=[{"id": "bbbbbbb2", "type": "blob", "path": "docs/sub/b.md"}], headers={"x-next-page": ""})
        return httpx.Response(200, json=[{"id": "aaaaaaa1", "type": "blob", "path": "docs/a.md"}, {"id": "ccccccc3", "type": "blob", "path": "docs/logo.png"}, {"id": "ddddddd4", "type": "tree", "path": "docs/sub"}], headers={"x-next-page": "2"})

    vendor.on("GET", api, lambda r: {"path_with_namespace": "docs/handbook", "default_branch": "main", "visibility": "private", "web_url": "https://gitlab.acme.test/docs/handbook"})
    vendor.on("GET", f"{api}/repository/tree", tree)
    vendor.on("GET", f"{api}/repository/blobs/aaaaaaa1/raw", lambda r: httpx.Response(200, content=b"# A\n"))
    assert c.test()["visibility"] == "private"
    items = list(c.items())
    assert [(it.external_id, it.version, it.filename) for it in items] == [("docs/a.md", "aaaaaaa1", "docs__a.md"), ("docs/sub/b.md", "bbbbbbb2", "docs__sub__b.md")]
    assert items[0].url == "https://gitlab.acme.test/docs/handbook/-/blob/main/docs/a.md"
    assert c.fetch(items[0]).content == b"# A\n"
    vendor.on("GET", api, lambda r: httpx.Response(404, json={"message": "404 Project Not Found"}))
    with pytest.raises(ConnectorError, match="Project not found"):
        c.test()
    with pytest.raises(ValidationFailed):
        build("gitlab", {"project": "../../etc"}, None).validate()


class FakeIMAP:
    """Just enough of imaplib.IMAP4 for the connector."""

    messages: dict[str, bytes] = {}
    password = "app-pass"
    searches: list[tuple[Any, ...]] = []

    def __init__(self, host: str, port: int, ssl_context: Any = None, timeout: float | None = None) -> None:
        self.host, self.port, self.ssl_context, self.timeout = host, port, ssl_context, timeout

    def login(self, user: str, password: str) -> tuple[str, list[bytes]]:
        if password != self.password:
            raise imaplib.IMAP4.error("AUTHENTICATIONFAILED")
        return "OK", [b"Logged in"]

    def select(self, mailbox: str, readonly: bool = False) -> tuple[str, list[bytes]]:
        assert readonly and mailbox == '"INBOX"'
        return "OK", [str(len(self.messages)).encode()]

    def uid(self, command: str, *args: Any) -> tuple[str, list[Any]]:
        if command == "SEARCH":
            FakeIMAP.searches.append(args)
            return "OK", [" ".join(self.messages).encode()]
        uids, what = args
        rows: list[Any] = []
        for uid in uids.split(","):
            raw = self.messages[uid]
            if "HEADER.FIELDS" in what:
                head = raw.split(b"\r\n\r\n", 1)[0] + b"\r\n\r\n"
                rows += [(f"1 (UID {uid} RFC822.SIZE {len(raw)} BODY[HEADER.FIELDS (MESSAGE-ID SUBJECT DATE)] {{{len(head)}}}".encode(), head), b")"]
            else:
                rows += [(f"1 (UID {uid} BODY[] {{{len(raw)}}}".encode(), raw), b")"]
        return "OK", rows

    def logout(self) -> tuple[str, list[bytes]]:
        return "BYE", []


def _mail(mid: str, subject: str, body: str, *, html: str | None = None) -> bytes:
    from email.message import EmailMessage

    m = EmailMessage()
    m["Message-ID"], m["Subject"], m["From"], m["To"] = mid, subject, "Ana <ana@acme.test>", "team@acme.test"
    m["Date"] = email.utils.format_datetime(datetime(2026, 10, 30, 9, 30, tzinfo=UTC))
    if body:
        m.set_content(body)
        if html:
            m.add_alternative(html, subtype="html")
    else:
        m.set_content(html or "", subtype="html")
    return bytes(m).replace(b"\n", b"\r\n").replace(b"\r\r\n", b"\r\n")


def test_email_imap_messages_become_markdown(monkeypatch, fake_dns):
    FakeIMAP.messages = {
        "11": _mail("<a1@acme.test>", "Quarterly numbers (older copy)", "dup\n"),
        "12": _mail("<a2@acme.test>", "Newsletter", "", html="<html><body><h1>News</h1><script>x()</script><p>Office closed <b>Friday</b>.</p></body></html>"),
        "13": _mail("<a1@acme.test>", "Quarterly numbers", "Revenue grew 12% in Q3.\n", html="<p>IGNORED html part</p>"),
    }
    FakeIMAP.searches = []
    monkeypatch.setattr(saas, "IMAP_SSL", FakeIMAP)
    cfg = {"host": "imap.acme.test", "username": "ana@acme.test", "lookback_days": 14}
    assert build("email", cfg, "app-pass").test() == {"ok": True, "folder": "INBOX", "messages": 3}
    c = build("email", cfg, "app-pass")
    items = list(c.items())
    since = FakeIMAP.searches[-1]
    assert since[0] is None and since[1] == "SINCE" and since[2][3:6] in ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    assert [it.external_id for it in items] == ["INBOX:<a1@acme.test>", "INBOX:<a2@acme.test>"]  # newest UID first; the older duplicate Message-ID is dropped
    assert items[0].version == f"<a1@acme.test>:{len(FakeIMAP.messages['13'])}" and items[0].filename.endswith(".md")
    md = c.fetch(items[0]).content.decode()
    assert md.startswith("# Quarterly numbers\n") and "- **From:** Ana <ana@acme.test>" in md and "- **To:** team@acme.test" in md and "Revenue grew 12% in Q3." in md and "IGNORED" not in md
    news = c.fetch(items[1]).content.decode()
    assert "Office closed Friday." in news and "<p>" not in news and "x()" not in news
    c.close()
    with pytest.raises(ConnectorError, match="login failed"):
        build("email", cfg, "wrong").test()
    with pytest.raises(UnsafeRequest):
        build("email", {**cfg, "host": "10.0.0.8"}, "app-pass").validate()
