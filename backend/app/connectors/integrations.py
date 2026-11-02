"""External integration connectors.

Each connector lists items (``Item``: stable external id, URL, version hint)
and fetches their content. The sync job compares content hashes with the
previous run to add, update or delete documents (deletion propagation).

Implemented here: website crawl, sitemap, RSS/Atom, GitHub (public repos or a
token), S3-compatible buckets (needs boto3). SaaS connectors (Google Drive,
OneDrive, SharePoint, Dropbox, Notion, Confluence, Slack, Teams, GitLab, IMAP
email) live in ``app.connectors.saas``; their OAuth flow in ``app.connectors.oauth``.

Catalog ``status`` is honest: ``tested`` means exercised end to end (the
automated suite, or a recorded live check, see ``verified``); ``beta`` means
implemented and unit-tested against stubbed vendor APIs, not verified live.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from app.core.netguard import check_url
from app.utils.errors import ConnectorError, NotAvailable, UnsafeRequest, ValidationFailed

USER_AGENT = "KnowledgeIntelligenceBot/3.0 (+https://github.com/Safiullah-Rahu/knowledge-intelligence)"
MAX_BYTES = 10 * 1024 * 1024

INTEGRATIONS: dict[str, dict[str, Any]] = {
    "website": {"label": "Website", "category": "web", "status": "tested", "auth": "none", "auth_modes": ["none"], "fields": ["start_url", "max_pages", "max_depth", "path_prefix"], "verified": "Automated suite (crawl, robots.txt, deletion propagation against a local HTTP server)"},
    "sitemap": {"label": "Sitemap", "category": "web", "status": "tested", "auth": "none", "auth_modes": ["none"], "fields": ["sitemap_url", "max_pages", "path_prefix"], "verified": "Automated suite (local HTTP server)"},
    "rss": {"label": "RSS / Atom feed", "category": "web", "status": "tested", "auth": "none", "auth_modes": ["none"], "fields": ["feed_url", "max_items"], "verified": "Automated suite (local HTTP server)"},
    "github": {"label": "GitHub", "category": "development", "status": "beta", "auth": "token (optional for public repos)", "auth_modes": ["token"], "fields": ["repository", "branch", "path_prefix", "extensions"]},
    "s3": {"label": "S3-compatible storage", "category": "storage", "status": "beta", "auth": "access key", "auth_modes": ["token"], "fields": ["bucket", "prefix", "endpoint_url", "region", "access_key_id"]},
    "google_drive": {"label": "Google Drive", "category": "storage", "status": "beta", "auth": "OAuth 2.0", "auth_modes": ["oauth"], "oauth": {"provider": "google", "scopes": ["https://www.googleapis.com/auth/drive.readonly"]}, "fields": ["folder_id", "max_items"]},
    "onedrive": {"label": "Microsoft OneDrive", "category": "storage", "status": "beta", "auth": "OAuth 2.0 (Microsoft Graph)", "auth_modes": ["oauth"], "oauth": {"provider": "microsoft", "scopes": ["offline_access", "User.Read", "Files.Read.All"]}, "fields": ["path", "drive_id", "tenant", "max_items"]},
    "sharepoint": {"label": "SharePoint", "category": "document_management", "status": "beta", "auth": "OAuth 2.0 (Microsoft Graph)", "auth_modes": ["oauth"], "oauth": {"provider": "microsoft", "scopes": ["offline_access", "User.Read", "Sites.Read.All"]}, "fields": ["hostname", "site_path", "library", "path", "tenant", "max_items"]},
    "dropbox": {"label": "Dropbox", "category": "storage", "status": "beta", "auth": "OAuth 2.0", "auth_modes": ["oauth"], "oauth": {"provider": "dropbox", "scopes": ["account_info.read", "files.metadata.read", "files.content.read"]}, "fields": ["path", "max_items"]},
    "notion": {"label": "Notion", "category": "document_management", "status": "beta", "auth": "OAuth 2.0 or internal integration token", "auth_modes": ["oauth", "token"], "oauth": {"provider": "notion", "scopes": []}, "fields": ["max_items"]},
    "confluence": {"label": "Confluence Cloud", "category": "document_management", "status": "beta", "auth": "OAuth 2.0 (Atlassian) or email + API token", "auth_modes": ["oauth", "token"], "oauth": {"provider": "atlassian", "scopes": ["read:page:confluence", "read:space:confluence", "offline_access"]}, "fields": ["space_keys", "site_url", "email", "max_items"]},
    "slack": {"label": "Slack", "category": "collaboration", "status": "beta", "auth": "OAuth 2.0 or bot token (xoxb-)", "auth_modes": ["oauth", "token"], "oauth": {"provider": "slack", "scopes": ["channels:read", "channels:history", "groups:read", "groups:history", "users:read"]}, "fields": ["channels", "lookback_days", "max_items"]},
    "teams": {"label": "Microsoft Teams", "category": "collaboration", "status": "beta", "auth": "OAuth 2.0 (Microsoft Graph)", "auth_modes": ["oauth"], "oauth": {"provider": "microsoft", "scopes": ["offline_access", "User.Read", "Team.ReadBasic.All", "Channel.ReadBasic.All", "ChannelMessage.Read.All"]}, "fields": ["team_id", "channel_ids", "lookback_days", "tenant", "max_items"]},
    "gitlab": {"label": "GitLab", "category": "development", "status": "tested", "auth": "token (optional for public projects)", "auth_modes": ["token"], "fields": ["project", "branch", "path_prefix", "extensions", "base_url", "max_files"], "verified": "Live check 2026-11-02 against public gitlab.com project gitlab-org/project-templates/express without a token (project lookup, recursive tree, raw blob fetch); token and self-managed paths are covered by stubbed tests only"},
    "email": {"label": "Email (IMAP)", "category": "collaboration", "status": "beta", "auth": "IMAP username + password", "auth_modes": ["token"], "fields": ["host", "port", "security", "username", "folder", "lookback_days", "max_items"]},
}


@dataclass
class Item:
    external_id: str
    url: str
    title: str
    filename: str
    version: str | None = None


@dataclass
class Fetched:
    item: Item
    content: bytes
    filename: str


def _client(timeout: float = 20.0) -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=False)


def buffered(r: httpx.Response, body: bytes) -> httpx.Response:
    """Re-wrap an already *decoded* streamed body. The encoding headers must go, or httpx
    would try to gunzip the plain bytes again (DecodingError on every gzip response)."""
    headers = [(k, v) for k, v in r.headers.multi_items() if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")]
    return httpx.Response(r.status_code, headers=headers, content=body, request=r.request)


def safe_get(client: httpx.Client, url: str, *, max_redirects: int = 5, accept: str | None = None) -> httpx.Response:
    """GET that re-validates every redirect hop against the SSRF guard and caps size."""
    for _ in range(max_redirects + 1):
        check_url(url, purpose="web")
        headers = {"Accept": accept} if accept else {}
        with client.stream("GET", url, headers=headers) as r:
            if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                url = urljoin(url, r.headers["location"])
                continue
            body = bytearray()
            for chunk in r.iter_bytes():
                body += chunk
                if len(body) > MAX_BYTES:
                    raise ConnectorError(f"{url} exceeds the {MAX_BYTES // (1024 * 1024)} MB page limit.")
            return buffered(r, bytes(body))
    raise ConnectorError("Too many redirects.")


def _ext_for(content_type: str, url: str) -> str:
    ct = content_type.split(";")[0].strip().lower()
    if ct == "application/pdf" or url.lower().endswith(".pdf"):
        return "pdf"
    if ct in ("text/markdown",) or url.lower().endswith(".md"):
        return "md"
    if ct.startswith("text/plain"):
        return "txt"
    return "html"


def _slug(url: str) -> str:
    p = urlparse(url)
    base = re.sub(r"[^A-Za-z0-9]+", "-", (p.netloc + p.path).strip("/")).strip("-")[:80] or "page"
    return base + "-" + hashlib.sha1(url.encode()).hexdigest()[:8]


class Connector:
    """``save_secret`` persists a rotated credential (refreshed OAuth tokens) from inside a sync."""

    def __init__(self, config: dict[str, Any], secret: str | None = None, save_secret: Callable[[str], None] | None = None) -> None:
        self.config = config
        self.secret = secret
        self.save_secret = save_secret

    def validate(self) -> None: ...
    def test(self) -> dict[str, Any]: ...
    def items(self) -> Iterator[Item]: ...
    def fetch(self, item: Item) -> Fetched: ...
    def close(self) -> None: ...


class WebsiteConnector(Connector):
    def validate(self) -> None:
        url = self.config.get("start_url", "")
        if not url:
            raise ValidationFailed("start_url is required.")
        check_url(url, purpose="web")

    def _robots(self, client: httpx.Client, base: str) -> RobotFileParser:
        rp = RobotFileParser()
        try:
            r = safe_get(client, urljoin(base, "/robots.txt"))
            rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        except (ConnectorError, UnsafeRequest, httpx.HTTPError):
            rp.parse([])
        return rp

    def test(self) -> dict[str, Any]:
        self.validate()
        with _client() as client:
            r = safe_get(client, self.config["start_url"])
        if r.status_code >= 400:
            raise ConnectorError(f"Start URL returned HTTP {r.status_code}.")
        return {"ok": True, "status": r.status_code, "content_type": r.headers.get("content-type", "")}

    def items(self) -> Iterator[Item]:
        from bs4 import BeautifulSoup

        self.validate()
        start = self.config["start_url"]
        host = urlparse(start).netloc
        prefix = self.config.get("path_prefix") or ""
        max_pages = min(int(self.config.get("max_pages", 50)), 2000)
        max_depth = min(int(self.config.get("max_depth", 2)), 6)
        seen: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        with _client() as client:
            robots = self._robots(client, start)
            while queue and len(seen) < max_pages:
                url, depth = queue.popleft()
                url = urldefrag(url)[0]
                if url in seen or not robots.can_fetch(USER_AGENT, url):
                    continue
                seen.add(url)
                try:
                    r = safe_get(client, url)
                except (ConnectorError, UnsafeRequest, httpx.HTTPError):
                    continue
                if r.status_code >= 400:
                    continue
                ctype = r.headers.get("content-type", "")
                ext = _ext_for(ctype, url)
                yield Item(external_id=url, url=url, title=url, filename=f"{_slug(url)}.{ext}", version=hashlib.sha256(r.content).hexdigest())
                self._cache = getattr(self, "_cache", {})
                self._cache[url] = r.content
                if ext == "html" and depth < max_depth:
                    soup = BeautifulSoup(r.text, "lxml")
                    for a in soup.find_all("a", href=True):
                        nxt = urldefrag(urljoin(url, a["href"]))[0]
                        p = urlparse(nxt)
                        if p.scheme in ("http", "https") and p.netloc == host and p.path.startswith(prefix or "/") and nxt not in seen:
                            queue.append((nxt, depth + 1))

    def fetch(self, item: Item) -> Fetched:
        cached = getattr(self, "_cache", {}).pop(item.url, None)
        if cached is None:
            with _client() as client:
                cached = safe_get(client, item.url).content
        return Fetched(item=item, content=cached, filename=item.filename)


class SitemapConnector(WebsiteConnector):
    def validate(self) -> None:
        if not self.config.get("sitemap_url"):
            raise ValidationFailed("sitemap_url is required.")
        check_url(self.config["sitemap_url"], purpose="web")

    def test(self) -> dict[str, Any]:
        self.validate()
        urls = list(self._urls(limit=5))
        return {"ok": True, "sample": urls}

    def _urls(self, limit: int) -> Iterator[str]:
        from lxml import etree

        parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
        pending = [self.config["sitemap_url"]]
        seen_maps: set[str] = set()
        count = 0
        prefix = self.config.get("path_prefix") or ""
        with _client() as client:
            while pending and count < limit:
                sm = pending.pop(0)
                if sm in seen_maps or len(seen_maps) > 50:
                    continue
                seen_maps.add(sm)
                r = safe_get(client, sm)
                if r.status_code >= 400:
                    raise ConnectorError(f"Sitemap returned HTTP {r.status_code}.")
                root = etree.fromstring(r.content, parser)
                for loc in root.iter("{*}loc"):
                    url = (loc.text or "").strip()
                    if not url:
                        continue
                    if root.tag.endswith("sitemapindex"):
                        pending.append(url)
                    elif urlparse(url).path.startswith(prefix or "/"):
                        yield url
                        count += 1
                        if count >= limit:
                            break

    def items(self) -> Iterator[Item]:
        self.validate()
        with _client() as client:
            for url in self._urls(limit=min(int(self.config.get("max_pages", 200)), 5000)):
                try:
                    r = safe_get(client, url)
                except (ConnectorError, UnsafeRequest, httpx.HTTPError):
                    continue
                if r.status_code >= 400:
                    continue
                ext = _ext_for(r.headers.get("content-type", ""), url)
                self._cache = getattr(self, "_cache", {})
                self._cache[url] = r.content
                yield Item(external_id=url, url=url, title=url, filename=f"{_slug(url)}.{ext}", version=hashlib.sha256(r.content).hexdigest())


class RSSConnector(Connector):
    def validate(self) -> None:
        if not self.config.get("feed_url"):
            raise ValidationFailed("feed_url is required.")
        check_url(self.config["feed_url"], purpose="web")

    def _entries(self) -> list[dict[str, str]]:
        from lxml import etree

        self.validate()
        with _client() as client:
            r = safe_get(client, self.config["feed_url"])
        if r.status_code >= 400:
            raise ConnectorError(f"Feed returned HTTP {r.status_code}.")
        parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
        root = etree.fromstring(r.content, parser)
        out = []
        for entry in list(root.iter("{*}item")) + list(root.iter("{*}entry")):
            def text(tag: str, entry=entry) -> str:  # noqa: ANN001
                el = entry.find(f"{{*}}{tag}")
                return (el.text or "").strip() if el is not None and el.text else ""

            link = text("link")
            if not link:
                el = entry.find("{*}link")
                link = el.get("href", "") if el is not None else ""
            body = text("description") or text("content") or text("summary")
            out.append({"id": text("guid") or text("id") or link, "title": text("title") or link, "link": link, "body": body, "date": text("pubDate") or text("updated") or text("published")})
        return out[: min(int(self.config.get("max_items", 100)), 1000)]

    def test(self) -> dict[str, Any]:
        entries = self._entries()
        return {"ok": True, "entries": len(entries), "sample": [e["title"] for e in entries[:3]]}

    def items(self) -> Iterator[Item]:
        self._entries_cache = {e["id"]: e for e in self._entries()}
        for e in self._entries_cache.values():
            html = f"<html><head><title>{e['title']}</title></head><body><h1>{e['title']}</h1><p>Published: {e['date']}</p>{e['body']}</body></html>"
            yield Item(external_id=e["id"], url=e["link"], title=e["title"][:300], filename=f"{_slug(e['id'])}.html", version=hashlib.sha256(html.encode()).hexdigest())

    def fetch(self, item: Item) -> Fetched:
        e = self._entries_cache[item.external_id]
        html = f"<html><head><title>{e['title']}</title></head><body><h1>{e['title']}</h1><p>Published: {e['date']}</p>{e['body']}</body></html>"
        return Fetched(item=item, content=html.encode("utf-8"), filename=item.filename)


class GitHubConnector(Connector):
    API = "https://api.github.com"

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT, "X-GitHub-Api-Version": "2022-11-28"}
        if self.secret:
            h["Authorization"] = f"Bearer {self.secret}"
        return h

    def validate(self) -> None:
        repo = self.config.get("repository", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValidationFailed("repository must look like 'owner/name'.")

    def _get(self, path: str) -> Any:
        r = httpx.get(f"{self.API}{path}", headers=self._headers(), timeout=30)
        if r.status_code == 404:
            raise ConnectorError("Repository not found (or the token lacks access).")
        if r.status_code == 403 and "rate limit" in r.text.lower():
            raise ConnectorError("GitHub API rate limit reached; add a token.")
        if r.status_code >= 400:
            raise ConnectorError(f"GitHub API HTTP {r.status_code}.")
        return r.json()

    def test(self) -> dict[str, Any]:
        self.validate()
        data = self._get(f"/repos/{self.config['repository']}")
        return {"ok": True, "default_branch": data.get("default_branch"), "private": data.get("private")}

    def items(self) -> Iterator[Item]:
        self.validate()
        repo = self.config["repository"]
        branch = self.config.get("branch") or self._get(f"/repos/{repo}")["default_branch"]
        tree = self._get(f"/repos/{repo}/git/trees/{branch}?recursive=1")
        exts = {e.lower().lstrip(".") for e in (self.config.get("extensions") or ["md", "mdx", "rst", "txt", "py", "ts", "js", "java", "go", "rs", "yaml", "yml", "json", "sql"])}
        prefix = self.config.get("path_prefix") or ""
        limit = min(int(self.config.get("max_files", 500)), 5000)
        n = 0
        for entry in tree.get("tree", []):
            if entry.get("type") != "blob" or not entry["path"].startswith(prefix) or entry.get("size", 0) > MAX_BYTES:
                continue
            ext = entry["path"].rsplit(".", 1)[-1].lower() if "." in entry["path"] else ""
            if ext not in exts:
                continue
            filename = entry["path"].replace("/", "__")
            if ext == "mdx":
                filename = filename[:-1]
            yield Item(external_id=entry["path"], url=f"https://github.com/{repo}/blob/{branch}/{entry['path']}", title=entry["path"], filename=filename, version=entry["sha"])
            n += 1
            if n >= limit:
                break

    def fetch(self, item: Item) -> Fetched:
        repo = self.config["repository"]
        r = httpx.get(f"{self.API}/repos/{repo}/git/blobs/{item.version}", headers={**self._headers(), "Accept": "application/vnd.github.raw+json"}, timeout=60)
        if r.status_code >= 400:
            raise ConnectorError(f"Could not fetch {item.external_id} (HTTP {r.status_code}).")
        return Fetched(item=item, content=r.content, filename=item.filename)


class S3Connector(Connector):
    def _client(self) -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise NotAvailable("S3 integrations need boto3 on the worker (pip extra 's3').") from exc
        endpoint = self.config.get("endpoint_url")
        if endpoint:
            check_url(endpoint, purpose="database")
        return boto3.client("s3", endpoint_url=endpoint or None, region_name=self.config.get("region") or "us-east-1", aws_access_key_id=self.config.get("access_key_id"), aws_secret_access_key=self.secret)

    def validate(self) -> None:
        if not self.config.get("bucket"):
            raise ValidationFailed("bucket is required.")

    def test(self) -> dict[str, Any]:
        self.validate()
        try:
            self._client().head_bucket(Bucket=self.config["bucket"])
        except NotAvailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ConnectorError(f"Bucket check failed: {type(exc).__name__}") from exc
        return {"ok": True}

    def items(self) -> Iterator[Item]:
        from app.ingestion.detect import SUPPORTED_EXT, extension_of

        self.validate()
        client = self._client()
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.config["bucket"], Prefix=self.config.get("prefix") or ""):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/") or extension_of(key) not in SUPPORTED_EXT or obj["Size"] > MAX_BYTES * 10:
                    continue
                yield Item(external_id=key, url=f"s3://{self.config['bucket']}/{key}", title=key, filename=key.replace("/", "__"), version=obj.get("ETag", "").strip('"'))

    def fetch(self, item: Item) -> Fetched:
        body = self._client().get_object(Bucket=self.config["bucket"], Key=item.external_id)["Body"].read()
        return Fetched(item=item, content=body, filename=item.filename)


CONNECTORS: dict[str, type[Connector]] = {"website": WebsiteConnector, "sitemap": SitemapConnector, "rss": RSSConnector, "github": GitHubConnector, "s3": S3Connector}


def build(provider: str, config: dict[str, Any], secret: str | None, save_secret: Callable[[str], None] | None = None) -> Connector:
    from app.connectors.saas import SAAS_CONNECTORS

    meta = INTEGRATIONS.get(provider)
    if meta is None:
        raise ValidationFailed(f"Unknown integration '{provider}'.")
    cls = CONNECTORS.get(provider) or SAAS_CONNECTORS.get(provider)
    if cls is None:
        raise NotAvailable(f"{meta['label']} is planned but not implemented yet ({meta['auth']} required).")
    return cls(config, secret, save_secret)


def polite_delay() -> None:
    time.sleep(0.05)
