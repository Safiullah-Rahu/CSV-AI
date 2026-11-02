"""SaaS connectors: Google Drive, OneDrive, SharePoint, Dropbox, Notion,
Confluence Cloud, Slack, Microsoft Teams, GitLab and email (IMAP).

Every connector lists items with a stable ``external_id`` and a ``version``
hint (vendor content hash, etag, revision or edit time) so incremental syncs
only fetch what changed, pages through the vendor API up to ``max_items``, and
goes through :class:`VendorHTTP` (timeouts, ``Retry-After``, ``MAX_BYTES``,
safe redirects). User-supplied hosts (GitLab ``base_url``, Confluence
``site_url``, IMAP ``host``) pass the SSRF guard. OAuth tokens come from
``app.connectors.oauth`` and are refreshed (and persisted) on expiry or a 401.
"""

from __future__ import annotations

import base64
import email
import email.policy
import hashlib
import html
import imaplib
import json
import re
import ssl
from collections import defaultdict, deque
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

import httpx

from app.connectors import integrations as _ic
from app.connectors import oauth
from app.connectors.integrations import INTEGRATIONS, Connector, Fetched, Item
from app.connectors.vendor_http import VendorHTTP, vendor_error
from app.core.netguard import check_host, check_url
from app.utils.errors import ConnectorError, ValidationFailed

GRAPH = "https://graph.microsoft.com/v1.0"
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


# ----------------------------------------------------------------------------- helpers


def _cap(config: dict[str, Any], key: str = "max_items", default: int = 500, hard: int = 5000) -> int:
    try:
        n = int(config.get(key) or default)
    except (TypeError, ValueError) as exc:
        raise ValidationFailed(f"{key} must be a whole number.") from exc
    return max(1, min(n, hard))


def _days(config: dict[str, Any], default: int = 30) -> int:
    try:
        n = int(config.get("lookback_days") or default)
    except (TypeError, ValueError) as exc:
        raise ValidationFailed("lookback_days must be a whole number.") from exc
    if not 1 <= n <= 365:
        raise ValidationFailed("lookback_days must be between 1 and 365.")
    return n


def _list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = re.split(r"[,\s]+", value)
    return [str(v).strip() for v in (value or []) if str(v).strip()]


def _safe_name(name: str, fallback: str = "item") -> str:
    base = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", name or "").strip(" .")[:150]
    return base or fallback


def _short(value: str) -> str:
    return hashlib.sha1(value.encode()).hexdigest()[:8]


def _supported(filename: str) -> bool:
    from app.ingestion.detect import SUPPORTED_EXT, extension_of

    return extension_of(filename) in SUPPORTED_EXT


def _html_text(markup: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(markup or "", "lxml")
    for tag in soup(["script", "style", "head", "title"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(["p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "table", "ul", "ol", "section", "article"]):
        block.insert_after("\n")  # block elements end a line; inline ones (b, a, span) do not
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in soup.get_text().splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _sha(parts: list[str]) -> str:
    return hashlib.sha256("|".join(sorted(parts)).encode()).hexdigest()


def _iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


# --------------------------------------------------------------------------- base class


class SaaSConnector(Connector):
    """Bearer-token HTTP connector: an OAuth token set, or a vendor token in token mode."""

    key = ""
    vendor = "Service"
    user_host = False  # True when the API host comes from the configuration (SSRF-checked)

    def __init__(self, config: dict[str, Any], secret: str | None = None, save_secret: Any = None) -> None:
        super().__init__(config, secret, save_secret)
        self.meta = INTEGRATIONS.get(self.key, {})
        self.http = VendorHTTP(check_hosts=self.user_host)
        self._tokens = oauth.parse(secret) if self.auth_mode == "oauth" else None

    @property
    def auth_mode(self) -> str:
        modes = self.meta.get("auth_modes") or ["token"]
        mode = self.config.get("auth_mode") or modes[0]
        if mode not in modes:
            raise ValidationFailed(f"auth_mode must be one of: {', '.join(modes)}.")
        return mode

    def validate(self) -> None:
        _ = self.auth_mode
        if "max_items" in self.config:
            _cap(self.config)

    def close(self) -> None:
        self.http.close()

    def _access_token(self, force: bool = False) -> str:
        if self.auth_mode != "oauth":
            if not self.secret:
                raise ConnectorError(f"No {self.vendor} token is stored for this integration.")
            return self.secret
        tokens = self._tokens
        if tokens is None:
            raise ConnectorError(f"Not authorized yet: use “Connect with {oauth.PROVIDERS[self.meta['oauth']['provider']].vendor}” to grant access.")
        if force or oauth.expired(tokens):
            tokens = oauth.refresh(tokens)
            self._tokens, self.secret = tokens, oauth.dump(tokens)
            if self.save_secret:
                self.save_secret(self.secret)
        return tokens["access_token"]

    def _auth_headers(self, force: bool = False) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token(force)}"}

    def _call(self, method: str, url: str, *, allow: tuple[int, ...] = (), headers: dict[str, str] | None = None, **kw: Any) -> httpx.Response:
        r = self.http.request(method, url, headers={**self._auth_headers(), **(headers or {})}, **kw)
        if r.status_code == 401 and self._tokens and self._tokens.get("refresh_token"):
            r = self.http.request(method, url, headers={**self._auth_headers(force=True), **(headers or {})}, **kw)
        if r.status_code in allow or r.status_code < 400:
            return r
        raise ConnectorError(vendor_error(r, self.vendor))

    def _json(self, method: str, url: str, **kw: Any) -> Any:
        r = self._call(method, url, **kw)
        try:
            return r.json()
        except ValueError as exc:
            raise ConnectorError(f"{self.vendor} returned a response that is not JSON.") from exc


# ------------------------------------------------------------------------- Google Drive


class GoogleDriveConnector(SaaSConnector):
    key = "google_drive"
    vendor = "Google Drive"
    API = "https://www.googleapis.com/drive/v3"
    FOLDER = "application/vnd.google-apps.folder"
    EXPORT = {
        "application/vnd.google-apps.document": ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        "application/vnd.google-apps.spreadsheet": ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        "application/vnd.google-apps.presentation": ("pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    }
    FIELDS = "nextPageToken,files(id,name,mimeType,md5Checksum,version,modifiedTime,size,webViewLink)"

    def validate(self) -> None:
        super().validate()
        folder = self.config.get("folder_id")
        if folder and not re.fullmatch(r"[A-Za-z0-9_-]{5,200}", str(folder)):
            raise ValidationFailed("folder_id must be a Google Drive folder id (the last part of the folder URL).")

    def test(self) -> dict[str, Any]:
        self.validate()
        about = self._json("GET", f"{self.API}/about", params={"fields": "user(displayName,emailAddress)"})
        out: dict[str, Any] = {"ok": True, "account": (about.get("user") or {}).get("emailAddress")}
        if self.config.get("folder_id"):
            f = self._json("GET", f"{self.API}/files/{self.config['folder_id']}", params={"fields": "id,name,mimeType", "supportsAllDrives": "true"})
            if f.get("mimeType") != self.FOLDER:
                raise ConnectorError("folder_id does not point to a folder.")
            out["folder"] = f.get("name")
        return out

    def _files(self, q: str) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] = {"q": q, "fields": self.FIELDS, "pageSize": 200, "supportsAllDrives": "true", "includeItemsFromAllDrives": "true", "corpora": "allDrives", "orderBy": "modifiedTime desc"}
        while True:
            data = self._json("GET", f"{self.API}/files", params=params)
            yield from data.get("files", [])
            token = data.get("nextPageToken")
            if not token:
                return
            params = {**params, "pageToken": token}

    def items(self) -> Iterator[Item]:
        self.validate()
        limit = _cap(self.config)
        self._mime: dict[str, str] = {}
        folder = self.config.get("folder_id")
        queue: deque[tuple[str | None, str]] = deque([(folder, "")])
        n, folders = 0, 0
        while queue:
            parent, prefix = queue.popleft()
            folders += 1
            if folders > 2000:
                return
            q = f"'{parent}' in parents and trashed = false" if parent else f"trashed = false and mimeType != '{self.FOLDER}'"
            for f in self._files(q):
                mime, name = f.get("mimeType", ""), f.get("name", "")
                if mime == self.FOLDER:
                    if parent:
                        queue.append((f["id"], f"{prefix}{name}/"))
                    continue
                if mime in self.EXPORT:
                    filename, version = f"{_safe_name(name)}.{self.EXPORT[mime][0]}", f.get("version") or f.get("modifiedTime")
                elif mime.startswith("application/vnd.google-apps."):
                    continue  # forms, drawings, shortcuts, sites: nothing to export
                else:
                    if int(f.get("size") or 0) > _ic.MAX_BYTES or not _supported(name):
                        continue
                    filename, version = _safe_name(name), f.get("md5Checksum") or f.get("version") or f.get("modifiedTime")
                self._mime[f["id"]] = mime
                yield Item(external_id=f["id"], url=f.get("webViewLink", ""), title=f"{prefix}{name}", filename=filename, version=version)
                n += 1
                if n >= limit:
                    return

    def fetch(self, item: Item) -> Fetched:
        mime = getattr(self, "_mime", {}).get(item.external_id)
        if mime is None:
            mime = self._json("GET", f"{self.API}/files/{item.external_id}", params={"fields": "mimeType", "supportsAllDrives": "true"}).get("mimeType", "")
        if mime in self.EXPORT:
            r = self._call("GET", f"{self.API}/files/{item.external_id}/export", params={"mimeType": self.EXPORT[mime][1]})
        else:
            r = self._call("GET", f"{self.API}/files/{item.external_id}", params={"alt": "media", "supportsAllDrives": "true"})
        return Fetched(item=item, content=r.content, filename=item.filename)


# ------------------------------------------------------------------- OneDrive / SharePoint


class _GraphDrive(SaaSConnector):
    vendor = "Microsoft Graph"
    SELECT = "id,name,size,file,folder,eTag,cTag,webUrl,lastModifiedDateTime"

    def validate(self) -> None:
        super().validate()
        oauth.tenant_for(self.config)
        path = self.config.get("path") or ""
        if any(c in path for c in ("?", "#", ":", "\\")) or ".." in path.split("/"):
            raise ValidationFailed("path must be a folder path such as /Documents/Handbook.")

    def _pages(self, url: str, params: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        next_url: str | None = url
        while next_url:
            data = self._json("GET", next_url, params=params)
            yield from data.get("value", [])
            next_url, params = data.get("@odata.nextLink"), None
            if next_url and not next_url.startswith(GRAPH):
                raise ConnectorError("Unexpected pagination link from Microsoft Graph.")

    def _folder(self, drive_id: str) -> dict[str, Any]:
        path = (self.config.get("path") or "").strip("/")
        url = f"{GRAPH}/drives/{drive_id}/root" + (f":/{quote(path)}" if path else "")
        item = self._json("GET", url, params={"$select": "id,name,folder,webUrl"})
        if "folder" not in item:
            raise ConnectorError(f"/{path} is not a folder.")
        return item

    def _walk(self, drive_id: str, root_id: str, limit: int) -> Iterator[Item]:
        queue: deque[tuple[str, str]] = deque([(root_id, "")])
        folders = 0
        while queue and self._count < limit:
            item_id, prefix = queue.popleft()
            folders += 1
            if folders > 2000:
                return
            for child in self._pages(f"{GRAPH}/drives/{drive_id}/items/{item_id}/children", {"$top": 200, "$select": self.SELECT}):
                name = child.get("name", "")
                if "folder" in child:
                    queue.append((child["id"], f"{prefix}{name}/"))
                    continue
                if "file" not in child or int(child.get("size") or 0) > _ic.MAX_BYTES or not _supported(name):
                    continue
                yield Item(external_id=f"{drive_id}:{child['id']}", url=child.get("webUrl", ""), title=f"{prefix}{name}", filename=_safe_name(name), version=child.get("cTag") or child.get("eTag") or child.get("lastModifiedDateTime"))
                self._count += 1
                if self._count >= limit:
                    return

    def fetch(self, item: Item) -> Fetched:
        drive_id, item_id = item.external_id.split(":", 1)
        r = self._call("GET", f"{GRAPH}/drives/{drive_id}/items/{item_id}/content")  # 302 to a pre-signed URL; auth is dropped on the hop
        return Fetched(item=item, content=r.content, filename=item.filename)

    def _me(self) -> str | None:
        me = self._json("GET", f"{GRAPH}/me", params={"$select": "displayName,userPrincipalName"})
        return me.get("userPrincipalName") or me.get("displayName")


class OneDriveConnector(_GraphDrive):
    key = "onedrive"
    vendor = "OneDrive"

    def validate(self) -> None:
        super().validate()
        drive = self.config.get("drive_id")
        if drive and not re.fullmatch(r"[A-Za-z0-9!_\-]{3,200}", str(drive)):
            raise ValidationFailed("drive_id must be a Graph drive id.")

    def _drive(self) -> dict[str, Any]:
        drive = self.config.get("drive_id")
        return self._json("GET", f"{GRAPH}/drives/{drive}" if drive else f"{GRAPH}/me/drive", params={"$select": "id,driveType,name"})

    def test(self) -> dict[str, Any]:
        self.validate()
        account = self._me()
        drive = self._drive()
        folder = self._folder(drive["id"])
        return {"ok": True, "account": account, "drive": drive.get("driveType"), "folder": folder.get("name")}

    def items(self) -> Iterator[Item]:
        self.validate()
        self._count = 0
        drive = self._drive()
        yield from self._walk(drive["id"], self._folder(drive["id"])["id"], _cap(self.config))


class SharePointConnector(_GraphDrive):
    key = "sharepoint"
    vendor = "SharePoint"

    def validate(self) -> None:
        super().validate()
        host = str(self.config.get("hostname") or "").strip().lower()
        if not re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", host):
            raise ValidationFailed("hostname must look like contoso.sharepoint.com.")
        site = str(self.config.get("site_path") or "")
        if site and not re.fullmatch(r"/(sites|teams|personal)/[A-Za-z0-9_.\-/ ]+", site):
            raise ValidationFailed("site_path must look like /sites/Handbook (empty for the root site).")

    def _site(self) -> dict[str, Any]:
        host = str(self.config["hostname"]).strip().lower()
        site = str(self.config.get("site_path") or "").rstrip("/")
        return self._json("GET", f"{GRAPH}/sites/{host}" + (f":{quote(site)}" if site else ""), params={"$select": "id,displayName,webUrl"})

    def _drives(self, site: dict[str, Any]) -> list[dict[str, Any]]:
        drives = [d for d in self._pages(f"{GRAPH}/sites/{site['id']}/drives", {"$select": "id,name,driveType,webUrl"}) if d.get("driveType") == "documentLibrary"]
        lib = (self.config.get("library") or "").strip().lower()
        if lib:
            drives = [d for d in drives if (d.get("name") or "").lower() == lib]
            if not drives:
                raise ConnectorError(f"No document library named '{self.config['library']}' in this site.")
        return drives

    def test(self) -> dict[str, Any]:
        self.validate()
        site = self._site()
        drives = self._drives(site)
        return {"ok": True, "account": self._me(), "site": site.get("displayName"), "libraries": [d.get("name") for d in drives]}

    def items(self) -> Iterator[Item]:
        self.validate()
        self._count = 0
        limit = _cap(self.config)
        for drive in self._drives(self._site()):
            if self._count >= limit:
                return
            try:
                root = self._folder(drive["id"])
            except ConnectorError:
                if self.config.get("library"):
                    raise
                continue  # the path does not exist in this library
            yield from self._walk(drive["id"], root["id"], limit)


# ------------------------------------------------------------------------------ Dropbox


class DropboxConnector(SaaSConnector):
    key = "dropbox"
    vendor = "Dropbox"
    API = "https://api.dropboxapi.com/2"
    CONTENT = "https://content.dropboxapi.com/2"

    def _path(self) -> str:
        path = (self.config.get("path") or "").strip().rstrip("/")
        return "/" + path.lstrip("/") if path else ""

    def validate(self) -> None:
        super().validate()
        if "\\" in self._path() or ".." in self._path().split("/"):
            raise ValidationFailed("path must be a Dropbox folder path such as /Team/Handbook.")

    def test(self) -> dict[str, Any]:
        self.validate()
        acct = self._json("POST", f"{self.API}/users/get_current_account", content=b"null", headers={"Content-Type": "application/json"})
        self._json("POST", f"{self.API}/files/list_folder", json={"path": self._path(), "limit": 1})
        return {"ok": True, "account": acct.get("email"), "folder": self._path() or "/"}

    def items(self) -> Iterator[Item]:
        self.validate()
        limit, n = _cap(self.config), 0
        data = self._json("POST", f"{self.API}/files/list_folder", json={"path": self._path(), "recursive": True, "include_deleted": False, "include_non_downloadable_files": False, "limit": 2000})
        while True:
            for e in data.get("entries", []):
                name = e.get("name", "")
                if e.get(".tag") != "file" or int(e.get("size") or 0) > _ic.MAX_BYTES or not _supported(name) or e.get("is_downloadable") is False:
                    continue
                shown = e.get("path_display") or name
                folder = shown.rsplit("/", 1)[0]
                yield Item(external_id=e["id"], url=f"https://www.dropbox.com/home{quote(folder)}?preview={quote(name)}", title=shown.lstrip("/"), filename=_safe_name(name), version=e.get("content_hash") or e.get("rev"))
                n += 1
                if n >= limit:
                    return
            if not data.get("has_more") or not data.get("cursor"):
                return
            data = self._json("POST", f"{self.API}/files/list_folder/continue", json={"cursor": data["cursor"]})

    def fetch(self, item: Item) -> Fetched:
        r = self._call("POST", f"{self.CONTENT}/files/download", headers={"Dropbox-API-Arg": json.dumps({"path": item.external_id})})
        return Fetched(item=item, content=r.content, filename=item.filename)


# ------------------------------------------------------------------------------- Notion


def _rich(rich: list[dict[str, Any]] | None) -> str:
    out = []
    for t in rich or []:
        s = t.get("plain_text", "")
        a = t.get("annotations") or {}
        if s.strip():
            if a.get("code"):
                s = f"`{s}`"
            if a.get("bold"):
                s = f"**{s}**"
            if a.get("italic"):
                s = f"*{s}*"
            if a.get("strikethrough"):
                s = f"~~{s}~~"
            if t.get("href"):
                s = f"[{s}]({t['href']})"
        out.append(s)
    return "".join(out)


class NotionConnector(SaaSConnector):
    key = "notion"
    vendor = "Notion"
    API = "https://api.notion.com/v1"
    VERSION = "2022-06-28"
    MAX_BLOCKS = 3000

    def validate(self) -> None:
        super().validate()
        if self.auth_mode == "token" and self.secret and not re.match(r"^(secret_|ntn_)", self.secret):
            raise ValidationFailed("A Notion internal integration token starts with 'ntn_' or 'secret_'.")

    def _auth_headers(self, force: bool = False) -> dict[str, str]:
        return {**super()._auth_headers(force), "Notion-Version": self.VERSION}

    def test(self) -> dict[str, Any]:
        self.validate()
        me = self._json("GET", f"{self.API}/users/me")
        found = self._json("POST", f"{self.API}/search", json={"filter": {"property": "object", "value": "page"}, "page_size": 5})
        return {"ok": True, "account": (me.get("bot") or {}).get("workspace_name") or me.get("name"), "sample": [self._title(p) for p in found.get("results", [])]}

    @staticmethod
    def _title(page: dict[str, Any]) -> str:
        for prop in (page.get("properties") or {}).values():
            if isinstance(prop, dict) and prop.get("type") == "title":
                return "".join(t.get("plain_text", "") for t in prop.get("title", [])).strip() or "Untitled"
        return "Untitled"

    def items(self) -> Iterator[Item]:
        self.validate()
        limit, n = _cap(self.config), 0
        self._titles: dict[str, str] = {}
        body: dict[str, Any] = {"filter": {"property": "object", "value": "page"}, "page_size": 100, "sort": {"direction": "descending", "timestamp": "last_edited_time"}}
        while True:
            data = self._json("POST", f"{self.API}/search", json=body)
            for page in data.get("results", []):
                if page.get("archived") or page.get("in_trash"):
                    continue
                title = self._title(page)
                self._titles[page["id"]] = title
                yield Item(external_id=page["id"], url=page.get("url", ""), title=title, filename=f"{_safe_name(title, 'page')}-{page['id'].replace('-', '')[:8]}.md", version=page.get("last_edited_time"))
                n += 1
                if n >= limit:
                    return
            if not data.get("has_more") or not data.get("next_cursor"):
                return
            body = {**body, "start_cursor": data["next_cursor"]}

    def _children(self, block_id: str) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] = {"page_size": 100}
        while True:
            data = self._json("GET", f"{self.API}/blocks/{block_id}/children", params=params)
            yield from data.get("results", [])
            if not data.get("has_more") or not data.get("next_cursor"):
                return
            params = {"page_size": 100, "start_cursor": data["next_cursor"]}

    def _render(self, block_id: str, depth: int, lines: list[str]) -> None:
        if depth > 6:
            return
        rows = 0
        for b in self._children(block_id):
            self._blocks += 1
            if self._blocks > self.MAX_BLOCKS:
                return
            t = b.get("type", "")
            d = b.get(t) or {}
            ind = "  " * depth
            text = _rich(d.get("rich_text"))
            nested = depth + 1
            if t in ("heading_1", "heading_2", "heading_3"):
                line, nested = "#" * (int(t[-1]) + 1) + " " + text, depth
            elif t == "bulleted_list_item" or t == "toggle":
                line = f"{ind}- {text}"
            elif t == "numbered_list_item":
                line = f"{ind}1. {text}"
            elif t == "to_do":
                line = f"{ind}- [{'x' if d.get('checked') else ' '}] {text}"
            elif t in ("quote", "callout"):
                line, nested = f"> {text}", depth
            elif t == "code":
                line, nested = f"```{d.get('language', '')}\n{''.join(x.get('plain_text', '') for x in d.get('rich_text', []))}\n```", depth
            elif t == "divider":
                line = "---"
            elif t == "equation":
                line = f"$${d.get('expression', '')}$$"
            elif t == "child_page":
                line = f"{ind}- Sub-page: {d.get('title', '')}"
            elif t == "child_database":
                line = f"{ind}- Database: {d.get('title', '')}"
            elif t in ("bookmark", "embed", "link_preview"):
                line = f"<{d.get('url', '')}>" if d.get("url") else ""
            elif t in ("image", "file", "pdf", "video", "audio"):
                cap = _rich(d.get("caption"))
                line = f"[{t}{': ' + cap if cap else ''}]"
            elif t == "table_row":
                line = "| " + " | ".join(_rich(c).replace("|", "\\|") for c in d.get("cells", [])) + " |"
                rows += 1
                if rows == 1:
                    line += "\n|" + " --- |" * len(d.get("cells", []))
                elif lines:
                    lines[-1] += "\n" + line  # rows of one table stay on consecutive lines
                    line = ""
            elif t == "table":
                line, nested = "", depth
            else:
                line, nested = (f"{ind}{text}" if text else ""), depth
            if line:
                lines.append(line)
            if b.get("has_children") and t not in ("child_page", "child_database"):
                self._render(b["id"], nested, lines)

    def fetch(self, item: Item) -> Fetched:
        self._blocks = 0
        title = getattr(self, "_titles", {}).get(item.external_id) or item.title
        lines: list[str] = [f"# {title}", ""]
        self._render(item.external_id, 0, lines)
        if self._blocks > self.MAX_BLOCKS:
            lines.append(f"\n_(truncated after {self.MAX_BLOCKS} blocks)_")
        return Fetched(item=item, content=("\n\n".join(x for x in lines if x is not None) + "\n").encode("utf-8"), filename=item.filename)


# --------------------------------------------------------------------------- Confluence


class ConfluenceConnector(SaaSConnector):
    key = "confluence"
    vendor = "Confluence"

    def __init__(self, config: dict[str, Any], secret: str | None = None, save_secret: Any = None) -> None:
        self.user_host = (config.get("auth_mode") or "oauth") == "token"
        super().__init__(config, secret, save_secret)

    def validate(self) -> None:
        super().validate()
        keys = _list(self.config.get("space_keys"))
        if not keys or not all(re.fullmatch(r"[A-Za-z0-9_~\-]{1,255}", k) for k in keys):
            raise ValidationFailed("space_keys must list one or more Confluence space keys (e.g. ENG, HR).")
        if self.auth_mode == "token":
            site = (self.config.get("site_url") or "").rstrip("/")
            if not re.fullmatch(r"https://[A-Za-z0-9.\-]+(:\d+)?", site):
                raise ValidationFailed("site_url must be your Confluence Cloud address, e.g. https://acme.atlassian.net.")
            check_url(site, purpose="integration", schemes=("https",))
            if not re.fullmatch(r"[^@\s]+@[^@\s]+", self.config.get("email") or ""):
                raise ValidationFailed("email is required with an API token (the Atlassian account the token belongs to).")

    def _root(self) -> str:
        if self.auth_mode == "token":
            return self.config["site_url"].rstrip("/")
        cloud = ((self._tokens or {}).get("extras") or {}).get("cloud_id")
        if not cloud:
            raise ConnectorError("Not authorized yet: use “Connect with Atlassian” to grant access.")
        return f"https://api.atlassian.com/ex/confluence/{cloud}"

    def _site(self) -> str:
        if self.auth_mode == "token":
            return self.config["site_url"].rstrip("/")
        return ((self._tokens or {}).get("extras") or {}).get("site_url", "")

    def _auth_headers(self, force: bool = False) -> dict[str, str]:
        if self.auth_mode == "token":
            if not self.secret:
                raise ConnectorError("No Confluence API token is stored for this integration.")
            raw = f"{self.config.get('email', '')}:{self.secret}".encode()
            return {"Authorization": "Basic " + base64.b64encode(raw).decode("ascii"), "Accept": "application/json"}
        return {**super()._auth_headers(force), "Accept": "application/json"}

    def _spaces(self) -> list[dict[str, Any]]:
        keys = _list(self.config.get("space_keys"))
        data = self._json("GET", f"{self._root()}/wiki/api/v2/spaces", params={"keys": ",".join(keys), "limit": 250})
        return data.get("results", [])

    def _next(self, link: str | None) -> str | None:
        if not link:
            return None
        if link.startswith("/wiki/"):
            return self._root() + link
        if link.startswith("/"):
            return self._root() + "/wiki" + link
        return None  # absolute links to another host are never followed

    def test(self) -> dict[str, Any]:
        self.validate()
        spaces = self._spaces()
        found = {s.get("key") for s in spaces}
        missing = [k for k in _list(self.config.get("space_keys")) if k not in found]
        if missing:
            raise ConnectorError(f"Space(s) not found or not visible to this account: {', '.join(missing)}.")
        return {"ok": True, "site": self._site(), "spaces": [s.get("name") for s in spaces]}

    def items(self) -> Iterator[Item]:
        self.validate()
        limit, n = _cap(self.config), 0
        site = self._site()
        for space in self._spaces():
            url: str | None = f"{self._root()}/wiki/api/v2/spaces/{space['id']}/pages"
            params: dict[str, Any] | None = {"limit": 250, "status": "current"}
            while url:
                data = self._json("GET", url, params=params)
                for page in data.get("results", []):
                    title = page.get("title") or "Untitled"
                    webui = (page.get("_links") or {}).get("webui", "")
                    yield Item(external_id=str(page["id"]), url=f"{site}/wiki{webui}" if site and webui else "", title=f"{space.get('key')}: {title}", filename=f"{_safe_name(title, 'page')}-{page['id']}.html", version=str((page.get("version") or {}).get("number", "")) or None)
                    n += 1
                    if n >= limit:
                        return
                url, params = self._next((data.get("_links") or {}).get("next")), None

    def fetch(self, item: Item) -> Fetched:
        page = self._json("GET", f"{self._root()}/wiki/api/v2/pages/{item.external_id}", params={"body-format": "storage"})
        title = html.escape(page.get("title") or item.title)
        body = ((page.get("body") or {}).get("storage") or {}).get("value", "")
        doc = f"<html><head><title>{title}</title></head><body><h1>{title}</h1>{body}</body></html>"
        return Fetched(item=item, content=doc.encode("utf-8"), filename=item.filename)


# -------------------------------------------------------------------------------- Slack


_SLACK_SKIP = {"channel_join", "channel_leave", "channel_topic", "channel_purpose", "channel_name", "channel_archive", "channel_unarchive", "group_join", "group_leave", "bot_add", "bot_remove", "pinned_item", "unpinned_item"}


class SlackConnector(SaaSConnector):
    key = "slack"
    vendor = "Slack"
    API = "https://slack.com/api"

    def validate(self) -> None:
        super().validate()
        _days(self.config)
        if self.auth_mode == "token" and self.secret and not self.secret.startswith("xoxb-"):
            raise ValidationFailed("Use a Slack bot token (it starts with 'xoxb-').")

    def _api(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        data = self._json("GET", f"{self.API}/{method}", params=params)
        if not data.get("ok"):
            raise ConnectorError(f"Slack {method} failed: {data.get('error', 'unknown error')}.")
        return data

    def _all_channels(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for types in ("public_channel,private_channel", "public_channel"):
            params: dict[str, Any] = {"types": types, "exclude_archived": "true", "limit": 200}
            try:
                for _ in range(50):
                    data = self._api("conversations.list", params)
                    out.extend(data.get("channels", []))
                    cursor = (data.get("response_metadata") or {}).get("next_cursor")
                    if not cursor:
                        break
                    params = {**params, "cursor": cursor}
                return out
            except ConnectorError as exc:
                if "missing_scope" not in exc.message or types == "public_channel":
                    raise
                out = []
        return out

    def _channels(self) -> list[dict[str, Any]]:
        wanted = [c.lstrip("#") for c in _list(self.config.get("channels"))]
        channels = self._all_channels()
        if not wanted:
            return [c for c in channels if c.get("is_member")]
        picked = [c for c in channels if c.get("id") in wanted or c.get("name") in wanted]
        missing = [w for w in wanted if not any(w in (c.get("id"), c.get("name")) for c in picked)]
        if missing:
            raise ConnectorError(f"Channel(s) not found or not visible to the bot: {', '.join(missing)}. Invite the app to private channels first.")
        return picked

    def test(self) -> dict[str, Any]:
        self.validate()
        me = self._api("auth.test")
        channels = self._channels()
        return {"ok": True, "account": me.get("team"), "bot": me.get("user"), "channels": [f"#{c.get('name')}" for c in channels][:20]}

    def _user(self, uid: str | None) -> str:
        if not uid:
            return "unknown"
        if uid not in self._users:
            name = uid
            if len(self._users) < 500:
                try:
                    u = self._api("users.info", {"user": uid}).get("user") or {}
                    name = (u.get("profile") or {}).get("display_name") or u.get("real_name") or u.get("name") or uid
                except ConnectorError:
                    pass
            self._users[uid] = name
        return self._users[uid]

    def _text(self, text: str) -> str:
        text = re.sub(r"<@([UW][A-Z0-9]+)(\|[^>]*)?>", lambda m: "@" + self._user(m.group(1)), text or "")
        text = re.sub(r"<#[CG][A-Z0-9]+\|([^>]*)>", r"#\1", text)
        text = re.sub(r"<!(here|channel|everyone)[^>]*>", r"@\1", text)
        text = re.sub(r"<(https?://[^|>]+)\|([^>]+)>", r"[\2](\1)", text)
        text = re.sub(r"<(https?://[^>]+)>", r"\1", text)
        return html.unescape(text)

    def items(self) -> Iterator[Item]:
        self.validate()
        limit, n = _cap(self.config), 0
        cutoff = datetime.now(UTC) - timedelta(days=_days(self.config))
        self._users: dict[str, str] = {}
        self._docs: dict[str, bytes] = {}
        for ch in self._channels():
            days: dict[str, list[dict[str, Any]]] = defaultdict(list)
            params: dict[str, Any] = {"channel": ch["id"], "oldest": f"{cutoff.timestamp():.6f}", "limit": 200}
            for _ in range(100):
                data = self._api("conversations.history", params)
                for m in data.get("messages", []):
                    if m.get("subtype") in _SLACK_SKIP or not m.get("ts"):
                        continue
                    days[datetime.fromtimestamp(float(m["ts"]), UTC).date().isoformat()].append(m)
                cursor = (data.get("response_metadata") or {}).get("next_cursor")
                if not data.get("has_more") or not cursor:
                    break
                params = {**params, "cursor": cursor}
            name = ch.get("name") or ch["id"]
            for day in sorted(days, reverse=True):
                msgs = sorted(days[day], key=lambda m: float(m["ts"]))
                lines = [f"# #{name} — {day}", "", f"Slack channel #{name}, messages from {day} (UTC).", ""]
                for m in msgs:
                    when = datetime.fromtimestamp(float(m["ts"]), UTC).strftime("%H:%M")
                    who = self._user(m.get("user")) if m.get("user") else (m.get("username") or (m.get("bot_profile") or {}).get("name") or "bot")
                    extra = " _(edited)_" if m.get("edited") else ""
                    if m.get("reply_count"):
                        extra += f" ({m['reply_count']} replies in thread)"
                    files = ", ".join(f.get("name", "file") for f in m.get("files", []) if isinstance(f, dict))
                    lines.append(f"**{when}** · **{who}**: {self._text(m.get('text', ''))}{extra}" + (f" [files: {files}]" if files else ""))
                ext_id = f"{ch['id']}:{day}"
                self._docs[ext_id] = ("\n\n".join(lines) + "\n").encode("utf-8")
                version = _sha([f"{m['ts']}:{(m.get('edited') or {}).get('ts', '')}:{m.get('reply_count', 0)}:{m.get('latest_reply', '')}" for m in msgs])
                yield Item(external_id=ext_id, url=f"https://slack.com/app_redirect?channel={ch['id']}", title=f"#{name} — {day}", filename=f"slack-{_safe_name(name, 'channel')}-{day}.md", version=version)
                n += 1
                if n >= limit:
                    return

    def fetch(self, item: Item) -> Fetched:
        if item.external_id not in getattr(self, "_docs", {}):
            for _ in self.items():
                pass
        content = self._docs.get(item.external_id)
        if content is None:
            raise ConnectorError(f"{item.title} is no longer available.")
        return Fetched(item=item, content=content, filename=item.filename)


# ------------------------------------------------------------------------ Microsoft Teams


class TeamsConnector(SaaSConnector):
    key = "teams"
    vendor = "Microsoft Teams"
    MAX_PAGES = 40

    def validate(self) -> None:
        super().validate()
        oauth.tenant_for(self.config)
        _days(self.config)
        if not re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", str(self.config.get("team_id") or "")):
            raise ValidationFailed("team_id must be the team's group id (a GUID; Teams → team → Get link to team → groupId).")
        for c in _list(self.config.get("channel_ids")):
            if not re.fullmatch(r"19:[A-Za-z0-9_\-.]+@thread\.(tacv2|skype)", c):
                raise ValidationFailed(f"'{c}' is not a Teams channel id (19:…@thread.tacv2).")

    def _channels(self) -> list[dict[str, Any]]:
        team = self.config["team_id"]
        data = self._json("GET", f"{GRAPH}/teams/{team}/channels", params={"$select": "id,displayName,webUrl"})
        channels = data.get("value", [])
        wanted = _list(self.config.get("channel_ids"))
        if wanted:
            channels = [c for c in channels if c.get("id") in wanted]
            missing = [w for w in wanted if w not in {c.get("id") for c in channels}]
            if missing:
                raise ConnectorError(f"Channel(s) not found in this team: {', '.join(missing)}.")
        return channels

    def test(self) -> dict[str, Any]:
        self.validate()
        me = self._json("GET", f"{GRAPH}/me", params={"$select": "displayName,userPrincipalName"})
        team = self._json("GET", f"{GRAPH}/teams/{self.config['team_id']}", params={"$select": "id,displayName"})
        return {"ok": True, "account": me.get("userPrincipalName"), "team": team.get("displayName"), "channels": [c.get("displayName") for c in self._channels()][:20]}

    @staticmethod
    def _who(m: dict[str, Any]) -> str:
        frm = m.get("from") or {}
        return ((frm.get("user") or {}).get("displayName") or (frm.get("application") or {}).get("displayName") or "unknown")

    @staticmethod
    def _body(m: dict[str, Any]) -> str:
        body = m.get("body") or {}
        text = body.get("content") or ""
        text = _html_text(text) if body.get("contentType") == "html" else text
        subject = m.get("subject")
        return (f"**{subject}** — " if subject else "") + re.sub(r"\s*\n\s*", " ", text).strip()

    def items(self) -> Iterator[Item]:
        self.validate()
        limit, n = _cap(self.config), 0
        cutoff = datetime.now(UTC) - timedelta(days=_days(self.config))
        self._docs: dict[str, bytes] = {}
        team = self.config["team_id"]
        for ch in self._channels():
            days: dict[str, list[dict[str, Any]]] = defaultdict(list)
            url: str | None = f"{GRAPH}/teams/{team}/channels/{quote(ch['id'], safe='')}/messages"
            params: dict[str, Any] | None = {"$top": 50, "$expand": "replies"}
            for _ in range(self.MAX_PAGES):
                if not url:
                    break
                data = self._json("GET", url, params=params)
                values = data.get("value", [])
                recent = 0
                for m in values:
                    created = _iso(m.get("createdDateTime"))
                    touched = max([d for d in (created, _iso(m.get("lastModifiedDateTime")), *(_iso(r.get("lastModifiedDateTime") or r.get("createdDateTime")) for r in m.get("replies") or [])) if d] or [cutoff])
                    if touched >= cutoff:
                        recent += 1
                    if created is None or created < cutoff or m.get("messageType") != "message" or m.get("deletedDateTime"):
                        continue
                    days[created.date().isoformat()].append(m)
                url, params = data.get("@odata.nextLink"), None
                if url and not url.startswith(GRAPH):
                    raise ConnectorError("Unexpected pagination link from Microsoft Graph.")
                if values and not recent:
                    break  # the whole page predates the lookback window
            name = ch.get("displayName") or "channel"
            for day in sorted(days, reverse=True):
                msgs = sorted(days[day], key=lambda m: m.get("createdDateTime", ""))
                lines = [f"# {name} — {day}", "", f"Microsoft Teams channel {name}, messages from {day} (UTC).", ""]
                parts: list[str] = []
                for m in msgs:
                    when = (_iso(m.get("createdDateTime")) or cutoff).strftime("%H:%M")
                    lines.append(f"**{when}** · **{self._who(m)}**: {self._body(m)}")
                    parts.append(f"{m.get('id')}:{m.get('lastModifiedDateTime')}")
                    for r in sorted((r for r in m.get("replies") or [] if not r.get("deletedDateTime") and r.get("messageType", "message") == "message"), key=lambda r: r.get("createdDateTime", "")):
                        rw = (_iso(r.get("createdDateTime")) or cutoff).strftime("%Y-%m-%d %H:%M")
                        lines.append(f"  - reply {rw} · **{self._who(r)}**: {self._body(r)}")
                        parts.append(f"{r.get('id')}:{r.get('lastModifiedDateTime')}")
                ext_id = f"{ch['id']}:{day}"
                self._docs[ext_id] = ("\n\n".join(lines) + "\n").encode("utf-8")
                yield Item(external_id=ext_id, url=msgs[0].get("webUrl") or ch.get("webUrl", ""), title=f"{name} — {day}", filename=f"teams-{_safe_name(name, 'channel')}-{day}.md", version=_sha(parts))
                n += 1
                if n >= limit:
                    return

    def fetch(self, item: Item) -> Fetched:
        if item.external_id not in getattr(self, "_docs", {}):
            for _ in self.items():
                pass
        content = self._docs.get(item.external_id)
        if content is None:
            raise ConnectorError(f"{item.title} is no longer available.")
        return Fetched(item=item, content=content, filename=item.filename)


# ------------------------------------------------------------------------------- GitLab


class GitLabConnector(SaaSConnector):
    key = "gitlab"
    vendor = "GitLab"
    user_host = True
    DEFAULT_EXT = ["md", "mdx", "rst", "txt", "py", "ts", "js", "java", "go", "rs", "yaml", "yml", "json", "sql"]

    def _base(self) -> str:
        return (self.config.get("base_url") or "https://gitlab.com").strip().rstrip("/")

    def validate(self) -> None:
        super().validate()
        project = str(self.config.get("project") or "").strip().strip("/")
        if not (project.isdigit() or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*(/[A-Za-z0-9_][A-Za-z0-9_.\-]*)+", project)):
            raise ValidationFailed("project must be a path like 'group/project' (subgroups allowed) or a numeric project id.")
        base = self._base()
        if not re.fullmatch(r"https?://[A-Za-z0-9.\-]+(:\d+)?(/[A-Za-z0-9._~\-/]*)?", base):
            raise ValidationFailed("base_url must be the GitLab address, e.g. https://gitlab.com or https://gitlab.example.com.")
        check_url(base, purpose="integration")
        _cap(self.config, "max_files")

    def _auth_headers(self, force: bool = False) -> dict[str, str]:
        return {"PRIVATE-TOKEN": self.secret} if self.secret else {}

    def _pid(self) -> str:
        return quote(str(self.config["project"]).strip().strip("/"), safe="")

    def _api(self, path: str) -> str:
        return f"{self._base()}/api/v4/projects/{self._pid()}{path}"

    def _project(self) -> dict[str, Any]:
        r = self._call("GET", self._api(""), allow=(404,))
        if r.status_code == 404:
            raise ConnectorError("Project not found (or the token lacks access).")
        return r.json()

    def test(self) -> dict[str, Any]:
        self.validate()
        p = self._project()
        return {"ok": True, "project": p.get("path_with_namespace"), "default_branch": p.get("default_branch"), "visibility": p.get("visibility")}

    def items(self) -> Iterator[Item]:
        self.validate()
        project = self._project()
        self._branch = branch = self.config.get("branch") or project.get("default_branch") or "main"
        exts = {e.lower().lstrip(".") for e in (_list(self.config.get("extensions")) or self.DEFAULT_EXT)}
        prefix = (self.config.get("path_prefix") or "").lstrip("/")
        limit, n = _cap(self.config, "max_files"), 0
        web = project.get("web_url") or f"{self._base()}/{project.get('path_with_namespace', '')}"
        params: dict[str, Any] = {"recursive": "true", "per_page": 100, "ref": branch, "page": 1}
        if prefix.endswith("/"):
            params["path"] = prefix.rstrip("/")
        for _ in range(500):
            r = self._call("GET", self._api("/repository/tree"), params=params, allow=(404,))
            if r.status_code == 404:
                raise ConnectorError(f"Branch or path not found: {branch}{' / ' + prefix if prefix else ''}.")
            for entry in r.json():
                path = entry.get("path", "")
                if entry.get("type") != "blob" or not path.startswith(prefix):
                    continue
                ext = path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
                if ext not in exts:
                    continue
                filename = path.replace("/", "__")
                if ext == "mdx":
                    filename = filename[:-1]
                yield Item(external_id=path, url=f"{web}/-/blob/{quote(branch)}/{quote(path)}", title=path, filename=filename, version=entry.get("id"))
                n += 1
                if n >= limit:
                    return
            nxt = r.headers.get("x-next-page", "").strip()
            if not nxt.isdigit():
                return
            params = {**params, "page": int(nxt)}

    def fetch(self, item: Item) -> Fetched:
        if not item.version or not re.fullmatch(r"[0-9a-f]{7,64}", item.version):
            raise ConnectorError(f"No blob id for {item.external_id}.")
        r = self._call("GET", self._api(f"/repository/blobs/{item.version}/raw"))
        return Fetched(item=item, content=r.content, filename=item.filename)


# ------------------------------------------------------------------------- Email (IMAP)


IMAP_SSL: Any = imaplib.IMAP4_SSL  # tests substitute fakes
IMAP_PLAIN: Any = imaplib.IMAP4


def _imap_date(d: datetime) -> str:
    return f"{d.day:02d}-{_MONTHS[d.month - 1]}-{d.year}"


def email_markdown(raw: bytes) -> tuple[str, str]:
    """Render an RFC 822 message as Markdown; returns (subject, markdown)."""
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    subject = str(msg.get("subject") or "").strip() or "(no subject)"
    part = msg.get_body(preferencelist=("plain", "html"))  # type: ignore[attr-defined]
    text = ""
    if part is not None:
        try:
            content = part.get_content()
        except (LookupError, ValueError):
            content = (part.get_payload(decode=True) or b"").decode("utf-8", "replace")
        text = content if part.get_content_type() == "text/plain" else _html_text(content)
    attachments = [p.get_filename() for p in msg.iter_attachments() if p.get_filename()]  # type: ignore[attr-defined]
    meta = [("From", msg.get("from")), ("To", msg.get("to")), ("Cc", msg.get("cc")), ("Date", msg.get("date"))]
    lines = [f"# {subject}", ""] + [f"- **{k}:** {v}" for k, v in meta if v]
    if attachments:
        lines.append(f"- **Attachments:** {', '.join(attachments)}")
    lines += ["", "---", "", text.strip(), ""]
    return subject, "\n".join(lines)


class EmailConnector(Connector):
    key = "email"

    def _port(self) -> int:
        security = self.config.get("security") or "ssl"
        try:
            port = int(self.config.get("port") or (993 if security == "ssl" else 143))
        except (TypeError, ValueError) as exc:
            raise ValidationFailed("port must be a number.") from exc
        if not 1 <= port <= 65535:
            raise ValidationFailed("port must be between 1 and 65535.")
        return port

    def _folder(self) -> str:
        folder = str(self.config.get("folder") or "INBOX")
        if any(c in folder for c in "\r\n\x00") or len(folder) > 200:
            raise ValidationFailed("folder is not a valid mailbox name.")
        return folder

    def validate(self) -> None:
        host = str(self.config.get("host") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9.\-]+|\[[0-9A-Fa-f:.]+\]", host):
            raise ValidationFailed("host must be the IMAP server name, e.g. imap.example.com.")
        if (self.config.get("security") or "ssl") not in ("ssl", "starttls"):
            raise ValidationFailed("security must be 'ssl' (port 993) or 'starttls' (port 143); unencrypted IMAP is not supported.")
        if not str(self.config.get("username") or "").strip():
            raise ValidationFailed("username is required.")
        self._folder()
        _days(self.config)
        _cap(self.config)
        check_host(host, self._port(), purpose="integration")

    def _connect(self) -> Any:
        if getattr(self, "_conn", None) is not None:
            return self._conn
        self.validate()
        if not self.secret:
            raise ConnectorError("No IMAP password is stored for this integration.")
        host, port = str(self.config["host"]).strip(), self._port()
        ctx = ssl.create_default_context()
        try:
            if (self.config.get("security") or "ssl") == "ssl":
                conn = IMAP_SSL(host, port, ssl_context=ctx, timeout=30)
            else:
                conn = IMAP_PLAIN(host, port, timeout=30)
                conn.starttls(ssl_context=ctx)
        except (OSError, ssl.SSLError, imaplib.IMAP4.error) as exc:
            raise ConnectorError(f"Could not open a TLS connection to {host}:{port} ({type(exc).__name__}).") from exc
        try:
            conn.login(str(self.config["username"]).strip(), self.secret)
        except imaplib.IMAP4.error as exc:
            raise ConnectorError("IMAP login failed: check the username and (app) password.") from exc
        quoted = '"' + self._folder().replace("\\", "\\\\").replace('"', '\\"') + '"'
        typ, data = conn.select(quoted, readonly=True)
        if typ != "OK":
            raise ConnectorError(f"Mailbox '{self._folder()}' was not found.")
        self._count = int((data or [b"0"])[0] or 0)
        self._conn = conn
        return conn

    def close(self) -> None:
        conn, self._conn = getattr(self, "_conn", None), None
        if conn is not None:
            try:
                conn.logout()
            except (OSError, imaplib.IMAP4.error):
                pass

    def test(self) -> dict[str, Any]:
        try:
            self._connect()
            return {"ok": True, "folder": self._folder(), "messages": self._count}
        finally:
            self.close()

    def items(self) -> Iterator[Item]:
        conn = self._connect()
        since = _imap_date(datetime.now(UTC) - timedelta(days=_days(self.config)))
        typ, data = conn.uid("SEARCH", None, "SINCE", since)
        if typ != "OK":
            raise ConnectorError("IMAP search failed.")
        uids = [u.decode() for u in (data[0] or b"").split()][::-1][: _cap(self.config)]
        self._uids: dict[str, str] = {}
        folder = self._folder()
        seen: set[str] = set()
        for i in range(0, len(uids), 100):
            batch = uids[i : i + 100]
            typ, rows = conn.uid("FETCH", ",".join(batch), "(RFC822.SIZE BODY.PEEK[HEADER.FIELDS (MESSAGE-ID SUBJECT DATE)])")
            if typ != "OK":
                raise ConnectorError("IMAP fetch failed.")
            for row in rows or []:
                if not isinstance(row, tuple) or len(row) < 2:
                    continue
                head = row[0].decode("ascii", "replace") if isinstance(row[0], bytes) else str(row[0])
                uid = re.search(r"UID (\d+)", head)
                size = re.search(r"RFC822\.SIZE (\d+)", head)
                if not uid:
                    continue
                hdr = email.message_from_bytes(row[1], policy=email.policy.default)
                mid = str(hdr.get("message-id") or "").strip() or f"uid:{uid.group(1)}"
                if size and int(size.group(1)) > _ic.MAX_BYTES:
                    continue
                subject = str(hdr.get("subject") or "").strip() or "(no subject)"
                date = hdr.get("date")
                day = date.datetime.date().isoformat() if date is not None and getattr(date, "datetime", None) else "undated"
                ext_id = f"{folder}:{mid}"
                if ext_id in seen:
                    continue  # the same message twice in one mailbox
                seen.add(ext_id)
                self._uids[ext_id] = uid.group(1)
                yield Item(external_id=ext_id, url="", title=subject[:300], filename=f"email-{day}-{_safe_name(subject, 'message')[:60]}-{_short(mid)}.md", version=f"{mid}:{size.group(1) if size else ''}")

    def fetch(self, item: Item) -> Fetched:
        conn = self._connect()
        uid = getattr(self, "_uids", {}).get(item.external_id)
        if uid is None:
            raise ConnectorError(f"{item.title} is no longer in the mailbox.")
        typ, rows = conn.uid("FETCH", uid, "(BODY.PEEK[])")
        raw = next((r[1] for r in rows or [] if isinstance(r, tuple) and len(r) > 1), None)
        if typ != "OK" or raw is None:
            raise ConnectorError(f"Could not download {item.title}.")
        if len(raw) > _ic.MAX_BYTES:
            raise ConnectorError(f"{item.title} exceeds the {_ic.MAX_BYTES // (1024 * 1024)} MB limit.")
        _, md = email_markdown(raw)
        return Fetched(item=item, content=md.encode("utf-8"), filename=item.filename)


SAAS_CONNECTORS: dict[str, type[Connector]] = {
    "google_drive": GoogleDriveConnector,
    "onedrive": OneDriveConnector,
    "sharepoint": SharePointConnector,
    "dropbox": DropboxConnector,
    "notion": NotionConnector,
    "confluence": ConfluenceConnector,
    "slack": SlackConnector,
    "teams": TeamsConnector,
    "gitlab": GitLabConnector,
    "email": EmailConnector,
}
