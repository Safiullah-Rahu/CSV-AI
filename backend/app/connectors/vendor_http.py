"""HTTP for SaaS connectors and OAuth token endpoints.

Timeouts on every call, ``Retry-After`` honoured on 429/503 (bounded), the
connector ``MAX_BYTES`` cap enforced while streaming, and GET redirects
followed by hand: every hop is re-checked by the SSRF guard and the
``Authorization`` header is dropped when the host changes (pre-signed
download URLs from Graph / Drive must not receive the bearer token).

Tests install an ``httpx.MockTransport`` in ``TRANSPORT``.
"""

from __future__ import annotations

import time
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from app.connectors import integrations as _ic
from app.core.netguard import check_url
from app.utils.errors import ConnectorError

TRANSPORT: httpx.BaseTransport | None = None
_sleep = time.sleep  # monkeypatched by tests


def client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(headers={"User-Agent": _ic.USER_AGENT}, timeout=httpx.Timeout(timeout, connect=10.0), follow_redirects=False, transport=TRANSPORT)


def retry_after_seconds(value: str | None, default: float = 5.0) -> float:
    if not value:
        return default
    value = value.strip()
    if value.replace(".", "", 1).isdigit():
        return float(value)
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return default


class VendorHTTP:
    """One client per connector run; ``check_hosts`` SSRF-checks the first hop too (user-supplied hosts)."""

    def __init__(self, *, check_hosts: bool = False, timeout: float = 30.0, max_retries: int = 3, max_wait: float = 60.0, max_bytes: int | None = None) -> None:
        self.check_hosts, self.timeout, self.max_retries, self.max_wait, self.max_bytes = check_hosts, timeout, max_retries, max_wait, max_bytes
        self._client: httpx.Client | None = None

    @property
    def http(self) -> httpx.Client:
        if self._client is None:
            self._client = client(self.timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def request(self, method: str, url: str, *, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None, json: Any = None, data: Any = None, content: bytes | None = None, auth: Any = None, max_redirects: int = 5) -> httpx.Response:
        headers = dict(headers or {})
        attempts = 0
        hops = 0
        origin = urlparse(url).netloc
        while True:
            if self.check_hosts or hops:
                check_url(url, purpose="integration")
            try:
                with self.http.stream(method, url, headers=headers, params=params, json=json, data=data, content=content, auth=auth) as r:
                    if r.status_code in (429, 503) and attempts < self.max_retries:
                        wait = retry_after_seconds(r.headers.get("retry-after"), default=2.0 * (attempts + 1))
                        if wait > self.max_wait:
                            raise ConnectorError(f"{urlparse(url).netloc} is rate limiting requests; retry after {int(wait)} s.")
                        attempts += 1
                        _sleep(wait)
                        continue
                    if method.upper() == "GET" and r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                        if hops >= max_redirects:
                            raise ConnectorError("Too many redirects.")
                        url = urljoin(url, r.headers["location"])
                        params = None
                        hops += 1
                        if urlparse(url).netloc != origin:
                            headers = {k: v for k, v in headers.items() if k.lower() not in ("authorization", "private-token", "cookie")}
                            auth = None
                        continue
                    cap = self.max_bytes or _ic.MAX_BYTES
                    body = bytearray()
                    for chunk in r.iter_bytes():
                        body += chunk
                        if len(body) > cap:
                            raise ConnectorError(f"Response from {urlparse(url).netloc} exceeds the {max(cap // (1024 * 1024), 1)} MB limit.")
                    return _ic.buffered(r, bytes(body))
            except httpx.TimeoutException as exc:
                raise ConnectorError(f"{urlparse(url).netloc} did not respond in time.") from exc
            except httpx.HTTPError as exc:
                raise ConnectorError(f"Could not reach {urlparse(url).netloc} ({type(exc).__name__}).") from exc


def vendor_error(r: httpx.Response, vendor: str) -> str:
    """A short, token-free description of a failed vendor response."""
    detail = ""
    try:
        body = r.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                detail = str(err.get("message") or err.get("code") or "")
            elif err:
                detail = str(body.get("error_description") or err)
            detail = detail or str(body.get("message") or body.get("error_summary") or "")
    except ValueError:
        detail = ""
    hint = {401: "the credentials are invalid or expired; reconnect", 403: "access denied; check the granted scopes and sharing", 404: "not found or not shared with this connection"}.get(r.status_code, "")
    parts = [p for p in (hint, detail[:200]) if p]
    return f"{vendor} returned HTTP {r.status_code}" + (f": {'; '.join(parts)}" if parts else "") + "."
