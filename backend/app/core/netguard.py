"""Outbound network guard (SSRF protection).

Every user-supplied destination — database hosts, webhook URLs, website and
sitemap crawls, custom AI endpoints — is resolved and checked here before a
connection is opened:

* cloud metadata / link-local ranges are always blocked;
* loopback and the platform's own database endpoint are always blocked
  (a tenant must never reach the platform's internals);
* private ranges are allowed for database connections only when
  ``ALLOW_PRIVATE_NETWORK_TARGETS`` (self-hosted default) and for web
  ingestion only when ``ALLOW_PRIVATE_WEB_INGESTION``;
* ``SSRF_ALLOWLIST`` hosts bypass the private-range rule (not metadata).
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from sqlalchemy.engine import make_url

from app.config import get_settings
from app.utils.errors import UnsafeRequest

_ALWAYS_BLOCKED = [ipaddress.ip_network(n) for n in ("169.254.0.0/16", "fe80::/10", "0.0.0.0/8", "100.100.100.200/32", "fd00:ec2::254/128", "224.0.0.0/4", "ff00::/8")]
_LOOPBACK = [ipaddress.ip_network(n) for n in ("127.0.0.0/8", "::1/128")]
_PRIVATE = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "fc00::/7")]


def _resolve(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeRequest(f"Host '{host}' could not be resolved.") from exc
    return sorted({ipaddress.ip_address(i[4][0].split("%")[0]) for i in infos}, key=str)


def _platform_db_endpoint() -> tuple[set[str], int | None]:
    try:
        url = make_url(get_settings().database_url)
        host = url.host or "localhost"
        return {str(ip) for ip in _resolve(host)}, url.port or 5432
    except Exception:  # noqa: BLE001
        return set(), None


def check_host(host: str, port: int | None, *, purpose: str) -> None:
    """Raise :class:`UnsafeRequest` if ``host`` is not an acceptable destination."""
    settings = get_settings()
    host = (host or "").strip().strip("[]")
    if not host:
        raise UnsafeRequest("Host is required.")
    allowlisted = host.lower() in {h.lower() for h in settings.ssrf_allowlist}
    ips = _resolve(host)
    platform_ips, platform_port = _platform_db_endpoint()
    for ip in ips:
        if any(ip in n for n in _ALWAYS_BLOCKED):
            raise UnsafeRequest(f"Destination {host} resolves to a blocked address range.")
        if port is not None and platform_port is not None and str(ip) in platform_ips and int(port) == int(platform_port):
            raise UnsafeRequest("Connecting to the platform's own database is not allowed.")
        if allowlisted:
            continue
        if any(ip in n for n in _LOOPBACK):
            raise UnsafeRequest(f"Destination {host} is a loopback address.")
        private = any(ip in n for n in _PRIVATE)
        if private:
            allowed = settings.allow_private_network_targets if purpose in ("database", "ai_endpoint") else settings.allow_private_web_ingestion
            if not allowed:
                raise UnsafeRequest(f"Destination {host} is on a private network, which is disabled for {purpose} targets.")


def check_url(url: str, *, purpose: str, schemes: tuple[str, ...] = ("https", "http")) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in schemes:
        raise UnsafeRequest(f"URL scheme must be one of {schemes}.")
    if parsed.username or parsed.password:
        raise UnsafeRequest("Credentials in URLs are not allowed.")
    check_host(parsed.hostname or "", parsed.port or (443 if parsed.scheme == "https" else 80), purpose=purpose)
