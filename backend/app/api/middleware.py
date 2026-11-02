"""Pure-ASGI middleware: request IDs, request context, security headers, timing.

Pure ASGI (not ``BaseHTTPMiddleware``) so streaming responses (SSE, chat)
are passed through untouched.
"""

from __future__ import annotations

import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import get_settings
from app.services.activity import client_ip_var, user_agent_var
from app.utils.logging import get_logger, org_id_var, request_id_var, user_id_var

log = get_logger("http")
_RID = re.compile(r"^[A-Za-z0-9\-_.]{8,64}$")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.prod = get_settings().is_production

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        rid = headers.get("x-request-id", "")
        if not _RID.match(rid):
            rid = uuid.uuid4().hex
        tokens = [
            request_id_var.set(rid),
            org_id_var.set("-"),
            user_id_var.set("-"),
        ]
        client = scope.get("client")
        ip_tok = client_ip_var.set(client[0] if client else "")
        ua_tok = user_agent_var.set(headers.get("user-agent", "")[:400])
        start = time.perf_counter()
        status_holder = {"status": 0}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                extra = [
                    (b"x-request-id", rid.encode()),
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                    (b"cross-origin-opener-policy", b"same-origin"),
                ]
                path = scope.get("path", "")
                if not path.startswith(("/docs", "/redoc")):
                    extra.append((b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"))
                if self.prod:
                    extra.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                message.setdefault("headers", [])
                message["headers"] = list(message["headers"]) + extra
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            if not scope.get("path", "").startswith("/api/v1/health"):
                log.info(
                    "%s %s -> %s (%.1f ms)",
                    scope.get("method"),
                    scope.get("path"),
                    status_holder["status"],
                    elapsed,
                )
            for tok, var in zip(tokens, (request_id_var, org_id_var, user_id_var)):
                var.reset(tok)
            client_ip_var.reset(ip_tok)
            user_agent_var.reset(ua_tok)
