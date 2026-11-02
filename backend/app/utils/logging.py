"""Structured logging with request IDs and secret redaction.

Each module gets a child logger of the ``csvai`` root (kept from CSV-AI so
existing ``get_logger`` callers work). Log records carry the current request
ID, organisation and user from context variables. A redaction filter masks
anything that looks like a credential before it reaches a handler, so
secrets cannot end up in log files even if a caller logs them by mistake.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
org_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("org_id", default="-")
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="-")

_CONFIGURED = False

_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)(password|passwd|pwd|secret|token|api[_-]?key|authorization)(\"?\s*[:=]\s*\"?)([^\s\"',;&]+)"), r"\1\2***"),
    (re.compile(r"(?i)bearer\s+[a-z0-9._\-]+"), "Bearer ***"),
    (re.compile(r"\b(sk|rk|pk)[-_][A-Za-z0-9._\-]{8,}"), r"\1_***"),
    # Vendor tokens used by SaaS integrations: Slack, GitLab, Notion, Google access tokens.
    (re.compile(r"\b(xox[abposr]|xapp|glpat|ntn|ya29)[-_.][A-Za-z0-9._\-]{8,}"), r"\1-***"),
    (re.compile(r"(?i)(code_verifier|refresh_token|access_token|client_secret)=([^\s&\"']+)"), r"\1=***"),
    (re.compile(r"(?i)(postgres(?:ql)?|mysql|mariadb|mssql|mongodb(?:\+srv)?|redis|oracle)(\+\w+)?://([^:/@\s]+):([^@\s]+)@"), r"\1\2://\3:***@"),
)


def redact(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.org_id = org_id_var.get()
        record.user_id = user_id_var.get()
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "org_id": getattr(record, "org_id", "-"),
            "user_id": getattr(record, "user_id", "-"),
        }
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    global _CONFIGURED
    root = logging.getLogger("csvai")
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(_ContextFilter())
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s rid=%(request_id)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Return a namespaced logger under ``csvai.<name>``."""
    if not _CONFIGURED:
        configure_logging(level or "INFO")
    return logging.getLogger(f"csvai.{name}")
