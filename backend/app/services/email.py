"""Outbound email (verification, password reset, invitations).

* ``smtp``    — real delivery via SMTP (STARTTLS by default).
* ``console`` — development only: logs the message *subject and link host*,
  and prints the full link to stderr so a developer can click it. Refused in
  production by :meth:`Settings.validate_for_production`.
* ``memory``  — tests: messages are kept in :data:`OUTBOX`.
"""

from __future__ import annotations

import smtplib
import sys
from dataclasses import dataclass
from email.message import EmailMessage

from app.config import get_settings
from app.utils.logging import get_logger

log = get_logger("email")


@dataclass
class SentEmail:
    to: str
    subject: str
    body: str
    link: str | None = None


OUTBOX: list[SentEmail] = []


def send_email(to: str, subject: str, body: str, *, link: str | None = None) -> None:
    settings = get_settings()
    backend = settings.email_backend
    if backend == "memory":
        OUTBOX.append(SentEmail(to, subject, body, link))
        return
    if backend == "console":
        log.info("email (console backend) to=%s subject=%s", to, subject)
        if link:
            print(f"[dev email] {subject} -> {to}: {link}", file=sys.stderr, flush=True)
        return
    msg = EmailMessage()
    msg["From"] = settings.smtp_from
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(settings.smtp_host or "localhost", settings.smtp_port, timeout=20) as smtp:
            if settings.smtp_starttls:
                smtp.starttls()
            if settings.smtp_username:
                smtp.login(settings.smtp_username, settings.smtp_password or "")
            smtp.send_message(msg)
    except Exception:  # noqa: BLE001 — surface as log; callers do not leak delivery status
        log.exception("SMTP delivery failed to=%s", to)
        raise


def link(path: str) -> str:
    return get_settings().public_app_url.rstrip("/") + path
