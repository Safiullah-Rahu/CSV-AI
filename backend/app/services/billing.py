"""Subscription plans and usage limits.

Billing is optional: with ``BILLING_ENABLED=false`` (self-hosted default)
organisations get the ``enterprise`` plan, which has no limits. Payment
providers plug in behind :class:`BillingProvider`; none is required.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import (
    DatabaseConnection,
    Document,
    Integration,
    Invitation,
    KnowledgeBase,
    Membership,
    Plan,
    Subscription,
    UsageRecord,
)
from app.utils.errors import LimitExceeded, NotAvailable


def plan_for(db: Session, org_id: uuid.UUID) -> tuple[Plan | None, Subscription | None]:
    sub = db.scalar(select(Subscription).where(Subscription.org_id == org_id))
    plan = db.get(Plan, sub.plan_key) if sub else db.get(Plan, get_settings().default_plan)
    return plan, sub


def limits_for(db: Session, org_id: uuid.UUID) -> dict[str, Any]:
    plan, sub = plan_for(db, org_id)
    limits = dict(plan.limits if plan else {})
    if sub and sub.limit_overrides:
        limits.update(sub.limit_overrides)
    return limits


def _month_start() -> datetime:
    now = datetime.now(UTC)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def current_usage(db: Session, org_id: uuid.UUID) -> dict[str, int]:
    def count(model: Any, *where: Any) -> int:
        return db.scalar(select(func.count()).select_from(model).where(model.org_id == org_id, *where)) or 0

    storage = db.scalar(select(func.coalesce(func.sum(Document.size_bytes), 0)).where(Document.org_id == org_id)) or 0
    tokens = db.scalar(
        select(func.coalesce(func.sum(UsageRecord.tokens_in + UsageRecord.tokens_out), 0)).where(
            UsageRecord.org_id == org_id, UsageRecord.created_at >= _month_start()
        )
    ) or 0
    day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    api_calls = db.scalar(
        select(func.count()).select_from(UsageRecord).where(
            UsageRecord.org_id == org_id, UsageRecord.api_key_id.is_not(None), UsageRecord.created_at >= day_start
        )
    ) or 0
    return {
        "users": count(Membership, Membership.status == "active") + count(Invitation, Invitation.status == "pending"),
        "knowledge_bases": count(KnowledgeBase),
        "documents": count(Document),
        "storage_mb": int(storage) // (1024 * 1024),
        "monthly_tokens": int(tokens),
        "api_calls_per_day": int(api_calls),
        "connected_sources": count(DatabaseConnection) + count(Integration),
    }


def check_limit(db: Session, org_id: uuid.UUID, key: str, add: int = 1) -> None:
    limit = limits_for(db, org_id).get(key)
    if limit is None:
        return
    used = current_usage(db, org_id).get(key, 0)
    if used + add > limit:
        raise LimitExceeded(
            f"Your plan allows {limit} {key.replace('_', ' ')}; this would exceed it.",
            details={"limit": limit, "used": used, "key": key},
        )


class BillingProvider(Protocol):
    name: str

    def create_checkout(self, org_id: uuid.UUID, plan_key: str, success_url: str) -> str: ...


def get_billing_provider() -> BillingProvider:
    settings = get_settings()
    if not settings.billing_enabled:
        raise NotAvailable("Billing is disabled in this deployment.")
    if not settings.stripe_secret_key:
        raise NotAvailable("No payment provider is configured (set STRIPE_SECRET_KEY).")
    raise NotAvailable("Stripe checkout is not implemented yet; plans can be changed by a platform administrator.")
