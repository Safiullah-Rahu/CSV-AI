"""Engine/session management with tenant binding for Row-Level Security.

Every transaction starts by publishing the caller's tenant context into
PostgreSQL settings (``app.current_org``, ``app.current_user``,
``app.rls_bypass``). RLS policies created by the migrations read those settings,
so a query that forgets an ``org_id`` filter still cannot see another tenant's
rows. Settings are applied with ``set_config(..., is_local => true)`` so they
vanish at commit/rollback and can never leak to the next pooled connection.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

_SET_CONTEXT = text(
    "SELECT set_config('app.current_org', :org, true),"
    " set_config('app.current_user', :usr, true),"
    " set_config('app.rls_bypass', :bypass, true)"
)


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    settings = get_settings()
    db_url = url or settings.database_url
    return create_engine(
        db_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
        connect_args={"options": f"-c statement_timeout={settings.db_statement_timeout_ms}"},
        future=True,
    )


@lru_cache(maxsize=4)
def _factory(url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(url), expire_on_commit=False, autoflush=False)


def _apply_context(session: Session, connection: Any) -> None:
    info = session.info
    connection.execute(
        _SET_CONTEXT,
        {
            "org": str(info.get("org_id") or ""),
            "usr": str(info.get("user_id") or ""),
            "bypass": "on" if info.get("rls_bypass") else "off",
        },
    )


@event.listens_for(Session, "after_begin")
def _after_begin(session: Session, transaction: Any, connection: Any) -> None:
    _apply_context(session, connection)


def bind_tenant(
    session: Session,
    *,
    org_id: uuid.UUID | None = None,
    user_id: uuid.UUID | None = None,
) -> None:
    """Attach tenant context to ``session`` and to its open transaction."""
    if org_id is not None:
        session.info["org_id"] = org_id
    if user_id is not None:
        session.info["user_id"] = user_id
    if session.in_transaction():
        _apply_context(session, session.connection())


def new_session(*, url: str | None = None) -> Session:
    return _factory(url)()


@contextmanager
def tenant_session(
    org_id: uuid.UUID | None, user_id: uuid.UUID | None = None
) -> Iterator[Session]:
    """Session confined to one organisation (used by request handlers)."""
    session = new_session()
    session.info.update({"org_id": org_id, "user_id": user_id, "rls_bypass": False})
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def system_session() -> Iterator[Session]:
    """Session that bypasses RLS — only for workers, schedulers and platform jobs.

    Code running here must filter by ``org_id`` explicitly. It is never used by
    request handlers that act for an organisation member.
    """
    session = new_session()
    session.info.update({"rls_bypass": True})
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engines() -> None:
    """Dispose cached engines (tests switch databases)."""
    for fn in (_factory, get_engine):
        fn.cache_clear()
