"""Liveness and readiness probes."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text

from app.config import get_settings
from app.db.session import get_engine, system_session
from app.models import Job
from app.storage import get_storage

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def live() -> dict[str, str]:
    return {"status": "ok"}


def readiness() -> tuple[bool, dict[str, Any]]:
    settings = get_settings()
    checks: dict[str, Any] = {}
    ok = True
    t = time.perf_counter()
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
            rev = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
        checks["database"] = {"ok": True, "latency_ms": round((time.perf_counter() - t) * 1000, 1), "migration": rev}
    except Exception as exc:  # noqa: BLE001
        ok = False
        checks["database"] = {"ok": False, "error": type(exc).__name__}
    try:
        checks["storage"] = {"ok": bool(get_storage().health()), "backend": settings.storage_backend}
    except Exception as exc:  # noqa: BLE001
        ok = False
        checks["storage"] = {"ok": False, "backend": settings.storage_backend, "error": type(exc).__name__}
    if settings.redis_url:
        try:
            import redis

            redis.Redis.from_url(settings.redis_url, socket_timeout=2).ping()
            checks["redis"] = {"ok": True}
        except Exception as exc:  # noqa: BLE001
            ok = False
            checks["redis"] = {"ok": False, "error": type(exc).__name__}
    try:
        with system_session() as db:
            counts = dict(db.execute(select(Job.status, func.count()).group_by(Job.status)).all())
        checks["jobs"] = {"ok": True, "backend": settings.job_backend, "counts": counts}
    except Exception as exc:  # noqa: BLE001
        checks["jobs"] = {"ok": False, "error": type(exc).__name__}
    return ok, checks


@router.get("/ready")
def ready() -> JSONResponse:
    ok, checks = readiness()
    return JSONResponse(status_code=200 if ok else 503, content={"status": "ready" if ok else "degraded", "checks": checks})
