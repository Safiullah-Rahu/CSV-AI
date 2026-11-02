"""FastAPI application factory.

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.middleware import RequestContextMiddleware
from app.config import get_settings
from app.db.session import get_engine
from app.utils.errors import CSVAIError
from app.utils.logging import configure_logging, get_logger, request_id_var

log = get_logger("app")


def _error(status: int, code: str, message: str, details: Any = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"code": code, "message": message, "request_id": request_id_var.get()}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


def _check_db_role() -> None:
    """Refuse to run as a role that bypasses Row-Level Security."""
    settings = get_settings()
    with get_engine().connect() as conn:
        row = conn.execute(text("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")).one()
    if row.rolsuper or row.rolbypassrls:
        msg = "Database role bypasses Row-Level Security (superuser/BYPASSRLS). Use a dedicated application role."
        if settings.is_production:
            raise RuntimeError(msg)
        log.critical(msg)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    settings = get_settings()
    problems = settings.validate_for_production()
    if problems:
        raise RuntimeError("Refusing to start: " + " ".join(problems))
    _check_db_role()
    if settings.job_backend == "embedded":
        from app.jobs.worker import start_embedded

        start_embedded()
    yield
    if settings.job_backend == "embedded":
        from app.jobs.worker import stop_embedded

        stop_embedded()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    app = FastAPI(
        title=settings.app_name,
        version="3.0.0",
        description="Knowledge Intelligence: multi-tenant, permission-aware RAG and knowledge intelligence platform.",
        openapi_url=f"{settings.api_prefix}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Org-Id", "X-Request-ID", "X-Requested-With", "Idempotency-Key", "Last-Event-ID"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(CSVAIError)
    async def _app_error(_: Request, exc: CSVAIError) -> JSONResponse:
        return _error(exc.status_code, exc.code, exc.message, exc.details or None)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        return _error(422, "validation_error", "Request validation failed.", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error: %s", type(exc).__name__)
        return _error(500, "internal_error", "An unexpected error occurred. Quote the request id when reporting it.")

    import app.jobs.handlers as _job_handlers  # noqa: F401 - register background job handlers
    from app.api.router import api_router

    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
