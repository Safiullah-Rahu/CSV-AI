"""Local development server: embedded PostgreSQL + migrations + API with embedded worker.

    python scripts/dev.py            # http://127.0.0.1:8000  (docs at /docs)
    python scripts/dev.py --env-file ../.env.keys [--env-file other.env]

``--env-file`` loads only the platform's model-provider credentials (``PROVIDER_KEYS``)
from KEY=VALUE files and turns on the platform model fallback, so organisations
without their own AI provider answer with Claude or OpenAI. Values are never printed.

Development defaults only: console email backend (links printed to stderr),
local storage under ./var, and localhost allowed for AI endpoints so a local
Ollama can be configured. Never use these settings in production.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import devdb  # noqa: E402

PROVIDER_KEYS = {"ANTHROPIC_API_KEY", "ANTHROPIC_WORKSPACE_ID", "OPENAI_API_KEY"}


def _load_keys(argv: list[str]) -> list[str]:
    loaded: list[str] = []
    for i, arg in enumerate(argv):
        if arg != "--env-file" or i + 1 >= len(argv):
            continue
        for line in Path(argv[i + 1]).read_text(encoding="utf-8").splitlines():
            key, _, value = line.strip().removeprefix("export ").partition("=")
            key = key.strip()
            if key in PROVIDER_KEYS and value.strip():
                os.environ.setdefault(key, value.strip().strip('"').strip("'"))
                loaded.append(key)
    return loaded


_keys = _load_keys(sys.argv[1:])
if _keys:
    os.environ.setdefault("PLATFORM_LLM_FALLBACK", "true")
    os.environ.setdefault("DEFAULT_PROVIDER", "anthropic" if "ANTHROPIC_API_KEY" in _keys else "openai")
    print(f"dev: loaded {', '.join(sorted(set(_keys)))} (values hidden); platform model fallback on", file=sys.stderr)

os.environ.setdefault("ENVIRONMENT", "dev")
os.environ.setdefault("EMAIL_BACKEND", "console")
os.environ.setdefault("JOB_BACKEND", "embedded")
os.environ.setdefault("STORAGE_LOCAL_ROOT", str(ROOT / "var" / "storage"))
os.environ.setdefault("SSRF_ALLOWLIST", "localhost,127.0.0.1")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
os.environ.setdefault("JWT_SECRET", "dev-only-local-secret-not-for-production-0001")
if not os.environ.get("DATABASE_URL"):  # an explicit database (e.g. a CI service) wins
    os.environ["DATABASE_URL"] = devdb.ensure("rag")

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402

cfg = Config(str(ROOT / "alembic.ini"))
cfg.set_main_option("script_location", str(ROOT / "alembic"))
cfg.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
command.upgrade(cfg, "head")

import uvicorn  # noqa: E402

uvicorn.run("app.main:app", host="127.0.0.1", port=int(os.environ.get("PORT", "8000")), log_level="info")
