"""Embedded PostgreSQL 16 + pgvector for local development and tests (no Docker).

    python scripts/devdb.py            # start (idempotent) and print DATABASE_URL
    python scripts/devdb.py --db rag_test

Creates a NON-superuser role ``rag`` that owns the database. The application
must connect as this role: superusers bypass Row-Level Security.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pgserver
import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DEVDB_DIR", ROOT / "var" / "pgdata"))


def sqlalchemy_url(admin_uri: str, db_name: str, role: str, password: str) -> str:
    """SQLAlchemy URL for ``role`` on the server behind ``admin_uri``.

    pgserver listens on TCP on Windows but on a Unix-domain socket on Linux/macOS
    (``host`` is then a directory), so derive the URL from parsed conninfo, not strings.
    """
    info = conninfo_to_dict(admin_uri)
    host, port = str(info.get("host") or "localhost"), info.get("port")
    if host.startswith("/"):
        return f"postgresql+psycopg://{role}:{password}@/{db_name}?host={host}" + (f"&port={port}" if port else "")
    return f"postgresql+psycopg://{role}:{password}@{host}:{port or 5432}/{db_name}"


def ensure(db_name: str = "rag", role: str = "rag", password: str = "rag") -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    server = pgserver.get_server(str(DATA_DIR), cleanup_mode=None)
    admin_uri = server.get_uri()
    with psycopg.connect(admin_uri, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role,)).fetchone()
        if not exists:
            from psycopg import sql

            conn.execute(
                sql.SQL(
                    "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOBYPASSRLS NOCREATEROLE"
                ).format(sql.Identifier(role), sql.Literal(password))
            )
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (db_name,)).fetchone():
            conn.execute(f'CREATE DATABASE "{db_name}" OWNER "{role}"')
    with psycopg.connect(make_conninfo(admin_uri, dbname=db_name), autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    return sqlalchemy_url(admin_uri, db_name, role, password)


def drop(db_name: str) -> None:
    server = pgserver.get_server(str(DATA_DIR), cleanup_mode=None)
    with psycopg.connect(server.get_uri(), autocommit=True) as conn:
        conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s", (db_name,)
        )
        conn.execute(f'DROP DATABASE IF EXISTS "{db_name}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="rag")
    parser.add_argument("--drop", action="store_true")
    args = parser.parse_args()
    if args.drop:
        drop(args.db)
        print(f"dropped {args.db}")
        sys.exit(0)
    print(ensure(args.db))
