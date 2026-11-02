"""Database connector framework.

A connector knows how to test a connection, discover schema (schemas,
tables/collections, columns, primary and foreign keys, row estimates), run a
validated read-only query with a timeout and row cap, and stream rows for
synchronisation. SQL engines share :class:`SQLAlchemyConnector`; NoSQL
engines have their own classes.

``PROVIDERS[...]["status"]`` is shown in the UI and must stay honest:
``tested`` (exercised by the automated suite against a live server),
``beta`` (implemented, not verified against a live server here) or
``planned`` (configuration only; connecting raises NotAvailable).
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus

from app.config import get_settings
from app.core.netguard import check_host
from app.utils.errors import ConnectorError, NotAvailable, UnsafeRequest, ValidationFailed

PROVIDERS: dict[str, dict[str, Any]] = {
    "postgresql": {"label": "PostgreSQL", "family": "sql", "driver": "postgresql+psycopg", "port": 5432, "dialect": "postgres", "status": "tested", "category": "database"},
    "supabase": {"label": "Supabase (PostgreSQL)", "family": "sql", "driver": "postgresql+psycopg", "port": 5432, "dialect": "postgres", "status": "beta", "category": "database"},
    "amazon_rds_postgres": {"label": "Amazon RDS for PostgreSQL", "family": "sql", "driver": "postgresql+psycopg", "port": 5432, "dialect": "postgres", "status": "beta", "category": "database"},
    "pgvector": {"label": "PostgreSQL + pgvector", "family": "sql", "driver": "postgresql+psycopg", "port": 5432, "dialect": "postgres", "status": "tested", "category": "vector"},
    "redshift": {"label": "Amazon Redshift", "family": "sql", "driver": "postgresql+psycopg", "port": 5439, "dialect": "redshift", "status": "beta", "category": "warehouse"},
    "mysql": {"label": "MySQL", "family": "sql", "driver": "mysql+pymysql", "port": 3306, "dialect": "mysql", "status": "beta", "category": "database", "module": "pymysql"},
    "mariadb": {"label": "MariaDB", "family": "sql", "driver": "mysql+pymysql", "port": 3306, "dialect": "mysql", "status": "beta", "category": "database", "module": "pymysql"},
    "amazon_rds_mysql": {"label": "Amazon RDS for MySQL", "family": "sql", "driver": "mysql+pymysql", "port": 3306, "dialect": "mysql", "status": "beta", "category": "database", "module": "pymysql"},
    "sqlserver": {"label": "Microsoft SQL Server", "family": "sql", "driver": "mssql+pymssql", "port": 1433, "dialect": "tsql", "status": "beta", "category": "database", "module": "pymssql"},
    "azure_sql": {"label": "Azure SQL Database", "family": "sql", "driver": "mssql+pymssql", "port": 1433, "dialect": "tsql", "status": "beta", "category": "database", "module": "pymssql"},
    "oracle": {"label": "Oracle Database", "family": "sql", "driver": "oracle+oracledb", "port": 1521, "dialect": "oracle", "status": "beta", "category": "database", "module": "oracledb"},
    "sqlite": {"label": "SQLite", "family": "sql", "driver": "sqlite", "port": None, "dialect": "sqlite", "status": "tested", "category": "database"},
    "mongodb": {"label": "MongoDB", "family": "mongodb", "port": 27017, "status": "beta", "category": "nosql", "module": "pymongo"},
    "snowflake": {"label": "Snowflake", "family": "planned", "status": "planned", "category": "warehouse", "requires": "snowflake-sqlalchemy"},
    "bigquery": {"label": "Google BigQuery", "family": "planned", "status": "planned", "category": "warehouse", "requires": "sqlalchemy-bigquery + service account"},
    "databricks": {"label": "Databricks SQL", "family": "planned", "status": "planned", "category": "warehouse", "requires": "databricks-sql-connector"},
    "redis": {"label": "Redis", "family": "planned", "status": "planned", "category": "nosql", "requires": "key-pattern mapping design"},
    "elasticsearch": {"label": "Elasticsearch", "family": "planned", "status": "planned", "category": "nosql"},
    "opensearch": {"label": "OpenSearch", "family": "planned", "status": "planned", "category": "nosql"},
    "cassandra": {"label": "Apache Cassandra", "family": "planned", "status": "planned", "category": "nosql", "requires": "cassandra-driver"},
    "qdrant": {"label": "Qdrant", "family": "planned", "status": "planned", "category": "vector"},
    "weaviate": {"label": "Weaviate", "family": "planned", "status": "planned", "category": "vector"},
    "milvus": {"label": "Milvus", "family": "planned", "status": "planned", "category": "vector"},
    "pinecone": {"label": "Pinecone", "family": "planned", "status": "planned", "category": "vector"},
}

SYSTEM_SCHEMAS = {"information_schema", "pg_catalog", "pg_toast", "mysql", "performance_schema", "sys", "INFORMATION_SCHEMA", "guest", "db_owner", "db_accessadmin"}


@dataclass
class ConnectionSpec:
    provider: str
    host: str = ""
    port: int | None = None
    database: str = ""
    username: str = ""
    password: str | None = None
    ssl_mode: str = "prefer"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    duration_ms: float


def provider_info(provider: str) -> dict[str, Any]:
    info = PROVIDERS.get(provider)
    if info is None:
        raise ValidationFailed(f"Unknown database provider '{provider}'.")
    return info


def _module_available(name: str | None) -> bool:
    if not name:
        return True
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def catalogue() -> list[dict[str, Any]]:
    out = []
    for key, info in PROVIDERS.items():
        available = info["status"] != "planned" and _module_available(info.get("module"))
        out.append({"key": key, **{k: v for k, v in info.items() if k not in ("driver", "module")}, "driver_installed": available})
    return out


class Connector:
    def __init__(self, spec: ConnectionSpec) -> None:
        self.spec = spec
        self.info = provider_info(spec.provider)

    def test(self) -> float: ...
    def discover(self) -> dict[str, Any]: ...
    def execute(self, sql: str, *, max_rows: int, timeout_s: int) -> QueryResult: ...
    def iter_rows(self, table: str, columns: list[str], *, where: str | None, batch: int, limit: int, params: dict[str, Any] | None = None) -> Iterator[list[dict[str, Any]]]: ...

    def quote(self, identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'


def _json_safe(v: Any) -> Any:
    import datetime as _dt
    import decimal

    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(v))} bytes>"
    return str(v)


class SQLAlchemyConnector(Connector):
    def _url(self) -> str:
        s, info = self.spec, self.info
        if s.provider == "sqlite":
            path = os.path.abspath(s.database or s.host)
            allowed = [os.path.abspath(d) for d in get_settings().sqlite_allowed_dirs]
            if not any(os.path.commonpath([path, d]) == d for d in allowed):
                raise UnsafeRequest("SQLite files must live in an administrator-approved directory (SQLITE_ALLOWED_DIRS).")
            if not os.path.exists(path):
                raise ConnectorError("SQLite database file not found.")
            return f"sqlite:///file:{path}?mode=ro&uri=true"
        if not _module_available(info.get("module")):
            raise NotAvailable(f"The {info['label']} driver is not installed on this server (pip extra 'connectors').")
        port = s.port or info["port"]
        check_host(s.host, port, purpose="database")
        auth = quote_plus(s.username)
        if s.password:
            auth += ":" + quote_plus(s.password)
        return f"{info['driver']}://{auth}@{s.host}:{port}/{quote_plus(s.database)}"

    def _engine(self, timeout_s: int = 15) -> Any:
        from sqlalchemy import create_engine
        from sqlalchemy.pool import NullPool

        args: dict[str, Any] = {}
        family = self.info["driver"].split("+")[0]
        if family == "postgresql":
            args = {"connect_timeout": 10, "sslmode": self.spec.ssl_mode or "prefer", "options": f"-c statement_timeout={timeout_s * 1000} -c default_transaction_read_only=on"}
        elif family == "mysql":
            args = {"connect_timeout": 10, "read_timeout": timeout_s}
        elif family == "mssql":
            args = {"login_timeout": 10, "timeout": timeout_s}
        return create_engine(self._url(), poolclass=NullPool, connect_args=args)

    def _readonly(self, conn: Any) -> None:
        from sqlalchemy import text

        family = self.info["driver"].split("+")[0]
        if family == "postgresql":
            conn.execute(text("SET TRANSACTION READ ONLY"))
        elif family == "mysql":
            conn.execute(text("SET SESSION TRANSACTION READ ONLY"))

    def test(self) -> float:
        from sqlalchemy import text

        start = time.perf_counter()
        try:
            with self._engine(10).connect() as conn:
                conn.execute(text("SELECT 1" if self.spec.provider != "oracle" else "SELECT 1 FROM dual"))
        except (UnsafeRequest, NotAvailable):
            raise
        except Exception as exc:  # noqa: BLE001
            raise ConnectorError(_clean_error(exc, self.spec)) from exc
        return (time.perf_counter() - start) * 1000

    def discover(self) -> dict[str, Any]:
        from sqlalchemy import inspect, text

        try:
            engine = self._engine(30)
            insp = inspect(engine)
            schemas = [s for s in insp.get_schema_names() if s not in SYSTEM_SCHEMAS and not s.startswith("pg_temp") and not s.startswith("pg_toast")]
            if self.spec.provider == "sqlite":
                schemas = [None]  # type: ignore[list-item]
            estimates: dict[str, int] = {}
            if self.info["driver"].startswith("postgresql"):
                with engine.connect() as conn:
                    for nsp, rel, n in conn.execute(text("SELECT n.nspname, c.relname, c.reltuples::bigint FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.relkind IN ('r','p','v','m')")):
                        estimates[f"{nsp}.{rel}"] = max(int(n), 0)
            tables: dict[str, Any] = {}
            for schema in schemas:
                names = [(t, "table") for t in insp.get_table_names(schema=schema)] + [(v, "view") for v in insp.get_view_names(schema=schema)]
                for name, kind in names:
                    key = f"{schema}.{name}" if schema else name
                    cols = insp.get_columns(name, schema=schema)
                    pk = insp.get_pk_constraint(name, schema=schema).get("constrained_columns", []) if kind == "table" else []
                    fks = insp.get_foreign_keys(name, schema=schema) if kind == "table" else []
                    tables[key] = {
                        "schema": schema, "name": name, "kind": kind,
                        "columns": [{"name": c["name"], "type": str(c["type"]), "nullable": bool(c.get("nullable", True))} for c in cols],
                        "primary_key": pk,
                        "foreign_keys": [{"columns": fk["constrained_columns"], "references": f"{fk.get('referred_schema') + '.' if fk.get('referred_schema') else ''}{fk['referred_table']}", "referred_columns": fk["referred_columns"]} for fk in fks],
                        "row_estimate": estimates.get(key),
                    }
            return {"schemas": [s for s in schemas if s], "tables": tables}
        except (UnsafeRequest, NotAvailable):
            raise
        except Exception as exc:  # noqa: BLE001
            raise ConnectorError(_clean_error(exc, self.spec)) from exc

    def execute(self, sql: str, *, max_rows: int, timeout_s: int) -> QueryResult:
        from sqlalchemy import text

        start = time.perf_counter()
        try:
            with self._engine(timeout_s).connect() as conn:
                self._readonly(conn)
                result = conn.execute(text(sql))
                columns = list(result.keys())
                rows = result.fetchmany(max_rows + 1)
                conn.rollback()
        except Exception as exc:  # noqa: BLE001
            raise ConnectorError(_clean_error(exc, self.spec)) from exc
        truncated = len(rows) > max_rows
        return QueryResult(columns=columns, rows=[[_json_safe(v) for v in r] for r in rows[:max_rows]], truncated=truncated, duration_ms=(time.perf_counter() - start) * 1000)

    def quote(self, identifier: str) -> str:
        return self._engine(10).dialect.identifier_preparer.quote(identifier)

    def iter_rows(self, table: str, columns: list[str], *, where: str | None, batch: int, limit: int, params: dict[str, Any] | None = None) -> Iterator[list[dict[str, Any]]]:
        from sqlalchemy import text

        preparer = self._engine(60).dialect.identifier_preparer
        qualified = ".".join(preparer.quote(p) for p in table.split("."))
        cols = ", ".join(preparer.quote(c) for c in columns)
        sql = f"SELECT {cols} FROM {qualified}" + (f" WHERE {where}" if where else "")
        engine = self._engine(300)
        try:
            with engine.connect() as conn:
                self._readonly(conn)
                result = conn.execution_options(stream_results=True).execute(text(sql), params or {})
                seen = 0
                while seen < limit:
                    rows = result.fetchmany(min(batch, limit - seen))
                    if not rows:
                        break
                    seen += len(rows)
                    yield [{c: _json_safe(v) for c, v in zip(columns, r)} for r in rows]
        except Exception as exc:  # noqa: BLE001
            raise ConnectorError(_clean_error(exc, self.spec)) from exc


class MongoConnector(Connector):
    def _client(self) -> Any:
        if not _module_available("pymongo"):
            raise NotAvailable("The MongoDB driver (pymongo) is not installed on this server.")
        import pymongo

        port = self.spec.port or 27017
        check_host(self.spec.host, port, purpose="database")
        return pymongo.MongoClient(host=self.spec.host, port=port, username=self.spec.username or None, password=self.spec.password or None, serverSelectionTimeoutMS=8000, tls=self.spec.ssl_mode in ("require", "verify-full"))

    def test(self) -> float:
        start = time.perf_counter()
        try:
            self._client().admin.command("ping")
        except (UnsafeRequest, NotAvailable):
            raise
        except Exception as exc:  # noqa: BLE001
            raise ConnectorError(_clean_error(exc, self.spec)) from exc
        return (time.perf_counter() - start) * 1000

    def discover(self) -> dict[str, Any]:
        db = self._client()[self.spec.database]
        tables = {}
        for name in db.list_collection_names():
            sample = list(db[name].find({}, limit=50))
            fields: dict[str, str] = {}
            for doc in sample:
                for k, v in doc.items():
                    fields.setdefault(k, type(v).__name__)
            tables[name] = {"schema": None, "name": name, "kind": "collection", "columns": [{"name": k, "type": t, "nullable": True} for k, t in fields.items()], "primary_key": ["_id"], "foreign_keys": [], "row_estimate": db[name].estimated_document_count()}
        return {"schemas": [], "tables": tables}

    def execute(self, sql: str, *, max_rows: int, timeout_s: int) -> QueryResult:
        raise NotAvailable("Natural-language querying of MongoDB is not supported yet; use synchronisation instead.")

    def iter_rows(self, table: str, columns: list[str], *, where: str | None, batch: int, limit: int, params: dict[str, Any] | None = None) -> Iterator[list[dict[str, Any]]]:
        coll = self._client()[self.spec.database][table]
        projection = {c: 1 for c in columns}
        cursor = coll.find({}, projection=projection, limit=limit, batch_size=batch)
        buf: list[dict[str, Any]] = []
        for doc in cursor:
            buf.append({c: _json_safe(doc.get(c)) for c in columns})
            if len(buf) >= batch:
                yield buf
                buf = []
        if buf:
            yield buf


def build_connector(spec: ConnectionSpec) -> Connector:
    info = provider_info(spec.provider)
    if info["family"] == "sql":
        return SQLAlchemyConnector(spec)
    if info["family"] == "mongodb":
        return MongoConnector(spec)
    raise NotAvailable(f"{info['label']} connections are planned but not implemented yet" + (f" (requires {info['requires']})." if info.get("requires") else "."))


def _clean_error(exc: Exception, spec: ConnectionSpec) -> str:
    """Connection errors without credentials or full connection strings."""
    msg = str(getattr(exc, "orig", exc)).splitlines()[0] if str(exc) else type(exc).__name__
    if spec.password:
        msg = msg.replace(spec.password, "***")
    from app.utils.logging import redact

    return redact(msg)[:500]
