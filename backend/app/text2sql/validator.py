"""SQL safety validator and rewriter.

Never executes anything. Given SQL (typically LLM-generated) and a policy
derived from the connection's table selection and the requesting user's
grants, it either rejects the query or returns a rewritten, safe query:

* exactly one statement, read-only (SELECT / set operations / CTEs);
* no DDL/DML/session/transaction/admin commands anywhere in the tree;
* no dangerous functions (sleep, file and network readers, config setters);
* only allow-listed tables, only allowed (non-sensitive) columns;
* ``SELECT *`` expanded to the allowed columns;
* admin row filters injected by wrapping each table in a filtered subquery;
* a hard row ``LIMIT``; join-count limit; cross joins and sensitive tables
  routed to human approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

_FORBIDDEN_NODES = tuple(
    getattr(exp, n)
    for n in (
        "Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable", "Merge", "TruncateTable", "Command", "Into",
        "Set", "Pragma", "Use", "Copy", "Transaction", "Commit", "Rollback", "Grant", "LoadData", "Describe", "Show",
        "Attach", "Detach", "Install", "Kill", "Analyze", "Export", "Lock", "Cache", "Uncache", "Refresh",
    )
    if hasattr(exp, n)
)

_FORBIDDEN_FUNCTIONS = {
    # sleeps / DoS
    "pg_sleep", "pg_sleep_for", "pg_sleep_until", "sleep", "benchmark", "waitfor", "generate_series",
    # filesystem / network / large objects
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file", "lo_import", "lo_export", "lo_get", "lo_put",
    "load_file", "dblink", "dblink_exec", "dblink_connect", "openrowset", "opendatasource", "openquery", "xp_cmdshell",
    "sp_executesql", "utl_http", "utl_file", "httpget", "read_csv", "read_csv_auto", "read_parquet", "read_json",
    "read_json_auto", "read_ndjson", "read_text", "read_blob", "glob", "parquet_scan", "csv_scan", "sniff_csv",
    "query", "query_table", "sqlite_scan", "postgres_scan", "mysql_scan",
    # configuration / session / admin
    "set_config", "current_setting", "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf", "pg_rotate_logfile",
    "txid_current", "nextval", "setval", "currval", "pg_advisory_lock", "pg_advisory_xact_lock",
}

_FUNCTION_NODE_NAMES = {"readcsv": "read_csv", "readparquet": "read_parquet"}


@dataclass
class TableSpec:
    name: str                                  # as referenced in SQL, e.g. "public.orders" or "orders"
    columns: list[str]                         # allowed (non-sensitive, non-excluded) columns
    row_filter: str | None = None              # SQL boolean expression over this table's columns
    requires_approval: bool = False

    @property
    def bare(self) -> str:
        return self.name.split(".")[-1].lower()


@dataclass
class Policy:
    tables: list[TableSpec]
    dialect: str = "postgres"
    max_rows: int = 500
    max_joins: int = 4
    max_subquery_depth: int = 3
    require_approval_for_cross_join: bool = True

    def lookup(self, table: exp.Table) -> TableSpec | None:
        parts = [p for p in (table.catalog, table.db, table.name) if p]
        full = ".".join(parts).lower()
        for spec in self.tables:
            if spec.name.lower() == full:
                return spec
        bare = table.name.lower()
        matches = [s for s in self.tables if s.bare == bare and ("." not in full)]
        return matches[0] if len(matches) == 1 else None


@dataclass
class Validation:
    ok: bool
    sql: str = ""
    errors: list[str] = field(default_factory=list)
    needs_approval: bool = False
    approval_reasons: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)


def _reject(*errors: str) -> Validation:
    return Validation(ok=False, errors=list(errors))


def _function_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    key = type(node).__name__.lower()
    return _FUNCTION_NODE_NAMES.get(key, node.sql_name().lower() if hasattr(node, "sql_name") else key)


def _cte_names(tree: exp.Expression) -> set[str]:
    return {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}


def _single_row(query: exp.Expression, ctes: dict[str, exp.Expression], seen: frozenset[str]) -> bool:
    """True when a query returns at most one row: a constant SELECT, an aggregate without GROUP BY, LIMIT 1, or a
    SELECT without joins over another such source (a date window in a CTE, for example)."""
    if not isinstance(query, exp.Select) or query.args.get("joins") or query.args.get("group"):
        return False
    limit = query.args.get("limit")
    if limit is not None and isinstance(limit.expression, exp.Literal) and limit.expression.this == "1":
        return True
    if query.expressions and all(isinstance(e.unalias(), exp.AggFunc) for e in query.expressions):
        return True
    source = query.args.get("from_") or query.args.get("from")
    return source is None or _single_row_source(source.this, ctes, seen)


def _single_row_source(node: exp.Expression, ctes: dict[str, exp.Expression], seen: frozenset[str] = frozenset()) -> bool:
    if isinstance(node, exp.Subquery):
        return _single_row(node.this, ctes, seen)
    if isinstance(node, exp.Table) and not node.db and node.name.lower() in ctes and node.name.lower() not in seen:
        return _single_row(ctes[node.name.lower()], ctes, seen | {node.name.lower()})
    return False


def _depth(node: exp.Expression) -> int:
    depth, cur = 0, node.parent
    while cur is not None:
        if isinstance(cur, (exp.Subquery, exp.Select)) and cur is not node:
            depth += 1
        cur = cur.parent
    return depth


def validate(sql: str, policy: Policy) -> Validation:
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return _reject("Empty query.")
    if len(sql) > 20_000:
        return _reject("Query is too long.")
    try:
        statements = [s for s in sqlglot.parse(sql, read=policy.dialect) if s is not None]
    except ParseError as exc:
        return _reject(f"SQL could not be parsed: {str(exc).splitlines()[0][:200]}")
    if len(statements) != 1:
        return _reject("Exactly one SQL statement is allowed.")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)):
        return _reject(f"Only read-only SELECT queries are allowed (got {type(tree).__name__.upper()}).")
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            return _reject(f"Forbidden operation: {type(node).__name__.upper()}.")
        if isinstance(node, exp.Func):
            name = _function_name(node)
            if name in _FORBIDDEN_FUNCTIONS:
                return _reject(f"Function '{name}' is not allowed.")
        if isinstance(node, exp.Table) and not isinstance(node.this, exp.Identifier):
            return _reject("Table functions and file readers are not allowed.")
    ctes = _cte_names(tree)
    errors: list[str] = []
    approval: list[str] = []
    used: dict[str, TableSpec] = {}
    alias_to_spec: dict[str, TableSpec] = {}
    for table in tree.find_all(exp.Table):
        if table.name.lower() in ctes and not table.db:
            continue
        spec = policy.lookup(table)
        if spec is None:
            errors.append(f"Table '{table.sql(dialect=policy.dialect)}' is not available.")
            continue
        used[spec.name] = spec
        alias_to_spec[(table.alias or table.name).lower()] = spec
        alias_to_spec[spec.bare] = spec
        if spec.requires_approval:
            approval.append(f"Table {spec.name} requires approval.")
        if _depth(table) > policy.max_subquery_depth + 1:
            errors.append("Query nesting is too deep.")
    if errors:
        return Validation(ok=False, errors=sorted(set(errors)))
    if not used:
        return _reject("The query does not read any allowed table.")
    # Columns: every real-table column reference must be allowed.
    select_aliases = {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}
    cte_columns: set[str] = set()
    for cte in tree.find_all(exp.CTE):
        for sel in cte.find_all(exp.Select):
            cte_columns |= {e.alias_or_name.lower() for e in sel.expressions}
        if cte.args.get("alias") is not None and cte.args["alias"].columns:
            cte_columns |= {c.name.lower() for c in cte.args["alias"].columns}
    subquery_aliases = {s.alias.lower() for s in tree.find_all(exp.Subquery) if s.alias}
    allowed_any = {c.lower() for s in used.values() for c in s.columns}
    for col in tree.find_all(exp.Column):
        name = col.name.lower()
        if not name:
            continue
        qualifier = (col.table or "").lower()
        if qualifier and qualifier in alias_to_spec and qualifier not in ctes and qualifier not in subquery_aliases:
            spec = alias_to_spec[qualifier]
            if name not in {c.lower() for c in spec.columns}:
                errors.append(f"Column '{qualifier}.{name}' is not available.")
        elif not qualifier:
            if name not in allowed_any and name not in select_aliases and name not in cte_columns:
                errors.append(f"Column '{name}' is not available.")
    if errors:
        return Validation(ok=False, errors=sorted(set(errors)))
    # Joins.
    joins = list(tree.find_all(exp.Join))
    if len(joins) > policy.max_joins:
        return _reject(f"Too many joins ({len(joins)} > {policy.max_joins}).")
    cte_queries = {cte.alias_or_name.lower(): cte.this for cte in tree.find_all(exp.CTE)}
    for j in joins:
        # A cross join against a one-row source (a date window, a total) cannot multiply rows, so it needs no approval.
        if not j.args.get("on") and not j.args.get("using") and policy.require_approval_for_cross_join and not _single_row_source(j.this, cte_queries):
            approval.append("Cross join (no join condition) requires approval.")
    # Expand stars to allowed columns (prevents reading excluded/sensitive columns).
    for star_parent in list(tree.find_all(exp.Select)):
        new_exprs: list[exp.Expression] = []
        changed = False
        for e in star_parent.expressions:
            if isinstance(e, exp.Star) or (isinstance(e, exp.Column) and isinstance(e.this, exp.Star)):
                qualifier = (e.table or "").lower() if isinstance(e, exp.Column) else ""
                sources = [t for t in star_parent.find_all(exp.Table) if t.parent_select is star_parent]
                specs = []
                for t in sources:
                    key = (t.alias or t.name).lower()
                    if (not qualifier or qualifier == key) and key in alias_to_spec and t.name.lower() not in ctes:
                        specs.append((key, alias_to_spec[key]))
                if not specs:
                    new_exprs.append(e)
                    continue
                changed = True
                multi = len(specs) > 1
                for key, spec in specs:
                    for c in spec.columns:
                        new_exprs.append(exp.column(c, table=key if multi or qualifier else None, quoted=True))
            else:
                new_exprs.append(e)
        if changed:
            star_parent.set("expressions", new_exprs)
    # Row filters: wrap each filtered table reference in a filtered subquery.
    for table in list(tree.find_all(exp.Table)):
        if table.name.lower() in ctes and not table.db:
            continue
        spec = policy.lookup(table)
        if spec is None or not spec.row_filter:
            continue
        alias = table.alias or table.name
        try:
            condition = sqlglot.parse_one(spec.row_filter, read=policy.dialect, into=exp.Condition)
        except ParseError:
            return _reject(f"Row filter for {spec.name} is invalid; contact an administrator.")
        bare = exp.Table(this=exp.to_identifier(table.name, quoted=True), db=exp.to_identifier(table.db, quoted=True) if table.db else None, catalog=exp.to_identifier(table.catalog, quoted=True) if table.catalog else None)
        sub = exp.select("*").from_(bare).where(condition).subquery(alias=exp.to_identifier(alias, quoted=True))
        table.replace(sub)
    # Enforce the row limit on the outermost query.
    limit_node = tree.args.get("limit")
    if isinstance(tree, exp.Select):
        current = None
        if limit_node is not None and isinstance(limit_node.expression, exp.Literal) and limit_node.expression.is_int:
            current = int(limit_node.expression.this)
        if current is None or current > policy.max_rows:
            tree = tree.limit(policy.max_rows)
    else:
        tree = exp.select("*").from_(tree.subquery(alias="q")).limit(policy.max_rows)
    out = tree.sql(dialect=policy.dialect, comments=False)
    return Validation(ok=True, sql=out, needs_approval=bool(approval), approval_reasons=sorted(set(approval)), tables=sorted(used))


def validate_row_filter(expression: str, columns: list[str], dialect: str = "postgres") -> list[str]:
    """Admin-side check for row-filter expressions."""
    try:
        cond = sqlglot.parse_one(expression, read=dialect, into=exp.Condition)
    except ParseError as exc:
        return [f"Invalid filter: {str(exc).splitlines()[0][:200]}"]
    problems = []
    allowed = {c.lower() for c in columns}
    for node in cond.walk():
        if isinstance(node, (exp.Select, exp.Subquery)) or isinstance(node, _FORBIDDEN_NODES):
            problems.append("Filters cannot contain subqueries or statements.")
        if isinstance(node, exp.Func) and _function_name(node) in _FORBIDDEN_FUNCTIONS:
            problems.append(f"Function '{_function_name(node)}' is not allowed in filters.")
        if isinstance(node, exp.Column) and node.name.lower() not in allowed:
            problems.append(f"Unknown column '{node.name}' in filter.")
    return sorted(set(problems))
