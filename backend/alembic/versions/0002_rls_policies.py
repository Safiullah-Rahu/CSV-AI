"""Row-Level Security, append-only audit trail, vector indexes, plan seeds.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-02
"""
from __future__ import annotations

import json

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

TENANT_TABLES = [
    "activity_events", "agents", "ai_providers", "api_keys", "approvals", "audit_events",
    "chunk_embeddings", "chunks", "citations", "conversation_shares", "conversations",
    "credentials", "database_connections", "datasets", "departments", "document_versions",
    "documents", "health_checks", "integrations", "invitations", "jobs", "kb_config_revisions",
    "knowledge_bases", "memberships", "message_feedback", "messages", "notifications",
    "resource_grants", "role_assignments", "role_permissions", "roles", "sql_query_logs",
    "subscriptions", "support_access_grants", "sync_runs", "team_members", "teams",
    "usage_records", "webhook_deliveries", "webhooks", "workspace_members", "workspaces",
]

IN_ORG = "org_id::text = current_setting('app.current_org', true)"
BYPASS = "current_setting('app.rls_bypass', true) = 'on'"
SELF_USER = "user_id::text = current_setting('app.current_user', true)"

PLANS = [
    ("free", "Free", {"users": 3, "knowledge_bases": 2, "storage_mb": 500, "documents": 200,
                      "monthly_tokens": 200_000, "api_calls_per_day": 500, "connected_sources": 1},
     ["chat", "documents"], 0, 0),
    ("professional", "Professional", {"users": 25, "knowledge_bases": 20, "storage_mb": 20_000,
                                      "documents": 10_000, "monthly_tokens": 5_000_000,
                                      "api_calls_per_day": 20_000, "connected_sources": 10},
     ["chat", "documents", "databases", "integrations", "api"], 49, 1),
    ("business", "Business", {"users": 250, "knowledge_bases": 200, "storage_mb": 250_000,
                              "documents": 250_000, "monthly_tokens": 50_000_000,
                              "api_calls_per_day": 250_000, "connected_sources": 100},
     ["chat", "documents", "databases", "integrations", "api", "agents", "audit", "sso"], 299, 2),
    ("enterprise", "Enterprise", {}, ["*"], None, 3),
]


def upgrade() -> None:
    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        using = f"({IN_ORG} OR {BYPASS})"
        if table == "memberships":
            using = f"({IN_ORG} OR {BYPASS} OR {SELF_USER})"
        check = f"({IN_ORG} OR {BYPASS})"
        if table == "audit_events":
            # Platform-level events (registration, login) have no organisation. They may be
            # written from any context but are readable only by system/platform code.
            check = f"({IN_ORG} OR {BYPASS} OR org_id IS NULL)"
        op.execute(f"CREATE POLICY tenant_isolation ON {table} USING {using} WITH CHECK {check}")

    # Organisations: visible inside their own context or to their members.
    op.execute("ALTER TABLE organizations ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE organizations FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY org_visibility ON organizations USING ("
        "id::text = current_setting('app.current_org', true) OR "
        f"{BYPASS} OR EXISTS (SELECT 1 FROM memberships m WHERE m.org_id = organizations.id "
        "AND m.user_id::text = current_setting('app.current_user', true))) "
        f"WITH CHECK (id::text = current_setting('app.current_org', true) OR {BYPASS})"
    )

    # Audit trail is append-only. Deletes are allowed only for system retention/org purge.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION audit_events_append_only() RETURNS trigger AS $$
        BEGIN
          IF TG_OP = 'DELETE' AND current_setting('app.rls_bypass', true) = 'on' THEN
            RETURN OLD;
          END IF;
          RAISE EXCEPTION 'audit_events is append-only';
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "CREATE TRIGGER audit_events_append_only BEFORE UPDATE OR DELETE ON audit_events "
        "FOR EACH ROW EXECUTE FUNCTION audit_events_append_only()"
    )

    # Approximate-NN indexes for common embedding sizes (others are created on demand).
    for dim in (384, 768, 1024, 1536):
        op.execute(
            f"CREATE INDEX IF NOT EXISTS ix_chunk_embeddings_hnsw_{dim} ON chunk_embeddings "
            f"USING hnsw ((embedding::vector({dim})) vector_cosine_ops) WHERE dim = {dim}"
        )

    for key, name, limits, features, price, sort in PLANS:
        feature_sql = "ARRAY[" + ",".join(f"'{f}'" for f in features) + "]::varchar[]"
        price_sql = "NULL" if price is None else str(price)
        op.execute(
            "INSERT INTO plans (key, name, limits, features, price_monthly_usd, is_public, sort) "
            f"VALUES ('{key}', '{name}', '{json.dumps(limits)}'::jsonb, {feature_sql}, "
            f"{price_sql}, true, {sort}) ON CONFLICT (key) DO NOTHING"
        )


def downgrade() -> None:
    op.execute("DELETE FROM plans")
    for dim in (384, 768, 1024, 1536):
        op.execute(f"DROP INDEX IF EXISTS ix_chunk_embeddings_hnsw_{dim}")
    op.execute("DROP TRIGGER IF EXISTS audit_events_append_only ON audit_events")
    op.execute("DROP FUNCTION IF EXISTS audit_events_append_only()")
    op.execute("DROP POLICY IF EXISTS org_visibility ON organizations")
    op.execute("ALTER TABLE organizations DISABLE ROW LEVEL SECURITY")
    for table in TENANT_TABLES:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
