"""Permission catalogue and predefined roles.

Permissions are *capabilities* ("may upload documents"). Which concrete
knowledge bases, documents and connections a capability applies to is decided
by resource grants (see :mod:`app.rbac.authorizer`).
"""

from __future__ import annotations

from dataclasses import dataclass

PERMISSIONS: dict[str, tuple[str, str]] = {
    # key: (category, description)
    "org.view": ("Organization", "View organization"),
    "org.edit": ("Organization", "Edit organization profile"),
    "org.settings": ("Organization", "Manage organization settings"),
    "org.billing": ("Organization", "Manage billing and subscription"),
    "org.users": ("Organization", "Manage members and invitations"),
    "org.integrations": ("Organization", "Manage external integrations"),
    "org.security": ("Organization", "Manage security policies and sessions"),
    "workspace.view": ("Workspace", "View workspaces"),
    "workspace.create": ("Workspace", "Create workspaces"),
    "workspace.edit": ("Workspace", "Edit workspaces"),
    "workspace.delete": ("Workspace", "Delete workspaces"),
    "workspace.members": ("Workspace", "Manage workspace members"),
    "kb.view": ("Knowledge base", "View knowledge bases"),
    "kb.create": ("Knowledge base", "Create knowledge bases"),
    "kb.edit": ("Knowledge base", "Edit knowledge bases"),
    "kb.delete": ("Knowledge base", "Delete knowledge bases"),
    "kb.upload": ("Knowledge base", "Upload documents"),
    "kb.delete_documents": ("Knowledge base", "Delete documents"),
    "kb.reprocess": ("Knowledge base", "Reprocess documents"),
    "kb.configure_ingestion": ("Knowledge base", "Configure ingestion and chunking"),
    "kb.configure_retrieval": ("Knowledge base", "Configure retrieval"),
    "kb.chat": ("Knowledge base", "Chat with knowledge bases"),
    "kb.export": ("Knowledge base", "Export knowledge base content"),
    "kb.permissions": ("Knowledge base", "Manage knowledge base permissions"),
    "kb.access_all": ("Knowledge base", "Access every knowledge base (unless explicitly denied)"),
    "db.view": ("Database connections", "View database connections"),
    "db.create": ("Database connections", "Create database connections"),
    "db.edit": ("Database connections", "Edit database connections"),
    "db.test": ("Database connections", "Test database connections"),
    "db.sync": ("Database connections", "Synchronize data"),
    "db.schemas": ("Database connections", "Manage schemas and table selection"),
    "db.credentials": ("Database connections", "Manage credentials"),
    "db.delete": ("Database connections", "Delete database connections"),
    "db.query": ("Database connections", "Run read-only queries"),
    "db.approve": ("Database connections", "Approve sensitive or expensive queries"),
    "integrations.view": ("Integrations", "View integrations"),
    "conv.create": ("Conversations", "Create conversations"),
    "conv.view_own": ("Conversations", "View own conversations"),
    "conv.view_team": ("Conversations", "View team conversations"),
    "conv.view_authorized": ("Conversations", "View all authorized conversations"),
    "conv.share": ("Conversations", "Share conversations"),
    "conv.export": ("Conversations", "Export conversations"),
    "conv.delete": ("Conversations", "Delete conversations"),
    "agents.view": ("Agents", "View agents"),
    "agents.use": ("Agents", "Chat with agents"),
    "agents.manage": ("Agents", "Create and configure agents"),
    "analytics.view": ("Analytics", "View analytics and usage"),
    "admin.users": ("Administration", "Manage users (suspend, deactivate)"),
    "admin.roles": ("Administration", "Manage roles"),
    "admin.permissions": ("Administration", "Manage role assignments and permissions"),
    "admin.api_keys": ("Administration", "Manage API keys"),
    "admin.audit": ("Administration", "View audit logs"),
    "admin.ai_providers": ("Administration", "Configure AI providers"),
    "admin.platform_settings": ("Administration", "Manage webhooks and developer settings"),
}

ALL_PERMISSIONS: frozenset[str] = frozenset(PERMISSIONS)


@dataclass(frozen=True)
class RoleTemplate:
    key: str
    name: str
    description: str
    rank: int
    permissions: frozenset[str]


def _p(*keys: str) -> frozenset[str]:
    unknown = set(keys) - ALL_PERMISSIONS
    if unknown:
        raise ValueError(f"Unknown permissions: {sorted(unknown)}")
    return frozenset(keys)


_KB_ALL = tuple(k for k in PERMISSIONS if k.startswith("kb."))
_DB_ALL = tuple(k for k in PERMISSIONS if k.startswith("db."))
_CONV_ALL = tuple(k for k in PERMISSIONS if k.startswith("conv."))
_WS_ALL = tuple(k for k in PERMISSIONS if k.startswith("workspace."))

ROLE_TEMPLATES: tuple[RoleTemplate, ...] = (
    RoleTemplate("owner", "Organization Owner", "Full control including billing.", 100, ALL_PERMISSIONS),
    RoleTemplate(
        "admin",
        "Organization Admin",
        "Manages the organization except billing.",
        90,
        ALL_PERMISSIONS - {"org.billing"},
    ),
    RoleTemplate(
        "security_admin",
        "Security Administrator",
        "Security policies, roles, sessions, API keys and audit — no content access.",
        80,
        _p(
            "org.view", "org.security", "workspace.view", "admin.users", "admin.roles",
            "admin.permissions", "admin.api_keys", "admin.audit", "kb.view", "kb.permissions",
            "db.view", "integrations.view", "agents.view", "analytics.view",
        ),
    ),
    RoleTemplate(
        "workspace_admin",
        "Workspace Admin",
        "Manages workspaces and their knowledge bases (assign at workspace scope).",
        70,
        _p("org.view", *_WS_ALL, *_KB_ALL, "agents.view", "agents.use", "agents.manage",
           "conv.create", "conv.view_own", "conv.view_team", "conv.share", "conv.export",
           "conv.delete", "db.view", "db.query", "integrations.view", "analytics.view"),
    ),
    RoleTemplate(
        "kb_manager",
        "Knowledge Base Manager",
        "Creates and curates knowledge bases they own or are granted.",
        60,
        _p("org.view", "workspace.view", *(k for k in _KB_ALL if k != "kb.access_all"),
           "conv.create", "conv.view_own", "conv.share", "conv.export", "conv.delete",
           "agents.view", "agents.use", "integrations.view"),
    ),
    RoleTemplate(
        "data_source_manager",
        "Data Source Manager",
        "Connects databases and integrations and runs synchronisation.",
        60,
        _p("org.view", "workspace.view", *_DB_ALL, "integrations.view", "org.integrations",
           "kb.view", "kb.chat", "conv.create", "conv.view_own", "agents.view"),
    ),
    RoleTemplate(
        "developer",
        "Developer",
        "Builds on the API: API keys, queries, agents.",
        50,
        _p("org.view", "workspace.view", "kb.view", "kb.chat", "db.view", "db.query",
           "integrations.view", "agents.view", "agents.use", "admin.api_keys", "conv.create",
           "conv.view_own", "conv.share", "conv.export", "conv.delete"),
    ),
    RoleTemplate(
        "analyst",
        "Analyst",
        "Asks questions, queries data and reads analytics.",
        40,
        _p("org.view", "workspace.view", "kb.view", "kb.chat", "kb.export", "db.view",
           "db.query", "agents.view", "agents.use", "analytics.view", "conv.create",
           "conv.view_own", "conv.share", "conv.export", "conv.delete", "integrations.view"),
    ),
    RoleTemplate(
        "member",
        "Standard Member",
        "Chats with the knowledge bases they are granted.",
        30,
        _p("org.view", "workspace.view", "kb.view", "kb.chat", "agents.view", "agents.use",
           "conv.create", "conv.view_own", "conv.share", "conv.export", "conv.delete"),
    ),
    RoleTemplate(
        "read_only",
        "Read-Only User",
        "Views granted knowledge bases and own conversations.",
        10,
        _p("org.view", "workspace.view", "kb.view", "conv.view_own"),
    ),
)

ROLE_BY_KEY: dict[str, RoleTemplate] = {r.key: r for r in ROLE_TEMPLATES}

PRIVILEGE_LEVEL: dict[str, int] = {"none": 0, "read": 1, "write": 2, "manage": 3}
LEVEL_NAME: dict[int, str] = {v: k for k, v in PRIVILEGE_LEVEL.items()}

RESOURCE_TYPES = ("knowledge_base", "document", "db_connection", "integration", "agent", "dataset")
SUBJECT_TYPES = ("user", "team", "department", "role", "workspace", "org")
