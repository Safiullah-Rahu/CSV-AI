"""Authorization engine.

Two layers:

1. **Capabilities** come from roles assigned to the user (directly or via a
   team) at organisation scope or workspace scope, with optional expiry. API
   keys narrow capabilities to their scopes.
2. **Resource privileges** (none < read < write < manage) for knowledge bases,
   documents, database connections and agents come from implicit rules
   (ownership, visibility, ``kb.access_all``) plus explicit resource grants to
   users, teams, departments, roles, workspaces or the whole organisation.
   An explicit *deny* caps the effective privilege below the denied level and
   always beats any allow, including implicit admin access.

Every retrieval path asks this engine for the authorized knowledge-base and
document filters *before* any content is read, so unauthorized text never
reaches ranking, the LLM context, citations, caches or exports.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.models import (
    Agent,
    DatabaseConnection,
    Document,
    KnowledgeBase,
    Membership,
    ResourceGrant,
    Role,
    RoleAssignment,
    RolePermission,
    SupportAccessGrant,
    TeamMember,
    WorkspaceMember,
)
from app.rbac.catalog import PRIVILEGE_LEVEL, ROLE_BY_KEY
from app.utils.errors import PermissionDenied

READ, WRITE, MANAGE = 1, 2, 3


@dataclass
class Principal:
    user_id: uuid.UUID
    email: str
    org_id: uuid.UUID | None = None
    is_platform_admin: bool = False
    session_id: uuid.UUID | None = None
    api_key_id: uuid.UUID | None = None
    api_scopes: frozenset[str] | None = None
    full_name: str = ""
    mfa_enabled: bool = False
    session_family_id: uuid.UUID | None = None  # sign-in this access token descends from


@dataclass
class AccessScope:
    """Pre-computed retrieval filter for one principal."""

    kb_levels: dict[uuid.UUID, int]
    denied_document_ids: set[uuid.UUID] = field(default_factory=set)
    restricted_allowed_document_ids: set[uuid.UUID] = field(default_factory=set)
    kb_manage_ids: set[uuid.UUID] = field(default_factory=set)

    def readable_kb_ids(self) -> list[uuid.UUID]:
        return [k for k, lvl in self.kb_levels.items() if lvl >= READ]


def _now() -> datetime:
    return datetime.now(UTC)


def _active(expires_at: datetime | None) -> bool:
    return expires_at is None or expires_at > _now()


class Authorizer:
    def __init__(self, session: Session, principal: Principal) -> None:
        self.session = session
        self.p = principal
        self._loaded = False
        self.is_member = False
        self.via_support = False
        self.org_perms: set[str] = set()
        self.ws_perms: dict[uuid.UUID, set[str]] = defaultdict(set)
        self.team_ids: set[uuid.UUID] = set()
        self.workspace_ids: set[uuid.UUID] = set()
        self.role_ids: set[uuid.UUID] = set()
        self.department_id: uuid.UUID | None = None
        self.max_rank = 0

    # ------------------------------------------------------------------ loading
    def load(self) -> Authorizer:
        if self._loaded:
            return self
        self._loaded = True
        org_id, user_id = self.p.org_id, self.p.user_id
        if org_id is None:
            return self
        s = self.session
        membership = s.scalar(
            select(Membership).where(
                Membership.org_id == org_id,
                Membership.user_id == user_id,
                Membership.status == "active",
            )
        )
        if membership is None:
            if self.p.is_platform_admin and self._support_grant_active():
                # Break-glass: read-only view of this organisation, audited by the caller.
                self.via_support = True
                self.org_perms = set(ROLE_BY_KEY["read_only"].permissions)
                self.max_rank = ROLE_BY_KEY["read_only"].rank
            return self
        self.is_member = True
        self.department_id = membership.department_id
        self.team_ids = set(
            s.scalars(
                select(TeamMember.team_id).where(
                    TeamMember.org_id == org_id, TeamMember.user_id == user_id
                )
            )
        )
        self.workspace_ids = set(
            s.scalars(
                select(WorkspaceMember.workspace_id).where(
                    WorkspaceMember.org_id == org_id, WorkspaceMember.user_id == user_id
                )
            )
        )
        subject_filter = RoleAssignment.user_id == user_id
        if self.team_ids:
            subject_filter = or_(subject_filter, RoleAssignment.team_id.in_(self.team_ids))
        rows = s.execute(
            select(RoleAssignment, Role.rank)
            .join(Role, Role.id == RoleAssignment.role_id)
            .where(RoleAssignment.org_id == org_id, subject_filter)
        ).all()
        assignments = [(a, rank) for a, rank in rows if _active(a.expires_at)]
        self.role_ids = {a.role_id for a, _ in assignments}
        perms_by_role: dict[uuid.UUID, set[str]] = defaultdict(set)
        if self.role_ids:
            for role_id, perm in s.execute(
                select(RolePermission.role_id, RolePermission.permission).where(
                    RolePermission.role_id.in_(self.role_ids)
                )
            ):
                perms_by_role[role_id].add(perm)
        for a, rank in assignments:
            self.max_rank = max(self.max_rank, rank)
            perms = perms_by_role[a.role_id]
            if a.scope_type == "workspace" and a.scope_id is not None:
                self.ws_perms[a.scope_id] |= perms
                self.workspace_ids.add(a.scope_id)
            else:
                self.org_perms |= perms
        if self.p.api_scopes is not None:
            self.org_perms &= self.p.api_scopes
            for ws in list(self.ws_perms):
                self.ws_perms[ws] &= self.p.api_scopes
        return self

    def _support_grant_active(self) -> bool:
        grant = self.session.scalar(
            select(SupportAccessGrant).where(
                SupportAccessGrant.org_id == self.p.org_id,
                SupportAccessGrant.platform_user_id == self.p.user_id,
                SupportAccessGrant.revoked_at.is_(None),
                SupportAccessGrant.expires_at > _now(),
            )
        )
        return grant is not None

    # ------------------------------------------------------------- capabilities
    def has(self, perm: str, workspace_id: uuid.UUID | None = None) -> bool:
        self.load()
        if perm in self.org_perms:
            return True
        return workspace_id is not None and perm in self.ws_perms.get(workspace_id, set())

    def has_anywhere(self, perm: str) -> bool:
        self.load()
        return perm in self.org_perms or any(perm in p for p in self.ws_perms.values())

    def require(self, perm: str, workspace_id: uuid.UUID | None = None) -> None:
        if not self.has(perm, workspace_id):
            raise PermissionDenied(f"Missing permission: {perm}")

    def effective_permissions(self) -> dict[str, object]:
        self.load()
        return {
            "org": sorted(self.org_perms),
            "workspaces": {str(k): sorted(v) for k, v in self.ws_perms.items()},
            "max_rank": self.max_rank,
            "via_support": self.via_support,
        }

    # ----------------------------------------------------------------- subjects
    def _subject_clause(self):
        self.load()
        clauses = [
            and_(ResourceGrant.subject_type == "user", ResourceGrant.subject_id == self.p.user_id),
            and_(ResourceGrant.subject_type == "org", ResourceGrant.subject_id.is_(None)),
        ]
        if self.team_ids:
            clauses.append(
                and_(ResourceGrant.subject_type == "team", ResourceGrant.subject_id.in_(self.team_ids))
            )
        if self.role_ids:
            clauses.append(
                and_(ResourceGrant.subject_type == "role", ResourceGrant.subject_id.in_(self.role_ids))
            )
        if self.workspace_ids:
            clauses.append(
                and_(
                    ResourceGrant.subject_type == "workspace",
                    ResourceGrant.subject_id.in_(self.workspace_ids),
                )
            )
        if self.department_id:
            clauses.append(
                and_(
                    ResourceGrant.subject_type == "department",
                    ResourceGrant.subject_id == self.department_id,
                )
            )
        return or_(*clauses)

    def _grant_levels(
        self, resource_type: str, resource_ids: Iterable[uuid.UUID]
    ) -> tuple[dict[uuid.UUID, int], dict[uuid.UUID, int]]:
        """Return (max allow level, min deny level) per resource for this principal."""
        ids = list(resource_ids)
        allow: dict[uuid.UUID, int] = {}
        deny: dict[uuid.UUID, int] = {}
        if not ids or not self.is_member:
            return allow, deny
        rows = self.session.scalars(
            select(ResourceGrant).where(
                ResourceGrant.org_id == self.p.org_id,
                ResourceGrant.resource_type == resource_type,
                ResourceGrant.resource_id.in_(ids),
                self._subject_clause(),
            )
        )
        for g in rows:
            if not _active(g.expires_at):
                continue
            level = PRIVILEGE_LEVEL[g.privilege]
            if g.effect == "deny":
                deny[g.resource_id] = min(deny.get(g.resource_id, 99), level)
            else:
                allow[g.resource_id] = max(allow.get(g.resource_id, 0), level)
        return allow, deny

    @staticmethod
    def _combine(implicit: int, allow: int, deny: int | None) -> int:
        level = max(implicit, allow)
        if deny is not None:
            level = min(level, deny - 1)
        return max(level, 0)

    # ------------------------------------------------------------ knowledge bases
    def kb_levels(self, kbs: Iterable[KnowledgeBase]) -> dict[uuid.UUID, int]:
        kbs = list(kbs)
        self.load()
        allow, deny = self._grant_levels("knowledge_base", (k.id for k in kbs))
        out: dict[uuid.UUID, int] = {}
        for kb in kbs:
            if not self.has("kb.view", kb.workspace_id):
                out[kb.id] = 0
                continue
            implicit = 0
            if self.is_member:
                if kb.owner_id == self.p.user_id:
                    implicit = MANAGE
                if self.has("kb.access_all", kb.workspace_id):
                    implicit = MANAGE
                if kb.visibility == "org" or kb.visibility == "workspace" and kb.workspace_id in self.workspace_ids:
                    implicit = max(implicit, READ)
            elif self.via_support:
                implicit = READ
            out[kb.id] = self._combine(implicit, allow.get(kb.id, 0), deny.get(kb.id))
        return out

    def kb_level(self, kb: KnowledgeBase) -> int:
        return self.kb_levels([kb])[kb.id]

    def require_kb(self, kb: KnowledgeBase, level: int, perm: str | None = None) -> None:
        if perm and not self.has(perm, kb.workspace_id):
            raise PermissionDenied(f"Missing permission: {perm}")
        if self.kb_level(kb) < level:
            raise PermissionDenied("You do not have access to this knowledge base.")

    def access_scope(self, kb_ids: Iterable[uuid.UUID] | None = None) -> AccessScope:
        """Authorized KB + document filters for retrieval (one call per request)."""
        self.load()
        query = select(KnowledgeBase).where(
            KnowledgeBase.org_id == self.p.org_id, KnowledgeBase.status == "active"
        )
        if kb_ids is not None:
            ids = list(kb_ids)
            if not ids:
                return AccessScope(kb_levels={})
            query = query.where(KnowledgeBase.id.in_(ids))
        kbs = list(self.session.scalars(query))
        levels = {k: v for k, v in self.kb_levels(kbs).items() if v >= READ}
        scope = AccessScope(kb_levels=levels, kb_manage_ids={k for k, v in levels.items() if v >= MANAGE})
        if not levels:
            return scope
        # Document-level grants inside the readable knowledge bases.
        doc_ids = list(
            self.session.scalars(
                select(ResourceGrant.resource_id)
                .join(Document, Document.id == ResourceGrant.resource_id)
                .where(
                    ResourceGrant.org_id == self.p.org_id,
                    ResourceGrant.resource_type == "document",
                    Document.kb_id.in_(list(levels)),
                )
            )
        )
        restricted = set(
            self.session.scalars(
                select(Document.id).where(
                    Document.kb_id.in_(list(levels)), Document.acl_mode == "restricted"
                )
            )
        )
        allow, deny = self._grant_levels("document", set(doc_ids) | restricted)
        scope.denied_document_ids = {d for d, lvl in deny.items() if lvl <= READ}
        scope.restricted_allowed_document_ids = {d for d, lvl in allow.items() if lvl >= READ}
        return scope

    def document_level(self, doc: Document, kb_level: int | None = None) -> int:
        if kb_level is None:
            kb = self.session.get(KnowledgeBase, doc.kb_id)
            kb_level = self.kb_level(kb) if kb else 0
        allow, deny = self._grant_levels("document", [doc.id])
        implicit = kb_level
        if doc.acl_mode == "restricted" and kb_level < MANAGE and doc.owner_id != self.p.user_id:
            implicit = 0
        return self._combine(implicit, allow.get(doc.id, 0) if kb_level >= READ else 0, deny.get(doc.id))

    # --------------------------------------------------------- other resources
    def connection_level(self, conn: DatabaseConnection) -> int:
        self.load()
        if not self.has("db.view", conn.workspace_id):
            return 0
        implicit = 0
        if self.has("db.edit", conn.workspace_id) or conn.owner_id == self.p.user_id:
            implicit = MANAGE
        allow, deny = self._grant_levels("db_connection", [conn.id])
        return self._combine(implicit, allow.get(conn.id, 0), deny.get(conn.id))

    def connection_table_constraint(self, conn: DatabaseConnection) -> set[str] | None:
        """Tables this principal is limited to by grant constraints (None = no narrowing)."""
        if self.connection_level(conn) >= MANAGE:
            return None
        rows = self.session.scalars(
            select(ResourceGrant).where(
                ResourceGrant.org_id == self.p.org_id,
                ResourceGrant.resource_type == "db_connection",
                ResourceGrant.resource_id == conn.id,
                ResourceGrant.effect == "allow",
                self._subject_clause(),
            )
        )
        tables: set[str] = set()
        unconstrained = False
        for g in rows:
            if not _active(g.expires_at):
                continue
            listed = g.constraints.get("tables") if g.constraints else None
            if listed:
                tables |= set(listed)
            else:
                unconstrained = True
        return None if unconstrained or not tables else tables

    def agent_level(self, agent: Agent) -> int:
        self.load()
        implicit = 0
        if self.has("agents.manage", agent.workspace_id):
            implicit = MANAGE
        elif self.has("agents.use", agent.workspace_id) and agent.enabled:
            implicit = READ
        allow, deny = self._grant_levels("agent", [agent.id])
        return self._combine(implicit, allow.get(agent.id, 0), deny.get(agent.id))

    def can_assign_role(self, role: Role) -> bool:
        self.load()
        if role.key == "owner":
            return "org.billing" in self.org_perms and self.max_rank >= 100
        return self.has("admin.permissions") and role.rank <= self.max_rank
