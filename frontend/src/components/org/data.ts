"use client";

import { useMemo } from "react";

import type { Department, Member, Page, Permission, Role, Team, Workspace } from "@/components/org/types";
import { type CurrentOrg, useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";

/** Shared directory lookups. Query keys match across pages so the cache is reused. */
export function useMembers(enabled = true) {
  const { can } = useAuth();
  return useApi<Page<Member>>(enabled && can("org.view") ? "/members" : null, { limit: 500 });
}

export function useTeams(enabled = true) {
  const { can } = useAuth();
  return useApi<Team[]>(enabled && can("org.view") ? "/teams" : null);
}

export function useRoles(enabled = true) {
  const { can } = useAuth();
  return useApi<Role[]>(enabled && can("org.view") ? "/roles" : null);
}

export function useDepartments(enabled = true) {
  const { can } = useAuth();
  return useApi<Department[]>(enabled && can("org.view") ? "/departments" : null);
}

export function useWorkspaces(enabled = true) {
  const { can } = useAuth();
  return useApi<Workspace[]>(enabled && can("workspace.view") ? "/workspaces" : null);
}

export function usePermissionCatalog() {
  return useApi<Permission[]>("/permissions", undefined, { staleTime: 10 * 60_000 });
}

/** id → entity maps for joining id-only API responses. */
export function useIndex<T>(rows: T[] | undefined, key: (row: T) => string) {
  return useMemo(() => {
    const m = new Map<string, T>();
    for (const r of rows ?? []) m.set(key(r), r);
    return m;
  }, [rows]); // eslint-disable-line react-hooks/exhaustive-deps
}

/** Mirrors Authorizer.can_assign_role on the server. */
export function canAssignRole(role: Pick<Role, "key" | "rank">, org: CurrentOrg | null): boolean {
  const perms = org?.permissions.org ?? [];
  const max = org?.permissions.max_rank ?? 0;
  if (role.key === "owner") return perms.includes("org.billing") && max >= 100;
  return perms.includes("admin.permissions") && role.rank <= max;
}

/** Workspace-scoped capability check (org-wide grant or a grant on that workspace). */
export function useWorkspaceCan() {
  const { org } = useAuth();
  return (perm: string, workspaceId: string | null | undefined) =>
    Boolean(org?.permissions.org.includes(perm) || (workspaceId && org?.permissions.workspaces[workspaceId]?.includes(perm)));
}

export function memberLabel(m: Pick<Member, "full_name" | "email"> | undefined | null, fallback = "Unknown user") {
  if (!m) return fallback;
  return m.full_name || m.email;
}

export function groupPermissions(catalog: Permission[] | undefined): { category: string; items: Permission[] }[] {
  const groups: { category: string; items: Permission[] }[] = [];
  const byCat = new Map<string, Permission[]>();
  for (const p of catalog ?? []) {
    let list = byCat.get(p.category);
    if (!list) {
      list = [];
      byCat.set(p.category, list);
      groups.push({ category: p.category, items: list });
    }
    list.push(p);
  }
  return groups;
}
