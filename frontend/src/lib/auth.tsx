"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { usePathname, useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { ApiError, get, getOrgId, hasAccessToken, post, refreshAccessToken, setAccessToken, setAuthLostHandler, setOrgId } from "@/lib/api";

export type User = {
  id: string;
  email: string;
  full_name: string;
  status: string;
  email_verified_at: string | null;
  is_platform_admin: boolean;
  mfa_enabled: boolean;
  last_login_at: string | null;
  locale: string;
  preferences: Record<string, unknown>;
  created_at: string;
};
export type OrgSummary = { id: string; name: string; slug: string; status: string; roles: string[] };
export type CurrentOrg = {
  organization: { id: string; name: string; slug: string; status: string; settings: Record<string, any>; created_at: string };
  permissions: { org: string[]; workspaces: Record<string, string[]>; max_rank: number; via_support: boolean };
  plan: string | null;
  member_count: number;
};

/** The organization's security policy is stopping this member (see enforce_org_policy). */
export type OrgBlock = { reason: "mfa_required" | "reauth_required"; message: string };

type AuthState = {
  status: "loading" | "authenticated" | "anonymous";
  user: User | null;
  orgs: OrgSummary[];
  org: CurrentOrg | null;
  orgId: string | null;
  orgBlock: OrgBlock | null;
  can: (perm: string) => boolean;
  canAnywhere: (perm: string) => boolean;
  switchOrg: (id: string) => void;
  signIn: (accessToken: string) => Promise<void>;
  signOut: () => Promise<void>;
  reload: () => Promise<void>;
};

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const qc = useQueryClient();
  const [booted, setBooted] = useState(false);
  const [authed, setAuthed] = useState(false);
  const [orgId, setOrgIdState] = useState<string | null>(null);

  useEffect(() => {
    setAuthLostHandler(() => {
      setAccessToken(null);
      setAuthed(false);
      qc.clear();
    });
    (async () => {
      const token = hasAccessToken() ? "present" : await refreshAccessToken();
      setAuthed(Boolean(token));
      setOrgIdState(getOrgId());
      setBooted(true);
    })();
  }, [qc]);

  const me = useQuery({
    queryKey: ["me"],
    queryFn: () => get<{ user: User; organizations: OrgSummary[] }>("/auth/me"),
    enabled: booted && authed,
    staleTime: 60_000,
  });

  const orgs = me.data?.organizations ?? [];
  useEffect(() => {
    if (!me.data) return;
    const stored = getOrgId();
    const valid = orgs.find((o) => o.id === stored);
    const next = valid ? valid.id : orgs[0]?.id ?? null;
    if (next !== stored) setOrgId(next);
    setOrgIdState(next);
  }, [me.data]); // eslint-disable-line react-hooks/exhaustive-deps

  const current = useQuery({
    queryKey: ["org", orgId],
    queryFn: () => get<CurrentOrg>("/orgs/current"),
    enabled: Boolean(authed && orgId),
    staleTime: 30_000,
    meta: { silent: true }, // a policy block is shown by the app shell, not as a toast
  });
  const err = current.error;
  const reason = err instanceof ApiError && err.status === 403 ? (err.details as { reason?: string } | undefined)?.reason : undefined;
  const orgBlock: OrgBlock | null = reason === "mfa_required" || reason === "reauth_required" ? { reason, message: err!.message } : null;

  const perms = useMemo(() => new Set(current.data?.permissions.org ?? []), [current.data]);
  const wsPerms = useMemo(() => {
    const s = new Set<string>();
    Object.values(current.data?.permissions.workspaces ?? {}).forEach((list) => list.forEach((p) => s.add(p)));
    return s;
  }, [current.data]);

  const switchOrg = useCallback(
    (id: string) => {
      setOrgId(id);
      setOrgIdState(id);
      qc.removeQueries({ predicate: (q) => q.queryKey[0] !== "me" });
    },
    [qc],
  );

  const signIn = useCallback(
    async (token: string) => {
      setAccessToken(token);
      setAuthed(true);
      await qc.invalidateQueries({ queryKey: ["me"] });
    },
    [qc],
  );

  const signOut = useCallback(async () => {
    try {
      await post("/auth/logout");
    } catch {
      /* already signed out */
    }
    setAccessToken(null);
    setAuthed(false);
    qc.clear();
  }, [qc]);

  const status: AuthState["status"] = !booted || (authed && (me.isLoading || (orgId && current.isLoading))) ? "loading" : authed && me.data ? "authenticated" : "anonymous";

  const value: AuthState = {
    status,
    user: me.data?.user ?? null,
    orgs,
    org: current.data ?? null,
    orgId,
    orgBlock,
    can: (p) => perms.has(p),
    canAnywhere: (p) => perms.has(p) || wsPerms.has(p),
    switchOrg,
    signIn,
    signOut,
    reload: async () => {
      await qc.invalidateQueries({ queryKey: ["me"] });
      await qc.invalidateQueries({ queryKey: ["org"] });
    },
  };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}

/** Redirects anonymous visitors to /login (client-side guard; the API enforces everything). */
export function useRequireAuth() {
  const auth = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  useEffect(() => {
    if (auth.status === "anonymous") router.replace(`/login?next=${encodeURIComponent(pathname)}`);
  }, [auth.status, pathname, router]);
  return auth;
}
