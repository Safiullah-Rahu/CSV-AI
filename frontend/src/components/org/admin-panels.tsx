"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Ban, Building2, CheckCircle2, CircleSlash, CreditCard, Database, HardDrive, RefreshCw, RotateCcw, Shield, ShieldOff, Users, XCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { CancelButton, formatDuration, KV, Pager, PersonCell, RowMenuButton, SearchBox, useDebounced, usd, When } from "@/components/org/kit";
import type { Page, Plan } from "@/components/org/types";
import { Button } from "@/components/ui/button";
import { type Column, DataTable } from "@/components/ui/data-table";
import { ConfirmDialog, Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, Select } from "@/components/ui/form";
import { Badge, Card, CardBody, CardHeader, Dot, EmptyState, Menu, MenuContent, MenuItem, MenuLabel, MenuSeparator, MenuTrigger, Mono, Skeleton, SkeletonRows, Stat, StatGroup, StatusBadge, Tip } from "@/components/ui/primitives";
import { patch, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatBytes, formatDate, formatNumber } from "@/lib/utils";

type AdminOrg = { id: string; name: string; slug: string; status: string; created_at: string; members: number; knowledge_bases: number; documents: number; storage_bytes: number; plan: string | null };
type AdminUser = { id: string; email: string; full_name: string; status: string; is_platform_admin: boolean; mfa_enabled: boolean; email_verified: boolean; organizations: number; last_login_at: string | null; created_at: string };
type Job = { id: string; org_id: string; kind: string; status: string; attempts: number; error: string; created_at: string; finished_at: string | null };
type AdminAudit = { id: number; org_id: string | null; action: string; outcome: string; actor_email: string | null; resource_type: string | null; resource_id: string | null; ip: string | null; created_at: string };
type Check = { ok: boolean; latency_ms?: number; migration?: string; backend?: string; error?: string; counts?: Record<string, number> };
type System = {
  ready: boolean;
  checks: Record<string, Check>;
  jobs: { by_status: Record<string, number>; dead_by_kind: Record<string, number>; oldest_queued: string | null };
  ai_providers: Record<string, number>;
  totals: { organizations: number; users: number; documents: number; chunks: number; storage_bytes: number };
  storage_backend: string;
  storage_bytes_on_disk: number | null;
  usage_30d: { tokens: number; cost_usd: number; events: number };
  settings: { environment: string; job_backend: string; billing_enabled: boolean; event_backend: string };
};

const LIMIT = 50;

/** Org names for id-only rows (jobs, audit). */
function useOrgNames() {
  const q = useApi<Page<AdminOrg>>("/admin/organizations", { limit: 200 });
  const map = new Map((q.data?.items ?? []).map((o) => [o.id, o.name]));
  return (id: string | null) => (id ? map.get(id) ?? `${id.slice(0, 8)}…` : "Platform");
}

/* ------------------------------------------------------------ organizations */
export function OrganizationsPanel() {
  const qc = useQueryClient();
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);
  const dq = useDebounced(q.trim(), 300);
  const list = useApi<Page<AdminOrg>>("/admin/organizations", { q: dq || undefined, limit: LIMIT, offset });
  const plans = useApi<Plan[]>("/admin/plans");
  const [statusChange, setStatusChange] = useState<{ org: AdminOrg; status: "active" | "suspended" | "deactivated" } | null>(null);
  const [planFor, setPlanFor] = useState<AdminOrg | null>(null);

  const update = useMutation({
    mutationFn: ({ org, body }: { org: AdminOrg; body: Record<string, unknown> }) => patch(`/admin/organizations/${org.id}`, body),
    onSuccess: (_r, { org, body }) => {
      toast.success(body.status ? `${org.name} is now ${body.status}` : `${org.name} moved to the ${String(body.plan)} plan`);
      setStatusChange(null);
      setPlanFor(null);
      qc.invalidateQueries({ queryKey: ["/admin/organizations"] });
    },
  });

  const rows = list.data?.items ?? [];
  return (
    <Card className="overflow-hidden">
      <div className="flex flex-wrap items-center gap-2 border-b px-3.5 py-3">
        <SearchBox
          value={q}
          onChange={(v) => {
            setQ(v);
            setOffset(0);
          }}
          placeholder="Search name or slug…"
          label="Search organizations"
        />
        <span className="num ml-auto font-mono text-[11.5px] text-subtle">{list.data ? `${formatNumber(list.data.total)} organizations` : null}</span>
      </div>
      {list.isLoading ? (
        <SkeletonRows rows={8} />
      ) : !rows.length ? (
        <EmptyState icon={<Building2 className="h-5 w-5" />} title={dq ? "No organizations match" : "No organizations yet"} />
      ) : (
        <div className="relative overflow-x-auto">
          <table className="w-full min-w-[860px] text-[13px]">
            <thead>
              <tr className="border-b bg-surface-2/50 text-left font-mono text-[10.5px] uppercase tracking-[0.1em] text-subtle [&>th]:h-9 [&>th]:font-medium">
                <th className="px-4">Organization</th>
                <th className="px-4">Status</th>
                <th className="px-4">Plan</th>
                <th className="px-4 text-right">Members</th>
                <th className="px-4 text-right">KBs</th>
                <th className="px-4 text-right">Documents</th>
                <th className="px-4 text-right">Storage</th>
                <th className="px-4">Created</th>
                <th className="w-10 px-3 py-2">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((o) => (
                <tr key={o.id} className="border-b last:border-0 hover:bg-surface-2/50">
                  <td className="px-4 py-3">
                    <div className="font-medium">{o.name}</div>
                    <Mono className="text-subtle">{o.slug}</Mono>
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={o.status} />
                  </td>
                  <td className="px-4 py-3">{o.plan ? <Badge tone="accent">{o.plan}</Badge> : <span className="text-subtle">—</span>}</td>
                  <td className="num px-4 py-3 text-right font-mono text-[12.5px]">{formatNumber(o.members)}</td>
                  <td className="num px-4 py-3 text-right font-mono text-[12.5px]">{formatNumber(o.knowledge_bases)}</td>
                  <td className="num px-4 py-3 text-right font-mono text-[12.5px]">{formatNumber(o.documents)}</td>
                  <td className="num px-4 py-3 text-right font-mono text-[12.5px]">{formatBytes(o.storage_bytes)}</td>
                  <td className="px-4 py-3 text-muted">
                    <When iso={o.created_at} />
                  </td>
                  <td className="px-4 py-3 text-right">
                    <Menu>
                      <MenuTrigger asChild>
                        <RowMenuButton label={`Actions for ${o.name}`} />
                      </MenuTrigger>
                      <MenuContent className="w-52">
                        <MenuItem onSelect={() => setPlanFor(o)}>
                          <CreditCard className="h-3.5 w-3.5" /> Change plan
                        </MenuItem>
                        <MenuSeparator />
                        <MenuLabel>Status</MenuLabel>
                        {o.status !== "active" ? (
                          <MenuItem onSelect={() => update.mutate({ org: o, body: { status: "active" } })}>
                            <CheckCircle2 className="h-3.5 w-3.5" /> Activate
                          </MenuItem>
                        ) : null}
                        {o.status !== "suspended" ? (
                          <MenuItem onSelect={() => setStatusChange({ org: o, status: "suspended" })}>
                            <Ban className="h-3.5 w-3.5" /> Suspend
                          </MenuItem>
                        ) : null}
                        {o.status !== "deactivated" ? (
                          <MenuItem danger onSelect={() => setStatusChange({ org: o, status: "deactivated" })}>
                            <CircleSlash className="h-3.5 w-3.5" /> Deactivate
                          </MenuItem>
                        ) : null}
                      </MenuContent>
                    </Menu>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {list.data ? <Pager total={list.data.total} limit={LIMIT} offset={offset} onChange={setOffset} /> : null}
      <ConfirmDialog
        open={Boolean(statusChange)}
        onOpenChange={(v) => !v && setStatusChange(null)}
        title={statusChange?.status === "suspended" ? `Suspend ${statusChange.org.name}?` : `Deactivate ${statusChange?.org.name ?? ""}?`}
        description={statusChange?.status === "suspended" ? "Every member is blocked from the organization until it is reactivated. Data is kept." : "The organization stops working for all members. Data is kept until purged; you can reactivate it later."}
        confirmLabel={statusChange?.status === "suspended" ? "Suspend organization" : "Deactivate organization"}
        requireText={statusChange?.org.slug}
        loading={update.isPending}
        onConfirm={() => {
          if (statusChange) update.mutate({ org: statusChange.org, body: { status: statusChange.status } });
        }}
      />
      <PlanDialog org={planFor} plans={plans.data ?? []} saving={update.isPending} onClose={() => setPlanFor(null)} onSave={(plan) => planFor && update.mutate({ org: planFor, body: { plan } })} />
    </Card>
  );
}

function PlanDialog({ org, plans, saving, onClose, onSave }: { org: AdminOrg | null; plans: Plan[]; saving: boolean; onClose: () => void; onSave: (plan: string) => void }) {
  const [plan, setPlan] = useState("");
  useEffect(() => {
    if (org) setPlan(org.plan ?? "");
  }, [org]);
  const selected = plans.find((p) => p.key === plan);
  return (
    <Dialog open={Boolean(org)} onOpenChange={(v) => !v && onClose()}>
      <DialogContent
        title={`Change plan · ${org?.name ?? ""}`}
        description="Takes effect immediately. Limits are enforced on new uploads, invites, sources and model usage."
        footer={
          <>
            <CancelButton />
            <Button loading={saving} disabled={!plan || plan === org?.plan} onClick={() => onSave(plan)}>
              Apply plan
            </Button>
          </>
        }
      >
        <div className="space-y-4">
          <Field label="Plan" htmlFor="admin-plan">
            <Select id="admin-plan" value={plan} onChange={(e) => setPlan(e.target.value)}>
              <option value="">Choose a plan…</option>
              {plans.map((p) => (
                <option key={p.key} value={p.key}>
                  {p.name} ({p.key}){p.price_monthly_usd ? ` · ${usd(p.price_monthly_usd, 0)}/mo` : ""}
                </option>
              ))}
            </Select>
          </Field>
          {selected ? (
            <div className="rounded-lg border bg-surface-2/50 px-3 py-2">
              {Object.keys(selected.limits).length ? (
                Object.entries(selected.limits).map(([k, v]) => (
                  <KV key={k} label={k.replace(/_/g, " ")}>
                    <span className="num">{formatNumber(v)}</span>
                  </KV>
                ))
              ) : (
                <p className="py-1 text-[13px] text-muted">No limits.</p>
              )}
            </div>
          ) : null}
        </div>
      </DialogContent>
    </Dialog>
  );
}

/* -------------------------------------------------------------------- users */
export function UsersPanel() {
  const { user: me } = useAuth();
  const qc = useQueryClient();
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);
  const dq = useDebounced(q.trim(), 300);
  const list = useApi<Page<AdminUser>>("/admin/users", { q: dq || undefined, limit: LIMIT, offset });
  const [pending, setPending] = useState<{ user: AdminUser; body: { status?: string; is_platform_admin?: boolean }; title: string; description: string; confirm: string } | null>(null);

  const update = useMutation({
    mutationFn: ({ user, body }: { user: AdminUser; body: Record<string, unknown> }) => patch(`/admin/users/${user.id}`, body),
    onSuccess: () => {
      toast.success("User updated");
      setPending(null);
      qc.invalidateQueries({ queryKey: ["/admin/users"] });
    },
  });

  const rows = list.data?.items ?? [];
  return (
    <Card className="overflow-hidden">
      <div className="flex flex-wrap items-center gap-2 border-b px-3.5 py-3">
        <SearchBox
          value={q}
          onChange={(v) => {
            setQ(v);
            setOffset(0);
          }}
          placeholder="Search email or name…"
          label="Search users"
        />
        <span className="num ml-auto font-mono text-[11.5px] text-subtle">{list.data ? `${formatNumber(list.data.total)} users` : null}</span>
      </div>
      {list.isLoading ? (
        <SkeletonRows rows={8} />
      ) : !rows.length ? (
        <EmptyState icon={<Users className="h-5 w-5" />} title={dq ? "No users match" : "No users yet"} />
      ) : (
        <div className="relative overflow-x-auto">
          <table className="w-full min-w-[820px] text-[13px]">
            <thead>
              <tr className="border-b bg-surface-2/50 text-left font-mono text-[10.5px] uppercase tracking-[0.1em] text-subtle [&>th]:h-9 [&>th]:font-medium">
                <th className="px-4">User</th>
                <th className="px-4">Status</th>
                <th className="px-4">Security</th>
                <th className="px-4 text-right">Orgs</th>
                <th className="px-4">Last login</th>
                <th className="px-4">Joined</th>
                <th className="w-10 px-3 py-2">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((u) => {
                const self = u.id === me?.id;
                return (
                  <tr key={u.id} className="border-b last:border-0 hover:bg-surface-2/50">
                    <td className="px-4 py-3">
                      <PersonCell name={u.full_name} email={u.email} you={self} />
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex flex-wrap items-center gap-1">
                        <StatusBadge status={u.status} />
                        {u.is_platform_admin ? (
                          <Badge tone="accent">
                            <Shield className="h-3 w-3" /> platform admin
                          </Badge>
                        ) : null}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex flex-wrap gap-1">
                        <Badge tone={u.mfa_enabled ? "ok" : "neutral"}>MFA {u.mfa_enabled ? "on" : "off"}</Badge>
                        <Badge tone={u.email_verified ? "ok" : "warn"}>{u.email_verified ? "verified" : "unverified"}</Badge>
                      </div>
                    </td>
                    <td className="num px-4 py-3 text-right font-mono text-[12.5px]">{u.organizations}</td>
                    <td className="px-4 py-3 text-muted">
                      <When iso={u.last_login_at} />
                    </td>
                    <td className="px-4 py-3 text-muted">
                      <When iso={u.created_at} />
                    </td>
                    <td className="px-4 py-3 text-right">
                      {self ? (
                        <Tip content="You cannot change your own platform account here.">
                          <span tabIndex={0} className="inline-flex">
                            <RowMenuButton label="No actions for your own account" disabled />
                          </span>
                        </Tip>
                      ) : (
                        <Menu>
                          <MenuTrigger asChild>
                            <RowMenuButton label={`Actions for ${u.email}`} />
                          </MenuTrigger>
                          <MenuContent className="w-56">
                            {u.status !== "active" ? (
                              <MenuItem onSelect={() => update.mutate({ user: u, body: { status: "active" } })}>
                                <CheckCircle2 className="h-3.5 w-3.5" /> Reactivate account
                              </MenuItem>
                            ) : null}
                            {u.status !== "suspended" ? (
                              <MenuItem onSelect={() => setPending({ user: u, body: { status: "suspended" }, title: `Suspend ${u.email}?`, description: "All of their sessions are revoked and they cannot sign in until reactivated.", confirm: "Suspend user" })}>
                                <Ban className="h-3.5 w-3.5" /> Suspend account
                              </MenuItem>
                            ) : null}
                            {u.status !== "deactivated" ? (
                              <MenuItem danger onSelect={() => setPending({ user: u, body: { status: "deactivated" }, title: `Deactivate ${u.email}?`, description: "Sessions are revoked and sign-in is blocked. Memberships are kept.", confirm: "Deactivate user" })}>
                                <CircleSlash className="h-3.5 w-3.5" /> Deactivate account
                              </MenuItem>
                            ) : null}
                            <MenuSeparator />
                            {u.is_platform_admin ? (
                              <MenuItem danger onSelect={() => setPending({ user: u, body: { is_platform_admin: false }, title: `Revoke platform admin from ${u.email}?`, description: "They lose access to this console immediately.", confirm: "Revoke platform admin" })}>
                                <ShieldOff className="h-3.5 w-3.5" /> Revoke platform admin
                              </MenuItem>
                            ) : (
                              <MenuItem onSelect={() => setPending({ user: u, body: { is_platform_admin: true }, title: `Make ${u.email} a platform administrator?`, description: "They will manage every tenant, user and plan on this deployment. They still cannot read customer content without an owner-granted support window.", confirm: "Grant platform admin" })}>
                                <Shield className="h-3.5 w-3.5" /> Make platform admin
                              </MenuItem>
                            )}
                          </MenuContent>
                        </Menu>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {list.data ? <Pager total={list.data.total} limit={LIMIT} offset={offset} onChange={setOffset} /> : null}
      <ConfirmDialog
        open={Boolean(pending)}
        onOpenChange={(v) => !v && setPending(null)}
        title={pending?.title ?? ""}
        description={pending?.description}
        confirmLabel={pending?.confirm}
        danger={pending?.body.is_platform_admin !== true}
        requireText={pending?.body.is_platform_admin === true ? pending.user.email : undefined}
        loading={update.isPending}
        onConfirm={() => {
          if (pending) update.mutate({ user: pending.user, body: pending.body });
        }}
      />
    </Card>
  );
}

/* ------------------------------------------------------------------- system */
export function SystemPanel() {
  const q = useApi<System>("/admin/system", undefined, { refetchInterval: 30_000 });
  const s = q.data;
  if (!s)
    return (
      <div className="space-y-6">
        <Skeleton className="h-16 rounded-xl" />
        <Skeleton className="h-[104px] rounded-xl" />
      </div>
    );
  const dead = Object.entries(s.jobs.dead_by_kind).sort((a, b) => b[1] - a[1]);
  return (
    <div className="space-y-6">
      <div className={cn("flex flex-wrap items-center gap-3.5 rounded-xl border px-4 py-3.5", s.ready ? "border-ok/25 bg-ok/[0.045]" : "border-danger/25 bg-danger/[0.045]")} role="status">
        <span className={cn("grid h-9 w-9 shrink-0 place-items-center rounded-lg", s.ready ? "bg-ok/10 text-ok" : "bg-danger/10 text-danger")}>{s.ready ? <CheckCircle2 className="h-[18px] w-[18px]" /> : <XCircle className="h-[18px] w-[18px]" />}</span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-[14px] font-semibold">
            <Dot tone={s.ready ? "ok" : "danger"} live />
            {s.ready ? "All readiness checks passing" : "Degraded — one or more readiness checks failing"}
          </div>
          <div className="mt-0.5 font-mono text-[11.5px] text-muted">
            {s.settings.environment} · jobs via {s.settings.job_backend} · events via {s.settings.event_backend} · billing {s.settings.billing_enabled ? "enabled" : "disabled"}
          </div>
        </div>
        <Button size="sm" variant="secondary" onClick={() => q.refetch()} loading={q.isFetching}>
          {!q.isFetching ? <RefreshCw className="h-3.5 w-3.5" /> : null} Refresh
        </Button>
      </div>

      <StatGroup cols={4}>
        <Stat label="Organizations" value={formatNumber(s.totals.organizations)} icon={<Building2 />} />
        <Stat label="Users" value={formatNumber(s.totals.users)} icon={<Users />} />
        <Stat label="Documents" value={formatNumber(s.totals.documents)} hint={`${formatNumber(s.totals.chunks)} chunks`} icon={<Database />} />
        <Stat label="Storage" value={formatBytes(s.totals.storage_bytes)} hint={`${s.storage_backend}${s.storage_bytes_on_disk !== null ? ` · ${formatBytes(s.storage_bytes_on_disk)} on disk` : ""}`} icon={<HardDrive />} />
      </StatGroup>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader title="Readiness checks" />
          <ul className="divide-y">
            {Object.entries(s.checks).map(([name, c]) => (
              <li key={name} className="flex items-start gap-3 px-5 py-2.5 text-[13px]">
                <Dot tone={c.ok ? "ok" : "danger"} className="mt-1.5" />
                <div className="min-w-0 flex-1">
                  <div className="font-medium capitalize">{name}</div>
                  <div className="text-xs text-subtle">
                    {[c.backend, c.migration ? `migration ${c.migration}` : null, c.error].filter(Boolean).join(" · ") || (c.ok ? "ok" : "failing")}
                  </div>
                </div>
                {c.latency_ms !== undefined ? <span className="num font-mono text-[11.5px] text-muted">{formatDuration(c.latency_ms)}</span> : null}
                <span className="sr-only">{c.ok ? "passing" : "failing"}</span>
              </li>
            ))}
          </ul>
        </Card>
        <Card>
          <CardHeader title="Background jobs" description={s.jobs.oldest_queued ? <>Oldest queued job waiting since <When iso={s.jobs.oldest_queued} /></> : "Queue is empty"} />
          <CardBody className="space-y-3">
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(s.jobs.by_status).length ? (
                Object.entries(s.jobs.by_status)
                  .sort((a, b) => b[1] - a[1])
                  .map(([st, n]) => (
                    <span key={st} className="inline-flex items-center gap-1.5">
                      <StatusBadge status={st} />
                      <span className="num text-[13px] font-medium">{formatNumber(n)}</span>
                    </span>
                  ))
              ) : (
                <span className="text-[13px] text-muted">No jobs recorded.</span>
              )}
            </div>
            <div>
              <div className="label mb-1.5">Failed permanently, by kind</div>
              {dead.length ? (
                <ul className="divide-y rounded-lg border">
                  {dead.map(([kind, n]) => (
                    <li key={kind} className="flex items-center justify-between px-3 py-1.5 text-[12.5px]">
                      <Mono>{kind}</Mono>
                      <span className="num font-medium text-danger">{formatNumber(n)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-[13px] text-ok">No dead jobs.</p>
              )}
            </div>
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Usage & AI providers" description="Across every tenant, last 30 days" />
          <CardBody className="divide-y py-2">
            <KV label="Model tokens">
              <span className="num">{formatNumber(s.usage_30d.tokens)}</span>
            </KV>
            <KV label="Estimated cost">
              <span className="num">{usd(s.usage_30d.cost_usd)}</span>
            </KV>
            <KV label="Usage events">
              <span className="num">{formatNumber(s.usage_30d.events)}</span>
            </KV>
            <div className="py-2">
              <div className="label mb-1.5">Configured AI providers by status</div>
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(s.ai_providers).length ? (
                  Object.entries(s.ai_providers).map(([st, n]) => (
                    <span key={st} className="inline-flex items-center gap-1.5">
                      <StatusBadge status={st} />
                      <span className="num text-[13px] font-medium">{n}</span>
                    </span>
                  ))
                ) : (
                  <span className="text-[13px] text-muted">None configured.</span>
                )}
              </div>
            </div>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------------- jobs */
export function JobsPanel() {
  const qc = useQueryClient();
  const [status, setStatus] = useState("dead");
  const jobs = useApi<Job[]>("/admin/jobs", { status, limit: 200 });
  const orgName = useOrgNames();
  const retry = useMutation({
    mutationFn: (j: Job) => post(`/admin/jobs/${j.id}/retry`),
    onSuccess: (_r, j) => {
      toast.success(`Job ${j.kind} re-queued`, { description: "Attempts were reset; it runs on the next worker tick." });
      qc.invalidateQueries({ queryKey: ["/admin/jobs"] });
      qc.invalidateQueries({ queryKey: ["/admin/system"] });
    },
  });
  const columns: Column<Job>[] = [
    { key: "kind", header: "Kind", sortValue: (j) => j.kind, cell: (j) => <Mono className="text-fg">{j.kind}</Mono> },
    { key: "org", header: "Organization", sortValue: (j) => orgName(j.org_id), cell: (j) => <span className="text-muted">{orgName(j.org_id)}</span> },
    { key: "status", header: "Status", cell: (j) => <StatusBadge status={j.status} /> },
    { key: "attempts", header: "Attempts", sortValue: (j) => j.attempts, cell: (j) => <span className="num">{j.attempts}</span> },
    { key: "error", header: "Error", className: "max-w-[340px]", cell: (j) => (j.error ? <span className="line-clamp-2 break-words font-mono text-[11.5px] text-danger">{j.error}</span> : <span className="text-subtle">—</span>) },
    { key: "created", header: "Created", sortValue: (j) => j.created_at, cell: (j) => <span title={formatDate(j.created_at)}><When iso={j.created_at} className="text-muted" /></span> },
    { key: "finished", header: "Finished", sortValue: (j) => j.finished_at || "", cell: (j) => <When iso={j.finished_at} className="text-muted" /> },
    {
      key: "actions",
      header: <span className="relative block"><span className="sr-only">Actions</span></span>,
      className: "text-right",
      cell: (j) =>
        j.status === "dead" || j.status === "failed" || j.status === "cancelled" ? (
          <Button size="xs" variant="outline" loading={retry.isPending && retry.variables?.id === j.id} onClick={() => retry.mutate(j)}>
            {!(retry.isPending && retry.variables?.id === j.id) ? <RotateCcw className="h-3.5 w-3.5" /> : null} Retry
          </Button>
        ) : null,
    },
  ];
  return (
    <DataTable
      rows={jobs.data}
      columns={columns}
      rowKey={(j) => j.id}
      loading={jobs.isLoading}
      filterText={(j) => `${j.kind} ${orgName(j.org_id)} ${j.error}`}
      searchPlaceholder="Filter by kind, org or error…"
      toolbar={
        <Select aria-label="Job status" value={status} onChange={(e) => setStatus(e.target.value)} className="h-8 w-44 text-[13px]">
          <option value="dead">Failed permanently</option>
          <option value="failed">Failed (will retry)</option>
          <option value="queued">Queued</option>
          <option value="running">Running</option>
          <option value="succeeded">Succeeded</option>
          <option value="cancelled">Cancelled</option>
        </Select>
      }
      empty={<EmptyState title={status === "dead" ? "No dead jobs" : `No ${status} jobs`} description={status === "dead" ? "Every background job has either succeeded or is still retrying." : undefined} />}
    />
  );
}

/* -------------------------------------------------------------------- audit */
export function PlatformAuditPanel() {
  const q = useApi<AdminAudit[]>("/admin/audit", { limit: 500 });
  const orgName = useOrgNames();
  const columns: Column<AdminAudit>[] = [
    {
      key: "time",
      header: "Time",
      sortValue: (e) => e.id,
      cell: (e) => (
        <div className="whitespace-nowrap">
          <div>{formatDate(e.created_at)}</div>
          <div className="text-2xs text-subtle">
            <When iso={e.created_at} />
          </div>
        </div>
      ),
    },
    { key: "actor", header: "Actor", sortValue: (e) => e.actor_email || "", cell: (e) => <span className="truncate">{e.actor_email || <span className="text-subtle">System</span>}</span> },
    { key: "action", header: "Action", sortValue: (e) => e.action, cell: (e) => <Mono className="text-fg">{e.action}</Mono> },
    { key: "outcome", header: "Outcome", cell: (e) => <StatusBadge status={e.outcome} /> },
    { key: "org", header: "Organization", sortValue: (e) => orgName(e.org_id), cell: (e) => <span className="text-muted">{orgName(e.org_id)}</span> },
    {
      key: "resource",
      header: "Resource",
      cell: (e) =>
        e.resource_type ? (
          <div className="max-w-[220px]">
            <div className="text-muted">{e.resource_type.replace(/_/g, " ")}</div>
            {e.resource_id ? <Mono className="block truncate text-[11px] text-subtle">{e.resource_id}</Mono> : null}
          </div>
        ) : (
          <span className="text-subtle">—</span>
        ),
    },
    { key: "ip", header: "IP", cell: (e) => <Mono className="text-muted">{e.ip || "—"}</Mono> },
  ];
  return (
    <DataTable
      rows={q.data}
      columns={columns}
      rowKey={(e) => String(e.id)}
      loading={q.isLoading}
      filterText={(e) => `${e.action} ${e.actor_email ?? ""} ${orgName(e.org_id)} ${e.resource_type ?? ""} ${e.resource_id ?? ""}`}
      searchPlaceholder="Filter platform events…"
      pageSize={50}
      dense
      empty={<EmptyState title="No platform events" description="Platform-level actions (tenant and user changes, job retries) and support-access events appear here." />}
    />
  );
}
