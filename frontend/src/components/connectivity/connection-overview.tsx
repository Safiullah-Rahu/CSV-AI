"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Activity, ChevronDown, Gauge, KeyRound, Lock, MoreHorizontal, Pause, Pencil, Play, PlugZap, RefreshCw, ScanSearch, Table2, Unplug } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";
import { Area, AreaChart, ResponsiveContainer, Tooltip, YAxis } from "recharts";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Badge, Card, CardBody, CardHeader, Dot, EmptyState, Menu, MenuContent, MenuItem, MenuLabel, MenuSeparator, MenuTrigger, Skeleton, Stat, StatGroup, StatusBadge, Tip } from "@/components/ui/primitives";
import { post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatDate, formatNumber, timeAgo } from "@/lib/utils";

import { DisconnectDialog, EditConnectionDialog, RotatePasswordDialog } from "./connection-dialogs";
import { AccessModeBadge, GatedButton, Notice, ProviderStatusBadge, When, formatMs, intervalLabel, invalidatePrefix, useJobWatcher, useKnowledgeBases } from "./shared";
import type { DbConnection, DiscoverResult, HealthPoint, TestResult } from "./types";

/* --------------------------------------------------------------- actions */

export function useConnectionPerms(c: DbConnection | undefined) {
  const { can } = useAuth();
  const manage = c?.access === "manage";
  return {
    test: can("db.test"),
    discover: can("db.schemas") && manage,
    sync: can("db.sync"),
    edit: can("db.edit") && manage,
    rotate: can("db.edit") && can("db.credentials") && manage,
    remove: can("db.delete") && manage,
    query: can("db.query"),
    grants: can("db.credentials") || can("admin.permissions"),
    manage,
  };
}

function needs(perm: string, manage: boolean, hasPerm: boolean) {
  if (!hasPerm) return `Requires ${perm}`;
  if (!manage) return "Requires manage access to this connection";
  return "";
}

export function ConnectionActions({ connection: c }: { connection: DbConnection }) {
  const { can } = useAuth();
  const qc = useQueryClient();
  const watch = useJobWatcher();
  const p = useConnectionPerms(c);
  const [edit, setEdit] = useState(false);
  const [rotate, setRotate] = useState(false);
  const [disconnect, setDisconnect] = useState(false);

  const test = useMutation({
    mutationFn: () => post<TestResult>(`/database-connections/${c.id}/test`),
    onSuccess: (r) => {
      invalidatePrefix(qc, "/database-connections", "/connections");
      if (r.ok) toast.success(`${c.name} is reachable`, { description: `Responded in ${r.latency_ms} ms` });
      else toast.error(`${c.name}: connection failed`, { description: r.error });
    },
  });
  const discover = useMutation({
    mutationFn: () => post<DiscoverResult>(`/database-connections/${c.id}/discover`),
    onSuccess: (r) => {
      invalidatePrefix(qc, "/database-connections");
      const delta = [r.added.length && `+${r.added.length} new`, r.removed.length && `−${r.removed.length} removed`, r.changed.length && `~${r.changed.length} changed`].filter(Boolean).join(" · ");
      toast.success(`Discovered ${formatNumber(r.tables)} tables`, { description: delta || "No changes since the last discovery." });
    },
  });
  const sync = useMutation({
    mutationFn: (mode: "incremental" | "full") => post<{ job_id: string }>(`/database-connections/${c.id}/sync`, { mode }),
    onSuccess: (r, mode) => {
      invalidatePrefix(qc, "/database-connections");
      watch(r.job_id, { label: `${mode === "full" ? "Full sync" : "Sync"} · ${c.name}`, invalidate: ["/database-connections", "/connections", "/documents"] });
    },
  });
  const pause = useMutation({
    mutationFn: (action: "pause" | "resume") => post(`/database-connections/${c.id}/${action}`),
    onSuccess: (_r, action) => {
      toast.success(action === "pause" ? "Scheduled sync paused" : "Scheduled sync resumed");
      invalidatePrefix(qc, "/database-connections");
    },
  });

  const syncBlocked = !p.sync
    ? "Requires db.sync"
    : c.access_mode === "live"
      ? "Live-query mode — switch to ingest or hybrid to synchronise"
      : !c.enabled
        ? "Connection is disabled"
        : !c.selected_tables.length
          ? "Select at least one table on the Schema tab first"
          : !c.kb_id
            ? "Choose a target knowledge base (Edit) first"
            : "";

  return (
    <>
      <GatedButton variant="secondary" size="sm" allowed={p.test} perm="db.test" onClick={() => test.mutate()} loading={test.isPending}>
        <PlugZap className="h-3.5 w-3.5" /> Test
      </GatedButton>
      <GatedButton variant="secondary" size="sm" allowed={p.discover} reason={needs("db.schemas", p.manage, can("db.schemas"))} onClick={() => discover.mutate()} loading={discover.isPending}>
        <ScanSearch className="h-3.5 w-3.5" /> Discover schema
      </GatedButton>
      {syncBlocked ? (
        <GatedButton size="sm" allowed={false} reason={syncBlocked}>
          <RefreshCw className="h-3.5 w-3.5" /> Sync now
        </GatedButton>
      ) : (
        <Menu>
          <MenuTrigger asChild>
            <Button size="sm" loading={sync.isPending}>
              <RefreshCw className="h-3.5 w-3.5" /> Sync now <ChevronDown className="h-3.5 w-3.5 opacity-70" />
            </Button>
          </MenuTrigger>
          <MenuContent>
            <MenuItem onSelect={() => sync.mutate("incremental")}>
              <div>
                <div className="font-medium">Incremental</div>
                <div className="text-2xs text-subtle">New and changed rows (needs an incremental column)</div>
              </div>
            </MenuItem>
            <MenuItem onSelect={() => sync.mutate("full")}>
              <div>
                <div className="font-medium">Full</div>
                <div className="text-2xs text-subtle">Re-extract every selected table</div>
              </div>
            </MenuItem>
          </MenuContent>
        </Menu>
      )}
      <Menu>
        <MenuTrigger asChild>
          <Button variant="secondary" size="icon" aria-label="More actions">
            <MoreHorizontal className="h-4 w-4" />
          </Button>
        </MenuTrigger>
        <MenuContent className="min-w-[230px]">
          <MenuLabel>Schedule</MenuLabel>
          <MenuItem disabled={!p.sync || c.access_mode === "live"} onSelect={() => pause.mutate(c.sync_paused ? "resume" : "pause")}>
            {c.sync_paused ? <Play className="h-3.5 w-3.5" /> : <Pause className="h-3.5 w-3.5" />}
            <span className="flex-1">{c.sync_paused ? "Resume scheduled sync" : "Pause scheduled sync"}</span>
            {!p.sync ? <span className="text-2xs text-subtle">db.sync</span> : c.access_mode === "live" ? <span className="text-2xs text-subtle">live mode</span> : null}
          </MenuItem>
          <MenuSeparator />
          <MenuLabel>Configuration</MenuLabel>
          <MenuItem disabled={!p.edit} onSelect={() => setEdit(true)}>
            <Pencil className="h-3.5 w-3.5" /> <span className="flex-1">Edit connection</span>
            {!p.edit ? <span className="text-2xs text-subtle">{can("db.edit") ? "manage access" : "db.edit"}</span> : null}
          </MenuItem>
          <MenuItem disabled={!p.rotate} onSelect={() => setRotate(true)}>
            <KeyRound className="h-3.5 w-3.5" /> <span className="flex-1">Rotate password</span>
            {!p.rotate ? <span className="text-2xs text-subtle">{!can("db.credentials") ? "db.credentials" : !can("db.edit") ? "db.edit" : "manage access"}</span> : null}
          </MenuItem>
          <MenuSeparator />
          <MenuItem danger disabled={!p.remove} onSelect={() => setDisconnect(true)}>
            <Unplug className="h-3.5 w-3.5" /> <span className="flex-1">Disconnect…</span>
            {!p.remove ? <span className="text-2xs text-subtle">{can("db.delete") ? "manage access" : "db.delete"}</span> : null}
          </MenuItem>
        </MenuContent>
      </Menu>
      {p.edit ? <EditConnectionDialog connection={c} open={edit} onOpenChange={setEdit} /> : null}
      {p.rotate ? <RotatePasswordDialog connection={c} open={rotate} onOpenChange={setRotate} /> : null}
      {p.remove ? <DisconnectDialog connection={c} open={disconnect} onOpenChange={setDisconnect} /> : null}
    </>
  );
}

/* -------------------------------------------------------------- overview */

export function ConnectionOverview({ connection: c, onGoTo }: { connection: DbConnection; onGoTo: (tab: string) => void }) {
  const kbs = useKnowledgeBases();
  const kb = kbs.data?.find((k) => k.id === c.kb_id);
  const limits = c.query_limits || {};
  const live = c.access_mode === "live";
  return (
    <div className="space-y-6">
      {c.last_error || !c.enabled || c.table_count === 0 || !c.selected_tables.length ? (
        <div className="space-y-3">
          {c.last_error ? (
            <Notice tone="danger" title={c.status === "failed" ? "Last check failed" : "Last error"}>
              <span className="font-mono text-[12px]">{c.last_error}</span>
            </Notice>
          ) : null}
          {!c.enabled ? <Notice tone="warn" title="Connection disabled">It cannot be queried or synchronised. Enable it from Edit.</Notice> : null}
          {c.table_count === 0 ? (
            <Notice tone="info" title="Schema not discovered yet" action={<Button size="xs" variant="secondary" onClick={() => onGoTo("schema")}>Open schema</Button>}>
              Discover the schema, then select which tables and columns are available for querying and sync.
            </Notice>
          ) : !c.selected_tables.length ? (
            <Notice tone="info" title="No tables selected" action={<Button size="xs" variant="secondary" onClick={() => onGoTo("schema")}>Select tables</Button>}>
              {c.table_count} tables were discovered, but none are selected — nothing can be queried or synchronised yet.
            </Notice>
          ) : null}
        </div>
      ) : null}

      <StatGroup cols={4}>
        <Stat label="Tables" value={<button className="text-left transition-colors hover:text-accent" onClick={() => onGoTo("schema")}>{c.selected_tables.length}<span className="text-[18px] font-medium text-subtle"> / {c.table_count}</span></button>} hint={c.schema_refreshed_at ? `selected · refreshed ${timeAgo(c.schema_refreshed_at)}` : "Schema not discovered"} icon={<Table2 />} />
        <Stat label="Last tested" value={timeAgo(c.last_tested_at)} hint={c.last_success_at ? `Last success ${timeAgo(c.last_success_at)}` : "No successful check yet"} icon={<PlugZap />} tone={c.status === "failed" ? "danger" : c.status === "healthy" ? "ok" : undefined} />
        <Stat label={live ? "Sync" : "Last synced"} value={live ? "Off" : timeAgo(c.last_sync_at)} hint={live ? "Live mode copies nothing" : `${intervalLabel(c.sync_interval_minutes)}${c.sync_paused ? " · paused" : ""}`} icon={<RefreshCw />} />
        <Stat label="Limits" value={<>{limits.max_rows ?? 500}<span className="text-[15px] font-medium text-subtle"> rows</span></>} hint={`${limits.timeout_s ?? 15} s timeout · ${limits.max_joins ?? 4} joins`} icon={<Gauge />} />
      </StatGroup>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="min-w-0 space-y-6 lg:col-span-2">
          <Card>
            <CardHeader title="Connection details" description="Credentials are encrypted at rest and never returned by the API." action={<StatusBadge status={c.status} />} />
            <dl className="grid sm:grid-cols-2 sm:divide-x">
              <div className="divide-y px-5 py-1.5">
                <Spec label="Engine">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="font-mono text-[12.5px]">{c.provider_label}</span> <ProviderStatusBadge status={c.provider_status} />
                  </span>
                </Spec>
                <Spec label="Endpoint">
                  <span className="font-mono text-[12px]" title={c.endpoint}>
                    {c.endpoint}
                  </span>
                </Spec>
                <Spec label="SSL mode">
                  <span className="font-mono text-[12px]">{c.ssl_mode}</span>
                </Spec>
                <Spec label="Credentials">{c.has_credentials ? <Badge tone="ok">Password stored</Badge> : <Badge>No password</Badge>}</Spec>
                <Spec label="Access mode">
                  <AccessModeBadge mode={c.access_mode} />
                </Spec>
                <Spec label="Your access">
                  <span className="capitalize">{c.access}</span>
                </Spec>
              </div>
              <div className="divide-y border-t px-5 py-1.5 sm:border-t-0">
                <Spec label="Target knowledge base">
                  {c.kb_id ? (
                    kb ? (
                      <Link href={`/knowledge-bases/${kb.id}`} className="text-accent hover:underline">
                        {kb.name}
                      </Link>
                    ) : (
                      <span className="text-muted">Not visible to you</span>
                    )
                  ) : (
                    <span className="text-muted">None</span>
                  )}
                </Spec>
                <Spec label="Sync schedule">
                  {live ? (
                    <span className="text-muted">Not applicable (live)</span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5">
                      {intervalLabel(c.sync_interval_minutes)}
                      {c.sync_paused ? <Badge tone="warn">paused</Badge> : null}
                    </span>
                  )}
                </Spec>
                <Spec label="Next sync">{c.next_sync_at ? <Tip content={formatDate(c.next_sync_at)}><span className="font-mono text-[12px]">{formatDate(c.next_sync_at)}</span></Tip> : <span className="text-subtle">—</span>}</Spec>
                <Spec label="Last success">
                  <When at={c.last_success_at} />
                </Spec>
                <Spec label="Last synced">
                  <When at={c.last_sync_at} />
                </Spec>
                <Spec label="Schema refreshed">
                  <When at={c.schema_refreshed_at} />
                </Spec>
              </div>
            </dl>
          </Card>

          {c.selected_tables.length ? (
            <Card>
              <CardHeader
                title="Selected tables"
                description="Available to live queries (subject to grants) and to synchronisation."
                action={
                  <Button size="xs" variant="ghost" onClick={() => onGoTo("schema")}>
                    Manage
                  </Button>
                }
              />
              <CardBody className="flex flex-wrap gap-1.5">
                {c.selected_tables.map((t) => {
                  const sel = c.selection?.[t];
                  return (
                    <span key={t} className="inline-flex h-7 items-center gap-1.5 rounded-md border bg-surface-2/60 px-2 font-mono text-[11.5px] tracking-[-0.01em]">
                      <Table2 className="h-3 w-3 text-subtle" aria-hidden />
                      {t}
                      {sel?.requires_approval ? (
                        <Badge tone="warn" className="px-1.5 py-[1px] font-sans text-[10.5px]">
                          <Lock className="h-2.5 w-2.5" /> approval
                        </Badge>
                      ) : null}
                      {sel?.sensitive?.length ? (
                        <Badge tone="danger" className="px-1.5 py-[1px] font-sans text-[10.5px]">
                          {sel.sensitive.length} sensitive
                        </Badge>
                      ) : null}
                      {sel && !sel.ingest ? <Badge className="px-1.5 py-[1px] font-sans text-[10.5px]">no sync</Badge> : null}
                    </span>
                  );
                })}
              </CardBody>
            </Card>
          ) : null}
        </div>

        <HealthCard connectionId={c.id} />
      </div>
    </div>
  );
}

function Spec({ label, children }: { label: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="flex min-h-[42px] items-center justify-between gap-4 py-2">
      <dt className="shrink-0 text-[12.5px] text-muted">{label}</dt>
      <dd className="min-w-0 truncate text-right text-[13px]">{children}</dd>
    </div>
  );
}

/* ---------------------------------------------------------------- health */

function HealthCard({ connectionId }: { connectionId: string }) {
  const health = useApi<HealthPoint[]>(`/database-connections/${connectionId}/health`, undefined, { refetchInterval: 60_000 });
  const points = useMemo(() => [...(health.data ?? [])].reverse(), [health.data]);
  const stats = useMemo(() => {
    const ok = points.filter((p) => p.ok);
    const lat = ok.map((p) => p.latency_ms).filter((v): v is number => typeof v === "number");
    const lastFail = [...points].reverse().find((p) => !p.ok);
    return {
      uptime: points.length ? (ok.length / points.length) * 100 : null,
      avg: lat.length ? lat.reduce((a, b) => a + b, 0) / lat.length : null,
      p95: lat.length ? [...lat].sort((a, b) => a - b)[Math.min(lat.length - 1, Math.floor(lat.length * 0.95))] : null,
      lastFail,
    };
  }, [points]);
  const series = points.map((p) => ({ t: p.checked_at, latency: p.ok ? p.latency_ms : null }));
  const strip = points.slice(-60);
  const latest = points[points.length - 1];

  return (
    <Card className="h-fit overflow-hidden lg:sticky lg:top-6">
      <CardHeader
        title="Health"
        description={points.length ? `Last ${points.length} check${points.length === 1 ? "" : "s"}` : "Connection tests and scheduled health checks"}
        action={
          latest ? (
            <span className="inline-flex items-center gap-1.5 text-[12px] text-muted">
              <Dot tone={latest.ok ? "ok" : "danger"} live={latest.ok} />
              <When at={latest.checked_at} />
            </span>
          ) : null
        }
      />
      {health.isLoading ? (
        <CardBody>
          <Skeleton className="h-40" />
        </CardBody>
      ) : !points.length ? (
        <EmptyState className="py-10" icon={<Activity className="h-5 w-5" />} title="No checks yet" description="Run Test to record the first health check." />
      ) : (
        <>
          <div className="grid grid-cols-3 divide-x border-b">
            {[
              { label: "Uptime", value: stats.uptime === null ? "—" : `${stats.uptime.toFixed(stats.uptime === 100 ? 0 : 1)}%`, warn: stats.uptime !== null && stats.uptime < 95 },
              { label: "Avg", value: formatMs(stats.avg) },
              { label: "p95", value: formatMs(stats.p95) },
            ].map((s) => (
              <div key={s.label} className="min-w-0 px-4 py-3.5">
                <div className="label">{s.label}</div>
                <div className={cn("num mt-1.5 truncate text-[19px] font-semibold tracking-[-0.03em]", s.warn && "text-warn")}>{s.value}</div>
              </div>
            ))}
          </div>
          <CardBody className="space-y-4">
            <div className="-mx-1 h-24" aria-label="Latency of successful checks over time">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={series} margin={{ left: 0, right: 0, top: 4, bottom: 0 }}>
                  <defs>
                    <linearGradient id="lat" x1="0" x2="0" y1="0" y2="1">
                      <stop offset="0" stopColor="hsl(var(--accent))" stopOpacity={0.24} />
                      <stop offset="1" stopColor="hsl(var(--accent))" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <YAxis hide domain={[0, "auto"]} />
                  <Tooltip
                    contentStyle={{ background: "hsl(var(--surface))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12, color: "hsl(var(--fg))" }}
                    labelFormatter={(_l, payload) => (payload?.[0]?.payload?.t ? formatDate(payload[0].payload.t as string) : "")}
                    formatter={(v) => [v === null || v === undefined ? "failed" : `${Number(v).toFixed(0)} ms`, "Latency"]}
                  />
                  <Area type="monotone" dataKey="latency" stroke="hsl(var(--accent))" strokeWidth={1.75} fill="url(#lat)" connectNulls={false} isAnimationActive={false} dot={false} activeDot={{ r: 4, strokeWidth: 2, stroke: "hsl(var(--surface))" }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div>
              <div className="mb-2 flex items-center justify-between">
                <span className="label">Recent checks</span>
                <span className="flex items-center gap-2.5 text-[11px] text-subtle">
                  <span className="inline-flex items-center gap-1"><span className="h-2 w-2 rounded-[2px] bg-ok/70" /> ok</span>
                  <span className="inline-flex items-center gap-1"><span className="h-2 w-2 rounded-[2px] bg-danger/80" /> failed</span>
                </span>
              </div>
              <div className="flex h-7 items-stretch gap-[2px]" role="list" aria-label="Recent health checks">
                {strip.map((p, i) => (
                  <Tip key={i} content={<span>{formatDate(p.checked_at)} · {p.ok ? `ok, ${formatMs(p.latency_ms)}` : `failed: ${p.error ?? "unknown error"}`}</span>}>
                    <span role="listitem" aria-label={`${p.ok ? "ok" : "failed"} at ${formatDate(p.checked_at)}`} tabIndex={0} className={cn("min-w-[3px] max-w-[10px] flex-1 rounded-[2px] transition-colors", p.ok ? "bg-ok/60 hover:bg-ok" : "bg-danger/80 hover:bg-danger")} />
                  </Tip>
                ))}
              </div>
            </div>

            {stats.lastFail ? (
              <div className="rounded-lg border border-danger/20 bg-danger/[0.045] px-3 py-2.5 text-[12px]">
                <div className="font-medium text-danger">Last failure · {timeAgo(stats.lastFail.checked_at)}</div>
                <div className="mt-0.5 break-words font-mono text-[11.5px] text-muted">{stats.lastFail.error}</div>
              </div>
            ) : null}
          </CardBody>
        </>
      )}
    </Card>
  );
}
