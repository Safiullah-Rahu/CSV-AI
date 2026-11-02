"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, Boxes, ChevronDown, CircleAlert, ExternalLink, HeartPulse, Loader2, Pause, Play, PlugZap, Radio, RefreshCw, ShieldCheck, Trash2, X } from "lucide-react";
import Link from "next/link";
import { useId, useState } from "react";
import { toast } from "sonner";

import { EVENT_TONE, NODE_FAMILY, NODE_ICON, STATUS_LABEL, STATUS_TONE, toneVar, TYPE_LABEL } from "@/components/connections/meta";
import type { GraphNode, GraphSummary } from "@/components/connections/types";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/dialog";
import { Badge, Dot, Skeleton, StatusBadge, Tip, type Tone } from "@/components/ui/primitives";
import { del, get, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import type { ActivityEvent } from "@/lib/hooks";
import { cn, formatBytes, formatDate, formatNumber, timeAgo } from "@/lib/utils";

/** Frosted floating surface used for everything that sits on the map. */
export function GlassPanel({ className, children, solid }: { className?: string; children: React.ReactNode; solid?: boolean }) {
  return <div className={cn("cx-glass rounded-xl text-fg", solid && "cx-glass-solid", className)}>{children}</div>;
}

/** A glass card with a one-line header that collapses the body away. */
export function FloatingCard({
  title,
  icon,
  meta,
  collapsed,
  onToggle,
  className,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  meta?: React.ReactNode;
  collapsed: boolean;
  onToggle: () => void;
  className?: string;
  children: React.ReactNode;
}) {
  const id = useId();
  return (
    <section className={cn("cx-glass overflow-hidden rounded-xl text-fg", className)} aria-labelledby={`${id}-t`}>
      <div className="flex h-10 items-center gap-2 pl-3.5 pr-1.5">
        <span className="shrink-0 text-subtle [&_svg]:h-3.5 [&_svg]:w-3.5" aria-hidden>
          {icon}
        </span>
        <h2 id={`${id}-t`} className="label shrink-0 !text-muted">
          {title}
        </h2>
        <div className="ml-auto flex min-w-0 items-center gap-2">{meta}</div>
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={!collapsed}
          aria-controls={`${id}-b`}
          aria-label={collapsed ? `Expand ${title.toLowerCase()}` : `Collapse ${title.toLowerCase()}`}
          className="grid h-7 w-7 shrink-0 place-items-center rounded-md text-subtle transition-colors hover:bg-surface-2 hover:text-fg"
        >
          <ChevronDown className={cn("h-3.5 w-3.5 transition-transform duration-200", collapsed && "-rotate-90")} />
        </button>
      </div>
      {!collapsed ? (
        <div id={`${id}-b`} className="animate-fade-in border-t border-border/80">
          {children}
        </div>
      ) : null}
    </section>
  );
}

/* --------------------------------------------------------------- summary */

function Hero({ label, value, tone, alert }: { label: string; value: React.ReactNode; tone?: Tone; alert?: boolean }) {
  return (
    <div className="min-w-0 px-3 py-2.5">
      <div className="flex items-center gap-1.5">
        {tone ? <Dot tone={tone} /> : null}
        <span className="truncate font-mono text-[9.5px] font-medium uppercase tracking-[0.1em] text-subtle">{label}</span>
      </div>
      <div className={cn("num mt-1.5 text-[22px] font-semibold leading-none tracking-[-0.035em]", alert && "text-danger")}>{value}</div>
    </div>
  );
}

function Readout({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="min-w-0 px-3 py-[7px]">
      <div className="truncate text-[11px] leading-4 text-subtle">{label}</div>
      <div className="num truncate font-mono text-[12.5px] font-medium leading-4 text-fg">{value}</div>
    </div>
  );
}

export function ConnectionSummaryPanel({ s, collapsed, onToggle }: { s: GraphSummary | undefined; collapsed: boolean; onToggle: () => void }) {
  const parts = s
    ? [
        { k: "active", v: s.active, c: "bg-ok" },
        { k: "syncing", v: s.syncing, c: "bg-info" },
        { k: "failed", v: s.failed, c: "bg-danger" },
        { k: "disconnected", v: s.disconnected, c: "bg-border-strong" },
      ]
    : [];
  const total = parts.reduce((a, p) => a + p.v, 0);
  return (
    <FloatingCard
      title="Ecosystem"
      icon={<Activity />}
      collapsed={collapsed}
      onToggle={onToggle}
      className={cn(collapsed ? "w-auto" : "w-[352px]", "max-w-full")}
      meta={
        s ? (
          collapsed ? (
            <span className="truncate font-mono text-[11px] text-muted">
              <span className="num text-fg">{formatNumber(s.total_sources)}</span> sources
              {s.failed ? (
                <>
                  {" · "}
                  <span className="num text-danger">{formatNumber(s.failed)}</span> failed
                </>
              ) : null}
            </span>
          ) : (
            <span className="truncate font-mono text-[10.5px] text-subtle">synced {s.last_successful_sync ? timeAgo(s.last_successful_sync) : "never"}</span>
          )
        ) : null
      }
    >
      {s ? (
        <>
          <div className="grid grid-cols-4 divide-x divide-border/80">
            <Hero label="Sources" value={formatNumber(s.total_sources)} />
            <Hero label="Active" value={formatNumber(s.active)} tone="ok" />
            <Hero label="Syncing" value={formatNumber(s.syncing)} tone="info" />
            <Hero label="Failed" value={formatNumber(s.failed)} tone="danger" alert={s.failed > 0} />
          </div>
          <div className="px-3 pb-2.5">
            <div className="flex h-1 gap-px overflow-hidden rounded-full bg-surface-2" role="img" aria-label={parts.map((p) => `${p.v} ${p.k}`).join(", ")}>
              {total > 0 ? parts.map((p) => (p.v ? <span key={p.k} className={cn("h-full first:rounded-l-full last:rounded-r-full", p.c)} style={{ width: `${(p.v / total) * 100}%` }} /> : null)) : null}
            </div>
          </div>
          <div className="grid grid-cols-3 divide-x divide-border/70 border-t border-border/80">
            <Readout label="Documents" value={formatNumber(s.documents)} />
            <Readout label="Records" value={formatNumber(s.records)} />
            <Readout label="Chunks" value={formatNumber(s.chunks)} />
          </div>
          <div className="grid grid-cols-3 divide-x divide-border/70 border-t border-border/60">
            <Readout label="Storage" value={formatBytes(s.storage_bytes)} />
            <Readout label="Active jobs" value={formatNumber(s.active_jobs)} />
            <Readout label="Disconnected" value={formatNumber(s.disconnected)} />
          </div>
        </>
      ) : (
        <div className="space-y-2.5 p-3.5">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-16 w-full" />
        </div>
      )}
    </FloatingCard>
  );
}

/* ---------------------------------------------------------------- health */

type Health = { status: string; infrastructure: Record<string, { ok: boolean; latency_ms?: number; backend?: string }>; warnings: { severity: string; resource: string; message: string; kind: string }[] };

const HEALTH_TONE: Record<string, Tone> = { operational: "ok", degraded: "warn" };

export function ConnectionHealthPanel({ orgKey, collapsed, onToggle }: { orgKey: string | null; collapsed: boolean; onToggle: () => void }) {
  const { data } = useQuery({ queryKey: ["connections-health", orgKey], queryFn: () => get<Health>("/connections/health"), refetchInterval: 30_000, enabled: Boolean(orgKey) });
  const tone: Tone = data ? HEALTH_TONE[data.status] ?? "danger" : "neutral";
  return (
    <FloatingCard
      title="System health"
      icon={<HeartPulse />}
      collapsed={collapsed}
      onToggle={onToggle}
      className={cn(collapsed ? "w-auto" : "w-[288px]", "max-w-full")}
      meta={
        data ? (
          <Badge tone={tone} className="!py-[2px]">
            <Dot tone={tone} />
            {data.status}
          </Badge>
        ) : null
      }
    >
      {data ? (
        <>
          <ul className="py-1.5">
            {Object.entries(data.infrastructure).map(([k, v]) => (
              <li key={k} className="flex h-7 items-center justify-between gap-3 px-3.5">
                <span className="text-[12.5px] capitalize text-muted">{k}</span>
                <span className="flex items-center gap-2 font-mono text-[11.5px] text-fg">
                  {v.latency_ms !== undefined ? `${v.latency_ms} ms` : v.backend ?? ""}
                  <span className={cn("h-1.5 w-1.5 rounded-full", v.ok ? "bg-ok shadow-[0_0_0_3px_hsl(var(--ok)/0.15)]" : "bg-danger shadow-[0_0_0_3px_hsl(var(--danger)/0.18)]")} aria-label={v.ok ? "ok" : "down"} />
                </span>
              </li>
            ))}
          </ul>
          {data.warnings.length ? (
            <ul className="max-h-36 space-y-2 overflow-y-auto border-t border-border/80 px-3.5 py-2.5">
              {data.warnings.map((w, i) => (
                <li key={i} className="flex gap-2 text-[12px]">
                  <CircleAlert className={cn("mt-0.5 h-3.5 w-3.5 shrink-0", w.severity === "error" ? "text-danger" : "text-warn")} />
                  <div className="min-w-0">
                    <div className="truncate font-medium text-fg">{w.resource}</div>
                    <div className="line-clamp-2 text-muted">{w.message}</div>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="flex items-center gap-2 border-t border-border/80 px-3.5 py-2.5 text-[12px] text-muted">
              <ShieldCheck className="h-3.5 w-3.5 text-ok" /> All connections healthy
            </div>
          )}
        </>
      ) : (
        <div className="space-y-2 p-3.5">
          <Skeleton className="h-5 w-full" />
          <Skeleton className="h-5 w-full" />
          <Skeleton className="h-5 w-3/4" />
        </div>
      )}
    </FloatingCard>
  );
}

/* ---------------------------------------------------------- live activity */

export function LiveActivityFeed({
  events,
  connected,
  onSelect,
  canSelect,
  collapsed,
  onToggle,
  animate,
}: {
  events: ActivityEvent[];
  connected: boolean;
  onSelect: (e: ActivityEvent) => void;
  canSelect: (e: ActivityEvent) => boolean;
  collapsed: boolean;
  onToggle: () => void;
  animate: boolean;
}) {
  const latest = events[0];
  return (
    <FloatingCard
      title="Live activity"
      icon={<Radio />}
      collapsed={collapsed}
      onToggle={onToggle}
      className={cn(collapsed ? "w-full sm:w-[300px]" : "w-full sm:w-[328px]", "max-w-full")}
      meta={
        collapsed && latest ? (
          <span className="flex min-w-0 items-center gap-1.5">
            <Dot tone={connected ? "ok" : "neutral"} live={connected && animate} />
            <span key={latest.id} className="animate-fade-in truncate text-[12px] text-muted">
              {latest.message || latest.type}
            </span>
          </span>
        ) : (
          <span className="flex items-center gap-1.5 font-mono text-[10.5px] text-subtle">
            <Dot tone={connected ? "ok" : "neutral"} live={connected && animate} />
            {connected ? "streaming" : "reconnecting"}
          </span>
        )
      }
    >
      <ul className="max-h-[188px] overflow-y-auto py-1" aria-live="polite">
        {events.length === 0 ? <li className="px-3.5 py-6 text-center text-[12.5px] text-subtle">No events yet. Activity appears here in real time.</li> : null}
        {events.map((e) => {
          const body = (
            <>
              <Dot tone={EVENT_TONE[e.severity] ?? "info"} className="mt-[7px]" />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[12.5px] text-fg">{e.message || e.type}</span>
                <span className="mt-0.5 flex min-w-0 items-center gap-1.5 font-mono text-[10.5px] text-subtle">
                  <span className="truncate">{e.type}</span>
                  <span aria-hidden>·</span>
                  <time dateTime={e.created_at} className="shrink-0">
                    {timeAgo(e.created_at)}
                  </time>
                </span>
              </span>
            </>
          );
          return (
            <li key={e.id} className="animate-fade-in">
              {canSelect(e) ? (
                <button type="button" onClick={() => onSelect(e)} className="flex w-full items-start gap-2.5 px-3.5 py-1.5 text-left transition-colors hover:bg-surface-2/80" title="Show on map">
                  {body}
                </button>
              ) : (
                <div className="flex items-start gap-2.5 px-3.5 py-1.5">{body}</div>
              )}
            </li>
          );
        })}
      </ul>
    </FloatingCard>
  );
}

/* --------------------------------------------------------- detail drawer */

type NodeDetail = Record<string, any>;

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 py-2 text-[13px]">
      <span className="shrink-0 text-muted">{k.charAt(0).toUpperCase() + k.slice(1)}</span>
      <span className="min-w-0 break-words text-right text-fg">{v ?? "—"}</span>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mt-5 first:mt-0">
      <h3 className="label mb-1">{title}</h3>
      <div className="divide-y divide-border/70">{children}</div>
    </section>
  );
}

const linkBtn = "inline-flex h-8 items-center justify-center gap-1.5 rounded-lg border bg-surface px-3 text-[12.5px] font-medium text-fg shadow-xs transition-colors hover:border-border-strong hover:bg-surface-2/60 [&_svg]:h-3.5 [&_svg]:w-3.5";

export function ConnectionDetailsPanel({ node, onClose, onChanged }: { node: GraphNode; onClose: () => void; onChanged: () => void }) {
  const { can } = useAuth();
  const qc = useQueryClient();
  const isSource = node.id.startsWith("db:") || node.id.startsWith("int:");
  const kind = node.id.startsWith("db:") ? "db" : node.id.startsWith("int:") ? "int" : null;
  const rid = node.id.split(":")[1];
  const detail = useQuery({ queryKey: ["node-detail", node.id], queryFn: () => get<NodeDetail>(`/connections/node/${node.id}`), enabled: isSource, retry: false });
  const [confirm, setConfirm] = useState(false);
  const base = kind === "db" ? `/database-connections/${rid}` : `/integrations/${rid}`;
  const canTest = kind === "db" ? can("db.test") : can("org.integrations");
  const canSync = kind === "db" ? can("db.sync") : can("org.integrations");
  const canDelete = kind === "db" ? can("db.delete") : can("org.integrations");
  const d = detail.data;
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["node-detail", node.id] });
    onChanged();
  };
  const test = useMutation({ mutationFn: () => post<{ ok: boolean; error?: string; latency_ms?: number }>(`${base}/test`), onSuccess: (r) => { r.ok ? toast.success(`Connection OK${r.latency_ms ? ` · ${Math.round(r.latency_ms)} ms` : ""}`) : toast.error(r.error || "Connection failed"); refresh(); } });
  const sync = useMutation({ mutationFn: () => post<{ job_id: string }>(`${base}/sync`, { mode: "incremental" }), onSuccess: () => { toast.success("Synchronization queued"); refresh(); } });
  const pause = useMutation({ mutationFn: (p: boolean) => post(`${base}/${p ? "pause" : "resume"}`), onSuccess: (_r, p) => { toast.success(p ? "Sync paused" : "Sync resumed"); refresh(); } });
  const remove = useMutation({
    mutationFn: () => del(base, { confirm_name: d?.name ?? node.label, delete_synced_documents: false }),
    onSuccess: () => { toast.success("Disconnected"); setConfirm(false); onClose(); onChanged(); },
  });
  const m = node.data;
  const Icon = NODE_ICON[node.type] ?? Boxes;
  const family = NODE_FAMILY[node.type] ?? "source";
  const tone = STATUS_TONE[node.status] ?? "neutral";
  return (
    <GlassPanel solid className="flex h-full w-full flex-col overflow-hidden rounded-2xl animate-scale-in sm:w-[372px]">
      <div className="flex items-start gap-3.5 border-b border-border/80 px-4 pb-4 pt-4">
        <div className="cx-node !w-auto shrink-0 pt-1" style={{ "--tone": toneVar(tone) } as React.CSSProperties}>
          <div className="cx-tile-wrap !h-10 !w-10">
            <span className="cx-ring !rounded-[15px]" data-status={node.status} aria-hidden />
            <div className={cn("cx-tile !rounded-[11px] [&_svg]:!h-[17px] [&_svg]:!w-[17px]", `cx-tile--${family}`)}>
              <Icon />
            </div>
          </div>
        </div>
        <div className="min-w-0 flex-1">
          <div className="label">{TYPE_LABEL[node.type] ?? node.type.replace(/_/g, " ")}</div>
          <h2 className="mt-1 truncate text-[16px] font-semibold tracking-[-0.015em]" title={node.label}>
            {node.label}
          </h2>
          <div className="mt-2">
            <Badge tone={tone}>
              <Dot tone={tone} live={node.status === "syncing"} />
              {STATUS_LABEL[node.status] ?? node.status}
            </Badge>
          </div>
        </div>
        <button type="button" onClick={onClose} className="-mr-1 -mt-1 grid h-8 w-8 shrink-0 place-items-center rounded-lg text-subtle transition-colors hover:bg-surface-2 hover:text-fg" aria-label="Close details">
          <X className="h-4 w-4" />
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        {isSource && detail.isLoading ? <Loader2 className="mx-auto my-8 h-5 w-5 animate-spin text-subtle" /> : null}
        {isSource && detail.isError ? <p className="text-[13px] text-muted">Details for this connection could not be loaded.</p> : null}
        {isSource && d ? (
          <>
            <Section title="Connection">
              <Row k="Provider" v={d.provider_label} />
              <Row k="Connection type" v={kind === "db" ? d.access_mode : d.category} />
              {kind === "db" ? <Row k="Endpoint" v={<span className="font-mono text-[12px]">{d.endpoint}</span>} /> : null}
              {kind === "db" ? <Row k="Database" v={d.database} /> : null}
              {kind === "int" && d.config?.start_url ? <Row k="Endpoint" v={<span className="font-mono text-[12px]">{d.config.start_url || d.config.feed_url || d.config.sitemap_url || d.config.repository}</span>} /> : null}
              <Row k="Status" v={<StatusBadge status={d.status} />} />
            </Section>
            <Section title="Synchronization">
              <Row k="Last successful connection" v={d.last_success_at ? formatDate(d.last_success_at) : "never"} />
              <Row k="Last synchronization" v={d.last_sync_at ? formatDate(d.last_sync_at) : "never"} />
              <Row k="Sync schedule" v={d.sync_interval_minutes ? `every ${d.sync_interval_minutes} min${d.sync_paused ? " (paused)" : ""}` : "manual"} />
              {kind === "db" ? <Row k="Tables" v={`${d.selected_tables?.length ?? 0} selected of ${d.table_count ?? 0}`} /> : null}
              <Row k="Indexed documents" v={<span className="num">{formatNumber(m.documents)}</span>} />
            </Section>
            <Section title="Access">
              <Row k="Knowledge base" v={d.knowledge_base ? <Link className="text-accent hover:underline" href={`/knowledge-bases/${d.knowledge_base.id}`}>{d.knowledge_base.name}</Link> : "—"} />
              {d.authorized ? <Row k="Authorized grants" v={d.authorized.length ? d.authorized.map((g: any) => `${g.effect === "deny" ? "deny " : ""}${g.subject_type}:${g.privilege}`).join(", ") : "managers only"} /> : null}
            </Section>
            {d.last_error ? (
              <div className="mt-4 rounded-lg border border-danger/25 bg-danger/[0.06] px-3 py-2.5 text-[12.5px] text-fg">
                <div className="mb-0.5 flex items-center gap-1.5 font-medium text-danger">
                  <CircleAlert className="h-3.5 w-3.5" /> Error diagnostics
                </div>
                <div className="break-words text-muted">{d.last_error}</div>
              </div>
            ) : null}
            {d.recent_runs?.length ? (
              <section className="mt-5">
                <h3 className="label mb-1.5">Recent syncs</h3>
                <ul className="divide-y divide-border/70">
                  {d.recent_runs.slice(0, 5).map((r: any, i: number) => (
                    <li key={i} className="flex items-center justify-between gap-3 py-2 text-[12.5px]">
                      <span className="flex min-w-0 items-center gap-2 text-muted">
                        <Dot tone={r.status === "succeeded" ? "ok" : r.status === "failed" ? "danger" : "info"} />
                        <span className="truncate">
                          {timeAgo(r.started_at)} · {r.mode}
                        </span>
                      </span>
                      <span className={cn("num shrink-0 font-mono text-[11.5px]", r.status === "succeeded" ? "text-fg" : r.status === "failed" ? "text-danger" : "text-info")}>
                        {r.status === "succeeded" ? `+${r.items_added} ~${r.items_updated} −${r.items_deleted}` : r.status}
                      </span>
                    </li>
                  ))}
                </ul>
              </section>
            ) : null}
          </>
        ) : null}
        {!isSource ? (
          <Section title="Properties">
            {Object.entries(m)
              .filter(([k, v]) => !["resource_id", "resource_kind", "members"].includes(k) && v !== null && v !== undefined && typeof v !== "object")
              .map(([k, v]) => (
                <Row key={k} k={k.replace(/_/g, " ")} v={typeof v === "number" ? <span className="num">{k.endsWith("bytes") ? formatBytes(v) : formatNumber(v)}</span> : String(v)} />
              ))}
            {Array.isArray(m.stages) ? <Row k="stages" v={m.stages.join(" → ")} /> : null}
          </Section>
        ) : null}
        {node.type === "knowledge_base" ? (
          <Link href={`/knowledge-bases/${rid}`} className={cn(linkBtn, "mt-5 w-full")}>
            <ExternalLink /> Open knowledge base
          </Link>
        ) : null}
      </div>
      {isSource && d ? (
        <div className="grid grid-cols-2 gap-2 border-t border-border/80 bg-surface-2/40 p-3">
          <ActionBtn icon={PlugZap} label="Test connection" onClick={() => test.mutate()} loading={test.isPending} disabled={!canTest} perm={kind === "db" ? "db.test" : "org.integrations"} />
          <ActionBtn icon={RefreshCw} label="Synchronize now" onClick={() => sync.mutate()} loading={sync.isPending} disabled={!canSync || (kind === "db" && d.access_mode === "live") || (kind === "int" && !d.kb_id)} perm={kind === "db" ? "db.sync" : "org.integrations"} hint={kind === "db" && d.access_mode === "live" ? "Live-query connections are not synchronized" : undefined} />
          <ActionBtn icon={d.sync_paused ? Play : Pause} label={d.sync_paused ? "Resume sync" : "Pause sync"} onClick={() => pause.mutate(!d.sync_paused)} loading={pause.isPending} disabled={!canSync} perm={kind === "db" ? "db.sync" : "org.integrations"} />
          <Link href={kind === "db" ? `${base}` : "/integrations"} className={linkBtn}>
            <ExternalLink /> Details & logs
          </Link>
          {kind === "db" ? (
            <Link href={`${base}?tab=permissions`} className={linkBtn}>
              <ShieldCheck /> Permissions
            </Link>
          ) : <span />}
          <ActionBtn icon={Trash2} label="Disconnect" danger onClick={() => setConfirm(true)} disabled={!canDelete} perm={kind === "db" ? "db.delete" : "org.integrations"} />
        </div>
      ) : null}
      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title={`Disconnect ${d?.name ?? node.label}?`}
        description="The connection and its stored credentials are deleted. Documents already synchronized stay in their knowledge base."
        requireText={d?.name ?? node.label}
        confirmLabel="Disconnect"
        loading={remove.isPending}
        onConfirm={() => remove.mutate()}
      />
    </GlassPanel>
  );
}

function ActionBtn({ icon: Icon, label, onClick, loading, disabled, danger, perm, hint }: { icon: React.ComponentType<{ className?: string }>; label: string; onClick: () => void; loading?: boolean; disabled?: boolean; danger?: boolean; perm: string; hint?: string }) {
  const btn = (
    <Button size="sm" variant="secondary" onClick={onClick} loading={loading} disabled={disabled} className={cn("w-full justify-center text-[12.5px]", danger && "text-danger hover:border-danger/40 hover:bg-danger/[0.06]")}>
      {!loading ? <Icon className="h-3.5 w-3.5" /> : null} {label}
    </Button>
  );
  if (disabled) {
    return (
      <Tip content={hint || `Requires the ${perm} permission`}>
        <span className="block">{btn}</span>
      </Tip>
    );
  }
  return btn;
}
