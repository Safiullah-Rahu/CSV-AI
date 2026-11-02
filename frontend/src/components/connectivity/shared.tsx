"use client";

import { type QueryClient, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Braces, Check, CheckCircle2, Copy, Database, Info, Orbit, Search, Warehouse, XCircle } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { toast } from "sonner";

import { Button, type ButtonProps } from "@/components/ui/button";
import { Input } from "@/components/ui/form";
import { Badge, Checkbox, Dot, Tip } from "@/components/ui/primitives";
import { errorMessage, get } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatDate, formatDuration, formatNumber, timeAgo } from "@/lib/utils";

import type { Job, KnowledgeBase, Member, ProviderStatus } from "./types";

/* ---------------------------------------------------------- permissions */

/**
 * A button that is disabled (with a tooltip naming the missing permission or
 * the reason) when `allowed` is false. Disabled buttons swallow pointer events,
 * so the tooltip is anchored on a focusable wrapper.
 */
export function GatedButton({ allowed, reason, perm, children, ...props }: ButtonProps & { allowed: boolean; perm?: string; reason?: React.ReactNode }) {
  if (allowed) return <Button {...props}>{children}</Button>;
  const why = reason ?? (perm ? <>Requires the <span className="font-mono">{perm}</span> permission</> : "Not available");
  return (
    <Tip content={why}>
      <span tabIndex={0} className="inline-flex rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30" aria-label={typeof why === "string" ? why : undefined}>
        <Button {...props} disabled aria-disabled>
          {children}
        </Button>
      </span>
    </Tip>
  );
}

/* --------------------------------------------------------- status badges */

export function ProviderStatusBadge({ status, className }: { status: ProviderStatus | string | null | undefined; className?: string }) {
  if (status === "tested")
    return (
      <Badge tone="ok" className={className}>
        <Dot tone="ok" /> Tested
      </Badge>
    );
  if (status === "beta")
    return (
      <Badge tone="warn" className={className}>
        <Dot tone="warn" /> Beta
      </Badge>
    );
  if (status === "planned")
    return (
      <Badge className={className}>
        <span className="h-1.5 w-1.5 shrink-0 rounded-full border border-subtle" aria-hidden /> Planned
      </Badge>
    );
  return null;
}

export const ACCESS_MODES: { key: "live" | "ingest" | "hybrid"; label: string; description: string }[] = [
  { key: "live", label: "Live query", description: "Questions run as validated, read-only SQL against the database at ask time. Nothing is copied." },
  { key: "ingest", label: "Ingest", description: "Selected tables are synchronised into a knowledge base and indexed. Live querying is disabled." },
  { key: "hybrid", label: "Hybrid", description: "Both: tables are synchronised for retrieval and live read-only queries are allowed for fresh numbers." },
];

export function AccessModeBadge({ mode }: { mode: string }) {
  const label = ACCESS_MODES.find((m) => m.key === mode)?.label ?? mode;
  return <Badge tone={mode === "live" ? "info" : mode === "ingest" ? "accent" : "neutral"}>{label}</Badge>;
}

/* -------------------------------------------------------------- notices */

const noticeTone = {
  info: { box: "border-info/20 bg-info/[0.045]", icon: Info, color: "text-info" },
  warn: { box: "border-warn/25 bg-warn/[0.05]", icon: AlertTriangle, color: "text-warn" },
  danger: { box: "border-danger/20 bg-danger/[0.045]", icon: XCircle, color: "text-danger" },
  ok: { box: "border-ok/20 bg-ok/[0.045]", icon: CheckCircle2, color: "text-ok" },
} as const;

export function Notice({ tone = "info", title, children, action, className }: { tone?: keyof typeof noticeTone; title?: React.ReactNode; children?: React.ReactNode; action?: React.ReactNode; className?: string }) {
  const t = noticeTone[tone];
  const Icon = t.icon;
  return (
    <div role={tone === "danger" ? "alert" : "status"} className={cn("flex flex-wrap items-start gap-x-3 gap-y-2 rounded-xl border px-3.5 py-3 text-[13px] leading-relaxed", t.box, className)}>
      <Icon className={cn("mt-[3px] h-4 w-4 shrink-0", t.color)} aria-hidden />
      <div className="min-w-0 flex-1 basis-48">
        {title ? <div className="font-medium tracking-[-0.005em] text-fg">{title}</div> : null}
        {children ? <div className={cn("break-words text-muted", title && "mt-0.5")}>{children}</div> : null}
      </div>
      {action ? <div className="ml-7 shrink-0 self-center sm:ml-0">{action}</div> : null}
    </div>
  );
}

export function KV({ label, children, mono }: { label: React.ReactNode; children: React.ReactNode; mono?: boolean }) {
  return (
    <div className="min-w-0">
      <dt className="label">{label}</dt>
      <dd className={cn("mt-1.5 truncate text-[13.5px]", mono && "font-mono text-[12.5px] tracking-[-0.01em]")}>{children}</dd>
    </div>
  );
}

/* ----------------------------------------------------------- engines */

/** Neutral engine glyphs by category — never third-party logos. */
const ENGINE_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  database: Database,
  warehouse: Warehouse,
  nosql: Braces,
  vector: Orbit,
};

export function engineIcon(category: string | null | undefined) {
  return ENGINE_ICONS[category ?? ""] ?? Database;
}

/** Square tile holding the engine glyph; `active` tints it with the accent. */
export function EngineTile({ category, size = "md", active, muted, className }: { category: string | null | undefined; size?: "sm" | "md" | "lg"; active?: boolean; muted?: boolean; className?: string }) {
  const Icon = engineIcon(category);
  return (
    <span
      aria-hidden
      className={cn(
        "relative grid shrink-0 place-items-center border bg-surface-2 shadow-[inset_0_1px_0_hsl(var(--highlight))]",
        { sm: "h-7 w-7 rounded-md [&_svg]:h-3.5 [&_svg]:w-3.5", md: "h-9 w-9 rounded-lg [&_svg]:h-4 [&_svg]:w-4", lg: "h-12 w-12 rounded-xl [&_svg]:h-5 [&_svg]:w-5" }[size],
        active ? "border-accent/30 bg-accent/10 text-accent" : muted ? "border-dashed text-subtle" : "text-muted",
        className,
      )}
    >
      <Icon />
    </span>
  );
}

/* ---------------------------------------------------------- code blocks */

const SQL_KEYWORDS =
  "select|from|where|and|or|not|in|is|null|as|on|join|inner|left|right|full|outer|cross|group|by|order|having|limit|offset|union|all|distinct|case|when|then|else|end|with|asc|desc|between|like|ilike|exists|insert|into|values|update|set|delete|create|table|view|drop|alter|true|false|over|partition|cast|interval|returning|fetch|first|rows|only|top";
const SQL_FUNCS = "count|sum|avg|min|max|round|coalesce|date_trunc|substr|substring|lower|upper|length|now|extract|strftime|nullif|abs|concat|to_char";
const SQL_TOKEN = new RegExp(`(--[^\\n]*|/\\*[\\s\\S]*?\\*/)|('(?:[^']|'')*')|(\\b\\d+(?:\\.\\d+)?\\b)|\\b(${SQL_KEYWORDS})\\b|\\b(${SQL_FUNCS})(?=\\s*\\()`, "gi");

/** Lightweight SQL colouring: keywords in the accent, literals muted. Presentation only. */
function highlightSql(src: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  let last = 0;
  let i = 0;
  for (const m of src.matchAll(SQL_TOKEN)) {
    const at = m.index ?? 0;
    if (at > last) out.push(src.slice(last, at));
    const [tok, comment, str, num, kw, fn] = m;
    const cls = comment ? "italic text-subtle" : str ? "text-muted" : num ? "text-muted" : kw ? "font-medium text-accent" : fn ? "text-fg" : "";
    out.push(
      <span key={i++} className={cls}>
        {tok}
      </span>,
    );
    last = at + tok.length;
  }
  if (last < src.length) out.push(src.slice(last));
  return out;
}

export function CodeBlock({ children, className, sql, copy, label }: { children: React.ReactNode; className?: string; sql?: boolean; copy?: boolean; label?: React.ReactNode }) {
  const [copied, setCopied] = useState(false);
  const text = typeof children === "string" ? children : null;
  const doCopy = async () => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      toast.error("Couldn't copy to the clipboard");
    }
  };
  const showBar = Boolean(label) || (copy && text);
  return (
    <div className="min-w-0 overflow-hidden rounded-lg border bg-surface-2/60">
      {showBar ? (
        <div className="flex h-8 items-center justify-between gap-2 border-b bg-surface-2/70 pl-3 pr-1">
          <span className="label truncate">{label}</span>
          {copy && text ? (
            <button type="button" onClick={doCopy} className="inline-flex h-6 items-center gap-1 rounded-md px-2 text-[11.5px] font-medium text-muted transition-colors hover:bg-surface hover:text-fg" aria-label="Copy to clipboard">
              {copied ? <Check className="h-3 w-3 text-ok" /> : <Copy className="h-3 w-3" />}
              {copied ? "Copied" : "Copy"}
            </button>
          ) : null}
        </div>
      ) : null}
      <pre className={cn("max-h-72 overflow-auto whitespace-pre-wrap break-words px-3.5 py-3 font-mono text-[12px] leading-[1.7] tracking-[-0.01em] text-fg", className)}>{sql && text ? highlightSql(text) : children}</pre>
    </div>
  );
}

/** Relative time in mono with the absolute timestamp on hover; em dash when absent. */
export function When({ at, className, empty = "never" }: { at: string | null | undefined; className?: string; empty?: string }) {
  if (!at) return <span className={cn("whitespace-nowrap font-mono text-[12px] tracking-[-0.01em] text-subtle", className)}>{empty}</span>;
  return (
    <time dateTime={at} title={formatDate(at)} className={cn("whitespace-nowrap font-mono text-[12px] tracking-[-0.01em] text-muted", className)}>
      {timeAgo(at)}
    </time>
  );
}

export function formatJson(v: unknown): string {
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

export function formatMs(ms: number | null | undefined): string {
  return formatDuration(ms);
}

export function runDuration(start: string, end: string | null): string {
  if (!end) return "—";
  return formatMs(new Date(end).getTime() - new Date(start).getTime());
}

/* ----------------------------------------------------------- intervals */

export const SYNC_INTERVALS: { value: number; label: string }[] = [
  { value: 0, label: "Manual only" },
  { value: 15, label: "Every 15 minutes" },
  { value: 30, label: "Every 30 minutes" },
  { value: 60, label: "Hourly" },
  { value: 360, label: "Every 6 hours" },
  { value: 720, label: "Every 12 hours" },
  { value: 1440, label: "Daily" },
  { value: 10080, label: "Weekly" },
];

export function intervalLabel(minutes: number | null | undefined): string {
  if (!minutes) return "Manual";
  const known = SYNC_INTERVALS.find((i) => i.value === minutes);
  if (known) return known.label;
  if (minutes % 1440 === 0) return `Every ${minutes / 1440} days`;
  if (minutes % 60 === 0) return `Every ${minutes / 60} hours`;
  return `Every ${minutes} minutes`;
}

/** Interval options that include a non-standard current value. */
export function intervalOptions(current: number | null | undefined, min = 0) {
  const opts = SYNC_INTERVALS.filter((o) => o.value === 0 || o.value >= min);
  if (current && !opts.some((o) => o.value === current)) opts.push({ value: current, label: intervalLabel(current) });
  return opts.sort((a, b) => a.value - b.value);
}

/* ----------------------------------------------------------- lookups */

export function useKnowledgeBases() {
  const { can } = useAuth();
  return useApi<KnowledgeBase[]>(can("kb.view") ? "/knowledge-bases" : null, undefined, { staleTime: 60_000, meta: { silent: true } });
}

export function writableKbs(kbs: KnowledgeBase[] | undefined) {
  return (kbs ?? []).filter((k) => k.access === "write" || k.access === "manage");
}

/** Organization members by user id, for showing who requested/decided something. */
export function useMemberNames() {
  const { can } = useAuth();
  const q = useApi<{ items: Member[] }>(can("org.view") ? "/members" : null, { limit: 500 }, { staleTime: 60_000, meta: { silent: true } });
  return useMemo(() => {
    const map = new Map<string, Member>();
    q.data?.items.forEach((m) => map.set(m.user_id, m));
    return { map, members: q.data?.items ?? [], name: (id: string | null | undefined) => (id ? map.get(id)?.full_name || map.get(id)?.email || `${id.slice(0, 8)}…` : "—") };
  }, [q.data]);
}

/* ------------------------------------------------------- query cache */

export function invalidatePrefix(qc: QueryClient, ...prefixes: string[]) {
  return qc.invalidateQueries({
    predicate: (q) => {
      const k = q.queryKey[0];
      return typeof k === "string" && prefixes.some((p) => k === p || k.startsWith(p + "/") || k.startsWith(p + "?"));
    },
  });
}

/* ---------------------------------------------------------- job watcher */

function countsLine(result: Record<string, unknown> | null | undefined): string | undefined {
  if (!result) return undefined;
  if (typeof result.skipped === "string") return `Skipped: ${result.skipped}`;
  const n = (k: string) => (typeof result[k] === "number" ? (result[k] as number) : null);
  const parts: string[] = [];
  if (n("added") !== null) parts.push(`${formatNumber(n("added"))} added`);
  if (n("updated") !== null) parts.push(`${formatNumber(n("updated"))} updated`);
  if (n("deleted") !== null) parts.push(`${formatNumber(n("deleted"))} deleted`);
  if (n("unchanged")) parts.push(`${formatNumber(n("unchanged"))} unchanged`);
  if (n("failed")) parts.push(`${formatNumber(n("failed"))} failed`);
  if (n("seen") !== null) parts.push(`${formatNumber(n("seen"))} seen`);
  return parts.join(" · ") || undefined;
}

/**
 * Polls `GET /jobs/{id}` every 2 s until the job finishes and reports the
 * outcome as a single updating toast. Polling continues if the user navigates
 * away (the toast is global) and gives up after 30 minutes.
 */
export function useJobWatcher() {
  const qc = useQueryClient();
  return useCallback(
    (jobId: string, opts: { label: string; invalidate?: string[] }) => {
      const id = `job-${jobId}`;
      const started = Date.now();
      toast.loading(`${opts.label}: queued`, { id, description: "Waiting for a worker…" });
      const refresh = () => (opts.invalidate?.length ? invalidatePrefix(qc, ...opts.invalidate) : undefined);
      const tick = async () => {
        let job: Job;
        try {
          job = await get<Job>(`/jobs/${jobId}`);
        } catch (err) {
          toast.error(`${opts.label}: could not read the job status`, { id, description: errorMessage(err) });
          return;
        }
        if (job.status === "succeeded") {
          toast.success(`${opts.label}: finished`, { id, description: countsLine(job.result) });
          refresh();
          return;
        }
        if (job.status === "dead" || job.status === "cancelled") {
          toast.error(`${opts.label}: ${job.status === "dead" ? "failed" : "cancelled"}`, { id, description: job.error || undefined });
          refresh();
          return;
        }
        if (Date.now() - started > 30 * 60_000) {
          toast.message(`${opts.label}: still ${job.status}`, { id, description: "Stopped watching after 30 minutes. Check the runs history for the outcome." });
          return;
        }
        const pct = job.progress ? ` · ${Math.round(job.progress * 100)}%` : "";
        const desc =
          job.status === "failed"
            ? `Attempt ${job.attempts}/${job.max_attempts} failed — retrying automatically. ${job.error ?? ""}`.trim()
            : job.message || (job.status === "queued" ? "Waiting for a worker…" : "Working…");
        toast.loading(`${opts.label}: ${job.status === "failed" ? "retrying" : job.status}${pct}`, { id, description: desc });
        if (job.status === "running") refresh();
        setTimeout(tick, 2000);
      };
      setTimeout(tick, 1200);
    },
    [qc],
  );
}

/* ----------------------------------------------------------- check list */

/** Inline, keyboard-usable multi-select (bordered list of checkboxes, searchable when long). */
export function CheckList({
  options,
  value,
  onChange,
  empty,
  label,
  maxHeight = "max-h-52",
}: {
  options: { value: string; label: React.ReactNode; hint?: React.ReactNode; search?: string; disabled?: boolean }[];
  value: string[];
  onChange: (v: string[]) => void;
  empty?: React.ReactNode;
  label: string;
  maxHeight?: string;
}) {
  const [q, setQ] = useState("");
  const set = new Set(value);
  const shown = q ? options.filter((o) => (o.search ?? String(o.label)).toLowerCase().includes(q.toLowerCase())) : options;
  if (!options.length) return <div className="rounded-lg border border-dashed px-3 py-3 text-[13px] text-subtle">{empty ?? "Nothing to choose from."}</div>;
  return (
    <div className="overflow-hidden rounded-lg border bg-surface" role="group" aria-label={label}>
      {options.length > 7 ? (
        <div className="relative border-b">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-subtle" />
          <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter…" aria-label={`Filter ${label}`} className="h-8 rounded-none border-0 pl-8 text-[13px] shadow-none focus:ring-0" />
        </div>
      ) : null}
      <div className={cn("overflow-y-auto", maxHeight)}>
        {shown.map((o) => (
          <label key={o.value} className={cn("flex cursor-pointer items-start gap-2.5 px-3 py-2 text-[13px] hover:bg-surface-2/70", o.disabled && "cursor-not-allowed opacity-50")}>
            <span className="pt-0.5">
              <Checkbox
                checked={set.has(o.value)}
                disabled={o.disabled}
                label={typeof o.label === "string" ? o.label : o.value}
                onCheckedChange={(v) => onChange(v ? [...value, o.value] : value.filter((x) => x !== o.value))}
              />
            </span>
            <span className="min-w-0 flex-1">
              <span className="block truncate">{o.label}</span>
              {o.hint ? <span className="block text-xs text-subtle">{o.hint}</span> : null}
            </span>
          </label>
        ))}
        {!shown.length ? <div className="px-3 py-3 text-[13px] text-subtle">No matches.</div> : null}
      </div>
      <div className="border-t bg-surface-2/50 px-3 py-1.5 text-2xs text-subtle">
        {value.length} of {options.length} selected
      </div>
    </div>
  );
}

/* ------------------------------------------------------- radio cards */

export function RadioCards<T extends string>({ value, onChange, options, name, layout = "row" }: { value: T; onChange: (v: T) => void; options: { key: T; label: string; description: string }[]; name: string; layout?: "row" | "stack" }) {
  return (
    <div role="radiogroup" aria-label={name} className={cn("grid gap-2", layout === "row" && "sm:grid-cols-3")}>
      {options.map((o) => {
        const active = o.key === value;
        return (
          <button
            key={o.key}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(o.key)}
            className={cn(
              "rounded-xl border bg-surface p-3.5 text-left shadow-xs transition-[border-color,background,box-shadow] duration-150 hover:border-border-strong",
              active && "border-accent/50 bg-accent/[0.04] shadow-[0_0_0_3px_hsl(var(--accent)/0.12)] hover:border-accent/60",
            )}
          >
            <div className="flex items-center gap-2 text-[13px] font-medium tracking-[-0.005em]">
              <span className={cn("grid h-4 w-4 shrink-0 place-items-center rounded-full border border-border-strong bg-surface transition-colors", active && "border-accent bg-accent")}>{active ? <span className="h-1.5 w-1.5 rounded-full bg-accent-fg" /> : null}</span>
              {o.label}
            </div>
            <p className="mt-1.5 text-[12px] leading-relaxed text-muted">{o.description}</p>
          </button>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------- form sections */

/** A titled group of form fields: title + explanation on the left at md+, fields on the right. */
export function FormSection({ title, description, children, className, stacked }: { title: React.ReactNode; description?: React.ReactNode; children: React.ReactNode; className?: string; stacked?: boolean }) {
  return (
    <section className={cn("grid gap-x-8 gap-y-3 border-t pt-5 first:border-t-0 first:pt-0", !stacked && "md:grid-cols-[168px_minmax(0,1fr)]", className)}>
      <div className="min-w-0">
        <h4 className="text-[13px] font-semibold tracking-[-0.01em]">{title}</h4>
        {description ? <p className="mt-1 text-[12px] leading-relaxed text-muted">{description}</p> : null}
      </div>
      <div className="min-w-0 space-y-4">{children}</div>
    </section>
  );
}
