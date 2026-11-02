"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Boxes, Cable, Cpu, Database, FileSearch, Layers, MessageSquareText, Scissors, Server, Wand2 } from "lucide-react";
import Link from "next/link";

import { NODE_FAMILY, NODE_ICON, type Family } from "@/components/connections/meta";
import type { GraphNode } from "@/components/connections/types";
import { Button } from "@/components/ui/button";
import { Badge, Card, Dot, EmptyState, Skeleton, StatusBadge, type Tone } from "@/components/ui/primitives";
import { get } from "@/lib/api";
import { cn, formatDate, formatNumber, timeAgo } from "@/lib/utils";

const TYPE_LABEL: Record<string, string> = { database: "Database", warehouse: "Warehouse", vector_db: "Vector DB", integration: "Integration", document_source: "File uploads", group: "Group" };

const TILE_CLASS: Record<Family, string> = {
  org: "bg-fg text-bg",
  source: "border bg-surface text-fg shadow-xs",
  process: "border bg-surface-2 text-muted",
  knowledge: "bg-accent/10 text-accent ring-1 ring-inset ring-accent/25",
  intelligence: "bg-accent text-accent-fg",
};

function TypeTile({ type, className }: { type: string; className?: string }) {
  const Icon = NODE_ICON[type] ?? Boxes;
  return (
    <span className={cn("grid h-8 w-8 shrink-0 place-items-center rounded-lg [&_svg]:h-4 [&_svg]:w-4", TILE_CLASS[NODE_FAMILY[type] ?? "source"], className)}>
      <Icon />
    </span>
  );
}

/* ------------------------------------------------------------------ list */

export function ConnectionsListView({
  nodes,
  kbNames,
  onSelect,
  selectedId,
  filtering,
  onClearFilters,
}: {
  nodes: GraphNode[];
  kbNames: Record<string, string>;
  onSelect: (n: GraphNode) => void;
  selectedId?: string | null;
  filtering?: boolean;
  onClearFilters?: () => void;
}) {
  return (
    <div className="px-4 py-5 sm:px-6 sm:py-6">
      <div className="mb-3 flex items-end justify-between gap-4">
        <div>
          <h2 className="text-[15px] font-semibold tracking-[-0.015em]">Connected sources</h2>
          <p className="mt-0.5 text-[13px] text-muted">
            <span className="num">{nodes.length}</span> {nodes.length === 1 ? "source" : "sources"} in view{filtering ? " (filtered)" : ""}. Inspect one to test, synchronize or review access.
          </p>
        </div>
      </div>
      <Card className="overflow-hidden">
        {nodes.length === 0 ? (
          filtering ? (
            <EmptyState
              icon={<Layers className="h-5 w-5" />}
              title="No sources match these filters"
              description="Try another category or status, or clear the filters to see every source you can access."
              action={onClearFilters ? <Button variant="secondary" size="sm" onClick={onClearFilters}>Clear filters</Button> : undefined}
            />
          ) : (
            <EmptyState
              icon={<Database className="h-5 w-5" />}
              title="No sources connected yet"
              description="Connect a database or an integration and it appears here and on the live map."
              action={
                <>
                  <Link href="/database-connections" className="inline-flex h-8 items-center gap-1.5 rounded-lg border bg-surface px-3 text-[13px] font-medium shadow-xs hover:border-border-strong">
                    <Database className="h-3.5 w-3.5" /> Database connections
                  </Link>
                  <Link href="/integrations" className="inline-flex h-8 items-center gap-1.5 rounded-lg border bg-surface px-3 text-[13px] font-medium shadow-xs hover:border-border-strong">
                    <Cable className="h-3.5 w-3.5" /> Integrations
                  </Link>
                </>
              }
            />
          )
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[880px] text-[13px]">
              <thead>
                <tr className="border-b bg-surface-2/60 text-left">
                  {["Resource", "Provider", "Type", "Status", "Last synchronization", "Knowledge base", "Documents", ""].map((h, i) => (
                    <th key={h || i} className={cn("label h-10 px-4 !font-medium", h === "Documents" && "text-right")}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {nodes.map((n) => {
                  const sel = n.id === selectedId;
                  return (
                    <tr key={n.id} className={cn("group h-12 transition-colors hover:bg-surface-2/50", sel && "bg-accent/[0.05]")}>
                      <td className={cn("relative px-4", sel && "before:absolute before:inset-y-2 before:left-0 before:w-[3px] before:rounded-r-full before:bg-accent")}>
                        <div className="flex items-center gap-3">
                          <TypeTile type={n.type} />
                          <span className="min-w-0 truncate font-medium text-fg">{n.label}</span>
                        </div>
                      </td>
                      <td className="px-4 text-muted">{n.data.provider_label ?? n.data.provider ?? "—"}</td>
                      <td className="px-4">
                        <span className="text-muted">{TYPE_LABEL[n.type] ?? n.type}</span>
                        {n.data.access_mode ? <Badge className="ml-2">{n.data.access_mode}</Badge> : null}
                      </td>
                      <td className="px-4">
                        <StatusBadge status={n.status} />
                      </td>
                      <td className="px-4">
                        {n.data.last_sync_at ? (
                          <span className="font-mono text-[12px] text-muted" title={formatDate(n.data.last_sync_at)}>
                            {timeAgo(n.data.last_sync_at)}
                          </span>
                        ) : (
                          <span className="text-subtle">—</span>
                        )}
                      </td>
                      <td className="px-4 text-muted">{n.data.kb_id ? kbNames[n.data.kb_id] ?? "restricted" : n.data.kbs ? `${n.data.kbs} KBs` : "—"}</td>
                      <td className="num px-4 text-right font-medium">{formatNumber(n.data.documents ?? 0)}</td>
                      <td className="px-4 text-right">
                        <Button variant={sel ? "primary" : "secondary"} size="xs" onClick={() => onSelect(n)}>
                          Inspect
                        </Button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------- data flow */

type Flow = { stages: { key: string; label: string; count: number; detail: string }[]; documents: Record<string, number>; active_jobs: number };

const STAGE_ICON: Record<string, React.ComponentType<{ className?: string }>> = {
  source: Database,
  parsing: FileSearch,
  normalization: Wand2,
  chunking: Scissors,
  embeddings: Cpu,
  indexing: Server,
  retrieval: Layers,
  response: MessageSquareText,
};
const STAGE_FAMILY: Record<string, Family> = { source: "source", retrieval: "knowledge", response: "intelligence" };
const PROCESSING = ["parsing", "normalization", "chunking", "embeddings", "indexing"];
/** "avg 113742 ms generation" -> "avg 1m 54s generation" (durations in human units). */
function humanizeMs(text: string): string {
  return text.replace(/(\d+(?:\.\d+)?)\s?ms\b/g, (_, raw: string) => {
    const ms = Number(raw);
    if (ms < 1000) return `${Math.round(ms)} ms`;
    if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)} s`;
    return `${Math.floor(ms / 60_000)}m ${Math.round((ms % 60_000) / 1000)}s`;
  });
}

const DOC_TONE: Record<string, Tone> = { ready: "ok", indexed: "ok", processing: "info", queued: "info", pending: "warn", failed: "danger", error: "danger" };

function StageTile({ stageKey, busy, animate, size = "md" }: { stageKey: string; busy: boolean; animate: boolean; size?: "md" | "sm" }) {
  const Icon = STAGE_ICON[stageKey] ?? Layers;
  const fam = STAGE_FAMILY[stageKey] ?? "process";
  return (
    <div className={cn("cx-node !w-auto", busy && "is-busy")} style={{ "--tone": "var(--info)" } as React.CSSProperties}>
      <div className={cn("cx-tile-wrap", size === "sm" && "!h-10 !w-10")}>
        {busy ? <span className="cx-ring" data-status="syncing" aria-hidden /> : null}
        {busy && animate ? <span className="cx-sweep" aria-hidden /> : null}
        <div className={cn("cx-tile", `cx-tile--${fam}`, size === "sm" && "!rounded-[11px] [&_svg]:!h-[17px] [&_svg]:!w-[17px]")}>
          <Icon />
        </div>
      </div>
    </div>
  );
}

export function DataFlowView({ orgKey, kbId, animate }: { orgKey: string | null; kbId: string; animate: boolean }) {
  const { data, isError, refetch } = useQuery({ queryKey: ["data-flow", orgKey, kbId], queryFn: () => get<Flow>("/connections/data-flow", { kb_id: kbId || undefined }), refetchInterval: 5000, enabled: Boolean(orgKey) });
  const stages = data?.stages ?? [];
  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1240px] px-4 py-6 sm:px-6 sm:py-8">
        <header className="mb-6 flex flex-wrap items-end justify-between gap-x-6 gap-y-4 animate-fade-up">
          <div className="min-w-0 max-w-2xl">
            <div className="label mb-2.5 flex items-center gap-2">
              <Dot tone="accent" live={animate} /> Pipeline · live
            </div>
            <h2 className="text-[22px] font-semibold leading-tight tracking-[-0.025em]">From source to grounded answer</h2>
            <p className="mt-1.5 text-[13.5px] leading-relaxed text-muted">Counts come from documents, chunks, vectors, running ingestion jobs and answered questions in your scope. They refresh every few seconds.</p>
          </div>
          {data ? (
            <div className="flex items-center gap-3 rounded-xl border bg-surface px-4 py-3 shadow-card">
              <span className="label">Active ingestion jobs</span>
              <span className="num text-[22px] font-semibold leading-none tracking-[-0.03em]">{formatNumber(data.active_jobs)}</span>
            </div>
          ) : null}
        </header>

        {isError ? (
          <Card>
            <EmptyState title="The pipeline could not be loaded" description="Check your connection and try again." action={<Button variant="secondary" size="sm" onClick={() => refetch()}>Try again</Button>} />
          </Card>
        ) : (
          <Card className="relative overflow-hidden">
            <div className="pointer-events-none absolute inset-0 bg-dots opacity-60 [mask-image:radial-gradient(ellipse_70%_80%_at_50%_45%,black,transparent)]" aria-hidden />
            {/* wide screens: one horizontal rail */}
            <ol className="relative hidden grid-cols-8 xl:grid">
              <span className="absolute left-[6.25%] right-[6.25%] top-[51px] h-px bg-gradient-to-r from-border-strong via-accent/40 to-border-strong" aria-hidden>
                {animate && data ? [0, 1, 2].map((i) => <span key={i} className="cx-rail-dot" style={{ animationDelay: `${i * 2.15}s` }} />) : null}
              </span>
              {(data ? stages : Array.from({ length: 8 }).map((_, i) => ({ key: String(i), label: "", count: 0, detail: "" }))).map((s, i) => {
                const busy = PROCESSING.includes(s.key) && s.count > 0;
                return (
                  <li key={s.key} className="relative flex flex-col items-center px-2.5 pb-7 pt-7 text-center">
                    {data ? <StageTile stageKey={s.key} busy={busy} animate={animate} /> : <Skeleton className="h-[46px] w-[46px] rounded-[13px]" />}
                    <span className="label mt-4 !text-[10px]">{String(i + 1).padStart(2, "0")}</span>
                    <span className="mt-1 text-[13px] font-semibold tracking-[-0.01em]">{s.label || <Skeleton className="h-4 w-16" />}</span>
                    <span className="num mt-2 text-[26px] font-semibold leading-none tracking-[-0.035em]">{data ? formatNumber(s.count) : "–"}</span>
                    <span className="mt-2 line-clamp-2 min-h-[32px] text-[11.5px] leading-snug text-muted">{humanizeMs(s.detail)}</span>
                    {busy ? <span className="mt-2 inline-flex items-center gap-1.5 font-mono text-[10.5px] text-info"><Dot tone="info" live={animate} /> processing</span> : null}
                  </li>
                );
              })}
            </ol>
            {/* everything narrower: a vertical rail */}
            <ol className="relative xl:hidden">
              {(data ? stages : Array.from({ length: 8 }).map((_, i) => ({ key: String(i), label: "", count: 0, detail: "" }))).map((s, i, all) => {
                const busy = PROCESSING.includes(s.key) && s.count > 0;
                return (
                  <li key={s.key} className="relative flex items-center gap-4 px-4 py-3.5 sm:px-5">
                    {i < all.length - 1 ? (
                      <span className="absolute bottom-[-10px] left-[35.5px] top-[58px] w-px bg-gradient-to-b from-border-strong to-accent/30 sm:left-[39.5px]" aria-hidden>
                        {animate && data ? <span className="cx-rail-dot-v" style={{ animationDelay: `${i * 0.3}s` }} /> : null}
                      </span>
                    ) : null}
                    {data ? <StageTile stageKey={s.key} busy={busy} animate={animate} size="sm" /> : <Skeleton className="h-10 w-10 rounded-[11px]" />}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="label !text-[10px]">{String(i + 1).padStart(2, "0")}</span>
                        <span className="truncate text-[13.5px] font-semibold tracking-[-0.01em]">{s.label}</span>
                        {busy ? <Dot tone="info" live={animate} /> : null}
                      </div>
                      <div className="mt-0.5 truncate text-[12px] text-muted">{humanizeMs(s.detail)}</div>
                    </div>
                    <span className="num shrink-0 text-[20px] font-semibold tracking-[-0.03em]">{data ? formatNumber(s.count) : "–"}</span>
                  </li>
                );
              })}
            </ol>
          </Card>
        )}

        {data ? (
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <span className="label mr-1">Documents</span>
            {Object.entries(data.documents).map(([k, v]) => (
              <Badge key={k} tone={DOC_TONE[k] ?? "neutral"}>
                <Dot tone={DOC_TONE[k] ?? "neutral"} />
                <span className="capitalize">{k}</span>
                <span className="num font-mono">{v}</span>
              </Badge>
            ))}
            <Link href="/documents" className="ml-auto inline-flex items-center gap-1.5 text-[13px] font-medium text-accent underline-offset-4 hover:underline">
              Open document library <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        ) : null}
      </div>
    </div>
  );
}
