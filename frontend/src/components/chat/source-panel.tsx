"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowUpRight, ExternalLink, FileSearch, Lock, ShieldAlert, X } from "lucide-react";
import Link from "next/link";

import { SourceIcon } from "@/components/chat/message";
import type { Citation } from "@/components/chat/types";
import { locatorParts } from "@/components/chat/utils";
import { Skeleton, Tip } from "@/components/ui/primitives";
import { get } from "@/lib/api";
import { cn } from "@/lib/utils";

type Preview = {
  citation: { title: string; locator: Record<string, any>; snippet: string };
  document: { id: string; title: string; filename: string; ext: string; source_url: string | null } | null;
  context: { id: string; ordinal: number; content: string; metadata: Record<string, any>; is_cited: boolean }[];
};

function Locator({ loc, className }: { loc: Record<string, any>; className?: string }) {
  const parts = locatorParts(loc);
  const section: string[] = Array.isArray(loc?.section) ? loc.section : [];
  if (!parts.length && !section.length) return null;
  return (
    <div className={cn("flex flex-wrap items-center gap-1", className)}>
      {parts.map((p) => (
        <span key={p} className="rounded-[5px] bg-surface-2 px-1.5 py-px font-mono text-[11px] text-muted ring-1 ring-inset ring-border">
          {p}
        </span>
      ))}
      {section.length ? (
        <span className="min-w-0 font-mono text-[11px] leading-5 text-subtle">
          {section.map((s, i) => (
            <span key={i}>
              {i ? <span className="px-1 text-border-strong">/</span> : null}
              <span className={i === section.length - 1 ? "text-muted" : undefined}>{s}</span>
            </span>
          ))}
        </span>
      ) : null}
    </div>
  );
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.max(0, Math.min(1, score));
  return (
    <Tip content="Relevance score">
      <span className="flex shrink-0 cursor-default items-center gap-1.5 self-center" aria-label={`Relevance ${Math.round(pct * 100)}`}>
        <span className="h-[3px] w-7 overflow-hidden rounded-full bg-border">
          <span className="block h-full rounded-full bg-accent/60" style={{ width: `${pct * 100}%` }} />
        </span>
        <span className="num w-5 text-right font-mono text-[10.5px] leading-none text-subtle">{Math.round(score * 100)}</span>
      </span>
    </Tip>
  );
}

function SourceRow({ c, active, onSelect }: { c: Citation; active: boolean; onSelect: () => void }) {
  const leaf = Array.isArray(c.locator?.section) && c.locator.section.length ? c.locator.section[c.locator.section.length - 1] : null;
  const parts = locatorParts(c.locator);
  const sub = [...parts, leaf ? `§ ${leaf}` : null].filter(Boolean).join(" · ") || c.filename || c.source_type;
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        aria-current={active || undefined}
        className={cn(
          "relative flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-inset focus-visible:ring-ring/30",
          active ? "bg-accent/[0.07]" : "hover:bg-surface-2/80",
        )}
      >
        {active ? <span className="absolute inset-y-2 left-0 w-[2px] rounded-full bg-accent" aria-hidden /> : null}
        <span
          className={cn(
            "mt-px grid h-5 min-w-5 shrink-0 place-items-center rounded-[5px] px-1 font-mono text-[10px] font-semibold",
            c.cited ? "bg-accent/10 text-accent ring-1 ring-inset ring-accent/20" : "bg-surface-2 text-subtle ring-1 ring-inset ring-border",
          )}
        >
          {c.label.slice(1)}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1.5">
            <SourceIcon c={c} className="h-3 w-3 shrink-0 text-subtle" />
            <span className={cn("truncate text-[13px] font-medium leading-5", c.accessible === false && "text-muted")}>{c.title}</span>
            {c.flagged ? <ShieldAlert className="h-3 w-3 shrink-0 text-warn" aria-label="Contains instruction-like text" /> : null}
          </span>
          <span className="mt-0.5 block truncate font-mono text-[10.5px] leading-4 text-subtle">{sub}</span>
        </span>
        {typeof c.score === "number" ? <ScoreBar score={c.score} /> : null}
      </button>
    </li>
  );
}

export function SourcePanel({
  sources,
  active,
  onSelect,
  onClose,
  earlier,
}: {
  sources: Citation[];
  active: Citation | null;
  onSelect: (c: Citation) => void;
  onClose: () => void;
  /** The list belongs to an earlier answer, not the latest one. */
  earlier?: boolean;
}) {
  const preview = useQuery({
    queryKey: ["citation-preview", active?.id],
    queryFn: () => get<Preview>(`/citations/${active!.id}/preview`),
    enabled: Boolean(active?.id && active.document_id && active.accessible !== false),
    retry: false,
    meta: { silent: true },
  });
  const cited = sources.filter((c) => c.cited);
  const other = sources.filter((c) => !c.cited);
  const groups = cited.length && other.length ? [{ label: `Cited in answer`, items: cited }, { label: `Also retrieved`, items: other }] : [{ label: cited.length ? "Cited in answer" : "Retrieved passages", items: sources }];

  return (
    <aside className="flex h-full w-full flex-col bg-surface" aria-label="Sources">
      <div className="flex h-12 shrink-0 items-center justify-between gap-3 border-b px-4">
        <div className="flex items-center gap-2">
          <span className="text-[13.5px] font-semibold tracking-[-0.01em]">Sources</span>
          <span className="num rounded-full bg-surface-2 px-1.5 py-px font-mono text-[10.5px] text-muted ring-1 ring-inset ring-border">{sources.length}</span>
          {earlier ? <span className="label">· earlier answer</span> : null}
        </div>
        <button className="-mr-1.5 rounded-lg p-1.5 text-subtle transition-colors hover:bg-surface-2 hover:text-fg" onClick={onClose} aria-label="Close sources">
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="flex min-h-0 flex-1 flex-col">
        <div className="max-h-[42%] shrink-0 overflow-y-auto border-b px-2 pb-2">
          {groups.map((g) => (
            <div key={g.label}>
              <div className="label px-2.5 pb-1 pt-3">
                {g.label} <span className="text-border-strong">/</span> <span className="num">{g.items.length}</span>
              </div>
              <ul className="space-y-px">
                {g.items.map((c) => (
                  <SourceRow key={c.label} c={c} active={active?.label === c.label} onSelect={() => onSelect(c)} />
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto">
          {!active ? (
            <div className="flex h-full flex-col items-center justify-center px-8 py-10 text-center">
              <span className="grid h-10 w-10 place-items-center rounded-xl border bg-surface text-subtle shadow-card">
                <FileSearch className="h-4 w-4" />
              </span>
              <p className="mt-3 text-[13px] font-medium">Inspect a passage</p>
              <p className="mt-1 max-w-[240px] text-[12.5px] leading-relaxed text-muted">Select a source to read the exact text the answer relied on.</p>
            </div>
          ) : (
            <div className="space-y-4 p-4" key={active.label}>
              <div className="animate-fade-in">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className={cn("rounded-[5px] px-1.5 py-px font-mono text-[10.5px] font-semibold ring-1 ring-inset", active.cited ? "bg-accent/10 text-accent ring-accent/20" : "bg-surface-2 text-muted ring-border")}>
                    {active.label} · {active.cited ? "cited" : "retrieved"}
                  </span>
                  <span className="rounded-[5px] bg-surface-2 px-1.5 py-px font-mono text-[10.5px] text-muted ring-1 ring-inset ring-border">{active.source_type}</span>
                  {active.flagged ? (
                    <span className="inline-flex items-center gap-1 rounded-[5px] bg-warn/10 px-1.5 py-px text-[11px] font-medium text-warn ring-1 ring-inset ring-warn/25">
                      <ShieldAlert className="h-3 w-3" /> instruction-like text
                    </span>
                  ) : null}
                </div>
                <h3 className="mt-2.5 text-[16px] font-semibold leading-snug tracking-[-0.015em] [overflow-wrap:anywhere]">{active.title}</h3>
                <Locator loc={active.locator} className="mt-1.5" />
                {active.locator?.url || (active.document_id && active.accessible !== false) ? (
                  <div className="mt-3 flex flex-wrap items-center gap-1.5">
                    {active.document_id && active.accessible !== false ? (
                      <Link
                        href={`/documents?doc=${active.document_id}`}
                        className="inline-flex h-7 items-center gap-1 rounded-md border bg-surface px-2.5 text-[12px] font-medium shadow-xs transition-colors hover:border-border-strong hover:bg-surface-2/60"
                      >
                        Open document <ArrowUpRight className="h-3 w-3 text-subtle" />
                      </Link>
                    ) : null}
                    {active.locator?.url ? (
                      <a
                        href={active.locator.url}
                        target="_blank"
                        rel="noopener noreferrer nofollow"
                        className="inline-flex h-7 items-center gap-1 rounded-md px-2 text-[12px] font-medium text-muted transition-colors hover:bg-surface-2 hover:text-fg"
                      >
                        <ExternalLink className="h-3 w-3" /> Original page
                      </a>
                    ) : null}
                  </div>
                ) : null}
              </div>

              {active.accessible === false ? (
                <div className="flex items-center gap-2.5 rounded-xl border bg-surface-2/60 px-3.5 py-3 text-[13px] text-muted">
                  <Lock className="h-4 w-4 shrink-0 text-subtle" /> You no longer have access to this source.
                </div>
              ) : active.locator?.sql ? (
                <div className="space-y-2">
                  <div className="label">Read-only query</div>
                  <pre className="overflow-x-auto rounded-lg border bg-surface-2/70 p-3 font-mono text-[11.5px] leading-relaxed">{active.locator.sql}</pre>
                  <div className="prose-chat whitespace-pre-wrap text-xs">{active.snippet.split("```").pop()}</div>
                </div>
              ) : preview.isLoading && active.id ? (
                <div className="space-y-2">
                  <Skeleton className="h-16" />
                  <Skeleton className="h-28" />
                  <Skeleton className="h-16" />
                </div>
              ) : preview.data ? (
                <div>
                  <div className="label mb-2">Passage in context</div>
                  <div className="space-y-2">
                    {preview.data.context.map((ctx) => (
                      <div
                        key={ctx.id}
                        className={cn(
                          "relative whitespace-pre-wrap rounded-lg px-3.5 py-2.5 text-[13px] leading-relaxed [overflow-wrap:anywhere]",
                          ctx.is_cited ? "bg-accent/[0.06] text-fg ring-1 ring-inset ring-accent/25" : "text-muted",
                        )}
                      >
                        {ctx.is_cited ? <span className="absolute inset-y-2.5 left-0 w-[2px] rounded-full bg-accent" aria-hidden /> : null}
                        {ctx.content}
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div>
                  <div className="label mb-2">Passage</div>
                  <div className="relative whitespace-pre-wrap rounded-lg bg-accent/[0.06] px-3.5 py-2.5 text-[13px] leading-relaxed ring-1 ring-inset ring-accent/25 [overflow-wrap:anywhere]">
                    <span className="absolute inset-y-2.5 left-0 w-[2px] rounded-full bg-accent" aria-hidden />
                    {active.snippet}
                  </div>
                </div>
              )}

              {active.scores ? (
                <div>
                  <div className="label mb-2">Scores</div>
                  <div className="grid grid-cols-3 divide-x overflow-hidden rounded-lg border">
                    {(["dense", "sparse", "rerank"] as const).map((k) => (
                      <div key={k} className="px-3 py-2">
                        <div className="label">{k === "sparse" ? "BM25" : k}</div>
                        <div className="num mt-1 font-mono text-[13px] font-semibold">{active.scores?.[k] !== undefined ? Number(active.scores[k]).toFixed(k === "sparse" ? 2 : 3) : "—"}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
