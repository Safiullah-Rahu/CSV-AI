"use client";

import { ChevronDown, History } from "lucide-react";
import { useMemo, useState } from "react";

import { Avatar, Badge, EmptyState, Skeleton } from "@/components/ui/primitives";
import { useApi } from "@/lib/hooks";
import { cn, formatDate, pluralize, timeAgo } from "@/lib/utils";

import { memberLabel, useMemberMap } from "./shared";
import type { KBRevision } from "./types";

const LABEL: Record<string, string> = {
  name: "Name",
  description: "Description",
  visibility: "Visibility",
  tags: "Tags",
  category: "Category",
  workspace_id: "Workspace",
  department_id: "Department",
  embedding_config: "Embeddings",
  chunking_config: "Chunking",
  retrieval_config: "Retrieval",
  generation_config: "Answers",
};

type Change = { path: string; from: unknown; to: unknown };

function flatten(cfg: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(cfg || {})) {
    if (v && typeof v === "object" && !Array.isArray(v)) {
      for (const [k2, v2] of Object.entries(v as Record<string, unknown>)) out[`${k}.${k2}`] = v2;
      if (!Object.keys(v as object).length) out[k] = {};
    } else out[k] = v;
  }
  return out;
}

function diff(prev: Record<string, unknown> | null, next: Record<string, unknown>): Change[] {
  if (!prev) return [];
  const a = flatten(prev);
  const b = flatten(next);
  const keys = [...new Set([...Object.keys(a), ...Object.keys(b)])].sort();
  return keys.filter((k) => JSON.stringify(a[k]) !== JSON.stringify(b[k])).map((k) => ({ path: k, from: a[k], to: b[k] }));
}

function show(v: unknown): string {
  if (v === undefined || v === null || v === "") return "—";
  if (Array.isArray(v)) return v.length ? v.join(", ") : "—";
  if (typeof v === "object") return JSON.stringify(v);
  const s = String(v);
  return s.length > 80 ? `${s.slice(0, 77)}…` : s;
}

function pathLabel(path: string) {
  const [head, ...rest] = path.split(".");
  return `${LABEL[head] ?? head}${rest.length ? ` › ${rest.join(".").replace(/_/g, " ")}` : ""}`;
}

export function KBHistoryTab({ kbId }: { kbId: string }) {
  const q = useApi<KBRevision[]>(`/knowledge-bases/${kbId}/revisions`);
  const members = useMemberMap();
  const [open, setOpen] = useState<number | null>(null);
  const rows = useMemo(() => {
    const list = q.data ?? [];
    return list.map((r, i) => ({ ...r, changes: diff(list[i + 1]?.config ?? null, r.config), first: i === list.length - 1 && r.version === 1 }));
  }, [q.data]);

  if (q.isLoading)
    return (
      <div className="space-y-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-16 rounded-xl" />
        ))}
      </div>
    );
  if (!rows.length)
    return (
      <div className="card">
        <EmptyState icon={<History className="h-5 w-5" />} title="No configuration history" description="Every change to this knowledge base's settings is recorded here." />
      </div>
    );

  return (
    <div className="card overflow-hidden">
      <div className="flex items-end justify-between gap-4 border-b px-5 py-4">
        <div>
          <h3 className="text-[14px] font-semibold tracking-[-0.01em]">Configuration history</h3>
          <p className="mt-0.5 text-[13px] text-muted">Each save creates a new version with a full snapshot (latest 100).</p>
        </div>
        <span className="num hidden font-mono text-[11.5px] text-subtle sm:inline">{pluralize(rows.length, "version")}</span>
      </div>
      <ol className="divide-y">
        {rows.map((r) => {
          const m = r.changed_by ? members.get(r.changed_by) : undefined;
          const expanded = open === r.version;
          return (
            <li key={r.version}>
              <button type="button" onClick={() => setOpen(expanded ? null : r.version)} aria-expanded={expanded} className="flex w-full items-start gap-3.5 px-5 py-3.5 text-left transition-colors hover:bg-surface-2/50">
                <span className="num mt-px grid h-6 min-w-[34px] shrink-0 place-items-center rounded-md border bg-gradient-to-b from-surface to-surface-2/80 px-1.5 font-mono text-[11px] font-semibold text-fg/80 shadow-xs">v{r.version}</span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[13px]">
                    <Avatar name={m?.full_name} email={m?.email} className="h-5 w-5 text-[9px]" />
                    <span className="font-medium">{memberLabel(members, r.changed_by, "System")}</span>
                    <span className="text-muted">{r.first ? "created the knowledge base" : r.changes.length ? `changed ${r.changes.length} setting${r.changes.length === 1 ? "" : "s"}` : "saved without changes"}</span>
                    <span className="font-mono text-[11px] text-subtle" title={formatDate(r.created_at)}>
                      {timeAgo(r.created_at)}
                    </span>
                  </div>
                  {r.changes.length ? (
                    <ul className="mt-2 space-y-1">
                      {r.changes.slice(0, expanded ? undefined : 4).map((c) => (
                        <li key={c.path} className="flex flex-wrap items-baseline gap-x-1.5 text-[12px]">
                          <span className="font-medium text-muted">{pathLabel(c.path)}</span>
                          <span className="font-mono text-[11.5px] text-subtle line-through decoration-subtle/60">{show(c.from)}</span>
                          <span className="text-subtle">→</span>
                          <span className="font-mono text-[11.5px] text-fg">{show(c.to)}</span>
                        </li>
                      ))}
                      {!expanded && r.changes.length > 4 ? <li className="text-[11.5px] text-subtle">+{r.changes.length - 4} more</li> : null}
                    </ul>
                  ) : null}
                </div>
                {r.version === rows[0]?.version ? <Badge tone="accent">current</Badge> : null}
                <ChevronDown className={cn("mt-0.5 h-4 w-4 shrink-0 text-subtle transition-transform", expanded && "rotate-180")} />
              </button>
              {expanded ? (
                <div className="border-t bg-surface-2/40 px-5 py-4">
                  <div className="label mb-2">Snapshot · {formatDate(r.created_at)}</div>
                  <pre className="max-h-80 overflow-auto rounded-lg border bg-surface p-3 font-mono text-[11.5px] leading-relaxed">{JSON.stringify(r.config, null, 2)}</pre>
                </div>
              ) : null}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
