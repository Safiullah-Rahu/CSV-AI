"use client";

import { History } from "lucide-react";

import { type Column, DataTable } from "@/components/ui/data-table";
import { EmptyState, StatusBadge, Tip } from "@/components/ui/primitives";
import { useApi } from "@/lib/hooks";
import { formatDate, formatNumber, timeAgo } from "@/lib/utils";

import { runDuration } from "./shared";
import type { SyncRun } from "./types";

/** Sync run history for a database connection or an integration (`GET …/runs`). */
export function SyncRunsTable({ path, showFailed, emptyHint }: { path: string; showFailed?: boolean; emptyHint?: string }) {
  const runs = useApi<SyncRun[]>(path, undefined, {
    refetchInterval: (q) => ((q.state.data as SyncRun[] | undefined)?.some((r) => r.status === "running" || r.status === "queued") ? 4000 : 30_000),
  });
  const n = (v: number | undefined | null) => <span className="num font-mono text-[12px]">{formatNumber(v ?? 0)}</span>;
  const columns: Column<SyncRun>[] = [
    { key: "status", header: "Status", cell: (r) => <StatusBadge status={r.status} />, sortValue: (r) => r.status },
    { key: "mode", header: "Mode", cell: (r) => <span className="capitalize text-muted">{r.mode}</span> },
    {
      key: "started",
      header: "Started",
      cell: (r) => (
        <Tip content={formatDate(r.started_at)}>
          <span className="whitespace-nowrap font-mono text-[12px]">{timeAgo(r.started_at)}</span>
        </Tip>
      ),
      sortValue: (r) => r.started_at,
    },
    { key: "duration", header: "Duration", cell: (r) => <span className="num whitespace-nowrap font-mono text-[12px] text-muted">{r.finished_at ? runDuration(r.started_at, r.finished_at) : r.status === "running" ? "running…" : "—"}</span> },
    { key: "seen", header: "Seen", cell: (r) => n(r.items_seen), className: "text-right", headerClassName: "text-right" },
    { key: "added", header: "Added", cell: (r) => <span className={r.items_added ? "text-ok" : "text-subtle"}>+{n(r.items_added)}</span>, className: "text-right", headerClassName: "text-right" },
    { key: "updated", header: "Updated", cell: (r) => <span className={r.items_updated ? "text-info" : "text-subtle"}>~{n(r.items_updated)}</span>, className: "text-right", headerClassName: "text-right" },
    { key: "deleted", header: "Deleted", cell: (r) => <span className={r.items_deleted ? "text-warn" : "text-subtle"}>−{n(r.items_deleted)}</span>, className: "text-right", headerClassName: "text-right" },
    ...(showFailed ? [{ key: "failed", header: "Failed", cell: (r: SyncRun) => <span className={r.items_failed ? "text-danger" : "text-subtle"}>{n(r.items_failed)}</span>, className: "text-right", headerClassName: "text-right" }] : []),
    {
      key: "error",
      header: "Error",
      cell: (r) =>
        r.error ? (
          <Tip content={<span className="whitespace-pre-wrap break-words font-mono text-[11px]">{r.error}</span>}>
            <span className="block max-w-[280px] truncate font-mono text-[11.5px] text-danger" tabIndex={0}>
              {r.error}
            </span>
          </Tip>
        ) : (
          <span className="text-subtle">—</span>
        ),
    },
  ];
  return (
    <DataTable
      rows={runs.data}
      columns={columns}
      rowKey={(r) => r.id}
      loading={runs.isLoading}
      searchable={false}
      dense
      empty={<EmptyState icon={<History className="h-5 w-5" />} title="No sync runs yet" description={emptyHint ?? "Runs appear here after the first manual or scheduled sync."} />}
    />
  );
}
