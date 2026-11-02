"use client";

import { ChevronLeft, ChevronRight, ScrollText } from "lucide-react";
import { useState } from "react";

import { type Column, DataTable } from "@/components/ui/data-table";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Select } from "@/components/ui/form";
import { EmptyState, StatusBadge, Tip } from "@/components/ui/primitives";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { formatDate, formatNumber, timeAgo } from "@/lib/utils";

import { CodeBlock, KV, Notice, formatMs, useMemberNames } from "./shared";
import type { Paged, QueryLog } from "./types";

const STATUSES = ["executed", "rejected", "pending_approval", "approved", "denied", "failed", "cannot_answer"];
const PAGE = 50;

export function QueryLogTable({ sourceId }: { sourceId: string }) {
  const { can } = useAuth();
  const [status, setStatus] = useState("");
  const [offset, setOffset] = useState(0);
  const [open, setOpen] = useState<QueryLog | null>(null);
  const members = useMemberNames();
  const logs = useApi<Paged<QueryLog>>("/query-logs", { source_id: sourceId, status: status || undefined, limit: PAGE, offset }, { placeholderData: (prev) => prev });
  const total = logs.data?.total ?? 0;

  const columns: Column<QueryLog>[] = [
    { key: "status", header: "Status", cell: (r) => <StatusBadge status={r.status} /> },
    {
      key: "query",
      header: "Question / SQL",
      cell: (r) => (
        <div className="min-w-[220px] max-w-[460px]">
          {r.question ? <div className="truncate font-medium tracking-[-0.005em]">{r.question}</div> : null}
          <div className={r.question ? "mt-0.5 truncate font-mono text-[11.5px] tracking-[-0.01em] text-muted" : "truncate font-mono text-[12px] tracking-[-0.01em]"}>{r.executed_sql || r.generated_sql || r.error || "—"}</div>
        </div>
      ),
    },
    { key: "user", header: "User", cell: (r) => <span className="whitespace-nowrap text-muted">{members.name(r.user_id)}</span> },
    { key: "rows", header: "Rows", cell: (r) => <span className="num font-mono text-[12px]">{r.row_count === null ? "—" : formatNumber(r.row_count)}</span>, className: "text-right", headerClassName: "text-right" },
    { key: "ms", header: "Time", cell: (r) => <span className="num whitespace-nowrap font-mono text-[12px] text-muted">{formatMs(r.duration_ms)}</span>, className: "text-right", headerClassName: "text-right" },
    {
      key: "when",
      header: "When",
      cell: (r) => (
        <Tip content={formatDate(r.created_at)}>
          <span className="whitespace-nowrap font-mono text-[12px] text-muted">{timeAgo(r.created_at)}</span>
        </Tip>
      ),
    },
  ];

  return (
    <div className="space-y-3">
      {!(can("db.edit") || can("admin.audit")) ? <Notice tone="info">You are seeing your own queries only. People with db.edit or admin.audit see everyone&apos;s.</Notice> : null}
      <DataTable
        rows={logs.data?.items}
        columns={columns}
        rowKey={(r) => r.id}
        loading={logs.isLoading}
        searchable={false}
        pageSize={PAGE}
        dense
        onRowClick={setOpen}
        toolbar={
          <div className="flex items-center gap-2">
            <label htmlFor="ql-status" className="label">
              Status
            </label>
            <Select
              id="ql-status"
              value={status}
              onChange={(e) => {
                setStatus(e.target.value);
                setOffset(0);
              }}
              className="h-8 w-44 text-[13px]"
            >
              <option value="">All statuses</option>
              {STATUSES.map((s) => (
                <option key={s} value={s}>
                  {s.replace(/_/g, " ")}
                </option>
              ))}
            </Select>
          </div>
        }
        empty={<EmptyState icon={<ScrollText className="h-5 w-5" />} title={status ? "No queries with this status" : "No queries yet"} description="Every live query — executed, rejected or waiting for approval — is logged here." />}
      />
      {total > PAGE ? (
        <div className="flex items-center justify-end gap-2 text-xs text-muted">
          <span className="num">
            {offset + 1}–{Math.min(total, offset + PAGE)} of {formatNumber(total)}
          </span>
          <button className="rounded p-1 hover:bg-surface-2 disabled:opacity-40" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))} aria-label="Previous page">
            <ChevronLeft className="h-4 w-4" />
          </button>
          <button className="rounded p-1 hover:bg-surface-2 disabled:opacity-40" disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)} aria-label="Next page">
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      ) : null}

      <Dialog open={Boolean(open)} onOpenChange={(v) => !v && setOpen(null)}>
        {open ? (
          <DialogContent wide title="Query details" description={formatDate(open.created_at)}>
            <div className="space-y-4">
              <dl className="grid grid-cols-2 gap-4 rounded-lg border bg-surface-2/40 p-3.5 sm:grid-cols-4">
                <KV label="Status">
                  <StatusBadge status={open.status} />
                </KV>
                <KV label="User">{members.name(open.user_id)}</KV>
                <KV label="Rows">{open.row_count === null ? "—" : formatNumber(open.row_count)}</KV>
                <KV label="Duration">{formatMs(open.duration_ms)}</KV>
              </dl>
              {open.question ? (
                <div>
                  <div className="label mb-1.5">Question</div>
                  <p className="text-[14px] leading-relaxed">{open.question}</p>
                </div>
              ) : null}
              {open.generated_sql && open.generated_sql !== open.executed_sql ? (
                <CodeBlock sql copy label="Submitted / generated SQL">
                  {open.generated_sql}
                </CodeBlock>
              ) : null}
              {open.executed_sql ? (
                <CodeBlock sql copy label="Validated SQL">
                  {open.executed_sql}
                </CodeBlock>
              ) : null}
              {open.reasons?.length ? (
                <Notice tone="warn" title="Approval reasons">
                  {open.reasons.join("; ")}
                </Notice>
              ) : null}
              {open.error ? (
                <Notice tone="danger" title="Error">
                  {open.error}
                </Notice>
              ) : null}
            </div>
          </DialogContent>
        ) : null}
      </Dialog>
    </div>
  );
}
