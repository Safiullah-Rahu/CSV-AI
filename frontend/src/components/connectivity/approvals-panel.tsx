"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Bot, Check, Code2, Gavel, Undo2, X } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent } from "@/components/ui/dialog";
import { Field, Textarea } from "@/components/ui/form";
import { Badge, Card, Dot, EmptyState, SkeletonRows, StatusBadge, Tabs, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";

import { OutcomeCard } from "./query-console";
import { CodeBlock, GatedButton, Notice, When, formatJson, invalidatePrefix, useMemberNames } from "./shared";
import type { AgentRunResult, Approval, QueryOutcome } from "./types";

type Decision = { approval: Approval; approve: boolean };

/**
 * Approval requests (`GET /approvals`). People with `db.approve` see and decide
 * everyone's requests; others see only their own. Optionally filtered to one
 * resource (a connection or an agent) client-side.
 */
export function ApprovalsPanel({ sourceId, title }: { sourceId?: string; title?: string }) {
  const { can, user } = useAuth();
  const qc = useQueryClient();
  const [status, setStatus] = useState<"pending" | "approved" | "denied" | "withdrawn">("pending");
  const [decision, setDecision] = useState<Decision | null>(null);
  const [result, setResult] = useState<{ approval: Approval; result: QueryOutcome } | null>(null);
  const members = useMemberNames();
  const approvals = useApi<Approval[]>("/approvals", { status }, { refetchInterval: status === "pending" ? 20_000 : false });
  const rows = (approvals.data ?? []).filter((a) => !sourceId || a.payload.source_id === sourceId);
  const approver = can("db.approve");
  const withdraw = useMutation({
    mutationFn: (a: Approval) => post<Approval>(`/approvals/${a.id}/withdraw`, {}),
    onSuccess: () => {
      toast.success("Request withdrawn");
      invalidatePrefix(qc, "/approvals", "/query-logs");
    },
  });

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Tabs value={status} onValueChange={(v) => setStatus(v as typeof status)}>
          <TabsList>
            <TabsTrigger value="pending">Pending</TabsTrigger>
            <TabsTrigger value="approved">Approved</TabsTrigger>
            <TabsTrigger value="denied">Denied</TabsTrigger>
            <TabsTrigger value="withdrawn">Withdrawn</TabsTrigger>
          </TabsList>
        </Tabs>
        <p className="max-w-xl text-[12.5px] leading-relaxed text-muted">{approver ? "You can approve requests from other people. Approved SQL and agent tools run with the requester's permissions." : "You see your own requests. Someone with db.approve decides them; you can withdraw a request until then."}</p>
      </div>

      {approvals.isLoading ? (
        <Card>
          <SkeletonRows rows={3} />
        </Card>
      ) : !rows.length ? (
        <Card>
          <EmptyState
            icon={<Gavel className="h-5 w-5" />}
            title={status === "pending" ? "Nothing waiting for approval" : `No ${status} requests`}
            description={status === "pending" ? (title ? `Queries on ${title} that touch tables marked “requires approval”, or exceed the join limit, appear here.` : "Queries on tables marked “requires approval”, queries over the join limit, and agent tools that need sign-off appear here.") : undefined}
          />
        </Card>
      ) : (
        <ul className="stagger space-y-3">
          {rows.map((a) => {
            const own = a.requested_by === user?.id;
            return (
              <li key={a.id}>
                <Card className="overflow-hidden">
                  <div className="flex flex-wrap items-start gap-3.5 px-5 py-4">
                    <div className="relative grid h-9 w-9 shrink-0 place-items-center rounded-lg border bg-surface-2 text-muted shadow-[inset_0_1px_0_hsl(var(--highlight))]">
                      {a.kind === "agent_tool" ? <Bot className="h-4 w-4" /> : <Code2 className="h-4 w-4" />}
                      {a.status === "pending" ? <Dot tone="warn" live className="absolute -right-0.5 -top-0.5" /> : null}
                    </div>
                    <div className="min-w-0 flex-1 basis-64">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-[14px] font-medium tracking-[-0.01em]">{a.summary}</span>
                        <StatusBadge status={a.status} />
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-[12px] text-muted">
                        <Badge className="px-1.5 py-[1px] text-[10.5px]">{a.kind === "agent_tool" ? "Agent tool" : a.kind === "sql_query" ? "SQL query" : a.kind}</Badge>
                        {own ? <Badge tone="accent" className="px-1.5 py-[1px] text-[10.5px]">your request</Badge> : null}
                        <span>
                          Requested by <span className="text-fg">{members.name(a.requested_by)}</span>
                        </span>
                        <span className="text-subtle" aria-hidden>·</span>
                        <When at={a.created_at} className="text-[11.5px]" />
                        {a.decided_by ? (
                          <>
                            <span className="text-subtle" aria-hidden>·</span>
                            <span>
                              decided by <span className="text-fg">{members.name(a.decided_by)}</span>
                            </span>
                          </>
                        ) : null}
                      </div>
                    </div>
                    {a.status === "pending" && own ? (
                      <Button size="sm" variant="secondary" loading={withdraw.isPending && withdraw.variables?.id === a.id} onClick={() => withdraw.mutate(a)}>
                        <Undo2 className="h-3.5 w-3.5" /> Withdraw
                      </Button>
                    ) : null}
                    {a.status === "pending" && approver ? (
                      <div className="flex shrink-0 items-center gap-2">
                        <GatedButton size="sm" variant="secondary" allowed={!own} reason="You cannot decide your own request" onClick={() => setDecision({ approval: a, approve: false })}>
                          <X className="h-3.5 w-3.5" /> Deny
                        </GatedButton>
                        <GatedButton size="sm" allowed={!own} reason="You cannot approve your own request" onClick={() => setDecision({ approval: a, approve: true })}>
                          <Check className="h-3.5 w-3.5" /> Approve
                        </GatedButton>
                      </div>
                    ) : null}
                  </div>
                  <div className="space-y-3 border-t bg-surface-2/30 px-5 py-4">
                    {a.payload.question || a.payload.reasons?.length || (a.kind === "agent_tool" && a.payload.tool) || a.reason ? (
                      <dl className="grid gap-x-4 gap-y-2 text-[13px] sm:grid-cols-[110px_minmax(0,1fr)]">
                        {a.payload.question ? (
                          <>
                            <dt className="label pt-[3px]">Question</dt>
                            <dd className="min-w-0">{a.payload.question}</dd>
                          </>
                        ) : null}
                        {a.payload.reasons?.length ? (
                          <>
                            <dt className="label pt-[3px]">Why</dt>
                            <dd className="min-w-0 text-warn">{a.payload.reasons.join("; ")}</dd>
                          </>
                        ) : null}
                        {a.kind === "agent_tool" ? (
                          <>
                            <dt className="label pt-[3px]">Tool</dt>
                            <dd className="min-w-0 font-mono text-[12.5px]">{a.payload.tool}</dd>
                          </>
                        ) : null}
                        {a.reason ? (
                          <>
                            <dt className="label pt-[3px]">Decision note</dt>
                            <dd className="min-w-0">{a.reason}</dd>
                          </>
                        ) : null}
                      </dl>
                    ) : null}
                    {a.payload.sql ? (
                      <CodeBlock sql copy label="SQL" className="max-h-48">
                        {a.payload.sql}
                      </CodeBlock>
                    ) : null}
                    {a.kind === "agent_tool" ? (
                      <>
                        {a.payload.input !== undefined ? (
                          <CodeBlock label="Tool input" className="max-h-40">
                            {formatJson(a.payload.input)}
                          </CodeBlock>
                        ) : null}
                        <AgentOutcome approval={a} />
                      </>
                    ) : null}
                    {a.kind === "sql_query" && a.result ? (
                      <Button size="xs" variant="secondary" onClick={() => setResult({ approval: a, result: a.result! })}>
                        View result · {a.result.row_count} rows
                      </Button>
                    ) : a.status === "approved" && a.kind === "sql_query" ? (
                      <p className="text-[12px] text-subtle">The result is visible to the requester and to approvers.</p>
                    ) : null}
                  </div>
                </Card>
              </li>
            );
          })}
        </ul>
      )}

      <DecideDialog decision={decision} onClose={() => setDecision(null)} onResult={(approval, r) => setResult({ approval, result: r })} />

      <Dialog open={Boolean(result)} onOpenChange={(v) => !v && setResult(null)}>
        {result ? (
          <DialogContent wide title="Approved query result" description={result.approval.summary} className="max-w-5xl">
            <OutcomeCard outcome={result.result} title="Result" />
          </DialogContent>
        ) : null}
      </Dialog>
    </div>
  );
}

function DecideDialog({ decision, onClose, onResult }: { decision: Decision | null; onClose: () => void; onResult: (a: Approval, r: QueryOutcome) => void }) {
  const qc = useQueryClient();
  const [reason, setReason] = useState("");
  const decide = useMutation({
    mutationFn: (d: Decision) => post<{ status: string; result: QueryOutcome | null }>(`/approvals/${d.approval.id}/decide`, { approve: d.approve, reason: reason.trim() }),
    onSuccess: (r, d) => {
      toast.success(d.approve ? "Approved" : "Denied", { description: d.approve && r.result ? `The query ran: ${r.result.status}, ${r.result.row_count} rows.` : d.approve && d.approval.kind === "agent_tool" ? "The agent run continues in the background." : undefined });
      invalidatePrefix(qc, "/approvals", "/query-logs");
      setReason("");
      onClose();
      if (r.result) onResult(d.approval, r.result);
    },
  });
  const d = decision;
  return (
    <Dialog
      open={Boolean(d)}
      onOpenChange={(v) => {
        if (!v) {
          setReason("");
          onClose();
        }
      }}
    >
      {d ? (
        <DialogContent
          title={d.approve ? "Approve request" : "Deny request"}
          description={d.approval.summary}
          footer={
            <>
              <DialogClose asChild>
                <Button variant="secondary">Cancel</Button>
              </DialogClose>
              <Button variant={d.approve ? "primary" : "danger"} loading={decide.isPending} onClick={() => decide.mutate(d)}>
                {d.approve ? (d.approval.kind === "sql_query" ? "Approve and run" : "Approve and continue") : "Deny"}
              </Button>
            </>
          }
        >
          <div className="space-y-3">
            {d.approve && d.approval.kind === "sql_query" ? (
              <Notice tone="info">The SQL is re-validated and executed now with the requester&apos;s permissions. The result is returned to you and stored for the requester.</Notice>
            ) : null}
            {d.approve && d.approval.kind === "agent_tool" ? (
              <Notice tone="info">The agent runs this tool and finishes its answer in the background, with the requester&apos;s permissions. The requester is notified when it is done.</Notice>
            ) : null}
            {d.approval.payload.sql ? (
              <CodeBlock sql label="SQL" className="max-h-40">
                {d.approval.payload.sql}
              </CodeBlock>
            ) : null}
            <Field label="Note (optional)" htmlFor="decision-reason" hint="Recorded with the decision and shown to the requester.">
              <Textarea id="decision-reason" value={reason} onChange={(e) => setReason(e.target.value)} maxLength={2000} />
            </Field>
          </div>
        </DialogContent>
      ) : null}
    </Dialog>
  );
}

/** What happened after an agent-tool decision. */
function AgentOutcome({ approval: a }: { approval: Approval }) {
  const r = a.result as unknown as (Partial<AgentRunResult> & { status?: string }) | null;
  if (a.status === "pending") return <p className="text-[12px] text-subtle">Approving runs this tool and lets the agent finish its answer, with the requester&apos;s permissions.</p>;
  if (a.status !== "approved") return null;
  if (!r || r.status === "resuming")
    return (
      <p className="inline-flex items-center gap-2 text-[12px] text-muted">
        <Dot tone="info" live /> Approved: the agent is finishing the run.
      </p>
    );
  return (
    <div className="rounded-lg border bg-surface px-3.5 py-3">
      <div className="mb-1.5 flex items-center gap-2">
        <span className="label">Resumed run</span>
        <StatusBadge status={r.status === "complete" ? "complete" : r.status === "error" ? "error" : r.status} />
      </div>
      <p className="line-clamp-4 whitespace-pre-line text-[13px] leading-relaxed">{r.answer || "No answer was produced."}</p>
    </div>
  );
}

/** Pending count for a tab label. */
export function usePendingApprovals(sourceId?: string) {
  const { can } = useAuth();
  const q = useApi<Approval[]>(can("org.view") ? "/approvals" : null, { status: "pending" }, { refetchInterval: 30_000, meta: { silent: true } });
  return (q.data ?? []).filter((a) => !sourceId || a.payload.source_id === sourceId).length;
}
