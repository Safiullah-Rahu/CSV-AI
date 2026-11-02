"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { BookOpen, ChevronRight, CornerDownLeft, Gavel, ListTree, Lock, Send, Sparkles, Undo2, User, Wrench } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Textarea } from "@/components/ui/form";
import { Button } from "@/components/ui/button";
import { Badge, Card, CardBody, CardHeader, Dot, Kbd, StatusBadge } from "@/components/ui/primitives";
import { ApiError, errorMessage, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatNumber, timeAgo } from "@/lib/utils";

import { ResultChart } from "./results";
import { CodeBlock, GatedButton, Notice, formatJson, formatMs } from "./shared";
import type { Agent, AgentRunResult, AgentStep, Approval, Citation } from "./types";

type RunEntry = { id: number; question: string; at: string; result: AgentRunResult };

const STATUS_LABEL: Record<string, string> = { complete: "complete", pending_approval: "pending_approval", max_steps: "stopped", error: "error" };

export function AgentRunPanel({ agent }: { agent: Agent }) {
  const { can } = useAuth();
  const [question, setQuestion] = useState("");
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const [shown, setShown] = useState<number | null>(null);
  const [noModel, setNoModel] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  const run = useMutation({
    mutationFn: (q: string) => post<AgentRunResult>(`/agents/${agent.id}/run`, { question: q }),
    meta: { silent: true },
    onMutate: () => {
      setFailure(null);
    },
    onSuccess: (result, q) => {
      setNoModel(null);
      const entry = { id: Date.now(), question: q, at: new Date().toISOString(), result };
      setRuns((r) => [entry, ...r].slice(0, 10));
      setShown(entry.id);
      setQuestion("");
    },
    onError: (err) => {
      if (err instanceof ApiError && err.code === "not_available") {
        setNoModel(err.message);
        return;
      }
      setFailure(errorMessage(err));
    },
  });

  const blocked = !can("agents.use") ? "Requires agents.use" : !agent.enabled ? "This agent is disabled" : "";
  const submit = () => {
    if (!question.trim() || blocked || run.isPending) return;
    run.mutate(question.trim());
  };
  const current = runs.find((r) => r.id === shown) ?? runs[0];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader title="Run" description="The agent answers with its tools, using your permissions." />
        <CardBody className="space-y-3 p-4 sm:p-5">
          <div className="overflow-hidden rounded-xl border bg-surface shadow-xs transition-[border-color,box-shadow] focus-within:border-accent/60 focus-within:ring-[3px] focus-within:ring-accent/15">
            <Textarea
              aria-label={`Ask ${agent.name}`}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
              }}
              placeholder={agent.description ? `Ask ${agent.name}… (${agent.description})` : `Ask ${agent.name}…`}
              maxLength={8000}
              className="min-h-[96px] resize-y rounded-none border-0 bg-transparent px-3.5 py-3 text-[14px] shadow-none hover:border-0 focus:border-0 focus:ring-0"
              disabled={Boolean(blocked)}
            />
            <div className="flex flex-wrap items-center justify-end gap-3 border-t bg-surface-2/50 px-3 py-2">
              <span className="mr-auto min-w-0 truncate text-[12px] text-muted">
                {run.isPending ? (
                  <span className="inline-flex items-center gap-2">
                    <Dot tone="accent" live /> Running — each step calls the model and a tool, so this can take a while.
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5">
                    <Wrench className="h-3 w-3 text-subtle" />
                    <span className="font-mono text-[11px]">{agent.tools.join(" · ")}</span>
                  </span>
                )}
              </span>
              <span className="hidden items-center gap-1 text-2xs text-subtle sm:flex">
                <Kbd>Ctrl</Kbd>
                <Kbd>
                  <CornerDownLeft className="inline h-2.5 w-2.5" />
                </Kbd>
              </span>
              <GatedButton size="sm" allowed={!blocked} reason={blocked} onClick={submit} disabled={!question.trim()} loading={run.isPending}>
                <Send className="h-3.5 w-3.5" /> Run agent
              </GatedButton>
            </div>
          </div>
        </CardBody>
      </Card>

      {noModel ? (
        <Notice
          tone="warn"
          title="No language model configured"
          action={
            <Link href="/settings?tab=ai" className="whitespace-nowrap text-xs font-medium text-accent hover:underline">
              Open AI settings
            </Link>
          }
        >
          {noModel} {can("admin.ai_providers") ? "Connect a chat model under Settings → AI." : "Ask an administrator to connect a chat model under Settings → AI."}
        </Notice>
      ) : null}
      {failure ? <Notice tone="danger" title="Run failed">{failure}</Notice> : null}

      {runs.length > 1 ? (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="label mr-1">This session</span>
          {runs.map((r) => (
            <button
              key={r.id}
              onClick={() => setShown(r.id)}
              className={cn("inline-flex max-w-[240px] items-center gap-1.5 truncate rounded-full border px-2.5 py-1 text-[12px] transition-colors", current?.id === r.id ? "border-accent/40 bg-accent/[0.06] text-fg" : "bg-surface text-muted hover:border-border-strong hover:text-fg")}
              title={r.question}
            >
              <Dot tone={r.result.status === "complete" ? "ok" : r.result.status === "error" ? "danger" : "warn"} />
              <span className="truncate">{r.question}</span>
            </button>
          ))}
        </div>
      ) : null}

      {current ? <RunResult entry={current} /> : null}
    </div>
  );
}

/**
 * Follows the approval a paused run is waiting on: pending (withdrawable), approved and
 * finishing, the resumed answer, or denied / withdrawn / expired.
 */
function ApprovalTracker({ approvalId, entry }: { approvalId: string; entry: RunEntry }) {
  const qc = useQueryClient();
  const q = useApi<Approval>(`/approvals/${approvalId}`, undefined, {
    refetchInterval: (query) => {
      const d = query.state.data;
      const resuming = d?.status === "approved" && (!d.result || (d.result as unknown as { status?: string }).status === "resuming");
      return !d || d.status === "pending" || resuming ? 4000 : false;
    },
    meta: { silent: true },
  });
  const withdraw = useMutation({
    mutationFn: () => post<Approval>(`/approvals/${approvalId}/withdraw`, {}),
    onSuccess: () => qc.invalidateQueries({ queryKey: [`/approvals/${approvalId}`] }),
  });
  const a = q.data;
  const resumed = a?.status === "approved" ? (a.result as unknown as AgentRunResult | { status: "resuming" } | null) : null;
  if (resumed && resumed.status !== "resuming" && "answer" in resumed) {
    return (
      <div className="space-y-3">
        <Notice tone="info" title="Approved and resumed">
          The approved tool ran with your permissions and the agent finished below.
        </Notice>
        <RunResult entry={{ ...entry, id: entry.id + 1, result: resumed }} />
      </div>
    );
  }
  if (a && a.status !== "pending" && a.status !== "approved") {
    return (
      <Notice tone={a.status === "withdrawn" ? "info" : "warn"} title={a.status === "withdrawn" ? "Request withdrawn" : a.status === "expired" ? "Request expired" : "Request denied"}>
        {a.reason || "The tool did not run, so the agent stopped here."}
      </Notice>
    );
  }
  return (
    <Notice
      tone="warn"
      title={a?.status === "approved" ? "Approved: finishing the run" : "Waiting for approval"}
      action={
        a?.status === "pending" ? (
          <div className="flex items-center gap-3">
            <Button size="xs" variant="secondary" loading={withdraw.isPending} onClick={() => withdraw.mutate()}>
              <Undo2 /> Withdraw
            </Button>
            <Link href="/database-connections?tab=approvals" className="inline-flex items-center gap-1 whitespace-nowrap text-xs font-medium text-accent hover:underline">
              <Gavel className="h-3.5 w-3.5" /> Approvals
            </Link>
          </div>
        ) : undefined
      }
    >
      {a?.status === "approved" ? (
        <span className="inline-flex items-center gap-2">
          <Dot tone="info" live /> The tool is running with your permissions; the answer appears here when it is done.
        </span>
      ) : (
        <>A tool in this run needs sign-off. The run continues automatically once it is approved, and you will be notified.</>
      )}
    </Notice>
  );
}

function RunResult({ entry }: { entry: RunEntry }) {
  const r = entry.result;
  const cited = r.citations.filter((c) => c.cited);
  const others = r.citations.filter((c) => !c.cited);
  const toolMs = r.steps.reduce((a, s) => a + (s.ms ?? 0), 0);
  return (
    <div className="space-y-4 animate-fade-up">
      <Card>
        <CardHeader
          title={
            <span className="flex flex-wrap items-center gap-2">
              <Sparkles className="h-4 w-4 text-accent" /> Answer <StatusBadge status={STATUS_LABEL[r.status] ?? r.status} />
            </span>
          }
          description={
            <span className="line-clamp-2">
              “{entry.question}” · <span className="font-mono text-[11.5px]">{timeAgo(entry.at)}</span>
            </span>
          }
          action={
            <span className="num hidden font-mono text-[11px] text-subtle sm:block">
              {formatNumber(r.usage.prompt_tokens)} in · {formatNumber(r.usage.completion_tokens)} out
            </span>
          }
        />
        <CardBody className="space-y-4">
          {r.status === "pending_approval" && r.approval_id ? <ApprovalTracker approvalId={r.approval_id} entry={entry} /> : null}
          {r.status === "max_steps" ? <Notice tone="warn" title="Step limit reached">The agent stopped before finishing. Ask a narrower question or raise Max steps.</Notice> : null}
          {r.answer ? (
            <div className="prose-chat">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{r.answer}</ReactMarkdown>
            </div>
          ) : (
            <p className="text-[13px] text-muted">No answer was produced.</p>
          )}
          {r.chart ? (
            <div className="rounded-lg border p-4 pb-3">
              <ResultChart chart={r.chart} />
            </div>
          ) : null}
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <ListTree className="h-4 w-4 text-subtle" /> Transcript
            </span>
          }
          description={r.steps.length ? `${r.steps.length} tool call${r.steps.length === 1 ? "" : "s"} · ${formatMs(toolMs)} in tools` : "The model answered without calling a tool."}
        />
        <ol className="relative px-5 py-4" aria-label="Run transcript">
          <TimelineNode icon={<User className="h-3 w-3" />} title="Question" meta={<span className="font-mono">{timeAgo(entry.at)}</span>}>
            <p className="text-[13px] leading-relaxed">{entry.question}</p>
          </TimelineNode>
          {r.steps.map((s) => (
            <StepRow key={s.step} s={s} />
          ))}
          <TimelineNode
            last
            tone={r.status === "complete" ? "accent" : r.status === "error" ? "danger" : "warn"}
            icon={<Sparkles className="h-3 w-3" />}
            title="Answer"
            meta={
              <span className="num font-mono">
                {formatNumber(r.usage.prompt_tokens + r.usage.completion_tokens)} tokens
              </span>
            }
          >
            <span className="inline-flex items-center gap-2 text-[12.5px] text-muted">
              <StatusBadge status={STATUS_LABEL[r.status] ?? r.status} />
              {r.status === "complete" ? `${cited.length} source${cited.length === 1 ? "" : "s"} cited` : r.status === "pending_approval" ? "Paused for approval" : r.status === "max_steps" ? "Stopped at the step limit" : null}
            </span>
          </TimelineNode>
        </ol>
      </Card>

      {r.citations.length ? (
        <Card>
          <CardHeader
            title={
              <span className="flex items-center gap-2">
                <BookOpen className="h-4 w-4 text-subtle" /> Sources
              </span>
            }
            description={`${cited.length} cited · ${others.length} retrieved but not cited`}
          />
          <ul className="divide-y">
            {[...cited, ...others].map((c) => (
              <CitationRow key={c.label} c={c} />
            ))}
          </ul>
        </Card>
      ) : null}
    </div>
  );
}

const NODE_TONE = {
  neutral: "border-border-strong bg-surface text-muted",
  accent: "border-accent/40 bg-accent/10 text-accent",
  warn: "border-warn/40 bg-warn/10 text-warn",
  danger: "border-danger/40 bg-danger/10 text-danger",
} as const;

/** One event in the transcript: a node on a vertical rail with a title row and optional body. */
function TimelineNode({ icon, title, meta, children, tone = "neutral", last }: { icon: React.ReactNode; title: React.ReactNode; meta?: React.ReactNode; children?: React.ReactNode; tone?: keyof typeof NODE_TONE; last?: boolean }) {
  return (
    <li className="relative pb-5 pl-9 last:pb-0">
      {!last ? <span className="absolute bottom-0 left-[11px] top-6 w-px bg-border" aria-hidden /> : null}
      <span className={cn("absolute left-0 top-0 grid h-6 w-6 place-items-center rounded-full border", NODE_TONE[tone])} aria-hidden>
        {icon}
      </span>
      <div className="flex min-h-6 flex-wrap items-center gap-x-2 gap-y-1">
        <span className="text-[13px] font-medium tracking-[-0.005em]">{title}</span>
        {meta ? <span className="ml-auto text-[11px] text-subtle">{meta}</span> : null}
      </div>
      {children ? <div className="mt-1.5 min-w-0">{children}</div> : null}
    </li>
  );
}

function CitationRow({ c }: { c: Citation }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="px-5 py-3">
      <button className="flex w-full items-start gap-3 text-left" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className={cn("num mt-0.5 grid h-5 min-w-[28px] place-items-center rounded px-1 font-mono text-[11px] font-semibold", c.cited ? "bg-accent/15 text-accent" : "bg-surface-2 text-subtle")}>{c.label}</span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-1.5">
            <span className="truncate text-[13px] font-medium">{c.title || c.filename}</span>
            <Badge>{c.source_type.replace(/_/g, " ")}</Badge>
            {c.flagged ? <Badge tone="warn">flagged</Badge> : null}
          </span>
          <span className="mt-0.5 block text-[12px] text-muted">
            {c.locator_text}
            {c.filename && c.filename !== c.title ? ` · ${c.filename}` : ""} · score <span className="num font-mono text-[11.5px]">{c.score.toFixed(3)}</span>
          </span>
        </span>
        <ChevronRight className={cn("mt-0.5 h-4 w-4 shrink-0 text-subtle transition-transform", open && "rotate-90")} />
      </button>
      {open ? <p className="mt-2 whitespace-pre-wrap rounded-lg border bg-surface-2/50 p-3 text-[12.5px] leading-relaxed text-muted">{c.snippet}</p> : null}
    </li>
  );
}

function StepRow({ s }: { s: AgentStep }) {
  const [open, setOpen] = useState(false);
  const pending = s.result === "pending approval";
  const failed = s.result?.startsWith("Tool error");
  return (
    <TimelineNode
      tone={pending ? "warn" : failed ? "danger" : "neutral"}
      icon={pending ? <Lock className="h-3 w-3" /> : <Wrench className="h-3 w-3" />}
      title={
        <span className="inline-flex items-center gap-2">
          <span className="font-mono text-[10.5px] text-subtle">#{s.step}</span>
          <span className="font-mono text-[12.5px]">{s.tool}</span>
          {pending ? <Badge tone="warn">pending approval</Badge> : failed ? <Badge tone="danger">error</Badge> : null}
        </span>
      }
      meta={<span className="num font-mono">{s.ms !== undefined ? formatMs(s.ms) : "—"}</span>}
    >
      <button className="group flex w-full min-w-0 items-center gap-2 rounded-md border bg-surface-2/50 px-2.5 py-1.5 text-left transition-colors hover:border-border-strong" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <ChevronRight className={cn("h-3.5 w-3.5 shrink-0 text-subtle transition-transform", open && "rotate-90")} />
        <span className="min-w-0 flex-1 truncate font-mono text-[11.5px] text-muted">{formatJson(s.input).replace(/\s+/g, " ")}</span>
      </button>
      {open ? (
        <div className="mt-2 grid gap-3 lg:grid-cols-2">
          <CodeBlock label="Input" className="max-h-56">
            {formatJson(s.input)}
          </CodeBlock>
          <CodeBlock label="Result" className="max-h-56">
            {s.result ?? "—"}
          </CodeBlock>
        </div>
      ) : null}
    </TimelineNode>
  );
}
