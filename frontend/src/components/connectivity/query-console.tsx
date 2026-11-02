"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Code2, CornerDownLeft, Lock, MessageSquareText, Play, Sparkles, Table2 } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/form";
import { Badge, Card, CardBody, CardHeader, Kbd, StatusBadge, Tabs, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { ApiError, errorMessage, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/utils";

import { ResultView } from "./results";
import { CodeBlock, GatedButton, Notice, formatMs, invalidatePrefix } from "./shared";
import type { DbConnection, QueryOutcome } from "./types";

type Mode = "question" | "sql";

export function QueryConsole({ connection: c }: { connection: DbConnection }) {
  const { can } = useAuth();
  const qc = useQueryClient();
  const [mode, setMode] = useState<Mode>("question");
  const [question, setQuestion] = useState("");
  const [sql, setSql] = useState("");
  const [llmMissing, setLlmMissing] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<QueryOutcome | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  const run = useMutation({
    mutationFn: (body: { question?: string; sql?: string }) => post<QueryOutcome>(`/database-connections/${c.id}/query`, body),
    meta: { silent: true },
    onMutate: () => setFailure(null),
    onSuccess: (r) => {
      setOutcome(r);
      invalidatePrefix(qc, "/query-logs", "/approvals");
      if (r.status === "pending_approval") toast.message("Query sent for approval", { description: "Someone with db.approve must approve it before it runs." });
    },
    onError: (err, body) => {
      if (err instanceof ApiError && err.code === "not_available" && body.question && !body.sql) {
        setLlmMissing(err.message);
        setMode("sql");
        return;
      }
      setFailure(errorMessage(err));
      toast.error(errorMessage(err));
    },
  });

  const blocked = !c.enabled ? "This connection is disabled." : c.access_mode === "ingest" ? "Live querying is disabled for this connection (ingest-only mode). Ask questions about synchronised data in AI Chat instead." : !c.selected_tables.length ? "No tables are selected on this connection yet." : null;
  const text = mode === "question" ? question : sql;
  const submit = () => {
    if (!text.trim() || blocked) return;
    run.mutate(mode === "question" ? { question: question.trim() } : { sql: sql.trim(), ...(question.trim() ? { question: question.trim() } : {}) });
  };

  return (
    <div className="space-y-4">
      {blocked ? <Notice tone="warn">{blocked}</Notice> : null}
      {llmMissing ? (
        <Notice
          tone="info"
          title="Natural-language questions need a language model"
          action={
            can("admin.ai_providers") ? (
              <Link href="/settings?tab=ai" className="text-xs font-medium text-accent hover:underline">
                Connect a model
              </Link>
            ) : null
          }
        >
          {llmMissing} {can("admin.ai_providers") ? "" : "Ask an administrator to connect an AI provider."} SQL mode works without one.
        </Notice>
      ) : null}

      <Card>
        <CardHeader
          title="Ask this database"
          description={
            <>
              Read-only. Every statement is parsed and validated against the selected tables, columns and row filters before it runs
              {c.query_limits?.max_rows ? <> · max <span className="num">{c.query_limits.max_rows}</span> rows</> : null}
              {c.query_limits?.timeout_s ? <> · <span className="num">{c.query_limits.timeout_s}</span>s timeout</> : null}.
            </>
          }
          action={
            <Tabs value={mode} onValueChange={(v) => setMode(v as Mode)}>
              <TabsList aria-label="Query mode">
                <TabsTrigger value="question">
                  <MessageSquareText className="h-3.5 w-3.5" /> Question
                </TabsTrigger>
                <TabsTrigger value="sql">
                  <Code2 className="h-3.5 w-3.5" /> SQL
                </TabsTrigger>
              </TabsList>
            </Tabs>
          }
          className="flex-wrap"
        />
        <CardBody className="space-y-3 p-4 sm:p-5">
          <div className={cn("overflow-hidden rounded-xl border bg-surface shadow-xs transition-[border-color,box-shadow] focus-within:border-accent/60 focus-within:ring-[3px] focus-within:ring-accent/15")}>
            <div className="flex h-8 items-center gap-2 border-b bg-surface-2/60 px-3">
              {mode === "sql" ? <Code2 className="h-3.5 w-3.5 text-subtle" /> : <Sparkles className="h-3.5 w-3.5 text-accent" />}
              <span className="label truncate">{mode === "sql" ? "SQL · read-only" : "Natural language → validated SQL"}</span>
              <span className="ml-auto hidden shrink-0 items-center gap-1 text-[11px] text-subtle sm:inline-flex">
                <Lock className="h-3 w-3" /> read-only transaction
              </span>
            </div>
            {mode === "question" ? (
              <Textarea
                aria-label="Question"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
                }}
                placeholder="e.g. Total revenue by month for 2024, highest first"
                className="min-h-[96px] resize-y rounded-none border-0 bg-transparent px-3.5 py-3 text-[14px] shadow-none hover:border-0 focus:border-0 focus:ring-0"
                maxLength={4000}
                disabled={Boolean(blocked)}
              />
            ) : (
              <Textarea
                aria-label="SQL"
                value={sql}
                onChange={(e) => setSql(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
                }}
                placeholder={`SELECT … FROM ${c.selected_tables[0] ?? "schema.table"} LIMIT 100`}
                className="min-h-[160px] resize-y rounded-none border-0 bg-transparent px-3.5 py-3 font-mono text-[12.5px] leading-[1.7] shadow-none hover:border-0 focus:border-0 focus:ring-0"
                spellCheck={false}
                maxLength={20000}
                disabled={Boolean(blocked)}
              />
            )}
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1.5 text-[11px] text-subtle">
              {c.selected_tables.length ? <span className="label mr-0.5">Insert</span> : null}
              {c.selected_tables.slice(0, 6).map((t) => (
                <button key={t} type="button" className="inline-flex h-6 items-center gap-1 rounded-md border bg-surface-2/60 px-1.5 font-mono text-[11px] text-muted transition-colors hover:border-border-strong hover:text-fg" onClick={() => (mode === "sql" ? setSql((s) => (s ? `${s} ${t}` : `SELECT * FROM ${t} LIMIT 50`)) : setQuestion((s) => `${s}${s ? " " : ""}${t}`))} title="Insert table name">
                  <Table2 className="h-3 w-3 text-subtle" /> {t}
                </button>
              ))}
              {c.selected_tables.length > 6 ? <span>+{c.selected_tables.length - 6} more</span> : null}
            </div>
            <span className="hidden items-center gap-1 text-2xs text-subtle sm:flex">
              <Kbd>Ctrl</Kbd>
              <Kbd>
                <CornerDownLeft className="inline h-2.5 w-2.5" />
              </Kbd>
            </span>
            <GatedButton allowed={can("db.query") && !blocked} reason={!can("db.query") ? "Requires db.query" : blocked ?? undefined} onClick={submit} disabled={!text.trim()} loading={run.isPending}>
              {mode === "question" ? <Sparkles className="h-4 w-4" /> : <Play className="h-4 w-4" />} {mode === "question" ? "Ask" : "Run SQL"}
            </GatedButton>
          </div>
        </CardBody>
      </Card>

      {failure ? <Notice tone="danger" title="Request failed">{failure}</Notice> : null}
      {outcome ? (
        <OutcomeCard
          outcome={outcome}
          onEditSql={(s) => {
            setSql(s);
            setMode("sql");
          }}
        />
      ) : null}
    </div>
  );
}

export function OutcomeCard({ outcome: o, onEditSql, title = "Result" }: { outcome: QueryOutcome; onEditSql?: (sql: string) => void; title?: string }) {
  const shownSql = o.sql || o.generated_sql;
  return (
    <Card className="animate-fade-up">
      <CardHeader
        title={
          <span className="flex flex-wrap items-center gap-2">
            {title} <StatusBadge status={o.status} />
            {o.status === "executed" ? (
              <span className="num font-mono text-[11.5px] font-normal text-muted">
                {o.row_count} row{o.row_count === 1 ? "" : "s"} · {formatMs(o.duration_ms)}
              </span>
            ) : null}
            {o.truncated ? <Badge tone="warn">truncated at row limit</Badge> : null}
          </span>
        }
        description={o.question ? <span className="line-clamp-2">“{o.question}”</span> : undefined}
        action={
          onEditSql && shownSql ? (
            <Button size="xs" variant="ghost" onClick={() => onEditSql(shownSql)}>
              <Code2 className="h-3.5 w-3.5" /> Edit as SQL
            </Button>
          ) : undefined
        }
      />
      <CardBody className="space-y-4">
        {o.explanation ? <p className="text-[13.5px] leading-relaxed text-muted">{o.explanation}</p> : null}
        {o.status === "pending_approval" ? (
          <Notice tone="warn" title="Waiting for approval">
            {o.approval_reasons.length ? <span className="block">Why: {o.approval_reasons.join("; ")}.</span> : null}
            {o.approval_id ? (
              <span className="block">
                Approval ID <span className="font-mono text-fg">{o.approval_id}</span> — an approver can run it from the Approvals tab. You&apos;ll see the result there once it is approved.
              </span>
            ) : null}
          </Notice>
        ) : null}
        {o.status === "cannot_answer" ? <Notice tone="info" title="Can’t be answered from the selected tables">{o.errors.join(" ") || "Try rephrasing, or ask a manager to select more tables."}</Notice> : null}
        {(o.status === "rejected" || o.status === "failed") && o.errors.length ? (
          <Notice tone="danger" title={o.status === "rejected" ? "Rejected by the SQL validator" : "Query failed"}>
            <ul className="list-disc space-y-0.5 pl-4">
              {o.errors.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
          </Notice>
        ) : null}
        {shownSql ? (
          <CodeBlock sql copy label={o.sql ? "Executed SQL" : "Generated SQL"}>
            {shownSql}
          </CodeBlock>
        ) : null}
        {o.status === "executed" || (o.columns.length && o.rows.length) ? <ResultView columns={o.columns} rows={o.rows} chart={o.chart} /> : null}
      </CardBody>
    </Card>
  );
}
