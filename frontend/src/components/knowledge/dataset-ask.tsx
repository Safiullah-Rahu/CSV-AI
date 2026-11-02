"use client";

import { useMutation } from "@tanstack/react-query";
import { AlertTriangle, Bot, Clock, Code2, Copy, MessageSquareText, Pencil, Play, Table2 } from "lucide-react";
import { useRef, useState, type KeyboardEvent } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/form";
import { Badge, Card, CardBody, EmptyState, Kbd, StatusBadge, Tip } from "@/components/ui/primitives";
import { ApiError, errorMessage, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { cn, formatDuration, formatNumber } from "@/lib/utils";

import { AXIS_TICK, Cell, TOOLTIP_STYLE } from "./dataset-panels";
import { Guarded, Segmented } from "./shared";
import type { DatasetTableRef, QueryChart, QueryResult } from "./types";

const SERIES = ["hsl(var(--accent))", "hsl(var(--ok))", "hsl(var(--warn))"];

type Entry = { key: number; mode: "question" | "sql"; input: string; result: QueryResult };
let seq = 0;

export function AskPanel({ datasetId, tables }: { datasetId: string; tables: DatasetTableRef[] }) {
  const { can } = useAuth();
  const allowed = can("kb.chat");
  const [mode, setMode] = useState<"question" | "sql">("question");
  const [question, setQuestion] = useState("");
  const [sql, setSql] = useState("");
  const [llmMissing, setLlmMissing] = useState(false);
  const [history, setHistory] = useState<Entry[]>([]);
  const sqlRef = useRef<HTMLTextAreaElement>(null);

  const run = useMutation({
    meta: { silent: true },
    mutationFn: (body: { question?: string; sql?: string }) => post<QueryResult>(`/datasets/${datasetId}/query`, body),
    onSuccess: (result, body) => {
      setHistory((h) => [{ key: ++seq, mode: (body.sql ? "sql" : "question") as Entry["mode"], input: body.sql ?? body.question ?? "", result }, ...h].slice(0, 20));
    },
    onError: (err) => {
      if (err instanceof ApiError && err.code === "not_available") {
        setLlmMissing(true);
        setMode("sql");
        if (!sql.trim()) setSql(defaultSql(tables));
        return;
      }
      toast.error(errorMessage(err), err instanceof ApiError && err.requestId ? { description: `Request ID ${err.requestId}` } : undefined);
    },
  });

  const submit = () => {
    if (!allowed || run.isPending) return;
    if (mode === "question") {
      const qn = question.trim();
      if (qn) run.mutate({ question: qn });
    } else {
      const s = sql.trim();
      if (s) run.mutate({ sql: s });
    }
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      submit();
    }
  };
  const switchMode = (m: "question" | "sql") => {
    setMode(m);
    if (m === "sql" && !sql.trim()) setSql(defaultSql(tables));
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardBody className="space-y-4 px-5 py-5">
          <div className="flex flex-wrap items-center gap-3">
            <Segmented
              label="Query mode"
              value={mode}
              onChange={switchMode}
              options={[
                { value: "question", label: "Ask a question", icon: <MessageSquareText className="h-3.5 w-3.5" /> },
                { value: "sql", label: "Write SQL", icon: <Code2 className="h-3.5 w-3.5" /> },
              ]}
            />
            <span className="text-[12.5px] text-muted">Read-only, validated SQL over a sandboxed copy of this dataset. Results are capped at 1,000 rows.</span>
          </div>

          {llmMissing ? (
            <div className="flex items-start gap-2.5 rounded-lg border border-warn/25 bg-warn/[0.05] px-3.5 py-3 text-[13px]" role="status">
              <Bot className="mt-0.5 h-4 w-4 shrink-0 text-warn" />
              <div>
                <div className="font-medium">A language model must be configured to ask questions in plain language.</div>
                <div className="text-muted">An administrator can connect one in Settings → AI. SQL mode keeps working without it.</div>
              </div>
            </div>
          ) : null}

          {mode === "question" ? (
            <div>
              <label htmlFor="ask-question" className="sr-only">
                Question about this dataset
              </label>
              <Textarea id="ask-question" value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={onKey} rows={3} maxLength={4000} placeholder="e.g. What are the top 5 categories by total amount, and how many rows does each have?" disabled={!allowed} />
            </div>
          ) : (
            <div>
              <label htmlFor="ask-sql" className="sr-only">
                SQL query
              </label>
              <Textarea ref={sqlRef} id="ask-sql" value={sql} onChange={(e) => setSql(e.target.value)} onKeyDown={onKey} rows={6} maxLength={20000} spellCheck={false} className="font-mono text-[12.5px]" placeholder={defaultSql(tables)} disabled={!allowed} />
              <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11.5px] text-subtle">
                <span>Tables (DuckDB dialect):</span>
                {tables.map((t) => (
                  <Tip key={t.name} content={<span className="font-mono text-[11px]">{t.columns.slice(0, 40).join(", ")}{t.columns.length > 40 ? "…" : ""}</span>}>
                    <button type="button" className="rounded-[5px] border bg-surface-2/70 px-1.5 py-0.5 font-mono text-[11px] text-muted transition-colors hover:border-accent/40 hover:text-accent" onClick={() => setSql((s) => (s.trim() ? `${s}${s.endsWith(" ") ? "" : " "}"${t.name}"` : `SELECT *\nFROM "${t.name}"\nLIMIT 20`))}>
                      {t.name} · {formatNumber(t.rows)} rows
                    </button>
                  </Tip>
                ))}
              </div>
            </div>
          )}
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="inline-flex items-center gap-1.5 text-[12px] text-subtle">
              Press <Kbd>Ctrl</Kbd> <Kbd>Enter</Kbd> to run
            </span>
            <Guarded allowed={allowed} reason="Querying datasets requires the kb.chat permission.">
              <Button onClick={submit} loading={run.isPending} disabled={!allowed || !(mode === "question" ? question.trim() : sql.trim())}>
                <Play className="h-3.5 w-3.5" /> {mode === "question" ? "Ask" : "Run SQL"}
              </Button>
            </Guarded>
          </div>
        </CardBody>
      </Card>

      {history.length ? (
        <div className="space-y-4">
          {history.map((h) => (
            <ResultCard
              key={h.key}
              entry={h}
              onEditSql={(s) => {
                setMode("sql");
                setSql(s);
                requestAnimationFrame(() => {
                  sqlRef.current?.focus();
                  sqlRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
                });
              }}
            />
          ))}
        </div>
      ) : (
        <Card>
          <EmptyState icon={<Table2 className="h-5 w-5" />} title="Ask about the data" description="Questions are translated to SQL, validated against the allowed tables and run read-only. You'll see the exact SQL, the rows and a chart when the result has a clear shape." className="py-10" />
        </Card>
      )}
    </div>
  );
}

function defaultSql(tables: DatasetTableRef[]) {
  const t = tables[0]?.name ?? "data";
  return `SELECT *\nFROM "${t}"\nLIMIT 20`;
}

function ResultCard({ entry, onEditSql }: { entry: Entry; onEditSql: (sql: string) => void }) {
  const r = entry.result;
  const shownSql = r.sql || r.generated_sql;
  const [view, setView] = useState<"table" | "chart">(r.chart ? "chart" : "table");
  return (
    <Card className="overflow-hidden">
      <div className="flex flex-wrap items-start gap-3 border-b px-5 py-3.5">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            {r.status === "cannot_answer" ? <Badge tone="warn">cannot answer</Badge> : <StatusBadge status={r.status} />}
            <Badge>{entry.mode === "sql" ? "SQL" : "Question"}</Badge>
            {r.status === "executed" ? (
              <span className="inline-flex items-center gap-1 font-mono text-[11px] text-subtle">
                <Clock className="h-3 w-3" /> {formatDuration(r.duration_ms)} · {formatNumber(r.row_count)} row{r.row_count === 1 ? "" : "s"}
                {r.truncated ? " · truncated" : ""}
              </span>
            ) : null}
          </div>
          {entry.mode === "question" ? <p className="mt-2 text-[14.5px] font-medium tracking-[-0.01em]">{r.question || entry.input}</p> : null}
        </div>
      </div>

      <div className="space-y-4 px-5 py-4">
        {shownSql ? (
          <div className="overflow-hidden rounded-lg border">
            <div className="flex items-center gap-2 border-b bg-surface-2/50 px-3 py-1.5">
              <span className="label">{r.sql ? "Executed SQL" : "Generated SQL"}</span>
              <span className="flex-1" />
              <Button size="xs" variant="ghost" onClick={() => navigator.clipboard?.writeText(shownSql).then(() => toast.success("SQL copied"))} aria-label="Copy SQL">
                <Copy className="h-3.5 w-3.5" />
              </Button>
              <Button size="xs" variant="ghost" onClick={() => onEditSql(shownSql)}>
                <Pencil className="h-3.5 w-3.5" /> Edit
              </Button>
            </div>
            <pre className="max-h-64 overflow-auto bg-surface-2/25 px-3.5 py-3 font-mono text-[12px] leading-relaxed">
              <code>{shownSql}</code>
            </pre>
          </div>
        ) : null}

        {r.errors.length ? (
          <div className={cn("rounded-lg border px-3.5 py-3 text-[13px]", r.status === "cannot_answer" ? "border-warn/25 bg-warn/[0.05]" : "border-danger/25 bg-danger/[0.05]")} role="alert">
            <div className={cn("flex items-center gap-1.5 font-medium", r.status === "cannot_answer" ? "text-warn" : "text-danger")}>
              <AlertTriangle className="h-4 w-4" /> {r.status === "rejected" ? "Rejected by the SQL validator" : r.status === "failed" ? "Query failed" : "Cannot answer"}
            </div>
            <ul className="mt-1 list-disc space-y-0.5 pl-5">
              {r.errors.map((e, i) => (
                <li key={i} className="break-words">
                  {e}
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {r.status === "pending_approval" ? (
          <div className="rounded-lg border border-warn/25 bg-warn/[0.05] px-3.5 py-3 text-[13px]" role="status">
            <div className="font-medium text-warn">This query needs approval before it runs.</div>
            {r.approval_reasons.length ? (
              <ul className="mt-1 list-disc space-y-0.5 pl-5 text-muted">
                {r.approval_reasons.map((x, i) => (
                  <li key={i}>{x}</li>
                ))}
              </ul>
            ) : null}
            {r.approval_id ? <div className="mt-1 font-mono text-2xs text-subtle">Approval {r.approval_id}</div> : null}
          </div>
        ) : null}

        {r.explanation ? <p className="text-[13px] text-muted">{r.explanation}</p> : null}

        {r.status === "executed" ? (
          r.rows.length ? (
            <div>
              {r.chart ? (
                <div className="mb-2 flex justify-end">
                  <Segmented label="Result view" value={view} onChange={setView} options={[{ value: "chart", label: "Chart" }, { value: "table", label: "Table" }]} />
                </div>
              ) : null}
              {view === "chart" && r.chart ? <ResultChart chart={r.chart} /> : <ResultTable columns={r.columns} rows={r.rows} />}
            </div>
          ) : (
            <p className="rounded-lg border bg-surface-2/40 px-3.5 py-3 text-[13px] text-muted">The query ran successfully and returned no rows.</p>
          )
        ) : null}
      </div>
    </Card>
  );
}

function ResultTable({ columns, rows }: { columns: string[]; rows: unknown[][] }) {
  return (
    <div className="max-h-[420px] overflow-auto rounded-lg border">
      <table className="w-full text-[12.5px]">
        <thead className="sticky top-0">
          <tr className="border-b bg-surface-2">
            {columns.map((c) => (
              <th key={c} className="h-9 whitespace-nowrap px-3 text-left font-mono text-[10.5px] font-medium uppercase tracking-[0.1em] text-subtle">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b transition-colors last:border-0 hover:bg-surface-2/40">
              {row.map((v, j) => (
                <td key={j} className="whitespace-nowrap px-3 py-1.5">
                  <Cell v={v} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ResultChart({ chart }: { chart: QueryChart }) {
  const common = { data: chart.data, margin: { left: 4, right: 12, top: 8, bottom: 4 } };
  const axes = (
    <>
      <CartesianGrid vertical={false} stroke="hsl(var(--border))" />
      <XAxis dataKey={chart.x} tick={AXIS_TICK} axisLine={false} tickLine={false} minTickGap={12} />
      <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={56} />
      <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "hsl(var(--surface-2))" }} />
      {chart.y.length > 1 ? <Legend wrapperStyle={{ fontSize: 12 }} /> : null}
    </>
  );
  return (
    <div className="h-72 rounded-lg border p-3">
      <ResponsiveContainer width="100%" height="100%">
        {chart.type === "line" ? (
          <LineChart {...common}>
            {axes}
            {chart.y.map((y, i) => (
              <Line key={y} type="monotone" dataKey={y} stroke={SERIES[i % SERIES.length]} strokeWidth={2} dot={chart.data.length <= 40} />
            ))}
          </LineChart>
        ) : (
          <BarChart {...common}>
            {axes}
            {chart.y.map((y, i) => (
              <Bar key={y} dataKey={y} fill={SERIES[i % SERIES.length]} radius={[4, 4, 0, 0]} maxBarSize={48} />
            ))}
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
