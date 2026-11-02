"use client";

import { useMutation } from "@tanstack/react-query";
import { BarChart3, CheckCircle2, Copy, FileText, Rows3, Sparkles } from "lucide-react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import remarkGfm from "remark-gfm";

import { Button } from "@/components/ui/button";
import { Badge, Card, CardBody, CardHeader, EmptyState, Skeleton, Stat, StatGroup, Tip } from "@/components/ui/primitives";
import { post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatNumber } from "@/lib/utils";

import { Guarded, Pager } from "./shared";
import type { DatasetAnalysis, DatasetPreview, DatasetSummary, TableProfile } from "./types";

export const TOOLTIP_STYLE = { background: "hsl(var(--surface))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12, color: "hsl(var(--fg))" };
export const AXIS_TICK = { fontSize: 11, fill: "hsl(var(--subtle))" };

/** Compact, type-aware cell formatting for tabular values. */
export function fmtValue(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    if (Number.isInteger(v)) return v.toLocaleString();
    return v.toLocaleString(undefined, { maximumFractionDigits: Math.abs(v) < 1 ? 4 : 2 });
  }
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

export function Cell({ v }: { v: unknown }) {
  if (v === null || v === undefined || (typeof v === "number" && Number.isNaN(v))) return <span className="text-subtle">∅</span>;
  const s = fmtValue(v);
  return (
    <span className={cn("block max-w-[320px] truncate", typeof v === "number" && "num text-right")} title={s.length > 40 ? s : undefined}>
      {s}
    </span>
  );
}

const DTYPE_TONE = (dtype: string) => (/int|float|double|decimal|number/i.test(dtype) ? "info" : /bool/i.test(dtype) ? "warn" : /date|time/i.test(dtype) ? "accent" : "neutral");

/* ---------------------------------------------------------------- profile */

export function ProfilePanel({ profile }: { profile: TableProfile | undefined }) {
  if (!profile) return <EmptyState icon={<Rows3 className="h-5 w-5" />} title="No profile for this table" description="Profiles are computed when the file is indexed. Reprocess the document to rebuild it." />;
  const missingCols = profile.columns.filter((c) => c.missing_pct > 0).length;
  return (
    <div className="space-y-4">
      <StatGroup cols={4}>
        <Stat label="Rows" value={formatNumber(profile.rows)} hint="Profiled rows" />
        <Stat label="Columns" value={formatNumber(profile.columns.length)} hint={<span className={missingCols ? "text-warn" : undefined}>{missingCols} with missing values</span>} />
        <Stat label="Duplicate rows" value={<span className={profile.duplicate_rows ? "text-warn" : undefined}>{formatNumber(profile.duplicate_rows)}</span>} hint={profile.rows > 50_000 ? "In the first 50,000 rows" : "Exact duplicates"} />
        <Stat label="Memory" value={`${formatNumber(profile.memory_mb)} MB`} hint="In-memory size of the profiled sample" />
      </StatGroup>
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[960px] text-[13px]">
            <thead>
              <tr className="border-b bg-surface-2/50 text-left font-mono text-[10.5px] font-medium uppercase tracking-[0.1em] text-subtle">
                <th className="h-9 px-3 pl-4 font-medium">Column</th>
                <th className="h-9 px-3 font-medium">Type</th>
                <th className="h-9 px-3 font-medium">Missing</th>
                <th className="h-9 px-3 text-right font-medium">Unique</th>
                <th className="h-9 px-3 text-right font-medium">Mean</th>
                <th className="h-9 px-3 text-right font-medium">Std</th>
                <th className="h-9 px-3 text-right font-medium">Min</th>
                <th className="h-9 px-3 text-right font-medium">Median</th>
                <th className="h-9 px-3 text-right font-medium">Max</th>
                <th className="h-9 px-3 pr-4 font-medium">Samples</th>
              </tr>
            </thead>
            <tbody>
              {profile.columns.map((c) => (
                <tr key={c.name} className="border-b transition-colors last:border-0 hover:bg-surface-2/40">
                  <td className="max-w-[220px] truncate px-3 py-2.5 pl-4 font-medium" title={c.name}>
                    {c.name}
                  </td>
                  <td className="px-3 py-2.5">
                    <Badge tone={DTYPE_TONE(c.dtype)} className="font-mono text-[11px]">
                      {c.dtype}
                    </Badge>
                  </td>
                  <td className="px-3 py-2.5">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-surface-2">
                        <div className={cn("h-full rounded-full", c.missing_pct > 20 ? "bg-danger" : c.missing_pct > 0 ? "bg-warn" : "bg-ok")} style={{ width: `${Math.max(c.missing_pct, c.missing_pct > 0 ? 3 : 0)}%` }} />
                      </div>
                      <span className="num font-mono text-[11.5px] text-muted">{c.missing_pct}%</span>
                    </div>
                  </td>
                  <td className="num px-3 py-2.5 text-right">{formatNumber(c.unique)}</td>
                  <td className="num px-3 py-2.5 text-right">{c.mean !== null && c.mean !== undefined ? fmtValue(c.mean) : "—"}</td>
                  <td className="num px-3 py-2.5 text-right">{c.std !== null && c.std !== undefined ? fmtValue(c.std) : "—"}</td>
                  <td className="num px-3 py-2.5 text-right">{c.min !== null && c.min !== undefined ? fmtValue(c.min) : "—"}</td>
                  <td className="num px-3 py-2.5 text-right">{c.median !== null && c.median !== undefined ? fmtValue(c.median) : "—"}</td>
                  <td className="num px-3 py-2.5 text-right">{c.max !== null && c.max !== undefined ? fmtValue(c.max) : "—"}</td>
                  <td className="max-w-[260px] px-3 py-2.5 pr-4">
                    <span className="flex flex-wrap gap-1">
                      {(c.sample ?? []).slice(0, 3).map((s, i) => (
                        <span key={i} className="max-w-[120px] truncate rounded-[5px] border bg-surface-2/70 px-1.5 py-px font-mono text-[11px] text-muted" title={fmtValue(s)}>
                          {fmtValue(s) || "∅"}
                        </span>
                      ))}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- preview */

export function PreviewPanel({ datasetId, table }: { datasetId: string; table: string }) {
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(50);
  const q = useApi<DatasetPreview>(`/datasets/${datasetId}/preview`, { table, limit, offset });
  if (q.isLoading) return <Skeleton className="h-96 rounded-xl" />;
  if (!q.data || !q.data.columns.length) return <EmptyState icon={<Rows3 className="h-5 w-5" />} title="No rows to preview" />;
  return (
    <div className="card overflow-hidden">
      <div className={cn("max-h-[560px] overflow-auto transition-opacity", q.isFetching && "opacity-70")}>
        <table className="w-full text-[12.5px]">
          <thead className="sticky top-0 z-10">
            <tr className="border-b bg-surface-2">
              <th className="sticky left-0 h-9 border-r bg-surface-2 px-3 text-right font-mono text-[10.5px] font-medium uppercase tracking-[0.1em] text-subtle">#</th>
              {q.data.columns.map((c) => (
                <th key={c} className="h-9 whitespace-nowrap px-3 text-left font-mono text-[10.5px] font-medium uppercase tracking-[0.1em] text-subtle">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {q.data.rows.map((r, i) => (
              <tr key={offset + i} className="border-b last:border-0 hover:bg-surface-2/50">
                <td className="num sticky left-0 border-r bg-surface px-3 py-1.5 text-right font-mono text-[11px] text-subtle">{offset + i + 1}</td>
                {r.map((v, j) => (
                  <td key={j} className="whitespace-nowrap px-3 py-1.5">
                    <Cell v={v} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Pager total={q.data.total} limit={limit} offset={offset} onChange={setOffset} onLimit={(l) => { setLimit(l); setOffset(0); }} limits={[25, 50, 100, 200]} />
    </div>
  );
}

/* --------------------------------------------------------------- analysis */

function corrColor(v: number) {
  const a = Math.min(1, Math.abs(v));
  return v >= 0 ? `hsl(var(--accent) / ${0.08 + a * 0.82})` : `hsl(var(--danger) / ${0.08 + a * 0.82})`;
}

export function AnalysisPanel({ datasetId, table }: { datasetId: string; table: string }) {
  const q = useApi<DatasetAnalysis>(`/datasets/${datasetId}/analysis`, { table });
  if (q.isLoading)
    return (
      <div className="space-y-4">
        <Skeleton className="h-[108px] rounded-xl" />
        <Skeleton className="h-64 rounded-xl" />
      </div>
    );
  const a = q.data;
  if (!a) return <EmptyState icon={<BarChart3 className="h-5 w-5" />} title="Analysis unavailable" />;
  const missing = a.missing.filter((m) => m.missing > 0).sort((x, y) => y.pct - x.pct);
  const shown = missing.slice(0, 30);
  const numericKeys = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];
  const tops = Object.entries(a.top_values);

  return (
    <div className="space-y-5">
      <StatGroup cols={4}>
        <Stat label="Rows analysed" value={formatNumber(a.rows)} hint={a.rows >= 200_000 ? "First 200,000 rows" : "All rows"} />
        <Stat label="Columns" value={formatNumber(a.columns)} hint={`${a.numeric.length} numeric · ${tops.length} categorical`} />
        <Stat label="Duplicate rows" value={<span className={a.duplicates ? "text-warn" : undefined}>{formatNumber(a.duplicates)}</span>} hint={a.duplicates ? "Exact duplicates found" : "No duplicates"} />
        <Stat label="Columns with gaps" value={<span className={missing.length ? "text-warn" : "text-ok"}>{formatNumber(missing.length)}</span>} hint={missing.length ? `Worst: ${missing[0].column} (${missing[0].pct}%)` : "No missing values"} />
      </StatGroup>

      <Card>
        <CardHeader title="Missing values" description={missing.length ? `Share of empty cells per column${missing.length > shown.length ? ` — top ${shown.length} of ${missing.length}` : ""}.` : undefined} />
        {missing.length ? (
          <CardBody style={{ height: Math.max(140, shown.length * 26 + 40) }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={shown} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
                <CartesianGrid horizontal={false} stroke="hsl(var(--border))" />
                <XAxis type="number" domain={[0, 100]} unit="%" tick={AXIS_TICK} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="column" width={140} tick={AXIS_TICK} axisLine={false} tickLine={false} interval={0} />
                <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "hsl(var(--surface-2))" }} formatter={(v, _n, item) => [`${v}% (${formatNumber((item.payload as { missing: number }).missing)} cells)`, "Missing"]} />
                <Bar dataKey="pct" fill="hsl(var(--warn))" radius={[0, 4, 4, 0]} maxBarSize={18} />
              </BarChart>
            </ResponsiveContainer>
          </CardBody>
        ) : (
          <CardBody className="flex items-center gap-2 text-[13px] text-ok">
            <CheckCircle2 className="h-4 w-4" /> No missing values in any column.
          </CardBody>
        )}
      </Card>

      {a.numeric.length ? (
        <Card>
          <CardHeader title="Numeric columns" description="Descriptive statistics." />
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-[13px]">
              <thead>
                <tr className="border-b bg-surface-2/50 font-mono text-[10.5px] font-medium uppercase tracking-[0.1em] text-subtle">
                  <th className="h-9 px-3 pl-5 text-left font-medium">Column</th>
                  {numericKeys.map((k) => (
                    <th key={k} className="h-9 px-3 text-right font-medium last:pr-5">
                      {k}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {a.numeric.map((r) => (
                  <tr key={String(r.column)} className="border-b transition-colors last:border-0 hover:bg-surface-2/40">
                    <td className="max-w-[200px] truncate px-3 py-2.5 pl-5 font-medium">{String(r.column)}</td>
                    {numericKeys.map((k) => (
                      <td key={k} className="num px-3 py-2.5 text-right last:pr-5">
                        {r[k] === null || r[k] === undefined ? "—" : fmtValue(r[k])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ) : null}

      {tops.length ? (
        <Card>
          <CardHeader title="Top values" description="Most frequent values in text and categorical columns (first 20 columns)." />
          <CardBody className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {tops.map(([col, vals]) => {
              const max = Math.max(1, ...vals.map((v) => v.count));
              return (
                <div key={col} className="min-w-0 rounded-lg border bg-surface-2/30 p-3.5">
                  <div className="mb-2.5 truncate font-mono text-[12px] font-medium" title={col}>
                    {col}
                  </div>
                  <ul className="space-y-1.5">
                    {vals.map((v, i) => (
                      <li key={i} className="text-[12px]">
                        <div className="flex items-center justify-between gap-2">
                          <span className={cn("truncate", v.value === null && "italic text-subtle")} title={v.value ?? "missing"}>
                            {v.value ?? "(missing)"}
                          </span>
                          <span className="num shrink-0 font-mono text-[11.5px] text-muted">{formatNumber(v.count)}</span>
                        </div>
                        <div className="mt-1 h-1 overflow-hidden rounded-full bg-border/70">
                          <div className="h-full rounded-full bg-accent/70" style={{ width: `${(v.count / max) * 100}%` }} />
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              );
            })}
          </CardBody>
        </Card>
      ) : null}

      {a.correlation && a.correlation.columns.length >= 2 ? <CorrelationHeatmap columns={a.correlation.columns} matrix={a.correlation.matrix} /> : null}
    </div>
  );
}

function CorrelationHeatmap({ columns, matrix }: { columns: string[]; matrix: number[][] }) {
  const n = columns.length;
  const dense = n > 12;
  return (
    <Card>
      <CardHeader
        title="Correlation"
        description="Pearson correlation between numeric columns."
        action={
          <div className="flex items-center gap-2 text-2xs text-subtle" aria-hidden>
            <span>−1</span>
            <span className="h-2 w-28 rounded-full" style={{ background: "linear-gradient(90deg, hsl(var(--danger)), hsl(var(--surface-2)), hsl(var(--accent)))" }} />
            <span>+1</span>
          </div>
        }
      />
      <CardBody className="overflow-x-auto">
        <div role="table" aria-label="Correlation matrix" className="grid gap-px text-[11px]" style={{ gridTemplateColumns: `minmax(90px, 160px) repeat(${n}, minmax(${dense ? 28 : 44}px, ${dense ? 40 : 76}px))`, minWidth: 100 + n * (dense ? 30 : 46) }}>
          <div role="row" className="contents">
            <div role="columnheader" />
            {columns.map((c) => (
              <div key={c} role="columnheader" className="flex h-20 items-end justify-center pb-1" title={c}>
                <span className="max-h-20 truncate text-subtle [writing-mode:vertical-rl] rotate-180">{c}</span>
              </div>
            ))}
          </div>
          {matrix.map((row, i) => (
            <div key={columns[i]} role="row" className="contents">
              <div role="rowheader" className="flex items-center truncate pr-2 text-subtle" title={columns[i]}>
                {columns[i]}
              </div>
              {row.map((v, j) => (
                <div
                  key={j}
                  role="cell"
                  title={`${columns[i]} × ${columns[j]}: ${v.toFixed(3)}`}
                  aria-label={`${columns[i]} and ${columns[j]}: ${v.toFixed(2)}`}
                  className={cn("grid aspect-square place-items-center rounded-[4px] font-mono text-[10.5px] font-medium tabular-nums", Math.abs(v) > 0.55 ? "text-white" : "text-fg")}
                  style={{ background: corrColor(v) }}
                >
                  {dense ? "" : v.toFixed(2)}
                </div>
              ))}
            </div>
          ))}
        </div>
      </CardBody>
    </Card>
  );
}

/* ---------------------------------------------------------------- summary */

export function SummaryPanel({ datasetId, table }: { datasetId: string; table: string }) {
  const { can } = useAuth();
  const allowed = can("kb.chat");
  const [copied, setCopied] = useState(false);
  const run = useMutation({ mutationFn: () => post<DatasetSummary>(`/datasets/${datasetId}/summary`, undefined, { query: { table } }) });
  const r = run.data;
  return (
    <Card>
      <CardHeader
        title="Summary"
        description="A written overview of this table. Uses the organization's language model when one is configured; otherwise a deterministic summary is built from the profile."
        action={
          <Guarded allowed={allowed} reason="Summaries require the kb.chat permission.">
            <Button size="sm" onClick={() => run.mutate()} loading={run.isPending} disabled={!allowed}>
              <Sparkles className="h-3.5 w-3.5" /> {r ? "Regenerate" : "Generate summary"}
            </Button>
          </Guarded>
        }
      />
      <CardBody>
        {run.isPending ? (
          <div className="space-y-2">
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6" />
            <Skeleton className="h-4 w-3/4" />
          </div>
        ) : r ? (
          <div>
            <div className="mb-3 flex flex-wrap items-center gap-2">
              {r.mode === "llm" ? (
                <Badge tone="accent">
                  <Sparkles className="h-3 w-3" /> Language model{r.model ? ` · ${r.model}` : ""}
                </Badge>
              ) : (
                <Tip content="No language model is configured for this organization, so the summary is computed from column statistics.">
                  <span>
                    <Badge>Deterministic</Badge>
                  </span>
                </Tip>
              )}
              <Button
                size="xs"
                variant="ghost"
                className="ml-auto"
                onClick={() => {
                  navigator.clipboard?.writeText(r.summary).then(() => {
                    setCopied(true);
                    setTimeout(() => setCopied(false), 1500);
                  });
                }}
              >
                <Copy className="h-3.5 w-3.5" /> {copied ? "Copied" : "Copy"}
              </Button>
            </div>
            <div className="prose-chat max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{r.summary}</ReactMarkdown>
            </div>
          </div>
        ) : (
          <EmptyState icon={<FileText className="h-5 w-5" />} title="No summary yet" description={allowed ? "Generate a summary of this table's shape, columns and notable statistics." : "You need the kb.chat permission to generate summaries."} className="py-8" />
        )}
      </CardBody>
    </Card>
  );
}
