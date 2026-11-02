"use client";

import { BarChart3, Table2 } from "lucide-react";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { useTheme } from "@/app/providers";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

import type { ChartSpec } from "./types";

/* Categorical series colours, validated for lightness band, chroma, CVD and
   normal-vision separation against each theme's surface (dataviz validator).
   Status hues (ok/warn/danger) are deliberately not used for series. */
const SERIES = {
  light: ["#1f71f4", "#e8701d", "#b8368f"],
  dark: ["#1a9fe0", "#d9692a", "#d4549f"],
};

function formatCell(v: unknown): React.ReactNode {
  if (v === null || v === undefined) return <span className="font-mono text-[11px] text-subtle">NULL</span>;
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number") return <span className="num font-mono text-[12px] tracking-[-0.01em]">{Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 6 })}</span>;
  if (typeof v === "object") return <span className="font-mono text-[11.5px]">{JSON.stringify(v)}</span>;
  return String(v);
}

/** Horizontally and vertically scrollable result grid with a sticky header. */
export function ResultTable({ columns, rows, className, maxRows = 500 }: { columns: string[]; rows: unknown[][]; className?: string; maxRows?: number }) {
  const shown = rows.slice(0, maxRows);
  const numeric = columns.map((_, j) => shown.length > 0 && shown.every((r) => r[j] === null || r[j] === undefined || typeof r[j] === "number"));
  return (
    <div className="overflow-hidden rounded-lg border bg-surface">
      <div className={cn("overflow-auto", className ?? "max-h-[420px]")}>
        <table className="w-full border-separate border-spacing-0 text-[12.5px]">
          <thead className="sticky top-0 z-10">
            <tr>
              <th scope="col" className="h-8 w-10 border-b border-r bg-surface-2 px-2 text-right font-mono text-[10.5px] font-medium text-subtle">
                #
              </th>
              {columns.map((c, j) => (
                <th key={c} scope="col" className={cn("h-8 whitespace-nowrap border-b bg-surface-2 px-3 font-mono text-[11px] font-medium tracking-[-0.01em] text-muted", numeric[j] ? "text-right" : "text-left")}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((r, i) => (
              <tr key={i} className="group/r">
                <td className="num h-8 border-b border-r bg-surface-2/30 px-2 text-right font-mono text-[10.5px] text-subtle group-last/r:border-b-0">{i + 1}</td>
                {columns.map((c, j) => (
                  <td
                    key={c}
                    className={cn("h-8 max-w-[320px] truncate whitespace-nowrap border-b px-3 transition-colors group-last/r:border-b-0 group-hover/r:bg-surface-2/50", (numeric[j] || typeof r[j] === "number") && "text-right")}
                    title={r[j] === null || r[j] === undefined ? undefined : String(typeof r[j] === "object" ? JSON.stringify(r[j]) : r[j])}
                  >
                    {formatCell(r[j])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between gap-3 border-t bg-surface-2/40 px-3 py-1.5 font-mono text-[11px] text-subtle">
        <span className="num">
          {rows.length} row{rows.length === 1 ? "" : "s"} · {columns.length} column{columns.length === 1 ? "" : "s"}
        </span>
        {rows.length > shown.length ? <span>Showing the first {shown.length} of {rows.length} rows.</span> : null}
      </div>
    </div>
  );
}

/** Renders the `{type, x, y[], data[]}` chart suggestion returned by the API. */
export function ResultChart({ chart, height = 260 }: { chart: ChartSpec; height?: number }) {
  const { resolved } = useTheme();
  const colors = SERIES[resolved];
  const multi = chart.y.length > 1;
  const axis = { fontSize: 11, fill: "hsl(var(--subtle))" };
  const tooltip = {
    contentStyle: { background: "hsl(var(--surface))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12, color: "hsl(var(--fg))" },
    labelStyle: { color: "hsl(var(--muted))", marginBottom: 2 },
    itemStyle: { color: "hsl(var(--fg))" },
  };
  const tickFormatter = (v: unknown) => {
    const s = String(v ?? "");
    return s.length > 14 ? s.slice(0, 13) + "…" : s;
  };
  const yFormatter = (v: unknown) => (typeof v === "number" ? new Intl.NumberFormat(undefined, { notation: Math.abs(v) >= 10_000 ? "compact" : "standard", maximumFractionDigits: 2 }).format(v) : String(v));
  return (
    <figure aria-label={`${chart.type} chart of ${chart.y.join(", ")} by ${chart.x}`}>
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          {chart.type === "line" ? (
            <LineChart data={chart.data} margin={{ left: 4, right: 12, top: 8, bottom: 0 }}>
              <CartesianGrid vertical={false} stroke="hsl(var(--border))" strokeDasharray="3 3" />
              <XAxis dataKey={chart.x} tick={axis} tickFormatter={tickFormatter} axisLine={false} tickLine={false} minTickGap={16} />
              <YAxis tick={axis} tickFormatter={yFormatter} axisLine={false} tickLine={false} width={56} />
              <Tooltip {...tooltip} cursor={{ stroke: "hsl(var(--subtle))", strokeDasharray: "3 3" }} />
              {multi ? <Legend wrapperStyle={{ fontSize: 12, color: "hsl(var(--muted))" }} iconType="plainline" /> : null}
              {chart.y.map((y, i) => (
                <Line key={y} type="monotone" dataKey={y} stroke={colors[i % colors.length]} strokeWidth={2} dot={chart.data.length <= 24 ? { r: 4, strokeWidth: 2, stroke: "hsl(var(--surface))", fill: colors[i % colors.length] } : false} activeDot={{ r: 5, strokeWidth: 2, stroke: "hsl(var(--surface))" }} isAnimationActive={false} />
              ))}
            </LineChart>
          ) : (
            <BarChart data={chart.data} margin={{ left: 4, right: 12, top: 8, bottom: 0 }} barGap={2} barCategoryGap="22%">
              <CartesianGrid vertical={false} stroke="hsl(var(--border))" strokeDasharray="3 3" />
              <XAxis dataKey={chart.x} tick={axis} tickFormatter={tickFormatter} axisLine={false} tickLine={false} interval="preserveStartEnd" minTickGap={8} />
              <YAxis tick={axis} tickFormatter={yFormatter} axisLine={false} tickLine={false} width={56} />
              <Tooltip {...tooltip} cursor={{ fill: "hsl(var(--surface-2))" }} />
              {multi ? <Legend wrapperStyle={{ fontSize: 12, color: "hsl(var(--muted))" }} iconType="square" /> : null}
              {chart.y.map((y, i) => (
                <Bar key={y} dataKey={y} fill={colors[i % colors.length]} radius={[4, 4, 0, 0]} maxBarSize={48} isAnimationActive={false} />
              ))}
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
      <figcaption className="mt-2 font-mono text-[11px] text-subtle">
        {chart.y.join(", ")} by {chart.x} · {chart.data.length} points
      </figcaption>
    </figure>
  );
}

/** Chart + table switcher for a result that may carry a chart suggestion. */
export function ResultView({ columns, rows, chart }: { columns: string[]; rows: unknown[][]; chart: ChartSpec | null }) {
  const [view, setView] = useState<"chart" | "table">(chart ? "chart" : "table");
  if (!columns.length) return <p className="text-[13px] text-muted">The query returned no columns.</p>;
  return (
    <div className="space-y-3">
      {chart ? (
        <Tabs value={view} onValueChange={(v) => setView(v as "chart" | "table")}>
          <TabsList aria-label="Result view">
            <TabsTrigger value="chart">
              <BarChart3 className="h-3.5 w-3.5" /> Chart
            </TabsTrigger>
            <TabsTrigger value="table">
              <Table2 className="h-3.5 w-3.5" /> Table
            </TabsTrigger>
          </TabsList>
        </Tabs>
      ) : null}
      {chart && view === "chart" ? (
        <div className="rounded-lg border bg-surface p-4 pb-3">
          <ResultChart chart={chart} />
        </div>
      ) : rows.length ? (
        <ResultTable columns={columns} rows={rows} />
      ) : (
        <p className="rounded-lg border border-dashed px-3 py-6 text-center text-[13px] text-muted">No rows matched.</p>
      )}
    </div>
  );
}
