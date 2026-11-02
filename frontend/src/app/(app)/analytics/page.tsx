"use client";

import { Activity, ArrowUpRight, BarChart3, Coins, Cpu, Database, KeyRound, MessagesSquare, RefreshCw, ThumbsDown, ThumbsUp, Users } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";
import { Area, Bar, BarChart, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { axisTick, ChartTooltip, Forbidden, formatDuration, gridProps, PlanMeters, Segmented, tooltipProps, usd, VIZ_1, VIZ_2, VIZ_VARS } from "@/components/org/kit";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader, EmptyState, Page, PageHeader, SectionTitle, Skeleton, Stat, StatGroup, StatusBadge } from "@/components/ui/primitives";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatNumber } from "@/lib/utils";

type Day = { date: string; questions: number; active_users: number; tokens_in: number; tokens_out: number; cost_usd: number; documents: number };
type Analytics = {
  range_days: number;
  series: Day[];
  by_model: { provider: string | null; model: string | null; tokens: number; requests: number; cost_usd: number }[];
  by_kb: { kb_id: string; name: string; questions: number }[];
  latency: { retrieval_avg_ms: number; generation_avg_ms: number; retrieval_p95_ms: number };
  answers: { complete: number; refused: number; stopped: number; error: number };
  feedback: { positive: number; negative: number };
  documents: Record<string, number>;
  sql_queries: Record<string, number>;
  api_calls: number;
  conversations: number;
  usage: Record<string, number>;
  limits: Record<string, number>;
};

const RANGES = [7, 30, 90] as const;
const compact = (n: number) => new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(n);
const shortDate = (d: string) => new Date(`${d}T00:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric" });
const longDate = (d: string) => new Date(`${d}T00:00:00`).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
const activeDot = { r: 4, strokeWidth: 2, stroke: "hsl(var(--surface))" };

export default function AnalyticsPage() {
  const { can } = useAuth();
  const [days, setDays] = useState<number>(30);
  const q = useApi<Analytics>(can("analytics.view") ? "/analytics/overview" : null, { days });
  const d = q.data;

  const totals = useMemo(() => {
    const s = d?.series ?? [];
    const sum = (k: keyof Day) => s.reduce((a, r) => a + (Number(r[k]) || 0), 0);
    const peakUsers = s.reduce((a, r) => Math.max(a, r.active_users), 0);
    const activeDays = s.filter((r) => r.active_users > 0).length;
    return {
      questions: sum("questions"),
      tokensIn: sum("tokens_in"),
      tokensOut: sum("tokens_out"),
      cost: sum("cost_usd"),
      documents: sum("documents"),
      peakUsers,
      avgUsers: activeDays ? sum("active_users") / s.length : 0,
    };
  }, [d]);

  if (!can("analytics.view")) {
    return (
      <Page>
        <PageHeader eyebrow="Insights" title="Analytics & usage" />
        <Forbidden perm="analytics.view" />
      </Page>
    );
  }

  return (
    <Page className={VIZ_VARS}>
      <PageHeader
        eyebrow="Insights"
        title="Analytics & usage"
        description="Questions, model usage, answer quality and latency across your organization. Usage counts include API traffic."
        actions={
          <>
            <Segmented label="Time range" value={days} onChange={setDays} options={RANGES.map((r) => ({ value: r, label: `${r} days` }))} />
            <Button variant="secondary" size="sm" onClick={() => q.refetch()} loading={q.isFetching && !q.isLoading} aria-label="Refresh analytics">
              {!q.isFetching || q.isLoading ? <RefreshCw className="h-3.5 w-3.5" /> : null} Refresh
            </Button>
          </>
        }
      />

      <div className="space-y-10">
        <StatGroup cols={6} className="animate-fade-up">
          {!d ? (
            Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="space-y-3 px-5 py-4">
                <Skeleton className="h-3 w-20" />
                <Skeleton className="h-7 w-16" />
                <Skeleton className="h-3 w-24" />
              </div>
            ))
          ) : (
            <>
              <Stat label="Questions" value={formatNumber(totals.questions)} hint={`${(totals.questions / Math.max(1, d.series.length)).toFixed(1)} per day`} icon={<MessagesSquare />} tone="accent" />
              <Stat label="Conversations" value={formatNumber(d.conversations)} hint="Started in range" icon={<Activity />} />
              <Stat label="Peak daily users" value={formatNumber(totals.peakUsers)} hint={`${totals.avgUsers.toFixed(1)} on an average day`} icon={<Users />} />
              <Stat label="Model tokens" value={compact(totals.tokensIn + totals.tokensOut)} hint={`${compact(totals.tokensIn)} in · ${compact(totals.tokensOut)} out`} icon={<Cpu />} />
              <Stat label="Model cost" value={usd(totals.cost)} hint="Estimated from provider pricing" icon={<Coins />} />
              <Stat label="API calls" value={formatNumber(d.api_calls)} hint="Requests made with API keys" icon={<KeyRound />} />
            </>
          )}
        </StatGroup>

        {/* ------------------------------------------------------------ activity */}
        <section>
          <SectionTitle title="Activity" description={`Questions and people over the last ${days} days`} />
          <div className="grid gap-5 lg:grid-cols-3">
            <ChartCard
              className="lg:col-span-2"
              title="Questions & active users"
              description="Questions asked per day, and distinct people chatting per day"
              legend={[
                { color: VIZ_1, label: "Questions" },
                { color: VIZ_2, label: "Active users", line: true },
              ]}
              loading={!d}
              empty={d && totals.questions === 0 ? "No questions were asked in this period." : undefined}
              table={d?.series.map((r) => [shortDate(r.date), formatNumber(r.questions), formatNumber(r.active_users)])}
              tableHead={["Date", "Questions", "Active users"]}
            >
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={d?.series} margin={{ left: -6, right: 14, top: 10, bottom: 0 }}>
                  <defs>
                    <linearGradient id="viz-q" x1="0" x2="0" y1="0" y2="1">
                      <stop offset="0" stopColor={VIZ_1} stopOpacity={0.26} />
                      <stop offset="0.7" stopColor={VIZ_1} stopOpacity={0.04} />
                      <stop offset="1" stopColor={VIZ_1} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid {...gridProps} />
                  <XAxis dataKey="date" tickFormatter={shortDate} tick={axisTick} axisLine={false} tickLine={false} minTickGap={32} tickMargin={10} />
                  <YAxis tick={axisTick} axisLine={false} tickLine={false} allowDecimals={false} width={40} tickFormatter={compact} tickMargin={6} />
                  <Tooltip cursor={tooltipProps.cursor} content={(p) => <ChartTooltip active={p.active} payload={p.payload} label={p.label} labelFormatter={longDate} valueFormatter={formatNumber} colors={{ questions: VIZ_1, active_users: VIZ_2 }} />} />
                  <Area type="monotone" dataKey="questions" name="Questions" stroke={VIZ_1} strokeWidth={2} fill="url(#viz-q)" activeDot={activeDot} />
                  <Line type="monotone" dataKey="active_users" name="Active users" stroke={VIZ_2} strokeWidth={2} dot={false} activeDot={activeDot} />
                </ComposedChart>
              </ResponsiveContainer>
            </ChartCard>

            <div className="grid min-w-0 grid-cols-1 gap-5">
              <OutcomesCard answers={d?.answers} />
              <FeedbackCard feedback={d?.feedback} />
            </div>
          </div>
        </section>

        {/* ------------------------------------------------------ models & spend */}
        <section>
          <SectionTitle title="Models & spend" description="Token consumption and estimated provider cost" />
          <div className="grid gap-5 lg:grid-cols-3">
            <ChartCard
              className="lg:col-span-2"
              title="Model tokens per day"
              description="Prompt (in) and completion (out) tokens across chat, agents, SQL and embeddings"
              legend={[
                { color: VIZ_1, label: "Tokens in" },
                { color: VIZ_2, label: "Tokens out" },
              ]}
              loading={!d}
              empty={d && totals.tokensIn + totals.tokensOut === 0 ? "No model tokens were used in this period." : undefined}
              table={d?.series.map((r) => [shortDate(r.date), formatNumber(r.tokens_in), formatNumber(r.tokens_out)])}
              tableHead={["Date", "Tokens in", "Tokens out"]}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={d?.series} margin={{ left: -6, right: 14, top: 10, bottom: 0 }} barCategoryGap="22%">
                  <CartesianGrid {...gridProps} />
                  <XAxis dataKey="date" tickFormatter={shortDate} tick={axisTick} axisLine={false} tickLine={false} minTickGap={32} tickMargin={10} />
                  <YAxis tick={axisTick} axisLine={false} tickLine={false} width={44} tickFormatter={compact} tickMargin={6} />
                  <Tooltip cursor={{ fill: "hsl(var(--fg) / 0.04)" }} content={(p) => <ChartTooltip active={p.active} payload={p.payload} label={p.label} labelFormatter={longDate} valueFormatter={formatNumber} colors={{ tokens_in: VIZ_1, tokens_out: VIZ_2 }} />} />
                  <Bar dataKey="tokens_in" name="Tokens in" stackId="t" fill={VIZ_1} stroke="hsl(var(--surface))" strokeWidth={1.5} maxBarSize={28} />
                  <Bar dataKey="tokens_out" name="Tokens out" stackId="t" fill={VIZ_2} stroke="hsl(var(--surface))" strokeWidth={1.5} radius={[4, 4, 0, 0]} maxBarSize={28} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard
              title="Cost per day"
              description={d ? `${usd(totals.cost)} over ${days} days` : "Estimated model spend"}
              loading={!d}
              empty={d && totals.cost === 0 ? "No billable model usage in this period (local or keyless models report no cost)." : undefined}
              table={d?.series.map((r) => [shortDate(r.date), usd(r.cost_usd)])}
              tableHead={["Date", "Cost"]}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={d?.series} margin={{ left: -6, right: 14, top: 10, bottom: 0 }} barCategoryGap="22%">
                  <CartesianGrid {...gridProps} />
                  <XAxis dataKey="date" tickFormatter={shortDate} tick={axisTick} axisLine={false} tickLine={false} minTickGap={32} tickMargin={10} />
                  <YAxis tick={axisTick} axisLine={false} tickLine={false} width={48} tickFormatter={(v) => usd(Number(v), 0)} tickMargin={6} />
                  <Tooltip cursor={{ fill: "hsl(var(--fg) / 0.04)" }} content={(p) => <ChartTooltip active={p.active} payload={p.payload} label={p.label} labelFormatter={longDate} valueFormatter={(v) => usd(v)} colors={{ cost_usd: VIZ_1 }} />} />
                  <Bar dataKey="cost_usd" name="Cost" fill={VIZ_1} radius={[4, 4, 0, 0]} maxBarSize={28} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <Card className="overflow-hidden lg:col-span-2">
              <CardHeader title="Usage by model" description="Requests, tokens and estimated cost per provider and model" />
              {!d ? (
                <CardBody>
                  <Skeleton className="h-32" />
                </CardBody>
              ) : d.by_model.length === 0 ? (
                <EmptyState icon={<Cpu className="h-5 w-5" />} title="No model usage in this period" description="Usage is recorded for chat answers, agents, SQL generation and embeddings." className="py-12" />
              ) : (
                <div className="relative overflow-x-auto">
                  <table className="w-full min-w-[560px] text-[13px]">
                    <thead>
                      <tr className="border-b bg-surface-2/50 text-left font-mono text-[10.5px] uppercase tracking-[0.1em] text-subtle">
                        <th className="h-9 px-5 font-medium">Model</th>
                        <th className="h-9 px-4 text-right font-medium">Requests</th>
                        <th className="h-9 px-4 text-right font-medium">Tokens</th>
                        <th className="h-9 px-4 text-right font-medium">Cost</th>
                        <th className="h-9 w-[24%] px-5 font-medium">Share of tokens</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...d.by_model]
                        .sort((a, b) => b.tokens - a.tokens)
                        .map((m, i, all) => {
                          const total = Math.max(1, all.reduce((a, x) => a + x.tokens, 0));
                          const share = (m.tokens / total) * 100;
                          return (
                            <tr key={`${m.provider}-${m.model}-${i}`} className="border-b transition-colors last:border-0 hover:bg-surface-2/40">
                              <td className="px-5 py-3">
                                <div className="font-mono text-[12.5px] text-fg">{m.model || "—"}</div>
                                <div className="mt-0.5 text-[12px] capitalize text-subtle">{(m.provider || "unknown").replace(/_/g, " ")}</div>
                              </td>
                              <td className="num px-4 py-3 text-right font-mono text-[12.5px]">{formatNumber(m.requests)}</td>
                              <td className="num px-4 py-3 text-right font-mono text-[12.5px]">{formatNumber(m.tokens)}</td>
                              <td className="num px-4 py-3 text-right font-mono text-[12.5px]">{usd(m.cost_usd)}</td>
                              <td className="px-5 py-3">
                                <div className="flex items-center gap-3">
                                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                                    <div className="h-full rounded-full" style={{ width: `${share}%`, background: VIZ_1 }} />
                                  </div>
                                  <span className="num w-10 text-right font-mono text-[11.5px] text-subtle">{share.toFixed(0)}%</span>
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>

            <BreakdownCard title="SQL query outcomes" description={`Natural-language database queries · last ${days} days`} icon={<Database className="h-5 w-5" />} data={d?.sql_queries} emptyText="No database queries in this period." />
          </div>
        </section>

        {/* ------------------------------------------------- quality & latency */}
        <section>
          <SectionTitle title="Quality & performance" description="Where questions go, how fast answers arrive and what is indexed" />
          <div className="grid gap-5 lg:grid-cols-3">
            <LatencyCard latency={d?.latency} />
            <TopKbCard rows={d?.by_kb} loading={!d} />
            <BreakdownCard title="Documents by status" description="All documents, all time" icon={<Database className="h-5 w-5" />} data={d?.documents} emptyText="No documents uploaded yet." />
          </div>
        </section>

        {/* -------------------------------------------------------------- plan */}
        <section>
          <SectionTitle
            title="Plan usage"
            description={d ? (Object.keys(d.limits).length ? "Current consumption against your plan's limits" : "Your plan has no usage limits") : "Current consumption against your plan"}
            action={
              <Link href="/settings?tab=billing" className="inline-flex items-center gap-1 text-[13px] font-medium text-accent hover:underline">
                Plan & billing <ArrowUpRight className="h-3.5 w-3.5" />
              </Link>
            }
          />
          <Card>
            <CardBody className="py-5">{d ? <PlanMeters usage={d.usage} limits={d.limits} className="lg:grid-cols-3" /> : <Skeleton className="h-28" />}</CardBody>
          </Card>
        </section>
      </div>
    </Page>
  );
}

/* ----------------------------------------------------------------- cards */

function Legend({ items }: { items: { color: string; label: string; line?: boolean }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1" aria-hidden>
      {items.map((it) => (
        <span key={it.label} className="inline-flex items-center gap-1.5 text-[12px] text-muted">
          {it.line ? <span className="h-[2px] w-3 rounded-full" style={{ background: it.color }} /> : <span className="h-2 w-2 rounded-[3px]" style={{ background: it.color }} />}
          {it.label}
        </span>
      ))}
    </div>
  );
}

function ChartCard({ title, description, legend, children, className, loading, empty, table, tableHead }: { title: string; description?: string; legend?: { color: string; label: string; line?: boolean }[]; children: React.ReactNode; className?: string; loading?: boolean; empty?: string; table?: string[][]; tableHead: string[] }) {
  const [view, setView] = useState<"chart" | "table">("chart");
  return (
    <Card className={cn("flex flex-col overflow-hidden", className)}>
      <CardHeader
        title={title}
        description={description}
        action={!loading && !empty ? <Segmented label={`${title} view`} value={view} onChange={setView} options={[{ value: "chart", label: "Chart" }, { value: "table", label: "Table" }]} /> : null}
      />
      {legend && view === "chart" && !loading && !empty ? (
        <div className="px-5 pt-4">
          <Legend items={legend} />
        </div>
      ) : null}
      {loading ? (
        <CardBody>
          <Skeleton className="h-56" />
        </CardBody>
      ) : empty ? (
        <EmptyState icon={<BarChart3 className="h-5 w-5" />} title="No data yet" description={empty} className="flex-1 py-12" />
      ) : view === "chart" ? (
        <CardBody className="h-64 pb-3 pt-3" role="img" aria-label={`${title} chart. Switch to the table view for exact values.`}>
          {children}
        </CardBody>
      ) : (
        <div className="max-h-[17.5rem] overflow-y-auto">
          <table className="w-full text-[13px]">
            <thead className="sticky top-0 z-10 bg-surface">
              <tr className="border-b text-left font-mono text-[10.5px] uppercase tracking-[0.1em] text-subtle">
                {tableHead.map((h, i) => (
                  <th key={h} className={cn("h-9 bg-surface-2/50 px-5 font-medium", i > 0 && "text-right")}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...(table ?? [])].reverse().map((row) => (
                <tr key={row[0]} className="border-b last:border-0">
                  {row.map((cell, i) => (
                    <td key={i} className={cn("px-5 py-2", i > 0 ? "num text-right font-mono text-[12.5px]" : "text-muted")}>
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

const OUTCOMES = [
  { key: "complete", label: "Answered", bar: "bg-ok", hint: "Grounded answer delivered" },
  { key: "refused", label: "Refused", bar: "bg-warn", hint: "Not enough evidence in sources" },
  { key: "stopped", label: "Stopped", bar: "bg-subtle", hint: "Cancelled by the user" },
  { key: "error", label: "Error", bar: "bg-danger", hint: "Provider or system failure" },
] as const;

function OutcomesCard({ answers }: { answers?: Analytics["answers"] }) {
  const total = answers ? OUTCOMES.reduce((a, o) => a + answers[o.key], 0) : 0;
  const answeredPct = answers && total ? (answers.complete / total) * 100 : null;
  return (
    <Card>
      <CardHeader title="Answer outcomes" description={answers ? `${formatNumber(total)} assistant answers` : undefined} action={answeredPct !== null ? <span className="num font-mono text-[12px] text-muted">{answeredPct.toFixed(0)}% answered</span> : null} />
      <CardBody className="space-y-4">
        {!answers ? (
          <Skeleton className="h-28" />
        ) : total === 0 ? (
          <p className="py-6 text-center text-[13px] text-muted">No answers in this period.</p>
        ) : (
          <>
            <div className="flex h-2 gap-[2px] overflow-hidden rounded-full" aria-hidden>
              {OUTCOMES.filter((o) => answers[o.key] > 0).map((o) => (
                <div key={o.key} className={cn(o.bar, "first:rounded-l-full last:rounded-r-full")} style={{ width: `${(answers[o.key] / total) * 100}%` }} />
              ))}
            </div>
            <ul className="space-y-2">
              {OUTCOMES.map((o) => (
                <li key={o.key} className="flex items-center gap-2.5 text-[13px]">
                  <span className={cn("h-2 w-2 shrink-0 rounded-[3px]", o.bar)} />
                  <span className="font-medium">{o.label}</span>
                  <span className="hidden min-w-0 flex-1 truncate text-[12px] text-subtle sm:block lg:hidden xl:block" title={o.hint}>
                    {o.hint}
                  </span>
                  <span className="num ml-auto font-mono text-[12.5px]">{formatNumber(answers[o.key])}</span>
                  <span className="num w-12 text-right font-mono text-[11.5px] text-subtle">{((answers[o.key] / total) * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </>
        )}
      </CardBody>
    </Card>
  );
}

function FeedbackCard({ feedback }: { feedback?: Analytics["feedback"] }) {
  const total = feedback ? feedback.positive + feedback.negative : 0;
  const rate = total ? (feedback!.positive / total) * 100 : null;
  return (
    <Card>
      <CardHeader title="User feedback" description={rate === null ? "Ratings on assistant answers" : `${rate.toFixed(0)}% positive · ${formatNumber(total)} ratings`} />
      {!feedback ? (
        <CardBody>
          <Skeleton className="h-16" />
        </CardBody>
      ) : (
        <div className="grid grid-cols-2 divide-x">
          <div className="px-5 py-4">
            <div className="label flex items-center gap-1.5">
              <ThumbsUp className="h-3 w-3 text-ok" /> Helpful
            </div>
            <div className="num mt-2 text-[24px] font-semibold leading-none tracking-[-0.03em]">{formatNumber(feedback.positive)}</div>
          </div>
          <div className="px-5 py-4">
            <div className="label flex items-center gap-1.5">
              <ThumbsDown className="h-3 w-3 text-danger" /> Not helpful
            </div>
            <div className="num mt-2 text-[24px] font-semibold leading-none tracking-[-0.03em]">{formatNumber(feedback.negative)}</div>
          </div>
        </div>
      )}
    </Card>
  );
}

function LatencyCard({ latency }: { latency?: Analytics["latency"] }) {
  const rows = latency
    ? [
        { label: "Retrieval", sub: "average", value: latency.retrieval_avg_ms, color: VIZ_1 },
        { label: "Retrieval", sub: "p95", value: latency.retrieval_p95_ms, color: VIZ_1, faint: true },
        { label: "Generation", sub: "average", value: latency.generation_avg_ms, color: VIZ_2 },
      ]
    : [];
  const max = Math.max(1, ...rows.map((r) => r.value));
  return (
    <Card className="flex flex-col">
      <CardHeader title="Latency" description="Retrieval (search + rerank) and generation (model), measured separately" />
      {!latency ? (
        <CardBody>
          <Skeleton className="h-28" />
        </CardBody>
      ) : (
        <>
          <div className="border-b px-5 py-4">
            <div className="label">Typical end-to-end</div>
            <div className="num mt-2 text-[28px] font-semibold leading-none tracking-[-0.035em]">{latency.retrieval_avg_ms + latency.generation_avg_ms ? formatDuration(latency.retrieval_avg_ms + latency.generation_avg_ms) : "—"}</div>
          </div>
          <CardBody className="flex-1 space-y-4">
            {rows.map((r) => (
              <div key={`${r.label}-${r.sub}`}>
                <div className="flex items-baseline justify-between gap-3 text-[13px]">
                  <span className="text-muted">
                    {r.label} <span className="text-subtle">· {r.sub}</span>
                  </span>
                  <span className="num font-mono text-[12.5px] font-medium">{r.value ? formatDuration(r.value) : "—"}</span>
                </div>
                <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-surface-2">
                  <div className="h-full rounded-full" style={{ width: `${(r.value / max) * 100}%`, background: r.color, opacity: r.faint ? 0.5 : 1 }} />
                </div>
              </div>
            ))}
            <p className="pt-1 text-[12px] leading-relaxed text-subtle">Extractive answers (no model) report generation time as zero.</p>
          </CardBody>
        </>
      )}
    </Card>
  );
}

function TopKbCard({ rows, loading }: { rows?: Analytics["by_kb"]; loading: boolean }) {
  const top = (rows ?? []).slice(0, 8);
  const max = Math.max(1, ...top.map((r) => r.questions));
  return (
    <Card>
      <CardHeader title="Top knowledge bases" description="By questions asked (only ones you can read)" />
      {loading ? (
        <CardBody>
          <Skeleton className="h-28" />
        </CardBody>
      ) : top.length === 0 ? (
        <EmptyState title="No questions against knowledge bases" description="Questions scoped to a knowledge base appear here." className="py-10" />
      ) : (
        <ol className="space-y-3.5 px-5 py-4">
          {top.map((r, i) => (
            <li key={r.kb_id} className="flex items-start gap-3">
              <span className="num mt-px w-4 shrink-0 font-mono text-[11px] text-subtle">{String(i + 1).padStart(2, "0")}</span>
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline justify-between gap-3 text-[13px]">
                  <Link href={`/knowledge-bases/${r.kb_id}`} className="truncate font-medium underline-offset-4 hover:text-accent hover:underline">
                    {r.name}
                  </Link>
                  <span className="num shrink-0 font-mono text-[12.5px] text-muted">{formatNumber(r.questions)}</span>
                </div>
                <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-surface-2">
                  <div className="h-full rounded-full" style={{ width: `${(r.questions / max) * 100}%`, background: VIZ_1 }} />
                </div>
              </div>
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}

function BreakdownCard({ title, description, data, emptyText, icon }: { title: string; description: string; data?: Record<string, number>; emptyText: string; icon: React.ReactNode }) {
  const entries = Object.entries(data ?? {}).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((a, [, n]) => a + n, 0);
  return (
    <Card>
      <CardHeader title={title} description={data ? `${description} · ${formatNumber(total)} total` : description} />
      {!data ? (
        <CardBody>
          <Skeleton className="h-28" />
        </CardBody>
      ) : entries.length === 0 ? (
        <EmptyState icon={icon} title="Nothing yet" description={emptyText} className="py-10" />
      ) : (
        <ul className="divide-y">
          {entries.map(([status, n]) => (
            <li key={status} className="flex items-center gap-3 px-5 py-2.5 text-[13px]">
              <StatusBadge status={status} />
              <div className="hidden h-1 flex-1 overflow-hidden rounded-full bg-surface-2 sm:block">
                <div className="h-full rounded-full bg-border-strong" style={{ width: `${(n / Math.max(1, total)) * 100}%` }} />
              </div>
              <span className="num ml-auto w-14 text-right font-mono text-[12.5px] font-medium">{formatNumber(n)}</span>
              <span className="num w-10 text-right font-mono text-[11.5px] text-subtle">{((n / Math.max(1, total)) * 100).toFixed(0)}%</span>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
