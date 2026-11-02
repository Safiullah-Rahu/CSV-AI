"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, ArrowUpRight, BookOpen, Check, CornerDownLeft, Database, FileText, Layers, Library, Network, Sparkles, Users } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader, Dot, EmptyState, Page, Skeleton, Stat, StatGroup, StatusBadge, type Tone } from "@/components/ui/primitives";
import { post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { type ActivityEvent, useApi } from "@/lib/hooks";
import { cn, formatBytes, formatDuration, formatNumber, timeAgo } from "@/lib/utils";

type Onboarding = { steps: Record<string, boolean>; complete: boolean };
type Graph = { summary: { total_sources: number; active: number; failed: number; syncing: number; documents: number; records: number; chunks: number; active_jobs: number; storage_bytes: number; last_successful_sync: string | null } };
type Health = { status: string; infrastructure: Record<string, { ok: boolean; latency_ms?: number; backend?: string }>; warnings: { severity: string; resource: string; message: string }[] };
type Analytics = { series: { date: string; questions: number }[]; answers: Record<string, number>; latency: { retrieval_avg_ms: number; generation_avg_ms: number } };

const STEPS: { key: string; label: string; href: string; perm?: string }[] = [
  { key: "knowledge_base", label: "Create a knowledge base", href: "/knowledge-bases", perm: "kb.create" },
  { key: "document", label: "Upload and index a document", href: "/documents", perm: "kb.upload" },
  { key: "question", label: "Ask a question in AI Chat", href: "/chat", perm: "kb.chat" },
  { key: "ai_provider", label: "Connect a language model", href: "/settings?tab=ai", perm: "admin.ai_providers" },
  { key: "data_source", label: "Connect a database or integration", href: "/database-connections", perm: "db.create" },
  { key: "teammate", label: "Invite a teammate", href: "/team", perm: "org.users" },
];

const SEVERITY: Record<string, Tone> = { error: "danger", warning: "warn", success: "ok" };

export default function OverviewPage() {
  const { org, user, can, canAnywhere } = useAuth();
  const router = useRouter();
  const qc = useQueryClient();
  const onboarding = useApi<Onboarding>("/onboarding/status");
  const graph = useApi<Graph>("/connections/graph");
  const health = useApi<Health>("/connections/health", undefined, { refetchInterval: 60_000 });
  const events = useApi<{ items: ActivityEvent[] }>("/events", { limit: 12 }, { refetchInterval: 20_000 });
  const analytics = useApi<Analytics>(can("analytics.view") ? "/analytics/overview" : null, { days: 14 });
  const sample = useMutation({
    mutationFn: () => post<{ id: string; created: boolean }>("/onboarding/sample-kb"),
    onSuccess: (r) => {
      toast.success(r.created ? "Sample knowledge base created — indexing documents" : "Sample knowledge base already exists");
      qc.invalidateQueries();
      router.push(`/knowledge-bases/${r.id}`);
    },
  });
  const s = graph.data?.summary;
  const steps = STEPS.filter((st) => !st.perm || canAnywhere(st.perm));
  const done = steps.filter((st) => onboarding.data?.steps[st.key]).length;
  const firstName = (user?.full_name || user?.email || "").split(/[\s@]/)[0];
  const totalQuestions = analytics.data?.series.reduce((n, d) => n + d.questions, 0) ?? 0;

  return (
    <Page>
      {/* Hero: editorial greeting + the one thing people come here to do. */}
      <section className="relative mb-8 animate-fade-up">
        <div className="pointer-events-none absolute -inset-x-5 -top-9 h-[260px] bg-dots sm:-inset-x-8 opacity-70 [mask-image:radial-gradient(ellipse_at_top_right,black,transparent_65%)]" />
        <div className="relative flex flex-wrap items-end justify-between gap-6">
          <div className="min-w-0">
            <div className="label mb-3 flex items-center gap-2">
              <span className="h-1 w-3 rounded-full bg-accent" />
              {org?.organization.name}
              <span className="text-border-strong">/</span>
              <span>{new Date().toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" })}</span>
            </div>
            <h1 className="font-display text-[44px] leading-[1.02] tracking-[-0.02em] sm:text-[52px]">
              Good {greeting()}, <span className="italic text-muted">{firstName}.</span>
            </h1>
            <p className="mt-3 max-w-xl text-[14.5px] leading-relaxed text-muted">Your organization&apos;s knowledge, data sources and AI usage at a glance.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="secondary" onClick={() => router.push("/connections")}>
              <Network /> Connections map
            </Button>
          </div>
        </div>
        {can("kb.chat") ? <AskBar /> : null}
      </section>

      {graph.isLoading ? (
        <Skeleton className="h-[118px] rounded-xl" />
      ) : (
        <StatGroup className="animate-fade-up [animation-delay:60ms]">
          <Stat label="Documents indexed" value={formatNumber(s?.documents)} hint={`${formatNumber(s?.chunks)} chunks · ${formatBytes(s?.storage_bytes)}`} icon={<FileText />} />
          <Stat label="Records indexed" value={formatNumber(s?.records)} hint="Rows from datasets and databases" icon={<Layers />} />
          <Stat label="Connected sources" value={formatNumber(s?.total_sources)} hint={s ? `${s.active} active · ${s.failed} failed · ${s.syncing} syncing` : undefined} icon={<Database />} tone={s?.failed ? "danger" : undefined} />
          <Stat label="Active jobs" value={formatNumber(s?.active_jobs)} hint={s?.last_successful_sync ? `Last sync ${timeAgo(s.last_successful_sync)}` : "No syncs yet"} icon={<Sparkles />} tone={s?.active_jobs ? "info" : undefined} />
        </StatGroup>
      )}

      <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-3 [&>*]:min-w-0">
        <div className="space-y-6 lg:col-span-2">
          {analytics.data ? (
            <Card className="animate-fade-up [animation-delay:120ms]">
              <div className="flex flex-wrap items-start justify-between gap-4 px-5 pt-5">
                <div>
                  <div className="label">Questions answered · 14 days</div>
                  <div className="num mt-2 text-[28px] font-semibold leading-none tracking-[-0.035em]">{formatNumber(totalQuestions)}</div>
                </div>
                <div className="flex gap-6 text-right">
                  <Metric label="Avg retrieval" value={formatDuration(analytics.data.latency.retrieval_avg_ms)} />
                  <Metric label="Avg generation" value={formatDuration(analytics.data.latency.generation_avg_ms)} />
                  <Link href="/analytics" className="hidden self-start rounded-md p-1 text-subtle transition-colors hover:bg-surface-2 hover:text-fg sm:block" aria-label="Open analytics">
                    <ArrowUpRight className="h-4 w-4" />
                  </Link>
                </div>
              </div>
              <div className="h-52 px-2 pb-3 pt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={analytics.data.series} margin={{ left: 4, right: 24, top: 6, bottom: 0 }}>
                    <defs>
                      <linearGradient id="q" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="0" stopColor="hsl(var(--accent))" stopOpacity={0.28} />
                        <stop offset="1" stopColor="hsl(var(--accent))" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid vertical={false} stroke="hsl(var(--border))" strokeDasharray="2 4" />
                    <XAxis dataKey="date" tickFormatter={(d) => d.slice(5)} tick={{ fontSize: 10.5, fill: "hsl(var(--subtle))", fontFamily: "var(--font-mono)" }} axisLine={false} tickLine={false} dy={6} minTickGap={24} />
                    <YAxis allowDecimals={false} width={28} tick={{ fontSize: 10.5, fill: "hsl(var(--subtle))", fontFamily: "var(--font-mono)" }} axisLine={false} tickLine={false} />
                    <Tooltip
                      cursor={{ stroke: "hsl(var(--border-strong))", strokeDasharray: "3 3" }}
                      contentStyle={{ background: "hsl(var(--surface))", border: "1px solid hsl(var(--border))", borderRadius: 10, fontSize: 12, boxShadow: "0 12px 32px -12px hsl(var(--shadow) / 0.3)" }}
                      labelStyle={{ color: "hsl(var(--muted))", fontFamily: "var(--font-mono)", fontSize: 11 }}
                      itemStyle={{ color: "hsl(var(--fg))" }}
                    />
                    <Area type="monotone" dataKey="questions" name="Questions" stroke="hsl(var(--accent))" strokeWidth={2} fill="url(#q)" activeDot={{ r: 4, strokeWidth: 2, stroke: "hsl(var(--surface))" }} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </Card>
          ) : null}

          <Card className="animate-fade-up [animation-delay:160ms]">
            <CardHeader
              title="Recent activity"
              description="Ingestion, sync and administration events"
              action={
                <Link href="/connections" className="inline-flex items-center gap-1 text-[12.5px] font-medium text-muted transition-colors hover:text-fg">
                  Live feed <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              }
            />
            {events.isLoading ? (
              <div className="space-y-3 p-5">{Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-7" />)}</div>
            ) : !events.data?.items.length ? (
              <EmptyState icon={<Library className="h-5 w-5" />} title="No activity yet" description="Create a knowledge base or connect a data source and events will stream in here." />
            ) : (
              <ol className="relative px-5 py-3">
                <span className="absolute bottom-6 left-[23px] top-6 w-px bg-border" aria-hidden />
                {events.data.items.map((e) => (
                  <li key={e.id} className="relative flex items-start gap-3.5 py-2">
                    <span className="relative z-[1] mt-[5px] grid h-[11px] w-[11px] shrink-0 place-items-center rounded-full bg-surface ring-1 ring-border">
                      <Dot tone={SEVERITY[e.severity] ?? "info"} />
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[13.5px]">{e.message || e.type}</div>
                      <div className="mt-0.5 font-mono text-[11px] text-subtle">{e.type}</div>
                    </div>
                    <span className="num shrink-0 pt-0.5 font-mono text-[11px] text-subtle">{timeAgo(e.created_at)}</span>
                  </li>
                ))}
              </ol>
            )}
          </Card>
        </div>

        <div className="space-y-6">
          {!onboarding.data?.complete && steps.length ? (
            <Card className="animate-fade-up [animation-delay:120ms]">
              <div className="flex items-center gap-4 px-5 pb-4 pt-5">
                <ProgressRing value={done} total={steps.length} />
                <div className="min-w-0">
                  <h3 className="text-[14px] font-semibold tracking-[-0.01em]">Finish setting up</h3>
                  <p className="mt-0.5 text-[12.5px] text-muted">
                    {done} of {steps.length} steps complete
                  </p>
                </div>
              </div>
              <ul className="border-t px-2 py-2">
                {steps.map((st) => {
                  const ok = onboarding.data?.steps[st.key];
                  return (
                    <li key={st.key}>
                      <Link href={st.href} className="group flex items-center gap-3 rounded-lg px-3 py-2 text-[13px] transition-colors hover:bg-surface-2/70">
                        {ok ? (
                          <span className="grid h-[18px] w-[18px] place-items-center rounded-full bg-ok/10 text-ok">
                            <Check className="h-3 w-3" strokeWidth={3} />
                          </span>
                        ) : (
                          <span className="h-[18px] w-[18px] rounded-full border border-dashed border-border-strong" />
                        )}
                        <span className={cn("flex-1", ok ? "text-subtle line-through decoration-border-strong" : "font-medium")}>{st.label}</span>
                        {!ok ? <ArrowRight className="h-3.5 w-3.5 text-subtle opacity-0 transition-opacity group-hover:opacity-100" /> : null}
                      </Link>
                    </li>
                  );
                })}
              </ul>
              {can("kb.create") && !onboarding.data?.steps.document ? (
                <div className="border-t px-5 py-3">
                  <Button size="sm" variant="secondary" className="w-full" onClick={() => sample.mutate()} loading={sample.isPending}>
                    <BookOpen /> Load the sample knowledge base
                  </Button>
                </div>
              ) : null}
            </Card>
          ) : null}

          <Card className="animate-fade-up [animation-delay:160ms]">
            <CardHeader title="System health" action={health.data ? <StatusBadge status={health.data.status === "operational" ? "healthy" : health.data.status === "degraded" ? "warning" : "failed"} /> : null} />
            <CardBody className="space-y-0 divide-y py-1">
              {health.data ? (
                Object.entries(health.data.infrastructure).map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between py-2.5 text-[13px]">
                    <span className="flex items-center gap-2.5 capitalize">
                      <Dot tone={v.ok ? "ok" : "danger"} />
                      {k}
                    </span>
                    <span className="font-mono text-[11.5px] text-subtle">{v.latency_ms !== undefined ? `${v.latency_ms} ms` : v.backend ?? (v.ok ? "ok" : "down")}</span>
                  </div>
                ))
              ) : (
                <Skeleton className="my-3 h-24" />
              )}
            </CardBody>
            {health.data?.warnings.length ? (
              <div className="space-y-2 border-t px-5 py-4">
                {health.data.warnings.slice(0, 4).map((w, i) => (
                  <div key={i} className="rounded-lg bg-warn/[0.07] px-3 py-2 text-[12px] ring-1 ring-inset ring-warn/20">
                    <div className="font-medium">{w.resource}</div>
                    <div className="mt-0.5 text-muted">{w.message}</div>
                  </div>
                ))}
              </div>
            ) : null}
          </Card>

          <Card className="animate-fade-up [animation-delay:200ms]">
            <CardHeader title="Jump to" />
            <ul className="p-2">
              {[
                { href: "/knowledge-bases", label: "Knowledge bases", hint: "Collections & access", icon: Library, perm: "kb.view" },
                { href: "/documents", label: "Documents", hint: "Every indexed file", icon: FileText, perm: "kb.view" },
                { href: "/database-connections", label: "Databases", hint: "Live SQL sources", icon: Database, perm: "db.view" },
                { href: "/team", label: "Team", hint: "Members & invitations", icon: Users, perm: "org.view" },
              ]
                .filter((l) => canAnywhere(l.perm))
                .map((l) => (
                  <li key={l.href}>
                    <Link href={l.href} className="group flex items-center gap-3 rounded-lg px-3 py-2 transition-colors hover:bg-surface-2/70">
                      <span className="grid h-8 w-8 place-items-center rounded-lg border bg-surface text-subtle shadow-xs transition-colors group-hover:text-accent">
                        <l.icon className="h-4 w-4" />
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block text-[13px] font-medium">{l.label}</span>
                        <span className="block text-[11.5px] text-subtle">{l.hint}</span>
                      </span>
                      <ArrowRight className="h-3.5 w-3.5 text-subtle transition-transform group-hover:translate-x-0.5" />
                    </Link>
                  </li>
                ))}
            </ul>
          </Card>
        </div>
      </div>
    </Page>
  );
}

/** Hands the question to AI Chat (prefilled there, not auto-sent). */
function AskBar() {
  const router = useRouter();
  const [q, setQ] = useState("");
  return (
    <form
      className="group relative mt-7 flex items-center gap-3 rounded-2xl border bg-surface px-4 shadow-lift transition-[border-color,box-shadow] focus-within:border-accent/50 focus-within:ring-4 focus-within:ring-accent/10"
      onSubmit={(e) => {
        e.preventDefault();
        router.push(q.trim() ? `/chat?q=${encodeURIComponent(q.trim())}` : "/chat");
      }}
    >
      <Sparkles className="h-[18px] w-[18px] shrink-0 text-accent" />
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Ask anything about your organization's knowledge…"
        aria-label="Ask a question"
        className="h-14 min-w-0 flex-1 bg-transparent text-[15px] placeholder:text-subtle focus:outline-none"
      />
      <button type="submit" className="hidden h-8 items-center gap-1.5 rounded-lg bg-fg px-3 text-[12.5px] font-medium text-bg transition-opacity hover:opacity-90 sm:inline-flex">
        Ask <CornerDownLeft className="h-3.5 w-3.5" />
      </button>
    </form>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className="num mt-2 font-mono text-[15px] font-medium">{value}</div>
    </div>
  );
}

function ProgressRing({ value, total }: { value: number; total: number }) {
  const r = 17;
  const c = 2 * Math.PI * r;
  const pct = total ? value / total : 0;
  return (
    <div className="relative grid h-11 w-11 shrink-0 place-items-center">
      <svg viewBox="0 0 40 40" className="absolute inset-0 -rotate-90">
        <circle cx="20" cy="20" r={r} fill="none" stroke="hsl(var(--border))" strokeWidth="3" />
        <circle cx="20" cy="20" r={r} fill="none" stroke="hsl(var(--accent))" strokeWidth="3" strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - pct)} className="transition-[stroke-dashoffset] duration-700" />
      </svg>
      <span className="num font-mono text-[11px] font-semibold">
        {value}/{total}
      </span>
    </div>
  );
}

function greeting() {
  const h = new Date().getHours();
  return h < 12 ? "morning" : h < 18 ? "afternoon" : "evening";
}
