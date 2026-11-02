"use client";

import { ArrowLeft, Database, Gavel, History, LayoutDashboard, MessageSquareCode, ScrollText, ShieldCheck, Table2 } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";

import { ApprovalsPanel, usePendingApprovals } from "@/components/connectivity/approvals-panel";
import { ConnectionActions, ConnectionOverview, useConnectionPerms } from "@/components/connectivity/connection-overview";
import { ConnectionGrants } from "@/components/connectivity/grants-panel";
import { QueryConsole } from "@/components/connectivity/query-console";
import { QueryLogTable } from "@/components/connectivity/query-log";
import { SyncRunsTable } from "@/components/connectivity/runs-table";
import { SchemaSelection } from "@/components/connectivity/schema-selection";
import { AccessModeBadge, EngineTile, ProviderStatusBadge } from "@/components/connectivity/shared";
import type { DbConnection } from "@/components/connectivity/types";
import { usePageCrumb } from "@/components/shell/breadcrumb";
import { Button } from "@/components/ui/button";
import { Card, EmptyState, Page, PageHeader, Skeleton, StatusBadge, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { ApiError } from "@/lib/api";
import { useApi } from "@/lib/hooks";

const TABS = ["overview", "schema", "query", "runs", "log", "permissions", "approvals"] as const;
type Tab = (typeof TABS)[number];

export default function ConnectionDetailPage() {
  return (
    <Suspense fallback={null}>
      <ConnectionDetail />
    </Suspense>
  );
}

function CountPill({ n, tone = "neutral" }: { n: number; tone?: "neutral" | "warn" }) {
  return (
    <span
      className={
        tone === "warn"
          ? "num ml-0.5 rounded-full bg-warn/10 px-1.5 font-mono text-[10.5px] font-semibold text-warn ring-1 ring-inset ring-warn/25"
          : "num ml-0.5 rounded-full bg-surface-2 px-1.5 font-mono text-[10.5px] font-medium text-muted ring-1 ring-inset ring-border"
      }
    >
      {n}
    </span>
  );
}

function ConnectionDetail() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const params = useSearchParams();
  const conn = useApi<DbConnection>(id ? `/database-connections/${id}` : null, undefined, {
    refetchInterval: (q) => ((q.state.data as DbConnection | undefined)?.status === "syncing" ? 4000 : false),
  });
  const c = conn.data;
  usePageCrumb(c?.name);
  const perms = useConnectionPerms(c);
  const pending = usePendingApprovals(id);
  const requested = params.get("tab") as Tab | null;
  const tab: Tab = requested && TABS.includes(requested) && (requested !== "permissions" || perms.grants) ? requested : "overview";
  const setTab = (t: string) => router.replace(t === "overview" ? `/database-connections/${id}` : `/database-connections/${id}?tab=${t}`, { scroll: false });

  if (conn.isLoading) {
    return (
      <Page>
        <Skeleton className="mb-3 h-3 w-40" />
        <Skeleton className="mb-3 h-8 w-80" />
        <Skeleton className="mb-8 h-4 w-96 max-w-full" />
        <Skeleton className="mb-6 h-10 w-full" />
        <div className="grid gap-6 lg:grid-cols-3">
          <Skeleton className="h-80 rounded-xl lg:col-span-2" />
          <Skeleton className="h-80 rounded-xl" />
        </div>
      </Page>
    );
  }

  if (!c) {
    const notFound = conn.error instanceof ApiError && (conn.error.status === 404 || conn.error.status === 422);
    return (
      <Page>
        <Card>
          <EmptyState
            icon={<Database className="h-5 w-5" />}
            title={notFound ? "Connection not found" : "Couldn't load this connection"}
            description={notFound ? "It may have been disconnected, or you no longer have access to it." : conn.error?.message}
            action={
              <Button variant="secondary" onClick={() => router.push("/database-connections")}>
                <ArrowLeft className="h-4 w-4" /> All connections
              </Button>
            }
          />
        </Card>
      </Page>
    );
  }

  return (
    <Page>
      <PageHeader
        eyebrow={
          <Link href="/database-connections" className="inline-flex items-center gap-1 transition-colors hover:text-fg">
            Database connections
          </Link>
        }
        title={
          <span className="flex flex-wrap items-center gap-x-3 gap-y-2">
            <span className="min-w-0 break-words">{c.name}</span>
            <StatusBadge status={c.status} className="translate-y-[1px]" />
          </span>
        }
        description={
          <span className="mt-1 flex flex-wrap items-center gap-x-2.5 gap-y-2 text-[13px]">
            <span className="inline-flex items-center gap-2">
              <EngineTile category={c.category} size="sm" />
              <span className="font-mono text-[12.5px] tracking-[-0.01em] text-fg">{c.provider_label}</span>
            </span>
            <ProviderStatusBadge status={c.provider_status} />
            <AccessModeBadge mode={c.access_mode} />
            <span className="hidden h-3.5 w-px bg-border sm:block" aria-hidden />
            <span className="min-w-0 max-w-full truncate font-mono text-[12px] tracking-[-0.01em] text-subtle" title={c.endpoint}>
              {c.endpoint}
            </span>
          </span>
        }
        actions={<ConnectionActions connection={c} />}
      />

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList variant="line" className="mb-6 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          <TabsTrigger value="overview">
            <LayoutDashboard className="h-3.5 w-3.5" /> Overview
          </TabsTrigger>
          <TabsTrigger value="schema">
            <Table2 className="h-3.5 w-3.5" /> Schema &amp; tables
            {c.table_count ? <CountPill n={c.selected_tables.length} /> : null}
          </TabsTrigger>
          <TabsTrigger value="query">
            <MessageSquareCode className="h-3.5 w-3.5" /> Query console
          </TabsTrigger>
          <TabsTrigger value="approvals">
            <Gavel className="h-3.5 w-3.5" /> Approvals
            {pending ? <CountPill n={pending} tone="warn" /> : null}
          </TabsTrigger>
          <TabsTrigger value="log">
            <ScrollText className="h-3.5 w-3.5" /> Query log
          </TabsTrigger>
          <TabsTrigger value="runs">
            <History className="h-3.5 w-3.5" /> Sync history
          </TabsTrigger>
          {perms.grants ? (
            <TabsTrigger value="permissions">
              <ShieldCheck className="h-3.5 w-3.5" /> Permissions
            </TabsTrigger>
          ) : null}
        </TabsList>

        <TabsContent value="overview" className="animate-fade-in focus-visible:outline-none">
          <ConnectionOverview connection={c} onGoTo={setTab} />
        </TabsContent>
        <TabsContent value="schema" className="animate-fade-in focus-visible:outline-none">
          <SchemaSelection connection={c} />
        </TabsContent>
        <TabsContent value="query" className="animate-fade-in focus-visible:outline-none">
          <QueryConsole connection={c} />
        </TabsContent>
        <TabsContent value="runs" className="animate-fade-in focus-visible:outline-none">
          <SyncRunsTable path={`/database-connections/${c.id}/runs`} emptyHint={c.access_mode === "live" ? "This connection is in live-query mode, so nothing is synchronised. Switch to ingest or hybrid to sync tables." : undefined} />
        </TabsContent>
        <TabsContent value="log" className="animate-fade-in focus-visible:outline-none">
          <QueryLogTable sourceId={c.id} />
        </TabsContent>
        {perms.grants ? (
          <TabsContent value="permissions" className="animate-fade-in focus-visible:outline-none">
            <ConnectionGrants connection={c} />
          </TabsContent>
        ) : null}
        <TabsContent value="approvals" className="animate-fade-in focus-visible:outline-none">
          <ApprovalsPanel sourceId={c.id} title={c.name} />
        </TabsContent>
      </Tabs>
    </Page>
  );
}
