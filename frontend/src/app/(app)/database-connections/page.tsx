"use client";

import { Activity, AlertTriangle, ChevronRight, Database, Gavel, Plus, RefreshCw } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";

import { ApprovalsPanel, usePendingApprovals } from "@/components/connectivity/approvals-panel";
import { AddConnectionDialog } from "@/components/connectivity/connection-wizard";
import { AccessModeBadge, EngineTile, GatedButton, ProviderStatusBadge, When } from "@/components/connectivity/shared";
import type { DbConnection } from "@/components/connectivity/types";
import { type Column, DataTable } from "@/components/ui/data-table";
import { Badge, EmptyState, Page, PageHeader, Skeleton, Stat, StatGroup, StatusBadge, Tabs, TabsContent, TabsList, TabsTrigger, Tip } from "@/components/ui/primitives";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";

export default function DatabaseConnectionsPage() {
  return (
    <Suspense fallback={null}>
      <DatabaseConnections />
    </Suspense>
  );
}

function DatabaseConnections() {
  const { can } = useAuth();
  const router = useRouter();
  const params = useSearchParams();
  const tab = params.get("tab") === "approvals" ? "approvals" : "connections";
  const [adding, setAdding] = useState(false);
  const conns = useApi<DbConnection[]>("/database-connections", undefined, {
    refetchInterval: (q) => ((q.state.data as DbConnection[] | undefined)?.some((c) => c.status === "syncing") ? 5000 : 60_000),
  });
  const pending = usePendingApprovals();
  const canCreate = can("db.create") && can("db.credentials");
  const list = conns.data ?? [];
  const failed = list.filter((c) => c.status === "failed").length;
  const syncing = list.filter((c) => c.status === "syncing").length;

  const columns: Column<DbConnection>[] = [
    {
      key: "name",
      header: "Connection",
      sortValue: (c) => c.name.toLowerCase(),
      cell: (c) => (
        <div className="flex min-w-[240px] items-center gap-3">
          <EngineTile category={c.category} />
          <div className="min-w-0">
            <Link href={`/database-connections/${c.id}`} className="block truncate font-medium tracking-[-0.01em] transition-colors hover:text-accent" onClick={(e) => e.stopPropagation()}>
              {c.name}
            </Link>
            <div className="mt-0.5 flex items-center gap-1.5">
              <span className="truncate font-mono text-[11.5px] tracking-[-0.01em] text-muted">{c.provider_label}</span>
              <ProviderStatusBadge status={c.provider_status} className="px-1.5 py-[1px] text-[10.5px]" />
            </div>
          </div>
        </div>
      ),
    },
    {
      key: "endpoint",
      header: "Endpoint",
      cell: (c) => (
        <span className="block max-w-[240px] truncate font-mono text-[11.5px] tracking-[-0.01em] text-muted" title={c.endpoint}>
          {c.endpoint}
        </span>
      ),
    },
    {
      key: "status",
      header: "Status",
      sortValue: (c) => c.status,
      cell: (c) => (
        <div className="flex flex-wrap items-center gap-1">
          {c.last_error && c.status === "failed" ? (
            <Tip content={<span className="font-mono text-[11px]">{c.last_error}</span>}>
              <span tabIndex={0}>
                <StatusBadge status={c.status} />
              </span>
            </Tip>
          ) : (
            <StatusBadge status={c.status} />
          )}
          {c.sync_paused && c.access_mode !== "live" ? <Badge tone="warn">sync paused</Badge> : null}
          {!c.has_credentials && c.provider !== "sqlite" ? <Badge>no password</Badge> : null}
        </div>
      ),
    },
    { key: "mode", header: "Access", sortValue: (c) => c.access_mode, cell: (c) => <AccessModeBadge mode={c.access_mode} /> },
    {
      key: "tested",
      header: "Last tested",
      sortValue: (c) => c.last_tested_at ?? "",
      cell: (c) => <When at={c.last_tested_at} />,
    },
    {
      key: "synced",
      header: "Last synced",
      sortValue: (c) => c.last_sync_at ?? "",
      cell: (c) =>
        c.access_mode === "live" ? (
          <Tip content="Live-query mode: nothing is synchronised">
            <span tabIndex={0} className="font-mono text-[12px] text-subtle">
              live
            </span>
          </Tip>
        ) : (
          <When at={c.last_sync_at} />
        ),
    },
    {
      key: "tables",
      header: "Tables",
      sortValue: (c) => c.selected_tables.length,
      cell: (c) =>
        c.selected_tables.length ? (
          <Tip content={<span className="font-mono text-[11px]">{c.selected_tables.slice(0, 20).join(", ")}{c.selected_tables.length > 20 ? ` +${c.selected_tables.length - 20} more` : ""}</span>}>
            <span className="inline-flex flex-col gap-1.5 whitespace-nowrap" tabIndex={0}>
              <span className="num font-mono text-[12px]">
                {c.selected_tables.length}
                <span className="text-subtle"> / {c.table_count}</span>
              </span>
              <span className="h-[3px] w-14 overflow-hidden rounded-full bg-surface-2 ring-1 ring-inset ring-border" aria-hidden>
                <span className="block h-full rounded-full bg-fg/35" style={{ width: `${Math.min(100, (c.selected_tables.length / Math.max(1, c.table_count)) * 100)}%` }} />
              </span>
            </span>
          </Tip>
        ) : (
          <span className="whitespace-nowrap text-[12.5px] text-subtle">{c.table_count ? <span className="num font-mono text-[12px]">0 / {c.table_count}</span> : "not discovered"}</span>
        ),
    },
    {
      key: "go",
      header: <span className="sr-only">Open</span>,
      className: "w-8 pl-0 pr-3",
      cell: () => <ChevronRight className="h-4 w-4 text-subtle opacity-0 transition-[opacity,transform] group-hover/row:translate-x-0.5 group-hover/row:opacity-100" aria-hidden />,
    },
  ];

  const setTab = (t: string) => router.replace(t === "approvals" ? "/database-connections?tab=approvals" : "/database-connections", { scroll: false });
  const addButton = (
    <GatedButton allowed={canCreate} reason={!can("db.create") ? "Requires db.create" : "Requires db.credentials"} onClick={() => setAdding(true)}>
      <Plus className="h-4 w-4" /> Add connection
    </GatedButton>
  );

  return (
    <Page>
      <PageHeader
        eyebrow="Connectivity"
        title="Database connections"
        description="Query production databases read-only, or synchronise selected tables into knowledge bases. Credentials are encrypted and never shown again."
        actions={addButton}
      />

      <div className="stagger space-y-8">
        {conns.isLoading ? (
          <Skeleton className="h-[106px] rounded-xl" />
        ) : (
          <StatGroup cols={4}>
            <Stat label="Connections" value={list.length} hint={`${list.filter((c) => c.enabled).length} enabled`} icon={<Database />} />
            <Stat label="Healthy" value={list.filter((c) => c.status === "healthy").length} hint={`${list.filter((c) => c.status === "unknown").length} not tested yet`} icon={<Activity />} tone="ok" />
            <Stat label="Failing" value={failed} hint={failed ? "Open a connection to see the error" : "No failing connections"} icon={<AlertTriangle />} tone={failed ? "danger" : undefined} valueTone={failed ? "danger" : undefined} />
            <Stat label="Syncing" value={syncing} hint={`${list.filter((c) => c.access_mode !== "live").length} ingest or hybrid`} icon={<RefreshCw className={syncing ? "animate-spin [animation-duration:2.4s]" : undefined} />} tone={syncing ? "info" : undefined} />
          </StatGroup>
        )}

        <Tabs value={tab} onValueChange={setTab}>
          <TabsList variant="line" className="mb-5">
            <TabsTrigger value="connections">
              <Database className="h-3.5 w-3.5" /> Connections
              {list.length ? <span className="num ml-0.5 rounded-full bg-surface-2 px-1.5 font-mono text-[10.5px] font-medium text-muted ring-1 ring-inset ring-border">{list.length}</span> : null}
            </TabsTrigger>
            <TabsTrigger value="approvals">
              <Gavel className="h-3.5 w-3.5" /> Approvals
              {pending ? <span className="num ml-0.5 rounded-full bg-warn/10 px-1.5 font-mono text-[10.5px] font-semibold text-warn ring-1 ring-inset ring-warn/25">{pending}</span> : null}
            </TabsTrigger>
          </TabsList>
          <TabsContent value="connections" className="animate-fade-in focus-visible:outline-none">
            <DataTable
              rows={conns.data}
              columns={columns}
              rowKey={(c) => c.id}
              loading={conns.isLoading}
              filterText={(c) => `${c.name} ${c.provider_label} ${c.endpoint} ${c.status} ${c.selected_tables.join(" ")}`}
              searchPlaceholder="Search connections, engines, tables…"
              onRowClick={(c) => router.push(`/database-connections/${c.id}`)}
              empty={
                <EmptyState
                  icon={<Database className="h-5 w-5" />}
                  title="No database connections yet"
                  description="Connect PostgreSQL, MySQL, SQL Server, SQLite and more. Start in live-query mode — nothing is copied until you choose to synchronise tables."
                  action={addButton}
                />
              }
            />
          </TabsContent>
          <TabsContent value="approvals" className="animate-fade-in focus-visible:outline-none">
            <ApprovalsPanel />
          </TabsContent>
        </Tabs>
      </div>

      {canCreate ? <AddConnectionDialog open={adding} onOpenChange={setAdding} /> : null}
    </Page>
  );
}
