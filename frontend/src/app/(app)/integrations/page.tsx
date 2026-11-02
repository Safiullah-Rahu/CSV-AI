"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, ArrowUpRight, Cable, CheckCircle2, CircleDashed, Globe, History, KeyRound, Lock, MoreHorizontal, Pause, Pencil, Play, PlugZap, Plus, RefreshCw, Settings2, Trash2 } from "lucide-react";
import Link from "next/link";
import { useCallback, useMemo, useState } from "react";
import { toast } from "sonner";

import { INTEGRATION_CATEGORIES, IntegrationDialog, integrationIcon } from "@/components/connectivity/integration-dialog";
import { OAuthSetupNotice, blockedByOAuthSetup, useOAuthConnect, useOAuthReturn } from "@/components/connectivity/oauth";
import { SyncRunsTable } from "@/components/connectivity/runs-table";
import { GatedButton, Notice, ProviderStatusBadge, When, formatJson, intervalLabel, invalidatePrefix, useJobWatcher, useKnowledgeBases } from "@/components/connectivity/shared";
import type { Integration, IntegrationCatalogItem, TestResult } from "@/components/connectivity/types";
import { Button } from "@/components/ui/button";
import { type Column, DataTable } from "@/components/ui/data-table";
import { ConfirmDialog, Dialog, DialogContent } from "@/components/ui/dialog";
import { Badge, Checkbox, EmptyState, Menu, MenuContent, MenuItem, MenuSeparator, MenuTrigger, Page, PageHeader, SectionTitle, Skeleton, Stat, StatGroup, StatusBadge, Tip } from "@/components/ui/primitives";
import { del, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatDate } from "@/lib/utils";

const CATEGORY_ORDER = ["web", "storage", "development", "document_management", "collaboration"];

export default function IntegrationsPage() {
  const { can } = useAuth();
  const canManage = can("org.integrations");
  const catalog = useApi<IntegrationCatalogItem[]>("/integrations/catalog", undefined, { staleTime: 5 * 60_000 });
  const list = useApi<Integration[]>("/integrations", undefined, {
    refetchInterval: (q) => ((q.state.data as Integration[] | undefined)?.some((i) => i.status === "syncing") ? 4000 : 60_000),
  });
  const kbs = useKnowledgeBases();
  const [creating, setCreating] = useState<IntegrationCatalogItem | null>(null);
  const [editing, setEditing] = useState<Integration | null>(null);
  const [runsFor, setRunsFor] = useState<Integration | null>(null);
  const [deleting, setDeleting] = useState<Integration | null>(null);
  const [testResult, setTestResult] = useState<{ integration: Integration; result: TestResult } | null>(null);
  const [setupFor, setSetupFor] = useState<IntegrationCatalogItem | null>(null);

  const qc = useQueryClient();
  // Back from a vendor consent screen: /integrations?oauth=connected|error&integration=…&message=…
  useOAuthReturn(
    useCallback(
      (r: { status: "connected" | "error"; message: string | null }) => {
        if (r.status === "connected") toast.success("Integration connected", { description: "Authorization succeeded and the test passed. Run Sync to index content." });
        else toast.error("Authorization did not complete", { description: r.message ?? "Try connecting again from the integration's menu." });
        invalidatePrefix(qc, "/integrations", "/connections");
      },
      [qc],
    ),
  );

  const kbName = (id: string | null) => (!id ? null : !kbs.data ? "…" : kbs.data.find((k) => k.id === id)?.name ?? "Not visible to you");
  // Planned providers and OAuth-only providers whose app is not configured cannot be connected.
  const available = (catalog.data ?? []).filter((c) => c.status !== "planned" && !blockedByOAuthSetup(c));
  const firstAvailable = available[0] ?? null;
  const configured = list.data ?? [];
  const counts = {
    connected: configured.filter((i) => i.status === "connected").length,
    failing: configured.filter((i) => i.status === "failed").length,
    untested: configured.filter((i) => i.status === "pending").length,
  };
  const catalogCounts = {
    tested: (catalog.data ?? []).filter((c) => c.status === "tested").length,
    beta: (catalog.data ?? []).filter((c) => c.status === "beta").length,
    planned: (catalog.data ?? []).filter((c) => c.status === "planned").length,
    setup: (catalog.data ?? []).filter((c) => c.oauth && !c.oauth.configured).length,
  };

  const grouped = useMemo(() => {
    const by: Record<string, IntegrationCatalogItem[]> = {};
    (catalog.data ?? []).forEach((c) => (by[c.category] ||= []).push(c));
    const rank: Record<string, number> = { tested: 0, beta: 1, planned: 2 };
    Object.values(by).forEach((arr) => arr.sort((a, b) => (rank[a.status] ?? 3) - (rank[b.status] ?? 3) || a.label.localeCompare(b.label)));
    return [...CATEGORY_ORDER, ...Object.keys(by).filter((k) => !CATEGORY_ORDER.includes(k))].filter((k) => by[k]?.length).map((k) => ({ key: k, items: by[k] }));
  }, [catalog.data]);

  const columns: Column<Integration>[] = [
    {
      key: "name",
      header: "Integration",
      sortValue: (i) => i.name.toLowerCase(),
      cell: (i) => {
        const Icon = integrationIcon(i.provider);
        return (
          <div className="flex min-w-[240px] items-center gap-3">
            <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border bg-surface-2 text-muted shadow-[inset_0_1px_0_hsl(var(--highlight))]">
              <Icon className="h-4 w-4" />
            </span>
            <div className="min-w-0">
              <div className="truncate font-medium tracking-[-0.01em]">{i.name}</div>
              <div className="mt-0.5 flex flex-wrap items-center gap-1.5">
                <span className="font-mono text-[11.5px] tracking-[-0.01em] text-muted">{i.provider_label}</span>
                <ProviderStatusBadge status={i.provider_status} className="px-1.5 py-[1px] text-[10.5px]" />
                {i.oauth ? (
                  i.oauth.connected ? (
                    <Tip content={i.oauth.account ? `Authorized with ${i.oauth.vendor} as ${i.oauth.account}` : `Authorized with ${i.oauth.vendor}`}>
                      <span tabIndex={0}>
                        <Badge tone="ok" className="px-1.5 py-[1px] text-[10.5px]">
                          <KeyRound className="h-2.5 w-2.5" /> authorized
                        </Badge>
                      </span>
                    </Tip>
                  ) : (
                    <Badge tone="warn" className="px-1.5 py-[1px] text-[10.5px]">
                      not authorized
                    </Badge>
                  )
                ) : i.has_secret ? (
                  <Badge tone="ok" className="px-1.5 py-[1px] text-[10.5px]">
                    <KeyRound className="h-2.5 w-2.5" /> secret stored
                  </Badge>
                ) : null}
              </div>
            </div>
          </div>
        );
      },
    },
    {
      key: "status",
      header: "Status",
      sortValue: (i) => i.status,
      cell: (i) => (
        <div className="flex flex-wrap items-center gap-1">
          {i.status === "pending" ? (
            <Tip content="Saved, but not verified yet. Run Test to check access; it shows as connected only after a successful Test or Sync.">
              <span tabIndex={0}>
                <Badge>
                  <span className="h-1.5 w-1.5 shrink-0 rounded-full border border-subtle" aria-hidden /> not tested yet
                </Badge>
              </span>
            </Tip>
          ) : (
            <StatusBadge status={i.status} />
          )}
          {!i.enabled ? <Badge>disabled</Badge> : null}
          {i.sync_paused ? <Badge tone="warn">paused</Badge> : null}
        </div>
      ),
    },
    {
      key: "kb",
      header: "Target KB",
      cell: (i) =>
        i.kb_id ? (
          <Link href={`/knowledge-bases/${i.kb_id}`} className="block max-w-[180px] truncate transition-colors hover:text-accent" onClick={(e) => e.stopPropagation()}>
            {kbName(i.kb_id)}
          </Link>
        ) : (
          <span className="whitespace-nowrap text-warn">None — sync disabled</span>
        ),
    },
    {
      key: "schedule",
      header: "Schedule",
      cell: (i) => (
        <div className="whitespace-nowrap">
          <div>{intervalLabel(i.sync_interval_minutes)}</div>
          {i.next_sync_at && !i.sync_paused ? (
            <div className="font-mono text-[11px] text-subtle" title={formatDate(i.next_sync_at)}>
              next {formatDate(i.next_sync_at)}
            </div>
          ) : null}
        </div>
      ),
    },
    {
      key: "last",
      header: "Last sync",
      sortValue: (i) => i.last_sync_at ?? "",
      cell: (i) => <When at={i.last_sync_at} />,
    },
    {
      key: "error",
      header: "Last error",
      cell: (i) =>
        i.last_error ? (
          <Tip content={<span className="whitespace-pre-wrap break-words font-mono text-[11px]">{i.last_error}</span>}>
            <span tabIndex={0} className="block max-w-[240px] truncate font-mono text-[11.5px] text-danger">
              {i.last_error}
            </span>
          </Tip>
        ) : (
          <span className="text-subtle">—</span>
        ),
    },
    {
      key: "actions",
      header: <span className="sr-only">Actions</span>,
      cell: (i) => (
        <RowActions
          integration={i}
          canManage={canManage}
          onEdit={() => setEditing(i)}
          onRuns={() => setRunsFor(i)}
          onDelete={() => setDeleting(i)}
          onTested={(result) => setTestResult({ integration: i, result })}
        />
      ),
      className: "w-[150px]",
    },
  ];

  return (
    <Page>
      <PageHeader
        eyebrow="Connectivity"
        title="Integrations"
        description="Keep knowledge bases in sync with websites, feeds, repositories, drives, wikis, chat and email. Changed content is re-indexed; removed content is deleted."
        actions={
          <GatedButton allowed={canManage && Boolean(firstAvailable)} reason={!canManage ? "Requires org.integrations" : "No integration provider is available"} onClick={() => setCreating(firstAvailable)}>
            <Plus className="h-4 w-4" /> Add integration
          </GatedButton>
        }
      />

      <div className="stagger space-y-10">
        {list.isLoading ? (
          <Skeleton className="h-[106px] rounded-xl" />
        ) : (
          <StatGroup cols={4}>
            <Stat label="Configured" value={configured.length} hint={`${configured.filter((i) => i.enabled).length} enabled`} icon={<Cable />} />
            <Stat label="Connected" value={counts.connected} hint="Verified by a successful Test or Sync" icon={<CheckCircle2 />} tone={counts.connected ? "ok" : undefined} />
            <Stat label="Not tested yet" value={counts.untested} hint={counts.untested ? "Run Test to verify access" : "Nothing waiting for a test"} icon={<CircleDashed />} tone={counts.untested ? "warn" : undefined} />
            <Stat label="Failing" value={counts.failing} hint={counts.failing ? "See the last error in the table" : "No failing integrations"} icon={<AlertTriangle />} tone={counts.failing ? "danger" : undefined} valueTone={counts.failing ? "danger" : undefined} />
          </StatGroup>
        )}

        <section aria-labelledby="configured-heading">
          <SectionTitle
            title={<span id="configured-heading">Configured integrations</span>}
            description={list.data ? `${list.data.length} configured · ${counts.connected} connected · ${counts.failing} failing` : "Loading…"}
          />
          <DataTable
            rows={list.data}
            columns={columns}
            rowKey={(i) => i.id}
            loading={list.isLoading}
            filterText={(i) => `${i.name} ${i.provider_label} ${i.status}`}
            searchPlaceholder="Search integrations…"
            empty={
              <EmptyState
                icon={<Cable className="h-5 w-5" />}
                title="No integrations yet"
                description="Pick a source from the catalog below. Website, sitemap, RSS and public GitLab projects need no credentials."
              />
            }
          />
        </section>

        <section aria-labelledby="catalog-heading">
          <SectionTitle
            title={<span id="catalog-heading">Catalog</span>}
            description="Status is honest: tested sources were verified end to end (hover the badge for how); beta sources are implemented and tested against stubbed vendor APIs, but not verified against the live service."
          />
          <div className="mb-5 flex flex-wrap items-center gap-x-5 gap-y-2 rounded-xl border border-dashed px-4 py-2.5 text-[12px] text-muted">
            <span className="inline-flex items-center gap-2">
              <ProviderStatusBadge status="tested" /> <span className="num font-mono text-[11.5px] text-subtle">{catalogCounts.tested}</span> verified end to end
            </span>
            <span className="inline-flex items-center gap-2">
              <ProviderStatusBadge status="beta" /> <span className="num font-mono text-[11.5px] text-subtle">{catalogCounts.beta}</span> implemented, not verified live
            </span>
            <span className="inline-flex items-center gap-2">
              <ProviderStatusBadge status="planned" /> <span className="num font-mono text-[11.5px] text-subtle">{catalogCounts.planned}</span> not connectable yet
            </span>
            {catalogCounts.setup ? (
              <span className="inline-flex items-center gap-2">
                <NeedsSetupBadge /> <span className="num font-mono text-[11.5px] text-subtle">{catalogCounts.setup}</span> OAuth app not configured on this deployment
              </span>
            ) : null}
          </div>
          {catalog.isLoading ? (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {Array.from({ length: 8 }).map((_, i) => (
                <Skeleton key={i} className="h-[148px] rounded-xl" />
              ))}
            </div>
          ) : (
            <div className="space-y-8">
              {grouped.map((g) => (
                <section key={g.key} aria-labelledby={`icat-${g.key}`}>
                  <h3 id={`icat-${g.key}`} className="label mb-3 flex items-center gap-3">
                    {INTEGRATION_CATEGORIES[g.key] ?? g.key.replace(/_/g, " ")}
                    <span className="h-px flex-1 bg-border" aria-hidden />
                    <span className="num">{g.items.length}</span>
                  </h3>
                  <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                    {g.items.map((item) => (
                      <CatalogCard key={item.key} item={item} count={configured.filter((i) => i.provider === item.key).length} canManage={canManage} onConnect={() => setCreating(item)} onSetup={() => setSetupFor(item)} />
                    ))}
                  </div>
                </section>
              ))}
            </div>
          )}
        </section>
      </div>

      {canManage ? (
        <>
          <IntegrationDialog open={Boolean(creating)} onOpenChange={(v) => !v && setCreating(null)} provider={creating} catalog={available} />
          <IntegrationDialog open={Boolean(editing)} onOpenChange={(v) => !v && setEditing(null)} provider={null} existing={editing} catalog={catalog.data} />
          <DeleteIntegrationDialog integration={deleting} onClose={() => setDeleting(null)} />
        </>
      ) : null}

      {canManage ? (
        <Dialog open={Boolean(setupFor)} onOpenChange={(v) => !v && setSetupFor(null)}>
          {setupFor?.oauth ? (
            <DialogContent title={`Set up ${setupFor.label}`} description="Configured once per deployment by a platform operator; every organization then connects its own account.">
              <OAuthSetupNotice app={setupFor.oauth} label={setupFor.label} />
            </DialogContent>
          ) : null}
        </Dialog>
      ) : null}

      <Dialog open={Boolean(runsFor)} onOpenChange={(v) => !v && setRunsFor(null)}>
        {runsFor ? (
          <DialogContent wide className="max-w-5xl" title={`Sync runs · ${runsFor.name}`} description="The last 50 runs, newest first.">
            <SyncRunsTable path={`/integrations/${runsFor.id}/runs`} showFailed />
          </DialogContent>
        ) : null}
      </Dialog>

      <Dialog open={Boolean(testResult)} onOpenChange={(v) => !v && setTestResult(null)}>
        {testResult ? (
          <DialogContent title={`Test · ${testResult.integration.name}`} description={testResult.integration.provider_label}>
            <TestResultView result={testResult.result} />
          </DialogContent>
        ) : null}
      </Dialog>
    </Page>
  );
}

/** An OAuth app the platform operator has not configured yet (state, not decoration). */
function NeedsSetupBadge({ className }: { className?: string }) {
  return (
    <Badge className={className}>
      <Settings2 className="h-2.5 w-2.5" /> Needs setup
    </Badge>
  );
}

/**
 * One catalog entry. Planned entries and OAuth-only entries whose app is not configured are
 * deliberately flat (dashed, no shadow, no Connect button) so they never read as connectable.
 */
function CatalogCard({ item, count, canManage, onConnect, onSetup }: { item: IntegrationCatalogItem; count: number; canManage: boolean; onConnect: () => void; onSetup: () => void }) {
  const Icon = integrationIcon(item.key);
  const planned = item.status === "planned";
  const blocked = blockedByOAuthSetup(item);
  const flat = planned || blocked;
  const tokenFallback = Boolean(item.oauth && !item.oauth.configured && !blocked);
  const needsCreds = item.auth && item.auth !== "none";
  const badge = <ProviderStatusBadge status={item.status} />;
  return (
    <div
      className={cn(
        "group relative flex min-h-[148px] flex-col rounded-xl border p-4 transition-[border-color,box-shadow] duration-200",
        flat ? "border-dashed bg-transparent" : "bg-surface shadow-card hover:border-border-strong hover:shadow-lift",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <span
          className={cn(
            "grid h-10 w-10 shrink-0 place-items-center rounded-xl border",
            flat ? "border-dashed text-subtle" : "bg-surface-2 text-fg shadow-[inset_0_1px_0_hsl(var(--highlight))]",
          )}
        >
          <Icon className="h-[18px] w-[18px]" />
        </span>
        <span className="flex flex-wrap items-center justify-end gap-1">
          {blocked ? <NeedsSetupBadge /> : null}
          {item.verified ? (
            <Tip content={<span className="block max-w-[260px] whitespace-normal">Verified: {item.verified}</span>}>
              <span tabIndex={0} aria-label={`Verified: ${item.verified}`}>
                {badge}
              </span>
            </Tip>
          ) : (
            badge
          )}
        </span>
      </div>
      <div className="mt-3.5 min-w-0">
        <div className={cn("truncate text-[14px] font-medium tracking-[-0.01em]", flat && "text-muted")}>{item.label}</div>
        <div className="mt-1 flex items-center gap-1.5 text-[12px] text-subtle" title={needsCreds ? `Requires ${item.auth}` : undefined}>
          {needsCreds ? <KeyRound className="h-3 w-3 shrink-0" /> : <Globe className="h-3 w-3 shrink-0" />}
          <span className="truncate">{needsCreds ? <>Requires <span className="font-mono text-[11.5px]">{item.auth}</span></> : "No credentials needed"}</span>
        </div>
        {tokenFallback ? <div className="mt-1 truncate text-[11.5px] text-subtle">OAuth needs setup on this deployment; the token option works now.</div> : null}
      </div>
      <div className={cn("mt-auto flex items-center justify-between gap-2", planned ? "pt-3 [&>span]:w-full" : "pt-4")}>
        {planned ? (
          <span className="inline-flex min-w-0 items-center gap-1.5 border-t border-dashed pt-3 text-[12px] text-subtle" title={`Coming soon · requires ${item.auth}`}>
            <Lock className="h-3 w-3 shrink-0" />
            Coming soon · not connectable yet
          </span>
        ) : blocked ? (
          <>
            <span className="min-w-0 truncate text-[12px] text-subtle" title={`${item.oauth?.vendor} OAuth app not configured on this deployment`}>
              {canManage ? "OAuth app needed" : "Not configured"}
            </span>
            <GatedButton size="xs" variant="secondary" allowed={canManage} perm="org.integrations" onClick={onSetup}>
              <Settings2 className="h-3 w-3" /> Setup
            </GatedButton>
          </>
        ) : (
          <>
            <span className="font-mono text-[11px] text-subtle">{count ? `${count} configured` : ""}</span>
            <GatedButton size="xs" variant="secondary" allowed={canManage} perm="org.integrations" onClick={onConnect}>
              <Plus className="h-3 w-3" /> Connect
            </GatedButton>
          </>
        )}
      </div>
    </div>
  );
}

function RowActions({ integration: i, canManage, onEdit, onRuns, onDelete, onTested }: { integration: Integration; canManage: boolean; onEdit: () => void; onRuns: () => void; onDelete: () => void; onTested: (r: TestResult) => void }) {
  const qc = useQueryClient();
  const watch = useJobWatcher();
  const test = useMutation({
    mutationFn: () => post<TestResult>(`/integrations/${i.id}/test`),
    onSuccess: (r) => {
      invalidatePrefix(qc, "/integrations", "/connections");
      onTested(r);
    },
  });
  const sync = useMutation({
    mutationFn: (mode: "incremental" | "full") => post<{ job_id: string }>(`/integrations/${i.id}/sync`, { mode }),
    onSuccess: (r, mode) => {
      invalidatePrefix(qc, "/integrations");
      watch(r.job_id, { label: `${mode === "full" ? "Full sync" : "Sync"} · ${i.name}`, invalidate: ["/integrations", "/connections", "/documents"] });
    },
  });
  const pause = useMutation({
    mutationFn: (action: "pause" | "resume") => post(`/integrations/${i.id}/${action}`),
    onSuccess: (_r, action) => {
      toast.success(action === "pause" ? `Scheduled sync paused for ${i.name}` : `Scheduled sync resumed for ${i.name}`);
      invalidatePrefix(qc, "/integrations");
    },
  });
  const connect = useOAuthConnect();
  const needsAuth = Boolean(i.oauth && !i.oauth.connected);
  const syncBlocked = !canManage ? "Requires org.integrations" : !i.enabled ? "Integration is disabled" : !i.kb_id ? "Choose a target knowledge base (Edit) first" : needsAuth ? `Authorize with ${i.oauth?.vendor} first` : "";

  return (
    <div className="flex items-center justify-end gap-1" onClick={(e) => e.stopPropagation()}>
      {needsAuth ? (
        <GatedButton size="xs" variant="secondary" allowed={canManage} perm="org.integrations" onClick={() => connect.mutate(i.id)} loading={connect.isPending} aria-label={`Connect ${i.name} with ${i.oauth?.vendor}`}>
          <ArrowUpRight className="h-3 w-3" /> Connect
        </GatedButton>
      ) : (
        <GatedButton size="xs" variant="secondary" allowed={canManage} perm="org.integrations" onClick={() => test.mutate()} loading={test.isPending}>
          <PlugZap className="h-3 w-3" /> Test
        </GatedButton>
      )}
      <GatedButton size="xs" variant="secondary" allowed={!syncBlocked} reason={syncBlocked} onClick={() => sync.mutate("incremental")} loading={sync.isPending}>
        <RefreshCw className="h-3 w-3" /> Sync
      </GatedButton>
      <Menu>
        <MenuTrigger asChild>
          <Button size="icon" variant="ghost" aria-label={`More actions for ${i.name}`} className="h-7 w-7">
            <MoreHorizontal className="h-4 w-4" />
          </Button>
        </MenuTrigger>
        <MenuContent className="min-w-[210px]">
          <MenuItem disabled={Boolean(syncBlocked)} onSelect={() => sync.mutate("full")}>
            <RefreshCw className="h-3.5 w-3.5" /> <span className="flex-1">Full re-sync</span>
          </MenuItem>
          <MenuItem disabled={!canManage} onSelect={() => pause.mutate(i.sync_paused ? "resume" : "pause")}>
            {i.sync_paused ? <Play className="h-3.5 w-3.5" /> : <Pause className="h-3.5 w-3.5" />}
            <span className="flex-1">{i.sync_paused ? "Resume schedule" : "Pause schedule"}</span>
          </MenuItem>
          <MenuItem onSelect={onRuns}>
            <History className="h-3.5 w-3.5" /> Runs history
          </MenuItem>
          {i.oauth ? (
            <MenuItem disabled={!canManage} onSelect={() => connect.mutate(i.id)}>
              <ArrowUpRight className="h-3.5 w-3.5" /> <span className="flex-1">{i.oauth.connected ? `Reconnect with ${i.oauth.vendor}` : `Connect with ${i.oauth.vendor}`}</span>
            </MenuItem>
          ) : null}
          <MenuSeparator />
          <MenuItem disabled={!canManage} onSelect={onEdit}>
            <Pencil className="h-3.5 w-3.5" /> <span className="flex-1">Edit</span>
            {!canManage ? <span className="text-2xs text-subtle">org.integrations</span> : null}
          </MenuItem>
          <MenuItem danger disabled={!canManage} onSelect={onDelete}>
            <Trash2 className="h-3.5 w-3.5" /> <span className="flex-1">Delete…</span>
            {!canManage ? <span className="text-2xs text-subtle">org.integrations</span> : null}
          </MenuItem>
        </MenuContent>
      </Menu>
    </div>
  );
}

function TestResultView({ result }: { result: TestResult }) {
  if (!result.ok) {
    return (
      <Notice tone="danger" title={`Test failed${result.code ? ` (${result.code})` : ""}`}>
        {result.error}
      </Notice>
    );
  }
  const details = Object.entries(result).filter(([k]) => k !== "ok");
  return (
    <div className="space-y-3">
      <Notice tone="ok" title="Connected">The source is reachable with the stored configuration. The integration is now marked connected.</Notice>
      {details.length ? (
        <dl className="space-y-2.5 text-[13px]">
          {details.map(([k, v]) => (
            <div key={k} className="grid grid-cols-[130px_1fr] gap-3">
              <dt className="label pt-0.5">{k.replace(/_/g, " ")}</dt>
              <dd className="min-w-0 break-words">
                {Array.isArray(v) ? (
                  <ul className="space-y-0.5">
                    {v.map((x, idx) => (
                      <li key={idx} className="truncate font-mono text-[12px]" title={String(x)}>
                        {String(x)}
                      </li>
                    ))}
                  </ul>
                ) : typeof v === "object" && v !== null ? (
                  <span className="font-mono text-[12px]">{formatJson(v)}</span>
                ) : (
                  String(v)
                )}
              </dd>
            </div>
          ))}
        </dl>
      ) : null}
    </div>
  );
}

function DeleteIntegrationDialog({ integration, onClose }: { integration: Integration | null; onClose: () => void }) {
  const qc = useQueryClient();
  const [deleteDocs, setDeleteDocs] = useState(false);
  const remove = useMutation({
    mutationFn: (i: Integration) => del(`/integrations/${i.id}`, { confirm_name: i.name, delete_synced_documents: deleteDocs }),
    onSuccess: (_r, i) => {
      toast.success(`Deleted “${i.name}”`, { description: deleteDocs ? "Synced documents were deleted too." : "Synced documents were kept." });
      invalidatePrefix(qc, "/integrations", "/connections", "/documents", "/knowledge-bases");
      setDeleteDocs(false);
      onClose();
    },
  });
  return (
    <ConfirmDialog
      open={Boolean(integration)}
      onOpenChange={(v) => {
        if (!v) {
          setDeleteDocs(false);
          onClose();
        }
      }}
      title="Delete integration"
      description="The integration and its sync schedule are removed. Documents it synchronised stay in the knowledge base unless you also delete them below."
      confirmLabel="Delete integration"
      requireText={integration?.name}
      loading={remove.isPending}
      onConfirm={() => {
        if (integration) remove.mutate(integration);
      }}
    >
      <label className="mb-4 flex cursor-pointer items-start gap-2.5 rounded-lg border p-3 text-[13px]">
        <span className="pt-0.5">
          <Checkbox checked={deleteDocs} onCheckedChange={setDeleteDocs} label="Also delete synced documents" />
        </span>
        <span>
          <span className="font-medium">Also delete synced documents</span>
          <span className="block text-xs text-muted">Removes every document this integration created, with chunks and embeddings. This cannot be undone.</span>
        </span>
      </label>
    </ConfirmDialog>
  );
}
