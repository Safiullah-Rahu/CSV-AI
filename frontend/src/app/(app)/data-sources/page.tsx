"use client";

import { BarChart3, Database, ExternalLink, Info, MessageSquareText, Rows3, Sparkles, Table2, UploadCloud } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";

import { AskPanel } from "@/components/knowledge/dataset-ask";
import { AnalysisPanel, PreviewPanel, ProfilePanel, SummaryPanel } from "@/components/knowledge/dataset-panels";
import { FileIcon, Guarded, kbAbilities, MonoChip, SearchField, usePerms } from "@/components/knowledge/shared";
import type { Dataset, DatasetDetail, KBOptions, KnowledgeBase } from "@/components/knowledge/types";
import { UploadDialog } from "@/components/knowledge/upload-dialog";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/form";
import { Badge, Card, EmptyState, Page, PageHeader, Skeleton, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { ApiError } from "@/lib/api";
import { useApi } from "@/lib/hooks";
import { cn, formatDate, formatNumber, timeAgo } from "@/lib/utils";

const TABS = ["profile", "preview", "analysis", "summary", "ask"] as const;
type Tab = (typeof TABS)[number];

export default function DataSourcesPage() {
  return (
    <Suspense
      fallback={
        <Page>
          <Skeleton className="mb-8 h-24 w-full max-w-xl" />
          <Skeleton className="h-96 rounded-xl" />
        </Page>
      }
    >
      <Datasets />
    </Suspense>
  );
}

function Datasets() {
  const router = useRouter();
  const pathname = usePathname();
  const search = useSearchParams();
  const perms = usePerms();
  const selectedId = search.get("ds");
  const rawTab = search.get("tab");
  const tab: Tab = (TABS as readonly string[]).includes(rawTab || "") ? (rawTab as Tab) : "profile";
  const [q, setQ] = useState("");
  const [kbFilter, setKbFilter] = useState("");
  const [uploadOpen, setUploadOpen] = useState(false);

  const list = useApi<Dataset[]>("/datasets");
  const kbs = useApi<KnowledgeBase[]>("/knowledge-bases");
  const options = useApi<KBOptions>("/knowledge-bases/options", undefined, { staleTime: 5 * 60_000 });
  const canUpload = (kbs.data ?? []).some((k) => kbAbilities(k, perms).upload);
  const tabularExts = (options.data?.formats ?? []).filter((f) => f.family === "tabular").map((f) => f.ext.toUpperCase());

  const setParams = (patch: Record<string, string | null>) => {
    const params = new URLSearchParams(search.toString());
    for (const [k, v] of Object.entries(patch)) {
      if (v) params.set(k, v);
      else params.delete(k);
    }
    const qs = params.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  };

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return (list.data ?? []).filter((d) => (!kbFilter || d.kb_id === kbFilter) && (!needle || `${d.name} ${d.filename ?? ""} ${d.kb_name} ${d.tables.map((t) => t.name).join(" ")}`.toLowerCase().includes(needle)));
  }, [list.data, q, kbFilter]);
  const kbsWithData = useMemo(() => {
    const m = new Map<string, string>();
    for (const d of list.data ?? []) m.set(d.kb_id, d.kb_name);
    return [...m.entries()].sort((a, b) => a[1].localeCompare(b[1]));
  }, [list.data]);

  // Select the first dataset when none (or an unknown one) is in the URL.
  useEffect(() => {
    if (!list.data?.length) return;
    if (!selectedId || !list.data.some((d) => d.id === selectedId)) setParams({ ds: list.data[0].id, tab: null });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [list.data, selectedId]);

  const uploadButton = (
    <Guarded allowed={canUpload || kbs.isLoading} reason="Uploading needs the kb.upload permission and write access to a knowledge base.">
      <Button onClick={() => setUploadOpen(true)} disabled={!canUpload}>
        <UploadCloud className="h-4 w-4" /> Upload a file
      </Button>
    </Guarded>
  );

  return (
    <Page>
      <PageHeader
        eyebrow="Knowledge"
        title="Data sources"
        description="Profile, preview, analyse and query your tabular data with validated, read-only SQL."
        actions={
          <>
            <Button variant="secondary" onClick={() => router.push("/database-connections")}>
              <Database className="h-4 w-4" /> Database connections
            </Button>
            {uploadButton}
          </>
        }
      />

      {list.isLoading ? (
        <div className="grid gap-6 lg:grid-cols-[300px_minmax(0,1fr)]">
          <Skeleton className="h-96 rounded-xl" />
          <Skeleton className="h-96 rounded-xl" />
        </div>
      ) : !list.data?.length ? (
        <Card>
          <EmptyState
            icon={<Table2 className="h-5 w-5" />}
            title="No datasets yet"
            description="Upload a CSV, Excel, JSON or Parquet file to any knowledge base. When indexing finishes it appears here with a profile, analysis and a SQL workspace."
            action={canUpload ? uploadButton : undefined}
          />
        </Card>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[300px_minmax(0,1fr)]">
          <aside className="min-w-0 space-y-3 lg:sticky lg:top-6 lg:self-start">
            <div className="card overflow-hidden">
              <div className="space-y-2 border-b p-3">
                <SearchField value={q} onChange={setQ} placeholder="Search datasets…" label="Search datasets" />
                {kbsWithData.length > 1 ? (
                  <Select value={kbFilter} onChange={(e) => setKbFilter(e.target.value)} className="h-8 text-[13px]" aria-label="Filter by knowledge base">
                    <option value="">All knowledge bases</option>
                    {kbsWithData.map(([id, name]) => (
                      <option key={id} value={id}>
                        {name}
                      </option>
                    ))}
                  </Select>
                ) : null}
              </div>
              <div className="flex items-center justify-between bg-surface-2/40 px-4 py-2">
                <span className="label">Datasets</span>
                <span className="num font-mono text-[11px] text-subtle">
                  {filtered.length}
                  {filtered.length !== list.data.length ? ` / ${list.data.length}` : ""}
                </span>
              </div>
              {filtered.length ? (
                <ul className="max-h-[60vh] divide-y overflow-y-auto border-t" role="listbox" aria-label="Datasets">
                  {filtered.map((d) => {
                    const on = d.id === selectedId;
                    return (
                      <li key={d.id}>
                        <button
                          type="button"
                          role="option"
                          aria-selected={on}
                          onClick={() => setParams({ ds: d.id })}
                          className={cn("flex w-full items-start gap-3 px-4 py-3 text-left transition-colors hover:bg-surface-2/50", on && "bg-accent/[0.06] shadow-[inset_2px_0_0_hsl(var(--accent))] hover:bg-accent/[0.08]")}
                        >
                          <FileIcon ext={d.filename?.split(".").pop() || "csv"} className="mt-0.5 h-7 w-7" />
                          <span className="min-w-0 flex-1">
                            <span className={cn("block truncate text-[13px] font-medium", on && "text-fg")}>{d.name}</span>
                            <span className="block truncate text-[12px] text-subtle">{d.kb_name}</span>
                            <span className="num mt-1 block font-mono text-[11px] text-muted">
                              {formatNumber(d.row_count)} rows · {formatNumber(d.column_count)} cols{d.tables.length > 1 ? ` · ${d.tables.length} tables` : ""}
                            </span>
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <p className="border-t px-4 py-8 text-center text-[12.5px] text-muted">No datasets match.</p>
              )}
            </div>
            <p className="flex items-start gap-2 px-1 text-[12px] leading-relaxed text-subtle">
              <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>
                Uploaded {tabularExts.length ? tabularExts.join(", ") : "CSV, XLSX, JSON and Parquet"} files become datasets once they&apos;re indexed. Each sheet or table is profiled and stays queryable here and in AI Chat.
              </span>
            </p>
          </aside>
          <section className="min-w-0">{selectedId ? <DatasetView key={selectedId} id={selectedId} tab={tab} onTab={(t) => setParams({ tab: t === "profile" ? null : t })} /> : <Skeleton className="h-96 rounded-xl" />}</section>
        </div>
      )}

      <UploadDialog open={uploadOpen} onOpenChange={setUploadOpen} kbs={kbs.data} />
    </Page>
  );
}

function DatasetView({ id, tab, onTab }: { id: string; tab: Tab; onTab: (t: Tab) => void }) {
  const detail = useApi<DatasetDetail>(`/datasets/${id}`);
  const d = detail.data;
  const [table, setTable] = useState<string>("");
  useEffect(() => {
    if (d && !table) setTable(d.table_name || d.tables[0]?.name || "");
  }, [d, table]);

  if (detail.error instanceof ApiError && detail.error.status === 404) {
    return (
      <Card>
        <EmptyState title="Dataset not found" description="It may have been deleted with its document, or you no longer have access." />
      </Card>
    );
  }
  if (!d || !table) return <Skeleton className="h-96 rounded-xl" />;

  const tables = d.profile?.tables ?? [];
  const current = tables.find((t) => t.name === table) ?? tables[0];
  const tableRefs = d.tables.length ? d.tables : tables.map((t) => ({ name: t.name, rows: t.rows, columns: t.columns }));

  return (
    <div className="space-y-6 animate-fade-in">
      <Card className="overflow-hidden">
        <div className="flex flex-wrap items-start gap-4 px-5 py-5">
          <FileIcon ext={d.filename?.split(".").pop() || "csv"} className="h-11 w-11 rounded-xl [&_svg]:h-5 [&_svg]:w-5" />
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-[19px] font-semibold tracking-[-0.025em]">{d.name}</h2>
            <div className="mt-1.5 flex flex-wrap items-center gap-x-2.5 gap-y-1.5 text-[12.5px] text-muted">
              {d.filename ? <MonoChip>{d.filename}</MonoChip> : null}
              <span>
                in{" "}
                <Link href={`/knowledge-bases/${d.kb_id}`} className="font-medium text-fg underline-offset-4 hover:underline">
                  {d.kb_name}
                </Link>
              </span>
              {d.source_type && d.source_type !== "upload" ? <Badge>{d.source_type}</Badge> : null}
              <span className="text-subtle" title={formatDate(d.updated_at)}>
                updated {timeAgo(d.updated_at)}
              </span>
            </div>
          </div>
          <Link href={`/documents?doc=${d.document_id}`} className="inline-flex h-8 w-full items-center justify-center gap-1.5 rounded-lg border bg-surface px-3 text-[13px] font-medium text-fg shadow-xs transition-colors hover:border-border-strong hover:bg-surface-2/60 sm:w-auto">
            <ExternalLink className="h-3.5 w-3.5" /> Open document
          </Link>
        </div>
        <div className="flex flex-wrap items-stretch border-t bg-surface-2/30">
          <Fact label="Rows" value={formatNumber(d.row_count)} />
          <Fact label="Columns" value={formatNumber(d.column_count)} />
          <Fact label="Tables" value={formatNumber(tableRefs.length)} />
          {tableRefs.length > 1 ? (
            <label className="flex w-full items-center gap-2.5 border-t px-5 py-2.5 sm:ml-auto sm:w-auto sm:border-l sm:border-t-0">
              <span className="label whitespace-nowrap">Sheet / table</span>
              <Select value={table} onChange={(e) => setTable(e.target.value)} className="h-8 w-auto min-w-[160px] flex-1 text-[13px] sm:flex-none" aria-label="Sheet or table">
                {tableRefs.map((t) => (
                  <option key={t.name} value={t.name}>
                    {t.name} ({formatNumber(t.rows)} rows)
                  </option>
                ))}
              </Select>
            </label>
          ) : null}
        </div>
      </Card>

      <Tabs value={tab} onValueChange={(v) => onTab(v as Tab)}>
        <TabsList variant="line" className="gap-4 sm:gap-6">
          <TabsTrigger value="profile">
            <Rows3 className="hidden h-3.5 w-3.5 sm:block" /> Profile
          </TabsTrigger>
          <TabsTrigger value="preview">
            <Table2 className="hidden h-3.5 w-3.5 sm:block" /> Preview
          </TabsTrigger>
          <TabsTrigger value="analysis">
            <BarChart3 className="hidden h-3.5 w-3.5 sm:block" /> Analysis
          </TabsTrigger>
          <TabsTrigger value="summary">
            <Sparkles className="hidden h-3.5 w-3.5 sm:block" /> Summary
          </TabsTrigger>
          <TabsTrigger value="ask">
            <MessageSquareText className="hidden h-3.5 w-3.5 sm:block" /> Ask
          </TabsTrigger>
        </TabsList>
        <TabsContent value="profile" className="mt-5 focus:outline-none">
          <ProfilePanel profile={current?.profile} />
        </TabsContent>
        <TabsContent value="preview" className="mt-5 focus:outline-none">
          <PreviewPanel key={table} datasetId={d.id} table={table} />
        </TabsContent>
        <TabsContent value="analysis" className="mt-5 focus:outline-none">
          <AnalysisPanel datasetId={d.id} table={table} />
        </TabsContent>
        {/* Summary and Ask stay mounted so results survive switching tabs. */}
        <TabsContent value="summary" forceMount className="mt-5 focus:outline-none data-[state=inactive]:hidden">
          <SummaryPanel key={table} datasetId={d.id} table={table} />
        </TabsContent>
        <TabsContent value="ask" forceMount className="mt-5 focus:outline-none data-[state=inactive]:hidden">
          <AskPanel datasetId={d.id} tables={tableRefs} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-[96px] flex-1 border-r px-5 py-2.5 sm:flex-none max-sm:[&:nth-child(3)]:border-r-0">
      <div className="label">{label}</div>
      <div className="num mt-0.5 text-[15px] font-semibold tracking-[-0.02em]">{value}</div>
    </div>
  );
}
