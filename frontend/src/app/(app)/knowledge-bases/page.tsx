"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, BookOpen, Database, FileText, HardDrive, Layers, LayoutGrid, Library, List, Plus, Search, X } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { toast } from "sonner";

import { CreateKBDialog } from "@/components/knowledge/kb-create-dialog";
import { AccessBadge, Guarded, invalidatePaths, KBMark, SearchField, Segmented, TagList, VisibilityBadge } from "@/components/knowledge/shared";
import type { KnowledgeBase, Visibility, Workspace } from "@/components/knowledge/types";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/form";
import { Badge, Card, Dot, EmptyState, Page, PageHeader, Skeleton, Stat, StatGroup, StatusBadge } from "@/components/ui/primitives";
import { post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatBytes, formatDate, formatNumber, pluralize, timeAgo } from "@/lib/utils";

type SortKey = "name" | "documents" | "indexed" | "updated";

export default function KnowledgeBasesPage() {
  const router = useRouter();
  const qc = useQueryClient();
  const { can, canAnywhere } = useAuth();
  const canCreate = canAnywhere("kb.create");
  const [createOpen, setCreateOpen] = useState(false);
  const [q, setQ] = useState("");
  const [visibility, setVisibility] = useState<"" | Visibility>("");
  const [workspace, setWorkspace] = useState("");
  const [sort, setSort] = useState<SortKey>("name");
  const [view, setView] = useState<"grid" | "list">("grid");

  const kbs = useApi<KnowledgeBase[]>("/knowledge-bases", undefined, {
    refetchInterval: (query) => ((query.state.data as KnowledgeBase[] | undefined)?.some((k) => (k.stats.processing ?? 0) > 0) ? 5000 : false),
  });
  const workspaces = useApi<Workspace[]>(can("workspace.view") ? "/workspaces" : null);
  const wsName = useMemo(() => new Map((workspaces.data ?? []).map((w) => [w.id, w.name])), [workspaces.data]);

  const sample = useMutation({
    mutationFn: () => post<{ id: string; created: boolean }>("/onboarding/sample-kb"),
    onSuccess: (r) => {
      toast.success(r.created ? "Sample knowledge base created — indexing documents" : "Sample knowledge base already exists");
      invalidatePaths(qc, "/knowledge-bases", "/onboarding", "/documents");
      router.push(`/knowledge-bases/${r.id}`);
    },
  });

  const list = useMemo(() => {
    const needle = q.trim().toLowerCase();
    let rows = (kbs.data ?? []).filter((k) => (!visibility || k.visibility === visibility) && (!workspace || (workspace === "none" ? !k.workspace_id : k.workspace_id === workspace)));
    if (needle) rows = rows.filter((k) => `${k.name} ${k.description} ${k.tags.join(" ")} ${k.category}`.toLowerCase().includes(needle));
    const by: Record<SortKey, (a: KnowledgeBase, b: KnowledgeBase) => number> = {
      name: (a, b) => a.name.localeCompare(b.name),
      documents: (a, b) => (b.stats.documents ?? 0) - (a.stats.documents ?? 0),
      indexed: (a, b) => (b.stats.last_indexed_at || "").localeCompare(a.stats.last_indexed_at || ""),
      updated: (a, b) => b.updated_at.localeCompare(a.updated_at),
    };
    return [...rows].sort(by[sort]);
  }, [kbs.data, q, visibility, workspace, sort]);

  const totals = useMemo(() => {
    const t = { documents: 0, chunks: 0, records: 0, storage: 0, failed: 0, processing: 0 };
    for (const k of kbs.data ?? []) {
      t.documents += k.stats.documents ?? 0;
      t.chunks += k.stats.chunks ?? 0;
      t.records += k.stats.records ?? 0;
      t.storage += k.stats.storage_bytes ?? 0;
      t.failed += k.stats.failed ?? 0;
      t.processing += k.stats.processing ?? 0;
    }
    return t;
  }, [kbs.data]);

  const filtered = Boolean(q || visibility || workspace);
  const clearFilters = () => {
    setQ("");
    setVisibility("");
    setWorkspace("");
  };
  const createButton = (
    <Guarded allowed={canCreate} reason="Creating knowledge bases requires the kb.create permission.">
      <Button onClick={() => setCreateOpen(true)} disabled={!canCreate}>
        <Plus className="h-4 w-4" /> New knowledge base
      </Button>
    </Guarded>
  );

  return (
    <Page>
      <PageHeader eyebrow="Knowledge" title="Knowledge bases" description="Collections of documents and datasets that AI Chat and agents can search, each with its own access rules and retrieval settings." actions={createButton} />

      {kbs.isLoading ? (
        <Skeleton className="mb-8 h-[108px] rounded-xl" />
      ) : kbs.data?.length ? (
        <StatGroup cols={4} className="mb-8 animate-fade-up">
          <Stat label="Knowledge bases" icon={<Library className="max-sm:hidden" />} value={formatNumber(kbs.data.length)} hint={`${formatNumber(kbs.data.filter((k) => k.access === "manage").length)} you manage`} />
          <Stat
            label="Documents"
            icon={<FileText className="max-sm:hidden" />}
            tone={totals.failed ? "danger" : totals.processing ? "info" : undefined}
            value={formatNumber(totals.documents)}
            hint={
              totals.processing ? (
                <span className="inline-flex items-center gap-1.5 text-info">
                  <Dot tone="info" live /> {formatNumber(totals.processing)} processing
                </span>
              ) : totals.failed ? (
                <span className="inline-flex items-center gap-1.5 text-danger">
                  <Dot tone="danger" /> {formatNumber(totals.failed)} failed
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5">
                  <Dot tone="ok" /> All indexed
                </span>
              )
            }
          />
          <Stat
            label="Chunks · records"
            icon={<Layers className="max-sm:hidden" />}
            value={
              <>
                {formatNumber(totals.chunks)}
                <span className="mx-1.5 text-subtle">/</span>
                {formatNumber(totals.records)}
              </>
            }
            hint="Passages · dataset rows"
          />
          <Stat label="Storage" icon={<HardDrive className="max-sm:hidden" />} value={formatBytes(totals.storage)} hint="Original files" />
        </StatGroup>
      ) : null}

      {kbs.data?.length ? (
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <SearchField value={q} onChange={setQ} placeholder="Search name, description or tag…" label="Search knowledge bases" className="w-full sm:w-72" />
          <Select value={visibility} onChange={(e) => setVisibility(e.target.value as "" | Visibility)} className="h-8 w-auto min-w-[136px] text-[13px]" aria-label="Filter by visibility">
            <option value="">Any visibility</option>
            <option value="restricted">Restricted</option>
            <option value="workspace">Workspace</option>
            <option value="org">Organization</option>
          </Select>
          {workspaces.data?.length ? (
            <Select value={workspace} onChange={(e) => setWorkspace(e.target.value)} className="h-8 w-auto min-w-[146px] text-[13px]" aria-label="Filter by workspace">
              <option value="">All workspaces</option>
              <option value="none">No workspace</option>
              {workspaces.data.map((w) => (
                <option key={w.id} value={w.id}>
                  {w.name}
                </option>
              ))}
            </Select>
          ) : null}
          {filtered ? (
            <Button size="sm" variant="ghost" onClick={clearFilters}>
              <X className="h-3.5 w-3.5" /> Clear
            </Button>
          ) : null}
          <div className="ml-auto flex items-center gap-2">
            <span className="num hidden font-mono text-[11.5px] text-subtle md:inline">
              {filtered ? `${list.length} of ${kbs.data.length}` : pluralize(kbs.data.length, "base")}
            </span>
            <Select value={sort} onChange={(e) => setSort(e.target.value as SortKey)} className="h-8 w-auto text-[13px]" aria-label="Sort knowledge bases">
              <option value="name">Sort: Name</option>
              <option value="documents">Sort: Most documents</option>
              <option value="indexed">Sort: Recently indexed</option>
              <option value="updated">Sort: Recently updated</option>
            </Select>
            <Segmented
              label="View"
              value={view}
              onChange={setView}
              options={[
                { value: "grid", label: <span className="sr-only">Grid</span>, icon: <LayoutGrid className="h-3.5 w-3.5" />, title: "Grid view" },
                { value: "list", label: <span className="sr-only">List</span>, icon: <List className="h-3.5 w-3.5" />, title: "List view" },
              ]}
            />
          </div>
        </div>
      ) : null}

      {kbs.isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-[236px] rounded-xl" />
          ))}
        </div>
      ) : !kbs.data?.length ? (
        <Card>
          <EmptyState
            icon={<Library className="h-5 w-5" />}
            title="No knowledge bases yet"
            description={canCreate ? "Create one to start uploading documents, or load a small sample to see how retrieval and citations work." : "You don't have access to any knowledge bases yet. Ask an administrator to share one with you."}
            action={
              canCreate ? (
                <div className="flex flex-wrap justify-center gap-2">
                  <Button onClick={() => setCreateOpen(true)}>
                    <Plus className="h-4 w-4" /> New knowledge base
                  </Button>
                  {can("kb.create") && can("kb.upload") ? (
                    <Button variant="secondary" onClick={() => sample.mutate()} loading={sample.isPending}>
                      <BookOpen className="h-4 w-4" /> Load sample
                    </Button>
                  ) : null}
                </div>
              ) : null
            }
          />
        </Card>
      ) : !list.length ? (
        <Card>
          <EmptyState
            icon={<Search className="h-5 w-5" />}
            title="No knowledge bases match"
            description="Try another search or clear the filters."
            action={
              filtered ? (
                <Button variant="secondary" size="sm" onClick={clearFilters}>
                  Clear filters
                </Button>
              ) : null
            }
          />
        </Card>
      ) : view === "grid" ? (
        <div className="stagger grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {list.map((kb) => (
            <KBCard key={kb.id} kb={kb} workspace={kb.workspace_id ? wsName.get(kb.workspace_id) : undefined} />
          ))}
        </div>
      ) : (
        <KBTable rows={list} wsName={wsName} />
      )}

      <CreateKBDialog open={createOpen} onOpenChange={setCreateOpen} />
    </Page>
  );
}

/** One-line index health: live processing, failures, or when it was last indexed. */
function IndexState({ kb }: { kb: KnowledgeBase }) {
  const s = kb.stats;
  const busy = Boolean(s.processing || s.failed);
  return (
    <span className="inline-flex flex-wrap items-center gap-x-2 gap-y-0.5">
      {s.processing ? (
        <span className="inline-flex items-center gap-1.5 text-info">
          <Dot tone="info" live /> {formatNumber(s.processing)} processing
        </span>
      ) : null}
      {s.failed ? (
        <span className="inline-flex items-center gap-1.5 text-danger">
          <Dot tone="danger" /> {formatNumber(s.failed)} failed
        </span>
      ) : null}
      {busy ? (
        s.last_indexed_at ? (
          <span className="text-subtle" title={formatDate(s.last_indexed_at)}>
            indexed {timeAgo(s.last_indexed_at)}
          </span>
        ) : null
      ) : (
        <span className="inline-flex items-center gap-1.5" title={s.last_indexed_at ? formatDate(s.last_indexed_at) : undefined}>
          <Dot tone={s.last_indexed_at ? "ok" : "neutral"} /> {s.last_indexed_at ? `Indexed ${timeAgo(s.last_indexed_at)}` : "Not indexed yet"}
        </span>
      )}
    </span>
  );
}

function KBCard({ kb, workspace }: { kb: KnowledgeBase; workspace?: string }) {
  const s = kb.stats;
  return (
    <Link
      href={`/knowledge-bases/${kb.id}`}
      className="card group flex min-w-0 flex-col overflow-hidden transition-[box-shadow,border-color,transform] duration-200 ease-out hover:-translate-y-px hover:border-border-strong hover:shadow-lift focus-visible:shadow-lift"
    >
      <div className="flex items-start gap-3.5 px-5 pt-5">
        <KBMark />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h3 className="truncate text-[15px] font-semibold tracking-[-0.015em]">{kb.name}</h3>
            <ArrowUpRight className="ml-auto h-4 w-4 shrink-0 -translate-x-1 text-subtle opacity-0 transition-all duration-200 group-hover:translate-x-0 group-hover:text-fg group-hover:opacity-100" aria-hidden />
          </div>
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            <VisibilityBadge visibility={kb.visibility} />
            <AccessBadge access={kb.access} />
            {kb.is_sample ? <Badge tone="warn">sample</Badge> : null}
            {kb.status !== "active" ? <StatusBadge status={kb.status} /> : null}
          </div>
        </div>
      </div>
      <p className="mt-4 line-clamp-2 min-h-[40px] px-5 text-[13px] leading-5 text-muted">{kb.description || <span className="text-subtle">No description</span>}</p>
      <dl className="mt-5 grid grid-cols-4 divide-x border-y bg-surface-2/35">
        <Metric icon={<FileText />} label="Docs" value={formatNumber(s.documents ?? 0)} />
        <Metric icon={<Layers />} label="Chunks" value={formatNumber(s.chunks ?? 0)} />
        <Metric icon={<Database />} label="Records" value={formatNumber(s.records ?? 0)} />
        <Metric icon={<HardDrive />} label="Size" value={formatBytes(s.storage_bytes ?? 0)} />
      </dl>
      <div className="mt-auto flex min-h-[46px] items-center gap-3 px-5 py-3 text-[12px] text-muted">
        <span className="shrink-0">
          <IndexState kb={kb} />
        </span>
        {workspace ? <span className="min-w-0 truncate text-subtle">· {workspace}</span> : null}
        {kb.tags.length ? (
          <span className="ml-auto min-w-0 shrink overflow-hidden">
            <TagList tags={kb.tags} max={2} />
          </span>
        ) : null}
      </div>
    </Link>
  );
}

function Metric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="min-w-0 px-3 py-3 first:pl-5">
      <dt className="flex items-center gap-1 font-mono text-[10px] font-medium uppercase tracking-[0.1em] text-subtle [&_svg]:h-3 [&_svg]:w-3">
        {icon}
        {label}
      </dt>
      <dd className="num mt-1 truncate text-[15px] font-semibold tracking-[-0.02em]">{value}</dd>
    </div>
  );
}

function KBTable({ rows, wsName }: { rows: KnowledgeBase[]; wsName: Map<string, string> }) {
  const router = useRouter();
  const th = "h-9 whitespace-nowrap px-4 text-left font-mono text-[10.5px] font-medium uppercase tracking-[0.1em] text-subtle";
  return (
    <div className="card overflow-hidden animate-fade-up">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[920px] text-[13px]">
          <thead>
            <tr className="border-b bg-surface-2/50">
              <th className={th}>Knowledge base</th>
              <th className={th}>Access</th>
              <th className={cn(th, "text-right")}>Docs</th>
              <th className={cn(th, "text-right")}>Chunks</th>
              <th className={cn(th, "text-right")}>Records</th>
              <th className={cn(th, "text-right")}>Storage</th>
              <th className={th}>Index</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((kb) => (
              <tr
                key={kb.id}
                tabIndex={0}
                onClick={() => router.push(`/knowledge-bases/${kb.id}`)}
                onKeyDown={(e) => e.key === "Enter" && router.push(`/knowledge-bases/${kb.id}`)}
                className="group/row cursor-pointer border-b transition-colors last:border-0 hover:bg-surface-2/50 focus-visible:bg-surface-2/60"
                aria-label={`Open ${kb.name}`}
              >
                <td className="max-w-[380px] px-4 py-3">
                  <div className="flex items-center gap-3">
                    <KBMark size="sm" />
                    <div className="min-w-0">
                      <div className="truncate font-medium text-fg">{kb.name}</div>
                      <div className="truncate text-[12px] text-subtle">{[kb.workspace_id ? wsName.get(kb.workspace_id) : null, kb.description].filter(Boolean).join(" · ") || "No description"}</div>
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-1.5">
                    <VisibilityBadge visibility={kb.visibility} />
                    <AccessBadge access={kb.access} />
                  </div>
                </td>
                <td className="num px-4 py-3 text-right font-medium">{formatNumber(kb.stats.documents ?? 0)}</td>
                <td className="num px-4 py-3 text-right">{formatNumber(kb.stats.chunks ?? 0)}</td>
                <td className="num px-4 py-3 text-right">{formatNumber(kb.stats.records ?? 0)}</td>
                <td className="num px-4 py-3 text-right text-muted">{formatBytes(kb.stats.storage_bytes ?? 0)}</td>
                <td className="whitespace-nowrap px-4 py-3 text-[12.5px] text-muted">
                  <IndexState kb={kb} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
