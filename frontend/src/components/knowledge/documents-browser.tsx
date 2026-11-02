"use client";

import { keepPreviousData, useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowDown, ArrowUp, FileText, LayoutGrid, List, RefreshCw, Search, ShieldAlert, SlidersHorizontal, Tags, Trash2, X } from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { ConfirmDialog, Dialog, DialogClose, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Select } from "@/components/ui/form";
import { Checkbox, Dot, EmptyState, Skeleton, SkeletonRows, StatusBadge, Tip } from "@/components/ui/primitives";
import { post } from "@/lib/api";
import { useApi } from "@/lib/hooks";
import { cn, formatBytes, formatDate, formatNumber, pluralize, timeAgo } from "@/lib/utils";

import { FileIcon, FormatBadge, Guarded, invalidatePaths, isActive, kbAbilities, memberLabel, Pager, SearchField, Segmented, TagInput, TagList, useDebounced, useMemberMap, usePerms } from "./shared";
import type { BulkResult, DocumentRow, KBOptions, KnowledgeBase, Page } from "./types";

type SortKey = "created_at" | "updated_at" | "title" | "size" | "status" | "chunks";
const SORTS: { value: SortKey; label: string }[] = [
  { value: "created_at", label: "Uploaded" },
  { value: "updated_at", label: "Updated" },
  { value: "title", label: "Title" },
  { value: "size", label: "Size" },
  { value: "status", label: "Status" },
  { value: "chunks", label: "Chunks" },
];
const STATUSES = ["queued", "processing", "ready", "failed"];
const SOURCES = [
  { value: "upload", label: "Uploads" },
  { value: "integration", label: "Integrations" },
  { value: "database", label: "Database syncs" },
];

export type DocFilters = { q: string; kb_id: string; status: string; ext: string; tag: string; source_type: string };
const EMPTY: DocFilters = { q: "", kb_id: "", status: "", ext: "", tag: "", source_type: "" };

/**
 * Server-side paginated, sorted and filtered document list with selection and bulk actions.
 * Used by the document library and a knowledge base's Documents tab (`fixedKbId`).
 */
export function DocumentsBrowser({
  kbs,
  fixedKbId,
  initialKbId,
  onOpen,
  selectedId,
  emptyAction,
  headerAction,
}: {
  kbs: KnowledgeBase[] | undefined;
  fixedKbId?: string;
  initialKbId?: string;
  onOpen: (id: string) => void;
  selectedId?: string | null;
  emptyAction?: ReactNode;
  headerAction?: ReactNode;
}) {
  const qc = useQueryClient();
  const perms = usePerms();
  const members = useMemberMap();
  const options = useApi<KBOptions>("/knowledge-bases/options", undefined, { staleTime: 5 * 60_000 });

  const [filters, setFilters] = useState<DocFilters>({ ...EMPTY, kb_id: fixedKbId || initialKbId || "" });
  const [sort, setSort] = useState<SortKey>("created_at");
  const [order, setOrder] = useState<"asc" | "desc">("desc");
  const [limit, setLimit] = useState(25);
  const [offset, setOffset] = useState(0);
  const [view, setView] = useState<"table" | "grid">("table");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [tagOpen, setTagOpen] = useState(false);
  const [bulkTags, setBulkTags] = useState<string[]>([]);

  const q = useDebounced(filters.q.trim(), 300);
  const tag = useDebounced(filters.tag.trim(), 300);

  useEffect(() => {
    try {
      const v = localStorage.getItem("rag.docs.view");
      if (v === "grid" || v === "table") setView(v);
    } catch {
      /* per-session preference */
    }
  }, []);
  const changeView = (v: "table" | "grid") => {
    setView(v);
    try {
      localStorage.setItem("rag.docs.view", v);
    } catch {
      /* not persisted */
    }
  };

  useEffect(() => {
    if (initialKbId !== undefined && !fixedKbId) setFilters((f) => ({ ...f, kb_id: initialKbId }));
  }, [initialKbId, fixedKbId]);

  const query = useMemo(
    () => ({ kb_id: fixedKbId || filters.kb_id || undefined, status: filters.status || undefined, ext: filters.ext || undefined, q: q || undefined, tag: tag || undefined, source_type: filters.source_type || undefined, sort, order, limit, offset }),
    [fixedKbId, filters.kb_id, filters.status, filters.ext, filters.source_type, q, tag, sort, order, limit, offset],
  );
  // Any change other than paging starts again from the first page.
  useEffect(() => {
    setOffset(0);
  }, [fixedKbId, filters.kb_id, filters.status, filters.ext, filters.source_type, q, tag, sort, order, limit]);
  useEffect(() => setSelected(new Set()), [query]);

  const docs = useApi<Page<DocumentRow>>("/documents", query, {
    placeholderData: keepPreviousData,
    refetchInterval: (qq) => ((qq.state.data as Page<DocumentRow> | undefined)?.items.some((d) => isActive(d.status)) ? 3000 : false),
  });
  const items = docs.data?.items ?? [];
  const total = docs.data?.total ?? 0;

  const kbMap = useMemo(() => new Map((kbs ?? []).map((k) => [k.id, k])), [kbs]);
  const selectedDocs = items.filter((d) => selected.has(d.id));
  const abilityOf = (d: DocumentRow) => kbAbilities(kbMap.get(d.kb_id), perms);
  const canBulkDelete = selectedDocs.some((d) => abilityOf(d).deleteDocuments);
  const canBulkReprocess = selectedDocs.some((d) => abilityOf(d).reprocess);
  const canBulkTag = selectedDocs.some((d) => abilityOf(d).level >= 2);

  const activeFilters = (Object.keys(EMPTY) as (keyof DocFilters)[]).filter((k) => k !== "kb_id" || !fixedKbId).some((k) => filters[k]);
  const processing = items.filter((d) => isActive(d.status)).length;

  const bulk = useMutation({
    mutationFn: (body: { action: "delete" | "reprocess" | "tag"; tags?: string[] }) => post<BulkResult>("/documents/bulk", { document_ids: [...selected], action: body.action, tags: body.tags ?? [] }),
    onSuccess: (r, body) => {
      const verb = { delete: "deleted", reprocess: "queued for reprocessing", tag: "tagged" }[body.action];
      if (r.succeeded) toast.success(`${pluralize(r.succeeded, "document")} ${verb}`);
      if (r.failed.length) toast.warning(`${pluralize(r.failed.length, "document")} could not be ${verb}`, { description: r.failed[0]?.error });
      setSelected(new Set());
      setConfirmDelete(false);
      setTagOpen(false);
      setBulkTags([]);
      invalidatePaths(qc, "/documents", "/knowledge-bases", "/datasets");
    },
  });

  const toggle = (id: string, on: boolean) =>
    setSelected((s) => {
      const n = new Set(s);
      if (on) n.add(id);
      else n.delete(id);
      return n;
    });
  const allOnPage = items.length > 0 && items.every((d) => selected.has(d.id));
  const someOnPage = items.some((d) => selected.has(d.id));

  const sortBy = (k: SortKey) => {
    if (sort === k) setOrder((o) => (o === "asc" ? "desc" : "asc"));
    else {
      setSort(k);
      setOrder(k === "title" ? "asc" : "desc");
    }
  };
  const thCls = "h-9 whitespace-nowrap px-3 text-left font-mono text-[10.5px] font-medium uppercase tracking-[0.1em] text-subtle first:pl-4";
  const sortHead = (k: SortKey, label: string, className?: string) => (
    <th key={k} className={cn(thCls, className)} aria-sort={sort === k ? (order === "asc" ? "ascending" : "descending") : "none"}>
      <button type="button" className={cn("inline-flex items-center gap-1 uppercase transition-colors hover:text-fg", sort === k && "text-fg")} onClick={() => sortBy(k)}>
        {label}
        {sort === k ? order === "asc" ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" /> : null}
      </button>
    </th>
  );
  const head = (label: string, className?: string) => (
    <th key={label} className={cn(thCls, className)}>
      {label}
    </th>
  );

  const formats = (options.data?.formats ?? []).filter((f) => !f.ext.includes(" ")).map((f) => f.ext);
  const filterCls = "h-8 w-auto text-[13px]";
  const clear = () => setFilters({ ...EMPTY, kb_id: fixedKbId || "" });

  return (
    <div className="card overflow-hidden">
      {/* toolbar: search + view controls */}
      <div className="flex flex-wrap items-center gap-2 px-3.5 py-3">
        <SearchField value={filters.q} onChange={(v) => setFilters((f) => ({ ...f, q: v }))} placeholder="Search title or filename…" label="Search documents" className="w-full sm:w-72" />
        {processing ? (
          <span className="hidden items-center gap-1.5 pl-1 text-[12px] text-info sm:inline-flex">
            <Dot tone="info" live /> <span className="num">{processing}</span> processing
          </span>
        ) : null}
        <div className="ml-auto flex items-center gap-2">
          {total > 0 ? <span className="num hidden pr-1 font-mono text-[11.5px] text-subtle md:inline">{pluralize(total, "document")}</span> : null}
          <Select value={sort} onChange={(e) => setSort(e.target.value as SortKey)} className={filterCls} aria-label="Sort by">
            {SORTS.map((s) => (
              <option key={s.value} value={s.value}>
                Sort: {s.label}
              </option>
            ))}
          </Select>
          <Button size="icon" variant="secondary" onClick={() => setOrder((o) => (o === "asc" ? "desc" : "asc"))} aria-label={order === "asc" ? "Sort descending" : "Sort ascending"}>
            {order === "asc" ? <ArrowUp className="h-3.5 w-3.5" /> : <ArrowDown className="h-3.5 w-3.5" />}
          </Button>
          <Segmented
            label="View"
            value={view}
            onChange={changeView}
            options={[
              { value: "table", label: <span className="sr-only">Table</span>, icon: <List className="h-3.5 w-3.5" />, title: "Table view" },
              { value: "grid", label: <span className="sr-only">Grid</span>, icon: <LayoutGrid className="h-3.5 w-3.5" />, title: "Grid view" },
            ]}
          />
          {headerAction}
        </div>
      </div>

      {/* filters */}
      <div className="flex flex-wrap items-center gap-2 border-y bg-surface-2/40 px-3.5 py-2">
        <span className="label mr-1 hidden items-center gap-1.5 sm:inline-flex">
          <SlidersHorizontal className="h-3 w-3" /> Filter
        </span>
        {!fixedKbId ? (
          <Select value={filters.kb_id} onChange={(e) => setFilters((f) => ({ ...f, kb_id: e.target.value }))} className={cn(filterCls, "min-w-[150px] max-w-[220px]")} aria-label="Filter by knowledge base">
            <option value="">All knowledge bases</option>
            {(kbs ?? []).map((k) => (
              <option key={k.id} value={k.id}>
                {k.name}
              </option>
            ))}
          </Select>
        ) : null}
        <Select value={filters.status} onChange={(e) => setFilters((f) => ({ ...f, status: e.target.value }))} className={cn(filterCls, "min-w-[118px]")} aria-label="Filter by status">
          <option value="">Any status</option>
          {STATUSES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </Select>
        <Select value={filters.ext} onChange={(e) => setFilters((f) => ({ ...f, ext: e.target.value }))} className={cn(filterCls, "min-w-[112px]")} aria-label="Filter by format">
          <option value="">Any format</option>
          {formats.map((x) => (
            <option key={x} value={x}>
              .{x}
            </option>
          ))}
        </Select>
        <Select value={filters.source_type} onChange={(e) => setFilters((f) => ({ ...f, source_type: e.target.value }))} className={cn(filterCls, "min-w-[118px]")} aria-label="Filter by source">
          <option value="">Any source</option>
          {SOURCES.map((s) => (
            <option key={s.value} value={s.value}>
              {s.label}
            </option>
          ))}
        </Select>
        <div className="relative">
          <Tags className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-subtle" />
          <Input value={filters.tag} onChange={(e) => setFilters((f) => ({ ...f, tag: e.target.value }))} placeholder="Tag" className="h-8 w-28 pl-8 text-[13px]" aria-label="Filter by tag" />
        </div>
        {activeFilters ? (
          <Button size="sm" variant="ghost" onClick={clear}>
            <X className="h-3.5 w-3.5" /> Clear
          </Button>
        ) : null}
      </div>

      {/* bulk bar */}
      {selected.size ? (
        <div className="flex flex-wrap items-center gap-2 border-b bg-accent/[0.06] px-3.5 py-2 animate-fade-in">
          <span className="grid h-5 min-w-5 place-items-center rounded-full bg-accent px-1.5 font-mono text-[11px] font-semibold text-accent-fg">{selected.size}</span>
          <span className="text-[13px] font-medium">{selected.size === 1 ? "document" : "documents"} selected</span>
          <Button size="xs" variant="ghost" onClick={() => setSelected(new Set())}>
            Clear selection
          </Button>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <Guarded allowed={canBulkTag} reason="Tagging requires write access to the documents' knowledge base.">
              <Button size="sm" variant="secondary" disabled={!canBulkTag || bulk.isPending} onClick={() => setTagOpen(true)}>
                <Tags className="h-3.5 w-3.5" /> Add tags
              </Button>
            </Guarded>
            <Guarded allowed={canBulkReprocess} reason="Reprocessing requires the kb.reprocess permission and write access.">
              <Button size="sm" variant="secondary" disabled={!canBulkReprocess} loading={bulk.isPending && bulk.variables?.action === "reprocess"} onClick={() => bulk.mutate({ action: "reprocess" })}>
                <RefreshCw className="h-3.5 w-3.5" /> Reprocess
              </Button>
            </Guarded>
            <Guarded allowed={canBulkDelete} reason="Deleting requires the kb.delete_documents permission and write access.">
              <Button size="sm" variant="danger" disabled={!canBulkDelete || bulk.isPending} onClick={() => setConfirmDelete(true)}>
                <Trash2 className="h-3.5 w-3.5" /> Delete
              </Button>
            </Guarded>
          </div>
        </div>
      ) : null}

      {/* body */}
      {docs.isLoading ? (
        view === "grid" ? (
          <div className="grid gap-3 p-3.5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} className="h-40 rounded-xl" />
            ))}
          </div>
        ) : (
          <SkeletonRows rows={8} />
        )
      ) : !items.length ? (
        activeFilters ? (
          <EmptyState
            icon={<Search className="h-5 w-5" />}
            title="No documents match these filters"
            description="Try a different search term, status or format."
            action={
              <Button variant="secondary" size="sm" onClick={clear}>
                Clear filters
              </Button>
            }
          />
        ) : (
          <EmptyState icon={<FileText className="h-5 w-5" />} title="No documents yet" description="Upload PDFs, Office files, web pages, slides, images, code or spreadsheets — they're parsed, chunked and embedded automatically." action={emptyAction} />
        )
      ) : view === "table" ? (
        <>
        {/* phones: a stacked list (the full table scrolls inside the card from md up) */}
        <ul className={cn("divide-y transition-opacity md:hidden", docs.isPlaceholderData && "opacity-60")}>
          {items.map((d) => (
            <li key={d.id} className={cn("flex items-start gap-3 px-3.5 py-3", (selected.has(d.id) || selectedId === d.id) && "bg-accent/[0.05]")}>
              <span className="pt-2">
                <Checkbox label={`Select ${d.title || d.filename}`} checked={selected.has(d.id)} onCheckedChange={(v) => toggle(d.id, v)} />
              </span>
              <button type="button" onClick={() => onOpen(d.id)} className="flex min-w-0 flex-1 items-start gap-3 text-left" aria-label={`Open ${d.title || d.filename}`}>
                <FileIcon ext={d.ext} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[13.5px] font-medium">{d.title || d.filename}</span>
                  <span className={cn("block truncate text-subtle", fixedKbId ? "font-mono text-[11px]" : "text-[12px]")}>{fixedKbId ? d.filename : d.kb_name || kbMap.get(d.kb_id)?.name || d.filename}</span>
                  <span className="mt-1.5 flex flex-wrap items-center gap-1.5">
                    <StatusBadge status={d.status} />
                    {d.meta?.security?.injection_suspect ? <ShieldAlert className="h-3.5 w-3.5 text-danger" aria-label="Injection suspect" /> : null}
                    <span className="num font-mono text-[11px] text-subtle">
                      {formatBytes(d.size_bytes)} · {formatNumber(d.chunk_count)} chunks · {timeAgo(d.created_at)}
                    </span>
                  </span>
                  {d.error ? <span className="mt-1 block truncate text-[11.5px] text-danger">{d.error}</span> : null}
                </span>
              </button>
            </li>
          ))}
        </ul>
        <div className={cn("hidden overflow-x-auto transition-opacity md:block", docs.isPlaceholderData && "opacity-60")}>
          <table className={cn("w-full text-[13px]", fixedKbId ? "min-w-[920px]" : "min-w-[1080px]")}>
            <thead>
              <tr className="border-b bg-surface-2/50">
                <th className="h-9 w-10 pl-4 pr-1">
                  <Checkbox label="Select all on this page" checked={allOnPage ? true : someOnPage ? "indeterminate" : false} onCheckedChange={(v) => setSelected((s) => { const n = new Set(s); items.forEach((d) => (v ? n.add(d.id) : n.delete(d.id))); return n; })} />
                </th>
                {sortHead("title", "Document")}
                {sortHead("status", "Status")}
                {!fixedKbId ? head("Knowledge base") : null}
                {sortHead("size", "Format · size")}
                {sortHead("chunks", "Chunks", "text-right [&>button]:flex-row-reverse")}
                {head("Index")}
                {sortHead("created_at", "Uploaded")}
              </tr>
            </thead>
            <tbody>
              {items.map((d) => {
                const active = selected.has(d.id) || selectedId === d.id;
                return (
                  <tr
                    key={d.id}
                    onClick={() => onOpen(d.id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && e.target === e.currentTarget) onOpen(d.id);
                    }}
                    tabIndex={0}
                    aria-label={`Open ${d.title || d.filename}`}
                    className={cn(
                      "group/row cursor-pointer border-b transition-colors last:border-0 hover:bg-surface-2/50 focus-visible:bg-surface-2/60",
                      active && "bg-accent/[0.05]",
                      selectedId === d.id && "shadow-[inset_2px_0_0_hsl(var(--accent))]",
                    )}
                  >
                    <td className="w-10 py-3 pl-4 pr-1" onClick={(e) => e.stopPropagation()}>
                      <Checkbox label={`Select ${d.title || d.filename}`} checked={selected.has(d.id)} onCheckedChange={(v) => toggle(d.id, v)} />
                    </td>
                    <td className="max-w-[340px] px-3 py-3">
                      <div className="flex items-center gap-3">
                        <FileIcon ext={d.ext} />
                        <div className="min-w-0">
                          <div className="truncate font-medium text-fg">{d.title || d.filename}</div>
                          <div className="mt-0.5 flex min-w-0 items-center gap-2">
                            <span className="truncate font-mono text-[11px] text-subtle">{d.filename}</span>
                            {d.tags.length ? (
                              <span className="hidden shrink-0 lg:inline-flex">
                                <TagList tags={d.tags} max={2} />
                              </span>
                            ) : null}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-3 py-3">
                      <span className="inline-flex items-center gap-1.5">
                        <StatusBadge status={d.status} />
                        {d.meta?.security?.injection_suspect ? (
                          <Tip content="Contains text that looks like prompt-injection instructions">
                            <span className="text-danger" aria-label="Injection suspect">
                              <ShieldAlert className="h-3.5 w-3.5" />
                            </span>
                          </Tip>
                        ) : null}
                      </span>
                      {d.error ? (
                        <Tip content={<span className="whitespace-pre-wrap">{d.error}</span>}>
                          <span className="mt-1 block max-w-[220px] truncate text-[11.5px] text-danger">{d.error}</span>
                        </Tip>
                      ) : null}
                    </td>
                    {!fixedKbId ? <td className="max-w-[180px] truncate px-3 py-3 text-muted">{d.kb_name || kbMap.get(d.kb_id)?.name || "—"}</td> : null}
                    <td className="whitespace-nowrap px-3 py-3">
                      <span className="inline-flex items-center gap-2">
                        <FormatBadge ext={d.ext} />
                        <span className="num text-muted">{formatBytes(d.size_bytes)}</span>
                      </span>
                    </td>
                    <td className="whitespace-nowrap px-3 py-3 text-right">
                      <div className="num font-medium">{formatNumber(d.chunk_count)}</div>
                      {d.page_count ? <div className="num text-[11.5px] text-subtle">{d.page_count} pp</div> : null}
                    </td>
                    <td className="whitespace-nowrap px-3 py-3">
                      <StatusBadge status={d.embedding_status} />
                      <div className="mt-1 font-mono text-[11px] text-subtle" title={d.last_indexed_at ? formatDate(d.last_indexed_at) : undefined}>
                        {d.parser || "—"} · {d.last_indexed_at ? timeAgo(d.last_indexed_at) : "not indexed"}
                      </div>
                    </td>
                    <td className="max-w-[160px] px-3 py-3 pr-4">
                      <div className="truncate text-fg/90">{memberLabel(members, d.owner_id)}</div>
                      <div className="text-[12px] text-subtle" title={formatDate(d.created_at)}>
                        {timeAgo(d.created_at)}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        </>
      ) : (
        <div className={cn("grid gap-3 p-3.5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4", docs.isPlaceholderData && "opacity-60")}>
          {items.map((d) => {
            const active = selected.has(d.id) || selectedId === d.id;
            return (
              <div
                key={d.id}
                role="button"
                tabIndex={0}
                onClick={() => onOpen(d.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && e.target === e.currentTarget) onOpen(d.id);
                }}
                aria-label={`Open ${d.title || d.filename}`}
                className={cn(
                  "group relative flex min-w-0 cursor-pointer flex-col rounded-xl border bg-surface p-4 text-left shadow-xs transition-[box-shadow,border-color,transform] duration-200 hover:-translate-y-px hover:border-border-strong hover:shadow-lift",
                  active && "border-accent/50 bg-accent/[0.04] ring-1 ring-accent/20",
                )}
              >
                <div className="flex items-start gap-3">
                  <FileIcon ext={d.ext} className="h-9 w-9" />
                  <div className="min-w-0 flex-1">
                    <div className="line-clamp-2 text-[13.5px] font-medium leading-snug">{d.title || d.filename}</div>
                    <div className={cn("mt-0.5 truncate text-subtle", fixedKbId ? "font-mono text-[11px]" : "text-[12px]")}>{fixedKbId ? d.filename : d.kb_name || kbMap.get(d.kb_id)?.name}</div>
                  </div>
                  <span onClick={(e) => e.stopPropagation()} className={cn("transition-opacity", !selected.has(d.id) && "sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100")}>
                    <Checkbox label={`Select ${d.title || d.filename}`} checked={selected.has(d.id)} onCheckedChange={(v) => toggle(d.id, v)} />
                  </span>
                </div>
                <div className="mt-3.5 flex flex-wrap items-center gap-1.5">
                  <StatusBadge status={d.status} />
                  <FormatBadge ext={d.ext} />
                  {d.meta?.security?.injection_suspect ? (
                    <Tip content="Contains text that looks like prompt-injection instructions">
                      <span className="text-danger" aria-label="Injection suspect">
                        <ShieldAlert className="h-3.5 w-3.5" />
                      </span>
                    </Tip>
                  ) : null}
                </div>
                {d.error ? <p className="mt-2 line-clamp-2 text-[12px] text-danger">{d.error}</p> : null}
                <div className="mt-auto pt-4">
                  <div className="flex items-center justify-between gap-2 border-t pt-3 text-[11.5px] text-subtle">
                    <span className="num truncate font-mono">
                      {formatBytes(d.size_bytes)} · {formatNumber(d.chunk_count)} chunks{d.page_count ? ` · ${d.page_count} pp` : ""}
                    </span>
                    <span className="shrink-0" title={formatDate(d.created_at)}>
                      {timeAgo(d.created_at)}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {total > 0 ? <Pager total={total} limit={limit} offset={offset} onChange={setOffset} onLimit={setLimit} /> : null}

      <ConfirmDialog
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
        title={`Delete ${pluralize(selected.size, "document")}?`}
        description="Each document's versions, chunks, embeddings, datasets and stored files are permanently deleted. Documents you cannot delete are skipped and reported."
        confirmLabel={`Delete ${pluralize(selected.size, "document")}`}
        requireText={selected.size >= 10 ? `delete ${selected.size}` : undefined}
        loading={bulk.isPending}
        onConfirm={() => bulk.mutate({ action: "delete" })}
      >
        <ul className="mb-3 max-h-40 divide-y overflow-y-auto rounded-lg border bg-surface-2/40 text-[12.5px]">
          {selectedDocs.map((d) => (
            <li key={d.id} className="flex items-center gap-2 px-3 py-1.5">
              <FileIcon ext={d.ext} className="h-6 w-6 rounded-md [&_svg]:h-3 [&_svg]:w-3" />
              <span className="truncate">{d.title || d.filename}</span>
            </li>
          ))}
        </ul>
      </ConfirmDialog>

      <Dialog open={tagOpen} onOpenChange={(v) => { setTagOpen(v); if (!v) setBulkTags([]); }}>
        <DialogContent
          title={`Tag ${pluralize(selected.size, "document")}`}
          description="Tags are added to each document's existing tags."
          footer={
            <>
              <DialogClose asChild>
                <Button variant="secondary">Cancel</Button>
              </DialogClose>
              <Button disabled={!bulkTags.length} loading={bulk.isPending} onClick={() => bulk.mutate({ action: "tag", tags: bulkTags })}>
                Add {bulkTags.length ? pluralize(bulkTags.length, "tag") : "tags"}
              </Button>
            </>
          }
        >
          <Field label="Tags" htmlFor="bulk-tags">
            <TagInput id="bulk-tags" value={bulkTags} onChange={setBulkTags} />
          </Field>
        </DialogContent>
      </Dialog>
    </div>
  );
}
