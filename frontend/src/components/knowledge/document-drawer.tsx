"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Activity, AlertTriangle, Download, FileStack, History, Info, Lock, RefreshCw, Save, ShieldAlert, Trash2 } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/dialog";
import { Field, Input } from "@/components/ui/form";
import { Badge, EmptyState, Skeleton, StatusBadge, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { ApiError, apiFetch, del, patch, post, type ApiErrorBody } from "@/lib/api";
import { useApi } from "@/lib/hooks";
import { cn, formatBytes, formatDate, formatDuration, formatNumber, timeAgo } from "@/lib/utils";

import { GrantsManager } from "./grants-manager";
import { fileFamily, FileIcon, FormatBadge, Guarded, invalidatePaths, isActive, kbAbilities, KV, memberLabel, MonoChip, needs, Pager, Segmented, Sheet, TagInput, TagList, useMemberMap, usePerms } from "./shared";
import type { Chunk, ChunkLocator, DocumentActivity, DocumentRow, DocumentVersion, KnowledgeBase, Ok, Page } from "./types";

/** Fetch a document original with the caller's credentials and save it as a file. */
export async function downloadDocument(id: string, fallbackName: string, version?: number) {
  const res = await apiFetch(`/documents/${id}/download`, { query: version ? { version } : undefined });
  if (!res.ok) {
    let body: ApiErrorBody | null = null;
    try {
      body = await res.json();
    } catch {
      body = null;
    }
    throw new ApiError(res.status, body, `Download failed (${res.status})`);
  }
  const blob = await res.blob();
  const match = /filename="?([^";]+)"?/i.exec(res.headers.get("content-disposition") || "");
  const name = match?.[1] || fallbackName;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
  return name;
}

export function locatorChips(m: ChunkLocator): string[] {
  const out: string[] = [];
  if (m.page !== undefined) out.push(m.page_end !== undefined ? `Pages ${m.page}–${m.page_end}` : `Page ${m.page}`);
  if (m.slide !== undefined && m.page === undefined) out.push(`Slide ${m.slide}`);
  if (m.sheet) out.push(`Sheet ${m.sheet}`);
  if (m.row_start !== undefined) out.push(m.row_end !== undefined && m.row_end !== m.row_start ? `Rows ${m.row_start}–${m.row_end}` : `Row ${m.row_start}`);
  if (m.line_start !== undefined) out.push(`Lines ${m.line_start}–${m.line_end ?? m.line_start}`);
  if (m.table_index !== undefined) out.push(`Table ${Number(m.table_index) + 1}`);
  if (m.chapter) out.push(`Chapter ${m.chapter}`);
  if (m.ocr) out.push("OCR");
  return out;
}

export function DocumentDrawer({ docId, onClose }: { docId: string | null; onClose: () => void }) {
  const qc = useQueryClient();
  const perms = usePerms();
  const members = useMemberMap();
  const [tab, setTab] = useState("overview");
  const [confirmDelete, setConfirmDelete] = useState(false);

  const doc = useApi<DocumentRow>(docId ? `/documents/${docId}` : null, undefined, {
    refetchInterval: (q) => (isActive((q.state.data as DocumentRow | undefined)?.status) ? 3000 : false),
  });
  const d = doc.data;
  const kb = useApi<KnowledgeBase>(d ? `/knowledge-bases/${d.kb_id}` : null);
  const ab = kbAbilities(kb.data, perms);
  const canWrite = ab.level >= 2;
  const canDownload = ab.exportFiles || (Boolean(d?.owner_id) && d?.owner_id === perms.userId);

  useEffect(() => setTab("overview"), [docId]);

  const refresh = () => invalidatePaths(qc, "/documents", "/knowledge-bases", "/datasets");

  const reprocess = useMutation({
    mutationFn: () => post<Ok>(`/documents/${docId}/reprocess`),
    onSuccess: () => {
      toast.success("Reprocessing queued");
      refresh();
    },
  });
  const download = useMutation({
    mutationFn: (version?: number) => downloadDocument(docId as string, d?.filename || "document", version),
    onSuccess: (name) => toast.success(`Downloading ${name}`),
  });
  const remove = useMutation({
    mutationFn: () => del<Ok>(`/documents/${docId}`),
    onSuccess: (r) => {
      toast.success(r?.detail || "Document deleted");
      setConfirmDelete(false);
      onClose();
      refresh();
    },
  });

  const notFound = doc.error instanceof ApiError && doc.error.status === 404;
  const suspect = Boolean(d?.meta?.security?.injection_suspect);
  const showAccessTab = d?.acl_mode === "restricted" && ab.permissions;

  return (
    <Sheet
      open={Boolean(docId)}
      onOpenChange={(v) => !v && onClose()}
      title={
        d ? (
          <span className="flex min-w-0 items-center gap-3">
            <FileIcon ext={d.ext} className="h-9 w-9 rounded-[10px]" />
            <span className="truncate">{d.title || d.filename}</span>
          </span>
        ) : (
          "Document"
        )
      }
      description={
        d ? (
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <StatusBadge status={d.status} />
            {suspect ? (
              <Badge tone="danger">
                <ShieldAlert className="h-3 w-3" /> Injection suspect
              </Badge>
            ) : null}
            {d.acl_mode === "restricted" ? (
              <Badge>
                <Lock className="h-3 w-3" /> Restricted
              </Badge>
            ) : null}
            <span className="truncate">
              in{" "}
              <Link href={`/knowledge-bases/${d.kb_id}`} className="font-medium text-fg underline-offset-4 hover:underline" onClick={onClose}>
                {d.kb_name || kb.data?.name || "knowledge base"}
              </Link>
            </span>
          </span>
        ) : undefined
      }
      actions={
        d ? (
          <>
            <Guarded allowed={ab.reprocess} reason={needs("kb.reprocess", "write")}>
              <Button size="sm" variant="secondary" disabled={!ab.reprocess || isActive(d.status)} loading={reprocess.isPending} onClick={() => reprocess.mutate()}>
                <RefreshCw className="h-3.5 w-3.5" /> Reprocess
              </Button>
            </Guarded>
            <Guarded allowed={canDownload} reason="Downloading originals requires the kb.export permission.">
              <Button size="sm" variant="secondary" disabled={!canDownload} loading={download.isPending} onClick={() => download.mutate(undefined)}>
                <Download className="h-3.5 w-3.5" /> Download
              </Button>
            </Guarded>
            <span className="flex-1" />
            <Guarded allowed={ab.deleteDocuments} reason={needs("kb.delete_documents", "write")}>
              <Button size="sm" variant="ghost" className="text-danger hover:bg-danger/10 hover:text-danger" disabled={!ab.deleteDocuments} onClick={() => setConfirmDelete(true)}>
                <Trash2 className="h-3.5 w-3.5" /> Delete
              </Button>
            </Guarded>
          </>
        ) : null
      }
    >
      {notFound ? (
        <EmptyState title="Document not found" description="It may have been deleted, or you no longer have access to it." />
      ) : !d ? (
        <div className="space-y-3 px-6 py-5">
          <Skeleton className="h-8 w-1/2" />
          <Skeleton className="h-24" />
          <Skeleton className="h-40" />
        </div>
      ) : (
        <Tabs value={tab} onValueChange={setTab} className="px-6 pb-8 pt-1">
          <TabsList variant="line" className="mb-5 gap-4 sm:gap-6">
            <TabsTrigger value="overview">
              <Info className="hidden h-3.5 w-3.5 sm:block" /> Overview
            </TabsTrigger>
            <TabsTrigger value="chunks">
              <FileStack className="hidden h-3.5 w-3.5 sm:block" /> Chunks <span className="num ml-0.5 rounded-full bg-surface-2 px-1.5 py-px font-mono text-[10.5px] text-muted ring-1 ring-inset ring-border">{formatNumber(d.chunk_count)}</span>
            </TabsTrigger>
            <TabsTrigger value="versions">
              <History className="hidden h-3.5 w-3.5 sm:block" /> Versions
            </TabsTrigger>
            <TabsTrigger value="activity">
              <Activity className="hidden h-3.5 w-3.5 sm:block" /> Activity
            </TabsTrigger>
            {showAccessTab ? (
              <TabsTrigger value="access">
                <Lock className="hidden h-3.5 w-3.5 sm:block" /> Access
              </TabsTrigger>
            ) : null}
          </TabsList>
          <TabsContent value="overview" className="focus:outline-none">
            <Overview doc={d} kb={kb.data} canWrite={canWrite} canPermissions={ab.restrictDocuments} members={members} onSaved={refresh} />
          </TabsContent>
          <TabsContent value="chunks" className="focus:outline-none">
            <ChunksPanel docId={d.id} status={d.status} />
          </TabsContent>
          <TabsContent value="versions" className="focus:outline-none">
            <VersionsPanel docId={d.id} members={members} canDownload={canDownload} downloading={download.isPending} onDownload={(v) => download.mutate(v)} />
          </TabsContent>
          <TabsContent value="activity" className="focus:outline-none">
            <ActivityPanel docId={d.id} members={members} live={isActive(d.status)} />
          </TabsContent>
          {showAccessTab ? (
            <TabsContent value="access" className="focus:outline-none">
              <p className="mb-3 text-[13px] text-muted">This document is restricted: only knowledge-base managers, its uploader and the grants below can see it.</p>
              <GrantsManager resourceType="document" resourceId={d.id} resourceName={d.title || d.filename} invalidate={["/documents"]} />
            </TabsContent>
          ) : null}
        </Tabs>
      )}
      <ConfirmDialog
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
        title="Delete document?"
        description={d ? <>“{d.title || d.filename}” and all of its versions, chunks, embeddings{fileFamily(d.ext) === "tabular" ? ", dataset" : ""} and stored files will be permanently deleted. Past answers keep their text, but their citations to it lose the quoted snippet.</> : null}
        confirmLabel="Delete document"
        loading={remove.isPending}
        onConfirm={() => remove.mutate()}
      />
    </Sheet>
  );
}

/* --------------------------------------------------------------- overview */

function Overview({ doc, kb, canWrite, canPermissions, members, onSaved }: { doc: DocumentRow; kb: KnowledgeBase | undefined; canWrite: boolean; canPermissions: boolean; members: ReturnType<typeof useMemberMap>; onSaved: () => void }) {
  const [title, setTitle] = useState(doc.title);
  const [tags, setTags] = useState<string[]>(doc.tags);
  const serverKey = `${doc.id}|${doc.title}|${JSON.stringify(doc.tags)}`;
  useEffect(() => {
    setTitle(doc.title);
    setTags(doc.tags);
    // Reset only when the stored values change (polling returns new array identities).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [serverKey]);
  const dirty = title.trim() !== doc.title || JSON.stringify(tags) !== JSON.stringify(doc.tags);

  const save = useMutation({
    mutationFn: () => patch<DocumentRow>(`/documents/${doc.id}`, { title: title.trim(), tags }),
    onSuccess: () => {
      toast.success("Document updated");
      onSaved();
    },
  });
  const setAcl = useMutation({
    mutationFn: (acl_mode: "inherit" | "restricted") => patch<DocumentRow>(`/documents/${doc.id}`, { acl_mode }),
    onSuccess: (r) => {
      toast.success(r.acl_mode === "restricted" ? "Document restricted — only managers, the uploader and document grants can see it" : "Document now inherits the knowledge base's access");
      onSaved();
    },
  });

  const m = doc.meta || {};
  const sec = m.security;
  const warnings = (m.warnings ?? []).filter(Boolean);
  const sourceMeta = Object.entries(m.source_metadata ?? {}).filter(([, v]) => v !== "" && v !== null);

  return (
    <div className="space-y-5">
      {doc.status === "failed" && doc.error ? (
        <div className="rounded-xl border border-danger/25 bg-danger/[0.05] p-4" role="alert">
          <div className="flex items-center gap-2 text-[13px] font-semibold text-danger">
            <AlertTriangle className="h-4 w-4" /> Processing failed
          </div>
          <p className="mt-1.5 whitespace-pre-wrap break-words font-mono text-[12px] leading-relaxed">{doc.error}</p>
          <p className="mt-2 text-[12px] text-muted">Fix the file and upload a new version, or reprocess if the failure was transient.</p>
        </div>
      ) : null}
      {isActive(doc.status) ? (
        <div className="flex items-center gap-2 rounded-xl border border-info/20 bg-info/[0.05] px-4 py-3 text-[13px] text-info">
          <RefreshCw className="h-4 w-4 animate-spin" /> {doc.status === "queued" ? "Waiting for an ingestion worker…" : "Parsing, chunking and embedding…"} This panel refreshes automatically.
        </div>
      ) : null}
      {sec?.injection_suspect ? (
        <div className="rounded-xl border border-danger/30 bg-danger/[0.05] p-4" role="alert">
          <div className="flex items-center gap-2 text-[13px] font-semibold text-danger">
            <ShieldAlert className="h-4 w-4" /> Possible prompt-injection content
          </div>
          <p className="mt-1 text-[13px]">
            {formatNumber(sec.flagged_chunks ?? 0)} chunk{sec.flagged_chunks === 1 ? "" : "s"} contain text that looks like instructions to an AI. Retrieved passages are always treated as data, but you can exclude flagged chunks entirely with “Quarantine suspicious chunks” in the knowledge base’s retrieval settings.
          </p>
          {sec.findings?.length ? (
            <ul className="mt-2 space-y-1">
              {sec.findings.map((f, i) => (
                <li key={i} className="truncate rounded-md border border-danger/20 bg-surface px-2 py-1 font-mono text-[11.5px]">
                  {f}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
      {warnings.length ? (
        <div className="rounded-xl border border-warn/25 bg-warn/[0.05] p-4">
          <div className="flex items-center gap-2 text-[13px] font-semibold text-warn">
            <AlertTriangle className="h-4 w-4" /> {warnings.length} parser warning{warnings.length === 1 ? "" : "s"}
          </div>
          <ul className="mt-1.5 max-h-40 list-disc space-y-0.5 overflow-y-auto pl-5 text-[12.5px]">
            {warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <section className="card p-4">
        <div className="grid gap-4">
          <Field label="Title" htmlFor="doc-title">
            <Input id="doc-title" value={title} onChange={(e) => setTitle(e.target.value)} maxLength={500} disabled={!canWrite} />
          </Field>
          <Field label="Tags" htmlFor="doc-tags" hint={m.auto_tags?.length ? "Suggestions are keywords detected in the content." : undefined}>
            <TagInput id="doc-tags" value={tags} onChange={setTags} disabled={!canWrite} suggestions={m.auto_tags} />
          </Field>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-1.5">
              <span className="text-[13px] font-medium">Access</span>
              <div className="flex items-center gap-2">
                <Guarded allowed={canPermissions} reason={needs("kb.permissions", "write")}>
                  <Segmented
                    label="Document access mode"
                    value={doc.acl_mode === "restricted" ? "restricted" : "inherit"}
                    onChange={(v) => v !== doc.acl_mode && setAcl.mutate(v)}
                    options={[
                      { value: "inherit", label: "Inherit from KB" },
                      { value: "restricted", label: "Restricted" },
                    ]}
                  />
                </Guarded>
                {setAcl.isPending ? <RefreshCw className="h-3.5 w-3.5 animate-spin text-subtle" /> : null}
              </div>
            </div>
            <Guarded allowed={canWrite} reason="Editing requires write access to this knowledge base.">
              <Button size="sm" disabled={!canWrite || !dirty || !title.trim()} loading={save.isPending} onClick={() => save.mutate()}>
                <Save className="h-3.5 w-3.5" /> Save changes
              </Button>
            </Guarded>
          </div>
        </div>
      </section>

      <section>
        <h4 className="mb-3 text-[13.5px] font-semibold tracking-[-0.01em]">Details</h4>
        <KV
          className="rounded-xl border bg-surface-2/30 p-4"
          items={[
            { k: "Filename", v: <span className="break-all">{doc.filename}</span>, wide: true },
            { k: "Format", v: <span className="flex items-center gap-2"><FormatBadge ext={doc.ext} /> <span className="truncate text-muted">{doc.mime_type}</span></span> },
            { k: "Size", v: <span className="num">{formatBytes(doc.size_bytes)}</span> },
            { k: "Knowledge base", v: <Link href={`/knowledge-bases/${doc.kb_id}`} className="text-accent hover:underline">{doc.kb_name || kb?.name || "Open"}</Link> },
            { k: "Owner", v: memberLabel(members, doc.owner_id) },
            { k: "Uploaded", v: formatDate(doc.created_at) },
            { k: "Last indexed", v: doc.last_indexed_at ? `${formatDate(doc.last_indexed_at)} (${timeAgo(doc.last_indexed_at)})` : "Not yet" },
            { k: "Version", v: <span className="font-mono text-[12px]">v{doc.current_version}</span> },
            { k: "Parser", v: doc.parser ? <span className="font-mono text-[12px]">{doc.parser}</span> : "—" },
            { k: "Pages", v: doc.page_count ?? "—" },
            { k: "Characters", v: <span className="num">{formatNumber(doc.char_count)}</span> },
            { k: "Tokens", v: <span className="num">{formatNumber(doc.token_count)}</span> },
            { k: "Chunks", v: <span className="num">{formatNumber(doc.chunk_count)}</span> },
            { k: "Embeddings", v: <span className="flex flex-wrap items-center gap-2"><StatusBadge status={doc.embedding_status} />{m.embedding?.model ? <span className="font-mono text-[11.5px] text-muted">{m.embedding.model}{m.embedding.dim ? ` · ${m.embedding.dim}d` : ""}</span> : null}</span> },
            { k: "Chunking", v: m.chunking?.strategy ? `${m.chunking.strategy.replace(/_/g, " ")} · ${m.chunking.chunk_size}` : "—" },
            { k: "Language", v: doc.language ? doc.language.toUpperCase() : "—" },
            { k: "Processing time", v: m.processing_ms !== undefined ? <span className="num">{formatDuration(m.processing_ms)}</span> : "—" },
            { k: "Source", v: <span className="capitalize">{doc.source_type}</span> },
            ...(doc.source_url ? [{ k: "Source URL", v: <a href={doc.source_url} target="_blank" rel="noreferrer noopener" className="break-all text-accent hover:underline">{doc.source_url}</a>, wide: true }] : []),
            ...(m.auto_tags?.length ? [{ k: "Auto tags", v: <TagList tags={m.auto_tags} max={10} tone="info" />, wide: true }] : []),
            { k: "Document ID", v: <span className="font-mono text-[11.5px] text-muted">{doc.id}</span>, wide: true },
          ]}
        />
      </section>

      {sourceMeta.length ? (
        <section>
          <h4 className="mb-3 text-[13.5px] font-semibold tracking-[-0.01em]">Source metadata</h4>
          <KV className="rounded-xl border bg-surface-2/30 p-4" items={sourceMeta.slice(0, 30).map(([k, v]) => ({ k: k.replace(/_/g, " "), v: Array.isArray(v) ? v.join(", ") : String(v) }))} />
        </section>
      ) : null}
    </div>
  );
}

/* ----------------------------------------------------------------- chunks */

function ChunksPanel({ docId, status }: { docId: string; status: string }) {
  const [offset, setOffset] = useState(0);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const limit = 20;
  const q = useApi<Page<Chunk>>(`/documents/${docId}/chunks`, { limit, offset });
  useEffect(() => setOffset(0), [docId]);

  if (q.isLoading) return <div className="space-y-2">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-24" />)}</div>;
  if (!q.data?.items.length)
    return <EmptyState icon={<FileStack className="h-5 w-5" />} title="No chunks yet" description={isActive(status) ? "Chunks appear once processing finishes." : status === "failed" ? "Processing failed before chunks were produced." : "This document produced no retrievable chunks."} />;

  return (
    <div className="overflow-hidden rounded-xl border bg-surface">
      <ol className="divide-y">
        {q.data.items.map((c) => {
          const open = expanded.has(c.id);
          const long = c.content.length > 700;
          const chips = locatorChips(c.metadata || {});
          const section = (c.metadata?.section ?? []).filter(Boolean);
          return (
            <li key={c.id} className="px-4 py-3.5">
              <div className="mb-2 flex flex-wrap items-center gap-1.5">
                <span className="num font-mono text-[11px] font-semibold text-fg/70">#{c.ordinal}</span>
                {chips.map((ch) => (
                  <MonoChip key={ch} className="text-fg/80">
                    {ch}
                  </MonoChip>
                ))}
                {c.metadata?.kind ? <span className="font-mono text-[11px] text-subtle">{c.metadata.kind}</span> : null}
                {c.parent_id ? <span className="text-[11.5px] text-subtle">child chunk</span> : null}
                {c.metadata?.injection_suspect ? (
                  <Badge tone="danger">
                    <ShieldAlert className="h-3 w-3" /> flagged
                  </Badge>
                ) : null}
                <span className="num ml-auto font-mono text-[11px] text-subtle">{formatNumber(c.token_count)} tok</span>
              </div>
              {section.length ? <div className="mb-1.5 truncate text-[12px] font-medium text-muted">{section.join(" › ")}</div> : null}
              <p className={cn("whitespace-pre-wrap break-words text-[13px] leading-relaxed text-fg/90", !open && long && "line-clamp-6")}>{c.content}</p>
              {long ? (
                <button
                  type="button"
                  className="mt-1.5 text-[12px] font-medium text-accent hover:underline"
                  onClick={() =>
                    setExpanded((s) => {
                      const n = new Set(s);
                      if (n.has(c.id)) n.delete(c.id);
                      else n.add(c.id);
                      return n;
                    })
                  }
                >
                  {open ? "Show less" : "Show full chunk"}
                </button>
              ) : null}
            </li>
          );
        })}
      </ol>
      <Pager total={q.data.total} limit={limit} offset={offset} onChange={setOffset} />
    </div>
  );
}

/* --------------------------------------------------------------- versions */

function VersionsPanel({ docId, members, canDownload, downloading, onDownload }: { docId: string; members: ReturnType<typeof useMemberMap>; canDownload: boolean; downloading: boolean; onDownload: (v: number) => void }) {
  const q = useApi<DocumentVersion[]>(`/documents/${docId}/versions`);
  if (q.isLoading) return <Skeleton className="h-32" />;
  if (!q.data?.length) return <EmptyState icon={<History className="h-5 w-5" />} title="No versions recorded" />;
  return (
    <div className="overflow-x-auto rounded-xl border bg-surface">
      <table className="w-full min-w-[560px] text-[13px]">
        <thead>
          <tr className="border-b bg-surface-2/50 text-left font-mono text-[10.5px] uppercase tracking-[0.1em] text-subtle">
            <th className="h-9 px-3 pl-4 font-medium">Version</th>
            <th className="h-9 px-3 font-medium">File</th>
            <th className="h-9 px-3 font-medium">Size</th>
            <th className="h-9 px-3 font-medium">SHA-256</th>
            <th className="h-9 px-3 font-medium">Uploaded</th>
            <th className="h-9 px-3" />
          </tr>
        </thead>
        <tbody>
          {q.data.map((v) => (
            <tr key={v.version} className="border-b last:border-0">
              <td className="whitespace-nowrap px-3 py-2.5 pl-4">
                <span className="font-mono text-[12px] font-semibold">v{v.version}</span> {v.current ? <Badge tone="accent">current</Badge> : null}
              </td>
              <td className="max-w-[200px] truncate px-3 py-2.5" title={v.filename}>
                {v.filename}
              </td>
              <td className="num whitespace-nowrap px-3 py-2.5 text-muted">{formatBytes(v.size_bytes)}</td>
              <td className="px-3 py-2.5 font-mono text-[11px] text-muted" title={v.sha256}>
                {v.sha256.slice(0, 12)}…
              </td>
              <td className="whitespace-nowrap px-3 py-2.5 text-muted">
                <div className="text-fg/90">{memberLabel(members, v.created_by)}</div>
                <div className="text-[11.5px] text-subtle">{formatDate(v.created_at)}</div>
              </td>
              <td className="px-3 py-2.5 pr-4 text-right">
                {canDownload ? (
                  <Button size="xs" variant="ghost" disabled={downloading} onClick={() => onDownload(v.version)} aria-label={`Download version ${v.version}`}>
                    <Download className="h-3.5 w-3.5" />
                  </Button>
                ) : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* --------------------------------------------------------------- activity */

const JOB_DOT: Record<string, string> = { succeeded: "bg-ok", failed: "bg-danger", dead: "bg-danger", running: "bg-info", queued: "bg-info", cancelled: "bg-subtle" };
const EVENT_DOT: Record<string, string> = { error: "bg-danger", warning: "bg-warn", success: "bg-ok" };

function ActivityPanel({ docId, members, live }: { docId: string; members: ReturnType<typeof useMemberMap>; live: boolean }) {
  const q = useApi<DocumentActivity[]>(`/documents/${docId}/activity`, undefined, { refetchInterval: live ? 3000 : false });
  const items = useMemo(() => [...(q.data ?? [])].sort((a, b) => (b.created_at || "").localeCompare(a.created_at || "")), [q.data]);
  if (q.isLoading) return <Skeleton className="h-40" />;
  if (!items.length) return <EmptyState icon={<Activity className="h-5 w-5" />} title="No activity recorded" />;
  return (
    <ol className="relative ml-1.5 space-y-0 border-l pl-6">
      {items.map((it) => (
        <li key={`${it.kind}-${it.id}`} className="relative pb-5">
          <span
            className={cn(
              "absolute -left-[29.5px] top-1.5 h-2.5 w-2.5 rounded-full ring-[3px] ring-surface",
              it.kind === "job" ? JOB_DOT[it.status] ?? "bg-subtle" : EVENT_DOT[it.severity] ?? "bg-info",
            )}
          />
          {it.kind === "job" ? (
            <div>
              <div className="flex flex-wrap items-center gap-2 text-[13px]">
                <span className="font-medium">Job · {it.type.replace(/_/g, " ")}</span>
                <StatusBadge status={it.status} />
                {it.attempts > 1 ? <span className="text-[11.5px] text-subtle">{it.attempts} attempts</span> : null}
              </div>
              {it.error ? <p className="mt-1 break-words font-mono text-[11.5px] text-danger">{it.error}</p> : null}
              <div className="mt-1 font-mono text-[11px] text-subtle">
                {formatDate(it.created_at)}
                {it.finished_at ? ` · took ${formatDuration(Math.max(0, new Date(it.finished_at).getTime() - new Date(it.created_at).getTime()))}` : ""}
              </div>
            </div>
          ) : (
            <div>
              <div className="text-[13px]">{it.message || it.type}</div>
              <div className="mt-1 flex flex-wrap items-center gap-x-2 text-[11.5px] text-subtle">
                <span className="font-mono text-[11px]">{it.type}</span>
                {it.actor_user_id ? <span>· {memberLabel(members, it.actor_user_id)}</span> : null}
                <span>· {it.created_at ? `${formatDate(it.created_at)} (${timeAgo(it.created_at)})` : ""}</span>
              </div>
            </div>
          )}
        </li>
      ))}
    </ol>
  );
}
