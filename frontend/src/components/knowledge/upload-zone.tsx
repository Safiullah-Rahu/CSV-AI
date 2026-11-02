"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertCircle, CheckCircle2, CircleSlash, Copy, FolderUp, Info, Loader2, RotateCcw, Trash2, UploadCloud, X } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Field } from "@/components/ui/form";
import { Badge, Switch, Tip, type Tone } from "@/components/ui/primitives";
import { api, errorMessage } from "@/lib/api";
import { cn, formatBytes, pluralize } from "@/lib/utils";

import { FileIcon, Guarded, invalidatePaths, Segmented, TagInput } from "./shared";
import type { KBOptions, UploadOutcome, UploadResponse, UploadResult } from "./types";

type Staged = { key: number; file: File; path: string; issue: null | "unsupported" | "too_large" };

const BATCH_FILES = 20;
const BATCH_BYTES = 200 * 1024 * 1024;
const MAX_STAGED = 1000;
const JUNK = /^(\.|~\$)|^(thumbs\.db|desktop\.ini)$/i;

const OUTCOME: Record<UploadOutcome, { label: string; tone: Tone; icon: typeof CheckCircle2 }> = {
  created: { label: "Queued for indexing", tone: "ok", icon: CheckCircle2 },
  new_version: { label: "New version queued", tone: "ok", icon: CheckCircle2 },
  duplicate: { label: "Duplicate", tone: "warn", icon: Copy },
  unchanged: { label: "Unchanged", tone: "neutral", icon: CircleSlash },
  rejected: { label: "Rejected", tone: "danger", icon: AlertCircle },
  skipped: { label: "Skipped", tone: "neutral", icon: CircleSlash },
};

let seq = 0;

function extOf(name: string) {
  const i = name.lastIndexOf(".");
  return i > 0 ? name.slice(i + 1).toLowerCase() : "";
}

async function walk(entry: FileSystemEntry, prefix: string, out: { file: File; path: string }[]): Promise<void> {
  if (entry.isFile) {
    const file = await new Promise<File | null>((res) => (entry as FileSystemFileEntry).file(res, () => res(null)));
    if (file) out.push({ file, path: prefix ? `${prefix}${file.name}` : "" });
    return;
  }
  if (entry.isDirectory) {
    const reader = (entry as FileSystemDirectoryEntry).createReader();
    const dir = `${prefix}${entry.name}/`;
    for (;;) {
      const batch = await new Promise<FileSystemEntry[]>((res) => reader.readEntries(res, () => res([])));
      if (!batch.length) break;
      for (const child of batch) await walk(child, dir, out);
    }
  }
}

/** Collect dropped files, descending into dropped folders (paths keep the folder structure). */
function collectDrop(dt: DataTransfer): Promise<{ file: File; path: string }[]> {
  const entries: FileSystemEntry[] = [];
  const loose: { file: File; path: string }[] = [];
  // DataTransfer items are only readable synchronously inside the event handler.
  for (const item of Array.from(dt.items ?? [])) {
    if (item.kind !== "file") continue;
    const entry = typeof item.webkitGetAsEntry === "function" ? item.webkitGetAsEntry() : null;
    if (entry) entries.push(entry);
    else {
      const f = item.getAsFile();
      if (f) loose.push({ file: f, path: "" });
    }
  }
  if (!entries.length && !loose.length) return Promise.resolve(Array.from(dt.files).map((file) => ({ file, path: "" })));
  return (async () => {
    const out = [...loose];
    for (const e of entries) await walk(e, "", out);
    return out;
  })();
}

export function UploadZone({
  kbId,
  kbName,
  options,
  allowed,
  deniedReason,
  canRestrict,
  onFinished,
  className,
}: {
  kbId: string;
  kbName?: string;
  options: KBOptions | undefined;
  allowed: boolean;
  deniedReason?: string;
  canRestrict: boolean;
  onFinished?: (results: UploadResult[]) => void;
  className?: string;
}) {
  const qc = useQueryClient();
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);
  const [dragging, setDragging] = useState(false);
  const [staged, setStaged] = useState<Staged[]>([]);
  const [junk, setJunk] = useState(0);
  const [tags, setTags] = useState<string[]>([]);
  const [acl, setAcl] = useState<"inherit" | "restricted">("inherit");
  const [allowDuplicate, setAllowDuplicate] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number; batch: number; batches: number } | null>(null);
  const [results, setResults] = useState<UploadResult[] | null>(null);

  const maxBytes = (options?.max_upload_mb ?? 100) * 1024 * 1024;
  const supported = useMemo(() => {
    if (!options) return null;
    return new Set(options.formats.filter((f) => f.available && !f.ext.includes(" ")).map((f) => f.ext));
  }, [options]);
  const families = useMemo(() => {
    const m = new Map<string, string[]>();
    for (const f of options?.formats ?? []) {
      if (f.ext.includes(" ")) continue;
      m.set(f.family, [...(m.get(f.family) ?? []), f.ext + (f.available ? "" : ` (needs ${f.requires})`)]);
    }
    return m;
  }, [options]);

  useEffect(() => {
    // `webkitdirectory` is not in React's input typings; set it on the element.
    folderInput.current?.setAttribute("webkitdirectory", "");
    folderInput.current?.setAttribute("directory", "");
  }, []);

  const stage = useCallback(
    (incoming: { file: File; path: string }[]) => {
      let skipped = 0;
      const next: Staged[] = [];
      for (const { file, path } of incoming) {
        if (JUNK.test(file.name)) {
          skipped++;
          continue;
        }
        const ext = extOf(file.name);
        const issue = file.size > maxBytes ? "too_large" : supported && !supported.has(ext) ? "unsupported" : null;
        next.push({ key: ++seq, file, path, issue });
      }
      const id = (x: Staged) => `${x.path || x.file.name}:${x.file.size}:${x.file.lastModified}`;
      const seen = new Set(staged.map(id));
      const merged = [...staged, ...next.filter((x) => !seen.has(id(x)))];
      if (merged.length > MAX_STAGED) toast.warning(`Only the first ${MAX_STAGED} files were added.`);
      setJunk((j) => j + skipped);
      setResults(null);
      setStaged(merged.slice(0, MAX_STAGED));
    },
    [maxBytes, supported, staged],
  );

  const ready = staged.filter((s) => !s.issue);
  const blocked = staged.filter((s) => s.issue);

  const upload = useMutation({
    mutationFn: async () => {
      const batches: Staged[][] = [];
      let cur: Staged[] = [];
      let bytes = 0;
      for (const s of ready) {
        if (cur.length && (cur.length >= BATCH_FILES || bytes + s.file.size > BATCH_BYTES)) {
          batches.push(cur);
          cur = [];
          bytes = 0;
        }
        cur.push(s);
        bytes += s.file.size;
      }
      if (cur.length) batches.push(cur);

      const out: UploadResult[] = blocked.map((s) => ({
        filename: s.file.name,
        path: s.path,
        outcome: "skipped" as const,
        error: s.issue === "too_large" ? `Exceeds the ${options?.max_upload_mb ?? 100} MB upload limit.` : `.${extOf(s.file.name) || "?"} files are not supported here.`,
      }));
      let done = 0;
      for (let i = 0; i < batches.length; i++) {
        const batch = batches[i];
        setProgress({ done, total: ready.length, batch: i + 1, batches: batches.length });
        const fd = new FormData();
        for (const s of batch) fd.append("files", s.file, s.file.name);
        fd.append("tags", tags.join(","));
        fd.append("acl_mode", acl);
        fd.append("allow_duplicate", allowDuplicate ? "true" : "false");
        if (batch.some((s) => s.path)) fd.append("relative_paths", JSON.stringify(batch.map((s) => s.path)));
        try {
          const res = await api<UploadResponse>(`/knowledge-bases/${kbId}/documents`, { method: "POST", body: fd });
          res.results.forEach((r, j) => out.push({ ...r, path: batch[j]?.path }));
        } catch (err) {
          // A whole batch failed (limit, permission, network): report it per file and keep going.
          const msg = errorMessage(err);
          for (const s of batch) out.push({ filename: s.file.name, path: s.path, outcome: "rejected", error: msg });
        }
        done += batch.length;
        setProgress({ done, total: ready.length, batch: i + 1, batches: batches.length });
      }
      return out;
    },
    onSuccess: (out) => {
      setResults(out);
      setStaged([]);
      setJunk(0);
      setProgress(null);
      const accepted = out.filter((r) => r.outcome === "created" || r.outcome === "new_version").length;
      const failed = out.filter((r) => r.outcome === "rejected").length;
      if (accepted) toast.success(`${pluralize(accepted, "file")} uploaded${kbName ? ` to ${kbName}` : ""} — indexing started`);
      if (failed) toast.warning(`${pluralize(failed, "file")} could not be uploaded. See the results below.`);
      if (!accepted && !failed) toast.message("Nothing new to index — files were duplicates, unchanged or skipped.");
      invalidatePaths(qc, "/documents", "/knowledge-bases", "/datasets", "/connections", "/onboarding");
      onFinished?.(out);
    },
    onError: () => setProgress(null),
  });

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    if (!allowed || upload.isPending) return;
    collectDrop(e.dataTransfer).then(stage);
  };

  const busy = upload.isPending;

  return (
    <div className={cn("space-y-3", className)}>
      <div
        onDragEnter={(e) => {
          e.preventDefault();
          if (!allowed) return;
          dragDepth.current++;
          setDragging(true);
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={() => {
          dragDepth.current = Math.max(0, dragDepth.current - 1);
          if (!dragDepth.current) setDragging(false);
        }}
        onDrop={onDrop}
        className={cn(
          "group/drop relative overflow-hidden rounded-xl border border-dashed transition-[border-color,background-color,box-shadow] duration-200",
          dragging ? "border-accent bg-accent/[0.045] shadow-[0_0_0_4px_hsl(var(--accent)/0.10)]" : "border-border-strong/80 bg-surface hover:border-border-strong",
          !allowed && "opacity-70",
        )}
      >
        <span aria-hidden className={cn("pointer-events-none absolute inset-0 bg-dots transition-opacity [mask-image:linear-gradient(100deg,black,transparent_55%)]", dragging ? "opacity-100" : "opacity-60")} />
        <div className="relative flex flex-col gap-4 px-5 py-5 sm:flex-row sm:items-center sm:gap-5">
          <div className="flex min-w-0 flex-1 items-center gap-4">
            <span className={cn("relative grid h-12 w-12 shrink-0 place-items-center rounded-xl border bg-surface text-accent shadow-card transition-transform duration-200", dragging && "-translate-y-0.5 scale-105")}>
              <span className="absolute inset-[3px] rounded-[9px] bg-accent/[0.07]" />
              <UploadCloud className="relative h-5 w-5" strokeWidth={1.75} />
            </span>
            <div className="min-w-0">
              <p className="text-[14px] font-semibold tracking-[-0.01em]">{dragging ? "Drop to add files" : "Drop files or folders to upload"}</p>
              <p className="mt-0.5 text-[12.5px] leading-relaxed text-muted">
                Up to <span className="num font-medium text-fg/80">{options?.max_upload_mb ?? 100} MB</span> per file. Folder names become tags. CSV, Excel, JSON and Parquet files also become queryable datasets.
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 sm:shrink-0 sm:justify-end">
          {families.size ? (
            <Tip
              content={
                <div className="space-y-1 py-0.5">
                  {[...families.entries()].map(([fam, exts]) => (
                    <div key={fam}>
                      <span className="font-medium capitalize">{fam}:</span> <span className="text-muted">{exts.join(", ")}</span>
                    </div>
                  ))}
                </div>
              }
            >
              <button type="button" aria-label="Supported formats" className="inline-flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-[12.5px] text-muted transition-colors hover:bg-surface-2 hover:text-fg">
                <Info className="h-3.5 w-3.5" /> <span className="max-sm:hidden">Formats</span>
              </button>
            </Tip>
          ) : null}
          <Guarded allowed={allowed} reason={deniedReason || "You cannot upload to this knowledge base."}>
            <Button size="sm" variant="secondary" disabled={!allowed || busy} onClick={() => folderInput.current?.click()}>
              <FolderUp className="h-3.5 w-3.5" /> Choose folder
            </Button>
          </Guarded>
          <Guarded allowed={allowed} reason={deniedReason || "You cannot upload to this knowledge base."}>
            <Button size="sm" disabled={!allowed || busy} onClick={() => fileInput.current?.click()}>
              <UploadCloud className="h-3.5 w-3.5" /> Choose files
            </Button>
          </Guarded>
          </div>
        </div>
        <input
          ref={fileInput}
          type="file"
          multiple
          className="hidden"
          aria-label="Choose files to upload"
          onChange={(e) => {
            stage(Array.from(e.target.files ?? []).map((file) => ({ file, path: "" })));
            e.target.value = "";
          }}
        />
        <input
          ref={folderInput}
          type="file"
          multiple
          className="hidden"
          aria-label="Choose a folder to upload"
          onChange={(e) => {
            stage(Array.from(e.target.files ?? []).map((file) => ({ file, path: file.webkitRelativePath || "" })));
            e.target.value = "";
          }}
        />
      </div>

      {staged.length ? (
        <div className="card overflow-hidden animate-fade-in">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b px-4 py-3">
            <span className="text-[13.5px] font-semibold tracking-[-0.01em]">{pluralize(staged.length, "file")} selected</span>
            <span className="num font-mono text-[11.5px] text-subtle">{formatBytes(ready.reduce((n, s) => n + s.file.size, 0))} to upload</span>
            {blocked.length ? <Badge tone="warn">{blocked.length} will be skipped</Badge> : null}
            {junk ? <span className="text-[11.5px] text-subtle">{junk} hidden/system files ignored</span> : null}
            <Button size="xs" variant="ghost" className="ml-auto" disabled={busy} onClick={() => { setStaged([]); setJunk(0); }}>
              <Trash2 className="h-3.5 w-3.5" /> Clear
            </Button>
          </div>
          <ul className="max-h-64 divide-y overflow-y-auto">
            {staged.map((s) => (
              <li key={s.key} className="flex items-center gap-3 px-4 py-2.5 transition-colors hover:bg-surface-2/40">
                <FileIcon ext={extOf(s.file.name)} className="h-7 w-7" />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[13px] font-medium">{s.file.name}</div>
                  {s.path && s.path !== s.file.name ? <div className="truncate font-mono text-[11px] text-subtle">{s.path}</div> : null}
                </div>
                <span className="num shrink-0 font-mono text-[11.5px] text-muted">{formatBytes(s.file.size)}</span>
                {s.issue ? <Badge tone="warn">{s.issue === "too_large" ? "Too large" : "Unsupported"}</Badge> : null}
                <button type="button" disabled={busy} onClick={() => setStaged((p) => p.filter((x) => x.key !== s.key))} className="grid h-6 w-6 place-items-center rounded-md text-subtle transition-colors hover:bg-surface-2 hover:text-fg disabled:opacity-40" aria-label={`Remove ${s.file.name}`}>
                  <X className="h-3.5 w-3.5" />
                </button>
              </li>
            ))}
          </ul>
          <div className="grid gap-4 border-t bg-surface-2/40 px-4 py-4 md:grid-cols-[1fr_auto_auto] md:items-end">
            <Field label="Tags for these files" htmlFor={`up-tags-${kbId}`}>
              <TagInput id={`up-tags-${kbId}`} value={tags} onChange={setTags} disabled={busy} />
            </Field>
            <div className="space-y-1.5">
              <span className="text-[13px] font-medium">Access</span>
              <div>
                <Guarded allowed={canRestrict || acl === "restricted"} reason="Restricting documents requires the kb.permissions permission.">
                  <Segmented
                    label="Document access"
                    value={acl}
                    onChange={(v) => (v === "restricted" && !canRestrict ? undefined : setAcl(v))}
                    options={[
                      { value: "inherit", label: "Inherit", title: "Same access as the knowledge base" },
                      { value: "restricted", label: "Restricted", title: "Only KB managers, the uploader and explicit document grants" },
                    ]}
                  />
                </Guarded>
              </div>
            </div>
            <label className="flex h-9 items-center gap-2.5 text-[13px] font-medium">
              <Switch label="Allow duplicates" checked={allowDuplicate} onCheckedChange={setAllowDuplicate} disabled={busy} />
              Allow duplicates
            </label>
          </div>
          <div className="flex flex-wrap items-center gap-3 border-t px-4 py-3">
            {progress ? (
              <div className="min-w-[200px] flex-1">
                <div className="mb-1.5 flex items-center justify-between text-[12px] text-muted">
                  <span className="inline-flex items-center gap-1.5">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" /> Uploading batch {progress.batch} of {progress.batches}
                  </span>
                  <span className="num font-mono text-[11.5px]">
                    {progress.done} / {progress.total}
                  </span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-surface-2">
                  <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${(progress.done / Math.max(progress.total, 1)) * 100}%` }} />
                </div>
              </div>
            ) : (
              <p className="flex-1 text-[12.5px] text-muted">{kbName ? <>Uploading to <span className="font-medium text-fg">{kbName}</span>. </> : null}Files are scanned, parsed, chunked and embedded in the background.</p>
            )}
            <Button onClick={() => upload.mutate()} loading={busy} disabled={!ready.length || !allowed}>
              <UploadCloud className="h-4 w-4" /> Upload {ready.length ? pluralize(ready.length, "file") : ""}
            </Button>
          </div>
        </div>
      ) : null}

      {results ? <UploadResults results={results} onDismiss={() => setResults(null)} /> : null}
    </div>
  );
}

export function UploadResults({ results, onDismiss }: { results: UploadResult[]; onDismiss?: () => void }) {
  const counts = results.reduce<Record<string, number>>((m, r) => ({ ...m, [r.outcome]: (m[r.outcome] ?? 0) + 1 }), {});
  return (
    <div className="card overflow-hidden animate-fade-in" aria-live="polite">
      <div className="flex flex-wrap items-center gap-2 border-b px-4 py-3">
        <span className="mr-1 text-[13.5px] font-semibold tracking-[-0.01em]">Upload results</span>
        {(Object.keys(OUTCOME) as UploadOutcome[])
          .filter((k) => counts[k])
          .map((k) => (
            <Badge key={k} tone={OUTCOME[k].tone}>
              {counts[k]} {OUTCOME[k].label.toLowerCase()}
            </Badge>
          ))}
        {onDismiss ? (
          <Button size="xs" variant="ghost" className="ml-auto" onClick={onDismiss}>
            <RotateCcw className="h-3.5 w-3.5" /> Upload more
          </Button>
        ) : null}
      </div>
      <ul className="max-h-72 divide-y overflow-y-auto">
        {results.map((r, i) => {
          const o = OUTCOME[r.outcome] ?? OUTCOME.rejected;
          const Icon = o.icon;
          return (
            <li key={i} className="flex items-start gap-3 px-4 py-2.5">
              <Icon className={cn("mt-0.5 h-4 w-4 shrink-0", { ok: "text-ok", warn: "text-warn", danger: "text-danger", neutral: "text-subtle", info: "text-info", accent: "text-accent" }[o.tone])} />
              <div className="min-w-0 flex-1">
                <div className="truncate text-[13px] font-medium">{r.path || r.filename}</div>
                {r.error ? <div className="text-[12px] text-danger">{r.error}</div> : null}
                {r.outcome === "duplicate" ? <div className="text-[12px] text-muted">Identical content already exists in this knowledge base. Enable “Allow duplicates” to keep a second copy.</div> : null}
                {r.outcome === "unchanged" ? <div className="text-[12px] text-muted">Identical to the current version.</div> : null}
              </div>
              <Badge tone={o.tone}>{o.label}</Badge>
              {r.document ? (
                <Link href={`/documents?doc=${r.document.id}`} className="shrink-0 pt-px text-[12px] font-medium text-accent hover:underline">
                  Open
                </Link>
              ) : null}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
