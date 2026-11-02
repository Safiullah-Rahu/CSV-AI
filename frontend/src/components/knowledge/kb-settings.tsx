"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Lock, RefreshCw, Save, Trash2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState, type ReactNode } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/dialog";
import { Field, Input, Select, Textarea } from "@/components/ui/form";
import { Card, CardBody, Dot, Skeleton, Switch } from "@/components/ui/primitives";
import { del, patch } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatNumber, pluralize } from "@/lib/utils";

import { ChunkingFields, chunkingError, chunkingPayload, EmbeddingFields, embeddingError, embeddingPayload, GroundingField, RetrievalFields, retrievalError, withChunkDefaults, withRetrievalDefaults } from "./kb-config";
import { Guarded, invalidatePaths, kbAbilities, needs, SettingsRow, TagInput, usePerms, VISIBILITY_OPTIONS } from "./shared";
import type { ChunkingConfig, Department, EmbeddingConfig, GroundingPolicy, KBOptions, KBUpdateResult, KnowledgeBase, Ok, RetrievalConfig, Visibility, Workspace } from "./types";

function useSave(kb: KnowledgeBase, onDone?: (r: KBUpdateResult) => void) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Record<string, unknown>) => patch<KBUpdateResult>(`/knowledge-bases/${kb.id}`, body),
    onSuccess: (r) => {
      if (r.reindex_scheduled) {
        toast.success("Settings saved — re-index scheduled", {
          description: kb.stats.documents ? `${pluralize(kb.stats.documents, "document")} will be re-chunked and re-embedded in the background.` : "New documents will be indexed with these settings.",
        });
      } else {
        toast.success("Settings saved");
      }
      invalidatePaths(qc, "/knowledge-bases", "/documents");
      onDone?.(r);
    },
  });
}

export function KBSettingsTab({ kb }: { kb: KnowledgeBase }) {
  const perms = usePerms();
  const ab = kbAbilities(kb, perms);
  const options = useApi<KBOptions>("/knowledge-bases/options", undefined, {
    staleTime: 5 * 60_000,
  });
  const [reindexAt, setReindexAt] = useState<string | null>(null);
  const markReindex = (r: KBUpdateResult) => r.reindex_scheduled && setReindexAt(new Date().toLocaleTimeString(undefined, { timeStyle: "short" }));

  return (
    <div className="space-y-8">
      {reindexAt ? (
        <div className="flex items-start gap-3 rounded-xl border border-info/25 bg-info/[0.06] px-4 py-3 text-[13px] animate-fade-in" role="status">
          <RefreshCw className="mt-0.5 h-4 w-4 shrink-0 animate-spin text-info" />
          <div>
            <div className="font-medium">Re-index scheduled at {reindexAt}</div>
            <div className="text-muted">Documents are re-processed with the new configuration in the background. Progress shows in the Documents tab and the activity feed.</div>
          </div>
        </div>
      ) : null}
      <GeneralSection key={`g-${kb.config_version}`} kb={kb} allowed={ab.edit} />
      {options.data ? (
        <>
          <IngestionSection key={`i-${kb.config_version}-${JSON.stringify(kb.embedding_config)}`} kb={kb} options={options.data} allowed={ab.configureIngestion} onSaved={markReindex} />
          <RetrievalSection key={`r-${kb.config_version}`} kb={kb} options={options.data} allowed={ab.configureRetrieval} />
        </>
      ) : (
        <>
          <SettingsRow title="Ingestion" description="Loading options…">
            <Skeleton className="h-72 rounded-xl" />
          </SettingsRow>
          <SettingsRow title="Retrieval & answers" description="Loading options…">
            <Skeleton className="h-72 rounded-xl" />
          </SettingsRow>
        </>
      )}
      <DangerZone kb={kb} allowed={ab.remove} />
    </div>
  );
}

function SectionFooter({ allowed, reason, dirty, saving, onSave, onReset, children }: { allowed: boolean; reason: string; dirty: boolean; saving: boolean; onSave: () => void; onReset: () => void; children?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-b-xl border-t bg-surface-2/40 px-5 py-3">
      <div className="min-w-0 flex-1 text-[12.5px] text-muted">
        {children ??
          (allowed ? (
            dirty ? (
              <span className="inline-flex items-center gap-2 text-fg">
                <Dot tone="warn" /> Unsaved changes
              </span>
            ) : (
              "All changes saved."
            )
          ) : (
            <span className="inline-flex items-start gap-1.5">
              <Lock className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {reason}
            </span>
          ))}
      </div>
      {dirty ? (
        <Button variant="ghost" size="sm" onClick={onReset} disabled={saving}>
          Discard
        </Button>
      ) : null}
      <Guarded allowed={allowed} reason={reason}>
        <Button size="sm" onClick={onSave} disabled={!allowed || !dirty} loading={saving}>
          <Save className="h-3.5 w-3.5" /> Save
        </Button>
      </Guarded>
    </div>
  );
}

function ReadOnlyNote({ allowed }: { allowed: boolean }) {
  if (allowed) return null;
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border bg-surface-2/60 px-2 py-[3px] text-[11.5px] font-medium text-muted">
      <Lock className="h-3 w-3" /> Read only
    </span>
  );
}

/** A small heading for a group of fields inside a settings card. */
function Group({ title, children, className }: { title: string; children: ReactNode; className?: string }) {
  return (
    <div className={cn("px-5 py-5", className)}>
      <div className="label mb-3">{title}</div>
      {children}
    </div>
  );
}

/* ---------------------------------------------------------------- general */

function GeneralSection({ kb, allowed }: { kb: KnowledgeBase; allowed: boolean }) {
  const { can } = useAuth();
  const perms = usePerms();
  const workspaces = useApi<Workspace[]>(can("workspace.view") ? "/workspaces" : null);
  const departments = useApi<Department[]>(can("org.view") ? "/departments" : null);
  const initial = {
    name: kb.name,
    description: kb.description,
    visibility: kb.visibility,
    tags: kb.tags,
    category: kb.category,
    workspace_id: kb.workspace_id ?? "",
    department_id: kb.department_id ?? "",
  };
  const [v, setV] = useState(initial);
  const [error, setError] = useState<string | null>(null);
  const save = useSave(kb);
  const set = <K extends keyof typeof v>(k: K, val: (typeof v)[K]) => setV((s) => ({ ...s, [k]: val }));

  const changed: Record<string, unknown> = {};
  if (v.name.trim() !== kb.name) changed.name = v.name.trim();
  if (v.description !== kb.description) changed.description = v.description;
  if (v.visibility !== kb.visibility) changed.visibility = v.visibility;
  if (JSON.stringify(v.tags) !== JSON.stringify(kb.tags)) changed.tags = v.tags;
  if (v.category.trim() !== kb.category) changed.category = v.category.trim();
  if (v.workspace_id && v.workspace_id !== (kb.workspace_id ?? "")) changed.workspace_id = v.workspace_id;
  if (v.department_id && v.department_id !== (kb.department_id ?? "")) changed.department_id = v.department_id;
  const dirty = Object.keys(changed).length > 0;
  const wsChoices = (workspaces.data ?? []).filter((w) => perms.orgHas("kb.create") || perms.workspacesWith("kb.create").includes(w.id) || w.id === kb.workspace_id);

  const submit = () => {
    if (v.name.trim().length < 2) return setError("Name must be at least 2 characters.");
    if (v.visibility === "workspace" && !v.workspace_id) return setError("Workspace visibility needs a workspace.");
    setError(null);
    save.mutate(changed);
  };

  return (
    <SettingsRow title="General" description="Name, description, ownership scope and who can see this knowledge base." aside={<ReadOnlyNote allowed={allowed} />}>
      <Card>
        <CardBody className="grid gap-4 px-5 py-5 sm:grid-cols-2">
          <Field label="Name" htmlFor="set-name">
            <Input id="set-name" value={v.name} onChange={(e) => set("name", e.target.value)} maxLength={160} disabled={!allowed} />
          </Field>
          <Field label="Category" htmlFor="set-cat">
            <Input id="set-cat" value={v.category} onChange={(e) => set("category", e.target.value)} maxLength={80} disabled={!allowed} placeholder="Optional" />
          </Field>
          <Field label="Description" htmlFor="set-desc" className="sm:col-span-2">
            <Textarea id="set-desc" value={v.description} onChange={(e) => set("description", e.target.value)} maxLength={4000} rows={3} disabled={!allowed} />
          </Field>
          {can("workspace.view") ? (
            <Field label="Workspace" htmlFor="set-ws" hint={kb.workspace_id ? "Can be moved to another workspace, but not unassigned." : "Assigning a workspace requires kb.create in it."}>
              <Select id="set-ws" value={v.workspace_id} onChange={(e) => set("workspace_id", e.target.value)} disabled={!allowed || workspaces.isLoading}>
                {!kb.workspace_id ? <option value="">No workspace</option> : null}
                {wsChoices.map((w) => (
                  <option key={w.id} value={w.id}>
                    {w.name}
                  </option>
                ))}
              </Select>
            </Field>
          ) : null}
          {can("org.view") ? (
            <Field label="Department" htmlFor="set-dep" hint={kb.department_id ? "Can be changed, but not unassigned." : undefined}>
              <Select id="set-dep" value={v.department_id} onChange={(e) => set("department_id", e.target.value)} disabled={!allowed || departments.isLoading}>
                {!kb.department_id ? <option value="">No department</option> : null}
                {(departments.data ?? []).map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </Select>
            </Field>
          ) : null}
          <Field label="Tags" htmlFor="set-tags" className="sm:col-span-2">
            <TagInput id="set-tags" value={v.tags} onChange={(t) => set("tags", t)} disabled={!allowed} />
          </Field>
          <fieldset className="sm:col-span-2" disabled={!allowed}>
            <legend className="mb-2 text-[13px] font-medium">Visibility</legend>
            <div className="grid gap-2 sm:grid-cols-3">
              {VISIBILITY_OPTIONS.map((o) => {
                const off = o.value === "workspace" && !v.workspace_id;
                const Icon = o.icon;
                return (
                  <label
                    key={o.value}
                    htmlFor={`set-vis-${o.value}`}
                    className={cn(
                      "flex cursor-pointer flex-col gap-1.5 rounded-lg border p-3 transition-[background-color,border-color,box-shadow]",
                      v.visibility === o.value ? "border-accent/50 bg-accent/[0.05] shadow-[0_0_0_3px_hsl(var(--accent)/0.08)]" : "hover:border-border-strong hover:bg-surface-2/50",
                      (off || !allowed) && "cursor-not-allowed opacity-50",
                    )}
                  >
                    <span className="flex items-center gap-2 text-[13px] font-medium">
                      <input id={`set-vis-${o.value}`} type="radio" name="set-visibility" className="accent-[hsl(var(--accent))]" checked={v.visibility === o.value} disabled={off} onChange={() => set("visibility", o.value as Visibility)} />
                      <Icon className="h-3.5 w-3.5 text-subtle" /> {o.label}
                    </span>
                    <span className="text-[12px] leading-relaxed text-muted">{off ? "Assign a workspace first." : o.hint}</span>
                  </label>
                );
              })}
            </div>
          </fieldset>
          {error ? (
            <p className="text-[12px] text-danger sm:col-span-2" role="alert">
              {error}
            </p>
          ) : null}
        </CardBody>
        <SectionFooter
          allowed={allowed}
          reason={needs("kb.edit", "write")}
          dirty={dirty}
          saving={save.isPending}
          onSave={submit}
          onReset={() => {
            setV(initial);
            setError(null);
          }}
        />
      </Card>
    </SettingsRow>
  );
}

/* -------------------------------------------------------------- ingestion */

const sameEmbedding = (a: EmbeddingConfig, b: EmbeddingConfig) => {
  const pa = embeddingPayload(a);
  const pb = embeddingPayload(b);
  return pa.provider === pb.provider && (pa.model ?? "") === (pb.model ?? "") && (pa.dim ?? null) === (pb.dim ?? null) && (pa.provider_id ?? "") === (pb.provider_id ?? "");
};

function IngestionSection({ kb, options, allowed, onSaved }: { kb: KnowledgeBase; options: KBOptions; allowed: boolean; onSaved: (r: KBUpdateResult) => void }) {
  const initialEmb: EmbeddingConfig = {
    provider: kb.embedding_config.provider || "hashing",
    model: kb.embedding_config.model,
    dim: kb.embedding_config.dim,
    provider_id: (kb.embedding_config.provider_id as string | undefined) ?? undefined,
  };
  const initialChunk = withChunkDefaults(options, kb.chunking_config);
  const [emb, setEmb] = useState<EmbeddingConfig>(initialEmb);
  const [chunk, setChunk] = useState<ChunkingConfig>(initialChunk);
  const [reindex, setReindex] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const save = useSave(kb, onSaved);

  const embChanged = !sameEmbedding(emb, initialEmb);
  const chunkChanged = JSON.stringify(chunkingPayload(chunk)) !== JSON.stringify(chunkingPayload(initialChunk));
  const dirty = embChanged || chunkChanged;
  const docs = kb.stats.documents ?? 0;

  const submit = () => {
    const err = (embChanged && embeddingError(emb)) || (chunkChanged && chunkingError(chunk)) || null;
    setError(err);
    if (err) return;
    const body: Record<string, unknown> = {};
    if (embChanged) body.embedding_config = embeddingPayload(emb);
    if (chunkChanged) {
      body.chunking_config = chunkingPayload(chunk);
      body.reindex = reindex;
    }
    save.mutate(body);
  };

  return (
    <SettingsRow title="Ingestion" description="How documents are split and embedded. Applies to documents indexed after saving." aside={<ReadOnlyNote allowed={allowed} />}>
      <Card>
        <div className="divide-y">
          <Group title="Embeddings">
            <EmbeddingFields idPrefix="set" value={emb} options={options} onChange={setEmb} disabled={!allowed} />
            {embChanged ? (
              <p className="mt-3 flex items-start gap-1.5 rounded-lg border border-warn/25 bg-warn/[0.06] px-3 py-2 text-[12.5px] text-warn">
                <AlertTriangle className="mt-px h-3.5 w-3.5 shrink-0" /> Changing the embedding model re-embeds every document
                {docs ? ` (${pluralize(docs, "document")}, ${formatNumber(kb.stats.chunks)} chunks)` : ""}.
              </p>
            ) : null}
          </Group>
          <Group title="Chunking">
            <ChunkingFields idPrefix="set" value={chunk} options={options} onChange={setChunk} disabled={!allowed} />
            {chunkChanged ? (
              <div className="mt-3 flex items-center justify-between gap-3 rounded-lg border border-accent/25 bg-accent/[0.05] px-3.5 py-3">
                <div>
                  <div className="text-[13px] font-medium">Re-index existing documents</div>
                  <div className="text-[12px] text-muted">{reindex ? "All documents are re-chunked with the new settings." : "Only new or re-processed documents use the new settings."}</div>
                </div>
                <Switch label="Re-index existing documents" checked={reindex} onCheckedChange={setReindex} />
              </div>
            ) : null}
          </Group>
          {error ? (
            <p className="px-5 py-3 text-[12px] text-danger" role="alert">
              {error}
            </p>
          ) : null}
        </div>
        <SectionFooter
          allowed={allowed}
          reason={needs("kb.configure_ingestion", "manage")}
          dirty={dirty}
          saving={save.isPending}
          onSave={submit}
          onReset={() => {
            setEmb(initialEmb);
            setChunk(initialChunk);
            setError(null);
          }}
        />
      </Card>
    </SettingsRow>
  );
}

/* -------------------------------------------------------------- retrieval */

function RetrievalSection({ kb, options, allowed }: { kb: KnowledgeBase; options: KBOptions; allowed: boolean }) {
  const initialRet = withRetrievalDefaults(options, kb.retrieval_config);
  const initialPolicy = (kb.generation_config?.grounding_policy ?? "") as GroundingPolicy | "";
  const [ret, setRet] = useState<RetrievalConfig>(initialRet);
  const [policy, setPolicy] = useState<GroundingPolicy | "">(initialPolicy);
  const [error, setError] = useState<string | null>(null);
  const save = useSave(kb);

  const retDiff = (Object.keys(ret) as (keyof RetrievalConfig)[]).filter((k) => JSON.stringify(ret[k]) !== JSON.stringify(initialRet[k]));
  const dirty = retDiff.length > 0 || policy !== initialPolicy;

  const submit = () => {
    const err = retrievalError(ret);
    setError(err);
    if (err) return;
    const body: Record<string, unknown> = {};
    if (retDiff.length) {
      // The API stores retrieval_config as given: keep existing overrides, add the changed keys.
      body.retrieval_config = {
        ...kb.retrieval_config,
        ...Object.fromEntries(retDiff.map((k) => [k, ret[k]])),
      };
    }
    if (policy !== initialPolicy) {
      const rest = { ...(kb.generation_config || {}) };
      delete rest.grounding_policy;
      body.generation_config = policy ? { ...rest, grounding_policy: policy } : rest;
    }
    save.mutate(body);
  };

  return (
    <SettingsRow title="Retrieval & answers" description="How passages are found and ranked for chat, and how strictly answers stick to them." aside={<ReadOnlyNote allowed={allowed} />}>
      <Card>
        <div className="divide-y">
          <Group title="Retrieval">
            <RetrievalFields idPrefix="set" value={ret} onChange={setRet} disabled={!allowed} />
          </Group>
          <Group title="Answers">
            <GroundingField idPrefix="set" value={policy} onChange={setPolicy} disabled={!allowed} />
          </Group>
          {error ? (
            <p className="px-5 py-3 text-[12px] text-danger" role="alert">
              {error}
            </p>
          ) : null}
        </div>
        <SectionFooter
          allowed={allowed}
          reason={needs("kb.configure_retrieval", "manage")}
          dirty={dirty}
          saving={save.isPending}
          onSave={submit}
          onReset={() => {
            setRet(initialRet);
            setPolicy(initialPolicy);
            setError(null);
          }}
        />
      </Card>
    </SettingsRow>
  );
}

/* ------------------------------------------------------------ danger zone */

function DangerZone({ kb, allowed }: { kb: KnowledgeBase; allowed: boolean }) {
  const qc = useQueryClient();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const remove = useMutation({
    mutationFn: () => del<Ok>(`/knowledge-bases/${kb.id}`, { confirm_name: kb.name }),
    onSuccess: (r) => {
      toast.success(r?.detail || `Deleted “${kb.name}”`);
      setOpen(false);
      router.push("/knowledge-bases");
      invalidatePaths(qc, "/knowledge-bases", "/documents", "/datasets", "/connections", "/onboarding");
    },
  });
  return (
    <SettingsRow title={<span className="text-danger">Danger zone</span>} description="Irreversible actions. Deleting removes the knowledge base for everyone.">
      <Card className="border-danger/30">
        <CardBody className="flex flex-wrap items-center gap-4 px-5 py-5">
          <div className="min-w-0 flex-1 text-[13px]">
            <div className="font-medium">Delete this knowledge base</div>
            <div className="mt-0.5 leading-relaxed text-muted">
              Permanently deletes {pluralize(kb.stats.documents ?? 0, "document")}, {formatNumber(kb.stats.chunks ?? 0)} chunks, their embeddings, datasets and stored files. This cannot be undone.
            </div>
          </div>
          <Guarded allowed={allowed} reason={needs("kb.delete", "manage")}>
            <Button variant="danger" disabled={!allowed} onClick={() => setOpen(true)}>
              <Trash2 className="h-4 w-4" /> Delete knowledge base
            </Button>
          </Guarded>
        </CardBody>
        <ConfirmDialog
          open={open}
          onOpenChange={setOpen}
          title={`Delete “${kb.name}”?`}
          description="Everything in it is deleted for everyone, including chunks, embeddings, datasets and stored originals. Past answers keep their text but lose the quoted snippets."
          confirmLabel="Delete permanently"
          requireText={kb.name}
          loading={remove.isPending}
          onConfirm={() => remove.mutate()}
        />
      </Card>
    </SettingsRow>
  );
}
