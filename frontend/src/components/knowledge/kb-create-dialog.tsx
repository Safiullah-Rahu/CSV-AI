"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, SlidersHorizontal } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Select, Textarea } from "@/components/ui/form";
import { Skeleton } from "@/components/ui/primitives";
import { post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn } from "@/lib/utils";

import {
  ChunkingFields,
  chunkingError,
  chunkingPayload,
  EmbeddingFields,
  embeddingError,
  embeddingPayload,
  GroundingField,
  RetrievalFields,
  retrievalError,
} from "./kb-config";
import { invalidatePaths, TagInput, usePerms, VISIBILITY_OPTIONS } from "./shared";
import type { ChunkingConfig, Department, EmbeddingConfig, GroundingPolicy, KBOptions, KnowledgeBase, RetrievalConfig, Visibility, Workspace } from "./types";

export function CreateKBDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  const router = useRouter();
  const qc = useQueryClient();
  const { can } = useAuth();
  const perms = usePerms();
  const orgLevelCreate = perms.orgHas("kb.create");
  const wsAllowed = new Set(perms.workspacesWith("kb.create"));

  const options = useApi<KBOptions>(open ? "/knowledge-bases/options" : null, undefined, { staleTime: 5 * 60_000 });
  const workspaces = useApi<Workspace[]>(open && can("workspace.view") ? "/workspaces" : null);
  const departments = useApi<Department[]>(open && can("org.view") ? "/departments" : null);
  const workspaceChoices = (workspaces.data ?? []).filter((w) => orgLevelCreate || wsAllowed.has(w.id));

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [workspaceId, setWorkspaceId] = useState("");
  const [departmentId, setDepartmentId] = useState("");
  const [visibility, setVisibility] = useState<Visibility>("restricted");
  const [tags, setTags] = useState<string[]>([]);
  const [category, setCategory] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [touched, setTouched] = useState({ embedding: false, chunking: false, retrieval: false, grounding: false });
  const [embedding, setEmbedding] = useState<EmbeddingConfig>({ provider: "hashing", dim: 768 });
  const [chunking, setChunking] = useState<ChunkingConfig | null>(null);
  const [retrieval, setRetrieval] = useState<RetrievalConfig | null>(null);
  const [grounding, setGrounding] = useState<GroundingPolicy | "">("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (options.data && !chunking) setChunking({ ...options.data.chunking_defaults });
    if (options.data && !retrieval) setRetrieval({ ...options.data.retrieval_defaults });
  }, [options.data, chunking, retrieval]);

  useEffect(() => {
    // Workspace-scoped creators must pick one of their workspaces.
    if (open && !orgLevelCreate && !workspaceId && workspaceChoices.length) setWorkspaceId(workspaceChoices[0].id);
  }, [open, orgLevelCreate, workspaceId, workspaceChoices]);

  const reset = () => {
    setName("");
    setDescription("");
    setWorkspaceId("");
    setDepartmentId("");
    setVisibility("restricted");
    setTags([]);
    setCategory("");
    setAdvanced(false);
    setTouched({ embedding: false, chunking: false, retrieval: false, grounding: false });
    setEmbedding({ provider: "hashing", dim: 768 });
    setChunking(options.data ? { ...options.data.chunking_defaults } : null);
    setRetrieval(options.data ? { ...options.data.retrieval_defaults } : null);
    setGrounding("");
    setError(null);
  };

  const create = useMutation({
    mutationFn: () => {
      const body: Record<string, unknown> = {
        name: name.trim(),
        description: description.trim(),
        visibility,
        tags,
        category: category.trim(),
        workspace_id: workspaceId || null,
        department_id: departmentId || null,
      };
      if (touched.embedding) body.embedding_config = embeddingPayload(embedding);
      if (touched.chunking && chunking) body.chunking_config = chunkingPayload(chunking);
      if (touched.retrieval && retrieval) body.retrieval_config = { mode: retrieval.mode, top_k: retrieval.top_k, rerank: retrieval.rerank };
      if (touched.grounding && grounding) body.generation_config = { grounding_policy: grounding };
      return post<KnowledgeBase>("/knowledge-bases", body);
    },
    onSuccess: (kb) => {
      toast.success(`Knowledge base “${kb.name}” created`);
      invalidatePaths(qc, "/knowledge-bases", "/onboarding", "/connections");
      onOpenChange(false);
      reset();
      router.push(`/knowledge-bases/${kb.id}`);
    },
  });

  const submit = () => {
    const n = name.trim();
    let err: string | null = null;
    if (n.length < 2) err = "Name must be at least 2 characters.";
    else if (!orgLevelCreate && !workspaceId) err = "Choose a workspace where you are allowed to create knowledge bases.";
    else if (visibility === "workspace" && !workspaceId) err = "Workspace visibility needs a workspace.";
    else if (touched.embedding) err = embeddingError(embedding);
    if (!err && touched.chunking && chunking) err = chunkingError(chunking);
    if (!err && touched.retrieval && retrieval) err = retrievalError(retrieval);
    setError(err);
    if (!err) create.mutate();
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => {
        onOpenChange(v);
        if (!v && !create.isPending) reset();
      }}
    >
      <DialogContent
        wide
        title="New knowledge base"
        description="A searchable collection of documents and datasets with its own access rules, chunking and retrieval settings."
        footer={
          <>
            <DialogClose asChild>
              <Button variant="secondary">Cancel</Button>
            </DialogClose>
            <Button onClick={submit} loading={create.isPending}>
              Create knowledge base
            </Button>
          </>
        }
      >
        <form
          className="space-y-5"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Name" htmlFor="kb-name" className="sm:col-span-2">
              <Input id="kb-name" value={name} onChange={(e) => setName(e.target.value)} maxLength={160} placeholder="e.g. HR policies 2026" autoFocus required />
            </Field>
            <Field label="Description" htmlFor="kb-desc" className="sm:col-span-2" hint="Shown to people choosing a knowledge base in chat.">
              <Textarea id="kb-desc" value={description} onChange={(e) => setDescription(e.target.value)} maxLength={4000} rows={3} placeholder="What's in here and who it's for" />
            </Field>
            {can("workspace.view") ? (
              <Field label="Workspace" htmlFor="kb-ws" hint={!orgLevelCreate ? "Limited to workspaces where you can create knowledge bases." : undefined}>
                {workspaces.isLoading ? (
                  <Skeleton className="h-9" />
                ) : (
                  <Select id="kb-ws" value={workspaceId} onChange={(e) => setWorkspaceId(e.target.value)}>
                    {orgLevelCreate ? <option value="">No workspace (organization)</option> : null}
                    {workspaceChoices.map((w) => (
                      <option key={w.id} value={w.id}>
                        {w.name}
                      </option>
                    ))}
                  </Select>
                )}
              </Field>
            ) : null}
            {can("org.view") ? (
              <Field label="Department" htmlFor="kb-dep">
                {departments.isLoading ? (
                  <Skeleton className="h-9" />
                ) : (
                  <Select id="kb-dep" value={departmentId} onChange={(e) => setDepartmentId(e.target.value)}>
                    <option value="">No department</option>
                    {(departments.data ?? []).map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.name}
                      </option>
                    ))}
                  </Select>
                )}
              </Field>
            ) : null}
          </div>

          <fieldset>
            <legend className="mb-2 text-[13px] font-medium">Visibility</legend>
            <div className="grid gap-2 sm:grid-cols-3">
              {VISIBILITY_OPTIONS.map((v) => {
                const disabled = v.value === "workspace" && !workspaceId;
                const Icon = v.icon;
                return (
                  <label key={v.value} htmlFor={`kb-vis-${v.value}`} className={cn("flex cursor-pointer flex-col gap-1.5 rounded-lg border p-3 transition-[background-color,border-color,box-shadow]", visibility === v.value ? "border-accent/50 bg-accent/[0.05] shadow-[0_0_0_3px_hsl(var(--accent)/0.08)]" : "hover:border-border-strong hover:bg-surface-2/50", disabled && "cursor-not-allowed opacity-50")}>
                    <span className="flex items-center gap-2 text-[13px] font-medium">
                      <input id={`kb-vis-${v.value}`} type="radio" name="kb-visibility" className="accent-[hsl(var(--accent))]" checked={visibility === v.value} disabled={disabled} onChange={() => setVisibility(v.value)} />
                      <Icon className="h-3.5 w-3.5 text-subtle" /> {v.label}
                    </span>
                    <span className="text-[12px] leading-relaxed text-muted">{disabled ? "Choose a workspace first." : v.hint}</span>
                  </label>
                );
              })}
            </div>
          </fieldset>

          <div className="grid gap-4 sm:grid-cols-[2fr_1fr]">
            <Field label="Tags" htmlFor="kb-tags">
              <TagInput id="kb-tags" value={tags} onChange={setTags} />
            </Field>
            <Field label="Category" htmlFor="kb-cat">
              <Input id="kb-cat" value={category} onChange={(e) => setCategory(e.target.value)} maxLength={80} placeholder="Optional" />
            </Field>
          </div>

          <div className="overflow-hidden rounded-xl border">
            <button type="button" onClick={() => setAdvanced((a) => !a)} aria-expanded={advanced} className="flex w-full items-center gap-2.5 px-4 py-3 text-left text-[13px] font-medium transition-colors hover:bg-surface-2/50">
              <span className="grid h-7 w-7 place-items-center rounded-lg border bg-surface-2/60 text-muted">
                <SlidersHorizontal className="h-3.5 w-3.5" />
              </span>
              <span className="flex-1">
                Advanced
                <span className="ml-2 font-normal text-muted">Embeddings, chunking, retrieval and grounding</span>
              </span>
              <ChevronDown className={cn("h-4 w-4 text-subtle transition-transform", advanced && "rotate-180")} />
            </button>
            {advanced ? (
              <div className="space-y-6 border-t bg-surface-2/25 px-4 py-5">
                {!options.data || !chunking || !retrieval ? (
                  <div className="space-y-2">
                    <Skeleton className="h-9" />
                    <Skeleton className="h-9" />
                    <Skeleton className="h-9" />
                  </div>
                ) : (
                  <>
                    <p className="text-[12.5px] text-muted">Untouched sections use the platform defaults. Everything here can be changed later in Settings (changing embeddings re-indexes the knowledge base).</p>
                    <ConfigSection title="Embeddings" dirty={touched.embedding}>
                      <EmbeddingFields idPrefix="new" value={embedding} options={options.data} onChange={(v) => { setEmbedding(v); setTouched((t) => ({ ...t, embedding: true })); }} />
                    </ConfigSection>
                    <ConfigSection title="Chunking" dirty={touched.chunking}>
                      <ChunkingFields idPrefix="new" value={chunking} options={options.data} onChange={(v) => { setChunking(v); setTouched((t) => ({ ...t, chunking: true })); }} />
                    </ConfigSection>
                    <ConfigSection title="Retrieval" dirty={touched.retrieval}>
                      <RetrievalFields idPrefix="new" compact value={retrieval} onChange={(v) => { setRetrieval(v); setTouched((t) => ({ ...t, retrieval: true })); }} />
                    </ConfigSection>
                    <ConfigSection title="Answers" dirty={touched.grounding}>
                      <GroundingField idPrefix="new" value={grounding} onChange={(v) => { setGrounding(v); setTouched((t) => ({ ...t, grounding: true })); }} />
                    </ConfigSection>
                  </>
                )}
              </div>
            ) : null}
          </div>

          {error ? <p className="rounded-lg border border-danger/25 bg-danger/5 px-3 py-2 text-[13px] text-danger" role="alert">{error}</p> : null}
          <button type="submit" className="hidden" aria-hidden tabIndex={-1} />
        </form>
      </DialogContent>
    </Dialog>
  );
}

function ConfigSection({ title, dirty, children }: { title: string; dirty: boolean; children: ReactNode }) {
  return (
    <section>
      <div className="mb-3 flex items-center gap-2">
        <span className="label">{title}</span>
        {dirty ? <span className="rounded-full bg-accent/10 px-1.5 py-px text-[10.5px] font-medium text-accent ring-1 ring-inset ring-accent/20">customized</span> : <span className="text-[10.5px] text-subtle">default</span>}
      </div>
      {children}
    </section>
  );
}
