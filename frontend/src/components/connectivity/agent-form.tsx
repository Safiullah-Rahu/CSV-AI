"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Select, Textarea } from "@/components/ui/form";
import { Badge, Skeleton, Switch } from "@/components/ui/primitives";
import { post, put } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";

import { CheckList, FormSection, Notice, invalidatePrefix, useKnowledgeBases } from "./shared";
import type { AIProvider, Agent, AgentTool, DbConnection, ProviderKind } from "./types";

/** Chat-capable, enabled providers (embedding-only kinds cannot drive an agent). */
export function useChatProviders() {
  const providers = useApi<AIProvider[]>("/ai/providers", undefined, { staleTime: 60_000 });
  const kinds = useApi<ProviderKind[]>("/ai/provider-kinds", undefined, { staleTime: 5 * 60_000, meta: { silent: true } });
  return useMemo(() => {
    const embeddingOnly = new Set((kinds.data ?? []).filter((k) => k.embedding_only).map((k) => k.kind));
    const all = providers.data ?? [];
    return { all, chat: all.filter((p) => !embeddingOnly.has(p.kind)), isLoading: providers.isLoading };
  }, [providers.data, providers.isLoading, kinds.data]);
}

function modelOptions(p: AIProvider | undefined): string[] {
  if (!p) return [];
  const ids = [p.default_model, ...p.models.map((m) => (typeof m.id === "string" ? m.id : null)), ...p.suggested_models].filter((x): x is string => Boolean(x));
  return [...new Set(ids)];
}

type State = {
  name: string;
  description: string;
  instructions: string;
  kb_ids: string[];
  connection_ids: string[];
  provider_id: string;
  model: string;
  tools: string[];
  approval_tools: string[];
  max_steps: string;
  enabled: boolean;
};

function initial(a?: Agent | null): State {
  return {
    name: a?.name ?? "",
    description: a?.description ?? "",
    instructions: a?.instructions ?? "",
    kb_ids: a?.kb_ids ?? [],
    connection_ids: a?.connection_ids ?? [],
    provider_id: a?.provider_id ?? "",
    model: a?.model ?? "",
    tools: a?.tools ?? ["kb_search"],
    approval_tools: a?.approval_tools ?? [],
    max_steps: String(a?.limits?.max_steps ?? 6),
    enabled: a?.enabled ?? true,
  };
}

export function AgentFormDialog({ open, onOpenChange, agent, onSaved }: { open: boolean; onOpenChange: (v: boolean) => void; agent?: Agent | null; onSaved?: (a: Agent) => void }) {
  const { can } = useAuth();
  const qc = useQueryClient();
  const kbs = useKnowledgeBases();
  const conns = useApi<DbConnection[]>(can("db.view") ? "/database-connections" : null, undefined, { staleTime: 60_000, meta: { silent: true } });
  const tools = useApi<AgentTool[]>("/agents/tools", undefined, { staleTime: 5 * 60_000 });
  const providers = useChatProviders();
  const [s, setS] = useState<State>(() => initial(agent));
  const [errors, setErrors] = useState<Record<string, string>>({});
  useEffect(() => {
    if (open) {
      setS(initial(agent));
      setErrors({});
    }
  }, [open, agent]);
  const set = <K extends keyof State>(k: K, v: State[K]) => setS((p) => ({ ...p, [k]: v }));
  const provider = providers.chat.find((p) => p.id === s.provider_id);
  const defaultProvider = providers.chat.find((p) => p.is_default && p.enabled) ?? providers.chat.find((p) => p.enabled);
  const models = modelOptions(provider ?? defaultProvider);

  const save = useMutation({
    mutationFn: (body: Record<string, unknown>) => (agent ? put<Agent>(`/agents/${agent.id}`, body) : post<Agent>("/agents", body)),
    onSuccess: (a) => {
      toast.success(agent ? `Saved “${a.name}”` : `Agent “${a.name}” created`);
      invalidatePrefix(qc, "/agents");
      onOpenChange(false);
      onSaved?.(a);
    },
  });

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const errs: Record<string, string> = {};
    if (s.name.trim().length < 2) errs.name = "At least 2 characters.";
    const steps = Number(s.max_steps);
    if (!Number.isInteger(steps) || steps < 1 || steps > 20) errs.max_steps = "Whole number from 1 to 20.";
    if (!s.tools.length) errs.tools = "Enable at least one tool.";
    setErrors(errs);
    if (Object.keys(errs).length) return;
    // PUT replaces the whole agent, so carry over settings this form doesn't edit.
    save.mutate({
      name: s.name.trim(),
      description: s.description.trim(),
      instructions: s.instructions,
      kb_ids: s.kb_ids,
      connection_ids: s.connection_ids,
      provider_id: s.provider_id || null,
      model: s.model.trim() || null,
      tools: s.tools,
      approval_tools: s.approval_tools.filter((t) => s.tools.includes(t)),
      retrieval_config: agent?.retrieval_config ?? {},
      limits: { ...(agent?.limits ?? {}), max_steps: steps },
      enabled: s.enabled,
      workspace_id: agent?.workspace_id ?? null,
    });
  };

  const readableKbs = (kbs.data ?? []).filter((k) => k.access !== "none");
  const queryable = (conns.data ?? []).filter((c) => c.access_mode !== "ingest");

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        wide
        title={agent ? `Edit ${agent.name}` : "New agent"}
        description="Agents run a tool loop with the asking user's permissions, intersected with the scope you set here — they can never reach data the user couldn't."
        footer={
          <>
            <DialogClose asChild>
              <Button variant="secondary">Cancel</Button>
            </DialogClose>
            <Button type="submit" form="agent-form" loading={save.isPending}>
              {agent ? "Save agent" : "Create agent"}
            </Button>
          </>
        }
      >
        <form id="agent-form" onSubmit={submit} className="space-y-5" noValidate>
          <FormSection stacked title="Identity" description="What the agent is called, what it's for, and how it should behave.">
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Name" htmlFor="a-name" error={errors.name}>
              <Input id="a-name" value={s.name} onChange={(e) => set("name", e.target.value)} maxLength={120} placeholder="e.g. Finance analyst" autoFocus />
            </Field>
            <Field label="Description" htmlFor="a-desc" hint="Shown to people choosing an agent.">
              <Input id="a-desc" value={s.description} onChange={(e) => set("description", e.target.value)} maxLength={2000} placeholder="Answers revenue questions from the finance KB and warehouse" />
            </Field>
          </div>
          <Field label="Instructions" htmlFor="a-instr" hint={`${s.instructions.length}/8000 · Added to the system prompt. Tool results are always treated as untrusted data.`}>
            <Textarea id="a-instr" value={s.instructions} onChange={(e) => set("instructions", e.target.value)} maxLength={8000} className="min-h-[110px]" placeholder="Be concise. Always cite sources. Prefer the warehouse for numbers and the policy KB for definitions." />
          </Field>
          </FormSection>

          <FormSection stacked title="Knowledge & data" description="The scope is intersected with the asking user's own access.">
          <div className="grid gap-4 md:grid-cols-2">
            <Field label="Knowledge bases" hint="None selected = every knowledge base the asking user can read.">
              {kbs.isLoading ? (
                <Skeleton className="h-28" />
              ) : (
                <CheckList label="Knowledge bases" options={readableKbs.map((k) => ({ value: k.id, label: k.name, search: k.name, hint: k.description || undefined }))} value={s.kb_ids} onChange={(v) => set("kb_ids", v)} empty={can("kb.view") ? "No knowledge bases yet." : "You can't list knowledge bases (kb.view)."} maxHeight="max-h-40" />
              )}
            </Field>
            <Field label="Database connections" hint="Used by sql_query when no attached dataset answers. Ingest-only connections can't be queried live.">
              {conns.isLoading ? (
                <Skeleton className="h-28" />
              ) : (
                <CheckList
                  label="Database connections"
                  options={queryable.map((c) => ({ value: c.id, label: c.name, search: `${c.name} ${c.provider_label}`, hint: `${c.provider_label} · ${c.selected_tables.length} tables` }))}
                  value={s.connection_ids}
                  onChange={(v) => set("connection_ids", v)}
                  empty={can("db.view") ? "No live-queryable connections." : "You can't list database connections (db.view)."}
                  maxHeight="max-h-40"
                />
              )}
            </Field>
          </div>
          </FormSection>

          <FormSection stacked title="Model" description="Which language model plans the run and calls tools.">
          <div className="grid gap-4 sm:grid-cols-2">
            <Field
              label="AI provider"
              htmlFor="a-prov"
              hint={
                !providers.isLoading && !providers.chat.length ? (
                  <>
                    No chat model is configured. {can("admin.ai_providers") ? <Link href="/settings?tab=ai" className="text-accent hover:underline">Connect one</Link> : "Ask an administrator to connect one"} — agents can&apos;t run without it.
                  </>
                ) : (
                  "The organization default is used unless you pick one."
                )
              }
            >
              <Select id="a-prov" value={s.provider_id} onChange={(e) => set("provider_id", e.target.value)}>
                <option value="">Organization default{defaultProvider ? ` (${defaultProvider.name})` : ""}</option>
                {providers.chat.map((p) => (
                  <option key={p.id} value={p.id} disabled={!p.enabled}>
                    {p.name} · {p.kind_label}
                    {!p.enabled ? " (disabled)" : ""}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Model" htmlFor="a-model" hint={models.length ? "Pick a suggestion or type any model name the provider serves." : "Leave empty to use the provider's default model."}>
              <Input id="a-model" list="agent-models" value={s.model} onChange={(e) => set("model", e.target.value)} placeholder={(provider ?? defaultProvider)?.default_model ?? "provider default"} spellCheck={false} autoComplete="off" />
              <datalist id="agent-models">
                {models.map((m) => (
                  <option key={m} value={m} />
                ))}
              </datalist>
            </Field>
          </div>
          </FormSection>

          <FormSection stacked title="Tools & approvals" description="Tools the agent may call, and which of them pause for sign-off.">
          <div className="grid gap-4 md:grid-cols-2">
            <Field label="Tools" error={errors.tools}>
              {tools.isLoading ? (
                <Skeleton className="h-40" />
              ) : (
                <CheckList
                  label="Tools"
                  options={(tools.data ?? []).map((t) => ({ value: t.key, label: <span className="font-mono text-[12.5px]">{t.key}</span>, search: t.key, hint: t.description }))}
                  value={s.tools}
                  onChange={(v) => setS((p) => ({ ...p, tools: v, approval_tools: p.approval_tools.filter((x) => v.includes(x)) }))}
                  maxHeight="max-h-64"
                />
              )}
            </Field>
            <div className="space-y-4">
              <Field label="Require approval before running" hint="The run pauses and creates an approval request instead of executing these tools.">
                <CheckList
                  label="Approval-required tools"
                  options={s.tools.map((t) => ({ value: t, label: <span className="font-mono text-[12.5px]">{t}</span>, search: t }))}
                  value={s.approval_tools}
                  onChange={(v) => set("approval_tools", v)}
                  empty="Enable tools first."
                  maxHeight="max-h-40"
                />
              </Field>
              <Field label="Max steps" htmlFor="a-steps" error={errors.max_steps} hint="Tool calls per run before the agent must answer (1–20).">
                <Input id="a-steps" inputMode="numeric" value={s.max_steps} onChange={(e) => set("max_steps", e.target.value.replace(/[^\d]/g, ""))} className="w-28" />
              </Field>
            </div>
          </div>
          </FormSection>

          <div className="flex items-center justify-between gap-4 rounded-xl border bg-surface-2/40 px-3.5 py-3">
            <div>
              <div className="flex items-center gap-2 text-[13px] font-medium">
                Enabled {!s.enabled ? <Badge>disabled</Badge> : null}
              </div>
              <p className="mt-0.5 text-[12px] text-muted">Disabled agents stay configured but cannot be run.</p>
            </div>
            <Switch checked={s.enabled} onCheckedChange={(v) => set("enabled", v)} label="Agent enabled" />
          </div>
          {s.approval_tools.length ? <Notice tone="info">Approving a paused tool records the decision; the run doesn&apos;t resume by itself.</Notice> : null}
        </form>
      </DialogContent>
    </Dialog>
  );
}
