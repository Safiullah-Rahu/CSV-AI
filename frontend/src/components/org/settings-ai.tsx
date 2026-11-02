"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Activity, Bot, CheckCircle2, Pencil, Plus, RefreshCw, Star, Trash2, XCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { Callout, CancelButton, GatedButton, permissionHint, When } from "@/components/org/kit";
import type { AIProvider, ProviderKind } from "@/components/org/types";
import { Button } from "@/components/ui/button";
import { ConfirmDialog, Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Select } from "@/components/ui/form";
import { Badge, Card, EmptyState, Mono, SectionTitle, Skeleton, StatusBadge, Switch } from "@/components/ui/primitives";
import { del, get, patch, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatNumber } from "@/lib/utils";

const URL_REQUIRED = new Set(["azure_openai", "vllm", "openai_compatible", "custom"]);
// Provider types whose API publishes a model list (live discovery; refreshed daily server-side).
const LISTABLE = new Set(["openai", "anthropic", "gemini", "ollama", "vllm", "openai_compatible", "huggingface", "custom"]);
type Health = { ok: boolean; latency_ms: number; error: string | null; at: string };

export function AIProvidersTab() {
  const { can } = useAuth();
  const qc = useQueryClient();
  const manage = can("admin.ai_providers");
  const providers = useApi<AIProvider[]>("/ai/providers");
  const platform = useApi<{ available: boolean; provider: string | null; model: string | null }>("/ai/platform-default", undefined, { staleTime: 5 * 60_000, meta: { silent: true } });
  const hasChatProvider = (providers.data ?? []).some((p) => p.enabled && !kinds.data?.find((k) => k.kind === p.kind)?.embedding_only);
  const kinds = useApi<ProviderKind[]>("/ai/provider-kinds", undefined, { staleTime: 10 * 60_000 });
  const [editing, setEditing] = useState<AIProvider | "new" | null>(null);
  const [deleting, setDeleting] = useState<AIProvider | null>(null);
  const [health, setHealth] = useState<Record<string, Health>>({});
  const byId = useMemo(() => new Map((providers.data ?? []).map((p) => [p.id, p])), [providers.data]);
  const invalidate = () => qc.invalidateQueries({ queryKey: ["/ai/providers"] });

  const test = useMutation({
    mutationFn: (p: AIProvider) => post<{ ok: boolean; latency_ms: number; error: string | null }>(`/ai/providers/${p.id}/health`),
    onSuccess: (r, p) => {
      setHealth((h) => ({ ...h, [p.id]: { ...r, at: new Date().toISOString() } }));
      if (r.ok) toast.success(`${p.name} is reachable`, { description: r.latency_ms ? `${Math.round(r.latency_ms)} ms` : undefined });
      else toast.error(`${p.name} health check failed`, { description: r.error ?? undefined });
      invalidate();
    },
  });
  const update = useMutation({
    mutationFn: ({ p, body }: { p: AIProvider; body: Record<string, unknown> }) => patch<AIProvider>(`/ai/providers/${p.id}`, body),
    onSuccess: (r, { body }) => {
      toast.success("is_default" in body ? `${r.name} is now the default provider` : r.enabled ? `${r.name} enabled` : `${r.name} disabled`);
      invalidate();
    },
  });
  const remove = useMutation({
    mutationFn: (p: AIProvider) => del(`/ai/providers/${p.id}`),
    onSuccess: (_r, p) => {
      toast.success(`${p.name} removed`, { description: "Its stored credential was deleted." });
      setDeleting(null);
      invalidate();
    },
  });

  return (
    <div className="space-y-5">
      <SectionTitle
        className="mb-0 flex-wrap"
        title="Model providers"
        description={
          <span className="block max-w-2xl leading-relaxed">
            Language and embedding models used for answers, agents and SQL generation, from any hosted or self-hosted provider. Model lists come live from each provider and refresh daily. API keys are encrypted and never shown again.
          </span>
        }
        action={
          <GatedButton perm="admin.ai_providers" onClick={() => setEditing("new")}>
            <Plus className="h-4 w-4" /> Add provider
          </GatedButton>
        }
      />
      {providers.data && !hasChatProvider ? (
        <Callout tone="neutral">
          {platform.data?.available ? (
            <>
              No chat provider is enabled here, so answers use the platform default model, <Mono>{platform.data.model}</Mono>, set by your administrator. Add a provider to choose your own.
            </>
          ) : (
            <>No chat provider is enabled, so answers fall back to extractive mode (quoted passages only). Add a provider to generate answers.</>
          )}
        </Callout>
      ) : null}
      {!manage ? <Callout tone="neutral">You can see which providers are configured. {permissionHint("admin.ai_providers")} to add, edit or test them.</Callout> : null}

      {providers.isLoading ? (
        <div className="grid gap-4 md:grid-cols-2">
          <Skeleton className="h-44 rounded-xl" />
          <Skeleton className="h-44 rounded-xl" />
        </div>
      ) : !providers.data?.length ? (
        <Card>
          <EmptyState icon={<Bot className="h-5 w-5" />} title="No AI providers configured" description="Connect OpenAI, Anthropic, Azure OpenAI, Bedrock, Gemini or a self-hosted model (Ollama, vLLM, any OpenAI-compatible server)." action={manage ? <Button onClick={() => setEditing("new")}>Add a provider</Button> : undefined} />
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {providers.data.map((p) => {
            const h = health[p.id];
            const s = p.settings as Record<string, number | string | undefined>;
            const fallback = s.fallback_provider_id ? byId.get(String(s.fallback_provider_id)) : undefined;
            return (
              <Card key={p.id} className={cn("flex flex-col overflow-hidden", !p.enabled && "opacity-75")}>
                <div className="flex items-start gap-3 border-b px-5 py-4">
                  <span className={cn("grid h-9 w-9 shrink-0 place-items-center rounded-lg border", p.is_default ? "border-accent/20 bg-accent/[0.08] text-accent" : "bg-surface-2/60 text-muted")}>
                    <Bot className="h-4 w-4" />
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <h3 className="truncate text-[15px] font-semibold">{p.name}</h3>
                      {p.is_default ? (
                        <Badge tone="accent">
                          <Star className="h-3 w-3" /> default
                        </Badge>
                      ) : null}
                      {!p.enabled ? <Badge>disabled</Badge> : null}
                    </div>
                    <div className="text-xs text-muted">{p.kind_label}</div>
                  </div>
                  <StatusBadge status={p.status} />
                </div>
                <dl className="grid flex-1 grid-cols-2 gap-x-5 gap-y-3.5 px-5 py-4 text-[12.5px] [&_dd]:mt-1 [&_dt]:font-mono [&_dt]:text-[10.5px] [&_dt]:uppercase [&_dt]:tracking-[0.1em] [&_dt]:text-subtle">
                  <div className="min-w-0">
                    <dt>Chat model</dt>
                    <dd className="truncate font-mono">{p.default_model || "—"}</dd>
                  </div>
                  <div className="min-w-0">
                    <dt>Embedding model</dt>
                    <dd className="truncate font-mono">{p.default_embedding_model || "—"}</dd>
                  </div>
                  {manage ? (
                    <div className="col-span-2 min-w-0">
                      <dt>Endpoint</dt>
                      <dd className="truncate font-mono">{p.base_url || "Provider default"}</dd>
                    </div>
                  ) : null}
                  <div>
                    <dt>Credentials</dt>
                    <dd>{p.has_credentials ? "Stored (encrypted)" : "None"}</dd>
                  </div>
                  <div>
                    <dt>Model list</dt>
                    <dd>{p.models_refreshed_at ? <>{formatNumber(p.models?.length ?? 0)} live · <When iso={p.models_refreshed_at} /></> : LISTABLE.has(p.kind) ? "Not fetched yet" : "Built-in suggestions"}</dd>
                  </div>
                  <div>
                    <dt>Last checked</dt>
                    <dd>
                      <When iso={p.last_checked_at} />
                    </dd>
                  </div>
                  {manage && (s.temperature !== undefined || s.max_tokens !== undefined || s.rpm_limit || s.monthly_token_limit || fallback) ? (
                    <div className="col-span-2 flex flex-wrap gap-1 pt-1">
                      {s.temperature !== undefined ? <Badge>temp {String(s.temperature)}</Badge> : null}
                      {s.max_tokens !== undefined ? <Badge>max {formatNumber(Number(s.max_tokens))} tokens</Badge> : null}
                      {s.rpm_limit ? <Badge>{String(s.rpm_limit)} req/min</Badge> : null}
                      {s.monthly_token_limit ? <Badge>{formatNumber(Number(s.monthly_token_limit))} tokens/month</Badge> : null}
                      {fallback ? <Badge tone="info">fallback → {fallback.name}</Badge> : null}
                    </div>
                  ) : null}
                </dl>
                {h || (manage && p.last_error) ? (
                  <div className="px-5 pb-3">
                    {h ? (
                      <div className={cn("flex items-start gap-2 rounded-lg border px-3 py-2 text-xs", h.ok ? "border-ok/25 bg-ok/5" : "border-danger/25 bg-danger/5")} role="status">
                        {h.ok ? <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ok" /> : <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-danger" />}
                        <span className="min-w-0 break-words">{h.ok ? `Healthy${h.latency_ms ? ` · ${Math.round(h.latency_ms)} ms` : ""}` : h.error || "Health check failed"}</span>
                      </div>
                    ) : (
                      <div className="rounded-lg border border-danger/25 bg-danger/5 px-3 py-2 text-xs text-danger">{p.last_error}</div>
                    )}
                  </div>
                ) : null}
                {manage ? (
                  <div className="flex flex-wrap items-center gap-2 border-t bg-surface-2/30 px-5 py-2.5">
                    <label className="mr-auto flex items-center gap-2 text-xs text-muted">
                      <Switch checked={p.enabled} onCheckedChange={(v) => update.mutate({ p, body: { enabled: v } })} label={`${p.enabled ? "Disable" : "Enable"} ${p.name}`} />
                      {p.enabled ? "Enabled" : "Disabled"}
                    </label>
                    <Button size="xs" variant="secondary" loading={test.isPending && test.variables?.id === p.id} onClick={() => test.mutate(p)}>
                      {!(test.isPending && test.variables?.id === p.id) ? <Activity className="h-3.5 w-3.5" /> : null} Test
                    </Button>
                    {!p.is_default && !kinds.data?.find((k) => k.kind === p.kind)?.embedding_only ? (
                      <Button size="xs" variant="ghost" onClick={() => update.mutate({ p, body: { is_default: true } })}>
                        <Star className="h-3.5 w-3.5" /> Make default
                      </Button>
                    ) : null}
                    <Button size="xs" variant="ghost" onClick={() => setEditing(p)} aria-label={`Edit ${p.name}`}>
                      <Pencil className="h-3.5 w-3.5" /> Edit
                    </Button>
                    <Button size="xs" variant="ghost" className="text-danger hover:text-danger" onClick={() => setDeleting(p)} aria-label={`Delete ${p.name}`}>
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                ) : null}
              </Card>
            );
          })}
        </div>
      )}

      <ProviderDialog state={editing} kinds={kinds.data ?? []} providers={providers.data ?? []} onClose={() => setEditing(null)} onSaved={invalidate} />
      <ConfirmDialog
        open={Boolean(deleting)}
        onOpenChange={(v) => !v && setDeleting(null)}
        title="Remove AI provider?"
        description={deleting ? `“${deleting.name}” and its stored API key will be deleted. Agents, knowledge-base embeddings or fallbacks configured to use it must be pointed at another provider.` : undefined}
        confirmLabel="Remove provider"
        requireText={deleting?.name}
        loading={remove.isPending}
        onConfirm={() => {
          if (deleting) remove.mutate(deleting);
        }}
      />
    </div>
  );
}

type Form = {
  kind: string;
  name: string;
  base_url: string;
  api_key: string;
  default_model: string;
  default_embedding_model: string;
  temperature: string;
  max_tokens: string;
  timeout_s: string;
  max_retries: string;
  rpm_limit: string;
  monthly_token_limit: string;
  fallback_provider_id: string;
  api_version: string;
  region: string;
  access_key_id: string;
  enabled: boolean;
  is_default: boolean;
};

const str = (v: unknown) => (v === undefined || v === null ? "" : String(v));

function ProviderDialog({ state, kinds, providers, onClose, onSaved }: { state: AIProvider | "new" | null; kinds: ProviderKind[]; providers: AIProvider[]; onClose: () => void; onSaved: () => void }) {
  const editing = state && state !== "new" ? state : null;
  const [f, setF] = useState<Form | null>(null);
  const [models, setModels] = useState<string[]>([]);

  useEffect(() => {
    if (!state) return;
    setModels([]);
    if (state === "new") {
      const k = kinds[0]?.kind ?? "openai";
      setF({ kind: k, name: "", base_url: "", api_key: "", default_model: kinds[0]?.suggested_models[0] ?? "", default_embedding_model: "", temperature: "0.2", max_tokens: "1024", timeout_s: "", max_retries: "2", rpm_limit: "", monthly_token_limit: "", fallback_provider_id: "", api_version: "", region: "", access_key_id: "", enabled: true, is_default: providers.length === 0 });
    } else {
      const s = state.settings;
      setF({
        kind: state.kind,
        name: state.name,
        base_url: state.base_url ?? "",
        api_key: "",
        default_model: state.default_model ?? "",
        default_embedding_model: state.default_embedding_model ?? "",
        temperature: str(s.temperature ?? 0.2),
        max_tokens: str(s.max_tokens ?? 1024),
        timeout_s: str(s.timeout_s),
        max_retries: str(s.max_retries ?? 2),
        rpm_limit: str(s.rpm_limit),
        monthly_token_limit: str(s.monthly_token_limit),
        fallback_provider_id: str(s.fallback_provider_id),
        api_version: str(s.api_version),
        region: str(s.region),
        access_key_id: str(s.access_key_id),
        enabled: state.enabled,
        is_default: state.is_default,
      });
    }
  }, [state]); // eslint-disable-line react-hooks/exhaustive-deps

  const kind = kinds.find((k) => k.kind === f?.kind);
  const set = <K extends keyof Form>(key: K, value: Form[K]) => setF((cur) => (cur ? { ...cur, [key]: value } : cur));

  const loadModels = useMutation({
    mutationFn: (refresh: boolean) => get<string[]>(`/ai/providers/${editing!.id}/models`, refresh ? { refresh: "true" } : undefined),
    onSuccess: (list, refresh) => {
      setModels(list);
      if (refresh) toast.success(list.length ? `${list.length} models available` : "The provider reported no models");
    },
    meta: { silent: true }, // shown inline under the model fields
  });
  // Editing a provider loads its live model list (cached server-side for a few minutes).
  useEffect(() => {
    if (editing && LISTABLE.has(editing.kind)) loadModels.mutate(false);
  }, [editing?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const save = useMutation({
    mutationFn: () => {
      const form = f!;
      const settings: Record<string, unknown> = {};
      const num = (k: keyof Form, key: string) => {
        if (String(form[k]).trim() !== "") settings[key] = Number(form[k]);
      };
      num("temperature", "temperature");
      num("max_tokens", "max_tokens");
      num("timeout_s", "timeout_s");
      num("max_retries", "max_retries");
      // Limits and fallback can be cleared: falsy values disable them server-side.
      settings.rpm_limit = form.rpm_limit.trim() ? Number(form.rpm_limit) : null;
      settings.monthly_token_limit = form.monthly_token_limit.trim() ? Number(form.monthly_token_limit) : null;
      settings.fallback_provider_id = form.fallback_provider_id || null;
      if (form.kind === "azure_openai" && form.api_version.trim()) settings.api_version = form.api_version.trim();
      if (form.kind === "bedrock") {
        if (form.region.trim()) settings.region = form.region.trim();
        if (form.access_key_id.trim()) settings.access_key_id = form.access_key_id.trim();
      }
      const body: Record<string, unknown> = {
        name: form.name.trim(),
        base_url: form.base_url.trim() || null,
        default_model: kind?.embedding_only ? null : form.default_model.trim() || null,
        default_embedding_model: form.default_embedding_model.trim() || null,
        settings,
        enabled: form.enabled,
        is_default: kind?.embedding_only ? false : form.is_default,
      };
      if (form.api_key) body.api_key = form.api_key;
      if (editing) return patch<AIProvider>(`/ai/providers/${editing.id}`, body);
      return post<AIProvider>("/ai/providers", { ...body, kind: form.kind });
    },
    onSuccess: (p) => {
      toast.success(editing ? `${p.name} updated` : `${p.name} added`, { description: editing ? undefined : "Run a health check to verify the connection." });
      onSaved();
      onClose();
    },
  });

  if (!f) return null;
  const numBad = (v: string, min: number, max: number, int = true) => v.trim() !== "" && (Number.isNaN(Number(v)) || Number(v) < min || Number(v) > max || (int && !Number.isInteger(Number(v))));
  const errors = {
    name: f.name.trim().length > 0 && f.name.trim().length < 2,
    base_url: URL_REQUIRED.has(f.kind) && !f.base_url.trim(),
    api_key: !editing && kind && !kind.keyless && !f.api_key,
    temperature: numBad(f.temperature, 0, 2, false),
    max_tokens: numBad(f.max_tokens, 1, 1_000_000),
    timeout_s: numBad(f.timeout_s, 1, 600, false),
    max_retries: numBad(f.max_retries, 0, 10),
    rpm_limit: numBad(f.rpm_limit, 1, 100_000),
    monthly_token_limit: numBad(f.monthly_token_limit, 1, 1e12),
  };
  const invalid = f.name.trim().length < 2 || Object.values(errors).some(Boolean);
  const suggestions = Array.from(new Set([...(models ?? []), ...(kind?.suggested_models ?? []), ...(editing?.suggested_models ?? [])]));
  const fallbacks = providers.filter((p) => p.id !== editing?.id && !kinds.find((k) => k.kind === p.kind)?.embedding_only);

  return (
    <Dialog open={Boolean(state)} onOpenChange={(v) => !v && onClose()}>
      <DialogContent
        wide
        title={editing ? `Edit ${editing.name}` : "Add AI provider"}
        description={kind?.embedding_only ? "Embedding and reranking provider — not used for chat answers." : "Chat model provider. Keys are stored encrypted and are write-only."}
        footer={
          <>
            <CancelButton />
            <Button type="submit" form="provider-form" loading={save.isPending} disabled={invalid}>
              {editing ? "Save provider" : "Add provider"}
            </Button>
          </>
        }
      >
        <form
          id="provider-form"
          className="space-y-5"
          onSubmit={(e) => {
            e.preventDefault();
            if (!invalid) save.mutate();
          }}
        >
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Provider type" htmlFor="pv-kind" hint={editing ? "The provider type cannot be changed." : kind?.keyless ? "Can run without an API key on a private network." : undefined}>
              <Select
                id="pv-kind"
                value={f.kind}
                disabled={Boolean(editing)}
                onChange={(e) => {
                  const k = kinds.find((x) => x.kind === e.target.value);
                  setF((cur) => (cur ? { ...cur, kind: e.target.value, default_model: k?.suggested_models[0] ?? "", is_default: k?.embedding_only ? false : cur.is_default } : cur));
                }}
              >
                {kinds.map((k) => (
                  <option key={k.kind} value={k.kind}>
                    {k.label}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Display name" htmlFor="pv-name" error={errors.name ? "At least 2 characters." : null}>
              <Input id="pv-name" value={f.name} onChange={(e) => set("name", e.target.value)} placeholder={kind ? `${kind.label} (production)` : "Name"} maxLength={120} autoFocus />
            </Field>
            <Field label={URL_REQUIRED.has(f.kind) ? "Endpoint URL" : "Endpoint URL (optional)"} htmlFor="pv-url" className="sm:col-span-2" error={errors.base_url ? `${kind?.label ?? "This provider"} needs an endpoint URL.` : null} hint={f.kind === "ollama" ? "e.g. http://ollama.internal:11434" : URL_REQUIRED.has(f.kind) ? undefined : "Leave empty to use the provider's public API."}>
              <Input id="pv-url" value={f.base_url} onChange={(e) => set("base_url", e.target.value)} placeholder="https://…" className="font-mono text-[13px]" />
            </Field>
            <Field
              label={kind?.keyless ? "API key (optional)" : "API key"}
              htmlFor="pv-key"
              className="sm:col-span-2"
              error={errors.api_key ? `${kind?.label} needs an API key.` : null}
              hint={editing ? (editing.has_credentials ? "A key is stored. Leave empty to keep it; type a new one to rotate it." : "No key stored.") : "Write-only: it is encrypted on save and never displayed again."}
            >
              <Input id="pv-key" type="password" autoComplete="off" value={f.api_key} onChange={(e) => set("api_key", e.target.value)} placeholder={editing?.has_credentials ? "•••••••••••• (stored)" : f.kind === "bedrock" ? "AWS secret access key" : "sk-…"} className="font-mono text-[13px]" />
            </Field>
            {f.kind === "azure_openai" ? (
              <Field label="API version" htmlFor="pv-apiv" hint="Defaults to 2024-06-01.">
                <Input id="pv-apiv" value={f.api_version} onChange={(e) => set("api_version", e.target.value)} placeholder="2024-06-01" className="font-mono text-[13px]" />
              </Field>
            ) : null}
            {f.kind === "bedrock" ? (
              <>
                <Field label="AWS region" htmlFor="pv-region" hint="Defaults to us-east-1.">
                  <Input id="pv-region" value={f.region} onChange={(e) => set("region", e.target.value)} placeholder="us-east-1" className="font-mono text-[13px]" />
                </Field>
                <Field label="Access key id" htmlFor="pv-akid" hint="Leave empty to use the server's IAM role.">
                  <Input id="pv-akid" value={f.access_key_id} onChange={(e) => set("access_key_id", e.target.value)} className="font-mono text-[13px]" />
                </Field>
              </>
            ) : null}
          </div>

          <fieldset className="space-y-3 rounded-lg border p-4">
            <legend className="px-1 text-[13px] font-semibold">Models</legend>
            <div className="grid gap-4 sm:grid-cols-2">
              {!kind?.embedding_only ? (
                <Field label="Default chat model" htmlFor="pv-model" hint={f.kind === "azure_openai" ? "Your deployment name." : "Pick a suggestion or type any model id."}>
                  <Input id="pv-model" list="pv-model-list" value={f.default_model} onChange={(e) => set("default_model", e.target.value)} className="font-mono text-[13px]" />
                </Field>
              ) : null}
              <Field label="Default embedding model (optional)" htmlFor="pv-emb">
                <Input id="pv-emb" list="pv-model-list" value={f.default_embedding_model} onChange={(e) => set("default_embedding_model", e.target.value)} className="font-mono text-[13px]" placeholder={f.kind === "ollama" ? "nomic-embed-text" : undefined} />
              </Field>
            </div>
            <datalist id="pv-model-list">
              {suggestions.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
            {LISTABLE.has(f.kind) ? (
              editing ? (
                <div className="space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <Button type="button" size="xs" variant="secondary" loading={loadModels.isPending} onClick={() => loadModels.mutate(true)}>
                      {!loadModels.isPending ? <RefreshCw /> : null} Refresh from provider
                    </Button>
                    <span className="text-xs text-subtle">
                      {loadModels.isPending ? "Asking the provider which models it offers…" : models.length ? `${formatNumber(models.length)} models available right now · type in a field to filter` : "Lists the models this provider offers right now."}
                    </span>
                  </div>
                  {loadModels.error ? <p className="text-xs text-danger">{(loadModels.error as Error).message}</p> : null}
                  {models.length ? (
                    <div className="flex flex-wrap gap-1.5">
                      {models.slice(0, 14).map((m) => (
                        <button key={m} type="button" onClick={() => set(kind?.embedding_only ? "default_embedding_model" : "default_model", m)} className={cn("rounded-md border px-2 py-0.5 font-mono text-[11.5px] transition-colors hover:bg-surface-2", (f.default_model === m || f.default_embedding_model === m) && "border-accent bg-accent/10 text-accent")}>
                          {m}
                        </button>
                      ))}
                      {models.length > 14 ? <span className="self-center font-mono text-[11px] text-subtle">+{models.length - 14} more in the list</span> : null}
                    </div>
                  ) : null}
                </div>
              ) : (
                <p className="text-xs text-subtle">After you save, this provider&apos;s live model list loads here and refreshes daily. Any model id can also be typed.</p>
              )
            ) : (
              <p className="text-xs text-subtle">{f.kind === "azure_openai" ? "Azure OpenAI addresses models by deployment name: type yours." : "This provider does not publish a model list: type the model id."}</p>
            )}
          </fieldset>

          <fieldset className="space-y-3 rounded-lg border p-4">
            <legend className="px-1 text-[13px] font-semibold">Generation & limits</legend>
            <div className="grid gap-4 sm:grid-cols-3">
              <Field label="Temperature" htmlFor="pv-temp" error={errors.temperature ? "0 – 2" : null}>
                <Input id="pv-temp" inputMode="decimal" value={f.temperature} onChange={(e) => set("temperature", e.target.value)} />
              </Field>
              <Field label="Max output tokens" htmlFor="pv-max" error={errors.max_tokens ? "Positive whole number" : null}>
                <Input id="pv-max" inputMode="numeric" value={f.max_tokens} onChange={(e) => set("max_tokens", e.target.value.replace(/[^\d]/g, ""))} />
              </Field>
              <Field label="Timeout (seconds)" htmlFor="pv-timeout" error={errors.timeout_s ? "1 – 600" : null}>
                <Input id="pv-timeout" inputMode="decimal" value={f.timeout_s} onChange={(e) => set("timeout_s", e.target.value)} placeholder="Server default" />
              </Field>
              <Field label="Max retries" htmlFor="pv-retries" error={errors.max_retries ? "0 – 10" : null}>
                <Input id="pv-retries" inputMode="numeric" value={f.max_retries} onChange={(e) => set("max_retries", e.target.value.replace(/[^\d]/g, ""))} />
              </Field>
              <Field label="Requests / minute" htmlFor="pv-rpm" error={errors.rpm_limit ? "Positive whole number" : null}>
                <Input id="pv-rpm" inputMode="numeric" value={f.rpm_limit} onChange={(e) => set("rpm_limit", e.target.value.replace(/[^\d]/g, ""))} placeholder="Unlimited" />
              </Field>
              <Field label="Monthly token cap" htmlFor="pv-monthly" error={errors.monthly_token_limit ? "Positive whole number" : null}>
                <Input id="pv-monthly" inputMode="numeric" value={f.monthly_token_limit} onChange={(e) => set("monthly_token_limit", e.target.value.replace(/[^\d]/g, ""))} placeholder="Unlimited" />
              </Field>
            </div>
            {!kind?.embedding_only ? (
              <Field label="Fallback provider" htmlFor="pv-fallback" hint="Used automatically if this provider errors before streaming starts.">
                <Select id="pv-fallback" value={f.fallback_provider_id} onChange={(e) => set("fallback_provider_id", e.target.value)}>
                  <option value="">No fallback</option>
                  {fallbacks.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name} · {p.kind_label}
                    </option>
                  ))}
                </Select>
              </Field>
            ) : null}
            {editing ? <p className="text-xs text-subtle">Saved values are kept when a field is left empty; enter a new number to change one. Request and token caps can be cleared.</p> : null}
          </fieldset>

          <div className="flex flex-wrap gap-6">
            <label className="flex items-center gap-2 text-[13px]">
              <Switch checked={f.enabled} onCheckedChange={(v) => set("enabled", v)} label="Enabled" /> Enabled
            </label>
            {!kind?.embedding_only ? (
              <label className="flex items-center gap-2 text-[13px]">
                <Switch checked={f.is_default} onCheckedChange={(v) => set("is_default", v)} label="Organization default" /> Organization default for chat
              </label>
            ) : null}
          </div>
          {editing ? (
            <p className="text-xs text-subtle">
              Provider id <Mono>{editing.id}</Mono>
            </p>
          ) : null}
        </form>
      </DialogContent>
    </Dialog>
  );
}
