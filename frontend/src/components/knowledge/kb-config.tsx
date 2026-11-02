"use client";

import { Field, Input, Select } from "@/components/ui/form";
import { Switch } from "@/components/ui/primitives";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";

import type { AIProvider, ChunkingConfig, EmbeddingConfig, GroundingPolicy, KBOptions, RetrievalConfig } from "./types";

/** Providers that run without stored credentials (no `provider_id` needed). */
export const KEYLESS_EMBEDDINGS = new Set(["hashing", "local", "ollama"]);

const STRATEGY_HINT: Record<string, string> = {
  structure: "Follows headings, pages, sheets and slides (recommended).",
  markdown: "Splits on Markdown headings.",
  fixed: "Fixed-size character windows.",
  recursive: "Recursive separators: paragraphs → lines → sentences.",
  token: "Fixed-size token windows (size is in tokens).",
  sentence: "Packs whole sentences up to the size.",
  semantic: "Splits where meaning shifts (embeds every sentence; slower).",
  table: "Keeps table rows together with their header.",
  parent_child: "Small child chunks for matching, larger parents for context.",
};

const MODE_LABEL: Record<string, string> = { hybrid: "Hybrid (dense + keyword)", dense: "Dense vectors", sparse: "Sparse (BM25-style)", keyword: "Keyword" };
const RERANK_LABEL: Record<string, string> = { lexical: "Lexical overlap", none: "No reranking", cohere: "Cohere rerank", cross_encoder: "Cross-encoder" };
export const GROUNDING: { value: GroundingPolicy; label: string; hint: string }[] = [
  { value: "strict", label: "Strict", hint: "Only cited source content; no general knowledge." },
  { value: "balanced", label: "Balanced", hint: "Sourced answer, plus optional uncited suggestions in a separate section." },
  { value: "open", label: "Open", hint: "May fill gaps with general knowledge, labelled as such." },
];

/* -------------------------------------------------------------- embedding */

export function EmbeddingFields({ value, onChange, options, disabled, idPrefix }: { value: EmbeddingConfig; onChange: (v: EmbeddingConfig) => void; options: KBOptions; disabled?: boolean; idPrefix: string }) {
  const { can } = useAuth();
  const provider = value.provider || "hashing";
  const needsProvider = !KEYLESS_EMBEDDINGS.has(provider);
  const providers = useApi<AIProvider[]>(needsProvider && can("org.view") ? "/ai/providers" : null, undefined, { staleTime: 60_000 });
  const usable = (providers.data ?? []).filter((p) => p.enabled);
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <Field label="Embedding provider" htmlFor={`${idPrefix}-emb-provider`} className="sm:col-span-2">
        <Select id={`${idPrefix}-emb-provider`} value={provider} disabled={disabled} onChange={(e) => onChange({ ...value, provider: e.target.value, model: e.target.value === "hashing" ? undefined : value.model, provider_id: KEYLESS_EMBEDDINGS.has(e.target.value) ? undefined : value.provider_id })}>
          {options.embedding_providers.map((p) => (
            <option key={p.key} value={p.key}>
              {p.label}
            </option>
          ))}
        </Select>
      </Field>
      {needsProvider ? (
        <Field label="Credentials" htmlFor={`${idPrefix}-emb-cred`} className="sm:col-span-2" hint={usable.length ? "The AI provider configuration whose API key is used for embeddings." : "No AI providers are configured yet. An administrator can add one in Settings → AI."}>
          <Select id={`${idPrefix}-emb-cred`} value={value.provider_id || ""} disabled={disabled || !usable.length} onChange={(e) => onChange({ ...value, provider_id: e.target.value || undefined })}>
            <option value="">Select a configured provider…</option>
            {usable.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} · {p.kind_label}
                {p.has_credentials ? "" : " (no key)"}
              </option>
            ))}
          </Select>
        </Field>
      ) : null}
      <Field label="Model" htmlFor={`${idPrefix}-emb-model`} hint={provider === "hashing" ? "Derived from the dimension." : "e.g. text-embedding-3-small, bge-small-en-v1.5"}>
        <Input
          id={`${idPrefix}-emb-model`}
          value={provider === "hashing" ? `hashing-${value.dim || 768}` : value.model || ""}
          disabled={disabled || provider === "hashing"}
          onChange={(e) => onChange({ ...value, model: e.target.value })}
          placeholder="Model name"
        />
      </Field>
      <Field label="Dimension" htmlFor={`${idPrefix}-emb-dim`} hint="8 – 8192. Leave empty to use the model's native size.">
        <Input
          id={`${idPrefix}-emb-dim`}
          type="number"
          min={8}
          max={8192}
          value={value.dim ?? ""}
          disabled={disabled}
          onChange={(e) => onChange({ ...value, dim: e.target.value === "" ? undefined : Number(e.target.value) })}
        />
      </Field>
    </div>
  );
}

/** Body sent to the API for an embedding config (hashing derives its own model name). */
export function embeddingPayload(v: EmbeddingConfig): EmbeddingConfig {
  const provider = v.provider || "hashing";
  if (provider === "hashing") return { provider, dim: v.dim || 768 };
  const out: EmbeddingConfig = { provider };
  if (v.model) out.model = v.model;
  if (v.dim) out.dim = v.dim;
  if (v.provider_id) out.provider_id = v.provider_id;
  return out;
}

export function embeddingError(v: EmbeddingConfig): string | null {
  const provider = v.provider || "hashing";
  if (!KEYLESS_EMBEDDINGS.has(provider) && !v.provider_id) return "Choose the AI provider credentials to embed with.";
  if (provider !== "hashing" && !v.model) return "Enter the embedding model name.";
  if (v.dim !== undefined && (v.dim < 8 || v.dim > 8192)) return "Embedding dimension must be between 8 and 8192.";
  return null;
}

/* --------------------------------------------------------------- chunking */

export function ChunkingFields({ value, onChange, options, disabled, idPrefix }: { value: ChunkingConfig; onChange: (v: ChunkingConfig) => void; options: KBOptions; disabled?: boolean; idPrefix: string }) {
  const set = <K extends keyof ChunkingConfig>(k: K, v: ChunkingConfig[K]) => onChange({ ...value, [k]: v });
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <Field label="Chunking strategy" htmlFor={`${idPrefix}-strategy`} hint={STRATEGY_HINT[value.strategy]} className="sm:col-span-2">
        <Select id={`${idPrefix}-strategy`} value={value.strategy} disabled={disabled} onChange={(e) => set("strategy", e.target.value)}>
          {options.chunking_strategies.map((s) => (
            <option key={s} value={s}>
              {s.replace(/_/g, " ")}
            </option>
          ))}
        </Select>
      </Field>
      <Field label={value.strategy === "token" ? "Chunk size (tokens)" : "Chunk size (characters)"} htmlFor={`${idPrefix}-size`} hint="Minimum 100.">
        <Input id={`${idPrefix}-size`} type="number" min={100} step={50} value={value.chunk_size} disabled={disabled} onChange={(e) => set("chunk_size", Number(e.target.value))} />
      </Field>
      <Field label="Overlap" htmlFor={`${idPrefix}-overlap`} hint="Capped at half the chunk size.">
        <Input id={`${idPrefix}-overlap`} type="number" min={0} step={10} value={value.chunk_overlap} disabled={disabled} onChange={(e) => set("chunk_overlap", Number(e.target.value))} />
      </Field>
      {value.strategy === "parent_child" ? (
        <>
          <Field label="Parent size" htmlFor={`${idPrefix}-parent`}>
            <Input id={`${idPrefix}-parent`} type="number" min={200} step={100} value={value.parent_size} disabled={disabled} onChange={(e) => set("parent_size", Number(e.target.value))} />
          </Field>
          <Field label="Child size" htmlFor={`${idPrefix}-child`}>
            <Input id={`${idPrefix}-child`} type="number" min={100} step={50} value={value.child_size} disabled={disabled} onChange={(e) => set("child_size", Number(e.target.value))} />
          </Field>
        </>
      ) : null}
      {value.strategy === "semantic" ? (
        <Field label="Breakpoint percentile" htmlFor={`${idPrefix}-pct`} hint="Higher = fewer, larger chunks.">
          <Input id={`${idPrefix}-pct`} type="number" min={50} max={99} value={value.semantic_percentile} disabled={disabled} onChange={(e) => set("semantic_percentile", Number(e.target.value))} />
        </Field>
      ) : null}
      <div className="flex items-center justify-between gap-3 rounded-lg border bg-surface-2/40 px-3.5 py-3 sm:col-span-2">
        <div>
          <div className="text-[13px] font-medium">Contextual headers</div>
          <div className="mt-0.5 text-[12px] text-muted">Prefix each chunk with its title and section path.</div>
        </div>
        <Switch label="Contextual headers" checked={Boolean(value.contextual_headers)} disabled={disabled} onCheckedChange={(v) => set("contextual_headers", v)} />
      </div>
    </div>
  );
}

export function chunkingPayload(v: ChunkingConfig): Partial<ChunkingConfig> {
  const out: Partial<ChunkingConfig> = { strategy: v.strategy, chunk_size: v.chunk_size, chunk_overlap: v.chunk_overlap, contextual_headers: v.contextual_headers };
  if (v.strategy === "parent_child") Object.assign(out, { parent_size: v.parent_size, child_size: v.child_size });
  if (v.strategy === "semantic") out.semantic_percentile = v.semantic_percentile;
  return out;
}

/* -------------------------------------------------------------- retrieval */

export function RetrievalFields({ value, onChange, disabled, idPrefix, compact }: { value: RetrievalConfig; onChange: (v: RetrievalConfig) => void; disabled?: boolean; idPrefix: string; compact?: boolean }) {
  const set = <K extends keyof RetrievalConfig>(k: K, v: RetrievalConfig[K]) => onChange({ ...value, [k]: v });
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <Field label="Retrieval mode" htmlFor={`${idPrefix}-mode`}>
        <Select id={`${idPrefix}-mode`} value={value.mode} disabled={disabled} onChange={(e) => set("mode", e.target.value)}>
          {Object.entries(MODE_LABEL).map(([k, l]) => (
            <option key={k} value={k}>
              {l}
            </option>
          ))}
        </Select>
      </Field>
      <Field label="Reranking" htmlFor={`${idPrefix}-rerank`}>
        <Select id={`${idPrefix}-rerank`} value={value.rerank} disabled={disabled} onChange={(e) => set("rerank", e.target.value)}>
          {Object.entries(RERANK_LABEL).map(([k, l]) => (
            <option key={k} value={k}>
              {l}
            </option>
          ))}
        </Select>
      </Field>
      <Field label="Top K passages" htmlFor={`${idPrefix}-topk`} hint="1 – 50 passages sent to the model.">
        <Input id={`${idPrefix}-topk`} type="number" min={1} max={50} value={value.top_k} disabled={disabled} onChange={(e) => set("top_k", Number(e.target.value))} />
      </Field>
      {!compact ? (
        <>
          <Field label="Candidates" htmlFor={`${idPrefix}-cand`} hint="Retrieved before reranking (≥ top K, ≤ 400).">
            <Input id={`${idPrefix}-cand`} type="number" min={1} max={400} value={value.candidate_k} disabled={disabled} onChange={(e) => set("candidate_k", Number(e.target.value))} />
          </Field>
          <Field label="Minimum evidence score" htmlFor={`${idPrefix}-minev`} hint="Below this, answers report insufficient evidence.">
            <Input id={`${idPrefix}-minev`} type="number" min={0} max={1} step={0.05} value={value.min_evidence} disabled={disabled} onChange={(e) => set("min_evidence", Number(e.target.value))} />
          </Field>
          <Field label="Query expansion" htmlFor={`${idPrefix}-qe`}>
            <Select id={`${idPrefix}-qe`} value={value.query_expansion} disabled={disabled} onChange={(e) => set("query_expansion", e.target.value)}>
              <option value="prf">Pseudo-relevance feedback</option>
              <option value="none">None</option>
            </Select>
          </Field>
          <ToggleRow label="Diversify results (MMR)" hint="Reduces near-duplicate passages." checked={Boolean(value.mmr)} disabled={disabled} onChange={(v) => set("mmr", v)} />
          <ToggleRow label="Quarantine suspicious chunks" hint="Exclude chunks flagged for prompt-injection patterns." checked={Boolean(value.quarantine_suspicious)} disabled={disabled} onChange={(v) => set("quarantine_suspicious", v)} />
        </>
      ) : null}
    </div>
  );
}

function ToggleRow({ label, hint, checked, onChange, disabled }: { label: string; hint: string; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border bg-surface-2/40 px-3.5 py-3">
      <div className="min-w-0">
        <div className="text-[13px] font-medium">{label}</div>
        <div className="mt-0.5 text-[12px] text-muted">{hint}</div>
      </div>
      <Switch label={label} checked={checked} disabled={disabled} onCheckedChange={onChange} />
    </div>
  );
}

export function retrievalError(v: RetrievalConfig): string | null {
  if (!Number.isFinite(v.top_k) || v.top_k < 1 || v.top_k > 50) return "Top K must be between 1 and 50.";
  if (v.candidate_k !== undefined && (v.candidate_k < 1 || v.candidate_k > 400)) return "Candidates must be between 1 and 400.";
  if (v.min_evidence !== undefined && (v.min_evidence < 0 || v.min_evidence > 1)) return "Minimum evidence must be between 0 and 1.";
  return null;
}

export function chunkingError(v: ChunkingConfig): string | null {
  if (!Number.isFinite(v.chunk_size) || v.chunk_size < 100) return "Chunk size must be at least 100.";
  if (!Number.isFinite(v.chunk_overlap) || v.chunk_overlap < 0) return "Overlap cannot be negative.";
  return null;
}

/* -------------------------------------------------------------- grounding */

export function GroundingField({ value, onChange, disabled, idPrefix }: { value: GroundingPolicy | ""; onChange: (v: GroundingPolicy | "") => void; disabled?: boolean; idPrefix: string }) {
  return (
    <fieldset className="space-y-2" disabled={disabled}>
      <legend className="mb-2 text-[13px] font-medium">Grounding policy</legend>
      <div className="grid gap-2 sm:grid-cols-2">
        {[{ value: "" as const, label: "Organization default", hint: "Use the policy set in organization AI settings." }, ...GROUNDING].map((g) => (
          <label key={g.value || "default"} htmlFor={`${idPrefix}-gp-${g.value || "default"}`} className={`flex cursor-pointer gap-2.5 rounded-lg border p-3 transition-[background-color,border-color,box-shadow] ${value === g.value ? "border-accent/50 bg-accent/[0.05] shadow-[0_0_0_3px_hsl(var(--accent)/0.08)]" : "hover:border-border-strong hover:bg-surface-2/50"} ${disabled ? "cursor-not-allowed opacity-60" : ""}`}>
            <input id={`${idPrefix}-gp-${g.value || "default"}`} type="radio" name={`${idPrefix}-grounding`} className="mt-0.5 accent-[hsl(var(--accent))]" checked={value === g.value} onChange={() => onChange(g.value)} />
            <span>
              <span className="block text-[13px] font-medium">{g.label}</span>
              <span className="mt-0.5 block text-[12px] leading-relaxed text-muted">{g.hint}</span>
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

/** Fill defaults for a stored (partial) config. */
export function withChunkDefaults(options: KBOptions, cfg: Partial<ChunkingConfig> | undefined): ChunkingConfig {
  return { ...options.chunking_defaults, ...(cfg || {}) };
}
export function withRetrievalDefaults(options: KBOptions, cfg: Partial<RetrievalConfig> | undefined): RetrievalConfig {
  return { ...options.retrieval_defaults, ...(cfg || {}) };
}
