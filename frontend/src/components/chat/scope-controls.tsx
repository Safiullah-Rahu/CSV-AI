"use client";

import * as Popover from "@radix-ui/react-popover";
import { Check, ChevronDown, Library, SlidersHorizontal } from "lucide-react";

import type { ChatFilters, KB, RetrievalSettings } from "@/components/chat/types";
import { Field, Input, Select } from "@/components/ui/form";
import { Switch } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

/* Compact chip used for scope controls inside the composer. */
const chip =
  "inline-flex h-7 min-w-0 items-center gap-1.5 rounded-lg px-2 text-[12.5px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/30 data-[state=open]:bg-surface-2 data-[state=open]:text-fg [&_svg]:shrink-0";
const chipIdle = "text-muted hover:bg-surface-2 hover:text-fg";
const chipOn = "bg-accent/10 text-accent hover:bg-accent/15 data-[state=open]:bg-accent/15 data-[state=open]:text-accent";

const popover = "z-50 rounded-xl border bg-surface shadow-float animate-scale-in focus:outline-none";

function CheckMark({ on }: { on: boolean }) {
  return (
    <span className={cn("grid h-4 w-4 shrink-0 place-items-center rounded-[5px] border border-border-strong bg-surface shadow-xs transition-colors", on && "border-accent bg-accent text-accent-fg")}>
      {on ? <Check className="h-3 w-3" strokeWidth={3} /> : null}
    </span>
  );
}

export function KBPicker({ kbs, selected, onChange }: { kbs: KB[]; selected: string[]; onChange: (ids: string[]) => void }) {
  const label = selected.length === 0 ? "All knowledge" : selected.length === 1 ? kbs.find((k) => k.id === selected[0])?.name ?? "1 knowledge base" : `${selected.length} knowledge bases`;
  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button type="button" className={cn(chip, "max-w-[148px] sm:max-w-[240px]", selected.length ? chipOn : chipIdle)} aria-label={`Search scope: ${label}`}>
          <Library className="h-3.5 w-3.5" />
          <span className="truncate">{label}</span>
          <ChevronDown className="h-3 w-3 opacity-60" />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content align="start" side="top" sideOffset={8} collisionPadding={12} className={cn(popover, "w-[min(320px,calc(100vw-24px))] p-1.5")}>
          <div className="label px-2 pb-1.5 pt-1.5">Search scope</div>
          <button type="button" onClick={() => onChange([])} className="flex w-full items-center gap-2.5 rounded-lg px-2 py-2 text-left text-[13px] transition-colors hover:bg-surface-2">
            <CheckMark on={selected.length === 0} />
            <span className="min-w-0 flex-1">
              <span className="block font-medium">All knowledge bases</span>
              <span className="block text-[12px] text-subtle">Everything you are allowed to access</span>
            </span>
          </button>
          <div className="-mx-1.5 my-1.5 h-px bg-border" />
          <div className="max-h-72 overflow-y-auto">
            {kbs.length === 0 ? <p className="px-2 py-3 text-[12.5px] text-muted">No knowledge bases are shared with you yet.</p> : null}
            {kbs.map((kb) => {
              const on = selected.includes(kb.id);
              return (
                <button
                  type="button"
                  key={kb.id}
                  onClick={() => onChange(on ? selected.filter((x) => x !== kb.id) : [...selected, kb.id])}
                  className="flex w-full items-start gap-2.5 rounded-lg px-2 py-2 text-left transition-colors hover:bg-surface-2"
                >
                  <span className="mt-0.5">
                    <CheckMark on={on} />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[13px] font-medium">{kb.name}</span>
                    <span className="mt-0.5 block truncate font-mono text-[11px] text-subtle">
                      {kb.stats.documents} {kb.stats.documents === 1 ? "doc" : "docs"} · {kb.access}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

const TYPES = ["pdf", "docx", "md", "html", "txt", "xlsx", "csv", "pptx", "json"];

function Toggle({ on, onClick, children, mono }: { on: boolean; onClick: () => void; children: React.ReactNode; mono?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={on}
      className={cn(
        "h-6 rounded-md px-2 text-[11.5px] ring-1 ring-inset transition-colors",
        mono ? "font-mono" : "capitalize",
        on ? "bg-accent/10 text-accent ring-accent/30" : "text-muted ring-border hover:text-fg hover:ring-border-strong",
      )}
    >
      {children}
    </button>
  );
}

export function AdvancedControls({
  filters,
  setFilters,
  retrieval,
  setRetrieval,
  policy,
  setPolicy,
  providers,
  providerId,
  setProviderId,
}: {
  filters: ChatFilters;
  setFilters: (f: ChatFilters) => void;
  retrieval: RetrievalSettings;
  setRetrieval: (r: RetrievalSettings) => void;
  policy: string;
  setPolicy: (p: string) => void;
  providers: { id: string; name: string; default_model: string | null; enabled: boolean }[];
  providerId: string;
  setProviderId: (id: string) => void;
}) {
  const active = filters.file_types.length + filters.tags.length + filters.source_types.length + (filters.created_from ? 1 : 0) + (filters.created_to ? 1 : 0);
  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button type="button" className={cn(chip, active ? chipOn : chipIdle)} aria-label={active ? `Filters and retrieval settings (${active} active)` : "Filters and retrieval settings"}>
          <SlidersHorizontal className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Filters</span>
          {active ? <span className="num grid h-4 min-w-4 place-items-center rounded-full bg-accent px-1 text-[10px] font-semibold leading-none text-accent-fg">{active}</span> : null}
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content align="start" side="top" sideOffset={8} collisionPadding={12} className={cn(popover, "max-h-[min(72vh,640px)] w-[min(372px,calc(100vw-24px))] overflow-y-auto")}>
          <div className="space-y-4 p-4">
            <div>
              <div className="label mb-2">Document type</div>
              <div className="flex flex-wrap gap-1.5">
                {TYPES.map((t) => {
                  const on = filters.file_types.includes(t);
                  return (
                    <Toggle key={t} mono on={on} onClick={() => setFilters({ ...filters, file_types: on ? filters.file_types.filter((x) => x !== t) : [...filters.file_types, t] })}>
                      .{t}
                    </Toggle>
                  );
                })}
              </div>
            </div>
            <div>
              <div className="label mb-2">Source</div>
              <div className="flex flex-wrap gap-1.5">
                {["upload", "database", "integration"].map((t) => {
                  const on = filters.source_types.includes(t);
                  return (
                    <Toggle key={t} on={on} onClick={() => setFilters({ ...filters, source_types: on ? filters.source_types.filter((x) => x !== t) : [...filters.source_types, t] })}>
                      {t}
                    </Toggle>
                  );
                })}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Field label="From">
                <Input type="date" value={filters.created_from} onChange={(e) => setFilters({ ...filters, created_from: e.target.value })} />
              </Field>
              <Field label="To">
                <Input type="date" value={filters.created_to} onChange={(e) => setFilters({ ...filters, created_to: e.target.value })} />
              </Field>
            </div>
            <Field label="Tags" hint="Comma separated">
              <Input value={filters.tags.join(", ")} onChange={(e) => setFilters({ ...filters, tags: e.target.value.split(",").map((t) => t.trim()).filter(Boolean) })} placeholder="policy, finance" />
            </Field>
          </div>
          <div className="space-y-4 border-t bg-surface-2/40 p-4">
            <div className="label">Retrieval</div>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Mode">
                <Select value={retrieval.mode} onChange={(e) => setRetrieval({ ...retrieval, mode: e.target.value as RetrievalSettings["mode"] })}>
                  <option value="hybrid">Hybrid (dense + BM25)</option>
                  <option value="dense">Dense (vector)</option>
                  <option value="sparse">Sparse (BM25)</option>
                  <option value="keyword">Keyword (all terms)</option>
                </Select>
              </Field>
              <Field label="Top-K">
                <Input type="number" min={1} max={20} value={retrieval.top_k} onChange={(e) => setRetrieval({ ...retrieval, top_k: Number(e.target.value) || 6 })} />
              </Field>
              <Field label="Reranker">
                <Select value={retrieval.rerank} onChange={(e) => setRetrieval({ ...retrieval, rerank: e.target.value as RetrievalSettings["rerank"] })}>
                  <option value="lexical">Feature reranker</option>
                  <option value="none">None (fusion order)</option>
                </Select>
              </Field>
              <Field label="Grounding">
                <Select value={policy} onChange={(e) => setPolicy(e.target.value)}>
                  <option value="">Knowledge base default</option>
                  <option value="strict">Strict — sources only</option>
                  <option value="balanced">Balanced — labelled suggestions</option>
                  <option value="open">Open — general knowledge labelled</option>
                </Select>
              </Field>
            </div>
            <div className="space-y-3 text-[13px]">
              <label className="flex items-center justify-between gap-3">
                Diversify results (MMR) <Switch checked={retrieval.mmr} onCheckedChange={(v) => setRetrieval({ ...retrieval, mmr: v })} label="MMR" />
              </label>
              <label className="flex items-center justify-between gap-3">
                Query expansion (pseudo-relevance) <Switch checked={retrieval.query_expansion === "prf"} onCheckedChange={(v) => setRetrieval({ ...retrieval, query_expansion: v ? "prf" : "none" })} label="Query expansion" />
              </label>
              <label className="flex items-center justify-between gap-3">
                Contextual compression <Switch checked={retrieval.compression} onCheckedChange={(v) => setRetrieval({ ...retrieval, compression: v })} label="Compression" />
              </label>
            </div>
            {providers.length ? (
              <Field label="Model">
                <Select value={providerId} onChange={(e) => setProviderId(e.target.value)}>
                  <option value="">Organization default</option>
                  {providers
                    .filter((p) => p.enabled)
                    .map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name} {p.default_model ? `· ${p.default_model}` : ""}
                      </option>
                    ))}
                </Select>
              </Field>
            ) : null}
          </div>
          <div className="flex items-center justify-between border-t px-4 py-2.5">
            <span className="text-[12px] text-subtle">{active ? `${active} filter${active === 1 ? "" : "s"} active` : "No filters applied"}</span>
            <button
              type="button"
              className="text-[12.5px] font-medium text-accent hover:underline disabled:pointer-events-none disabled:opacity-40"
              disabled={!active}
              onClick={() => setFilters({ file_types: [], tags: [], source_types: [], created_from: "", created_to: "" })}
            >
              Clear filters
            </button>
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
