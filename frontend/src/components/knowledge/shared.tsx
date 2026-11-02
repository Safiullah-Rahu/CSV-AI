"use client";

import * as DialogPrimitive from "@radix-ui/react-dialog";
import type { QueryClient } from "@tanstack/react-query";
import {
  ChevronLeft,
  ChevronRight,
  File as FileGeneric,
  FileCode2,
  FileImage,
  FileSpreadsheet,
  FileText,
  Globe2,
  Library,
  Lock,
  Presentation,
  Search,
  Users,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState, type KeyboardEvent, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { Input, Select } from "@/components/ui/form";
import { Badge, Tip, type Tone } from "@/components/ui/primitives";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatNumber } from "@/lib/utils";

import type { Access, KnowledgeBase, Member, Page, Visibility } from "./types";

/* ------------------------------------------------------------ permissions */

export const LEVEL: Record<string, number> = { none: 0, read: 1, write: 2, manage: 3 };

/** Workspace-aware permission check, mirroring `Authorizer.has(perm, workspace_id)` on the API. */
export function usePerms() {
  const { org, user } = useAuth();
  return useMemo(() => {
    const orgPerms = new Set(org?.permissions.org ?? []);
    const ws = org?.permissions.workspaces ?? {};
    const has = (perm: string, workspaceId?: string | null) => orgPerms.has(perm) || Boolean(workspaceId && (ws[workspaceId] ?? []).includes(perm));
    const workspacesWith = (perm: string) => Object.entries(ws).filter(([, perms]) => perms.includes(perm)).map(([id]) => id);
    return { has, orgHas: (p: string) => orgPerms.has(p), workspacesWith, userId: user?.id ?? null };
  }, [org, user]);
}

export type KBAbilities = {
  level: number;
  upload: boolean;
  edit: boolean;
  configureIngestion: boolean;
  configureRetrieval: boolean;
  remove: boolean;
  permissions: boolean;
  /** Set a document's ACL mode to restricted (kb.permissions + write). */
  restrictDocuments: boolean;
  reprocess: boolean;
  deleteDocuments: boolean;
  exportFiles: boolean;
};

/** What the current user may do on one knowledge base (the API re-checks every call). */
export function kbAbilities(kb: Pick<KnowledgeBase, "access" | "workspace_id"> | null | undefined, perms: ReturnType<typeof usePerms>): KBAbilities {
  const level = kb ? LEVEL[kb.access] ?? 0 : 0;
  const ws = kb?.workspace_id ?? null;
  const h = (p: string) => Boolean(kb) && perms.has(p, ws);
  return {
    level,
    upload: h("kb.upload") && level >= 2,
    edit: h("kb.edit") && level >= 2,
    configureIngestion: h("kb.configure_ingestion") && level >= 3,
    configureRetrieval: h("kb.configure_retrieval") && level >= 3,
    remove: h("kb.delete") && level >= 3,
    permissions: (h("kb.permissions") && level >= 3) || perms.orgHas("admin.permissions"),
    restrictDocuments: h("kb.permissions") && level >= 2,
    reprocess: h("kb.reprocess") && level >= 2,
    deleteDocuments: h("kb.delete_documents") && level >= 2,
    exportFiles: h("kb.export"),
  };
}

/** Requirement text used in disabled-action tooltips. */
export function needs(perm: string, level?: Access) {
  return level && level !== "read" ? `Requires the ${perm} permission and ${level} access to this knowledge base.` : `Requires the ${perm} permission.`;
}

/* ---------------------------------------------------------------- queries */

/** Invalidate every `useApi` query whose path starts with one of the prefixes. */
export function invalidatePaths(qc: QueryClient, ...prefixes: string[]) {
  return qc.invalidateQueries({ predicate: (q) => typeof q.queryKey[0] === "string" && prefixes.some((p) => (q.queryKey[0] as string).startsWith(p)) });
}

export function useDebounced<T>(value: T, ms = 250): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

/** Org members keyed by user id (only fetched when the viewer may list members). */
export function useMemberMap() {
  const { can } = useAuth();
  const q = useApi<Page<Member>>(can("org.view") ? "/members" : null, { limit: 500 }, { staleTime: 60_000 });
  return useMemo(() => new Map((q.data?.items ?? []).map((m) => [m.user_id, m])), [q.data]);
}

export function memberLabel(map: Map<string, Member>, id: string | null | undefined, fallback = "—") {
  if (!id) return fallback;
  const m = map.get(id);
  return m ? m.full_name || m.email : `User ${id.slice(0, 8)}`;
}

/* ----------------------------------------------------------------- badges */

const VIS: Record<Visibility, { label: string; tone: Tone; icon: typeof Lock; hint: string }> = {
  restricted: { label: "Restricted", tone: "neutral", icon: Lock, hint: "Only the owner, explicit grants and admins with full knowledge access." },
  workspace: { label: "Workspace", tone: "info", icon: Users, hint: "Every member of the knowledge base's workspace can read it." },
  org: { label: "Organization", tone: "accent", icon: Globe2, hint: "Everyone in the organization with knowledge access can read it." },
};

export const VISIBILITY_OPTIONS = (Object.keys(VIS) as Visibility[]).map((k) => ({ value: k, ...VIS[k] }));

export function VisibilityBadge({ visibility }: { visibility: Visibility }) {
  const v = VIS[visibility] ?? VIS.restricted;
  const Icon = v.icon;
  return (
    <Tip content={v.hint}>
      <span>
        <Badge tone={v.tone}>
          <Icon className="h-3 w-3" /> {v.label}
        </Badge>
      </span>
    </Tip>
  );
}

const ACCESS_TONE: Record<string, Tone> = { manage: "accent", write: "ok", read: "neutral", none: "danger" };
export function AccessBadge({ access }: { access: Access | string }) {
  return (
    <Tip content={`Your access: ${access}`}>
      <span>
        <Badge tone={ACCESS_TONE[access] ?? "neutral"} className="capitalize">
          {access}
        </Badge>
      </span>
    </Tip>
  );
}

const TABULAR = new Set(["csv", "tsv", "xls", "xlsx", "xlsm", "ods", "json", "jsonl", "ndjson", "parquet"]);
const SLIDES = new Set(["ppt", "pptx", "odp"]);
const IMAGES = new Set(["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"]);
const DOCS = new Set(["pdf", "doc", "docx", "txt", "rtf", "odt", "html", "htm", "md", "markdown", "xml", "epub"]);

export function fileFamily(ext: string | null | undefined): "tabular" | "slides" | "image" | "document" | "code" | "other" {
  const e = (ext || "").toLowerCase();
  if (TABULAR.has(e)) return "tabular";
  if (SLIDES.has(e)) return "slides";
  if (IMAGES.has(e)) return "image";
  if (DOCS.has(e)) return "document";
  return e ? "code" : "other";
}

/*
 * Categorical file-type palette, defined once here (DESIGN.md: never decorate with
 * status colours). The tile stays neutral; only the glyph carries the family hue.
 */
const FAMILY_STYLE = {
  tabular: { icon: FileSpreadsheet, cls: "text-[hsl(152_52%_34%)] dark:text-[hsl(152_48%_62%)]" },
  slides: { icon: Presentation, cls: "text-[hsl(22_78%_48%)] dark:text-[hsl(24_85%_66%)]" },
  image: { icon: FileImage, cls: "text-[hsl(322_55%_50%)] dark:text-[hsl(322_68%_72%)]" },
  document: { icon: FileText, cls: "text-accent" },
  code: { icon: FileCode2, cls: "text-muted" },
  other: { icon: FileGeneric, cls: "text-subtle" },
} as const;

export function FileIcon({ ext, className }: { ext: string | null | undefined; className?: string }) {
  const s = FAMILY_STYLE[fileFamily(ext)];
  const Icon = s.icon;
  return (
    <span className={cn("grid h-8 w-8 shrink-0 place-items-center rounded-lg border bg-gradient-to-b from-surface to-surface-2/80 shadow-xs", s.cls, className)} aria-hidden>
      <Icon className="h-[15px] w-[15px]" strokeWidth={1.75} />
    </span>
  );
}

export function FormatBadge({ ext }: { ext: string | null | undefined }) {
  return <span className="inline-flex h-5 items-center rounded-[5px] border bg-surface-2/70 px-1.5 font-mono text-[10.5px] font-medium uppercase tracking-[0.04em] text-muted">{ext || "?"}</span>;
}

/** The knowledge-base mark: an accent-tinted tile used on cards, rows and the detail header. */
export function KBMark({ size = "md", className }: { size?: "sm" | "md" | "lg"; className?: string }) {
  const dims = { sm: "h-7 w-7 rounded-lg [&_svg]:h-3.5 [&_svg]:w-3.5", md: "h-10 w-10 rounded-[11px] [&_svg]:h-[18px] [&_svg]:w-[18px]", lg: "h-12 w-12 rounded-xl [&_svg]:h-5 [&_svg]:w-5" }[size];
  return (
    <span className={cn("relative grid shrink-0 place-items-center bg-accent/[0.09] text-accent ring-1 ring-inset ring-accent/20", dims, className)} aria-hidden>
      <span className="absolute inset-x-1.5 top-0 h-px bg-gradient-to-r from-transparent via-white/70 to-transparent dark:via-white/15" />
      <Library strokeWidth={1.75} />
    </span>
  );
}

/** Mono chip for locators, ids and short machine values. */
export function MonoChip({ children, className, title }: { children: ReactNode; className?: string; title?: string }) {
  return (
    <span title={title} className={cn("inline-flex h-5 max-w-full items-center truncate rounded-[5px] border bg-surface-2/70 px-1.5 font-mono text-[11px] tracking-[-0.01em] text-muted", className)}>
      {children}
    </span>
  );
}

/** Toolbar search field: icon + compact input, with a clear button. */
export function SearchField({ value, onChange, placeholder, label, className }: { value: string; onChange: (v: string) => void; placeholder: string; label: string; className?: string }) {
  return (
    <div className={cn("relative", className)}>
      <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-subtle" />
      <Input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} className="h-8 pl-8 pr-7 text-[13px]" aria-label={label} />
      {value ? (
        <button type="button" onClick={() => onChange("")} className="absolute right-1.5 top-1/2 grid h-5 w-5 -translate-y-1/2 place-items-center rounded text-subtle transition-colors hover:bg-surface-2 hover:text-fg" aria-label="Clear search">
          <X className="h-3 w-3" />
        </button>
      ) : null}
    </div>
  );
}

/**
 * Calm settings pattern: a label + description column beside a controls column.
 * Stacks on small screens.
 */
export function SettingsRow({ title, description, aside, children, className }: { title: ReactNode; description?: ReactNode; aside?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cn("grid gap-x-10 gap-y-4 border-t pt-8 first:border-t-0 first:pt-0 lg:grid-cols-[minmax(0,260px)_minmax(0,1fr)]", className)}>
      <div className="min-w-0 lg:sticky lg:top-6 lg:self-start">
        <h3 className="text-[15px] font-semibold tracking-[-0.015em]">{title}</h3>
        {description ? <p className="mt-1 text-[13px] leading-relaxed text-muted">{description}</p> : null}
        {aside ? <div className="mt-3">{aside}</div> : null}
      </div>
      <div className="min-w-0">{children}</div>
    </section>
  );
}

export function TagList({ tags, max = 4, tone = "neutral" }: { tags: string[]; max?: number; tone?: Tone }) {
  if (!tags?.length) return null;
  const shown = tags.slice(0, max);
  return (
    <span className="inline-flex flex-wrap items-center gap-1">
      {shown.map((t) => (
        <Badge key={t} tone={tone} className="font-normal">
          {t}
        </Badge>
      ))}
      {tags.length > max ? (
        <Tip content={tags.slice(max).join(", ")}>
          <span className="text-2xs text-subtle">+{tags.length - max}</span>
        </Tip>
      ) : null}
    </span>
  );
}

/* --------------------------------------------------------- guarded action */

/** A button that is disabled (with an explanation) when the user lacks a permission. */
export function Guarded({ allowed, reason, children }: { allowed: boolean; reason: string; children: ReactNode }) {
  if (allowed) return <>{children}</>;
  return (
    <Tip content={reason}>
      <span tabIndex={0} className="inline-flex cursor-not-allowed rounded-lg" aria-label={reason}>
        <span className="pointer-events-none inline-flex">{children}</span>
      </span>
    </Tip>
  );
}

/* ------------------------------------------------------------- tag input */

export function TagInput({ value, onChange, id, placeholder = "Add a tag and press Enter", disabled, suggestions }: { value: string[]; onChange: (v: string[]) => void; id?: string; placeholder?: string; disabled?: boolean; suggestions?: string[] }) {
  const [draft, setDraft] = useState("");
  const add = (raw: string) => {
    const parts = raw.split(",").map((t) => t.trim()).filter(Boolean);
    if (!parts.length) return;
    const next = [...value];
    for (const p of parts) if (!next.some((t) => t.toLowerCase() === p.toLowerCase())) next.push(p.slice(0, 64));
    onChange(next.slice(0, 30));
    setDraft("");
  };
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      add(draft);
    } else if (e.key === "Backspace" && !draft && value.length) {
      onChange(value.slice(0, -1));
    }
  };
  const extra = (suggestions ?? []).filter((s) => !value.some((t) => t.toLowerCase() === s.toLowerCase()));
  return (
    <div>
      <div className={cn("flex min-h-9 flex-wrap items-center gap-1.5 rounded-lg border bg-surface px-2 py-1.5 shadow-xs transition-[border-color,box-shadow] hover:border-border-strong focus-within:border-accent/70 focus-within:ring-[3px] focus-within:ring-accent/15", disabled && "opacity-60")}>
        {value.map((t) => (
          <span key={t} className="inline-flex h-6 items-center gap-1 rounded-md border bg-surface-2/80 pl-2 pr-0.5 text-[12px] font-medium">
            {t}
            {!disabled ? (
              <button type="button" className="rounded p-0.5 text-subtle hover:bg-border/60 hover:text-fg" onClick={() => onChange(value.filter((x) => x !== t))} aria-label={`Remove tag ${t}`}>
                <X className="h-3 w-3" />
              </button>
            ) : null}
          </span>
        ))}
        <input
          id={id}
          value={draft}
          disabled={disabled}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKey}
          onBlur={() => add(draft)}
          placeholder={value.length ? "" : placeholder}
          className="h-6 min-w-[120px] flex-1 bg-transparent px-1 text-[13.5px] outline-none placeholder:text-subtle"
        />
      </div>
      {extra.length && !disabled ? (
        <div className="mt-1.5 flex flex-wrap items-center gap-1">
          <span className="text-2xs text-subtle">Suggested:</span>
          {extra.slice(0, 8).map((s) => (
            <button key={s} type="button" onClick={() => add(s)} className="rounded-md border border-dashed px-1.5 py-0.5 text-2xs text-muted hover:border-accent/40 hover:text-accent">
              + {s}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

/* --------------------------------------------------------------- segmented */

export function Segmented<T extends string>({ value, onChange, options, label }: { value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode; icon?: ReactNode; title?: string }[]; label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex items-center gap-0.5 rounded-[10px] border bg-surface-2/80 p-[3px]">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          title={o.title}
          onClick={() => onChange(o.value)}
          className={cn("inline-flex h-[26px] items-center gap-1.5 rounded-[7px] px-2.5 text-[13px] font-medium text-muted transition-all hover:text-fg", value === o.value && "bg-surface text-fg shadow-xs ring-1 ring-border")}
        >
          {o.icon}
          {o.label}
        </button>
      ))}
    </div>
  );
}

/* -------------------------------------------------------------- pagination */

export function Pager({ total, limit, offset, onChange, onLimit, limits = [25, 50, 100] }: { total: number; limit: number; offset: number; onChange: (offset: number) => void; onLimit?: (limit: number) => void; limits?: number[] }) {
  const page = Math.floor(offset / limit);
  const pages = Math.max(1, Math.ceil(total / limit));
  return (
    <div className="flex flex-wrap items-center justify-between gap-2 border-t bg-surface-2/30 px-4 py-2 text-[12px] text-muted">
      <span className="num font-mono text-[11.5px]">{total ? `${formatNumber(offset + 1)}–${formatNumber(Math.min(total, offset + limit))} of ${formatNumber(total)}` : "0 results"}</span>
      <div className="flex items-center gap-2">
        {onLimit ? (
          <label className="flex items-center gap-1.5">
            <span className="hidden sm:inline">Rows</span>
            <Select value={limit} onChange={(e) => onLimit(Number(e.target.value))} className="h-7 w-[70px] py-0 text-[12px]" aria-label="Rows per page">
              {limits.map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </Select>
          </label>
        ) : null}
        <button className="grid h-7 w-7 place-items-center rounded-md text-muted transition-colors hover:bg-surface-2 hover:text-fg disabled:opacity-40" disabled={page === 0} onClick={() => onChange(Math.max(0, offset - limit))} aria-label="Previous page">
          <ChevronLeft className="h-4 w-4" />
        </button>
        <span className="num px-1 font-mono text-[11.5px]">
          {page + 1} / {pages}
        </span>
        <button className="grid h-7 w-7 place-items-center rounded-md text-muted transition-colors hover:bg-surface-2 hover:text-fg disabled:opacity-40" disabled={page >= pages - 1} onClick={() => onChange(offset + limit)} aria-label="Next page">
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ sheet */

/** Right-hand drawer built on Radix Dialog (focus trap, Esc to close, labelled). */
export function Sheet({ open, onOpenChange, title, description, actions, children, className }: { open: boolean; onOpenChange: (v: boolean) => void; title: ReactNode; description?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-[hsl(var(--shadow)/0.28)] backdrop-blur-[2px] animate-fade-in dark:bg-black/60" />
        <DialogPrimitive.Content className={cn("fixed inset-y-0 right-0 z-50 flex w-full max-w-[min(780px,100vw)] flex-col border-l bg-surface shadow-float animate-fade-in focus:outline-none sm:inset-y-2 sm:right-2 sm:max-w-[min(780px,calc(100vw-16px))] sm:rounded-2xl sm:border", className)}>
          <div className="flex items-start gap-3 px-6 pb-4 pt-5">
            <div className="min-w-0 flex-1">
              <DialogPrimitive.Title className="truncate text-[17px] font-semibold tracking-[-0.02em]">{title}</DialogPrimitive.Title>
              {description ? <DialogPrimitive.Description asChild><div className="mt-2.5 text-[13px] text-muted">{description}</div></DialogPrimitive.Description> : <DialogPrimitive.Description className="sr-only">Details</DialogPrimitive.Description>}
            </div>
            <DialogPrimitive.Close asChild>
              <Button variant="ghost" size="icon" aria-label="Close panel">
                <X className="h-4 w-4" />
              </Button>
            </DialogPrimitive.Close>
          </div>
          {actions ? <div className="flex flex-wrap items-center gap-2 border-y bg-surface-2/40 px-6 py-2.5">{actions}</div> : <div className="border-b" />}
          <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

/* ------------------------------------------------------------- key/value */

export function KV({ items, className }: { items: { k: ReactNode; v: ReactNode; wide?: boolean }[]; className?: string }) {
  return (
    <dl className={cn("grid grid-cols-1 gap-x-8 gap-y-4 sm:grid-cols-2", className)}>
      {items.map((it, i) => (
        <div key={i} className={cn("min-w-0", it.wide && "sm:col-span-2")}>
          <dt className="label">{it.k}</dt>
          <dd className="mt-1 break-words text-[13px] text-fg">{it.v ?? "—"}</dd>
        </div>
      ))}
    </dl>
  );
}

export const ACTIVE_STATUSES = new Set(["queued", "processing"]);
export const isActive = (status: string | null | undefined) => ACTIVE_STATUSES.has(status || "");
