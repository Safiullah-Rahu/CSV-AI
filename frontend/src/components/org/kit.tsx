"use client";

import { AlertTriangle, ArrowLeft, Check, ChevronLeft, ChevronRight, Copy, Info, KeyRound, Lock, MoreHorizontal, Search, ShieldAlert } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { forwardRef, useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { groupPermissions } from "@/components/org/data";
import type { Member, Permission } from "@/components/org/types";
import { Button, type ButtonProps } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent } from "@/components/ui/dialog";
import { Input } from "@/components/ui/form";
import { Avatar, Checkbox, EmptyState, Mono, Tip, type Tone } from "@/components/ui/primitives";
import { ApiError, type ApiErrorBody, apiFetch } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { cn, formatBytes, formatDate, formatNumber, timeAgo } from "@/lib/utils";

/* ------------------------------------------------------------- clipboard */
async function copyText(value: string) {
  try {
    await navigator.clipboard.writeText(value);
    return true;
  } catch {
    // Fallback for non-secure contexts.
    const ta = document.createElement("textarea");
    ta.value = value;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  }
}

export function CopyButton({ value, label = "Copy", variant = "outline", size = "sm", iconOnly, className }: { value: string; label?: string; variant?: ButtonProps["variant"]; size?: ButtonProps["size"]; iconOnly?: boolean; className?: string }) {
  const [done, setDone] = useState(false);
  useEffect(() => {
    if (!done) return;
    const t = setTimeout(() => setDone(false), 1600);
    return () => clearTimeout(t);
  }, [done]);
  return (
    <Button
      type="button"
      variant={variant}
      size={iconOnly ? "icon" : size}
      className={cn(iconOnly && "h-7 w-7", className)}
      aria-label={iconOnly ? label : undefined}
      onClick={async () => {
        if (await copyText(value)) {
          setDone(true);
          toast.success("Copied to clipboard");
        } else toast.error("Copy failed — select the text and copy it manually.");
      }}
    >
      {done ? <Check className="h-3.5 w-3.5 text-ok" /> : <Copy className="h-3.5 w-3.5" />}
      {iconOnly ? null : done ? "Copied" : label}
    </Button>
  );
}

/** Read-only, selectable value with a copy button (keys, secrets, URIs). */
export function CopyField({ value, label, multiline, className }: { value: string; label: string; multiline?: boolean; className?: string }) {
  return (
    <div className={cn("space-y-1.5", className)}>
      <div className="label">{label}</div>
      {multiline ? (
        <div className="relative">
          <textarea
            readOnly
            value={value}
            aria-label={label}
            rows={Math.min(12, Math.max(3, value.split("\n").length))}
            onFocus={(e) => e.currentTarget.select()}
            className="block min-h-[64px] w-full resize-none rounded-lg border bg-surface-2/50 py-2.5 pl-3 pr-11 font-mono text-[12px] leading-relaxed text-fg focus:border-accent/60 focus:outline-none focus:ring-[3px] focus:ring-accent/15"
          />
          <CopyButton value={value} label={`Copy ${label.toLowerCase()}`} iconOnly variant="ghost" className="absolute right-1.5 top-1.5" />
        </div>
      ) : (
        <div className="flex h-9 items-center rounded-lg border bg-surface-2/50 pl-3 pr-1 focus-within:border-accent/60 focus-within:ring-[3px] focus-within:ring-accent/15">
          <input readOnly value={value} aria-label={label} onFocus={(e) => e.currentTarget.select()} className="h-full min-w-0 flex-1 bg-transparent font-mono text-[12px] text-fg focus:outline-none" />
          <CopyButton value={value} label={`Copy ${label.toLowerCase()}`} iconOnly variant="ghost" />
        </div>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------- callout */
const calloutTone: Record<Tone, string> = {
  neutral: "border-border bg-surface-2/50",
  accent: "border-accent/20 bg-accent/[0.045]",
  ok: "border-ok/20 bg-ok/[0.045]",
  warn: "border-warn/25 bg-warn/[0.055]",
  danger: "border-danger/20 bg-danger/[0.045]",
  info: "border-info/20 bg-info/[0.045]",
};
const calloutIcon: Record<Tone, string> = { neutral: "text-subtle", accent: "text-accent", ok: "text-ok", warn: "text-warn", danger: "text-danger", info: "text-info" };

export function Callout({ tone = "info", title, children, icon, className, action }: { tone?: Tone; title?: React.ReactNode; children?: React.ReactNode; icon?: React.ReactNode; className?: string; action?: React.ReactNode }) {
  const Icon = tone === "danger" || tone === "warn" ? AlertTriangle : Info;
  return (
    <div className={cn("flex flex-wrap gap-x-3 gap-y-2.5 rounded-xl border px-4 py-3 text-[13px] leading-relaxed sm:flex-nowrap", calloutTone[tone], className)} role={tone === "danger" || tone === "warn" ? "alert" : undefined}>
      <span className={cn("mt-[3px] shrink-0 [&_svg]:h-4 [&_svg]:w-4", calloutIcon[tone])}>{icon ?? <Icon />}</span>
      <div className="min-w-0 flex-1">
        {title ? <div className="font-medium text-fg">{title}</div> : null}
        {children ? <div className={cn("text-muted", title ? "mt-0.5" : undefined)}>{children}</div> : null}
      </div>
      {action ? <div className="shrink-0 self-center max-sm:basis-full max-sm:pl-7">{action}</div> : null}
    </div>
  );
}

/* --------------------------------------------------------- one-time secret */
/** Shows a secret exactly once. Closing requires acknowledging it was stored. */
export function SecretOnceDialog({ open, onClose, title, secret, secretLabel, description, children }: { open: boolean; onClose: () => void; title: string; secret: string; secretLabel: string; description?: React.ReactNode; children?: React.ReactNode }) {
  const [ack, setAck] = useState(false);
  useEffect(() => {
    if (open) setAck(false);
  }, [open]);
  return (
    <Dialog
      open={open}
      onOpenChange={(v) => {
        if (!v && ack) onClose();
        else if (!v) toast.warning("Confirm you have stored the secret before closing.");
      }}
    >
      <DialogContent
        title={title}
        description={description}
        footer={
          <>
            <span className="mr-auto hidden text-[12px] text-subtle sm:inline">{ack ? "You can close this now." : "Confirm you stored it to close."}</span>
            <Button disabled={!ack} onClick={onClose}>
              Done
            </Button>
          </>
        }
      >
        <div className="space-y-4">
          <section className="overflow-hidden rounded-xl border border-warn/30 bg-surface shadow-xs" aria-label={`${secretLabel}, shown only once`}>
            <div className="flex items-center gap-2.5 border-b border-warn/20 bg-warn/[0.07] px-4 py-2.5">
              <span className="grid h-6 w-6 shrink-0 place-items-center rounded-md bg-warn/15 text-warn">
                <KeyRound className="h-3.5 w-3.5" />
              </span>
              <span className="text-[13px] font-medium text-fg">Shown only once</span>
              <span className="label ml-auto hidden sm:inline">one-time reveal</span>
            </div>
            <div className="space-y-3 px-4 py-4">
              <div className="label">{secretLabel}</div>
              <div className="flex items-center gap-2 rounded-lg border bg-surface-2/60 p-1 pl-3 focus-within:border-accent/60 focus-within:ring-[3px] focus-within:ring-accent/15">
                <Lock className="h-3.5 w-3.5 shrink-0 text-subtle" aria-hidden />
                <input readOnly value={secret} aria-label={secretLabel} onFocus={(e) => e.currentTarget.select()} className="h-8 min-w-0 flex-1 bg-transparent font-mono text-[12.5px] tracking-[-0.01em] text-fg focus:outline-none" />
                <CopyButton value={secret} label="Copy" variant="primary" size="sm" />
              </div>
              <p className="text-[12.5px] leading-relaxed text-muted">Copy it now and store it in a secrets manager. It is stored hashed or encrypted and can never be displayed again — if you lose it, create a new one.</p>
            </div>
          </section>
          {children}
          <label className={cn("flex cursor-pointer items-center gap-3 rounded-xl border px-4 py-3 text-[13px] transition-colors", ack ? "border-ok/30 bg-ok/[0.05]" : "hover:bg-surface-2/50")}>
            <Checkbox checked={ack} onCheckedChange={setAck} label="I have stored this secret" />
            <span className="flex-1">I have copied and stored this secret somewhere safe</span>
            {ack ? <Check className="h-4 w-4 text-ok" aria-hidden /> : null}
          </label>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/* --------------------------------------------------------- permission gates */
export function permissionHint(perm: string) {
  return (
    <>
      Requires the <span className="font-mono">{perm}</span> permission
    </>
  );
}

/**
 * Button that stays visible but disabled (with a tooltip naming the missing
 * permission) when the current user lacks `perm`. `allowed` overrides the check.
 */
export function GatedButton({ perm, allowed, reason, children, ...props }: ButtonProps & { perm?: string; allowed?: boolean; reason?: React.ReactNode }) {
  const { can } = useAuth();
  const ok = allowed ?? (perm ? can(perm) : true);
  if (ok) return <Button {...props}>{children}</Button>;
  return (
    <Tip content={reason ?? (perm ? permissionHint(perm) : "Not available")}>
      <span tabIndex={0} className="inline-flex rounded-lg" aria-disabled="true">
        <Button {...props} disabled>
          {children}
        </Button>
      </span>
    </Tip>
  );
}

export function Forbidden({ perm, title = "You don't have access to this page", description }: { perm?: string; title?: string; description?: React.ReactNode }) {
  return (
    <div className="card overflow-hidden">
      <EmptyState
        icon={<Lock className="h-5 w-5" />}
        title={title}
        description={
          description ?? (
            <>
              {perm ? (
                <>
                  This area requires the <Mono className="rounded-[5px] border bg-surface-2 px-1.5 py-0.5 text-fg">{perm}</Mono> permission.{" "}
                </>
              ) : null}
              Ask an organization administrator to grant you a role that includes it.
            </>
          )
        }
      />
    </div>
  );
}

/** Link styled as a button (secondary or ink); for navigation, not actions. */
export function LinkButton({ href, children, variant = "secondary", className }: { href: string; children: React.ReactNode; variant?: "secondary" | "ink"; className?: string }) {
  return (
    <Link
      href={href}
      className={cn(
        "inline-flex h-9 select-none items-center justify-center gap-2 whitespace-nowrap rounded-lg px-3.5 text-[13.5px] font-medium tracking-[-0.005em] transition-[background,color,border-color] duration-150 focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/35 [&_svg]:h-4 [&_svg]:w-4",
        variant === "ink" ? "bg-fg text-bg shadow-[inset_0_1px_0_hsl(0_0%_100%/0.14),0_1px_2px_hsl(var(--shadow)/0.25)] hover:bg-fg/90" : "border bg-surface text-fg shadow-xs hover:border-border-strong hover:bg-surface-2/60",
        className,
      )}
    >
      {children}
    </Link>
  );
}

const PLATFORM_AREAS = ["Tenants, plans and limits", "User accounts across organizations", "System health and background jobs", "The platform audit log"];

export function PlatformForbidden() {
  const { can } = useAuth();
  return (
    <div className="card relative overflow-hidden">
      <div aria-hidden className="pointer-events-none absolute inset-0 bg-grid [mask-image:radial-gradient(ellipse_80%_70%_at_20%_0%,black,transparent_75%)]" />
      <div className="relative grid gap-10 px-6 py-10 sm:px-10 sm:py-14 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)] lg:items-center lg:gap-16">
        <div className="min-w-0">
          <div className="label flex items-center gap-2.5">
            <span className="inline-flex h-5 items-center rounded-[5px] border border-danger/25 bg-danger/[0.06] px-1.5 font-mono text-[10.5px] tracking-[0.08em] text-danger">403</span>
            Restricted area
          </div>
          <h2 className="mt-5 font-display text-[36px] leading-[1.04] tracking-[-0.01em] sm:text-[44px]">Platform administrators only</h2>
          <p className="mt-3 max-w-md text-[14px] leading-relaxed text-muted">This console manages tenants, users and system health across the whole deployment. Your account is not a platform administrator.</p>
          <div className="mt-7 flex flex-wrap gap-2">
            <LinkButton href="/overview" variant="ink">
              <ArrowLeft /> Back to overview
            </LinkButton>
            <LinkButton href="/settings?tab=organization">Organization settings</LinkButton>
          </div>
        </div>
        <div className="min-w-0 rounded-xl border bg-surface/80 shadow-xs backdrop-blur-[2px]">
          <div className="flex items-center gap-2 border-b px-4 py-3">
            <ShieldAlert className="h-4 w-4 text-subtle" />
            <span className="text-[13px] font-medium">What this console covers</span>
          </div>
          <ul className="divide-y">
            {PLATFORM_AREAS.map((a) => (
              <li key={a} className="flex items-center gap-3 px-4 py-2.5 text-[13px] text-muted">
                <Lock className="h-3.5 w-3.5 shrink-0 text-subtle" aria-hidden />
                {a}
              </li>
            ))}
          </ul>
          <p className="rounded-b-xl border-t bg-surface-2/40 px-4 py-3 text-[12.5px] leading-relaxed text-muted">
            Platform administrators can&apos;t read an organization&apos;s content either, unless one of its owners grants a time-boxed{" "}
            {can("org.security") ? (
              <Link href="/settings?tab=support" className="font-medium text-accent hover:underline">
                support access
              </Link>
            ) : (
              "support access"
            )}{" "}
            window.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ----------------------------------------------------------------- inputs */
export function SearchBox({ value, onChange, placeholder = "Search…", className, label = "Search" }: { value: string; onChange: (v: string) => void; placeholder?: string; className?: string; label?: string }) {
  return (
    <div className={cn("relative w-full sm:max-w-xs", className)}>
      <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-subtle" />
      <Input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} className="h-8 bg-surface-2/50 pl-8 text-[13px] shadow-none focus:bg-surface" aria-label={label} type="search" />
    </div>
  );
}

export function Segmented<T extends string | number>({ value, onChange, options, label, className }: { value: T; onChange: (v: T) => void; options: { value: T; label: React.ReactNode }[]; label: string; className?: string }) {
  return (
    <div role="group" aria-label={label} className={cn("inline-flex max-w-full items-center gap-0.5 overflow-x-auto rounded-[10px] border bg-surface-2/80 p-[3px]", className)}>
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          aria-pressed={o.value === value}
          onClick={() => onChange(o.value)}
          className={cn("inline-flex h-7 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-[7px] px-2.5 text-[12.5px] font-medium text-muted transition-all hover:text-fg [&_svg]:h-3.5 [&_svg]:w-3.5", o.value === value && "bg-surface text-fg shadow-xs ring-1 ring-border")}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function useDebounced<T>(value: T, ms = 300): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

/** `?tab=` synced tab state (call inside a Suspense boundary). */
export function useUrlTab<T extends string>(allowed: readonly T[], fallback: T): [T, (t: string) => void] {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const raw = params.get("tab");
  const current = (allowed as readonly string[]).includes(raw || "") ? (raw as T) : fallback;
  const set = useCallback(
    (t: string) => {
      const next = new URLSearchParams(params.toString());
      if (t === fallback) next.delete("tab");
      else next.set("tab", t);
      const qs = next.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [params, router, pathname, fallback],
  );
  return [current, set];
}

/** Spacing wrapper under a page-level `TabsList variant="line"` (which scrolls itself on small screens). */
export function TabBar({ children }: { children: React.ReactNode }) {
  return <div className="mb-7 min-w-0">{children}</div>;
}

/* ------------------------------------------------------------- pagination */
export function Pager({ total, limit, offset, onChange, className }: { total: number; limit: number; offset: number; onChange: (offset: number) => void; className?: string }) {
  if (total <= limit && offset === 0) return null;
  const page = Math.floor(offset / limit);
  const pages = Math.max(1, Math.ceil(total / limit));
  return (
    <div className={cn("flex items-center justify-between gap-3 border-t bg-surface-2/30 px-4 py-2.5 text-[12px] text-muted", className)}>
      <span className="num">
        {total === 0 ? 0 : offset + 1}–{Math.min(total, offset + limit)} of {formatNumber(total)}
      </span>
      <div className="flex items-center gap-1">
        <button className="rounded p-1 hover:bg-surface-2 disabled:opacity-40" disabled={page === 0} onClick={() => onChange(Math.max(0, offset - limit))} aria-label="Previous page">
          <ChevronLeft className="h-4 w-4" />
        </button>
        <span className="num px-1">
          {page + 1} / {pages}
        </span>
        <button className="rounded p-1 hover:bg-surface-2 disabled:opacity-40" disabled={page >= pages - 1} onClick={() => onChange(offset + limit)} aria-label="Next page">
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------- displays */
export function When({ iso, className }: { iso: string | null | undefined; className?: string }) {
  if (!iso) return <span className={cn("text-subtle", className)}>never</span>;
  return (
    <time dateTime={iso} title={formatDate(iso)} className={cn("whitespace-nowrap", className)}>
      {timeAgo(iso)}
    </time>
  );
}

export function PersonCell({ name, email, sub, you }: { name?: string | null; email?: string | null; sub?: React.ReactNode; you?: boolean }) {
  return (
    <div className="flex min-w-0 items-center gap-3">
      <Avatar name={name} email={email || undefined} className="h-8 w-8 text-[11px]" />
      <div className="min-w-0 leading-tight">
        <div className="flex items-center gap-1.5 font-medium text-fg">
          <span className="truncate">{name || email || "Unknown"}</span>
          {you ? <span className="shrink-0 rounded-[5px] bg-accent/10 px-1.5 py-px font-mono text-[9.5px] font-medium uppercase tracking-[0.08em] text-accent ring-1 ring-inset ring-accent/20">you</span> : null}
        </div>
        {name && email ? <div className="mt-0.5 truncate text-[12px] text-subtle">{email}</div> : null}
        {sub ? <div className="mt-0.5 truncate text-[12px] text-subtle">{sub}</div> : null}
      </div>
    </div>
  );
}

export function JsonBlock({ value, className }: { value: unknown; className?: string }) {
  const text = useMemo(() => {
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  }, [value]);
  return <pre className={cn("max-h-80 overflow-auto rounded-lg border bg-surface-2/50 p-3.5 font-mono text-[11.5px] leading-[1.7] text-fg", className)}>{text}</pre>;
}

export const RowMenuButton = forwardRef<HTMLButtonElement, ButtonProps & { label: string }>(function RowMenuButton({ label, className, ...props }, ref) {
  return (
    <Button ref={ref} variant="ghost" size="icon" aria-label={label} className={cn("h-7 w-7", className)} {...props}>
      <MoreHorizontal className="h-4 w-4" />
    </Button>
  );
});

export function KV({ label, children, className }: { label: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("flex items-baseline justify-between gap-4 py-2 text-[13px]", className)}>
      <span className="shrink-0 text-muted">{label}</span>
      <span className="min-w-0 truncate text-right">{children}</span>
    </div>
  );
}

/* ------------------------------------------------------------ plan meters */
export const LIMIT_LABELS: Record<string, string> = {
  users: "Members & pending invites",
  knowledge_bases: "Knowledge bases",
  documents: "Documents",
  storage_mb: "Storage",
  monthly_tokens: "Model tokens this month",
  api_calls_per_day: "API calls today",
  connected_sources: "Connected sources",
};

export function formatLimitValue(key: string, n: number | null | undefined) {
  if (n === null || n === undefined) return "—";
  if (key === "storage_mb") return formatBytes(n * 1024 * 1024);
  return formatNumber(n);
}

export function Meter({ label, used, limit, format = formatNumber }: { label: string; used: number; limit?: number | null; format?: (n: number) => string }) {
  const unlimited = limit === null || limit === undefined;
  const pct = unlimited ? 0 : limit === 0 ? 100 : Math.min(100, (used / limit) * 100);
  const tone = pct >= 100 ? "danger" : pct >= 80 ? "warn" : "ok";
  return (
    <div className="min-w-0">
      <div className="flex items-baseline justify-between gap-3 text-[13px]">
        <span className="truncate text-muted">{label}</span>
        <span className="num shrink-0 font-mono text-[12px] font-medium text-fg">
          {format(used)}
          <span className="font-normal text-subtle"> / {unlimited ? "Unlimited" : format(limit)}</span>
        </span>
      </div>
      <div
        className="mt-2 h-1.5 overflow-hidden rounded-full bg-surface-2 ring-1 ring-inset ring-border/60"
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={unlimited ? undefined : limit}
        aria-valuenow={used}
        aria-valuetext={unlimited ? `${format(used)} used, no limit` : `${format(used)} of ${format(limit)} (${Math.round(pct)}%)`}
      >
        {unlimited ? (
          <div className="h-full w-full bg-[repeating-linear-gradient(135deg,hsl(var(--border-strong)/0.7)_0_3px,transparent_3px_7px)]" />
        ) : (
          <div className={cn("h-full rounded-full transition-all", tone === "danger" ? "bg-danger" : tone === "warn" ? "bg-warn" : "bg-accent")} style={{ width: `${Math.max(pct, used > 0 ? 2 : 0)}%` }} />
        )}
      </div>
      {!unlimited && pct >= 80 ? <div className={cn("mt-1.5 text-[11.5px] font-medium", tone === "danger" ? "text-danger" : "text-warn")}>{pct >= 100 ? "Limit reached" : `${Math.round(pct)}% used`}</div> : null}
    </div>
  );
}

export function PlanMeters({ usage, limits, className }: { usage: Record<string, number>; limits: Record<string, number>; className?: string }) {
  const keys = Array.from(new Set([...Object.keys(LIMIT_LABELS).filter((k) => k in usage || k in limits), ...Object.keys(usage), ...Object.keys(limits)]));
  return (
    <div className={cn("grid gap-x-10 gap-y-5 sm:grid-cols-2", className)}>
      {keys.map((k) => (
        <Meter key={k} label={LIMIT_LABELS[k] || k.replace(/_/g, " ")} used={usage[k] ?? 0} limit={limits[k]} format={(n) => formatLimitValue(k, n)} />
      ))}
    </div>
  );
}

/* --------------------------------------------------------------- downloads */
/** Download an authenticated API response as a file (uses Content-Disposition when present). */
export async function downloadFile(path: string, query: Record<string, string | number | boolean | undefined | null> | undefined, fallbackName: string) {
  const res = await apiFetch(path, { query });
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
  const cd = res.headers.get("content-disposition") || "";
  const match = /filename\*=UTF-8''([^;]+)|filename="?([^";]+)"?/i.exec(cd);
  const name = match ? decodeURIComponent(match[1] || match[2]) : fallbackName;
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

/* ------------------------------------------------------------ chart theme */
/**
 * Two-series categorical palette: the accent (iris, or the org's branding accent)
 * and a coral partner. Iris/coral validated for CVD + normal-vision separation and
 * 3:1 contrast on both surfaces (dataviz validate_palette.js, light and dark).
 */
export const VIZ_VARS = "[--viz-2:17_82%_56%] dark:[--viz-2:17_70%_52%]";
export const VIZ_1 = "hsl(var(--accent))";
export const VIZ_2 = "hsl(var(--viz-2))";
export const axisTick = { fontSize: 10.5, fill: "hsl(var(--subtle))", fontFamily: "var(--font-mono)" };
export const gridProps = { vertical: false, stroke: "hsl(var(--border))", strokeDasharray: "2 4" } as const;
export const tooltipProps = {
  contentStyle: { background: "hsl(var(--surface))", border: "1px solid hsl(var(--border))", borderRadius: 10, fontSize: 12, color: "hsl(var(--fg))", boxShadow: "0 16px 40px -12px hsl(var(--shadow) / 0.28)" },
  labelStyle: { color: "hsl(var(--muted))", marginBottom: 4 },
  itemStyle: { color: "hsl(var(--fg))", padding: 0 },
  cursor: { stroke: "hsl(var(--border-strong))", strokeWidth: 1, strokeDasharray: "3 3" },
};

type TipItem = { name?: unknown; value?: unknown; color?: string; dataKey?: unknown };
/** Token-styled recharts tooltip: mono label, swatch + series name, tabular value. */
export function ChartTooltip({ active, payload, label, labelFormatter, valueFormatter, colors }: { active?: boolean; payload?: ReadonlyArray<TipItem>; label?: unknown; labelFormatter?: (l: string) => string; valueFormatter?: (v: number) => string; colors?: Record<string, string> }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="min-w-[168px] rounded-[10px] border bg-surface/95 px-3 py-2.5 shadow-pop backdrop-blur-sm">
      <div className="label mb-2 text-muted">{labelFormatter ? labelFormatter(String(label)) : String(label ?? "")}</div>
      <ul className="space-y-1.5">
        {payload.map((p) => {
          const key = String(p.dataKey ?? p.name);
          return (
            <li key={key} className="flex items-center gap-2 text-[12.5px]">
              <span className="h-2 w-2 shrink-0 rounded-[3px]" style={{ background: colors?.[key] ?? p.color }} />
              <span className="text-muted">{String(p.name ?? key)}</span>
              <span className="num ml-auto pl-5 font-mono text-[12px] font-medium text-fg">{valueFormatter ? valueFormatter(Number(p.value)) : String(p.value ?? "")}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/** Human duration: `640 ms`, `4.2 s`, `1m 54s`, `2h 05m`. */
export function formatDuration(ms: number | null | undefined) {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;
  const total = Math.round(s);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  return `${m}m ${String(sec).padStart(2, "0")}s`;
}

/* --------------------------------------------------------- settings layout */
/** Calm settings row: title + description on the left, controls on the right (stacked below lg). */
export function SettingsSection({ title, description, aside, children, className }: { title: React.ReactNode; description?: React.ReactNode; aside?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <section className={cn("grid gap-x-12 gap-y-4 border-t py-8 first:border-t-0 first:pt-0 lg:grid-cols-[minmax(0,280px)_minmax(0,1fr)]", className)}>
      <div className="min-w-0 lg:pt-1">
        <h2 className="text-[15px] font-semibold tracking-[-0.015em]">{title}</h2>
        {description ? <div className="mt-1 text-[13px] leading-relaxed text-muted">{description}</div> : null}
        {aside ? <div className="mt-3">{aside}</div> : null}
      </div>
      <div className="min-w-0 space-y-4">{children}</div>
    </section>
  );
}

/* -------------------------------------------------------- permission picker */
export function PermissionPicker({ catalog, value, onChange, held, readOnly, heldReason = "You don't hold this permission, so you can't grant it." }: { catalog: Permission[]; value: string[]; onChange?: (next: string[]) => void; held?: Set<string>; readOnly?: boolean; heldReason?: string }) {
  const [q, setQ] = useState("");
  const selected = useMemo(() => new Set(value), [value]);
  const groups = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return groupPermissions(catalog)
      .map((g) => ({ ...g, items: needle ? g.items.filter((p) => `${p.key} ${p.description} ${p.category}`.toLowerCase().includes(needle)) : g.items }))
      .filter((g) => g.items.length);
  }, [catalog, q]);
  const selectable = (p: Permission) => !readOnly && (!held || held.has(p.key));
  const toggle = (key: string, on: boolean) => {
    const next = new Set(selected);
    if (on) next.add(key);
    else next.delete(key);
    onChange?.(Array.from(next).sort());
  };
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <SearchBox value={q} onChange={setQ} placeholder="Filter permissions…" label="Filter permissions" />
        <span className="ml-auto text-xs text-muted">
          <span className="num font-medium text-fg">{selected.size}</span> of {catalog.length} selected
        </span>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        {groups.map((g) => {
          const pickable = g.items.filter(selectable);
          const on = g.items.filter((p) => selected.has(p.key)).length;
          const allOn = pickable.length > 0 && pickable.every((p) => selected.has(p.key));
          return (
            <fieldset key={g.category} className="overflow-hidden rounded-xl border bg-surface">
              <legend className="sr-only">{g.category}</legend>
              <div className="flex items-center gap-2.5 border-b bg-surface-2/50 px-3 py-2">
                {!readOnly ? (
                  <Checkbox
                    checked={allOn ? true : on > 0 ? "indeterminate" : false}
                    disabled={!pickable.length}
                    label={`Select all ${g.category} permissions`}
                    onCheckedChange={(v) => {
                      const next = new Set(selected);
                      if (v && !allOn) pickable.forEach((p) => next.add(p.key));
                      else g.items.forEach((p) => next.delete(p.key));
                      onChange?.(Array.from(next).sort());
                    }}
                  />
                ) : null}
                <span className="text-[12.5px] font-semibold">{g.category}</span>
                <span className="num ml-auto font-mono text-[11px] text-subtle">
                  {on}/{g.items.length}
                </span>
              </div>
              <ul className="divide-y">
                {g.items.map((p) => {
                  // Already-selected permissions the user lacks can still be removed, never added.
                  const can = selectable(p) || (!readOnly && selected.has(p.key));
                  const row = (
                    <label className={cn("flex items-start gap-2.5 px-3 py-2 transition-colors", can ? "cursor-pointer hover:bg-surface-2/50" : readOnly ? "" : "cursor-not-allowed opacity-55")}>
                      {readOnly ? (
                        <span className={cn("mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded-[5px]", selected.has(p.key) ? "bg-accent text-accent-fg" : "border border-dashed border-border-strong")}>{selected.has(p.key) ? <Check className="h-3 w-3" strokeWidth={3} /> : null}</span>
                      ) : (
                        <span className="mt-0.5">
                          <Checkbox checked={selected.has(p.key)} disabled={!can} onCheckedChange={(v) => toggle(p.key, v)} label={p.key} />
                        </span>
                      )}
                      <span className="min-w-0">
                        <span className="block font-mono text-[11.5px] text-fg">{p.key}</span>
                        <span className="mt-0.5 block text-[12px] leading-snug text-muted">{p.description}</span>
                      </span>
                    </label>
                  );
                  return (
                    <li key={p.key}>
                      {!can && !readOnly ? (
                        <Tip content={heldReason}>
                          <div>{row}</div>
                        </Tip>
                      ) : (
                        row
                      )}
                    </li>
                  );
                })}
              </ul>
            </fieldset>
          );
        })}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------ member picker */
export function MemberPicker({ members, value, onChange, exclude }: { members: Member[]; value: string[]; onChange: (ids: string[]) => void; exclude?: Set<string> }) {
  const [q, setQ] = useState("");
  const selected = useMemo(() => new Set(value), [value]);
  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return members.filter((m) => !exclude?.has(m.user_id)).filter((m) => !needle || `${m.full_name} ${m.email}`.toLowerCase().includes(needle));
  }, [members, q, exclude]);
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <SearchBox value={q} onChange={setQ} placeholder="Find people…" label="Find members" className="sm:max-w-none" />
        <span className="num shrink-0 text-xs text-muted">{selected.size} selected</span>
      </div>
      <ul className="max-h-72 divide-y overflow-y-auto rounded-xl border">
        {rows.length === 0 ? <li className="px-3 py-6 text-center text-[13px] text-muted">No matching members.</li> : null}
        {rows.map((m) => (
          <li key={m.user_id}>
            <label className="flex cursor-pointer items-center gap-3 px-3 py-2 hover:bg-surface-2/60">
              <Checkbox
                checked={selected.has(m.user_id)}
                label={m.full_name || m.email}
                onCheckedChange={(v) => {
                  const next = new Set(selected);
                  if (v) next.add(m.user_id);
                  else next.delete(m.user_id);
                  onChange(Array.from(next));
                }}
              />
              <PersonCell name={m.full_name} email={m.email} />
              {m.status !== "active" ? <span className="ml-auto text-2xs text-subtle">{m.status}</span> : null}
            </label>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Close helper for dialog footers. */
export function CancelButton({ label = "Cancel" }: { label?: string }) {
  return (
    <DialogClose asChild>
      <Button variant="secondary">{label}</Button>
    </DialogClose>
  );
}

export function usd(n: number | null | undefined, digits = 2) {
  if (n === null || n === undefined) return "—";
  return new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", currencyDisplay: "narrowSymbol", minimumFractionDigits: n > 0 && n < 1 ? Math.max(digits, 3) : digits, maximumFractionDigits: n > 0 && n < 1 ? 4 : digits }).format(n);
}
