"use client";

import * as CheckboxPrimitive from "@radix-ui/react-checkbox";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import * as SwitchPrimitive from "@radix-ui/react-switch";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { Check, Inbox } from "lucide-react";
import { createContext, useContext } from "react";

import { cn } from "@/lib/utils";

/* ------------------------------------------------------------------- page */
/** Standard page frame: centred column, generous gutters, staggered entrance. */
export function Page({ children, className, width = "default" }: { children: React.ReactNode; className?: string; width?: "default" | "narrow" | "wide" | "full" }) {
  return (
    <div
      className={cn(
        "mx-auto w-full px-5 pb-16 pt-7 sm:px-8 sm:pt-9",
        { default: "max-w-[1320px]", narrow: "max-w-3xl", wide: "max-w-[1560px]", full: "max-w-none" }[width],
        className,
      )}
    >
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ cards */
export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("card", className)} {...props} />;
}
export function CardHeader({ title, description, action, className }: { title: React.ReactNode; description?: React.ReactNode; action?: React.ReactNode; className?: string }) {
  return (
    <div className={cn("flex items-start justify-between gap-4 border-b px-5 py-4", className)}>
      <div className="min-w-0">
        <h3 className="text-[14px] font-semibold tracking-[-0.01em]">{title}</h3>
        {description ? <p className="mt-0.5 text-[13px] leading-relaxed text-muted">{description}</p> : null}
      </div>
      {action ? <div className="flex shrink-0 items-center gap-2">{action}</div> : null}
    </div>
  );
}
export function CardBody({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-5 py-4", className)} {...props} />;
}

/** A heading for a group of cards inside a page (not inside a card). */
export function SectionTitle({ title, description, action, className }: { title: React.ReactNode; description?: React.ReactNode; action?: React.ReactNode; className?: string }) {
  return (
    <div className={cn("mb-3 flex items-end justify-between gap-4", className)}>
      <div className="min-w-0">
        <h2 className="text-[15px] font-semibold tracking-[-0.015em]">{title}</h2>
        {description ? <p className="mt-0.5 text-[13px] text-muted">{description}</p> : null}
      </div>
      {action ? <div className="flex shrink-0 items-center gap-2">{action}</div> : null}
    </div>
  );
}

/* --------------------------------------------------------------- badges */
const badgeTone = {
  neutral: "bg-surface-2 text-muted ring-border",
  accent: "bg-accent/10 text-accent ring-accent/20",
  ok: "bg-ok/10 text-ok ring-ok/20",
  warn: "bg-warn/10 text-warn ring-warn/25",
  danger: "bg-danger/10 text-danger ring-danger/20",
  info: "bg-info/10 text-info ring-info/20",
} as const;
export type Tone = keyof typeof badgeTone;

export function Badge({ tone = "neutral", className, children }: { tone?: Tone; className?: string; children: React.ReactNode }) {
  return <span className={cn("inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2 py-[3px] text-[11.5px] font-medium leading-[14px] ring-1 ring-inset", badgeTone[tone], className)}>{children}</span>;
}

const STATUS: Record<string, { tone: Tone; label?: string }> = {
  ready: { tone: "ok" }, healthy: { tone: "ok" }, connected: { tone: "ok" }, succeeded: { tone: "ok" }, executed: { tone: "ok" }, complete: { tone: "ok" }, active: { tone: "ok" }, approved: { tone: "ok" }, delivered: { tone: "ok" }, success: { tone: "ok" }, tested: { tone: "ok" },
  processing: { tone: "info" }, queued: { tone: "info" }, running: { tone: "info" }, syncing: { tone: "info", label: "syncing" }, streaming: { tone: "info" },
  failed: { tone: "danger" }, dead: { tone: "danger", label: "failed permanently" }, error: { tone: "danger" }, rejected: { tone: "danger" }, denied: { tone: "danger" }, suspended: { tone: "danger" }, failure: { tone: "danger" },
  pending: { tone: "warn" }, pending_approval: { tone: "warn", label: "needs approval" }, paused: { tone: "warn" }, warning: { tone: "warn" }, beta: { tone: "warn" }, refused: { tone: "warn" }, stopped: { tone: "warn" },
  disabled: { tone: "neutral" }, unknown: { tone: "neutral" }, idle: { tone: "neutral" }, planned: { tone: "neutral" }, cancelled: { tone: "neutral" }, deactivated: { tone: "neutral" },
};

const DOT: Record<Tone, string> = { ok: "bg-ok", info: "bg-info", danger: "bg-danger", warn: "bg-warn", neutral: "bg-subtle", accent: "bg-accent" };

/** A small status dot; `live` adds a soft pulse ring. */
export function Dot({ tone = "neutral", live, className }: { tone?: Tone; live?: boolean; className?: string }) {
  return (
    <span className={cn("relative inline-flex h-1.5 w-1.5 shrink-0", className)}>
      {live ? <span className={cn("absolute inset-0 rounded-full opacity-60 animate-pulse-ring", DOT[tone])} /> : null}
      <span className={cn("relative h-1.5 w-1.5 rounded-full", DOT[tone])} />
    </span>
  );
}

export function StatusBadge({ status, className }: { status: string | null | undefined; className?: string }) {
  const s = STATUS[status || "unknown"] || { tone: "neutral" as Tone };
  return (
    <Badge tone={s.tone} className={className}>
      <Dot tone={s.tone} live={s.tone === "info"} />
      {(s.label || status || "unknown").replace(/_/g, " ")}
    </Badge>
  );
}

/* -------------------------------------------------------------- skeleton */
export function Skeleton({ className }: { className?: string }) {
  return (
    <div className={cn("relative overflow-hidden rounded-md bg-surface-2", className)}>
      <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-white/40 to-transparent dark:via-white/5" />
    </div>
  );
}

export function SkeletonRows({ rows = 5 }: { rows?: number }) {
  return (
    <div className="space-y-2.5 p-4">
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className={cn("h-9", i % 3 === 2 ? "w-4/5" : "w-full")} />
      ))}
    </div>
  );
}

/* ------------------------------------------------------------ empty state */
export function EmptyState({ icon, title, description, action, className }: { icon?: React.ReactNode; title: string; description?: React.ReactNode; action?: React.ReactNode; className?: string }) {
  return (
    <div className={cn("flex flex-col items-center justify-center px-6 py-16 text-center", className)}>
      <div className="relative mb-5 grid h-24 w-24 place-items-center">
        {/* concentric rings + dot field: a quiet, consistent illustration */}
        <span className="absolute inset-0 rounded-full border border-dashed border-border-strong/70" />
        <span className="absolute inset-[14px] rounded-full border border-border" />
        <span className="absolute inset-0 rounded-full bg-dots opacity-60 [mask-image:radial-gradient(circle,black_30%,transparent_70%)]" />
        <span className="relative grid h-11 w-11 place-items-center rounded-xl border bg-surface text-muted shadow-card">{icon || <Inbox className="h-5 w-5" />}</span>
      </div>
      <h3 className="text-[15px] font-semibold tracking-[-0.01em]">{title}</h3>
      {description ? <div className="mt-1.5 max-w-md text-[13px] leading-relaxed text-muted">{description}</div> : null}
      {action ? <div className="mt-6 flex flex-wrap items-center justify-center gap-2">{action}</div> : null}
    </div>
  );
}

/* ----------------------------------------------------------- page header */
export function PageHeader({ title, description, actions, eyebrow }: { title: React.ReactNode; description?: React.ReactNode; actions?: React.ReactNode; eyebrow?: React.ReactNode }) {
  return (
    <header className="mb-8 flex flex-wrap items-end justify-between gap-x-6 gap-y-4 animate-fade-up">
      <div className="min-w-0 max-w-3xl">
        {eyebrow ? (
          <div className="label mb-2.5 flex items-center gap-2">
            <span className="h-1 w-3 rounded-full bg-accent" />
            {eyebrow}
          </div>
        ) : null}
        <h1 className="text-[28px] font-semibold leading-[1.12] tracking-[-0.03em]">{title}</h1>
        {description ? <p className="mt-2 text-[14.5px] leading-relaxed text-muted">{description}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </header>
  );
}

/* ------------------------------------------------------------------ stat */
const StatGroupCtx = createContext(false);

/** One card holding several stats separated by hairlines (preferred over stat-card grids). */
export function StatGroup({ children, className, cols = 4 }: { children: React.ReactNode; className?: string; cols?: 2 | 3 | 4 | 5 | 6 }) {
  // Every cell draws its right and bottom hairline; the grid is pulled 1px past the
  // card's clipped edge so the outer ones disappear, for any count and column number.
  return (
    <StatGroupCtx.Provider value={true}>
      <div className={cn("card overflow-hidden", className)}>
        <div
          className={cn(
            "-mb-px -mr-px grid grid-cols-2 [&>*]:border-b [&>*]:border-r [&>*]:border-border",
            { 2: "sm:grid-cols-2", 3: "sm:grid-cols-3", 4: "sm:grid-cols-4", 5: "sm:grid-cols-3 lg:grid-cols-5", 6: "sm:grid-cols-3 lg:grid-cols-6" }[cols],
          )}
        >
          {children}
        </div>
      </div>
    </StatGroupCtx.Provider>
  );
}

const TONE_TEXT: Record<Tone, string> = { ok: "text-ok", warn: "text-warn", danger: "text-danger", info: "text-info", accent: "text-accent", neutral: "text-subtle" };

/** `tone` colours the icon; `valueTone` colours the number (state only, e.g. failures). */
export function Stat({ label, value, hint, icon, tone, valueTone, children }: { label: string; value: React.ReactNode; hint?: React.ReactNode; icon?: React.ReactNode; tone?: Tone; valueTone?: Tone; children?: React.ReactNode }) {
  const grouped = useContext(StatGroupCtx);
  const body = (
    <>
      <div className="flex items-center justify-between gap-2">
        <span className="label truncate">{label}</span>
        {icon ? <span className={cn("hidden shrink-0 sm:inline [&_svg]:h-3.5 [&_svg]:w-3.5", tone ? TONE_TEXT[tone] : "text-subtle")}>{icon}</span> : null}
      </div>
      <div className={cn("num mt-3 text-[28px] font-semibold leading-none tracking-[-0.035em]", valueTone && valueTone !== "neutral" && TONE_TEXT[valueTone])}>{value}</div>
      {hint ? <div className="mt-2 truncate text-[12.5px] text-muted">{hint}</div> : null}
      {children}
    </>
  );
  return grouped ? <div className="min-w-0 px-5 py-4">{body}</div> : <Card className="min-w-0 px-5 py-4">{body}</Card>;
}

/* ------------------------------------------------------------------ tabs */
type TabsVariant = "segmented" | "line";
const TabsVariantCtx = createContext<TabsVariant>("segmented");

export const Tabs = TabsPrimitive.Root;
/** `segmented` for compact view switches; `line` for page-level sections. */
export function TabsList({ className, variant = "segmented", ...props }: TabsPrimitive.TabsListProps & { variant?: TabsVariant }) {
  return (
    <TabsVariantCtx.Provider value={variant}>
      <TabsPrimitive.List
        className={cn(
          variant === "line" ? "flex items-center gap-6 overflow-x-auto border-b" : "inline-flex items-center gap-0.5 rounded-[10px] border bg-surface-2/80 p-[3px]",
          className,
        )}
        {...props}
      />
    </TabsVariantCtx.Provider>
  );
}
export function TabsTrigger({ className, ...props }: TabsPrimitive.TabsTriggerProps) {
  const variant = useContext(TabsVariantCtx);
  return (
    <TabsPrimitive.Trigger
      className={cn(
        variant === "line"
          ? "relative -mb-px inline-flex h-10 shrink-0 items-center gap-1.5 border-b-2 border-transparent text-[13.5px] font-medium text-muted transition-colors hover:text-fg data-[state=active]:border-fg data-[state=active]:text-fg"
          : "inline-flex h-7 items-center gap-1.5 rounded-[7px] px-3 text-[13px] font-medium text-muted transition-all hover:text-fg data-[state=active]:bg-surface data-[state=active]:text-fg data-[state=active]:shadow-xs data-[state=active]:ring-1 data-[state=active]:ring-border",
        className,
      )}
      {...props}
    />
  );
}
export const TabsContent = TabsPrimitive.Content;

/* --------------------------------------------------------------- tooltip */
export function Tip({ content, children, side = "top" }: { content: React.ReactNode; children: React.ReactNode; side?: "top" | "bottom" | "left" | "right" }) {
  return (
    <TooltipPrimitive.Provider delayDuration={250}>
      <TooltipPrimitive.Root>
        <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
        <TooltipPrimitive.Portal>
          <TooltipPrimitive.Content side={side} sideOffset={6} className="z-50 max-w-xs rounded-md bg-fg px-2 py-1 text-[12px] font-medium text-bg shadow-pop animate-scale-in">
            {content}
          </TooltipPrimitive.Content>
        </TooltipPrimitive.Portal>
      </TooltipPrimitive.Root>
    </TooltipPrimitive.Provider>
  );
}

/* ----------------------------------------------------- switch & checkbox */
export function Switch({ checked, onCheckedChange, disabled, label }: { checked: boolean; onCheckedChange: (v: boolean) => void; disabled?: boolean; label?: string }) {
  return (
    <SwitchPrimitive.Root
      aria-label={label}
      checked={checked}
      onCheckedChange={onCheckedChange}
      disabled={disabled}
      className="relative h-5 w-9 shrink-0 rounded-full bg-border-strong/80 shadow-[inset_0_1px_2px_hsl(var(--shadow)/0.12)] transition-colors data-[state=checked]:bg-accent disabled:opacity-50"
    >
      <SwitchPrimitive.Thumb className="block h-4 w-4 translate-x-0.5 rounded-full bg-white shadow-[0_1px_3px_hsl(0_0%_0%/0.25)] transition-transform duration-200 data-[state=checked]:translate-x-[18px]" />
    </SwitchPrimitive.Root>
  );
}

export function Checkbox({ checked, onCheckedChange, label, disabled }: { checked: boolean | "indeterminate"; onCheckedChange: (v: boolean) => void; label?: string; disabled?: boolean }) {
  return (
    <CheckboxPrimitive.Root
      aria-label={label}
      checked={checked}
      disabled={disabled}
      onCheckedChange={(v) => onCheckedChange(v === true)}
      className="grid h-4 w-4 shrink-0 place-items-center rounded-[5px] border border-border-strong bg-surface shadow-xs transition-colors data-[state=checked]:border-accent data-[state=checked]:bg-accent data-[state=indeterminate]:border-accent data-[state=indeterminate]:bg-accent disabled:opacity-50"
    >
      <CheckboxPrimitive.Indicator className="text-accent-fg">{checked === "indeterminate" ? <span className="block h-0.5 w-2 bg-current" /> : <Check className="h-3 w-3" strokeWidth={3} />}</CheckboxPrimitive.Indicator>
    </CheckboxPrimitive.Root>
  );
}

/* --------------------------------------------------------------- menus */
export const Menu = DropdownMenu.Root;
export const MenuTrigger = DropdownMenu.Trigger;
export function MenuContent({ className, align = "end", ...props }: DropdownMenu.DropdownMenuContentProps) {
  return (
    <DropdownMenu.Portal>
      <DropdownMenu.Content align={align} sideOffset={6} className={cn("z-50 min-w-[200px] rounded-xl border bg-surface p-1.5 shadow-float animate-scale-in", className)} {...props} />
    </DropdownMenu.Portal>
  );
}
export function MenuItem({ className, danger, ...props }: DropdownMenu.DropdownMenuItemProps & { danger?: boolean }) {
  return (
    <DropdownMenu.Item
      className={cn(
        "flex h-8 cursor-pointer select-none items-center gap-2.5 rounded-lg px-2 text-[13px] outline-none transition-colors data-[disabled]:pointer-events-none data-[highlighted]:bg-surface-2 data-[disabled]:opacity-40 [&_svg]:text-subtle",
        danger ? "text-danger [&_svg]:text-danger" : "text-fg",
        className,
      )}
      {...props}
    />
  );
}
export function MenuLabel({ className, ...props }: DropdownMenu.DropdownMenuLabelProps) {
  return <DropdownMenu.Label className={cn("label px-2 pb-1 pt-2", className)} {...props} />;
}
export function MenuSeparator() {
  return <DropdownMenu.Separator className="-mx-1.5 my-1.5 h-px bg-border" />;
}

/* ----------------------------------------------------------------- misc */
export function Kbd({ children, className }: { children: React.ReactNode; className?: string }) {
  return <kbd className={cn("inline-flex h-5 items-center rounded-[5px] border border-b-2 bg-surface px-1.5 font-mono text-[10.5px] font-medium text-muted", className)}>{children}</kbd>;
}

export function Avatar({ name, email, className }: { name?: string | null; email?: string; className?: string }) {
  const text = (name || email || "?").trim();
  const parts = text.split(/[\s@._-]+/).filter(Boolean);
  const ini = ((parts[0]?.[0] || "?") + (parts[1]?.[0] || "")).toUpperCase();
  let hash = 0;
  for (const ch of text) hash = (hash * 31 + ch.charCodeAt(0)) % 360;
  return (
    <span
      className={cn("grid h-7 w-7 shrink-0 place-items-center rounded-full text-[10.5px] font-semibold tracking-wide text-white ring-1 ring-inset ring-black/10", className)}
      style={{ background: `linear-gradient(140deg, hsl(${hash} 55% 58%), hsl(${(hash + 50) % 360} 60% 40%))` }}
    >
      {ini}
    </span>
  );
}

export function Mono({ children, className }: { children: React.ReactNode; className?: string }) {
  return <span className={cn("font-mono text-[12px] tracking-[-0.01em]", className)}>{children}</span>;
}
