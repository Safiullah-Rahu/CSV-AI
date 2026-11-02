"use client";

import { useStore } from "@xyflow/react";
import { Boxes, Check, Crosshair, Expand, Layers, Maximize2, Minimize, Minus, MoreHorizontal, Plus, RotateCcw, Search, Sparkles, Type } from "lucide-react";
import { useMemo, useState } from "react";

import { NODE_FAMILY, NODE_ICON, TYPE_LABEL } from "@/components/connections/meta";
import { CATEGORY_FILTERS, STATUS_FILTERS, type GraphNode } from "@/components/connections/types";
import { Select } from "@/components/ui/form";
import { Dot, Menu, MenuContent, MenuItem, MenuLabel, MenuSeparator, MenuTrigger, Tip, type Tone } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

/* ----------------------------------------------------------- map toolbar */

function CtlButton({ label, onClick, active, children }: { label: string; onClick: () => void; active?: boolean; children: React.ReactNode }) {
  return (
    <Tip content={label} side="bottom">
      <button
        type="button"
        onClick={onClick}
        aria-label={label}
        aria-pressed={active}
        className={cn(
          "grid h-8 w-8 place-items-center rounded-lg text-muted transition-colors hover:bg-surface-2 hover:text-fg [&_svg]:h-4 [&_svg]:w-4",
          active && "bg-accent/10 text-accent hover:bg-accent/15 hover:text-accent",
        )}
      >
        {children}
      </button>
    </Tip>
  );
}

const Sep = () => <span className="mx-1 h-5 w-px shrink-0 bg-border" aria-hidden />;

/** Live zoom readout, instrument style. */
function ZoomReadout() {
  const zoom = useStore((s) => s.transform[2]);
  return (
    <span className="num w-11 text-center font-mono text-[11px] text-muted" aria-live="off">
      {Math.round(zoom * 100)}%
    </span>
  );
}

export function MapControls({
  onZoomIn,
  onZoomOut,
  onFit,
  onReset,
  onCenterOrg,
  animate,
  setAnimate,
  labels,
  setLabels,
  grouped,
  setGrouped,
  onFullscreen,
  fullscreen,
  compact,
}: {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFit: () => void;
  onReset: () => void;
  onCenterOrg: () => void;
  animate: boolean;
  setAnimate: (v: boolean) => void;
  labels: boolean;
  setLabels: (v: boolean) => void;
  grouped: boolean;
  setGrouped: (v: boolean) => void;
  onFullscreen: () => void;
  fullscreen: boolean;
  compact?: boolean;
}) {
  if (compact) {
    // Phones: the essentials inline, everything else in one menu.
    return (
      <div role="toolbar" aria-label="Map controls" className="cx-glass flex items-center gap-0.5 rounded-xl p-1">
        <CtlButton label="Zoom out" onClick={onZoomOut}><Minus /></CtlButton>
        <CtlButton label="Zoom in" onClick={onZoomIn}><Plus /></CtlButton>
        <CtlButton label="Fit all nodes" onClick={onFit}><Maximize2 /></CtlButton>
        <Menu>
          <MenuTrigger asChild>
            <button type="button" aria-label="More map options" className="grid h-8 w-8 place-items-center rounded-lg text-muted transition-colors hover:bg-surface-2 hover:text-fg">
              <MoreHorizontal className="h-4 w-4" />
            </button>
          </MenuTrigger>
          <MenuContent>
            <MenuItem onSelect={onCenterOrg}><Crosshair className="h-4 w-4" /> Center organization</MenuItem>
            <MenuItem onSelect={onReset}><RotateCcw className="h-4 w-4" /> Reset view</MenuItem>
            <MenuSeparator />
            <MenuLabel>Display</MenuLabel>
            <ToggleItem on={animate} onSelect={() => setAnimate(!animate)}>Animation</ToggleItem>
            <ToggleItem on={labels} onSelect={() => setLabels(!labels)}>Labels</ToggleItem>
            <ToggleItem on={grouped} onSelect={() => setGrouped(!grouped)}>Group sources by provider</ToggleItem>
            <ToggleItem on={fullscreen} onSelect={onFullscreen}>Full screen</ToggleItem>
          </MenuContent>
        </Menu>
      </div>
    );
  }
  return (
    <div role="toolbar" aria-label="Map controls" className="cx-glass flex items-center gap-0.5 rounded-xl p-1">
      <CtlButton label="Zoom out" onClick={onZoomOut}><Minus /></CtlButton>
      <ZoomReadout />
      <CtlButton label="Zoom in" onClick={onZoomIn}><Plus /></CtlButton>
      <CtlButton label="Fit all nodes" onClick={onFit}><Maximize2 /></CtlButton>
      <Sep />
      <CtlButton label="Center organization" onClick={onCenterOrg}><Crosshair /></CtlButton>
      <CtlButton label="Reset view" onClick={onReset}><RotateCcw /></CtlButton>
      <Sep />
      <CtlButton label={animate ? "Disable animation" : "Enable animation"} onClick={() => setAnimate(!animate)} active={animate}><Sparkles /></CtlButton>
      <CtlButton label={labels ? "Hide labels" : "Show labels"} onClick={() => setLabels(!labels)} active={labels}><Type /></CtlButton>
      <CtlButton label={grouped ? "Ungroup sources" : "Group sources by provider"} onClick={() => setGrouped(!grouped)} active={grouped}><Layers /></CtlButton>
      <CtlButton label={fullscreen ? "Exit full screen" : "Full screen"} onClick={onFullscreen} active={fullscreen}>{fullscreen ? <Minimize /> : <Expand />}</CtlButton>
    </div>
  );
}

function ToggleItem({ on, onSelect, children }: { on: boolean; onSelect: () => void; children: React.ReactNode }) {
  return (
    <MenuItem onSelect={(e) => { e.preventDefault(); onSelect(); }} role="menuitemcheckbox" aria-checked={on}>
      <span className={cn("grid h-4 w-4 place-items-center rounded-[5px] border", on ? "border-accent bg-accent text-accent-fg" : "border-border-strong")}>
        {on ? <Check className="!h-3 !w-3 !text-accent-fg" strokeWidth={3} /> : null}
      </span>
      {children}
    </MenuItem>
  );
}

/* ----------------------------------------------------------- view switch */

export function ViewSwitch<V extends string>({ value, onChange, options }: { value: V; onChange: (v: V) => void; options: readonly (readonly [V, React.ComponentType<{ className?: string }>, string])[] }) {
  return (
    <div role="group" aria-label="View" className="inline-flex items-center gap-0.5 rounded-[10px] border bg-surface-2/80 p-[3px]">
      {options.map(([key, Icon, label]) => (
        <button
          key={key}
          type="button"
          onClick={() => onChange(key)}
          aria-pressed={value === key}
          aria-label={label}
          className={cn(
            "inline-flex h-7 items-center gap-1.5 rounded-[7px] px-2.5 text-[13px] font-medium text-muted transition-all hover:text-fg sm:px-3",
            value === key && "bg-surface text-fg shadow-xs ring-1 ring-border",
          )}
        >
          <Icon className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">{label}</span>
        </button>
      ))}
    </div>
  );
}

/* --------------------------------------------------------------- filters */

const STATUS_DOT: Record<string, Tone> = { active: "ok", failed: "danger", syncing: "info" };

function Chip({ on, onClick, children, count }: { on: boolean; onClick: () => void; children: React.ReactNode; count?: number }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={on}
      className={cn(
        "inline-flex h-7 shrink-0 items-center gap-1.5 rounded-full border px-2.5 text-[12px] font-medium transition-colors",
        on ? "border-accent/35 bg-accent/10 text-accent" : "bg-surface text-muted shadow-xs hover:border-border-strong hover:text-fg",
      )}
    >
      {children}
      {count !== undefined ? <span className={cn("num font-mono text-[10.5px]", on ? "text-accent/75" : "text-subtle")}>{count}</span> : null}
    </button>
  );
}

/** Server-side scope: which workspace / knowledge base the graph is built for. */
export function ScopeSelects({
  workspaces,
  workspaceId,
  setWorkspaceId,
  kbs,
  kbId,
  setKbId,
  className,
}: {
  workspaces: { id: string; name: string }[];
  workspaceId: string;
  setWorkspaceId: (id: string) => void;
  kbs: { id: string; name: string }[];
  kbId: string;
  setKbId: (id: string) => void;
  className?: string;
}) {
  return (
    <div className={cn("flex shrink-0 items-center gap-1.5", className)}>
      <Select value={workspaceId} onChange={(e) => setWorkspaceId(e.target.value)} className={cn("!h-8 w-auto max-w-[170px] text-[12.5px]", workspaceId && "border-accent/40 text-accent")} aria-label="Filter by workspace">
        <option value="">All workspaces</option>
        {workspaces.map((w) => <option key={w.id} value={w.id}>{w.name}</option>)}
      </Select>
      <Select value={kbId} onChange={(e) => setKbId(e.target.value)} className={cn("!h-8 w-auto max-w-[190px] text-[12.5px]", kbId && "border-accent/40 text-accent")} aria-label="Filter by knowledge base">
        <option value="">All knowledge bases</option>
        {kbs.map((k) => <option key={k.id} value={k.id}>{k.name}</option>)}
      </Select>
    </div>
  );
}

export function ConnectionFilters({
  nodes,
  category,
  setCategory,
  statuses,
  setStatuses,
  children,
}: {
  nodes: GraphNode[];
  category: string;
  setCategory: (c: string) => void;
  statuses: string[];
  setStatuses: (s: string[]) => void;
  /** Extra controls appended to the scrolling row (e.g. scope selects on narrow screens). */
  children?: React.ReactNode;
}) {
  const countable = nodes.filter((n) => n.type !== "organization");
  return (
    <div className="cx-noscroll cx-fade-x flex min-w-0 items-center gap-1.5 overflow-x-auto pr-6">
      {CATEGORY_FILTERS.map((f) => (
        <Chip key={f.key} on={category === f.key} onClick={() => setCategory(f.key)} count={nodes.length ? countable.filter(f.match).length : undefined}>
          {f.label}
        </Chip>
      ))}
      <span className="mx-1 h-4 w-px shrink-0 bg-border" aria-hidden />
      {STATUS_FILTERS.map((f) => {
        const on = statuses.includes(f.key);
        return (
          <Chip key={f.key} on={on} onClick={() => setStatuses(on ? statuses.filter((s) => s !== f.key) : [...statuses, f.key])} count={nodes.length ? countable.filter(f.match).length : undefined}>
            <Dot tone={STATUS_DOT[f.key] ?? "neutral"} />
            {f.label}
          </Chip>
        );
      })}
      {children}
    </div>
  );
}

/* ---------------------------------------------------------------- search */

export function SearchFocus({ nodes, onFocus, className }: { nodes: GraphNode[]; onFocus: (n: GraphNode) => void; className?: string }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const hits = useMemo(() => (q ? nodes.filter((n) => n.label.toLowerCase().includes(q.toLowerCase()) || n.type.includes(q.toLowerCase())).slice(0, 8) : []), [q, nodes]);
  const pick = (n: GraphNode) => {
    onFocus(n);
    setOpen(false);
    setQ(n.label);
  };
  return (
    <div className={cn("relative", className)}>
      <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-subtle" />
      <input
        value={q}
        onChange={(e) => { setQ(e.target.value); setOpen(true); setActive(0); }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown" && hits.length) { e.preventDefault(); setActive((a) => (a + 1) % hits.length); }
          if (e.key === "ArrowUp" && hits.length) { e.preventDefault(); setActive((a) => (a - 1 + hits.length) % hits.length); }
          if (e.key === "Enter" && hits[active]) pick(hits[active]);
          if (e.key === "Escape") setOpen(false);
        }}
        placeholder="Search and focus a resource"
        className="h-8 w-full rounded-lg border bg-surface pl-8 pr-3 text-[13px] text-fg shadow-xs transition-[border-color,box-shadow] placeholder:text-subtle hover:border-border-strong focus:border-accent/70 focus:outline-none focus:ring-[3px] focus:ring-accent/15"
        aria-label="Search resources on the map"
        role="combobox"
        aria-expanded={open && hits.length > 0}
        aria-autocomplete="list"
      />
      {open && q && hits.length === 0 ? (
        <div className="absolute left-0 top-10 z-30 w-full min-w-[260px] rounded-xl border bg-surface px-3 py-2.5 text-[12.5px] text-subtle shadow-float animate-scale-in">No resource on the map matches “{q}”.</div>
      ) : null}
      {open && hits.length ? (
        <ul role="listbox" className="absolute left-0 top-10 z-30 w-full min-w-[300px] rounded-xl border bg-surface p-1.5 shadow-float animate-scale-in">
          {hits.map((n, i) => {
            const Icon = NODE_ICON[n.type] ?? Boxes;
            const fam = NODE_FAMILY[n.type] ?? "source";
            return (
              <li key={n.id} role="option" aria-selected={i === active}>
                <button
                  type="button"
                  onMouseDown={() => pick(n)}
                  onMouseEnter={() => setActive(i)}
                  className={cn("flex w-full items-center gap-2.5 rounded-lg px-2 py-1.5 text-left text-[13px] text-fg", i === active && "bg-surface-2")}
                >
                  <span className={cn("grid h-6 w-6 shrink-0 place-items-center rounded-md [&_svg]:h-3.5 [&_svg]:w-3.5", fam === "intelligence" ? "bg-accent text-accent-fg" : fam === "knowledge" ? "bg-accent/10 text-accent" : fam === "org" ? "bg-fg text-bg" : "border bg-surface-2 text-muted")}>
                    <Icon />
                  </span>
                  <span className="min-w-0 flex-1 truncate">{n.label}</span>
                  <span className="shrink-0 font-mono text-[10.5px] text-subtle">{TYPE_LABEL[n.type] ?? n.type}</span>
                </button>
              </li>
            );
          })}
        </ul>
      ) : null}
    </div>
  );
}
