"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, Eye, KeyRound, Link2, Lock, RotateCcw, ScanSearch, Search, ShieldAlert, Table2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Field, Input, Select, Textarea } from "@/components/ui/form";
import { Badge, Card, CardHeader, Checkbox, Dot, EmptyState, Skeleton, Switch } from "@/components/ui/primitives";
import { errorMessage, post, put } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatNumber, timeAgo } from "@/lib/utils";

import { GatedButton, Notice, invalidatePrefix } from "./shared";
import type { DbConnection, DiscoverResult, SchemaResponse, SchemaTable, TableSelection } from "./types";

type Draft = Record<string, TableSelection>;
type ColumnMode = "normal" | "excluded" | "sensitive";

function defaultSelection(t: SchemaTable): TableSelection {
  return { columns: t.columns.map((c) => c.name), excluded: [], sensitive: [], row_filter: null, incremental_column: null, primary_key: t.primary_key ?? [], requires_approval: false, ingest: true };
}

/** Normalise a stored selection against the discovered table (empty `columns` means "all"). */
function normalise(sel: Partial<TableSelection> | undefined, t: SchemaTable | undefined): TableSelection {
  const names = t?.columns.map((c) => c.name) ?? sel?.columns ?? [];
  return {
    columns: sel?.columns?.length ? sel.columns : names,
    excluded: sel?.excluded ?? [],
    sensitive: sel?.sensitive ?? [],
    row_filter: sel?.row_filter || null,
    incremental_column: sel?.incremental_column || null,
    primary_key: sel?.primary_key?.length ? sel.primary_key : t?.primary_key ?? [],
    requires_approval: Boolean(sel?.requires_approval),
    ingest: sel?.ingest ?? true,
  };
}

function stable(d: Draft) {
  return JSON.stringify(
    Object.keys(d)
      .sort()
      .map((k) => [k, { ...d[k], columns: [...d[k].columns].sort(), excluded: [...d[k].excluded].sort(), sensitive: [...d[k].sensitive].sort() }]),
  );
}

export function SchemaSelection({ connection: c }: { connection: DbConnection }) {
  const { can } = useAuth();
  const qc = useQueryClient();
  const manage = c.access === "manage";
  const canEdit = manage && can("db.schemas");
  const schema = useApi<SchemaResponse>(`/database-connections/${c.id}/schema`);
  const tables = schema.data?.tables ?? {};

  const saved = useMemo<Draft>(() => {
    const out: Draft = {};
    Object.entries(c.selection ?? {}).forEach(([k, v]) => (out[k] = normalise(v, tables[k])));
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [c.selection, schema.data]);

  const [draft, setDraft] = useState<Draft>(saved);
  const [active, setActive] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [onlySelected, setOnlySelected] = useState(false);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [saveError, setSaveError] = useState<string | null>(null);

  const savedKey = stable(saved);
  // Reset the draft only when the stored selection really changed (not on every background refetch).
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => setDraft(saved), [savedKey]);
  useEffect(() => {
    if (!active && schema.data) {
      const first = Object.keys(saved)[0] ?? Object.keys(schema.data.tables).sort()[0] ?? null;
      setActive(first);
    }
  }, [schema.data, saved, active]);

  const dirty = stable(draft) !== savedKey;

  const discover = useMutation({
    mutationFn: () => post<DiscoverResult>(`/database-connections/${c.id}/discover`),
    onSuccess: (r) => {
      toast.success(`Discovered ${formatNumber(r.tables)} tables`, { description: [r.added.length && `+${r.added.length} new`, r.removed.length && `−${r.removed.length} removed`, r.changed.length && `~${r.changed.length} changed`].filter(Boolean).join(" · ") || "No changes." });
      invalidatePrefix(qc, "/database-connections");
    },
  });

  const save = useMutation({
    mutationFn: (body: { tables: Draft }) => put<{ tables: Draft }>(`/database-connections/${c.id}/selection`, body),
    meta: { silent: true },
    onMutate: () => setSaveError(null),
    onSuccess: (r) => {
      toast.success(`Selection saved · ${Object.keys(r.tables).length} tables`);
      invalidatePrefix(qc, "/database-connections");
    },
    onError: (err) => {
      setSaveError(errorMessage(err));
      toast.error("Selection not saved", { description: errorMessage(err) });
    },
  });

  const grouped = useMemo(() => {
    const needle = q.toLowerCase();
    const by = new Map<string, string[]>();
    Object.keys(tables)
      .sort()
      .forEach((key) => {
        const t = tables[key];
        if (onlySelected && !draft[key]) return;
        if (needle && !key.toLowerCase().includes(needle) && !t.columns.some((col) => col.name.toLowerCase().includes(needle))) return;
        const g = t.schema || "default";
        by.set(g, [...(by.get(g) ?? []), key]);
      });
    return [...by.entries()];
  }, [tables, q, onlySelected, draft]);

  const toggleTable = (key: string, on: boolean) =>
    setDraft((d) => {
      const next = { ...d };
      if (on) next[key] = d[key] ?? saved[key] ?? defaultSelection(tables[key]);
      else delete next[key];
      return next;
    });
  const updateTable = (key: string, patch: Partial<TableSelection>) => setDraft((d) => (d[key] ? { ...d, [key]: { ...d[key], ...patch } } : d));

  const submit = () => {
    const empty = Object.entries(draft).filter(([, s]) => !s.columns.length).map(([k]) => k);
    if (empty.length) {
      setSaveError(`Choose at least one column for ${empty.join(", ")} or deselect the table.`);
      return;
    }
    save.mutate({ tables: draft });
  };

  if (schema.isLoading) {
    return (
      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        <Skeleton className="h-[420px] rounded-xl" />
        <Skeleton className="h-[420px] rounded-xl" />
      </div>
    );
  }

  if (!Object.keys(tables).length) {
    return (
      <Card>
        <EmptyState
          icon={<ScanSearch className="h-5 w-5" />}
          title={manage ? "Schema not discovered yet" : "No tables available to you"}
          description={
            manage
              ? "Discovery reads schemas, tables, columns, keys and row estimates using the stored credentials. Nothing is copied."
              : "A connection manager has not selected any tables you can see, or your grant limits you to tables that are not selected."
          }
          action={
            manage ? (
              <GatedButton allowed={canEdit} perm="db.schemas" onClick={() => discover.mutate()} loading={discover.isPending}>
                <ScanSearch className="h-4 w-4" /> Discover schema
              </GatedButton>
            ) : null
          }
        />
      </Card>
    );
  }

  const activeTable = active ? tables[active] : undefined;
  const selCount = Object.keys(draft).length;

  return (
    <div className="space-y-4">
      {!manage ? <Notice tone="info">You are seeing only the selected tables and the columns you are allowed to query. Managers configure the selection.</Notice> : null}
      <div className="grid gap-5 lg:grid-cols-[320px_minmax(0,1fr)]">
        {/* ---------------------------------------------------- table tree */}
        <Card className="flex max-h-[640px] min-h-[320px] flex-col overflow-hidden lg:sticky lg:top-6">
          <div className="space-y-2.5 border-b p-3">
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-subtle" />
              <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter tables or columns…" className="h-8 bg-surface-2/50 pl-8 text-[13px] shadow-none focus:bg-surface" aria-label="Filter tables or columns" />
            </div>
            {manage ? (
              <div className="flex items-center justify-between px-0.5 text-[12px] text-muted">
                <span>
                  <span className="num font-mono font-medium text-fg">{selCount}</span> of <span className="num font-mono">{Object.keys(tables).length}</span> selected
                </span>
                <label className="flex cursor-pointer items-center gap-1.5">
                  <Switch checked={onlySelected} onCheckedChange={setOnlySelected} label="Show selected only" /> Selected only
                </label>
              </div>
            ) : (
              <div className="px-0.5 text-[12px] text-muted">
                <span className="num font-mono font-medium text-fg">{Object.keys(tables).length}</span> tables available to you
              </div>
            )}
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto py-1">
            {grouped.map(([group, keys]) => {
              const selectedInGroup = keys.filter((k) => draft[k]).length;
              const isCollapsed = collapsed.has(group);
              return (
                <div key={group}>
                  <div className="sticky top-0 z-[1] flex h-8 items-center gap-2 border-b border-border/60 bg-surface/95 px-3 backdrop-blur">
                    <button
                      className="flex min-w-0 flex-1 items-center gap-1 text-left font-mono text-[10.5px] font-medium uppercase tracking-[0.1em] text-subtle transition-colors hover:text-fg"
                      onClick={() => setCollapsed((s) => {
                        const n = new Set(s);
                        if (n.has(group)) n.delete(group);
                        else n.add(group);
                        return n;
                      })}
                      aria-expanded={!isCollapsed}
                    >
                      {isCollapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                      <span className="truncate">{group}</span>
                      <span className="num ml-1 normal-case tracking-normal">
                        {selectedInGroup}/{keys.length}
                      </span>
                    </button>
                    {canEdit ? (
                      <Checkbox
                        checked={selectedInGroup === 0 ? false : selectedInGroup === keys.length ? true : "indeterminate"}
                        onCheckedChange={(v) => keys.forEach((k) => toggleTable(k, v))}
                        label={`Select all tables in ${group}`}
                      />
                    ) : null}
                  </div>
                  {!isCollapsed
                    ? keys.map((key) => {
                        const t = tables[key];
                        const sel = draft[key];
                        return (
                          <div key={key} className={cn("group flex h-9 items-center gap-2.5 px-3 text-[13px] transition-colors", active === key ? "bg-accent/[0.07] shadow-[inset_2px_0_0_hsl(var(--accent))]" : "hover:bg-surface-2/70")}>
                            {canEdit ? <Checkbox checked={Boolean(sel)} onCheckedChange={(v) => toggleTable(key, v)} label={`Select ${key}`} /> : <Table2 className="h-3.5 w-3.5 text-subtle" />}
                            <button className="flex min-w-0 flex-1 items-center gap-1.5 text-left" onClick={() => setActive(key)} aria-current={active === key}>
                              <span className={cn("truncate font-mono text-[12px] tracking-[-0.01em]", !sel && canEdit && "text-muted", active === key && "font-medium text-fg")}>{t.name}</span>
                              {t.kind !== "table" ? <Badge className="px-1.5 py-0 font-mono text-[10px]">{t.kind}</Badge> : null}
                              {sel?.requires_approval ? <Lock className="h-3 w-3 shrink-0 text-warn" aria-label="Requires approval" /> : null}
                              {sel?.sensitive.length ? <ShieldAlert className="h-3 w-3 shrink-0 text-danger" aria-label="Has sensitive columns" /> : null}
                            </button>
                            <span className="num shrink-0 font-mono text-[10.5px] text-subtle">{t.row_estimate !== null && t.row_estimate !== undefined ? formatNumber(t.row_estimate) : `${t.columns.length} col`}</span>
                          </div>
                        );
                      })
                    : null}
                </div>
              );
            })}
            {!grouped.length ? <p className="px-3 py-6 text-center text-[13px] text-muted">No tables match.</p> : null}
          </div>
          <div className="flex items-center border-t bg-surface-2/40 px-3 py-2 text-[11.5px] text-subtle">
            <span className="min-w-0 flex-1 truncate">
              Schema refreshed <span className="font-mono">{timeAgo(schema.data?.refreshed_at)}</span>
            </span>
            {canEdit ? (
              <button className="ml-2 shrink-0 font-medium text-accent hover:underline disabled:opacity-50" onClick={() => discover.mutate()} disabled={discover.isPending}>
                {discover.isPending ? "Discovering…" : "Re-discover"}
              </button>
            ) : null}
          </div>
        </Card>

        {/* ---------------------------------------------------- table detail */}
        <div className="min-w-0">
          {activeTable && active ? (
            <TableDetail key={active} tableKey={active} table={activeTable} sel={draft[active]} canEdit={canEdit} onToggle={(on) => toggleTable(active, on)} onChange={(p) => updateTable(active, p)} />
          ) : (
            <Card>
              <EmptyState title="Pick a table" description="Choose a table on the left to see its columns and configure access." />
            </Card>
          )}
        </div>
      </div>

      {canEdit ? (
        <div className={cn("sticky bottom-3 z-20 flex flex-wrap items-center gap-3 rounded-xl border bg-surface/95 px-4 py-2.5 backdrop-blur transition-shadow", dirty || saveError ? "border-border-strong shadow-float" : "shadow-card")}>
          <div className="flex min-w-0 flex-1 basis-56 items-center gap-2.5 text-[13px]">
            <Dot tone={saveError ? "danger" : dirty ? "warn" : "ok"} live={dirty && !saveError} />
            {saveError ? (
              <span className="text-danger">{saveError}</span>
            ) : dirty ? (
              <span>
                <span className="font-medium">Unsaved changes</span> <span className="text-muted">· <span className="num font-mono">{selCount}</span> tables selected</span>
              </span>
            ) : (
              <span className="text-muted">
                Selection is saved · <span className="num font-mono">{selCount}</span> tables
              </span>
            )}
          </div>
          <Button
            variant="ghost"
            size="sm"
            disabled={!dirty}
            onClick={() => {
              setDraft(saved);
              setSaveError(null);
            }}
          >
            <RotateCcw className="h-3.5 w-3.5" /> Discard
          </Button>
          <Button size="sm" disabled={!dirty} loading={save.isPending} onClick={submit}>
            Save selection
          </Button>
        </div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------- detail */

function TableDetail({ tableKey, table: t, sel, canEdit, onToggle, onChange }: { tableKey: string; table: SchemaTable; sel: TableSelection | undefined; canEdit: boolean; onToggle: (on: boolean) => void; onChange: (p: Partial<TableSelection>) => void }) {
  const fkCols = new Map<string, string>();
  t.foreign_keys.forEach((fk) => fk.columns.forEach((col, i) => fkCols.set(col, `${fk.references}.${fk.referred_columns[i] ?? ""}`)));
  const pk = new Set(sel?.primary_key?.length ? sel.primary_key : t.primary_key);
  const included = new Set(sel?.columns ?? []);
  const modeOf = (name: string): ColumnMode => (sel?.sensitive.includes(name) ? "sensitive" : sel?.excluded.includes(name) ? "excluded" : "normal");
  const editable = canEdit && Boolean(sel);
  const usable = sel ? sel.columns.filter((col) => modeOf(col) === "normal") : [];

  const setMode = (name: string, mode: ColumnMode) => {
    if (!sel) return;
    onChange({
      excluded: mode === "excluded" ? [...sel.excluded.filter((x) => x !== name), name] : sel.excluded.filter((x) => x !== name),
      sensitive: mode === "sensitive" ? [...sel.sensitive.filter((x) => x !== name), name] : sel.sensitive.filter((x) => x !== name),
      ...(mode !== "normal" && sel.incremental_column === name ? { incremental_column: null } : {}),
    });
  };
  const toggleColumn = (name: string, on: boolean) => {
    if (!sel) return;
    onChange({
      columns: on ? t.columns.map((c) => c.name).filter((n) => included.has(n) || n === name) : sel.columns.filter((x) => x !== name),
      ...(on ? {} : { excluded: sel.excluded.filter((x) => x !== name), sensitive: sel.sensitive.filter((x) => x !== name) }),
      ...(!on && sel.incremental_column === name ? { incremental_column: null } : {}),
    });
  };
  const allOn = t.columns.every((c) => included.has(c.name));

  return (
    <Card>
      <CardHeader
        title={
          <span className="flex items-center gap-2">
            <Table2 className="h-4 w-4 text-subtle" />
            <span className="font-mono text-[14px] tracking-[-0.01em]">{tableKey}</span>
          </span>
        }
        description={
          <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="capitalize">{t.kind}</span>
            <span className="num">{t.columns.length} columns</span>
            {t.row_estimate !== null && t.row_estimate !== undefined ? <span className="num">≈ {formatNumber(t.row_estimate)} rows</span> : null}
            {t.primary_key.length ? <span>PK: <span className="font-mono">{t.primary_key.join(", ")}</span></span> : <span className="text-warn">No primary key</span>}
          </span>
        }
        action={
          canEdit ? (
            <label className="flex cursor-pointer items-center gap-2 text-[13px] font-medium">
              <Switch checked={Boolean(sel)} onCheckedChange={onToggle} label={`Include ${tableKey}`} /> Include table
            </label>
          ) : sel ? (
            <Badge tone="ok">Selected</Badge>
          ) : null
        }
      />

      {canEdit && !sel ? (
        <div className="border-b bg-surface-2/40 px-5 py-2.5 text-[12.5px] text-muted">This table is not selected. Include it to configure columns, filters and sync behaviour.</div>
      ) : null}

      <div className="overflow-x-auto">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b bg-surface-2/50 text-left font-mono text-[10.5px] uppercase tracking-[0.1em] text-subtle">
              <th className="h-9 w-9 px-3">
                {editable ? <Checkbox checked={allOn ? true : included.size ? "indeterminate" : false} onCheckedChange={(v) => onChange({ columns: v ? t.columns.map((c) => c.name) : [] })} label="Select all columns" /> : null}
              </th>
              <th className="h-9 px-3 font-medium">Column</th>
              <th className="h-9 px-3 font-medium">Type</th>
              <th className="h-9 px-3 font-medium">Keys</th>
              {canEdit ? <th className="h-9 px-3 font-medium">Access</th> : null}
            </tr>
          </thead>
          <tbody>
            {t.columns.map((col) => {
              const on = included.has(col.name);
              const mode = modeOf(col.name);
              return (
                <tr key={col.name} className={cn("border-b transition-colors last:border-0 hover:bg-surface-2/40", sel && !on && "opacity-55")}>
                  <td className="h-10 px-3">{editable ? <Checkbox checked={on} onCheckedChange={(v) => toggleColumn(col.name, v)} label={`Select column ${col.name}`} /> : null}</td>
                  <td className="whitespace-nowrap px-3 font-mono text-[12.5px] tracking-[-0.01em]">
                    {col.name}
                    {!col.nullable ? <span className="ml-2 font-mono text-[10px] uppercase tracking-[0.08em] text-subtle">not null</span> : null}
                  </td>
                  <td className="whitespace-nowrap px-3 font-mono text-[11.5px] lowercase text-muted">{col.type}</td>
                  <td className="whitespace-nowrap px-3">
                    <span className="flex items-center gap-1">
                      {pk.has(col.name) ? (
                        <Badge tone="accent">
                          <KeyRound className="h-2.5 w-2.5" /> PK
                        </Badge>
                      ) : null}
                      {fkCols.has(col.name) ? (
                        <Badge tone="info">
                          <Link2 className="h-2.5 w-2.5" /> <span className="font-mono">{fkCols.get(col.name)}</span>
                        </Badge>
                      ) : null}
                    </span>
                  </td>
                  {canEdit ? (
                    <td className="px-3 py-1">
                      <Select aria-label={`Access for ${col.name}`} value={mode} disabled={!editable || !on} onChange={(e) => setMode(col.name, e.target.value as ColumnMode)} className={cn("h-7 w-[132px] text-xs", mode === "sensitive" && "border-danger/40 text-danger", mode === "excluded" && "border-warn/40 text-warn")}>
                        <option value="normal">Queryable</option>
                        <option value="excluded">Excluded</option>
                        <option value="sensitive">Sensitive</option>
                      </Select>
                    </td>
                  ) : null}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {canEdit ? (
        <div className="space-y-4 border-t px-5 py-4">
          <p className="flex items-start gap-1.5 text-[12px] leading-relaxed text-subtle">
            <Eye className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            Unchecked columns are not part of the selection. Excluded and sensitive columns are never queried or synchronised and are hidden from non-managers; mark personal or confidential data as sensitive.
          </p>
          <div className="grid gap-4 md:grid-cols-2">
            <Field label="Row filter" htmlFor="row-filter" hint="SQL boolean expression applied to every query and sync, e.g. region = 'EU' AND deleted_at IS NULL. Validated on save." className="md:col-span-2">
              <Textarea
                id="row-filter"
                value={sel?.row_filter ?? ""}
                disabled={!editable}
                onChange={(e) => onChange({ row_filter: e.target.value || null })}
                placeholder="status = 'active'"
                className="min-h-[64px] font-mono text-[12.5px]"
                spellCheck={false}
              />
            </Field>
            <Field label="Incremental column" htmlFor="inc-col" hint="A monotonically increasing timestamp or id. Incremental syncs fetch rows above the last value and propagate deletions by primary key.">
              <Select id="inc-col" value={sel?.incremental_column ?? ""} disabled={!editable} onChange={(e) => onChange({ incremental_column: e.target.value || null })}>
                <option value="">None — always full extract</option>
                {usable.map((name) => (
                  <option key={name} value={name}>
                    {name} ({t.columns.find((c) => c.name === name)?.type})
                  </option>
                ))}
              </Select>
            </Field>
            <div className="space-y-3">
              <label className="flex cursor-pointer items-start justify-between gap-3 rounded-lg border bg-surface px-3 py-2.5 transition-colors hover:border-border-strong">
                <span>
                  <span className="block text-[13px] font-medium">Requires approval</span>
                  <span className="mt-0.5 block text-[12px] leading-relaxed text-muted">Live queries touching this table wait for someone with db.approve.</span>
                </span>
                <Switch checked={Boolean(sel?.requires_approval)} disabled={!editable} onCheckedChange={(v) => onChange({ requires_approval: v })} label="Requires approval" />
              </label>
              <label className="flex cursor-pointer items-start justify-between gap-3 rounded-lg border bg-surface px-3 py-2.5 transition-colors hover:border-border-strong">
                <span>
                  <span className="block text-[13px] font-medium">Include in sync</span>
                  <span className="mt-0.5 block text-[12px] leading-relaxed text-muted">Ingest and hybrid modes copy this table into the target knowledge base.</span>
                </span>
                <Switch checked={sel?.ingest ?? true} disabled={!editable} onCheckedChange={(v) => onChange({ ingest: v })} label="Include in sync" />
              </label>
            </div>
          </div>
          {sel && sel.incremental_column && !(sel.primary_key?.length || t.primary_key.length) ? (
            <Notice tone="warn">This table has no primary key, so incremental syncs cannot merge updates or propagate deletions; they append new rows only.</Notice>
          ) : null}
        </div>
      ) : sel ? (
        <div className="grid gap-3 border-t px-5 py-4 text-[13px] sm:grid-cols-3">
          <div>
            <div className="label">Row filter</div>
            <div className="mt-1 font-mono text-[12px]">{sel.row_filter || "—"}</div>
          </div>
          <div>
            <div className="label">Approval</div>
            <div className="mt-1">{sel.requires_approval ? "Required" : "Not required"}</div>
          </div>
          <div>
            <div className="label">Sync</div>
            <div className="mt-1">{sel.ingest ? "Included" : "Excluded"}{sel.incremental_column ? <span className="text-muted"> · incremental on <span className="font-mono">{sel.incremental_column}</span></span> : null}</div>
          </div>
        </div>
      ) : null}
    </Card>
  );
}
