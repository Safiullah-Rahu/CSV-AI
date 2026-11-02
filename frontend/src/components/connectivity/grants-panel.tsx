"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Plus, ShieldCheck, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { type Column, DataTable } from "@/components/ui/data-table";
import { ConfirmDialog, Dialog, DialogClose, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Select, Textarea } from "@/components/ui/form";
import { Badge, EmptyState, Switch, Tip } from "@/components/ui/primitives";
import { del, post } from "@/lib/api";
import { useApi } from "@/lib/hooks";
import { formatDate, timeAgo } from "@/lib/utils";

import { CheckList, Notice, invalidatePrefix, useMemberNames } from "./shared";
import type { DbConnection, Grant, Named, Role } from "./types";

const SUBJECT_TYPES = [
  { key: "user", label: "User" },
  { key: "team", label: "Team" },
  { key: "department", label: "Department" },
  { key: "role", label: "Role" },
  { key: "workspace", label: "Workspace" },
  { key: "org", label: "Everyone in the organization" },
] as const;

const PRIVILEGES = [
  { key: "read", label: "Read", hint: "View the connection and run read-only queries (with db.query)." },
  { key: "write", label: "Write", hint: "Currently equivalent to read: no connection action requires more than read or manage." },
  { key: "manage", label: "Manage", hint: "Full control: edit, discover schema, table selection, disconnect (each still needs its db.* permission)." },
] as const;

/** Look-ups that turn subject ids into names for every subject type. */
function useSubjects() {
  const members = useMemberNames();
  const teams = useApi<Named[]>("/teams", undefined, { staleTime: 60_000, meta: { silent: true } });
  const departments = useApi<Named[]>("/departments", undefined, { staleTime: 60_000, meta: { silent: true } });
  const roles = useApi<Role[]>("/roles", undefined, { staleTime: 60_000, meta: { silent: true } });
  const workspaces = useApi<Named[]>("/workspaces", undefined, { staleTime: 60_000, meta: { silent: true } });
  return useMemo(() => {
    const options: Record<string, { value: string; label: string }[]> = {
      user: members.members.map((m) => ({ value: m.user_id, label: m.full_name ? `${m.full_name} · ${m.email}` : m.email })),
      team: (teams.data ?? []).map((t) => ({ value: t.id, label: t.name })),
      department: (departments.data ?? []).map((d) => ({ value: d.id, label: d.name })),
      role: (roles.data ?? []).map((r) => ({ value: r.id, label: r.name })),
      workspace: (workspaces.data ?? []).map((w) => ({ value: w.id, label: w.name })),
      org: [],
    };
    const name = (type: string, id: string | null) => {
      if (type === "org") return "Everyone";
      if (!id) return "—";
      if (type === "user") return members.name(id);
      return options[type]?.find((o) => o.value === id)?.label ?? `${id.slice(0, 8)}…`;
    };
    return { options, name };
  }, [members, teams.data, departments.data, roles.data, workspaces.data]);
}

export function ConnectionGrants({ connection: c }: { connection: DbConnection }) {
  const qc = useQueryClient();
  const subjects = useSubjects();
  const grants = useApi<Grant[]>("/grants", { resource_type: "db_connection", resource_id: c.id });
  const [adding, setAdding] = useState(false);
  const [removing, setRemoving] = useState<Grant | null>(null);

  const remove = useMutation({
    mutationFn: (g: Grant) => del(`/grants/${g.id}`),
    onSuccess: () => {
      toast.success("Grant removed");
      invalidatePrefix(qc, "/grants", "/database-connections");
      setRemoving(null);
    },
  });

  const columns: Column<Grant>[] = [
    {
      key: "subject",
      header: "Subject",
      cell: (g) => (
        <div className="flex min-w-0 items-center gap-2">
          <Badge className="capitalize">{g.subject_type}</Badge>
          <span className="truncate font-medium tracking-[-0.005em]">{subjects.name(g.subject_type, g.subject_id)}</span>
        </div>
      ),
    },
    { key: "privilege", header: "Privilege", cell: (g) => <span className="capitalize">{g.privilege}</span> },
    { key: "effect", header: "Effect", cell: (g) => <Badge tone={g.effect === "deny" ? "danger" : "ok"}>{g.effect}</Badge> },
    {
      key: "tables",
      header: "Tables",
      cell: (g) =>
        g.constraints?.tables?.length ? (
          <Tip content={<span className="font-mono text-[11px]">{g.constraints.tables.join(", ")}</span>}>
            <span className="inline-flex flex-wrap gap-1" tabIndex={0}>
              {g.constraints.tables.slice(0, 2).map((t) => (
                <span key={t} className="rounded border bg-surface-2 px-1.5 py-0.5 font-mono text-[11px]">
                  {t}
                </span>
              ))}
              {g.constraints.tables.length > 2 ? <span className="text-2xs text-subtle">+{g.constraints.tables.length - 2}</span> : null}
            </span>
          </Tip>
        ) : (
          <span className="text-muted">All selected</span>
        ),
    },
    { key: "expires", header: "Expires", cell: (g) => (g.expires_at ? <span className={new Date(g.expires_at) < new Date() ? "text-danger" : ""}>{formatDate(g.expires_at)}</span> : <span className="text-subtle">Never</span>) },
    { key: "note", header: "Note", cell: (g) => <span className="block max-w-[200px] truncate text-muted" title={g.note}>{g.note || "—"}</span> },
    { key: "created", header: "Added", cell: (g) => <span className="whitespace-nowrap font-mono text-[12px] text-muted">{timeAgo(g.created_at)}</span> },
    {
      key: "actions",
      header: <span className="sr-only">Actions</span>,
      cell: (g) => (
        <Button size="icon" variant="ghost" aria-label="Remove grant" onClick={() => setRemoving(g)}>
          <Trash2 className="h-3.5 w-3.5" />
        </Button>
      ),
      className: "w-10",
    },
  ];

  return (
    <div className="space-y-4">
      <Notice tone="info">
        Grants give specific people, teams or roles access to this connection on top of their role permissions — they still need <span className="font-mono">db.view</span> to see it and <span className="font-mono">db.query</span> to query it. A table limit narrows a grantee to some of the selected tables. Deny grants override allows.
      </Notice>
      <DataTable
        rows={grants.data}
        columns={columns}
        rowKey={(g) => g.id}
        loading={grants.isLoading}
        searchable={false}
        toolbar={
          grants.data?.length ? (
            <Button size="sm" onClick={() => setAdding(true)}>
              <Plus className="h-3.5 w-3.5" /> Add grant
            </Button>
          ) : undefined
        }
        empty={
          <EmptyState
            icon={<ShieldCheck className="h-5 w-5" />}
            title="No connection-specific grants"
            description="Access currently comes from roles only: the owner and people with db.edit manage it; others with db.view can read it."
            action={
              <Button size="sm" onClick={() => setAdding(true)}>
                <Plus className="h-3.5 w-3.5" /> Add grant
              </Button>
            }
          />
        }
      />
      <AddGrantDialog connection={c} open={adding} onOpenChange={setAdding} options={subjects.options} />
      <ConfirmDialog
        open={Boolean(removing)}
        onOpenChange={(v) => !v && setRemoving(null)}
        title="Remove grant"
        description={removing ? `${subjects.name(removing.subject_type, removing.subject_id)} loses the ${removing.effect === "deny" ? "deny rule" : `${removing.privilege} access`} granted here. Role-based access is unaffected.` : undefined}
        confirmLabel="Remove grant"
        loading={remove.isPending}
        onConfirm={() => {
          if (removing) remove.mutate(removing);
        }}
      />
    </div>
  );
}

function AddGrantDialog({ connection: c, open, onOpenChange, options }: { connection: DbConnection; open: boolean; onOpenChange: (v: boolean) => void; options: Record<string, { value: string; label: string }[]> }) {
  const qc = useQueryClient();
  const [subjectType, setSubjectType] = useState<string>("user");
  const [subjectId, setSubjectId] = useState("");
  const [privilege, setPrivilege] = useState("read");
  const [effect, setEffect] = useState("allow");
  const [limit, setLimit] = useState(false);
  const [tables, setTables] = useState<string[]>([]);
  const [expires, setExpires] = useState("");
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);

  const reset = () => {
    setSubjectType("user");
    setSubjectId("");
    setPrivilege("read");
    setEffect("allow");
    setLimit(false);
    setTables([]);
    setExpires("");
    setNote("");
    setError(null);
  };

  const create = useMutation({
    mutationFn: () =>
      post<Grant>("/grants", {
        resource_type: "db_connection",
        resource_id: c.id,
        subject_type: subjectType,
        subject_id: subjectType === "org" ? null : subjectId,
        privilege,
        effect,
        expires_at: expires ? new Date(expires).toISOString() : null,
        constraints: limit && tables.length ? { tables } : {},
        note: note.trim(),
      }),
    onSuccess: () => {
      toast.success("Grant added");
      invalidatePrefix(qc, "/grants", "/database-connections");
      onOpenChange(false);
      reset();
    },
  });

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (subjectType !== "org" && !subjectId) return setError("Choose who this grant applies to.");
    if (limit && !tables.length) return setError("Choose at least one table, or turn off the table limit.");
    if (expires && new Date(expires) <= new Date()) return setError("The expiry must be in the future.");
    setError(null);
    create.mutate();
  };

  const list = options[subjectType] ?? [];
  return (
    <Dialog
      open={open}
      onOpenChange={(v) => {
        onOpenChange(v);
        if (!v) reset();
      }}
    >
      <DialogContent
        title="Add connection grant"
        description={`Give access to ${c.name}.`}
        footer={
          <>
            <DialogClose asChild>
              <Button variant="secondary">Cancel</Button>
            </DialogClose>
            <Button type="submit" form="grant-form" loading={create.isPending}>
              Add grant
            </Button>
          </>
        }
      >
        <form id="grant-form" onSubmit={submit} className="space-y-4" noValidate>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Applies to" htmlFor="g-type">
              <Select
                id="g-type"
                value={subjectType}
                onChange={(e) => {
                  setSubjectType(e.target.value);
                  setSubjectId("");
                }}
              >
                {SUBJECT_TYPES.map((s) => (
                  <option key={s.key} value={s.key}>
                    {s.label}
                  </option>
                ))}
              </Select>
            </Field>
            {subjectType !== "org" ? (
              <Field label={SUBJECT_TYPES.find((s) => s.key === subjectType)?.label} htmlFor="g-subject" hint={!list.length ? "Nothing to choose from (or you can't list these)." : undefined}>
                <Select id="g-subject" value={subjectId} onChange={(e) => setSubjectId(e.target.value)} disabled={!list.length}>
                  <option value="">Choose…</option>
                  {list.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </Select>
              </Field>
            ) : null}
            <Field label="Privilege" htmlFor="g-priv" hint={PRIVILEGES.find((p) => p.key === privilege)?.hint}>
              <Select id="g-priv" value={privilege} onChange={(e) => setPrivilege(e.target.value)}>
                {PRIVILEGES.map((p) => (
                  <option key={p.key} value={p.key}>
                    {p.label}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Effect" htmlFor="g-effect" hint={effect === "deny" ? "Blocks this access even if a role or another grant allows it." : undefined}>
              <Select id="g-effect" value={effect} onChange={(e) => setEffect(e.target.value)}>
                <option value="allow">Allow</option>
                <option value="deny">Deny</option>
              </Select>
            </Field>
          </div>

          {effect === "allow" ? (
            <div className="space-y-2">
              <label className="flex items-center justify-between gap-3">
                <span>
                  <span className="block text-[13px] font-medium">Limit to specific tables</span>
                  <span className="block text-xs text-muted">The grantee can only query these selected tables.</span>
                </span>
                <Switch checked={limit} onCheckedChange={setLimit} label="Limit to specific tables" />
              </label>
              {limit ? (
                <CheckList
                  label="Tables"
                  options={c.selected_tables.map((t) => ({ value: t, label: t, search: t }))}
                  value={tables}
                  onChange={setTables}
                  empty="No tables are selected on this connection yet — select tables on the Schema tab first."
                />
              ) : null}
            </div>
          ) : null}

          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Expires" htmlFor="g-exp" hint="Optional. Leave empty for no expiry.">
              <Input id="g-exp" type="datetime-local" value={expires} onChange={(e) => setExpires(e.target.value)} />
            </Field>
            <Field label="Note" htmlFor="g-note" hint="Why this access exists (audit trail).">
              <Textarea id="g-note" value={note} onChange={(e) => setNote(e.target.value)} maxLength={1000} className="min-h-[36px]" rows={1} />
            </Field>
          </div>
          {error ? <Notice tone="danger">{error}</Notice> : null}
        </form>
      </DialogContent>
    </Dialog>
  );
}
