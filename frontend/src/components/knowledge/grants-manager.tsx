"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Building, Clock, Globe2, KeyRound, Plus, ShieldBan, ShieldCheck, Trash2, UserRound, Users } from "lucide-react";
import { useMemo, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/dialog";
import { Field, Input, Select } from "@/components/ui/form";
import { Avatar, Badge, EmptyState, Skeleton } from "@/components/ui/primitives";
import { del, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatDate } from "@/lib/utils";

import { invalidatePaths, memberLabel, Segmented, useMemberMap } from "./shared";
import type { AccessGrantSummary, Department, Effect, Grant, Privilege, Role, SubjectType, Team, Workspace } from "./types";

const SUBJECTS: { value: SubjectType; label: string; icon: typeof UserRound }[] = [
  { value: "user", label: "User", icon: UserRound },
  { value: "team", label: "Team", icon: Users },
  { value: "department", label: "Department", icon: Building },
  { value: "role", label: "Role", icon: KeyRound },
  { value: "workspace", label: "Workspace", icon: Users },
  { value: "org", label: "Whole organization", icon: Globe2 },
];
const SUBJECT_ICON = Object.fromEntries(SUBJECTS.map((s) => [s.value, s.icon])) as Record<SubjectType, typeof UserRound>;

/** Names for every kind of grant subject the viewer is allowed to list. */
export function useSubjectDirectory(enabled = true) {
  const { can } = useAuth();
  const members = useMemberMap();
  const orgView = enabled && can("org.view");
  const teams = useApi<Team[]>(orgView ? "/teams" : null, undefined, { staleTime: 60_000 });
  const departments = useApi<Department[]>(orgView ? "/departments" : null, undefined, { staleTime: 60_000 });
  const roles = useApi<Role[]>(orgView ? "/roles" : null, undefined, { staleTime: 60_000 });
  const workspaces = useApi<Workspace[]>(enabled && can("workspace.view") ? "/workspaces" : null, undefined, { staleTime: 60_000 });
  return useMemo(() => {
    const label = (type: SubjectType, id: string | null): string => {
      if (type === "org") return "Everyone in the organization";
      if (!id) return "—";
      if (type === "user") return memberLabel(members, id);
      const list: { id: string; name: string }[] | undefined = { team: teams.data, department: departments.data, role: roles.data, workspace: workspaces.data }[type];
      return list?.find((x) => x.id === id)?.name ?? `${type} ${id.slice(0, 8)}`;
    };
    return { members, teams: teams.data ?? [], departments: departments.data ?? [], roles: roles.data ?? [], workspaces: workspaces.data ?? [], label };
  }, [members, teams.data, departments.data, roles.data, workspaces.data]);
}

type Directory = ReturnType<typeof useSubjectDirectory>;

export function GrantRow({ g, dir, onRemove }: { g: AccessGrantSummary & { id?: string; note?: string }; dir: Directory; onRemove?: () => void }) {
  const Icon = SUBJECT_ICON[g.subject_type] ?? UserRound;
  const expired = g.expires_at ? new Date(g.expires_at).getTime() < Date.now() : false;
  const m = g.subject_type === "user" && g.subject_id ? dir.members.get(g.subject_id) : undefined;
  return (
    <li className={cn("flex flex-wrap items-center gap-3 px-5 py-3 transition-colors hover:bg-surface-2/40", expired && "opacity-60")}>
      {m ? <Avatar name={m.full_name} email={m.email} /> : <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full border bg-surface-2 text-muted"><Icon className="h-3.5 w-3.5" /></span>}
      <div className="min-w-0 flex-1">
        <div className="truncate text-[13px] font-medium">{dir.label(g.subject_type, g.subject_id)}</div>
        <div className="flex flex-wrap items-center gap-x-2 text-[11.5px] text-subtle">
          <span className="capitalize">{g.subject_type === "org" ? "organization" : g.subject_type}</span>
          {m?.email && m.full_name ? <span className="truncate">{m.email}</span> : null}
          {g.note ? <span className="truncate">· {g.note}</span> : null}
        </div>
      </div>
      <Badge tone={g.effect === "deny" ? "danger" : "ok"}>
        {g.effect === "deny" ? <ShieldBan className="h-3 w-3" /> : <ShieldCheck className="h-3 w-3" />}
        {g.effect} {g.privilege}
      </Badge>
      {g.expires_at ? (
        <Badge tone={expired ? "neutral" : "warn"}>
          <Clock className="h-3 w-3" /> {expired ? "expired" : `until ${formatDate(g.expires_at, false)}`}
        </Badge>
      ) : null}
      {onRemove ? (
        <Button variant="ghost" size="icon" onClick={onRemove} aria-label={`Remove grant for ${dir.label(g.subject_type, g.subject_id)}`}>
          <Trash2 className="h-3.5 w-3.5" />
        </Button>
      ) : null}
    </li>
  );
}

/** Lists and edits the explicit grants on one resource (knowledge base or document). */
export function GrantsManager({ resourceType, resourceId, resourceName, invalidate = [] }: { resourceType: "knowledge_base" | "document"; resourceId: string; resourceName: string; invalidate?: string[] }) {
  const qc = useQueryClient();
  const dir = useSubjectDirectory();
  const grants = useApi<Grant[]>("/grants", { resource_type: resourceType, resource_id: resourceId });
  const [removing, setRemoving] = useState<Grant | null>(null);
  const [adding, setAdding] = useState(false);

  const remove = useMutation({
    mutationFn: (g: Grant) => del(`/grants/${g.id}`),
    onSuccess: () => {
      toast.success("Grant removed");
      setRemoving(null);
      invalidatePaths(qc, "/grants", ...invalidate);
    },
  });

  return (
    <div className="card h-fit overflow-hidden">
      <div className="flex flex-wrap items-start gap-3 border-b px-5 py-4">
        <div className="min-w-0 flex-1">
          <h3 className="text-[14px] font-semibold tracking-[-0.01em]">Explicit grants</h3>
          <p className="mt-0.5 text-[13px] leading-relaxed text-muted">Allow raises access; deny caps it below the chosen privilege (deny read removes access entirely).</p>
        </div>
        {!adding ? (
          <Button size="sm" onClick={() => setAdding(true)}>
            <Plus className="h-3.5 w-3.5" /> Add grant
          </Button>
        ) : null}
      </div>
      {adding ? (
        <AddGrantForm
          dir={dir}
          resourceType={resourceType}
          resourceId={resourceId}
          onCancel={() => setAdding(false)}
          onCreated={() => {
            setAdding(false);
            invalidatePaths(qc, "/grants", ...invalidate);
          }}
        />
      ) : null}
      {grants.isLoading ? (
        <div className="space-y-2 p-5">
          <Skeleton className="h-10" />
          <Skeleton className="h-10" />
        </div>
      ) : !grants.data?.length ? (
        <EmptyState icon={<ShieldCheck className="h-5 w-5" />} title="No explicit grants" description="Access currently comes only from visibility, ownership and roles. Add a grant to share with a person, team, department, role or workspace." className="py-10" />
      ) : (
        <ul className="divide-y">
          {grants.data.map((g) => (
            <GrantRow key={g.id} g={g} dir={dir} onRemove={() => setRemoving(g)} />
          ))}
        </ul>
      )}
      <ConfirmDialog
        open={Boolean(removing)}
        onOpenChange={(v) => !v && setRemoving(null)}
        title="Remove grant?"
        description={removing ? <>“{dir.label(removing.subject_type, removing.subject_id)}” loses the {removing.effect} {removing.privilege} grant on {resourceName}. Access from visibility or roles is unaffected.</> : null}
        confirmLabel="Remove grant"
        loading={remove.isPending}
        onConfirm={() => {
          if (removing) remove.mutate(removing);
        }}
      />
    </div>
  );
}

function AddGrantForm({ dir, resourceType, resourceId, onCancel, onCreated }: { dir: Directory; resourceType: string; resourceId: string; onCancel: () => void; onCreated: () => void }) {
  const [subjectType, setSubjectType] = useState<SubjectType>("user");
  const [subjectId, setSubjectId] = useState("");
  const [filter, setFilter] = useState("");
  const [privilege, setPrivilege] = useState<Privilege>("read");
  const [effect, setEffect] = useState<Effect>("allow");
  const [expires, setExpires] = useState("");
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);

  const choices = useMemo(() => {
    const f = filter.trim().toLowerCase();
    let list: { id: string; label: string }[] = [];
    if (subjectType === "user") list = [...dir.members.values()].map((m) => ({ id: m.user_id, label: m.full_name ? `${m.full_name} · ${m.email}` : m.email }));
    else if (subjectType === "team") list = dir.teams.map((t) => ({ id: t.id, label: t.name }));
    else if (subjectType === "department") list = dir.departments.map((d) => ({ id: d.id, label: d.name }));
    else if (subjectType === "role") list = dir.roles.map((r) => ({ id: r.id, label: r.name }));
    else if (subjectType === "workspace") list = dir.workspaces.map((w) => ({ id: w.id, label: w.name }));
    return f ? list.filter((x) => x.label.toLowerCase().includes(f)) : list;
  }, [subjectType, filter, dir]);

  const create = useMutation({
    mutationFn: () =>
      post<Grant>("/grants", {
        resource_type: resourceType,
        resource_id: resourceId,
        subject_type: subjectType,
        subject_id: subjectType === "org" ? null : subjectId,
        privilege,
        effect,
        expires_at: expires ? new Date(expires).toISOString() : null,
        note: note.trim(),
      }),
    onSuccess: () => {
      toast.success(`${effect === "deny" ? "Deny" : "Grant"} added for ${dir.label(subjectType, subjectType === "org" ? null : subjectId)}`);
      onCreated();
    },
  });

  const submit = () => {
    if (subjectType !== "org" && !subjectId) return setError("Choose who this grant applies to.");
    if (expires && new Date(expires).getTime() <= Date.now()) return setError("Expiry must be in the future.");
    setError(null);
    create.mutate();
  };

  return (
    <form
      className="space-y-4 border-b bg-surface-2/40 px-5 py-5 animate-fade-in"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <div className="grid gap-3 md:grid-cols-[200px_1fr]">
        <Field label="Subject type" htmlFor="grant-type">
          <Select
            id="grant-type"
            value={subjectType}
            onChange={(e) => {
              setSubjectType(e.target.value as SubjectType);
              setSubjectId("");
              setFilter("");
            }}
          >
            {SUBJECTS.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </Select>
        </Field>
        {subjectType === "org" ? (
          <div className="flex items-end pb-2 text-[13px] text-muted">Applies to every member of the organization.</div>
        ) : (
          <div className="grid gap-2 sm:grid-cols-[minmax(0,180px)_1fr]">
            <Field label="Filter" htmlFor="grant-filter">
              <Input id="grant-filter" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder={`Search ${subjectType}s`} />
            </Field>
            <Field label={SUBJECTS.find((s) => s.value === subjectType)?.label} htmlFor="grant-subject" hint={!choices.length ? (filter ? "No matches." : `No ${subjectType}s available to you.`) : undefined}>
              <Select id="grant-subject" value={subjectId} onChange={(e) => setSubjectId(e.target.value)}>
                <option value="">Select…</option>
                {choices.slice(0, 300).map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.label}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
        )}
      </div>
      <div className="flex flex-wrap items-end gap-4">
        <div className="space-y-1.5">
          <span className="text-[13px] font-medium">Privilege</span>
          <div>
            <Segmented label="Privilege" value={privilege} onChange={setPrivilege} options={[{ value: "read", label: "Read" }, { value: "write", label: "Write" }, { value: "manage", label: "Manage" }]} />
          </div>
        </div>
        <div className="space-y-1.5">
          <span className="text-[13px] font-medium">Effect</span>
          <div>
            <Segmented label="Effect" value={effect} onChange={setEffect} options={[{ value: "allow", label: "Allow" }, { value: "deny", label: "Deny" }]} />
          </div>
        </div>
        <Field label="Expires (optional)" htmlFor="grant-exp">
          <Input id="grant-exp" type="datetime-local" value={expires} onChange={(e) => setExpires(e.target.value)} className="w-[210px]" />
        </Field>
        <Field label="Note (optional)" htmlFor="grant-note" className="min-w-[160px] flex-1">
          <Input id="grant-note" value={note} onChange={(e) => setNote(e.target.value)} maxLength={1000} placeholder="Why this access exists" />
        </Field>
      </div>
      <p className="rounded-lg border bg-surface px-3 py-2 text-[12.5px] text-muted">
        {effect === "allow"
          ? { read: "Search, chat with and view documents.", write: "Read, plus upload, edit and reprocess — where their role allows it.", manage: "Write, plus settings, permissions and deletion — where their role allows it." }[privilege]
          : { read: "Removes all access, even if granted elsewhere.", write: "Limits access to read-only.", manage: "Prevents managing; read and write still apply." }[privilege]}
      </p>
      {error ? <p className="text-[12px] text-danger" role="alert">{error}</p> : null}
      <div className="flex justify-end gap-2">
        <Button type="button" variant="secondary" size="sm" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit" size="sm" loading={create.isPending}>
          Add {effect === "deny" ? "deny rule" : "grant"}
        </Button>
      </div>
    </form>
  );
}
