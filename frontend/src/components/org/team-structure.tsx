"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Building, FolderKanban, Pencil, Plus, Trash2, UserMinus, UserPlus, Users } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { memberLabel, useDepartments, useIndex, useMembers, useTeams, useWorkspaceCan, useWorkspaces } from "@/components/org/data";
import { CancelButton, Forbidden, GatedButton, MemberPicker, permissionHint, PersonCell, RowMenuButton, When } from "@/components/org/kit";
import type { Department, Team, Workspace } from "@/components/org/types";
import { Button } from "@/components/ui/button";
import { type Column, DataTable } from "@/components/ui/data-table";
import { ConfirmDialog, Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Select, Textarea } from "@/components/ui/form";
import { Avatar, Badge, EmptyState, Menu, MenuContent, MenuItem, MenuSeparator, MenuTrigger, Skeleton } from "@/components/ui/primitives";
import { del, patch, post, put } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { pluralize } from "@/lib/utils";

/* ------------------------------------------------------------ shared dialog */
function NamedDialog({ open, onOpenChange, title, description, initial, saving, onSubmit, children, submitLabel = "Save" }: { open: boolean; onOpenChange: (v: boolean) => void; title: string; description?: string; initial?: { name: string; description: string }; saving: boolean; onSubmit: (v: { name: string; description: string }) => void; children?: React.ReactNode; submitLabel?: string }) {
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  useEffect(() => {
    if (open) {
      setName(initial?.name ?? "");
      setDesc(initial?.description ?? "");
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps -- reset only when (re)opened
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        title={title}
        description={description}
        footer={
          <>
            <CancelButton />
            <Button type="submit" form="named-form" loading={saving} disabled={!name.trim()}>
              {submitLabel}
            </Button>
          </>
        }
      >
        <form
          id="named-form"
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (name.trim()) onSubmit({ name: name.trim(), description: desc });
          }}
        >
          <Field label="Name" htmlFor="named-name">
            <Input id="named-name" value={name} onChange={(e) => setName(e.target.value)} maxLength={120} autoFocus />
          </Field>
          <Field label="Description" htmlFor="named-desc">
            <Textarea id="named-desc" value={desc} onChange={(e) => setDesc(e.target.value)} maxLength={2000} rows={3} />
          </Field>
          {children}
        </form>
      </DialogContent>
    </Dialog>
  );
}

function AvatarStack({ ids, lookup }: { ids: string[]; lookup: (id: string) => { full_name: string; email: string } | undefined }) {
  const shown = ids.slice(0, 5);
  return (
    <div className="flex items-center">
      <div className="flex -space-x-1.5">
        {shown.map((id) => {
          const m = lookup(id);
          return <Avatar key={id} name={m?.full_name} email={m?.email ?? id} className="h-6 w-6 ring-2 ring-surface text-[9px]" />;
        })}
      </div>
      <span className="num ml-2.5 whitespace-nowrap text-[12.5px] text-muted">{ids.length ? pluralize(ids.length, "member") : "No members"}</span>
    </div>
  );
}

function EntityCell({ icon, name, description, badge }: { icon: React.ReactNode; name: string; description?: string | null; badge?: React.ReactNode }) {
  return (
    <div className="flex min-w-0 items-center gap-3">
      <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg border bg-surface-2/60 text-subtle">{icon}</span>
      <div className="min-w-0 leading-tight">
        <div className="flex items-center gap-1.5 font-medium">
          <span className="truncate">{name}</span>
          {badge}
        </div>
        {description ? <div className="mt-0.5 line-clamp-1 text-[12px] text-subtle">{description}</div> : null}
      </div>
    </div>
  );
}

/* --------------------------------------------------------------------- teams */
export function TeamsTab() {
  const { can } = useAuth();
  const qc = useQueryClient();
  const teams = useTeams();
  const members = useMembers();
  const departments = useDepartments();
  const workspaces = useWorkspaces();
  const memberById = useIndex(members.data?.items, (m) => m.user_id);
  const deptById = useIndex(departments.data, (d) => d.id);
  const wsById = useIndex(workspaces.data, (w) => w.id);
  const manage = can("org.users");
  const [editing, setEditing] = useState<Team | "new" | null>(null);
  const [staffing, setStaffing] = useState<Team | null>(null);
  const [deleting, setDeleting] = useState<Team | null>(null);
  const [deptId, setDeptId] = useState("");
  const [wsId, setWsId] = useState("");

  useEffect(() => {
    if (editing && editing !== "new") {
      setDeptId(editing.department_id ?? "");
      setWsId(editing.workspace_id ?? "");
    } else if (editing === "new") {
      setDeptId("");
      setWsId("");
    }
  }, [editing]);

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["/teams"] });
    qc.invalidateQueries({ queryKey: ["/members"] });
  };
  const save = useMutation({
    mutationFn: (v: { name: string; description: string }) => {
      const body = { ...v, department_id: deptId || null, workspace_id: wsId || null };
      return editing === "new" ? post<Team>("/teams", body) : patch<Team>(`/teams/${(editing as Team).id}`, body);
    },
    onSuccess: (t) => {
      toast.success(editing === "new" ? `Team “${t.name}” created` : "Team updated");
      setEditing(null);
      invalidate();
    },
  });
  const remove = useMutation({
    mutationFn: (t: Team) => del(`/teams/${t.id}`),
    onSuccess: (_r, t) => {
      toast.success(`Team “${t.name}” deleted`);
      setDeleting(null);
      invalidate();
      qc.invalidateQueries({ queryKey: ["/role-assignments"] });
    },
  });

  const columns: Column<Team>[] = [
    {
      key: "name",
      header: "Team",
      sortValue: (t) => t.name.toLowerCase(),
      className: "min-w-[200px]",
      cell: (t) => (
        <EntityCell icon={<Users className="h-4 w-4" />} name={t.name} description={t.description} />
      ),
    },
    { key: "dept", header: "Department", cell: (t) => (t.department_id ? <span className="whitespace-nowrap text-muted">{deptById.get(t.department_id)?.name ?? "—"}</span> : <span className="text-subtle">—</span>) },
    {
      key: "ws",
      header: "Workspace",
      cell: (t) =>
        t.workspace_id ? (
          <Badge className="gap-1">
            <FolderKanban className="h-3 w-3 text-subtle" aria-hidden /> {wsById.get(t.workspace_id)?.name ?? "workspace"}
          </Badge>
        ) : (
          <span className="text-subtle">—</span>
        ),
    },
    { key: "members", header: "Members", sortValue: (t) => t.member_ids.length, cell: (t) => <AvatarStack ids={t.member_ids} lookup={(id) => memberById.get(id)} /> },
    { key: "created", header: "Created", sortValue: (t) => t.created_at, cell: (t) => <When iso={t.created_at} className="text-muted" /> },
    {
      key: "actions",
      header: <span className="relative block"><span className="sr-only">Actions</span></span>,
      className: "w-10 text-right",
      cell: (t) =>
        manage ? (
          <Menu>
            <MenuTrigger asChild>
              <RowMenuButton label={`Actions for ${t.name}`} />
            </MenuTrigger>
            <MenuContent>
              <MenuItem onSelect={() => setStaffing(t)}>
                <Users className="h-3.5 w-3.5" /> Manage members
              </MenuItem>
              <MenuItem onSelect={() => setEditing(t)}>
                <Pencil className="h-3.5 w-3.5" /> Edit team
              </MenuItem>
              <MenuSeparator />
              <MenuItem danger onSelect={() => setDeleting(t)}>
                <Trash2 className="h-3.5 w-3.5" /> Delete team
              </MenuItem>
            </MenuContent>
          </Menu>
        ) : null,
    },
  ];

  return (
    <>
      <DataTable
        rows={teams.data}
        columns={columns}
        rowKey={(t) => t.id}
        loading={teams.isLoading}
        filterText={(t) => `${t.name} ${t.description}`}
        searchPlaceholder="Search teams…"
        toolbar={
          <GatedButton size="sm" variant={teams.data?.length ? "primary" : "secondary"} perm="org.users" onClick={() => setEditing("new")}>
            <Plus className="h-3.5 w-3.5" /> New team
          </GatedButton>
        }
        empty={<EmptyState icon={<Users className="h-5 w-5" />} title="No teams yet" description="Teams group people so you can share knowledge bases, conversations and roles with them at once." action={manage ? <Button onClick={() => setEditing("new")}>Create a team</Button> : undefined} />}
      />
      <NamedDialog
        open={editing !== null}
        onOpenChange={(v) => !v && setEditing(null)}
        title={editing === "new" ? "Create team" : "Edit team"}
        initial={editing && editing !== "new" ? { name: editing.name, description: editing.description } : undefined}
        saving={save.isPending}
        submitLabel={editing === "new" ? "Create team" : "Save"}
        onSubmit={(v) => save.mutate(v)}
      >
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Department" htmlFor="team-dept">
            <Select id="team-dept" value={deptId} onChange={(e) => setDeptId(e.target.value)}>
              <option value="">None</option>
              {(departments.data ?? []).map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Workspace" htmlFor="team-ws">
            <Select id="team-ws" value={wsId} onChange={(e) => setWsId(e.target.value)} disabled={!workspaces.data}>
              <option value="">None</option>
              {(workspaces.data ?? []).map((w) => (
                <option key={w.id} value={w.id}>
                  {w.name}
                </option>
              ))}
            </Select>
          </Field>
        </div>
      </NamedDialog>
      <TeamMembersDialog team={staffing} onClose={() => setStaffing(null)} onSaved={invalidate} />
      <ConfirmDialog
        open={Boolean(deleting)}
        onOpenChange={(v) => !v && setDeleting(null)}
        title="Delete team?"
        description={deleting ? `“${deleting.name}” will be deleted. Members stay in the organization, but lose any roles, grants and shared conversations that were given to the team.` : undefined}
        confirmLabel="Delete team"
        loading={remove.isPending}
        onConfirm={() => {
          if (deleting) remove.mutate(deleting);
        }}
      />
    </>
  );
}

function TeamMembersDialog({ team, onClose, onSaved }: { team: Team | null; onClose: () => void; onSaved: () => void }) {
  const members = useMembers(Boolean(team));
  const [ids, setIds] = useState<string[]>([]);
  useEffect(() => {
    if (team) setIds(team.member_ids);
  }, [team]);
  const m = useMutation({
    mutationFn: () => put(`/teams/${team!.id}/members`, { user_ids: ids }),
    onSuccess: () => {
      toast.success(`Members of “${team!.name}” updated`);
      onSaved();
      onClose();
    },
  });
  const added = ids.filter((id) => !team?.member_ids.includes(id)).length;
  const removed = (team?.member_ids ?? []).filter((id) => !ids.includes(id)).length;
  return (
    <Dialog open={Boolean(team)} onOpenChange={(v) => !v && onClose()}>
      <DialogContent
        title={`Team members · ${team?.name ?? ""}`}
        description="Check everyone who should belong to this team."
        footer={
          <>
            <span className="mr-auto text-xs text-muted">{added || removed ? `${added} added · ${removed} removed` : "No changes"}</span>
            <CancelButton />
            <Button loading={m.isPending} disabled={!added && !removed} onClick={() => m.mutate()}>
              Save members
            </Button>
          </>
        }
      >
        {members.isLoading ? <Skeleton className="h-60" /> : <MemberPicker members={members.data?.items ?? []} value={ids} onChange={setIds} />}
      </DialogContent>
    </Dialog>
  );
}

/* --------------------------------------------------------------- departments */
export function DepartmentsTab() {
  const { can } = useAuth();
  const qc = useQueryClient();
  const departments = useDepartments();
  const manage = can("org.users");
  const [editing, setEditing] = useState<Department | "new" | null>(null);
  const [deleting, setDeleting] = useState<Department | null>(null);
  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["/departments"] });
    qc.invalidateQueries({ queryKey: ["/members"] });
  };
  const save = useMutation({
    mutationFn: (v: { name: string; description: string }) => (editing === "new" ? post<Department>("/departments", v) : patch<Department>(`/departments/${(editing as Department).id}`, v)),
    onSuccess: (d) => {
      toast.success(editing === "new" ? `Department “${d.name}” created` : "Department updated");
      setEditing(null);
      invalidate();
    },
  });
  const remove = useMutation({
    mutationFn: (d: Department) => del(`/departments/${d.id}`),
    onSuccess: (_r, d) => {
      toast.success(`Department “${d.name}” deleted`);
      setDeleting(null);
      invalidate();
    },
  });
  const columns: Column<Department>[] = [
    {
      key: "name",
      header: "Department",
      sortValue: (d) => d.name.toLowerCase(),
      cell: (d) => (
        <EntityCell icon={<Building className="h-4 w-4" />} name={d.name} description={d.description} />
      ),
    },
    { key: "members", header: "Members", sortValue: (d) => d.member_count, cell: (d) => <span className="num whitespace-nowrap text-muted">{pluralize(d.member_count, "member")}</span> },
    { key: "created", header: "Created", sortValue: (d) => d.created_at, cell: (d) => <When iso={d.created_at} className="text-muted" /> },
    {
      key: "actions",
      header: <span className="relative block"><span className="sr-only">Actions</span></span>,
      className: "w-10 text-right",
      cell: (d) =>
        manage ? (
          <Menu>
            <MenuTrigger asChild>
              <RowMenuButton label={`Actions for ${d.name}`} />
            </MenuTrigger>
            <MenuContent>
              <MenuItem onSelect={() => setEditing(d)}>
                <Pencil className="h-3.5 w-3.5" /> Edit
              </MenuItem>
              <MenuItem danger onSelect={() => setDeleting(d)}>
                <Trash2 className="h-3.5 w-3.5" /> Delete
              </MenuItem>
            </MenuContent>
          </Menu>
        ) : null,
    },
  ];
  return (
    <>
      <DataTable
        rows={departments.data}
        columns={columns}
        rowKey={(d) => d.id}
        loading={departments.isLoading}
        filterText={(d) => `${d.name} ${d.description}`}
        searchPlaceholder="Search departments…"
        toolbar={
          <GatedButton size="sm" variant={departments.data?.length ? "primary" : "secondary"} perm="org.users" onClick={() => setEditing("new")}>
            <Plus className="h-3.5 w-3.5" /> New department
          </GatedButton>
        }
        empty={<EmptyState icon={<Building className="h-5 w-5" />} title="No departments yet" description="Departments describe your org chart. They can be used as a subject when granting access to knowledge." action={manage ? <Button onClick={() => setEditing("new")}>Create a department</Button> : undefined} />}
      />
      <NamedDialog
        open={editing !== null}
        onOpenChange={(v) => !v && setEditing(null)}
        title={editing === "new" ? "Create department" : "Edit department"}
        initial={editing && editing !== "new" ? { name: editing.name, description: editing.description } : undefined}
        saving={save.isPending}
        submitLabel={editing === "new" ? "Create department" : "Save"}
        onSubmit={(v) => save.mutate(v)}
      />
      <ConfirmDialog
        open={Boolean(deleting)}
        onOpenChange={(v) => !v && setDeleting(null)}
        title="Delete department?"
        description={deleting ? `“${deleting.name}” will be deleted and its ${pluralize(deleting.member_count, "member")} left without a department. Grants given to the department stop applying.` : undefined}
        confirmLabel="Delete department"
        loading={remove.isPending}
        onConfirm={() => {
          if (deleting) remove.mutate(deleting);
        }}
      />
    </>
  );
}

/* ---------------------------------------------------------------- workspaces */
export function WorkspacesTab() {
  const { can } = useAuth();
  const wsCan = useWorkspaceCan();
  const qc = useQueryClient();
  const workspaces = useWorkspaces();
  const [editing, setEditing] = useState<Workspace | "new" | null>(null);
  const [staffing, setStaffing] = useState<Workspace | null>(null);
  const [deleting, setDeleting] = useState<Workspace | null>(null);
  const invalidate = () => qc.invalidateQueries({ queryKey: ["/workspaces"] });
  const save = useMutation({
    mutationFn: (v: { name: string; description: string }) => (editing === "new" ? post<Workspace>("/workspaces", v) : patch<Workspace>(`/workspaces/${(editing as Workspace).id}`, v)),
    onSuccess: (w) => {
      toast.success(editing === "new" ? `Workspace “${w.name}” created — you were added as a member` : "Workspace updated");
      setEditing(null);
      invalidate();
    },
  });
  const remove = useMutation({
    mutationFn: (w: Workspace) => del(`/workspaces/${w.id}`),
    onSuccess: (_r, w) => {
      toast.success(`Workspace “${w.name}” deleted`);
      setDeleting(null);
      invalidate();
    },
  });

  if (!can("workspace.view")) return <Forbidden perm="workspace.view" />;

  const columns: Column<Workspace>[] = [
    {
      key: "name",
      header: "Workspace",
      sortValue: (w) => w.name.toLowerCase(),
      className: "min-w-[220px]",
      cell: (w) => (
        <EntityCell icon={<FolderKanban className="h-4 w-4" />} name={w.name} description={w.description} badge={w.is_member ? <Badge tone="accent">member</Badge> : null} />
      ),
    },
    { key: "members", header: "Members", sortValue: (w) => w.member_count, cell: (w) => <span className="num whitespace-nowrap text-muted">{pluralize(w.member_count, "member")}</span> },
    { key: "kbs", header: "Knowledge bases", sortValue: (w) => w.kb_count, cell: (w) => <span className="num font-mono text-[12.5px] text-muted">{w.kb_count}</span> },
    { key: "created", header: "Created", sortValue: (w) => w.created_at, cell: (w) => <When iso={w.created_at} className="text-muted" /> },
    {
      key: "actions",
      header: <span className="relative block"><span className="sr-only">Actions</span></span>,
      className: "w-10 text-right",
      cell: (w) => (
        <Menu>
          <MenuTrigger asChild>
            <RowMenuButton label={`Actions for ${w.name}`} />
          </MenuTrigger>
          <MenuContent className="w-56">
            <MenuItem onSelect={() => setStaffing(w)}>
              <Users className="h-3.5 w-3.5" /> {wsCan("workspace.members", w.id) ? "Manage members" : "View members"}
            </MenuItem>
            <MenuItem disabled={!wsCan("workspace.edit", w.id)} onSelect={() => setEditing(w)}>
              <Pencil className="h-3.5 w-3.5" /> Edit workspace
              {!wsCan("workspace.edit", w.id) ? <span className="ml-auto font-mono text-[10px] text-subtle">workspace.edit</span> : null}
            </MenuItem>
            <MenuSeparator />
            <MenuItem danger disabled={!wsCan("workspace.delete", w.id) || w.kb_count > 0} onSelect={() => setDeleting(w)}>
              <Trash2 className="h-3.5 w-3.5" /> Delete workspace
              {w.kb_count > 0 ? <span className="ml-auto text-[10px] text-subtle">has KBs</span> : !wsCan("workspace.delete", w.id) ? <span className="ml-auto font-mono text-[10px] text-subtle">workspace.delete</span> : null}
            </MenuItem>
          </MenuContent>
        </Menu>
      ),
    },
  ];
  return (
    <>
      <DataTable
        rows={workspaces.data}
        columns={columns}
        rowKey={(w) => w.id}
        loading={workspaces.isLoading}
        filterText={(w) => `${w.name} ${w.description}`}
        searchPlaceholder="Search workspaces…"
        toolbar={
          <GatedButton size="sm" variant={workspaces.data?.length ? "primary" : "secondary"} perm="workspace.create" onClick={() => setEditing("new")}>
            <Plus className="h-3.5 w-3.5" /> New workspace
          </GatedButton>
        }
        empty={<EmptyState icon={<FolderKanban className="h-5 w-5" />} title="No workspaces yet" description="Workspaces partition knowledge bases and let you assign roles that only apply inside them." action={can("workspace.create") ? <Button onClick={() => setEditing("new")}>Create a workspace</Button> : undefined} />}
      />
      <NamedDialog
        open={editing !== null}
        onOpenChange={(v) => !v && setEditing(null)}
        title={editing === "new" ? "Create workspace" : "Edit workspace"}
        description={editing === "new" ? "You will be added as the first member." : undefined}
        initial={editing && editing !== "new" ? { name: editing.name, description: editing.description } : undefined}
        saving={save.isPending}
        submitLabel={editing === "new" ? "Create workspace" : "Save"}
        onSubmit={(v) => save.mutate(v)}
      />
      <WorkspaceMembersDialog workspace={staffing} onClose={() => setStaffing(null)} />
      <ConfirmDialog
        open={Boolean(deleting)}
        onOpenChange={(v) => !v && setDeleting(null)}
        title="Delete workspace?"
        description={deleting ? `“${deleting.name}” and its memberships and workspace-scoped role assignments will be deleted.` : undefined}
        confirmLabel="Delete workspace"
        requireText={deleting?.name}
        loading={remove.isPending}
        onConfirm={() => {
          if (deleting) remove.mutate(deleting);
        }}
      />
    </>
  );
}

function WorkspaceMembersDialog({ workspace, onClose }: { workspace: Workspace | null; onClose: () => void }) {
  const qc = useQueryClient();
  const wsCan = useWorkspaceCan();
  const ids = useApi<string[]>(workspace ? `/workspaces/${workspace.id}/members` : null);
  const members = useMembers(Boolean(workspace));
  const memberById = useIndex(members.data?.items, (m) => m.user_id);
  const manage = workspace ? wsCan("workspace.members", workspace.id) : false;
  const [adding, setAdding] = useState<string[]>([]);
  const [picker, setPicker] = useState(false);
  const [removing, setRemoving] = useState<string | null>(null);
  const current = useMemo(() => new Set(ids.data ?? []), [ids.data]);

  useEffect(() => {
    if (!workspace) {
      setAdding([]);
      setPicker(false);
    }
  }, [workspace]);

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: [`/workspaces/${workspace?.id}/members`] });
    qc.invalidateQueries({ queryKey: ["/workspaces"] });
  };
  const add = useMutation({
    mutationFn: () => post(`/workspaces/${workspace!.id}/members`, { user_ids: adding }),
    onSuccess: () => {
      toast.success(`${pluralize(adding.length, "member")} added`);
      setAdding([]);
      setPicker(false);
      invalidate();
    },
  });
  const remove = useMutation({
    mutationFn: (uid: string) => del(`/workspaces/${workspace!.id}/members/${uid}`),
    onSuccess: () => {
      toast.success("Member removed from workspace");
      setRemoving(null);
      invalidate();
    },
  });

  return (
    <>
      <Dialog open={Boolean(workspace)} onOpenChange={(v) => !v && onClose()}>
        <DialogContent
          title={`Workspace members · ${workspace?.name ?? ""}`}
          description="Members can see the workspace's knowledge bases according to their roles and grants."
          footer={
            picker ? (
              <>
                <Button variant="secondary" onClick={() => setPicker(false)}>
                  Back
                </Button>
                <Button loading={add.isPending} disabled={!adding.length} onClick={() => add.mutate()}>
                  Add {adding.length ? pluralize(adding.length, "member") : "members"}
                </Button>
              </>
            ) : (
              <>
                {manage ? (
                  <Button variant="secondary" className="mr-auto" onClick={() => setPicker(true)}>
                    <UserPlus className="h-3.5 w-3.5" /> Add members
                  </Button>
                ) : null}
                <CancelButton label="Close" />
              </>
            )
          }
        >
          {ids.isLoading || members.isLoading ? (
            <Skeleton className="h-48" />
          ) : picker ? (
            <MemberPicker members={(members.data?.items ?? []).filter((m) => m.status === "active")} value={adding} onChange={setAdding} exclude={current} />
          ) : !ids.data?.length ? (
            <EmptyState title="No members" description="Add people to give them access to this workspace." className="py-8" />
          ) : (
            <ul className="divide-y rounded-xl border">
              {ids.data.map((uid) => {
                const m = memberById.get(uid);
                return (
                  <li key={uid} className="flex items-center gap-3 px-3 py-2">
                    <PersonCell name={m?.full_name} email={m?.email ?? "Former member"} />
                    {manage ? (
                      <Button variant="ghost" size="icon" className="ml-auto h-7 w-7" aria-label={`Remove ${memberLabel(m)}`} onClick={() => setRemoving(uid)}>
                        <UserMinus className="h-3.5 w-3.5" />
                      </Button>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          )}
          {!manage && workspace ? <p className="mt-3 text-xs text-subtle">{permissionHint("workspace.members")} to add or remove people.</p> : null}
        </DialogContent>
      </Dialog>
      <ConfirmDialog
        open={Boolean(removing)}
        onOpenChange={(v) => !v && setRemoving(null)}
        title="Remove from workspace?"
        description={removing ? `${memberLabel(memberById.get(removing))} will lose access granted through membership of “${workspace?.name}”.` : undefined}
        confirmLabel="Remove"
        loading={remove.isPending}
        onConfirm={() => {
          if (removing) remove.mutate(removing);
        }}
      />
    </>
  );
}
