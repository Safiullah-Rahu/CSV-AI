"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Building2, FolderKanban, Plus, Trash2, UserRound, Users } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { canAssignRole, memberLabel, useIndex, useMembers, useRoles, useTeams, useWorkspaces } from "@/components/org/data";
import { CancelButton, GatedButton, PersonCell, Segmented, When } from "@/components/org/kit";
import type { RoleAssignment } from "@/components/org/types";
import { Button } from "@/components/ui/button";
import { type Column, DataTable } from "@/components/ui/data-table";
import { ConfirmDialog, Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Select } from "@/components/ui/form";
import { Badge, EmptyState, Tip } from "@/components/ui/primitives";
import { del, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { formatDate } from "@/lib/utils";

export function AssignmentsTab() {
  const { can, org, user } = useAuth();
  const qc = useQueryClient();
  const list = useApi<RoleAssignment[]>("/role-assignments");
  const roles = useRoles();
  const members = useMembers();
  const teams = useTeams();
  const workspaces = useWorkspaces();
  const roleById = useIndex(roles.data, (r) => r.id);
  const memberById = useIndex(members.data?.items, (m) => m.user_id);
  const teamById = useIndex(teams.data, (t) => t.id);
  const wsById = useIndex(workspaces.data, (w) => w.id);
  const [kind, setKind] = useState<"all" | "user" | "team">("all");
  const [assigning, setAssigning] = useState(false);
  const [removing, setRemoving] = useState<RoleAssignment | null>(null);
  const manage = can("admin.permissions");

  const remove = useMutation({
    mutationFn: (a: RoleAssignment) => del(`/role-assignments/${a.id}`),
    onSuccess: () => {
      toast.success("Role assignment removed");
      setRemoving(null);
      qc.invalidateQueries({ queryKey: ["/role-assignments"] });
      qc.invalidateQueries({ queryKey: ["/roles"] });
      qc.invalidateQueries({ queryKey: ["/members"] });
      qc.invalidateQueries({ queryKey: ["org"] });
    },
  });

  const rows = useMemo(() => (list.data ?? []).filter((a) => kind === "all" || (kind === "user" ? a.user_id : a.team_id)), [list.data, kind]);
  const subjectName = (a: RoleAssignment) => (a.user_id ? memberLabel(memberById.get(a.user_id), "Former member") : teamById.get(a.team_id || "")?.name ?? "Deleted team");

  const columns: Column<RoleAssignment>[] = [
    {
      key: "subject",
      header: "Assigned to",
      sortValue: (a) => subjectName(a).toLowerCase(),
      className: "min-w-[220px]",
      cell: (a) => {
        if (a.user_id) {
          const m = memberById.get(a.user_id);
          return <PersonCell name={m?.full_name} email={m?.email ?? "Former member"} you={a.user_id === user?.id} />;
        }
        const t = teamById.get(a.team_id || "");
        return (
          <div className="flex items-center gap-2.5">
            <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full border bg-surface-2 text-muted">
              <Users className="h-3.5 w-3.5" />
            </span>
            <div className="min-w-0 leading-tight">
              <div className="truncate font-medium">{t?.name ?? "Deleted team"}</div>
              <div className="mt-0.5 text-[12px] text-subtle">Team · {t ? `${t.member_ids.length} members` : "—"}</div>
            </div>
          </div>
        );
      },
    },
    {
      key: "role",
      header: "Role",
      sortValue: (a) => roleById.get(a.role_id)?.rank ?? 0,
      cell: (a) => {
        const r = roleById.get(a.role_id);
        return (
          <div className="flex items-center gap-1.5">
            <Badge tone={r?.key === "owner" ? "accent" : "neutral"}>{r?.name ?? "Unknown role"}</Badge>
            {r ? <span className="num font-mono text-[11px] text-subtle">rank {r.rank}</span> : null}
          </div>
        );
      },
    },
    {
      key: "scope",
      header: "Scope",
      sortValue: (a) => a.scope_type,
      cell: (a) =>
        a.scope_type === "workspace" ? (
          <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
            <FolderKanban className="h-3.5 w-3.5 text-subtle" /> {wsById.get(a.scope_id || "")?.name ?? "Workspace"}
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-muted">
            <Building2 className="h-3.5 w-3.5 text-subtle" /> Organization
          </span>
        ),
    },
    {
      key: "expires",
      header: "Expires",
      sortValue: (a) => a.expires_at || "9999",
      cell: (a) =>
        !a.expires_at ? (
          <span className="text-subtle">Never</span>
        ) : new Date(a.expires_at).getTime() < Date.now() ? (
          <Badge tone="danger">expired {formatDate(a.expires_at, false)}</Badge>
        ) : (
          <span title={formatDate(a.expires_at)} className="whitespace-nowrap text-muted">
            {formatDate(a.expires_at)}
          </span>
        ),
    },
    { key: "granted", header: "Granted", sortValue: (a) => a.created_at, cell: (a) => <When iso={a.created_at} className="text-muted" /> },
    {
      key: "actions",
      header: <span className="relative block"><span className="sr-only">Actions</span></span>,
      className: "text-right",
      cell: (a) => {
        if (!manage) return null;
        const r = roleById.get(a.role_id);
        const allowed = r ? canAssignRole(r, org) : false;
        return allowed ? (
          <Button variant="ghost" size="icon" className="h-7 w-7" aria-label={`Remove ${r?.name} from ${subjectName(a)}`} onClick={() => setRemoving(a)}>
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        ) : (
          <Tip content="This role ranks above yours, so you cannot remove it.">
            <span tabIndex={0} className="inline-flex">
              <Button variant="ghost" size="icon" className="h-7 w-7" disabled aria-label="Cannot remove">
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </span>
          </Tip>
        );
      },
    },
  ];

  return (
    <>
      <DataTable
        rows={rows}
        columns={columns}
        rowKey={(a) => a.id}
        loading={list.isLoading}
        filterText={(a) => `${subjectName(a)} ${memberById.get(a.user_id || "")?.email ?? ""} ${roleById.get(a.role_id)?.name ?? ""}`}
        searchPlaceholder="Search people, teams, roles…"
        toolbar={
          <>
            <Segmented
              label="Subject filter"
              value={kind}
              onChange={setKind}
              options={[
                { value: "all", label: "All" },
                { value: "user", label: "People" },
                { value: "team", label: "Teams" },
              ]}
            />
            <GatedButton size="sm" perm="admin.permissions" onClick={() => setAssigning(true)}>
              <Plus className="h-3.5 w-3.5" /> Assign role
            </GatedButton>
          </>
        }
        empty={<EmptyState title="No role assignments match" />}
      />
      <AssignRoleDialog open={assigning} onOpenChange={setAssigning} />
      <ConfirmDialog
        open={Boolean(removing)}
        onOpenChange={(v) => !v && setRemoving(null)}
        title="Remove role assignment?"
        description={removing ? `${subjectName(removing)} will lose the “${roleById.get(removing.role_id)?.name ?? "role"}” role${removing.scope_type === "workspace" ? ` in ${wsById.get(removing.scope_id || "")?.name ?? "the workspace"}` : ""}. Permissions update immediately.` : undefined}
        confirmLabel="Remove assignment"
        loading={remove.isPending}
        onConfirm={() => {
          if (removing) remove.mutate(removing);
        }}
      />
    </>
  );
}

function AssignRoleDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  const { org } = useAuth();
  const qc = useQueryClient();
  const roles = useRoles(open);
  const members = useMembers(open);
  const teams = useTeams(open);
  const workspaces = useWorkspaces(open);
  const assignable = useMemo(() => (roles.data ?? []).filter((r) => canAssignRole(r, org)), [roles.data, org]);
  const [subjectType, setSubjectType] = useState<"user" | "team">("user");
  const [subjectId, setSubjectId] = useState("");
  const [roleId, setRoleId] = useState("");
  const [scope, setScope] = useState<"org" | "workspace">("org");
  const [wsId, setWsId] = useState("");
  const [expires, setExpires] = useState("");

  useEffect(() => {
    if (open) {
      setSubjectType("user");
      setSubjectId("");
      setRoleId("");
      setScope("org");
      setWsId("");
      setExpires("");
    }
  }, [open]);

  const m = useMutation({
    mutationFn: () =>
      post<RoleAssignment>("/role-assignments", {
        role_id: roleId,
        user_id: subjectType === "user" ? subjectId : null,
        team_id: subjectType === "team" ? subjectId : null,
        workspace_id: scope === "workspace" ? wsId : null,
        expires_at: expires ? new Date(expires).toISOString() : null,
      }),
    onSuccess: () => {
      toast.success("Role assigned");
      qc.invalidateQueries({ queryKey: ["/role-assignments"] });
      qc.invalidateQueries({ queryKey: ["/roles"] });
      qc.invalidateQueries({ queryKey: ["/members"] });
      qc.invalidateQueries({ queryKey: ["org"] });
      onOpenChange(false);
    },
  });
  const expiresInPast = expires ? new Date(expires).getTime() <= Date.now() : false;
  const valid = subjectId && roleId && (scope === "org" || wsId) && !expiresInPast;
  const role = assignable.find((r) => r.id === roleId);
  const minLocal = useMemo(() => {
    const d = new Date(Date.now() + 5 * 60_000);
    d.setSeconds(0, 0);
    return new Date(d.getTime() - d.getTimezoneOffset() * 60_000).toISOString().slice(0, 16);
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        title="Assign a role"
        description="Grant a role to a person or a whole team, across the organization or inside one workspace."
        footer={
          <>
            <CancelButton />
            <Button type="submit" form="assign-form" loading={m.isPending} disabled={!valid}>
              Assign role
            </Button>
          </>
        }
      >
        <form
          id="assign-form"
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (valid) m.mutate();
          }}
        >
          <div className="space-y-2">
            <Segmented
              label="Assign to"
              value={subjectType}
              onChange={(v) => {
                setSubjectType(v);
                setSubjectId("");
              }}
              options={[
                { value: "user", label: <><UserRound className="h-3.5 w-3.5" /> Person</> },
                { value: "team", label: <><Users className="h-3.5 w-3.5" /> Team</> },
              ]}
            />
            {subjectType === "user" ? (
              <Select aria-label="Member" value={subjectId} onChange={(e) => setSubjectId(e.target.value)}>
                <option value="">Choose a member…</option>
                {(members.data?.items ?? []).map((mm) => (
                  <option key={mm.user_id} value={mm.user_id}>
                    {mm.full_name ? `${mm.full_name} (${mm.email})` : mm.email}
                  </option>
                ))}
              </Select>
            ) : (
              <Select aria-label="Team" value={subjectId} onChange={(e) => setSubjectId(e.target.value)}>
                <option value="">Choose a team…</option>
                {(teams.data ?? []).map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.name} · {t.member_ids.length} members
                  </option>
                ))}
              </Select>
            )}
          </div>
          <Field label="Role" htmlFor="assign-role" hint={role ? role.description : "Only roles at or below your rank are listed."}>
            <Select id="assign-role" value={roleId} onChange={(e) => setRoleId(e.target.value)}>
              <option value="">Choose a role…</option>
              {assignable.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name} · rank {r.rank}
                  {r.is_system ? "" : " · custom"}
                </option>
              ))}
            </Select>
          </Field>
          <div className="space-y-2">
            <div className="text-[13px] font-medium">Scope</div>
            <Segmented
              label="Scope"
              value={scope}
              onChange={setScope}
              options={[
                { value: "org", label: <><Building2 className="h-3.5 w-3.5" /> Whole organization</> },
                { value: "workspace", label: <><FolderKanban className="h-3.5 w-3.5" /> One workspace</> },
              ]}
            />
            {scope === "workspace" ? (
              <Select aria-label="Workspace" value={wsId} onChange={(e) => setWsId(e.target.value)} disabled={!workspaces.data}>
                <option value="">{workspaces.data ? "Choose a workspace…" : "You cannot view workspaces"}</option>
                {(workspaces.data ?? []).map((w) => (
                  <option key={w.id} value={w.id}>
                    {w.name}
                  </option>
                ))}
              </Select>
            ) : null}
          </div>
          <Field label="Expires (optional)" htmlFor="assign-expires" error={expiresInPast ? "Expiry must be in the future." : null} hint="Leave empty for a permanent assignment. Expired assignments stop granting permissions automatically.">
            <Input id="assign-expires" type="datetime-local" min={minLocal} value={expires} onChange={(e) => setExpires(e.target.value)} />
          </Field>
        </form>
      </DialogContent>
    </Dialog>
  );
}
