"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { FolderKanban, MailPlus, Send, Users, XCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { canAssignRole, useIndex, useRoles, useTeams, useWorkspaces } from "@/components/org/data";
import { Callout, CancelButton, Forbidden, GatedButton, permissionHint, PersonCell, When } from "@/components/org/kit";
import type { Invitation } from "@/components/org/types";
import { Button } from "@/components/ui/button";
import { type Column, DataTable } from "@/components/ui/data-table";
import { ConfirmDialog, Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Select } from "@/components/ui/form";
import { Badge, EmptyState, StatusBadge } from "@/components/ui/primitives";
import { del, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";

function effectiveStatus(inv: Invitation) {
  if (inv.status === "pending" && new Date(inv.expires_at).getTime() < Date.now()) return "expired";
  return inv.status;
}

export function InvitationsTab() {
  const { can } = useAuth();
  const qc = useQueryClient();
  const allowed = can("org.users");
  const list = useApi<Invitation[]>(allowed ? "/invitations" : null);
  const roles = useRoles();
  const teams = useTeams();
  const workspaces = useWorkspaces();
  const roleById = useIndex(roles.data, (r) => r.id);
  const teamById = useIndex(teams.data, (t) => t.id);
  const wsById = useIndex(workspaces.data, (w) => w.id);
  const [inviting, setInviting] = useState(false);
  const [revoking, setRevoking] = useState<Invitation | null>(null);

  const revoke = useMutation({
    mutationFn: (inv: Invitation) => del(`/invitations/${inv.id}`),
    onSuccess: (_r, inv) => {
      toast.success(`Invitation for ${inv.email} revoked`);
      setRevoking(null);
      qc.invalidateQueries({ queryKey: ["/invitations"] });
    },
  });

  if (!allowed) return <Forbidden perm="org.users" title="Invitations are managed by organization administrators" />;

  const columns: Column<Invitation>[] = [
    { key: "email", header: "Email", sortValue: (i) => i.email, className: "min-w-[220px]", cell: (i) => <PersonCell email={i.email} /> },
    { key: "role", header: "Role", cell: (i) => (i.role_id ? <Badge>{roleById.get(i.role_id)?.name ?? "Deleted role"}</Badge> : <span className="text-subtle">Member</span>) },
    {
      key: "placement",
      header: "Joins",
      cell: (i) => (
        <div className="flex flex-wrap gap-1">
          {i.team_id ? (
            <Badge className="gap-1">
              <Users className="h-3 w-3 text-subtle" aria-hidden /> {teamById.get(i.team_id)?.name ?? "deleted team"}
            </Badge>
          ) : null}
          {i.workspace_id ? (
            <Badge className="gap-1">
              <FolderKanban className="h-3 w-3 text-subtle" aria-hidden /> {wsById.get(i.workspace_id)?.name ?? "deleted workspace"}
            </Badge>
          ) : null}
          {!i.team_id && !i.workspace_id ? <span className="text-[12.5px] text-subtle">Organization only</span> : null}
        </div>
      ),
    },
    { key: "status", header: "Status", sortValue: (i) => effectiveStatus(i), cell: (i) => <StatusBadge status={effectiveStatus(i)} /> },
    { key: "sent", header: "Sent", sortValue: (i) => i.created_at, cell: (i) => <When iso={i.created_at} className="text-muted" /> },
    { key: "expires", header: "Expires", sortValue: (i) => i.expires_at, cell: (i) => (i.status === "pending" ? <When iso={i.expires_at} className="text-muted" /> : <span className="text-subtle">—</span>) },
    {
      key: "actions",
      header: <span className="relative block"><span className="sr-only">Actions</span></span>,
      className: "text-right",
      cell: (i) =>
        effectiveStatus(i) === "pending" ? (
          <Button variant="ghost" size="xs" onClick={() => setRevoking(i)}>
            <XCircle className="h-3.5 w-3.5" /> Revoke
          </Button>
        ) : null,
    },
  ];

  return (
    <>
      <DataTable
        rows={list.data}
        columns={columns}
        rowKey={(i) => i.id}
        loading={list.isLoading}
        filterText={(i) => `${i.email} ${i.status}`}
        searchPlaceholder="Search invitations…"
        toolbar={
          <GatedButton size="sm" variant={list.data?.length ? "primary" : "secondary"} allowed={can("admin.permissions")} reason={<>Invitations assign a role, which {permissionHint("admin.permissions")}</>} onClick={() => setInviting(true)}>
            <MailPlus className="h-3.5 w-3.5" /> Invite people
          </GatedButton>
        }
        empty={
          <EmptyState
            icon={<MailPlus className="h-5 w-5" />}
            title="No invitations yet"
            description="Invite colleagues by email. They choose a password (or sign in) when they accept, and land with the role you pick."
            action={can("admin.permissions") ? <Button onClick={() => setInviting(true)}>Invite people</Button> : undefined}
          />
        }
      />
      <InviteDialog open={inviting} onOpenChange={setInviting} />
      <ConfirmDialog
        open={Boolean(revoking)}
        onOpenChange={(v) => !v && setRevoking(null)}
        title="Revoke invitation?"
        description={revoking ? `The invitation link sent to ${revoking.email} will stop working. You can invite them again later.` : undefined}
        confirmLabel="Revoke"
        loading={revoke.isPending}
        onConfirm={() => {
          if (revoking) revoke.mutate(revoking);
        }}
      />
    </>
  );
}

export function InviteDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  const { org } = useAuth();
  const qc = useQueryClient();
  const roles = useRoles(open);
  const teams = useTeams(open);
  const workspaces = useWorkspaces(open);
  const assignable = useMemo(() => (roles.data ?? []).filter((r) => canAssignRole(r, org)), [roles.data, org]);
  const [email, setEmail] = useState("");
  const [roleId, setRoleId] = useState("");
  const [teamId, setTeamId] = useState("");
  const [wsId, setWsId] = useState("");

  useEffect(() => {
    if (open) {
      setEmail("");
      setTeamId("");
      setWsId("");
    }
  }, [open]);
  useEffect(() => {
    if (open && !roleId && assignable.length) setRoleId(assignable.find((r) => r.key === "member")?.id ?? assignable[assignable.length - 1].id);
  }, [open, assignable, roleId]);

  const m = useMutation({
    mutationFn: () => post<{ invitation: Invitation; email_sent: boolean }>("/invitations", { email: email.trim(), role_id: roleId || null, team_id: teamId || null, workspace_id: wsId || null }),
    onSuccess: (r) => {
      toast.success(`Invitation sent to ${r.invitation.email}`, { description: "The link is valid for 7 days." });
      qc.invalidateQueries({ queryKey: ["/invitations"] });
      onOpenChange(false);
    },
  });
  const validEmail = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim());
  const role = assignable.find((r) => r.id === roleId);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        title="Invite to organization"
        description="We'll email a one-time link. Existing accounts sign in to accept; new users choose a password."
        footer={
          <>
            <CancelButton />
            <Button type="submit" form="invite-form" loading={m.isPending} disabled={!validEmail || !roleId}>
              <Send className="h-3.5 w-3.5" /> Send invitation
            </Button>
          </>
        }
      >
        <form
          id="invite-form"
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (validEmail && roleId) m.mutate();
          }}
        >
          <Field label="Email address" htmlFor="invite-email">
            <Input id="invite-email" type="email" autoComplete="off" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="colleague@company.com" autoFocus />
          </Field>
          <Field label="Role" htmlFor="invite-role" hint={role?.description || "Only roles at or below your own rank can be assigned."}>
            <Select id="invite-role" value={roleId} onChange={(e) => setRoleId(e.target.value)} disabled={!assignable.length}>
              {!assignable.length ? <option value="">No roles you can assign</option> : null}
              {assignable.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name} {r.is_system ? "" : "(custom)"}
                </option>
              ))}
            </Select>
          </Field>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Add to team (optional)" htmlFor="invite-team">
              <Select id="invite-team" value={teamId} onChange={(e) => setTeamId(e.target.value)}>
                <option value="">No team</option>
                {(teams.data ?? []).map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.name}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Add to workspace (optional)" htmlFor="invite-ws">
              <Select id="invite-ws" value={wsId} onChange={(e) => setWsId(e.target.value)} disabled={!workspaces.data}>
                <option value="">No workspace</option>
                {(workspaces.data ?? []).map((w) => (
                  <option key={w.id} value={w.id}>
                    {w.name}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
          {org && org.permissions.max_rank < 100 ? <Callout tone="neutral">Roles ranked above yours (rank {org.permissions.max_rank}) are hidden.</Callout> : null}
        </form>
      </DialogContent>
    </Dialog>
  );
}
