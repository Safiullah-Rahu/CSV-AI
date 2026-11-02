"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Clock, Crown, FolderKanban, Pencil, ShieldCheck, ShieldOff, Trash2, UserCheck, Users, UserX } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { useDepartments, useIndex, useMembers, useTeams, useWorkspaces } from "@/components/org/data";
import { CancelButton, PersonCell, RowMenuButton, Segmented, When } from "@/components/org/kit";
import type { Member } from "@/components/org/types";
import { Button } from "@/components/ui/button";
import { type Column, DataTable } from "@/components/ui/data-table";
import { ConfirmDialog, Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Select } from "@/components/ui/form";
import { Badge, EmptyState, Menu, MenuContent, MenuItem, MenuSeparator, MenuTrigger, Skeleton, Stat, StatGroup, StatusBadge, Tip } from "@/components/ui/primitives";
import { del, patch } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { formatDate } from "@/lib/utils";

export function MembersTab() {
  const { can, user } = useAuth();
  const qc = useQueryClient();
  const members = useMembers();
  const teams = useTeams();
  const workspaces = useWorkspaces();
  const teamById = useIndex(teams.data, (t) => t.id);
  const wsById = useIndex(workspaces.data, (w) => w.id);
  const manage = can("org.users");
  const [status, setStatus] = useState<"all" | "active" | "suspended">("all");
  const [editing, setEditing] = useState<Member | null>(null);
  const [removing, setRemoving] = useState<Member | null>(null);

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["/members"] });
    qc.invalidateQueries({ queryKey: ["/departments"] });
    qc.invalidateQueries({ queryKey: ["/teams"] });
  };
  const setMemberStatus = useMutation({
    mutationFn: ({ m, next }: { m: Member; next: "active" | "suspended" }) => patch(`/members/${m.user_id}`, { status: next }),
    onSuccess: (_r, { m, next }) => {
      toast.success(next === "suspended" ? `${m.full_name || m.email} suspended` : `${m.full_name || m.email} reactivated`);
      invalidate();
    },
  });
  const [suspending, setSuspending] = useState<Member | null>(null);
  const remove = useMutation({
    mutationFn: (m: Member) => del(`/members/${m.user_id}`),
    onSuccess: (_r, m) => {
      toast.success(`${m.full_name || m.email} removed from the organization`);
      setRemoving(null);
      invalidate();
      qc.invalidateQueries({ queryKey: ["/role-assignments"] });
      qc.invalidateQueries({ queryKey: ["/workspaces"] });
    },
  });

  const rows = useMemo(() => (members.data?.items ?? []).filter((m) => status === "all" || m.status === status), [members.data, status]);
  const summary = useMemo(() => {
    const all = members.data?.items ?? [];
    const active = all.filter((m) => m.status === "active").length;
    const mfa = all.filter((m) => m.mfa_enabled).length;
    return { total: all.length, active, suspended: all.filter((m) => m.status === "suspended").length, mfa, mfaPct: all.length ? Math.round((mfa / all.length) * 100) : 0 };
  }, [members.data]);

  const columns: Column<Member>[] = [
    {
      key: "member",
      header: "Member",
      sortValue: (m) => (m.full_name || m.email).toLowerCase(),
      cell: (m) => <PersonCell name={m.full_name} email={m.email} sub={m.title || undefined} you={m.user_id === user?.id} />,
      className: "min-w-[220px]",
    },
    {
      key: "roles",
      header: "Roles",
      cell: (m) =>
        m.roles.length ? (
          <div className="flex max-w-[300px] flex-wrap gap-1">
            {m.roles.map((r) => (
              <Badge key={r.assignment_id} tone={r.key === "owner" ? "accent" : "neutral"} className="gap-1">
                {r.key === "owner" ? <Crown className="h-3 w-3" aria-hidden /> : null}
                {r.name}
                {r.scope_type === "workspace" ? (
                  <span className="inline-flex items-center gap-0.5 text-subtle">
                    · <FolderKanban className="h-3 w-3" aria-hidden /> {wsById.get(r.scope_id || "")?.name ?? "workspace"}
                  </span>
                ) : null}
                {r.expires_at ? (
                  <Tip content={`Expires ${formatDate(r.expires_at)}`}>
                    <Clock className="h-3 w-3 text-warn" aria-label={`Expires ${formatDate(r.expires_at)}`} />
                  </Tip>
                ) : null}
              </Badge>
            ))}
          </div>
        ) : (
          <span className="text-[12px] text-subtle">Team roles only</span>
        ),
    },
    { key: "department", header: "Department", sortValue: (m) => m.department || "", cell: (m) => (m.department ? <span className="whitespace-nowrap text-muted">{m.department}</span> : <span className="text-subtle">—</span>) },
    {
      key: "teams",
      header: "Teams",
      cell: (m) =>
        m.teams.length ? (
          <div className="flex max-w-[220px] flex-wrap gap-1">
            {m.teams.map((t) => (
              <Badge key={t} className="gap-1">
                <Users className="h-3 w-3 text-subtle" aria-hidden />
                {teamById.get(t)?.name ?? "team"}
              </Badge>
            ))}
          </div>
        ) : (
          <span className="text-subtle">—</span>
        ),
    },
    {
      key: "status",
      header: "Status",
      sortValue: (m) => m.status,
      cell: (m) => (
        <div className="flex flex-col items-start gap-1">
          <StatusBadge status={m.status} />
          {m.user_status !== "active" ? <span className="text-[11px] text-subtle">account {m.user_status}</span> : null}
        </div>
      ),
    },
    {
      key: "mfa",
      header: "MFA",
      sortValue: (m) => (m.mfa_enabled ? 1 : 0),
      cell: (m) =>
        m.mfa_enabled ? (
          <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-[12.5px] font-medium text-ok">
            <ShieldCheck className="h-3.5 w-3.5" /> On
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-[12.5px] text-subtle">
            <ShieldOff className="h-3.5 w-3.5" /> Off
          </span>
        ),
    },
    { key: "login", header: "Last login", sortValue: (m) => m.last_login_at || "", cell: (m) => <When iso={m.last_login_at} className="text-muted" /> },
    {
      key: "actions",
      header: <span className="relative block"><span className="sr-only">Actions</span></span>,
      className: "w-10 text-right",
      cell: (m) =>
        manage ? (
          <Menu>
            <MenuTrigger asChild>
              <RowMenuButton label={`Actions for ${m.full_name || m.email}`} />
            </MenuTrigger>
            <MenuContent className="w-52">
              <MenuItem onSelect={() => setEditing(m)}>
                <Pencil className="h-3.5 w-3.5" /> Edit details
              </MenuItem>
              {m.user_id !== user?.id ? (
                m.status === "active" ? (
                  <MenuItem onSelect={() => setSuspending(m)}>
                    <UserX className="h-3.5 w-3.5" /> Suspend access
                  </MenuItem>
                ) : (
                  <MenuItem onSelect={() => setMemberStatus.mutate({ m, next: "active" })}>
                    <UserCheck className="h-3.5 w-3.5" /> Reactivate
                  </MenuItem>
                )
              ) : null}
              {m.user_id !== user?.id ? (
                <>
                  <MenuSeparator />
                  <MenuItem danger onSelect={() => setRemoving(m)}>
                    <Trash2 className="h-3.5 w-3.5" /> Remove from organization
                  </MenuItem>
                </>
              ) : null}
            </MenuContent>
          </Menu>
        ) : null,
    },
  ];

  return (
    <div className="space-y-5">
      <StatGroup cols={4}>
        {members.isLoading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="space-y-3 px-5 py-4">
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-7 w-12" />
            </div>
          ))
        ) : (
          <>
            <Stat label="Members" value={summary.total} hint="In this organization" icon={<Users />} />
            <Stat label="Active" value={summary.active} hint={summary.total ? `${Math.round((summary.active / summary.total) * 100)}% of members` : "No members yet"} icon={<UserCheck />} tone="ok" />
            <Stat label="Suspended" value={summary.suspended} hint={summary.suspended ? "Access paused, content kept" : "Nobody is suspended"} icon={<UserX />} tone={summary.suspended ? "warn" : undefined} />
            <Stat label="Two-factor" value={`${summary.mfaPct}%`} hint={`${summary.mfa} of ${summary.total} protected`} icon={<ShieldCheck />} tone={summary.total && summary.mfa === summary.total ? "ok" : undefined} />
          </>
        )}
      </StatGroup>
      <div className="space-y-3">
      <DataTable
        rows={rows}
        columns={columns}
        rowKey={(m) => m.user_id}
        loading={members.isLoading}
        filterText={(m) => `${m.full_name} ${m.email} ${m.title} ${m.department ?? ""} ${m.roles.map((r) => r.name).join(" ")}`}
        searchPlaceholder="Search name, email, role…"
        toolbar={
          <Segmented
            label="Status filter"
            value={status}
            onChange={setStatus}
            options={[
              { value: "all", label: `All${members.data ? ` · ${members.data.items.length}` : ""}` },
              { value: "active", label: "Active" },
              { value: "suspended", label: "Suspended" },
            ]}
          />
        }
        empty={<EmptyState title="No members match" description="Try a different search or status filter." />}
      />
      <p className="text-[12px] leading-relaxed text-subtle">
        Roles are assigned and removed on the{" "}
        <Link href="/roles?tab=assignments" className="font-medium text-accent hover:underline">
          Roles & permissions
        </Link>{" "}
        page. Suspended members keep their account but cannot access this organization.
      </p>
      </div>

      <EditMemberDialog member={editing} onClose={() => setEditing(null)} onSaved={invalidate} />
      <ConfirmDialog
        open={Boolean(suspending)}
        onOpenChange={(v) => !v && setSuspending(null)}
        title="Suspend member?"
        description={suspending ? `${suspending.full_name || suspending.email} will immediately lose access to this organization. Their account, roles and content are kept, and you can reactivate them at any time.` : undefined}
        confirmLabel="Suspend"
        loading={setMemberStatus.isPending}
        onConfirm={() => {
          if (suspending) setMemberStatus.mutate({ m: suspending, next: "suspended" }, { onSuccess: () => setSuspending(null) });
        }}
      />
      <ConfirmDialog
        open={Boolean(removing)}
        onOpenChange={(v) => !v && setRemoving(null)}
        title="Remove member?"
        description={removing ? `${removing.full_name || removing.email} will be removed from this organization, together with their role assignments, team and workspace memberships and direct resource grants. Their user account itself is not deleted.` : undefined}
        confirmLabel="Remove member"
        requireText={removing?.email}
        loading={remove.isPending}
        onConfirm={() => {
          if (removing) remove.mutate(removing);
        }}
      />
    </div>
  );
}

function EditMemberDialog({ member, onClose, onSaved }: { member: Member | null; onClose: () => void; onSaved: () => void }) {
  const { user } = useAuth();
  const departments = useDepartments(Boolean(member));
  const [title, setTitle] = useState("");
  const [dept, setDept] = useState("");
  const [status, setStatus] = useState<"active" | "suspended">("active");
  useEffect(() => {
    if (member) {
      setTitle(member.title || "");
      setDept(member.department_id || "");
      setStatus(member.status === "suspended" ? "suspended" : "active");
    }
  }, [member]);
  const self = member?.user_id === user?.id;
  const m = useMutation({
    mutationFn: () => {
      const body: Record<string, unknown> = {};
      if (title !== (member!.title || "")) body.title = title;
      if (dept !== (member!.department_id || "")) {
        if (dept) body.department_id = dept;
        else body.clear_department = true;
      }
      if (!self && status !== member!.status) body.status = status;
      return patch(`/members/${member!.user_id}`, body);
    },
    onSuccess: () => {
      toast.success("Member updated");
      onSaved();
      onClose();
    },
  });
  const dirty = member && (title !== (member.title || "") || dept !== (member.department_id || "") || (!self && status !== member.status));
  return (
    <Dialog open={Boolean(member)} onOpenChange={(v) => !v && onClose()}>
      <DialogContent
        title="Edit member"
        description={member ? member.full_name || member.email : undefined}
        footer={
          <>
            <CancelButton />
            <Button type="submit" form="edit-member" loading={m.isPending} disabled={!dirty}>
              Save changes
            </Button>
          </>
        }
      >
        <form
          id="edit-member"
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (dirty) m.mutate();
          }}
        >
          <Field label="Job title" htmlFor="member-title">
            <Input id="member-title" value={title} onChange={(e) => setTitle(e.target.value)} maxLength={200} placeholder="e.g. Senior Analyst" />
          </Field>
          <Field label="Department" htmlFor="member-dept" hint={departments.data?.length === 0 ? "No departments yet — create them on the Departments tab." : undefined}>
            <Select id="member-dept" value={dept} onChange={(e) => setDept(e.target.value)}>
              <option value="">No department</option>
              {(departments.data ?? []).map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Membership status" htmlFor="member-status" hint={self ? "You cannot change your own membership status." : "Suspended members cannot sign in to this organization."}>
            <Select id="member-status" value={status} disabled={self} onChange={(e) => setStatus(e.target.value as "active" | "suspended")}>
              <option value="active">Active</option>
              <option value="suspended">Suspended</option>
            </Select>
          </Field>
        </form>
      </DialogContent>
    </Dialog>
  );
}
