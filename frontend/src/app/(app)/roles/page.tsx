"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Copy, Eye, Grid3x3, KeyRound, Lock, Pencil, Plus, ShieldCheck, Trash2, UserCog } from "lucide-react";
import { Fragment, Suspense, useMemo, useState } from "react";
import { toast } from "sonner";

import { groupPermissions, usePermissionCatalog, useRoles } from "@/components/org/data";
import { GatedButton, RowMenuButton, SearchBox, TabBar, useUrlTab } from "@/components/org/kit";
import { AssignmentsTab } from "@/components/org/role-assignments";
import { type RoleEditorState, RoleEditorDialog, RoleViewDialog } from "@/components/org/role-editor";
import type { Role } from "@/components/org/types";
import { type Column, DataTable } from "@/components/ui/data-table";
import { ConfirmDialog } from "@/components/ui/dialog";
import { Badge, Card, EmptyState, Menu, MenuContent, MenuItem, MenuSeparator, MenuTrigger, Mono, Page, PageHeader, Skeleton, SkeletonRows, Tabs, TabsContent, TabsList, TabsTrigger, Tip } from "@/components/ui/primitives";
import { del } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/utils";

const TABS = ["roles", "matrix", "assignments"] as const;

export default function RolesPage() {
  return (
    <Suspense fallback={<Page><Skeleton className="h-96 rounded-xl" /></Page>}>
      <RolesInner />
    </Suspense>
  );
}

function RolesInner() {
  const { can } = useAuth();
  const [tab, setTab] = useUrlTab(TABS, "roles");
  const [editor, setEditor] = useState<RoleEditorState>(null);

  return (
    <Page>
      <PageHeader
        eyebrow="Organization"
        title="Roles & permissions"
        description="Roles bundle capabilities; assignments give them to people or teams, organization-wide or inside one workspace. Access to specific knowledge is controlled by grants on each resource."
        actions={
          <GatedButton perm="admin.roles" onClick={() => setEditor({ mode: "create" })}>
            <Plus className="h-4 w-4" /> New custom role
          </GatedButton>
        }
      />
      <Tabs value={tab} onValueChange={setTab}>
        <TabBar>
          <TabsList variant="line">
            <TabsTrigger value="roles">
              <ShieldCheck className="h-3.5 w-3.5" /> Roles
            </TabsTrigger>
            <TabsTrigger value="matrix">
              <Grid3x3 className="h-3.5 w-3.5" /> Permission matrix
            </TabsTrigger>
            <TabsTrigger value="assignments">
              <UserCog className="h-3.5 w-3.5" /> Assignments
            </TabsTrigger>
          </TabsList>
        </TabBar>
        <TabsContent value="roles" className="animate-fade-in focus-visible:outline-none">
          <RolesList onEdit={setEditor} />
        </TabsContent>
        <TabsContent value="matrix" className="animate-fade-in focus-visible:outline-none">
          <PermissionMatrix />
        </TabsContent>
        <TabsContent value="assignments" className="animate-fade-in focus-visible:outline-none">
          <AssignmentsTab />
          {!can("admin.permissions") ? <p className="mt-3 text-[12px] text-subtle">You can view assignments. Changing them requires the admin.permissions permission.</p> : null}
        </TabsContent>
      </Tabs>
      <RoleEditorDialog state={editor} onClose={() => setEditor(null)} />
    </Page>
  );
}

function RankMeter({ rank, dim }: { rank: number; dim?: boolean }) {
  return (
    <span className={cn("inline-flex items-center gap-2", dim && "opacity-55")}>
      <span className="num w-7 font-mono text-[12.5px]">{rank}</span>
      <span className="h-1 w-14 overflow-hidden rounded-full bg-surface-2 ring-1 ring-inset ring-border/60" aria-hidden>
        <span className="block h-full rounded-full bg-fg/45" style={{ width: `${Math.min(100, Math.max(0, rank))}%` }} />
      </span>
    </span>
  );
}

function RolesList({ onEdit }: { onEdit: (s: RoleEditorState) => void }) {
  const { can, org } = useAuth();
  const qc = useQueryClient();
  const roles = useRoles();
  const [viewing, setViewing] = useState<Role | null>(null);
  const [deleting, setDeleting] = useState<Role | null>(null);
  const maxRank = org?.permissions.max_rank ?? 0;
  const manage = can("admin.roles");

  const remove = useMutation({
    mutationFn: (r: Role) => del(`/roles/${r.id}`),
    onSuccess: (_x, r) => {
      toast.success(`Role “${r.name}” deleted`);
      setDeleting(null);
      qc.invalidateQueries({ queryKey: ["/roles"] });
    },
  });

  const columns: Column<Role>[] = [
    {
      key: "name",
      header: "Role",
      sortValue: (r) => r.name.toLowerCase(),
      className: "min-w-[280px]",
      cell: (r) => (
        <button className="group/role flex min-w-0 items-center gap-3 text-left" onClick={() => setViewing(r)}>
          <span className={cn("grid h-8 w-8 shrink-0 place-items-center rounded-lg border transition-colors", r.is_system ? "bg-surface-2/60 text-subtle" : "border-accent/20 bg-accent/[0.07] text-accent")}>
            {r.is_system ? <Lock className="h-3.5 w-3.5" aria-label="System role" /> : <ShieldCheck className="h-4 w-4" aria-label="Custom role" />}
          </span>
          <span className="min-w-0 leading-tight">
            <span className="block truncate font-medium text-fg group-hover/role:text-accent">{r.name}</span>
            <span className="mt-0.5 line-clamp-1 text-[12px] text-subtle">{r.description || "No description"}</span>
          </span>
        </button>
      ),
    },
    { key: "key", header: "Key", sortValue: (r) => r.key, cell: (r) => <Mono className="text-muted">{r.key}</Mono> },
    { key: "rank", header: "Rank", sortValue: (r) => r.rank, cell: (r) => <RankMeter rank={r.rank} dim={r.rank > maxRank} /> },
    { key: "type", header: "Type", sortValue: (r) => (r.is_system ? 0 : 1), cell: (r) => <Badge tone={r.is_system ? "neutral" : "accent"}>{r.is_system ? "system" : "custom"}</Badge> },
    { key: "perms", header: "Permissions", sortValue: (r) => r.permissions.length, headerClassName: "text-right", className: "text-right", cell: (r) => <span className="num font-mono text-[12.5px]">{r.permissions.length}</span> },
    { key: "assignments", header: "Assigned", sortValue: (r) => r.assignment_count, headerClassName: "text-right", className: "text-right", cell: (r) => <span className={cn("num font-mono text-[12.5px]", r.assignment_count ? "text-fg" : "text-subtle")}>{r.assignment_count}</span> },
    {
      key: "actions",
      header: <span className="relative block"><span className="sr-only">Actions</span></span>,
      className: "w-12 text-right",
      cell: (r) => {
        const editable = manage && !r.is_system && r.rank <= maxRank;
        return (
          <Menu>
            <MenuTrigger asChild>
              <RowMenuButton label={`Actions for ${r.name}`} />
            </MenuTrigger>
            <MenuContent className="w-60">
              <MenuItem onSelect={() => setViewing(r)}>
                <Eye className="h-3.5 w-3.5" /> View permissions
              </MenuItem>
              {manage ? (
                <MenuItem onSelect={() => onEdit({ mode: "duplicate", role: r })}>
                  <Copy className="h-3.5 w-3.5" /> Duplicate as custom role
                </MenuItem>
              ) : null}
              {!r.is_system && manage ? (
                <>
                  <MenuItem disabled={!editable} onSelect={() => onEdit({ mode: "edit", role: r })}>
                    <Pencil className="h-3.5 w-3.5" /> Edit role
                    {!editable ? <span className="ml-auto text-[10px] text-subtle">above your rank</span> : null}
                  </MenuItem>
                  <MenuSeparator />
                  <MenuItem danger disabled={r.assignment_count > 0} onSelect={() => setDeleting(r)}>
                    <Trash2 className="h-3.5 w-3.5" /> Delete role
                    {r.assignment_count > 0 ? <span className="ml-auto text-[10px] text-subtle">in use</span> : null}
                  </MenuItem>
                </>
              ) : null}
            </MenuContent>
          </Menu>
        );
      },
    },
  ];

  return (
    <>
      <DataTable
        rows={roles.data}
        columns={columns}
        rowKey={(r) => r.id}
        loading={roles.isLoading}
        filterText={(r) => `${r.name} ${r.key} ${r.description}`}
        searchPlaceholder="Search roles…"
        pageSize={50}
        toolbar={
          <span className="inline-flex items-center gap-2 text-[12px] text-muted">
            Your highest rank <span className="num rounded-[5px] border bg-surface-2 px-1.5 py-0.5 font-mono text-[11.5px] text-fg">{maxRank}</span>
          </span>
        }
        empty={<EmptyState icon={<KeyRound className="h-5 w-5" />} title="No roles" />}
      />
      <p className="mt-3 max-w-3xl text-[12px] leading-relaxed text-subtle">
        You can assign and edit roles at or below your rank. System roles are read-only; duplicate one to customise it. A role must have no assignments before it can be deleted.
      </p>
      <RoleViewDialog role={viewing} onClose={() => setViewing(null)} />
      <ConfirmDialog
        open={Boolean(deleting)}
        onOpenChange={(v) => !v && setDeleting(null)}
        title="Delete custom role?"
        description={deleting ? `“${deleting.name}” will be permanently deleted. It currently has no assignments.` : undefined}
        confirmLabel="Delete role"
        requireText={deleting?.key}
        loading={remove.isPending}
        onConfirm={() => {
          if (deleting) remove.mutate(deleting);
        }}
      />
    </>
  );
}

/* Opaque tint for the sticky "You" header (content scrolls beneath it). */
const YOU_HEAD = { background: "linear-gradient(hsl(var(--accent) / 0.07), hsl(var(--accent) / 0.07)), hsl(var(--surface))" };

function Allow({ label }: { label: string }) {
  return (
    <span className="mx-auto grid h-[18px] w-[18px] place-items-center rounded-full bg-accent/[0.12] text-accent" role="img" aria-label={label}>
      <Check className="h-3 w-3" strokeWidth={3} />
    </span>
  );
}
function Deny({ label }: { label: string }) {
  return <span className="mx-auto block h-[2px] w-2.5 rounded-full bg-border-strong" role="img" aria-label={label} />;
}

function PermissionMatrix() {
  const { org } = useAuth();
  const roles = useRoles();
  const catalog = usePermissionCatalog();
  const [q, setQ] = useState("");
  const held = useMemo(() => new Set(org?.permissions.org ?? []), [org]);
  const ordered = useMemo(() => [...(roles.data ?? [])].sort((a, b) => b.rank - a.rank || a.name.localeCompare(b.name)), [roles.data]);
  const roleSets = useMemo(() => new Map(ordered.map((r) => [r.id, new Set(r.permissions)])), [ordered]);
  const groups = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return groupPermissions(catalog.data)
      .map((g) => ({ ...g, items: needle ? g.items.filter((p) => `${p.key} ${p.description} ${p.category}`.toLowerCase().includes(needle)) : g.items }))
      .filter((g) => g.items.length);
  }, [catalog.data, q]);
  const shown = groups.reduce((a, g) => a + g.items.length, 0);

  if (roles.isLoading || catalog.isLoading)
    return (
      <Card>
        <SkeletonRows rows={10} />
      </Card>
    );

  return (
    <Card className="overflow-hidden">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2.5 border-b px-3.5 py-3">
        <SearchBox value={q} onChange={setQ} placeholder="Filter permissions…" label="Filter permissions" />
        <span className="num font-mono text-[11.5px] text-subtle">
          {shown} of {catalog.data?.length ?? 0} permissions · {ordered.length} roles
        </span>
        <span className="flex items-center gap-4 text-[12px] text-muted sm:ml-auto">
          <span className="inline-flex items-center gap-1.5">
            <Allow label="Granted" /> Granted
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="grid h-[18px] w-[18px] place-items-center">
              <Deny label="Not granted" />
            </span>
            Not granted
          </span>
        </span>
      </div>
      <div className="max-h-[70vh] overflow-auto overscroll-contain">
        <table className="w-full border-separate border-spacing-0 text-[12.5px]">
          <thead>
            <tr>
              <th className="sticky left-0 top-0 z-30 min-w-[200px] border-b border-r bg-surface px-4 py-3 text-left align-bottom font-mono text-[10.5px] font-medium uppercase tracking-[0.1em] text-subtle sm:min-w-[280px]">Permission</th>
              <th className="sticky top-0 z-20 min-w-[72px] border-b px-2 py-3 text-center align-bottom" style={YOU_HEAD}>
                <span className="block text-[11.5px] font-semibold leading-tight text-accent">You</span>
                <span className="mt-1 block font-mono text-[10px] font-normal text-subtle">effective</span>
              </th>
              {ordered.map((r) => (
                <th key={r.id} className="sticky top-0 z-20 min-w-[96px] max-w-[120px] border-b bg-surface px-2 py-3 text-center align-bottom">
                  <Tip content={`${r.name} · rank ${r.rank} · ${r.permissions.length} permissions`}>
                    <span tabIndex={0} className="line-clamp-2 block rounded text-[11.5px] font-medium leading-tight text-fg">
                      {r.name}
                    </span>
                  </Tip>
                  <span className="num mt-1 block font-mono text-[10px] font-normal text-subtle">
                    {r.is_system ? "system" : "custom"} · {r.rank}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {groups.map((g) => (
              <Fragment key={g.category}>
                <tr>
                  <th colSpan={ordered.length + 2} className="border-b bg-surface-2 p-0 text-left">
                    <span className="sticky left-0 inline-flex items-center gap-2 px-4 py-2 font-mono text-[10.5px] font-medium uppercase tracking-[0.12em] text-muted">
                      {g.category}
                      <span className="num font-normal text-subtle">{g.items.length}</span>
                    </span>
                  </th>
                </tr>
                {g.items.map((p) => (
                  <tr key={p.key} className="group/perm">
                    <td className="sticky left-0 z-10 border-b border-r bg-surface px-4 py-2 transition-colors group-hover/perm:bg-surface-2">
                      <div className="font-mono text-[11.5px] text-fg">{p.key}</div>
                      <div className="mt-0.5 line-clamp-1 text-[11.5px] text-subtle">{p.description}</div>
                    </td>
                    <td className="border-b bg-accent/[0.05] text-center transition-colors group-hover/perm:bg-surface-2">{held.has(p.key) ? <Allow label="You hold this" /> : <Deny label="You don't hold this" />}</td>
                    {ordered.map((r) => {
                      const has = roleSets.get(r.id)?.has(p.key);
                      return (
                        <td key={r.id} className="border-b text-center transition-colors group-hover/perm:bg-surface-2">
                          {has ? <Allow label={`${r.name} has ${p.key}`} /> : <Deny label={`${r.name} lacks ${p.key}`} />}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </Fragment>
            ))}
          </tbody>
        </table>
        {groups.length === 0 ? <EmptyState title="No permissions match" description="Try a permission key such as kb.chat, or a category name." /> : null}
      </div>
    </Card>
  );
}
