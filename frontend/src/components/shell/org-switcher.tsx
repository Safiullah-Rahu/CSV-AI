"use client";

import { useMutation } from "@tanstack/react-query";
import { Check, ChevronsUpDown, LogOut, Monitor, Moon, Plus, Sun, UserRound } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { toast } from "sonner";

import { useTheme } from "@/app/providers";
import { OrgAvatar } from "@/components/shell/branding";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, Input } from "@/components/ui/form";
import { Avatar, Menu, MenuContent, MenuItem, MenuLabel, MenuSeparator, MenuTrigger } from "@/components/ui/primitives";
import { post } from "@/lib/api";
import { useAuth } from "@/lib/auth";

/** Organization switcher at the top of the sidebar. */
export function OrgSwitcher({ onNavigate }: { onNavigate?: () => void }) {
  const { orgs, org, orgId, switchOrg } = useAuth();
  const router = useRouter();
  const [createOpen, setCreateOpen] = useState(false);
  const current = orgs.find((o) => o.id === orgId);
  const name = org?.organization.name ?? current?.name ?? "Select organization";
  const logo = org?.organization.settings?.branding?.logo_url as string | null | undefined;
  return (
    <>
      <Menu>
        <MenuTrigger asChild>
          <button
            className="group flex w-full min-w-0 items-center gap-2.5 rounded-xl px-2 py-1.5 text-left transition-colors hover:bg-black/[0.04] data-[state=open]:bg-black/[0.04] dark:hover:bg-white/[0.05] dark:data-[state=open]:bg-white/[0.05]"
            aria-label="Switch organization"
          >
            <OrgAvatar name={name} logo={logo} />
            <span className="min-w-0 flex-1 leading-tight">
              <span className="flex items-center gap-1.5">
                <span className="truncate text-[13.5px] font-semibold tracking-[-0.01em]">{name}</span>
                {org?.permissions.via_support ? <span className="rounded bg-warn/15 px-1 font-mono text-[9.5px] font-semibold text-warn">SUPPORT</span> : null}
              </span>
              <span className="block truncate text-[11.5px] text-subtle">{current?.roles[0] ?? (org?.plan ? `${org.plan} plan` : "Organization")}</span>
            </span>
            <ChevronsUpDown className="h-3.5 w-3.5 shrink-0 text-subtle transition-colors group-hover:text-muted" />
          </button>
        </MenuTrigger>
        <MenuContent align="start" className="w-[260px]">
          <MenuLabel>Organizations</MenuLabel>
          {orgs.map((o) => (
            <MenuItem
              key={o.id}
              onSelect={() => {
                switchOrg(o.id);
                onNavigate?.();
                router.push("/overview");
              }}
            >
              <OrgAvatar name={o.name} className="h-5 w-5 rounded-md text-[9px]" />
              <span className="min-w-0 flex-1 truncate">{o.name}</span>
              {o.id === orgId ? <Check className="h-3.5 w-3.5 !text-accent" /> : <span className="truncate text-[11px] text-subtle">{o.roles[0]}</span>}
            </MenuItem>
          ))}
          <MenuSeparator />
          <MenuItem onSelect={() => setCreateOpen(true)}>
            <Plus className="h-3.5 w-3.5" /> Create organization
          </MenuItem>
        </MenuContent>
      </Menu>
      <CreateOrgDialog open={createOpen} onOpenChange={setCreateOpen} />
    </>
  );
}

function CreateOrgDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  const { switchOrg, reload } = useAuth();
  const router = useRouter();
  const [name, setName] = useState("");
  const m = useMutation({
    mutationFn: () => post<{ id: string }>("/orgs", { name }, { noOrg: true }),
    onSuccess: async (org) => {
      await reload();
      switchOrg(org.id);
      onOpenChange(false);
      setName("");
      toast.success("Organization created");
      router.push("/overview");
    },
  });
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        title="Create organization"
        description="A separate, isolated tenant with its own members, knowledge and settings."
        footer={
          <Button onClick={() => m.mutate()} loading={m.isPending} disabled={name.trim().length < 2}>
            Create organization
          </Button>
        }
      >
        <Field htmlFor="new-org-name" label="Organization name">
          <Input id="new-org-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Acme Corporation" autoFocus />
        </Field>
      </DialogContent>
    </Dialog>
  );
}

/** Account menu at the bottom of the sidebar. */
export function UserMenu() {
  const { user, signOut } = useAuth();
  const { theme, setTheme } = useTheme();
  const router = useRouter();
  return (
    <Menu>
      <MenuTrigger asChild>
        <button
          className="flex w-full min-w-0 items-center gap-2.5 rounded-xl px-2 py-1.5 text-left transition-colors hover:bg-black/[0.04] data-[state=open]:bg-black/[0.04] dark:hover:bg-white/[0.05] dark:data-[state=open]:bg-white/[0.05]"
          aria-label="Account menu"
        >
          <Avatar name={user?.full_name} email={user?.email} />
          <span className="min-w-0 flex-1 leading-tight">
            <span className="block truncate text-[13px] font-medium">{user?.full_name || user?.email}</span>
            <span className="block truncate text-[11.5px] text-subtle">{user?.email}</span>
          </span>
          <ChevronsUpDown className="h-3.5 w-3.5 shrink-0 text-subtle" />
        </button>
      </MenuTrigger>
      <MenuContent align="start" side="top" className="w-[240px]">
        <MenuItem onSelect={() => router.push("/settings")}>
          <UserRound className="h-3.5 w-3.5" /> Profile & security
        </MenuItem>
        <MenuSeparator />
        <MenuLabel>Theme</MenuLabel>
        {(
          [
            ["light", Sun, "Light"],
            ["dark", Moon, "Dark"],
            ["system", Monitor, "System"],
          ] as const
        ).map(([value, Icon, label]) => (
          <MenuItem key={value} onSelect={() => setTheme(value)}>
            <Icon className="h-3.5 w-3.5" /> {label}
            {theme === value ? <Check className="ml-auto h-3.5 w-3.5 !text-accent" /> : null}
          </MenuItem>
        ))}
        <MenuSeparator />
        <MenuItem
          onSelect={async () => {
            await signOut();
            router.replace("/login");
          }}
        >
          <LogOut className="h-3.5 w-3.5" /> Sign out
        </MenuItem>
      </MenuContent>
    </Menu>
  );
}
