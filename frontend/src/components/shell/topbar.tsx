"use client";

import * as Popover from "@radix-ui/react-popover";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bell, BellOff, ChevronRight, Menu as MenuIcon, Moon, Search, Sun } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

import { useTheme } from "@/app/providers";
import { CommandPalette } from "@/components/shell/command-palette";
import { useBreadcrumbLabel } from "@/components/shell/breadcrumb";
import { NAV } from "@/components/shell/nav";
import { Kbd, Tip } from "@/components/ui/primitives";
import { get, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { cn, timeAgo } from "@/lib/utils";

type Notification = { id: string; kind: string; title: string; body: string; link: string | null; severity: string; read: boolean; created_at: string };

export function Topbar({ onMenu }: { onMenu: () => void }) {
  const pathname = usePathname();
  const pageLabel = useBreadcrumbLabel();
  const { resolved, setTheme } = useTheme();
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [mod, setMod] = useState("Ctrl");
  useEffect(() => {
    if (/Mac|iPhone|iPad/.test(navigator.platform)) setMod("⌘");
  }, []);

  const crumb = useMemo(() => {
    for (const section of NAV) {
      const item = section.items.find((i) => pathname === i.href || pathname.startsWith(i.href + "/"));
      if (item) return { group: section.group, label: item.label, href: item.href, nested: pathname !== item.href };
    }
    return null;
  }, [pathname]);

  return (
    <header className="sticky top-0 z-30 flex h-12 shrink-0 items-center gap-2 border-b bg-bg/85 px-3 backdrop-blur-md sm:px-5">
      <button className="-ml-1 rounded-lg p-1.5 text-muted hover:bg-surface-2 hover:text-fg lg:hidden" onClick={onMenu} aria-label="Open navigation">
        <MenuIcon className="h-[18px] w-[18px]" />
      </button>

      {crumb ? (
        <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-1.5 text-[13px]">
          <span className="hidden text-subtle sm:inline">{crumb.group}</span>
          <ChevronRight className="hidden h-3.5 w-3.5 shrink-0 text-border-strong sm:block" />
          {crumb.nested ? (
            <>
              <Link href={crumb.href} className="truncate text-subtle transition-colors hover:text-fg">
                {crumb.label}
              </Link>
              <ChevronRight className="h-3.5 w-3.5 shrink-0 text-border-strong" />
              <span className="truncate font-medium text-fg">{pageLabel ?? "Details"}</span>
            </>
          ) : (
            <span className="truncate font-medium text-fg">{crumb.label}</span>
          )}
        </nav>
      ) : null}

      <div className="ml-auto flex items-center gap-1">
        <button
          onClick={() => setPaletteOpen(true)}
          className="mr-1 hidden h-8 w-64 items-center gap-2 rounded-lg border bg-surface px-2.5 text-[13px] text-subtle shadow-xs transition-colors hover:border-border-strong hover:text-muted md:flex"
          aria-label="Search"
        >
          <Search className="h-3.5 w-3.5" />
          <span className="flex-1 text-left">Search everything…</span>
          <Kbd>{mod} K</Kbd>
        </button>
        <button className="rounded-lg p-2 text-muted hover:bg-surface-2 hover:text-fg md:hidden" onClick={() => setPaletteOpen(true)} aria-label="Search">
          <Search className="h-4 w-4" />
        </button>
        <Tip content={resolved === "dark" ? "Switch to light" : "Switch to dark"} side="bottom">
          <button
            className="rounded-lg p-2 text-muted transition-colors hover:bg-surface-2 hover:text-fg"
            onClick={() => setTheme(resolved === "dark" ? "light" : "dark")}
            aria-label={resolved === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          >
            {resolved === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </Tip>
        <Notifications />
      </div>

      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} />
    </header>
  );
}

function Notifications() {
  const qc = useQueryClient();
  const { orgId } = useAuth();
  const { data } = useQuery({
    queryKey: ["notifications", orgId],
    queryFn: () => get<{ unread: number; items: Notification[] }>("/notifications"),
    enabled: Boolean(orgId),
    refetchInterval: 60_000,
    meta: { silent: true },
  });
  const markRead = useMutation({
    mutationFn: () => post("/notifications/read"),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["notifications"] }),
  });
  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button className="relative rounded-lg p-2 text-muted transition-colors hover:bg-surface-2 hover:text-fg data-[state=open]:bg-surface-2 data-[state=open]:text-fg" aria-label={`Notifications${data?.unread ? ` (${data.unread} unread)` : ""}`}>
          <Bell className="h-4 w-4" />
          {data?.unread ? <span className="absolute right-[7px] top-[7px] h-2 w-2 rounded-full bg-accent ring-2 ring-bg" /> : null}
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content align="end" sideOffset={8} className="z-50 w-[380px] overflow-hidden rounded-2xl border bg-surface shadow-float animate-scale-in">
          <div className="flex items-center justify-between border-b px-4 py-3">
            <span className="text-[13.5px] font-semibold tracking-[-0.01em]">Notifications</span>
            {data?.unread ? (
              <button className="text-xs text-accent hover:underline" onClick={() => markRead.mutate()}>
                Mark all read
              </button>
            ) : null}
          </div>
          <div className="max-h-96 overflow-y-auto">
            {!data?.items.length ? (
              <div className="flex flex-col items-center px-4 py-10 text-center">
                <BellOff className="mb-2 h-5 w-5 text-subtle" />
                <p className="text-[13px] font-medium">You&apos;re all caught up</p>
                <p className="mt-0.5 text-[12px] text-subtle">Background jobs you started that fail, such as ingestion or syncs, show up here.</p>
              </div>
            ) : (
              data.items.map((n) => (
                <Link key={n.id} href={n.link || "#"} className={cn("block border-b px-4 py-3 transition-colors last:border-0 hover:bg-surface-2/70", !n.read && "bg-accent/[0.04]")}>
                  <div className="flex items-center gap-2">
                    <span className={cn("h-1.5 w-1.5 rounded-full", n.severity === "error" ? "bg-danger" : n.severity === "warning" ? "bg-warn" : "bg-info")} />
                    <span className="flex-1 truncate text-[13px] font-medium">{n.title}</span>
                    <span className="text-2xs text-subtle">{timeAgo(n.created_at)}</span>
                  </div>
                  {n.body ? <p className="mt-1 line-clamp-2 text-xs text-muted">{n.body}</p> : null}
                </Link>
              ))
            )}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
