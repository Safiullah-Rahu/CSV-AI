"use client";

import { SquarePen } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { NAV } from "@/components/shell/nav";
import { OrgSwitcher, UserMenu } from "@/components/shell/org-switcher";
import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/utils";

/** Product mark: an ink tile with an orbit — a knowledge core and one cited source. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={cn("h-7 w-7 shrink-0", className)} aria-hidden>
      <rect width="32" height="32" rx="9" className="fill-fg" />
      <circle cx="16" cy="16" r="7.25" fill="none" className="stroke-bg" strokeWidth="1.6" opacity=".55" />
      <circle cx="16" cy="16" r="2.6" className="fill-bg" />
      <circle cx="21.13" cy="10.87" r="2.3" className="fill-accent" />
    </svg>
  );
}

export function Logo({ compact }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <LogoMark />
      {!compact ? (
        <div className="leading-tight">
          <div className="text-[13.5px] font-semibold tracking-[-0.015em]">Knowledge Intelligence</div>
          <div className="text-[11px] text-subtle">Enterprise RAG Platform</div>
        </div>
      ) : null}
    </div>
  );
}

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  const { can, canAnywhere, user } = useAuth();
  return (
    <nav className="flex h-full flex-col" aria-label="Main navigation">
      <div className="px-2.5 pt-2.5">
        <OrgSwitcher onNavigate={onNavigate} />
      </div>
      {canAnywhere("kb.chat") ? (
        <div className="px-3 pt-3">
          <Link
            href="/chat"
            onClick={onNavigate}
            className="flex h-9 items-center gap-2 rounded-lg border bg-surface px-3 text-[13px] font-medium shadow-xs transition-colors hover:border-border-strong"
          >
            <SquarePen className="h-4 w-4 text-accent" />
            Ask your knowledge
          </Link>
        </div>
      ) : null}
      <div className="flex-1 space-y-6 overflow-y-auto px-3 pb-5 pt-5">
        {NAV.map((section) => {
          const items = section.items.filter((item) => {
            if (item.platform) return Boolean(user?.is_platform_admin);
            if (item.anyOf) return item.anyOf.some((p) => canAnywhere(p));
            return !item.perm || canAnywhere(item.perm) || can(item.perm);
          });
          if (!items.length) return null;
          return (
            <div key={section.group}>
              <div className="label mb-1.5 px-2.5 !text-[10px]">{section.group}</div>
              <ul className="space-y-px">
                {items.map((item) => {
                  const active = pathname === item.href || pathname.startsWith(item.href + "/");
                  const Icon = item.icon;
                  return (
                    <li key={item.href}>
                      <Link
                        href={item.href}
                        onClick={onNavigate}
                        aria-current={active ? "page" : undefined}
                        className={cn(
                          "group relative flex h-8 items-center gap-2.5 rounded-lg px-2.5 text-[13.5px] font-medium transition-colors",
                          active
                            ? "bg-surface text-fg shadow-xs ring-1 ring-border"
                            : "text-muted hover:bg-black/[0.035] hover:text-fg dark:hover:bg-white/[0.045]",
                        )}
                      >
                        <Icon className={cn("h-4 w-4 shrink-0 transition-colors", active ? "text-accent" : "text-subtle group-hover:text-muted")} />
                        <span className="truncate">{item.label}</span>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </div>
          );
        })}
      </div>
      <div className="space-y-2 border-t px-2.5 pb-2.5 pt-2.5">
        <UserMenu />
        <div className="flex items-center gap-2 px-2.5 pb-0.5 pt-1">
          <LogoMark className="h-4 w-4" />
          <span className="truncate text-[11px] text-subtle">Knowledge Intelligence</span>
        </div>
      </div>
    </nav>
  );
}
