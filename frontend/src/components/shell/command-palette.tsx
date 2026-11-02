"use client";

import * as DialogPrimitive from "@radix-ui/react-dialog";
import { useQuery } from "@tanstack/react-query";
import { CornerDownLeft, FileText, Library, Loader2, MessagesSquare, Network, Search, Users } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

import { NAV } from "@/components/shell/nav";
import { Kbd } from "@/components/ui/primitives";
import { get } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/utils";

type Hit = { id: string; title: string; snippet?: string; link: string; relevance?: number | null };
type SearchResult = { query: string; results: Record<string, Hit[]> };

const GROUPS: Record<string, { label: string; icon: React.ComponentType<{ className?: string }> }> = {
  knowledge_bases: { label: "Knowledge bases", icon: Library },
  documents: { label: "Documents", icon: FileText },
  conversations: { label: "Conversations", icon: MessagesSquare },
  connections: { label: "Connections", icon: Network },
  teams: { label: "Teams", icon: Users },
  workspaces: { label: "Workspaces", icon: Users },
};

export function CommandPalette({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  const router = useRouter();
  const { can, canAnywhere, user } = useAuth();
  const [q, setQ] = useState("");
  const [debounced, setDebounced] = useState("");
  const [active, setActive] = useState(0);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        onOpenChange(!open);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onOpenChange]);

  useEffect(() => {
    const t = setTimeout(() => setDebounced(q.trim()), 220);
    return () => clearTimeout(t);
  }, [q]);

  const { data, isFetching } = useQuery({
    queryKey: ["search", debounced],
    queryFn: () => get<SearchResult>("/search", { q: debounced, mode: "semantic" }),
    enabled: open && debounced.length >= 2,
    meta: { silent: true },
  });

  const pages = useMemo(
    () =>
      NAV.flatMap((s) => s.items)
        .filter((i) => (i.platform ? user?.is_platform_admin : i.anyOf ? i.anyOf.some(canAnywhere) : !i.perm || canAnywhere(i.perm) || can(i.perm)))
        .filter((i) => !q || i.label.toLowerCase().includes(q.toLowerCase()))
        .map((i) => ({ id: i.href, title: i.label, link: i.href, group: "Pages" })),
    [q, can, canAnywhere, user],
  );

  const flat = useMemo(() => {
    const hits: (Hit & { group: string })[] = [...pages];
    for (const [group, items] of Object.entries(data?.results ?? {})) {
      for (const h of items) hits.push({ ...h, group });
    }
    return hits;
  }, [pages, data]);

  const go = (link: string) => {
    onOpenChange(false);
    setQ("");
    router.push(link);
  };

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-[hsl(var(--shadow)/0.3)] backdrop-blur-[3px] animate-fade-in dark:bg-black/60" />
        <DialogPrimitive.Content className="fixed left-1/2 top-[12vh] z-50 w-[calc(100vw-32px)] max-w-[640px] -translate-x-1/2 overflow-hidden rounded-2xl border bg-surface shadow-float animate-scale-in">
          <DialogPrimitive.Title className="sr-only">Search</DialogPrimitive.Title>
          <DialogPrimitive.Description className="sr-only">Search pages, knowledge bases, documents and conversations you can access.</DialogPrimitive.Description>
          <div className="flex items-center gap-3 border-b px-5">
            <Search className="h-[18px] w-[18px] text-accent" />
            <input
              autoFocus
              value={q}
              onChange={(e) => {
                setQ(e.target.value);
                setActive(0);
              }}
              onKeyDown={(e) => {
                if (e.key === "ArrowDown") {
                  e.preventDefault();
                  setActive((a) => Math.min(a + 1, flat.length - 1));
                } else if (e.key === "ArrowUp") {
                  e.preventDefault();
                  setActive((a) => Math.max(a - 1, 0));
                } else if (e.key === "Enter" && flat[active]) {
                  go(flat[active].link);
                }
              }}
              placeholder="Search pages, knowledge, documents, conversations…"
              className="h-14 flex-1 bg-transparent text-[15px] outline-none placeholder:text-subtle"
              aria-label="Search"
            />
            {isFetching ? <Loader2 className="h-4 w-4 animate-spin text-subtle" /> : null}
          </div>
          <div className="max-h-[52vh] overflow-y-auto p-2">
            {flat.length === 0 ? (
              <p className="px-3 py-10 text-center text-[13px] text-muted">{debounced.length >= 2 ? "No results you have access to." : "Type to search."}</p>
            ) : (
              Object.entries(
                flat.reduce<Record<string, (Hit & { group: string; index: number })[]>>((acc, h, index) => {
                  (acc[h.group] ||= []).push({ ...h, index });
                  return acc;
                }, {}),
              ).map(([group, items]) => {
                const meta = GROUPS[group];
                const Icon = meta?.icon || CornerDownLeft;
                return (
                  <div key={group} className="mb-1">
                    <div className="label px-2.5 pb-1.5 pt-2">{meta?.label || group}</div>
                    {items.map((h) => (
                      <button
                        key={`${group}-${h.id}`}
                        onMouseEnter={() => setActive(h.index)}
                        onClick={() => go(h.link)}
                        className={cn("flex w-full items-center gap-3 rounded-[10px] px-2.5 py-2 text-left transition-colors", active === h.index ? "bg-surface-2" : "")}
                      >
                        <span className={cn("grid h-8 w-8 shrink-0 place-items-center rounded-lg border bg-surface shadow-xs", active === h.index ? "text-accent" : "text-subtle")}>
                          <Icon className="h-4 w-4" />
                        </span>
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-[13px] font-medium">{h.title}</span>
                          {h.snippet ? <span className="mt-0.5 line-clamp-1 block text-xs text-muted">{h.snippet}</span> : null}
                        </span>
                        {typeof h.relevance === "number" ? <span className="num font-mono text-[11px] text-subtle">{Math.round(h.relevance * 100)}%</span> : null}
                        {active === h.index ? <CornerDownLeft className="h-3.5 w-3.5 shrink-0 text-subtle" /> : null}
                      </button>
                    ))}
                  </div>
                );
              })
            )}
          </div>
          <div className="flex items-center gap-4 border-t bg-surface-2/40 px-5 py-2.5 text-[11.5px] text-subtle">
            <span className="flex items-center gap-1.5"><Kbd>↑</Kbd><Kbd>↓</Kbd> navigate</span>
            <span className="flex items-center gap-1.5"><Kbd>↵</Kbd> open</span>
            <span className="hidden items-center gap-1.5 sm:flex"><Kbd>esc</Kbd> close</span>
            <span className="ml-auto">Results respect your permissions</span>
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
