"use client";

import * as DialogPrimitive from "@radix-ui/react-dialog";
import { ArrowUpRight, Download, MoreHorizontal, Pencil, Search, SquarePen, Trash2, X } from "lucide-react";
import Link from "next/link";

import { groupByRecency, shortAgo } from "@/components/chat/utils";
import { Menu, MenuContent, MenuItem, MenuSeparator, MenuTrigger, Skeleton } from "@/components/ui/primitives";
import { cn, formatDate } from "@/lib/utils";

export type ConvRow = { id: string; title: string; updated_at: string; message_count: number };

type ListProps = {
  rows: ConvRow[] | undefined;
  loading: boolean;
  activeId: string | null;
  query: string;
  onQuery: (q: string) => void;
  onOpen: (id: string) => void;
  onNew: () => void;
  onRename: (row: ConvRow) => void;
  onExport?: (row: ConvRow) => void;
  onDelete?: (row: ConvRow) => void;
  /** Touch layout (drawer): row actions always visible. */
  touch?: boolean;
};

/** Conversation history: new chat, search, then rows grouped by recency. */
export function ConversationList({ rows, loading, activeId, query, onQuery, onOpen, onNew, onRename, onExport, onDelete, touch }: ListProps) {
  const groups = groupByRecency(rows ?? []);
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="space-y-2 p-3">
        <button
          type="button"
          onClick={onNew}
          className="flex h-9 w-full items-center gap-2 rounded-lg border bg-surface px-3 text-[13px] font-medium shadow-xs transition-colors hover:border-border-strong hover:bg-surface-2/60 focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/30"
        >
          <SquarePen className="h-4 w-4 text-accent" />
          New chat
        </button>
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-subtle" />
          <input
            value={query}
            onChange={(e) => onQuery(e.target.value)}
            placeholder="Search conversations"
            aria-label="Search conversations"
            className="h-8 w-full rounded-lg bg-surface/60 pl-8 pr-2.5 text-[13px] text-fg ring-1 ring-inset ring-border transition-[box-shadow,background] placeholder:text-subtle hover:ring-border-strong focus:bg-surface focus:outline-none focus:ring-accent/45"
          />
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {loading ? (
          <div className="space-y-1.5 px-1 pt-2">
            {Array.from({ length: 7 }).map((_, i) => (
              <Skeleton key={i} className={cn("h-[46px]", i % 3 === 1 && "w-11/12")} />
            ))}
          </div>
        ) : !rows?.length ? (
          <div className="px-3 py-10 text-center">
            <p className="text-[12.5px] font-medium text-muted">{query ? "No matching conversations" : "No conversations yet"}</p>
            <p className="mt-1 text-[12px] leading-relaxed text-subtle">{query ? "Try a different word from the title or an answer." : "Questions you ask appear here."}</p>
          </div>
        ) : (
          groups.map((g) => (
            <div key={g.label} className="pt-2">
              <div className="label px-2.5 pb-1.5 pt-2">{g.label}</div>
              <ul className="space-y-px">
                {g.items.map((c) => {
                  const active = c.id === activeId;
                  return (
                    <li key={c.id} data-conv-row className={cn("group relative flex items-center rounded-lg transition-colors", active ? "bg-surface shadow-xs ring-1 ring-border" : "hover:bg-surface-2/80")}>
                      {active ? <span className="absolute inset-y-2 left-0 w-[2px] rounded-full bg-accent" aria-hidden /> : null}
                      <button
                        type="button"
                        onClick={() => onOpen(c.id)}
                        aria-current={active ? "page" : undefined}
                        className="min-w-0 flex-1 rounded-lg px-2.5 py-2 text-left focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-inset focus-visible:ring-ring/30"
                      >
                        <span className={cn("block truncate text-[13px] leading-5", active ? "font-medium text-fg" : "text-fg/90")}>{c.title}</span>
                        <span className="mt-0.5 flex items-center gap-1.5 font-mono text-[10.5px] leading-4 text-subtle">
                          <time dateTime={c.updated_at} title={formatDate(c.updated_at)}>
                            {shortAgo(c.updated_at)}
                          </time>
                          <span className="text-border-strong">/</span>
                          <span className="num">{c.message_count} msg</span>
                        </span>
                      </button>
                      <Menu>
                        <MenuTrigger asChild>
                          <button
                            type="button"
                            className={cn(
                              "mr-1 grid h-7 w-7 shrink-0 place-items-center rounded-md text-subtle transition-[opacity,background,color] hover:bg-surface-2 hover:text-fg focus-visible:opacity-100 data-[state=open]:bg-surface-2 data-[state=open]:text-fg data-[state=open]:opacity-100",
                              touch ? "opacity-100" : "opacity-0 group-hover:opacity-100 [@media(hover:none)]:opacity-100",
                            )}
                            aria-label={`Actions for ${c.title}`}
                          >
                            <MoreHorizontal className="h-4 w-4" />
                          </button>
                        </MenuTrigger>
                        <MenuContent align="start" className="min-w-[180px]">
                          <MenuItem onSelect={() => onRename(c)}>
                            <Pencil className="h-3.5 w-3.5" /> Rename
                          </MenuItem>
                          {onExport ? (
                            <MenuItem onSelect={() => onExport(c)}>
                              <Download className="h-3.5 w-3.5" /> Export Markdown
                            </MenuItem>
                          ) : null}
                          {onDelete ? (
                            <>
                              <MenuSeparator />
                              <MenuItem danger onSelect={() => onDelete(c)}>
                                <Trash2 className="h-3.5 w-3.5" /> Delete
                              </MenuItem>
                            </>
                          ) : null}
                        </MenuContent>
                      </Menu>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))
        )}
      </div>

      <Link href="/conversations" className="group flex items-center justify-between border-t px-4 py-3 text-[12.5px] text-muted transition-colors hover:text-fg">
        All conversations
        <ArrowUpRight className="h-3.5 w-3.5 text-subtle transition-transform group-hover:-translate-y-px group-hover:translate-x-px group-hover:text-fg" />
      </Link>
    </div>
  );
}

/** The same list in a left drawer for narrow screens. */
export function HistoryDrawer({ open, onOpenChange, ...props }: ListProps & { open: boolean; onOpenChange: (v: boolean) => void }) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-[hsl(var(--shadow)/0.32)] backdrop-blur-[2px] animate-fade-in dark:bg-black/60" />
        <DialogPrimitive.Content className="fixed inset-y-0 left-0 z-50 flex w-[min(320px,88vw)] flex-col border-r bg-bg shadow-float animate-fade-in focus:outline-none">
          <div className="flex h-12 shrink-0 items-center justify-between border-b px-4">
            <DialogPrimitive.Title className="text-[13.5px] font-semibold tracking-[-0.01em]">History</DialogPrimitive.Title>
            <DialogPrimitive.Description className="sr-only">Your recent conversations</DialogPrimitive.Description>
            <DialogPrimitive.Close className="-mr-1.5 rounded-lg p-1.5 text-subtle transition-colors hover:bg-surface-2 hover:text-fg" aria-label="Close history">
              <X className="h-4 w-4" />
            </DialogPrimitive.Close>
          </div>
          <div className="min-h-0 flex-1">
            <ConversationList {...props} touch />
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
