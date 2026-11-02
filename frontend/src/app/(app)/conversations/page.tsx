"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Archive, ArchiveRestore, Bot, Download, FileJson, FileText, Library, MessageSquareText, MessagesSquare, Pencil, Share2, Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { toast } from "sonner";

import { RenameConversationDialog, ShareConversationDialog } from "@/components/org/conversation-dialogs";
import { memberLabel, useIndex, useMembers } from "@/components/org/data";
import { downloadFile, Pager, PersonCell, RowMenuButton, SearchBox, Segmented, useDebounced, When } from "@/components/org/kit";
import type { Conversation, Page as PageOf } from "@/components/org/types";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/dialog";
import { Badge, Card, EmptyState, Menu, MenuContent, MenuItem, MenuLabel, MenuSeparator, MenuTrigger, Page, PageHeader, SkeletonRows } from "@/components/ui/primitives";
import { del, patch } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatNumber, pluralize } from "@/lib/utils";

type Scope = "own" | "shared" | "team" | "all";
const LIMIT = 50;

export default function ConversationsPage() {
  const { can, user } = useAuth();
  const router = useRouter();
  const qc = useQueryClient();
  const [scope, setScope] = useState<Scope>("own");
  const [archived, setArchived] = useState(false);
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);
  const query = useDebounced(q.trim(), 300);
  const [renaming, setRenaming] = useState<Conversation | null>(null);
  const [sharing, setSharing] = useState<Conversation | null>(null);
  const [deleting, setDeleting] = useState<Conversation | null>(null);

  const list = useApi<PageOf<Conversation>>(can("conv.view_own") ? "/conversations" : null, { q: query || undefined, scope, archived, limit: LIMIT, offset });
  const members = useMembers(scope !== "own");
  const memberById = useIndex(members.data?.items, (m) => m.user_id);

  const scopes: { value: Scope; label: string }[] = [
    { value: "own", label: "Mine" },
    { value: "shared", label: "Shared with me" },
    ...(can("conv.view_team") ? [{ value: "team" as const, label: "My teams" }] : []),
    ...(can("conv.view_authorized") ? [{ value: "all" as const, label: "All" }] : []),
  ];

  const invalidate = () => qc.invalidateQueries({ queryKey: ["/conversations"] });
  const archive = useMutation({
    mutationFn: (c: Conversation) => patch(`/conversations/${c.id}`, { archived: !c.archived }),
    onSuccess: (_r, c) => {
      toast.success(c.archived ? "Conversation restored" : "Conversation archived");
      invalidate();
    },
  });
  const remove = useMutation({
    mutationFn: (c: Conversation) => del(`/conversations/${c.id}`),
    onSuccess: () => {
      toast.success("Conversation deleted");
      setDeleting(null);
      invalidate();
    },
  });
  const exporter = useMutation({
    mutationFn: ({ c, format }: { c: Conversation; format: "md" | "json" }) => downloadFile(`/conversations/${c.id}/export`, { format }, `${slug(c.title) || "conversation"}-${c.id.slice(0, 8)}.${format}`),
    onSuccess: (name) => toast.success(`Downloaded ${name}`),
  });

  const rows = list.data?.items ?? [];
  const canDelete = (c: Conversation) => can("conv.delete") && (c.own || can("conv.view_authorized"));

  return (
    <Page>
      <PageHeader
        eyebrow="Workspace"
        title="Conversations"
        description="Your AI chat history, conversations shared with you and — with the right permissions — your teams' conversations. Sources stay permission-checked when a conversation is reopened."
        actions={
          can("kb.chat") ? (
            <Button onClick={() => router.push("/chat")}>
              <MessageSquareText className="h-4 w-4" /> New chat
            </Button>
          ) : null
        }
      />

      <Card className="overflow-hidden">
        <div className="flex flex-wrap items-center gap-2 border-b px-3.5 py-3">
          <Segmented
            label="Conversation scope"
            value={scope}
            onChange={(v) => {
              setScope(v);
              setOffset(0);
            }}
            options={scopes}
          />
          <Segmented
            label="Archive filter"
            value={archived ? "archived" : "active"}
            onChange={(v) => {
              setArchived(v === "archived");
              setOffset(0);
            }}
            options={[
              { value: "active", label: "Active" },
              { value: "archived", label: <><Archive className="h-3.5 w-3.5" /> Archived</> },
            ]}
          />
          <SearchBox
            value={q}
            onChange={(v) => {
              setQ(v);
              setOffset(0);
            }}
            placeholder="Search titles and messages…"
            label="Search conversations"
            className="sm:ml-auto sm:w-72"
          />
        </div>

        {list.isLoading ? (
          <SkeletonRows rows={8} />
        ) : rows.length === 0 ? (
          <EmptyState
            icon={<MessagesSquare className="h-5 w-5" />}
            title={query ? "No conversations match your search" : archived ? "No archived conversations" : scope === "shared" ? "Nothing has been shared with you yet" : scope === "own" ? "No conversations yet" : "No conversations in this scope"}
            description={query ? "Search looks at titles and message text." : scope === "own" && !archived ? "Ask a question in AI Chat and the conversation will appear here." : undefined}
            action={scope === "own" && !archived && !query && can("kb.chat") ? <Button onClick={() => router.push("/chat")}>Start a conversation</Button> : undefined}
          />
        ) : (
          <div className={cn("relative overflow-x-auto transition-opacity", list.isFetching && !list.isLoading && "opacity-70")}>
            <table className={cn("w-full text-[13px]", scope !== "own" ? "min-w-[620px] sm:min-w-[760px]" : "sm:min-w-[640px]")}>
              <thead>
                <tr className="border-b bg-surface-2/50 text-left font-mono text-[10.5px] font-medium uppercase tracking-[0.1em] text-subtle">
                  <th className="h-9 px-4 font-medium">Conversation</th>
                  {scope !== "own" ? <th className="h-9 px-4 font-medium">Owner</th> : null}
                  <th className="hidden h-9 px-4 text-right font-medium sm:table-cell">Messages</th>
                  <th className="hidden h-9 px-4 font-medium sm:table-cell">Updated</th>
                  <th className="h-9 w-12 px-4">
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((c) => {
                  const owner = memberById.get(c.user_id);
                  return (
                    <tr key={c.id} className="group/row border-b transition-colors last:border-0 hover:bg-surface-2/50">
                      <td className="max-w-[250px] px-4 py-3 sm:max-w-[480px]">
                        <div className="flex min-w-0 items-start gap-3">
                          <span className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-lg border bg-surface-2/60 text-subtle transition-colors group-hover/row:border-accent/25 group-hover/row:text-accent">
                            {c.agent_id ? <Bot className="h-4 w-4" /> : <MessagesSquare className="h-4 w-4" />}
                          </span>
                          <div className="min-w-0 flex-1">
                        <Link href={`/chat?c=${c.id}`} className="block truncate font-medium text-fg underline-offset-4 hover:text-accent hover:underline">
                          {c.title || "Untitled conversation"}
                        </Link>
                        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px] text-subtle">
                          {c.kb_ids.length ? (
                            <span className="inline-flex items-center gap-1">
                              <Library className="h-3 w-3" /> {pluralize(c.kb_ids.length, "knowledge base")}
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1">
                              <Library className="h-3 w-3" /> All accessible knowledge
                            </span>
                          )}
                          {c.agent_id ? (
                            <span className="inline-flex items-center gap-1">
                              <Bot className="h-3 w-3" /> Agent
                            </span>
                          ) : null}
                          {c.archived ? <Badge>archived</Badge> : null}
                          {!c.own && scope === "own" ? <Badge tone="info">shared</Badge> : null}
                          <span className="whitespace-nowrap sm:hidden">
                            {pluralize(c.message_count, "message")} · <When iso={c.updated_at} />
                          </span>
                        </div>
                          </div>
                        </div>
                      </td>
                      {scope !== "own" ? (
                        <td className="px-4 py-3">
                          {c.own ? <PersonCell name={user?.full_name} email={user?.email} you /> : owner ? <PersonCell name={owner.full_name} email={owner.email} /> : <span className="text-subtle">{memberLabel(undefined, "Member")}</span>}
                        </td>
                      ) : null}
                      <td className="num hidden px-4 py-3 text-right font-mono text-[12.5px] text-muted sm:table-cell">{formatNumber(c.message_count)}</td>
                      <td className="hidden px-4 py-3 text-muted sm:table-cell">
                        <When iso={c.updated_at} />
                      </td>
                      <td className="w-12 px-3 py-3 text-right sm:px-4">
                        <Menu>
                          <MenuTrigger asChild>
                            <RowMenuButton label={`Actions for ${c.title}`} />
                          </MenuTrigger>
                          <MenuContent className="w-56">
                            <MenuItem onSelect={() => router.push(`/chat?c=${c.id}`)}>
                              <MessageSquareText className="h-3.5 w-3.5" /> Open in chat
                            </MenuItem>
                            {c.own ? (
                              <>
                                <MenuItem onSelect={() => setRenaming(c)}>
                                  <Pencil className="h-3.5 w-3.5" /> Rename
                                </MenuItem>
                                <MenuItem onSelect={() => archive.mutate(c)}>
                                  {c.archived ? <ArchiveRestore className="h-3.5 w-3.5" /> : <Archive className="h-3.5 w-3.5" />}
                                  {c.archived ? "Unarchive" : "Archive"}
                                </MenuItem>
                                <MenuItem disabled={!can("conv.share")} onSelect={() => setSharing(c)}>
                                  <Share2 className="h-3.5 w-3.5" /> Share…
                                  {!can("conv.share") ? <span className="ml-auto font-mono text-[10px] text-subtle">conv.share</span> : null}
                                </MenuItem>
                              </>
                            ) : null}
                            <MenuSeparator />
                            <MenuLabel>
                              <span className="inline-flex items-center gap-1">
                                <Download className="h-3 w-3" /> Export
                              </span>
                            </MenuLabel>
                            <MenuItem disabled={!can("conv.export")} onSelect={() => exporter.mutate({ c, format: "md" })}>
                              <FileText className="h-3.5 w-3.5" /> Markdown (.md)
                              {!can("conv.export") ? <span className="ml-auto font-mono text-[10px] text-subtle">conv.export</span> : null}
                            </MenuItem>
                            <MenuItem disabled={!can("conv.export")} onSelect={() => exporter.mutate({ c, format: "json" })}>
                              <FileJson className="h-3.5 w-3.5" /> JSON (.json)
                            </MenuItem>
                            {canDelete(c) ? (
                              <>
                                <MenuSeparator />
                                <MenuItem danger onSelect={() => setDeleting(c)}>
                                  <Trash2 className="h-3.5 w-3.5" /> Delete
                                </MenuItem>
                              </>
                            ) : null}
                          </MenuContent>
                        </Menu>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        {list.data ? <Pager total={list.data.total} limit={LIMIT} offset={offset} onChange={setOffset} /> : null}
      </Card>
      {list.data ? (
        <p className="mt-3 flex items-center gap-2 text-[12px] text-subtle">
          <span className="num font-medium text-muted">{pluralize(list.data.total, "conversation")}</span>
          <span aria-hidden>·</span> sorted by most recent activity
        </p>
      ) : null}

      <RenameConversationDialog conv={renaming} onClose={() => setRenaming(null)} />
      <ShareConversationDialog conv={sharing} onClose={() => setSharing(null)} />
      <ConfirmDialog
        open={Boolean(deleting)}
        onOpenChange={(v) => !v && setDeleting(null)}
        title="Delete conversation?"
        description={deleting ? `“${deleting.title}” and all of its messages, citations and feedback will be permanently deleted. Anyone it was shared with loses access.` : undefined}
        confirmLabel="Delete conversation"
        loading={remove.isPending}
        onConfirm={() => { if (deleting) remove.mutate(deleting); }}
      />
    </Page>
  );
}

function slug(s: string) {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60);
}
