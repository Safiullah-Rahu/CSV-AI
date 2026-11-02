"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BookOpen, Download, History, PanelRight, SquarePen } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { Composer } from "@/components/chat/composer";
import { ChatHero } from "@/components/chat/hero";
import { ConversationList, HistoryDrawer, type ConvRow } from "@/components/chat/history";
import { MessageView } from "@/components/chat/message";
import { AdvancedControls, KBPicker } from "@/components/chat/scope-controls";
import { SourcePanel } from "@/components/chat/source-panel";
import type { ChatFilters, ChatMessage, Citation, Conversation, KB, RetrievalSettings } from "@/components/chat/types";
import { Button } from "@/components/ui/button";
import { ConfirmDialog, Dialog, DialogContent } from "@/components/ui/dialog";
import { Input } from "@/components/ui/form";
import { EmptyState, Skeleton, Tip } from "@/components/ui/primitives";
import { apiFetch, del, errorMessage, get, patch, post, streamSSE } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn } from "@/lib/utils";

const DEFAULT_RETRIEVAL: RetrievalSettings = { mode: "hybrid", top_k: 6, rerank: "lexical", mmr: true, query_expansion: "prf", compression: false };
const EMPTY_FILTERS: ChatFilters = { file_types: [], tags: [], source_types: [], created_from: "", created_to: "" };

function ChatInner() {
  const params = useSearchParams();
  const router = useRouter();
  const qc = useQueryClient();
  const { user, orgId, can } = useAuth();
  const [convId, setConvId] = useState<string | null>(params.get("c"));
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [kbIds, setKbIds] = useState<string[]>([]);
  const [filters, setFilters] = useState<ChatFilters>(EMPTY_FILTERS);
  const [retrieval, setRetrieval] = useState<RetrievalSettings>(DEFAULT_RETRIEVAL);
  const [policy, setPolicy] = useState("");
  const [providerId, setProviderId] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [panelMsgId, setPanelMsgId] = useState<string | null>(null);
  const [listQ, setListQ] = useState("");
  const [renaming, setRenaming] = useState<ConvRow | null>(null);
  const [deleting, setDeleting] = useState<ConvRow | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  const kbs = useApi<KB[]>("/knowledge-bases");
  const providers = useApi<{ id: string; name: string; default_model: string | null; enabled: boolean }[]>("/ai/providers");
  const list = useQuery({
    queryKey: ["conversations", orgId, listQ],
    queryFn: () => get<{ items: ConvRow[] }>("/conversations", { scope: "own", q: listQ || undefined, limit: 100 }),
    enabled: Boolean(orgId),
  });
  const conv = useQuery({
    queryKey: ["conversation", convId],
    queryFn: () => get<Conversation>(`/conversations/${convId}`),
    enabled: Boolean(convId) && !streaming,
  });

  useEffect(() => {
    if (conv.data && !streaming) {
      setMessages(conv.data.messages.filter((m) => m.status !== "superseded"));
      setKbIds(conv.data.kb_ids);
    }
  }, [conv.data, streaming]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: streaming ? "auto" : "smooth", block: "end" });
  }, [messages, streaming]);

  // `/chat?q=...` (e.g. from the Overview "Ask anything" bar): prefill and focus, never auto-send.
  const qParam = params.get("q");
  const cParam = params.get("c");
  useEffect(() => {
    if (qParam && !cParam) {
      setInput(qParam);
      requestAnimationFrame(() => {
        const el = inputRef.current;
        if (!el) return;
        el.focus();
        el.setSelectionRange(el.value.length, el.value.length);
      });
    }
  }, [qParam, cParam]);

  // Escape closes the sources sheet.
  useEffect(() => {
    if (!panelOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented) return;
      // Let open menus, popovers and dialogs take the first Escape.
      if (document.querySelector("[data-radix-popper-content-wrapper], [role=dialog]")) return;
      setPanelOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [panelOpen]);

  const lastAssistant = useMemo(() => [...messages].reverse().find((m) => m.role === "assistant"), [messages]);
  const latestSources = useMemo(() => sourcesOf(lastAssistant), [lastAssistant]);
  // The panel shows the sources of the answer a citation was clicked in; null (or a message that
  // no longer exists, e.g. after regenerate) means the latest answer.
  const panelMessage = useMemo(() => (panelMsgId ? messages.find((m) => m.id === panelMsgId) : undefined) ?? lastAssistant, [messages, panelMsgId, lastAssistant]);
  const panelSources = useMemo(() => sourcesOf(panelMessage), [panelMessage]);
  const panelIsLatest = !panelMessage || panelMessage === lastAssistant;

  // Header "Sources" button: toggles the panel for the latest answer.
  function toggleLatestSources() {
    if (panelOpen) {
      setPanelOpen(false);
      return;
    }
    setPanelMsgId(null);
    if (activeCitation && !latestSources.some((s) => s === activeCitation || (s.id && s.id === activeCitation.id))) setActiveCitation(null);
    setPanelOpen(true);
  }

  const openConversation = useCallback(
    (id: string | null) => {
      abortRef.current?.abort();
      setConvId(id);
      setActiveCitation(null);
      setPanelMsgId(null);
      if (!id) {
        setMessages([]);
        setKbIds([]);
      }
      router.replace(id ? `/chat?c=${id}` : "/chat");
    },
    [router],
  );

  async function send(question: string, regenerateId?: string) {
    const q = question.trim();
    if ((!q && !regenerateId) || streaming) return;
    setInput("");
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;
    const tempAssistant: ChatMessage = { id: `live-${Date.now()}`, role: "assistant", content: "", status: "streaming", citations: [], streaming: true, stage: "retrieving" };
    setMessages((prev) => {
      const base = regenerateId ? prev.filter((m) => m.id !== regenerateId) : [...prev, { id: `u-${Date.now()}`, role: "user" as const, content: q, status: "complete", citations: [] }];
      return [...base, tempAssistant];
    });
    const update = (fn: (m: ChatMessage) => ChatMessage) => setMessages((prev) => prev.map((m) => (m.id === tempAssistant.id ? fn(m) : m)));
    let newConvId = convId;
    try {
      await streamSSE("/chat/stream", {
        method: "POST",
        signal: controller.signal,
        body: {
          question: q || "regenerate",
          conversation_id: convId,
          kb_ids: kbIds.length ? kbIds : null,
          regenerate_message_id: regenerateId ?? null,
          provider_id: providerId || null,
          policy: policy || null,
          retrieval,
          filters: {
            file_types: filters.file_types.length ? filters.file_types : null,
            tags: filters.tags.length ? filters.tags : null,
            source_types: filters.source_types.length ? filters.source_types : null,
            created_from: filters.created_from ? new Date(filters.created_from).toISOString() : null,
            created_to: filters.created_to ? new Date(filters.created_to + "T23:59:59").toISOString() : null,
          },
        },
        onEvent: ({ event, data }) => {
          if (event === "meta") {
            newConvId = data.conversation_id;
            if (!convId) {
              setConvId(data.conversation_id);
              window.history.replaceState(null, "", `/chat?c=${data.conversation_id}`);
            }
          } else if (event === "status") update((m) => ({ ...m, stage: data.stage }));
          else if (event === "sources") update((m) => ({ ...m, liveSources: data.sources }));
          else if (event === "token") update((m) => ({ ...m, content: m.content + data.text }));
          else if (event === "done")
            update((m) => ({
              ...m,
              id: data.message_id,
              content: data.content,
              status: data.mode === "refused" ? "refused" : "complete",
              streaming: false,
              citations: data.citations,
              confidence: data.confidence,
              grounded: data.grounded,
              model: data.usage?.model,
              provider: data.usage?.provider,
              retrieval_ms: data.timings?.retrieval_ms,
              generation_ms: data.timings?.generation_ms,
              metadata: { mode: data.mode, follow_ups: data.follow_ups, unsupported_sentences: data.unsupported_sentences, invalid_citations: data.invalid_citations },
            }));
          else if (event === "error") update((m) => ({ ...m, streaming: false, status: "error", metadata: { ...m.metadata, error: data.message } }));
        },
      });
    } catch (err) {
      if (controller.signal.aborted) update((m) => ({ ...m, streaming: false, status: "stopped" }));
      else update((m) => ({ ...m, streaming: false, status: "error", metadata: { error: errorMessage(err) } }));
    } finally {
      setStreaming(false);
      abortRef.current = null;
      qc.invalidateQueries({ queryKey: ["conversations"] });
      if (newConvId) {
        await qc.invalidateQueries({ queryKey: ["conversation", newConvId] });
      }
      inputRef.current?.focus();
    }
  }

  const feedback = useMutation({ mutationFn: ({ id, rating }: { id: string; rating: 1 | -1 }) => post(`/messages/${id}/feedback`, { rating }), onSuccess: () => toast.success("Thanks for the feedback") });
  const rename = useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) => patch(`/conversations/${id}`, { title }),
    onSuccess: () => {
      setRenaming(null);
      qc.invalidateQueries({ queryKey: ["conversations"] });
      qc.invalidateQueries({ queryKey: ["conversation"] });
    },
  });
  const remove = useMutation({
    mutationFn: (id: string) => del(`/conversations/${id}`),
    onSuccess: (_d, id) => {
      setDeleting(null);
      qc.invalidateQueries({ queryKey: ["conversations"] });
      if (id === convId) openConversation(null);
      toast.success("Conversation deleted");
    },
  });

  async function exportConversation(id: string) {
    const res = await apiFetch(`/conversations/${id}/export?format=md`);
    if (!res.ok) return toast.error("Export failed");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `conversation-${id}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const kbList = kbs.data ?? [];
  const noKnowledge = kbs.isSuccess && kbList.length === 0;
  const canChat = can("kb.chat");
  const isNew = messages.length === 0 && !convId;
  const showHero = isNew && !noKnowledge;
  const title = convId ? (conv.data?.title ?? (messages.find((m) => m.role === "user")?.content || "")) : "";

  const historyProps = {
    rows: list.data?.items,
    loading: list.isLoading,
    activeId: convId,
    query: listQ,
    onQuery: setListQ,
    onRename: (row: ConvRow) => setRenaming(row),
    onExport: can("conv.export") ? (row: ConvRow) => exportConversation(row.id) : undefined,
    onDelete: can("conv.delete") ? (row: ConvRow) => setDeleting(row) : undefined,
  };

  const scope = (
    <>
      <KBPicker kbs={kbList} selected={kbIds} onChange={setKbIds} />
      <AdvancedControls filters={filters} setFilters={setFilters} retrieval={retrieval} setRetrieval={setRetrieval} policy={policy} setPolicy={setPolicy} providers={providers.data ?? []} providerId={providerId} setProviderId={setProviderId} />
    </>
  );

  const composer = (hero: boolean) => (
    <Composer
      value={input}
      onChange={setInput}
      onSubmit={() => send(input)}
      onStop={() => abortRef.current?.abort()}
      streaming={streaming}
      canChat={canChat}
      inputRef={inputRef}
      scope={scope}
      hero={hero}
      autoFocus
    />
  );

  return (
    <div className="flex h-full min-h-0">
      {/* Conversation history (collapses while sources are inspected on mid-size screens) */}
      <aside className={cn("hidden w-[264px] shrink-0 flex-col border-r bg-surface-2/35 md:flex", panelOpen && "xl:hidden 2xl:flex")} aria-label="Conversation history">
        <ConversationList {...historyProps} onOpen={(id) => openConversation(id)} onNew={() => openConversation(null)} />
      </aside>
      <HistoryDrawer
        {...historyProps}
        open={historyOpen}
        onOpenChange={setHistoryOpen}
        onOpen={(id) => {
          setHistoryOpen(false);
          openConversation(id);
        }}
        onNew={() => {
          setHistoryOpen(false);
          openConversation(null);
        }}
      />

      {/* Thread */}
      <section className="relative flex min-w-0 flex-1 flex-col">
        <div className="flex h-12 shrink-0 items-center gap-1.5 border-b px-3 sm:px-4">
          <Tip content="History">
            <button type="button" onClick={() => setHistoryOpen(true)} className="-ml-1 grid h-8 w-8 shrink-0 place-items-center rounded-lg text-muted transition-colors hover:bg-surface-2 hover:text-fg md:hidden" aria-label="Open conversation history">
              <History className="h-4 w-4" />
            </button>
          </Tip>
          <div className="min-w-0 flex-1">
            {convId ? (
              title ? (
                <h2 className="truncate text-[13.5px] font-medium tracking-[-0.01em]" title={title}>
                  {title}
                </h2>
              ) : (
                <Skeleton className="h-4 w-48" />
              )
            ) : (
              <h2 className="text-[13.5px] font-medium tracking-[-0.01em] text-muted">New chat</h2>
            )}
          </div>
          <div className="flex shrink-0 items-center gap-0.5">
            {convId && can("conv.export") ? (
              <Tip content="Export as Markdown">
                <button type="button" onClick={() => exportConversation(convId)} className="grid h-8 w-8 place-items-center rounded-lg text-muted transition-colors hover:bg-surface-2 hover:text-fg" aria-label="Export conversation">
                  <Download className="h-4 w-4" />
                </button>
              </Tip>
            ) : null}
            {!isNew ? (
              <Button size="sm" variant={panelOpen ? "secondary" : "ghost"} onClick={toggleLatestSources} disabled={!latestSources.length} aria-label="Toggle sources panel" aria-pressed={panelOpen}>
                <PanelRight className="h-4 w-4" />
                <span className="hidden sm:inline">Sources</span>
                {latestSources.length ? <span className="num font-mono text-[11px] text-subtle">{latestSources.length}</span> : null}
              </Button>
            ) : null}
            {!isNew ? (
              <Tip content="New chat">
                <button type="button" onClick={() => openConversation(null)} className="grid h-8 w-8 place-items-center rounded-lg text-muted transition-colors hover:bg-surface-2 hover:text-fg md:hidden" aria-label="New chat">
                  <SquarePen className="h-4 w-4" />
                </button>
              </Tip>
            ) : null}
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto">
          {showHero ? (
            <ChatHero kbs={kbList} loading={kbs.isLoading} composer={composer(true)} onAsk={(q) => send(q)} canChat={canChat} />
          ) : (
            <div className="mx-auto w-full max-w-[760px] px-4 pb-12 pt-8 sm:px-8 sm:pt-10">
              {isNew && noKnowledge ? (
                <EmptyState
                  icon={<BookOpen className="h-5 w-5" />}
                  title="No knowledge bases available to you"
                  description="Ask an administrator for access, or create a knowledge base and upload documents first."
                  action={
                    can("kb.create") ? (
                      <Link href="/knowledge-bases">
                        <Button>Create a knowledge base</Button>
                      </Link>
                    ) : undefined
                  }
                />
              ) : null}
              {conv.isLoading && !messages.length ? <ThreadSkeleton /> : null}
              {convId && conv.isError && !messages.length ? (
                <EmptyState
                  title="This conversation could not be loaded"
                  description={errorMessage(conv.error)}
                  action={
                    <Button variant="secondary" onClick={() => openConversation(null)}>
                      <SquarePen className="h-4 w-4" /> Start a new chat
                    </Button>
                  }
                />
              ) : null}
              {convId && conv.isSuccess && !messages.length && !streaming ? <p className="py-16 text-center text-[13px] text-muted">No messages yet. Ask your first question below.</p> : null}
              <div className="space-y-10">
                {messages.map((m, i) => (
                  <MessageView
                    key={m.id}
                    m={m}
                    userName={user?.full_name || user?.email}
                    last={i === messages.length - 1}
                    onCite={(c) => {
                      setActiveCitation(c);
                      setPanelMsgId(m.id);
                      setPanelOpen(true);
                    }}
                    onRegenerate={!streaming && m.role === "assistant" && !m.id.startsWith("live-") ? () => send("", m.id) : undefined}
                    onFeedback={m.role === "assistant" && !m.id.startsWith("live-") ? (rating) => feedback.mutate({ id: m.id, rating }) : undefined}
                    onFollowUp={(q) => send(q)}
                  />
                ))}
              </div>
            </div>
          )}
          {/* below the column padding so the latest turn clears the composer fade */}
          {!showHero ? <div ref={bottomRef} aria-hidden /> : null}
        </div>

        {!showHero ? (
          <div className="relative shrink-0 px-3 pb-3 sm:px-6">
            <div className="pointer-events-none absolute inset-x-0 -top-10 h-10 bg-gradient-to-t from-bg to-transparent" aria-hidden />
            <div className="mx-auto max-w-[760px] sm:px-2">
              {composer(false)}
              <p className="mt-2 text-center text-[11px] text-subtle">Answers can be wrong. Check the cited sources for anything important.</p>
            </div>
          </div>
        ) : null}
      </section>

      {/* Sources: a side column on wide screens, a sheet over the thread otherwise */}
      {panelOpen ? (
        <div className="fixed inset-0 z-40 xl:static xl:z-auto xl:w-[400px] xl:shrink-0">
          <div className="absolute inset-0 bg-[hsl(var(--shadow)/0.32)] backdrop-blur-[2px] animate-fade-in dark:bg-black/60 xl:hidden" onClick={() => setPanelOpen(false)} aria-hidden />
          <div className="absolute inset-y-0 right-0 w-[min(420px,100vw)] border-l shadow-float animate-fade-in xl:static xl:h-full xl:w-full xl:shadow-none">
            <SourcePanel sources={panelSources} active={activeCitation} onSelect={setActiveCitation} onClose={() => setPanelOpen(false)} earlier={!panelIsLatest} />
          </div>
        </div>
      ) : null}

      <Dialog open={Boolean(renaming)} onOpenChange={(v) => !v && setRenaming(null)}>
        {renaming ? <RenameForm row={renaming} onSave={(title) => rename.mutate({ id: renaming.id, title })} loading={rename.isPending} /> : null}
      </Dialog>
      <ConfirmDialog
        open={Boolean(deleting)}
        onOpenChange={(v) => !v && setDeleting(null)}
        title="Delete conversation?"
        description={`“${deleting?.title}” and its answers will be permanently deleted.`}
        confirmLabel="Delete"
        loading={remove.isPending}
        onConfirm={() => {
          if (deleting) remove.mutate(deleting.id);
        }}
      />
    </div>
  );
}

/** Final citations once an answer is done, otherwise the passages retrieved while streaming. */
function sourcesOf(m: ChatMessage | undefined): Citation[] {
  if (!m) return [];
  return m.citations?.length ? m.citations : m.liveSources ?? [];
}

function ThreadSkeleton() {
  return (
    <div className="space-y-10" aria-label="Loading conversation">
      <div className="flex justify-end">
        <Skeleton className="h-11 w-2/5 rounded-[18px]" />
      </div>
      <div className="flex gap-4">
        <Skeleton className="h-7 w-7 shrink-0 rounded-[9px]" />
        <div className="flex-1 space-y-2.5 pt-1">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-11/12" />
          <Skeleton className="h-4 w-3/5" />
        </div>
      </div>
    </div>
  );
}

function RenameForm({ row, onSave, loading }: { row: ConvRow; onSave: (title: string) => void; loading: boolean }) {
  const [title, setTitle] = useState(row.title);
  return (
    <DialogContent title="Rename conversation" footer={<Button onClick={() => onSave(title)} loading={loading} disabled={!title.trim()}>Save</Button>}>
      <Input value={title} onChange={(e) => setTitle(e.target.value)} autoFocus aria-label="Conversation title" />
    </DialogContent>
  );
}

export default function ChatPage() {
  return (
    <Suspense>
      <ChatInner />
    </Suspense>
  );
}
