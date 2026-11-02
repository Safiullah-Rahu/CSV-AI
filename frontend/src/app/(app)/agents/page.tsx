"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Bot, ChevronDown, Cpu, Database, Footprints, Library, Lock, Pencil, Plus, Search, ShieldCheck, Trash2, Wrench, Zap } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useMemo, useState } from "react";
import { toast } from "sonner";

import { AgentFormDialog, useChatProviders } from "@/components/connectivity/agent-form";
import { AgentRunPanel } from "@/components/connectivity/agent-run";
import { GatedButton, Notice, invalidatePrefix, useKnowledgeBases } from "@/components/connectivity/shared";
import type { Agent, AgentTool, DbConnection } from "@/components/connectivity/types";
import { ConfirmDialog } from "@/components/ui/dialog";
import { Input } from "@/components/ui/form";
import { Badge, Card, Dot, EmptyState, Page, PageHeader, Skeleton } from "@/components/ui/primitives";
import { del } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatDate } from "@/lib/utils";

export default function AgentsPage() {
  return (
    <Suspense fallback={null}>
      <Agents />
    </Suspense>
  );
}

function Agents() {
  const { can } = useAuth();
  const router = useRouter();
  const params = useSearchParams();
  const qc = useQueryClient();
  const agents = useApi<Agent[]>("/agents");
  const providers = useChatProviders();
  const kbs = useKnowledgeBases();
  const conns = useApi<DbConnection[]>(can("db.view") ? "/database-connections" : null, undefined, { staleTime: 60_000, meta: { silent: true } });
  const toolInfo = useApi<AgentTool[]>("/agents/tools", undefined, { staleTime: 5 * 60_000, meta: { silent: true } });
  const [q, setQ] = useState("");
  const [creating, setCreating] = useState(false);
  const [editing, setEditing] = useState<Agent | null>(null);
  const [deleting, setDeleting] = useState<Agent | null>(null);

  const list = agents.data ?? [];
  const filtered = useMemo(() => list.filter((a) => !q || `${a.name} ${a.description} ${a.tools.join(" ")}`.toLowerCase().includes(q.toLowerCase())), [list, q]);
  const selectedId = params.get("agent");
  const selected = list.find((a) => a.id === selectedId) ?? filtered[0] ?? null;
  const select = (id: string) => router.replace(`/agents?agent=${id}`, { scroll: false });
  const canManage = can("agents.manage");
  const platform = useApi<{ available: boolean }>("/ai/platform-default", undefined, { staleTime: 5 * 60_000, meta: { silent: true } });
  // Agents run on an enabled org provider or, failing that, the deployment's platform default model.
  const noModel = !providers.isLoading && !platform.isLoading && !providers.chat.some((p) => p.enabled) && !platform.data?.available;

  const remove = useMutation({
    mutationFn: (a: Agent) => del(`/agents/${a.id}`),
    onSuccess: (_r, a) => {
      toast.success(`Deleted “${a.name}”`);
      invalidatePrefix(qc, "/agents");
      setDeleting(null);
      router.replace("/agents", { scroll: false });
    },
  });

  const kbName = (id: string) => kbs.data?.find((k) => k.id === id)?.name;
  const connName = (id: string) => conns.data?.find((c) => c.id === id)?.name;
  const providerOf = (a: Agent) => (a.provider_id ? providers.all.find((p) => p.id === a.provider_id) : null);
  const toolDescription = (key: string) => toolInfo.data?.find((t) => t.key === key)?.description;
  /** First sentence only for the summary row; the full text (with the input schema) stays in the tooltip. */
  const firstSentence = (text: string) => text.split(/(?<=\.)\s+(?=[A-Z])/)[0];

  return (
    <Page>
      <PageHeader
        eyebrow="Workspace"
        title="Agents & Workflows"
        description="Agents combine knowledge bases, live databases and tools behind one set of instructions. Every tool call runs with the asking user's permissions."
        actions={
          <GatedButton allowed={canManage} perm="agents.manage" onClick={() => setCreating(true)}>
            <Plus className="h-4 w-4" /> New agent
          </GatedButton>
        }
      />

      {noModel ? (
        <Notice
          tone="warn"
          className="mb-6"
          title="No chat model configured"
          action={
            <Link href="/settings?tab=ai" className="whitespace-nowrap text-xs font-medium text-accent hover:underline">
              Open AI settings
            </Link>
          }
        >
          Agents need a language model to plan and call tools. You can configure agents now; runs will fail until a model is connected{can("admin.ai_providers") ? "" : " by an administrator"}.
        </Notice>
      ) : null}

      {agents.isLoading ? (
        <div className="grid gap-6 lg:grid-cols-[300px_minmax(0,1fr)]">
          <Skeleton className="h-[360px] rounded-xl" />
          <Skeleton className="h-[480px] rounded-xl" />
        </div>
      ) : !list.length ? (
        <Card>
          <EmptyState
            icon={<Bot className="h-5 w-5" />}
            title="No agents yet"
            description="Create an agent that searches your knowledge bases, queries a database read-only, summarises documents or builds charts — with approval gates on the tools you choose."
            action={
              <GatedButton allowed={canManage} perm="agents.manage" onClick={() => setCreating(true)}>
                <Plus className="h-4 w-4" /> Create your first agent
              </GatedButton>
            }
          />
        </Card>
      ) : (
        <div className="grid items-start gap-6 lg:grid-cols-[300px_minmax(0,1fr)]">
          {/* ---------------------------------------------------- agent list */}
          <Card className="overflow-hidden lg:sticky lg:top-6">
            <div className="border-b p-3">
              <div className="relative">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-subtle" />
                <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search agents…" className="h-8 bg-surface-2/50 pl-8 text-[13px] shadow-none focus:bg-surface" aria-label="Search agents" />
              </div>
            </div>
            <div className="flex items-center justify-between border-b bg-surface-2/40 px-4 py-2">
              <span className="label">Agents</span>
              <span className="num font-mono text-[11px] text-subtle">
                {filtered.length}
                {q ? ` of ${list.length}` : ""}
              </span>
            </div>
            <ul className="max-h-[560px] divide-y overflow-y-auto" aria-label="Agents">
              {filtered.map((a) => {
                const active = selected?.id === a.id;
                return (
                  <li key={a.id}>
                    <button
                      onClick={() => select(a.id)}
                      aria-current={active}
                      className={cn("w-full px-4 py-3.5 text-left transition-colors", active ? "bg-accent/[0.06] shadow-[inset_2px_0_0_hsl(var(--accent))]" : "hover:bg-surface-2/60")}
                    >
                      <div className="flex items-center gap-2.5">
                        <span className={cn("grid h-7 w-7 shrink-0 place-items-center rounded-lg border", active ? "border-accent/30 bg-accent/10 text-accent" : "bg-surface-2 text-muted")}>
                          <Bot className="h-3.5 w-3.5" />
                        </span>
                        <span className="min-w-0 flex-1 truncate text-[13.5px] font-medium tracking-[-0.01em]">{a.name}</span>
                        {a.enabled ? <Dot tone="ok" /> : <Badge className="px-1.5 py-[1px] text-[10.5px]">disabled</Badge>}
                      </div>
                      {a.description ? <p className="mt-1.5 line-clamp-2 pl-[38px] text-[12px] leading-relaxed text-muted">{a.description}</p> : null}
                      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 pl-[38px] font-mono text-[11px] text-subtle">
                        <span className="inline-flex items-center gap-1">
                          <Wrench className="h-3 w-3" /> {a.tools.length}
                        </span>
                        <span className="inline-flex items-center gap-1">
                          <Library className="h-3 w-3" /> {a.kb_ids.length ? a.kb_ids.length : "all"}
                        </span>
                        {a.connection_ids.length ? (
                          <span className="inline-flex items-center gap-1">
                            <Database className="h-3 w-3" /> {a.connection_ids.length}
                          </span>
                        ) : null}
                        {a.approval_tools.length ? (
                          <span className="inline-flex items-center gap-1 text-warn">
                            <Lock className="h-3 w-3" /> {a.approval_tools.length} gated
                          </span>
                        ) : null}
                      </div>
                    </button>
                  </li>
                );
              })}
              {!filtered.length ? <li className="px-4 py-8 text-center text-[13px] text-muted">No agents match.</li> : null}
            </ul>
          </Card>

          {/* ---------------------------------------------------- detail */}
          {selected ? (
            <div className="min-w-0 space-y-6">
              <Card className="overflow-hidden">
                <div className="flex flex-wrap items-start gap-4 px-5 py-5">
                  <span className={cn("grid h-11 w-11 shrink-0 place-items-center rounded-xl border shadow-[inset_0_1px_0_hsl(var(--highlight))]", selected.enabled ? "border-accent/25 bg-accent/10 text-accent" : "bg-surface-2 text-subtle")}>
                    <Bot className="h-5 w-5" />
                  </span>
                  <div className="min-w-0 flex-1 basis-60">
                    <div className="flex flex-wrap items-center gap-2">
                      <h2 className="text-[18px] font-semibold tracking-[-0.02em]">{selected.name}</h2>
                      {selected.enabled ? (
                        <Badge tone="ok">
                          <Dot tone="ok" /> enabled
                        </Badge>
                      ) : (
                        <Badge>disabled</Badge>
                      )}
                    </div>
                    <p className="mt-1 text-[13.5px] leading-relaxed text-muted">{selected.description || "No description."}</p>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <GatedButton size="sm" variant="secondary" allowed={canManage && selected.access === "manage"} reason={!canManage ? "Requires agents.manage" : "You don't manage this agent"} onClick={() => setEditing(selected)}>
                      <Pencil className="h-3.5 w-3.5" /> Edit
                    </GatedButton>
                    <GatedButton size="icon" variant="ghost" aria-label="Delete agent" allowed={canManage && selected.access === "manage"} reason={!canManage ? "Requires agents.manage" : "You don't manage this agent"} onClick={() => setDeleting(selected)}>
                      <Trash2 className="h-3.5 w-3.5" />
                    </GatedButton>
                  </div>
                </div>

                {/* scope */}
                <dl className="grid grid-cols-2 border-t [&>*]:border-border max-md:[&>*:nth-child(odd)]:border-r max-md:[&>*:nth-child(n+3)]:border-t md:grid-cols-4 md:divide-x">
                  <ScopeCell icon={<Cpu />} label="Model">
                    <span className="block truncate">{providerOf(selected) ? providerOf(selected)!.name : "Org default"}</span>
                    {selected.model ? <span className="mt-0.5 block truncate font-mono text-[11.5px] text-muted">{selected.model}</span> : null}
                  </ScopeCell>
                  <ScopeCell icon={<Library />} label="Knowledge">
                    {selected.kb_ids.length ? (
                      <span className="block truncate" title={selected.kb_ids.map((id) => kbName(id) ?? "hidden").join(", ")}>
                        {selected.kb_ids.map((id) => kbName(id) ?? "hidden").join(", ")}
                      </span>
                    ) : (
                      <span className="block truncate text-muted">All the user can read</span>
                    )}
                  </ScopeCell>
                  <ScopeCell icon={<Database />} label="Databases">
                    {selected.connection_ids.length ? (
                      <span className="block truncate" title={selected.connection_ids.map((id) => connName(id) ?? "hidden").join(", ")}>
                        {selected.connection_ids.map((id) => connName(id) ?? "hidden").join(", ")}
                      </span>
                    ) : (
                      <span className="block truncate text-muted">None</span>
                    )}
                  </ScopeCell>
                  <ScopeCell icon={<Footprints />} label="Max steps">
                    <span className="num font-mono">{selected.limits?.max_steps ?? 6}</span>
                    <span className="text-muted"> tool calls / run</span>
                  </ScopeCell>
                </dl>

                {/* tools & permissions */}
                <div className="border-t">
                  <div className="flex flex-wrap items-center justify-between gap-2 px-5 pb-2 pt-4">
                    <h3 className="text-[13.5px] font-semibold tracking-[-0.01em]">Tools &amp; permissions</h3>
                    <span className="font-mono text-[11px] text-subtle">
                      {selected.tools.length} tool{selected.tools.length === 1 ? "" : "s"} · {selected.approval_tools.length} need approval
                    </span>
                  </div>
                  <ul className="mx-5 mb-4 divide-y rounded-lg border">
                    {selected.tools.map((t) => {
                      const gated = selected.approval_tools.includes(t);
                      const desc = toolDescription(t);
                      return (
                        <li key={t} className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3.5 py-2.5">
                          <span className={cn("grid h-6 w-6 shrink-0 place-items-center rounded-md border", gated ? "border-warn/30 bg-warn/10 text-warn" : "bg-surface-2 text-subtle")}>
                            {gated ? <Lock className="h-3 w-3" /> : <Wrench className="h-3 w-3" />}
                          </span>
                          <div className="min-w-0 flex-1 basis-48">
                            <div className="font-mono text-[12.5px] font-medium tracking-[-0.01em]">{t}</div>
                            {desc ? <div className="mt-0.5 line-clamp-1 text-[12px] text-muted" title={desc}>{firstSentence(desc)}</div> : null}
                          </div>
                          {gated ? (
                            <Badge tone="warn">
                              <Lock className="h-2.5 w-2.5" /> requires approval
                            </Badge>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-[11.5px] text-subtle">
                              <Zap className="h-3 w-3" /> runs automatically
                            </span>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                  <div className="flex flex-wrap items-start gap-2 border-t bg-surface-2/40 px-5 py-3 text-[12px] text-muted">
                    <ShieldCheck className="mt-[1px] h-3.5 w-3.5 shrink-0 text-subtle" />
                    <span className="min-w-0 flex-1">
                      Runs with the asking user&apos;s permissions, intersected with this scope. Created <span className="font-mono text-[11.5px]">{formatDate(selected.created_at)}</span> · your access: <span className="font-medium text-fg">{selected.access}</span>
                    </span>
                  </div>
                </div>

                {selected.instructions ? (
                  <details className="group border-t">
                    <summary className="flex cursor-pointer list-none items-center gap-2 px-5 py-3 text-[13px] font-medium text-muted transition-colors hover:text-fg [&::-webkit-details-marker]:hidden">
                      <ChevronDown className="h-3.5 w-3.5 -rotate-90 transition-transform group-open:rotate-0" />
                      Instructions
                      <span className="font-mono text-[11px] font-normal text-subtle">{selected.instructions.length} chars</span>
                    </summary>
                    <p className="mx-5 mb-4 whitespace-pre-wrap rounded-lg border bg-surface-2/50 p-3.5 text-[13px] leading-relaxed">{selected.instructions}</p>
                  </details>
                ) : null}
              </Card>
              <AgentRunPanel key={selected.id} agent={selected} />
            </div>
          ) : (
            <Card>
              <EmptyState title="Select an agent" description="Choose an agent on the left to see its configuration and run it." />
            </Card>
          )}
        </div>
      )}

      {canManage ? (
        <>
          <AgentFormDialog open={creating} onOpenChange={setCreating} onSaved={(a) => select(a.id)} />
          <AgentFormDialog open={Boolean(editing)} onOpenChange={(v) => !v && setEditing(null)} agent={editing} />
          <ConfirmDialog
            open={Boolean(deleting)}
            onOpenChange={(v) => !v && setDeleting(null)}
            title="Delete agent"
            description={deleting ? `“${deleting.name}” and its configuration are deleted. Knowledge bases, connections and past audit entries are not affected.` : undefined}
            confirmLabel="Delete agent"
            requireText={deleting?.name}
            loading={remove.isPending}
            onConfirm={() => {
              if (deleting) remove.mutate(deleting);
            }}
          />
        </>
      ) : null}
    </Page>
  );
}

function ScopeCell({ icon, label, children }: { icon: React.ReactNode; label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0 px-5 py-3.5">
      <dt className="label flex items-center gap-1.5 [&_svg]:h-3 [&_svg]:w-3">
        {icon}
        {label}
      </dt>
      <dd className="mt-1.5 min-w-0 text-[13px]">{children}</dd>
    </div>
  );
}
