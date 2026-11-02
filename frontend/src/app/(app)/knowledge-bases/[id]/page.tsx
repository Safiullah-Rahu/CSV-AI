"use client";

import { ArrowLeft, Database, ExternalLink, FileText, HardDrive, History, Layers, Library, Lock, Settings2, ShieldCheck, UploadCloud } from "lucide-react";
import Link from "next/link";
import { useParams, usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";

import { DocumentDrawer } from "@/components/knowledge/document-drawer";
import { DocumentsBrowser } from "@/components/knowledge/documents-browser";
import { KBAccessTab } from "@/components/knowledge/kb-access";
import { GROUNDING } from "@/components/knowledge/kb-config";
import { KBHistoryTab } from "@/components/knowledge/kb-history";
import { KBSettingsTab } from "@/components/knowledge/kb-settings";
import { AccessBadge, kbAbilities, KBMark, needs, TagList, usePerms, VisibilityBadge } from "@/components/knowledge/shared";
import type { KBOptions, KnowledgeBase } from "@/components/knowledge/types";
import { UploadZone } from "@/components/knowledge/upload-zone";
import { usePageCrumb } from "@/components/shell/breadcrumb";
import { Button } from "@/components/ui/button";
import { Badge, Card, Dot, EmptyState, Page, Skeleton, Stat, StatGroup, StatusBadge, Tabs, TabsContent, TabsList, TabsTrigger, Tip } from "@/components/ui/primitives";
import { ApiError } from "@/lib/api";
import { useApi } from "@/lib/hooks";
import { formatBytes, formatDate, formatNumber, timeAgo } from "@/lib/utils";

const TABS = ["documents", "access", "settings", "history"] as const;
type Tab = (typeof TABS)[number];

export default function KnowledgeBasePage() {
  return (
    <Suspense fallback={<PageSkeleton />}>
      <KnowledgeBaseDashboard />
    </Suspense>
  );
}

function PageSkeleton() {
  return (
    <Page>
      <Skeleton className="mb-4 h-3.5 w-32" />
      <div className="mb-8 flex items-start gap-4">
        <Skeleton className="h-12 w-12 rounded-xl" />
        <div className="flex-1 space-y-2.5">
          <Skeleton className="h-8 w-72" />
          <Skeleton className="h-5 w-56" />
        </div>
      </div>
      <Skeleton className="h-[108px] rounded-xl" />
      <Skeleton className="mt-8 h-10 w-80" />
      <Skeleton className="mt-6 h-96 rounded-xl" />
    </Page>
  );
}

function KnowledgeBaseDashboard() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const pathname = usePathname();
  const search = useSearchParams();
  const perms = usePerms();
  const raw = search.get("tab");
  const tab: Tab = (TABS as readonly string[]).includes(raw || "") ? (raw as Tab) : "documents";
  const [docId, setDocId] = useState<string | null>(null);

  const kbQ = useApi<KnowledgeBase>(id ? `/knowledge-bases/${id}` : null, undefined, {
    refetchInterval: (q) => (((q.state.data as KnowledgeBase | undefined)?.stats.processing ?? 0) > 0 ? 4000 : false),
  });
  usePageCrumb(kbQ.data?.name);
  const options = useApi<KBOptions>("/knowledge-bases/options", undefined, { staleTime: 5 * 60_000 });
  const kb = kbQ.data;
  const ab = kbAbilities(kb, perms);

  const setTab = (t: string) => {
    const params = new URLSearchParams(search.toString());
    if (t === "documents") params.delete("tab");
    else params.set("tab", t);
    const qs = params.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  };

  if (kbQ.error instanceof ApiError && (kbQ.error.status === 404 || kbQ.error.status === 422)) {
    return (
      <Page>
        <Card>
          <EmptyState
            icon={<Library className="h-5 w-5" />}
            title="Knowledge base not found"
            description="It may have been deleted, or you don't have access to it."
            action={
              <Link href="/knowledge-bases" className="inline-flex items-center gap-1.5 text-[13px] font-medium text-accent hover:underline">
                <ArrowLeft className="h-3.5 w-3.5" /> All knowledge bases
              </Link>
            }
          />
        </Card>
      </Page>
    );
  }
  if (!kb) return <PageSkeleton />;

  const s = kb.stats;
  const emb = kb.embedding_config || {};
  const chunk = { ...(options.data?.chunking_defaults ?? {}), ...kb.chunking_config };
  const ret = { ...(options.data?.retrieval_defaults ?? {}), ...kb.retrieval_config };
  const grounding = GROUNDING.find((g) => g.value === kb.generation_config?.grounding_policy);

  return (
    <Page>
      {/* header */}
      <header className="mb-8 animate-fade-up">
        <Link href="/knowledge-bases" className="group mb-5 inline-flex items-center gap-1.5 text-[12.5px] font-medium text-muted transition-colors hover:text-fg">
          <ArrowLeft className="h-3.5 w-3.5 transition-transform group-hover:-translate-x-0.5" /> Knowledge bases
        </Link>
        <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-4">
          <div className="flex min-w-0 max-w-3xl items-start gap-4">
            <KBMark size="lg" className="mt-0.5" />
            <div className="min-w-0">
              <h1 className="text-[28px] font-semibold leading-[1.12] tracking-[-0.03em] [overflow-wrap:anywhere]">{kb.name}</h1>
              <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
                <VisibilityBadge visibility={kb.visibility} />
                <AccessBadge access={kb.access} />
                {kb.is_sample ? <Badge tone="warn">sample</Badge> : null}
                {kb.status !== "active" ? <StatusBadge status={kb.status} /> : null}
                {kb.category ? <Badge>{kb.category}</Badge> : null}
                <TagList tags={kb.tags} max={5} />
              </div>
              {kb.description ? <p className="mt-3 text-[14.5px] leading-relaxed text-muted">{kb.description}</p> : null}
              <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-[12.5px] text-muted">
                {s.processing ? (
                  <span className="inline-flex items-center gap-1.5 text-info">
                    <Dot tone="info" live /> Indexing {formatNumber(s.processing)} {s.processing === 1 ? "document" : "documents"} · refreshing live
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5" title={s.last_indexed_at ? formatDate(s.last_indexed_at) : undefined}>
                    <Dot tone={s.last_indexed_at ? "ok" : "neutral"} /> {s.last_indexed_at ? <>Last indexed {timeAgo(s.last_indexed_at)}</> : "No documents indexed yet"}
                  </span>
                )}
                <span className="h-3 w-px bg-border" aria-hidden />
                <span className="font-mono text-[11.5px]" title={formatDate(kb.updated_at)}>
                  config v{kb.config_version} · updated {timeAgo(kb.updated_at)}
                </span>
              </div>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="secondary" onClick={() => router.push(`/documents?kb_id=${kb.id}`)}>
              <ExternalLink className="h-4 w-4" /> Open in library
            </Button>
            {tab !== "documents" && ab.upload ? (
              <Button onClick={() => setTab("documents")}>
                <UploadCloud className="h-4 w-4" /> Upload
              </Button>
            ) : null}
          </div>
        </div>
      </header>

      {/* metrics */}
      <StatGroup cols={4} className="animate-fade-up">
        <Stat
          label="Documents"
          icon={<FileText />}
          tone={s.failed ? "danger" : s.processing ? "info" : undefined}
          value={formatNumber(s.documents ?? 0)}
          hint={
            <span className="inline-flex flex-wrap items-center gap-x-2">
              <span>{formatNumber(s.ready ?? 0)} ready</span>
              {s.processing ? (
                <span className="inline-flex items-center gap-1.5 text-info">
                  <Dot tone="info" live /> {formatNumber(s.processing)} processing
                </span>
              ) : null}
              {s.failed ? <span className="text-danger">· {formatNumber(s.failed)} failed</span> : null}
            </span>
          }
        />
        <Stat label="Chunks" icon={<Layers />} value={formatNumber(s.chunks ?? 0)} hint="Searchable passages" />
        <Stat label="Records" icon={<Database />} value={formatNumber(s.records ?? 0)} hint="Dataset rows" />
        <Stat label="Storage" icon={<HardDrive />} value={formatBytes(s.storage_bytes ?? 0)} hint="Original files" />
      </StatGroup>

      {/* pipeline readout */}
      <dl className="mt-3 grid grid-cols-1 gap-px overflow-hidden rounded-xl border bg-border sm:grid-cols-2 xl:grid-cols-4">
        <ConfigChip label="Embeddings" value={`${emb.provider ?? "hashing"}${emb.model ? ` · ${emb.model}` : ""}${emb.dim ? ` · ${emb.dim}d` : ""}`} />
        <ConfigChip label="Chunking" value={chunk.strategy ? `${String(chunk.strategy).replace(/_/g, " ")} · ${chunk.chunk_size} / ${chunk.chunk_overlap}` : "—"} />
        <ConfigChip label="Retrieval" value={ret.mode ? `${ret.mode} · top ${ret.top_k} · rerank ${ret.rerank}` : "—"} />
        <ConfigChip label="Grounding" value={grounding ? grounding.label : "Organization default"} />
      </dl>

      <Tabs value={tab} onValueChange={setTab} className="mt-8">
        <TabsList variant="line" className="gap-4 sm:gap-6">
          <TabsTrigger value="documents">
            <FileText className="hidden h-3.5 w-3.5 sm:block" /> Documents
            <span className="num ml-0.5 rounded-full bg-surface-2 px-1.5 py-px font-mono text-[10.5px] text-muted ring-1 ring-inset ring-border">{formatNumber(s.documents ?? 0)}</span>
          </TabsTrigger>
          <TabsTrigger value="access">
            <ShieldCheck className="hidden h-3.5 w-3.5 sm:block" /> Access
          </TabsTrigger>
          <TabsTrigger value="settings">
            <Settings2 className="hidden h-3.5 w-3.5 sm:block" /> Settings
          </TabsTrigger>
          <TabsTrigger value="history">
            <History className="hidden h-3.5 w-3.5 sm:block" /> History
          </TabsTrigger>
        </TabsList>

        <TabsContent value="documents" className="mt-6 space-y-4 focus:outline-none">
          {ab.upload ? (
            <UploadZone kbId={kb.id} kbName={kb.name} options={options.data} allowed={ab.upload} canRestrict={ab.restrictDocuments} />
          ) : (
            <Tip content={needs("kb.upload", "write")}>
              <p className="inline-flex items-center gap-2 rounded-lg border bg-surface-2/50 px-3 py-2 text-[12.5px] text-muted">
                <Lock className="h-3.5 w-3.5" /> You can browse this knowledge base but not upload to it.
              </p>
            </Tip>
          )}
          <DocumentsBrowser kbs={[kb]} fixedKbId={kb.id} onOpen={setDocId} selectedId={docId} />
        </TabsContent>
        <TabsContent value="access" className="mt-6 focus:outline-none">
          <KBAccessTab kb={kb} />
        </TabsContent>
        <TabsContent value="settings" className="mt-8 focus:outline-none">
          <KBSettingsTab kb={kb} />
        </TabsContent>
        <TabsContent value="history" className="mt-6 focus:outline-none">
          <KBHistoryTab kbId={kb.id} />
        </TabsContent>
      </Tabs>

      <DocumentDrawer docId={docId} onClose={() => setDocId(null)} />
    </Page>
  );
}

function ConfigChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 bg-bg px-4 py-2.5">
      <dt className="label">{label}</dt>
      <dd className="mt-1 truncate font-mono text-[12px] text-fg/85" title={value}>
        {value}
      </dd>
    </div>
  );
}
