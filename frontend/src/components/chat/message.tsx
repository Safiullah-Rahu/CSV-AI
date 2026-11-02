"use client";

import * as Popover from "@radix-ui/react-popover";
import { Check, CircleSlash, Copy, CornerDownRight, Database, FileText, Info, Layers, ListTree, RefreshCw, ShieldAlert, ShieldCheck, ShieldQuestion, Table2, ThumbsDown, ThumbsUp, TriangleAlert } from "lucide-react";
import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import styles from "@/components/chat/chat.module.css";
import { AssistantMark } from "@/components/chat/mark";
import type { ChatMessage, Citation } from "@/components/chat/types";
import { formatDuration, shortLocator } from "@/components/chat/utils";
import { Badge, Tip } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

const CITE = /\[(S\d+(?:\s*,\s*S\d+)*)\]/g;

/** Turn [S1] / [S1, S2] markers into links the renderer shows as citation chips. */
function linkCitations(text: string) {
  return text.replace(CITE, (_, group: string) =>
    group
      .split(",")
      .map((l) => l.trim())
      .map((l) => `[${l}](#cite-${l})`)
      .join(""),
  );
}

const STAGES = [
  { key: "retrieving", label: "Searching authorized sources" },
  { key: "generating", label: "Writing a grounded answer" },
  { key: "verifying", label: "Verifying citations" },
] as const;

export function SourceIcon({ c, className }: { c: Citation; className?: string }) {
  const Icon = c.source_type === "database" ? Database : c.source_type === "dataset" ? Table2 : FileText;
  return <Icon className={cn("h-3.5 w-3.5", className)} />;
}

/** Plain text of a hast node (used to recognise the engine's verification notice). */
function hastText(node: any): string {
  if (!node) return "";
  if (node.type === "text") return node.value ?? "";
  return (node.children ?? []).map(hastText).join("");
}

/* -------------------------------------------------------------- callouts */
const CALLOUT_TONE = {
  warn: { box: "border-warn/25 bg-warn/[0.06]", icon: "text-warn" },
  danger: { box: "border-danger/25 bg-danger/[0.05]", icon: "text-danger" },
  neutral: { box: "border-border bg-surface-2/60", icon: "text-subtle" },
} as const;

function Callout({ tone, icon, title, children, className }: { tone: keyof typeof CALLOUT_TONE; icon: React.ReactNode; title?: React.ReactNode; children?: React.ReactNode; className?: string }) {
  const t = CALLOUT_TONE[tone];
  return (
    <div role="note" className={cn("flex gap-3 rounded-xl border px-3.5 py-3 text-[13px] leading-relaxed", t.box, className)}>
      <span className={cn("mt-[2px] shrink-0 [&_svg]:h-4 [&_svg]:w-4", t.icon)}>{icon}</span>
      <div className="min-w-0 flex-1">
        {title ? <div className="font-medium text-fg">{title}</div> : null}
        {children ? <div className={cn("text-muted", title && "mt-0.5")}>{children}</div> : null}
      </div>
    </div>
  );
}

/* ----------------------------------------------------------- ghost icon */
function GhostIcon({ label, onClick, active, children }: { label: string; onClick: () => void; active?: string | false; children: React.ReactNode }) {
  return (
    <Tip content={label}>
      <button
        type="button"
        aria-label={label}
        onClick={onClick}
        className={cn(
          "grid h-7 w-7 place-items-center rounded-md text-subtle transition-colors hover:bg-surface-2 hover:text-fg focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/30 [&_svg]:h-3.5 [&_svg]:w-3.5",
          active,
        )}
      >
        {children}
      </button>
    </Tip>
  );
}

/* ---------------------------------------------------------------- view */
export function MessageView({
  m,
  userName,
  onCite,
  onRegenerate,
  onFeedback,
  onFollowUp,
  last,
}: {
  m: ChatMessage;
  userName?: string;
  onCite: (c: Citation) => void;
  onRegenerate?: () => void;
  onFeedback?: (rating: 1 | -1) => void;
  onFollowUp?: (q: string) => void;
  last?: boolean;
}) {
  if (m.role === "user") {
    return (
      <div className="flex justify-end animate-fade-in" aria-label={userName ? `${userName} asked` : "You asked"}>
        <div className="max-w-[min(85%,560px)] whitespace-pre-wrap rounded-[18px] rounded-br-[6px] bg-surface-2 px-4 py-2.5 text-[14.5px] leading-relaxed text-fg ring-1 ring-inset ring-border/70 [overflow-wrap:anywhere]">
          {m.content}
        </div>
      </div>
    );
  }
  return <AssistantTurn m={m} onCite={onCite} onRegenerate={onRegenerate} onFeedback={onFeedback} onFollowUp={onFollowUp} last={last} />;
}

function AssistantTurn({
  m,
  onCite,
  onRegenerate,
  onFeedback,
  onFollowUp,
  last,
}: {
  m: ChatMessage;
  onCite: (c: Citation) => void;
  onRegenerate?: () => void;
  onFeedback?: (rating: 1 | -1) => void;
  onFollowUp?: (q: string) => void;
  last?: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const [rated, setRated] = useState<1 | -1 | null>(null);
  const sources = m.citations?.length ? m.citations : m.liveSources ?? [];
  const byLabel = useMemo(() => Object.fromEntries(sources.map((c) => [c.label, c])), [sources]);
  const mode = m.metadata?.mode;
  const refused = mode === "refused" || m.status === "refused";
  const cited = sources.filter((c) => c.cited);
  const settled = !m.streaming && m.status !== "error" && Boolean(m.content || cited.length);

  return (
    <article className="flex flex-col gap-2.5 animate-fade-in md:flex-row md:gap-4" aria-busy={m.streaming || undefined}>
      <div className="flex items-center gap-2 md:block md:pt-[1px]">
        <AssistantMark live={m.streaming} className="h-6 w-6 md:h-7 md:w-7" />
        <span className="text-[12.5px] font-medium text-muted md:hidden">Assistant</span>
      </div>

      <div className="min-w-0 flex-1">
        {/* in-flight progress */}
        {m.streaming && !m.content ? <StageTracker stage={m.stage} found={m.liveSources?.length ?? 0} /> : null}

        {/* answer body */}
        {m.content && refused && !m.streaming ? (
          <Callout tone="warn" icon={<ShieldQuestion />} title="Not enough evidence in your sources">
            {m.content}
          </Callout>
        ) : m.content ? (
          <div className="prose-chat text-fg [overflow-wrap:anywhere] [&>*:first-child]:mt-0">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                a: ({ href, children }) => {
                  if (href?.startsWith("#cite-")) {
                    const label = href.slice(6);
                    const c = byLabel[label];
                    return <InlineCite label={label} c={c} onCite={onCite} />;
                  }
                  return (
                    <a href={href} target="_blank" rel="noopener noreferrer nofollow">
                      {children}
                    </a>
                  );
                },
                blockquote: ({ node, children }) => {
                  const text = hastText(node).trim();
                  if (text.startsWith("⚠")) {
                    return (
                      <div role="note" className="my-4 flex items-start gap-2.5 border-l-2 border-warn/50 py-0.5 pl-3 text-[13px] leading-relaxed text-muted">
                        <ShieldAlert className="mt-[3px] h-3.5 w-3.5 shrink-0 text-warn" />
                        <span>{text.replace(/^⚠️?\s*/, "")}</span>
                      </div>
                    );
                  }
                  return <div className="my-3 border-l-2 border-border-strong pl-4 text-muted">{children}</div>;
                },
                table: ({ children }) => (
                  <div className="my-3 overflow-x-auto rounded-lg border">
                    <table className="!my-0 w-full">{children}</table>
                  </div>
                ),
              }}
            >
              {linkCitations(m.content)}
            </ReactMarkdown>
            {m.streaming ? <span className={styles.caret} aria-hidden /> : null}
          </div>
        ) : null}

        {m.streaming && m.content && m.stage === "verifying" ? (
          <div className="mt-3 text-[12.5px]">
            <span className={styles.shimmerText}>Verifying citations</span>
          </div>
        ) : null}

        {m.status === "error" ? (
          <Callout tone="danger" icon={<TriangleAlert />} title="The answer could not be generated" className={m.content ? "mt-4" : undefined}>
            {m.metadata?.error || "Something went wrong while answering. Try again in a moment."}
          </Callout>
        ) : null}
        {m.status === "stopped" || (!m.streaming && !m.content && m.status === "streaming") ? (
          <div className={cn("flex flex-wrap items-center gap-2", m.content ? "mt-3" : "pt-0.5")}>
            <span className="inline-flex items-center gap-1.5 rounded-md bg-surface-2 px-2 py-1 text-[12px] text-muted ring-1 ring-inset ring-border">
              <CircleSlash className="h-3.5 w-3.5 text-subtle" /> {m.status === "stopped" ? "Generation stopped" : "This answer did not finish"}
            </span>
            {!m.content && onRegenerate && last ? (
              <button type="button" onClick={onRegenerate} className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-[12px] font-medium text-accent transition-colors hover:bg-accent/10">
                <RefreshCw className="h-3 w-3" /> Regenerate
              </button>
            ) : null}
          </div>
        ) : null}

        {settled ? (
          <>
            {m.metadata?.unsupported_sentences?.length ? (
              <Callout
                tone="warn"
                icon={<ShieldAlert />}
                className="mt-4"
                title={`${m.metadata.unsupported_sentences.length} ${m.metadata.unsupported_sentences.length === 1 ? "statement" : "statements"} could not be matched to the cited text`}
              >
                <ul className="mt-1 space-y-1">
                  {m.metadata.unsupported_sentences.slice(0, 3).map((s, i) => (
                    <li key={i} className="line-clamp-2 border-l border-warn/30 pl-2.5 text-[12.5px] [overflow-wrap:anywhere]">
                      {s}
                    </li>
                  ))}
                </ul>
              </Callout>
            ) : null}

            {cited.length ? (
              <div className="mt-5">
                <div className="label mb-2">Sources</div>
                <div className="flex flex-wrap gap-1.5">
                  {cited.map((c) => (
                    <SourceCard key={c.label} c={c} onClick={() => onCite(c)} />
                  ))}
                </div>
              </div>
            ) : sources.length ? (
              <button
                type="button"
                data-cite-chip
                onClick={() => onCite(sources[0])}
                className="mt-4 inline-flex items-center gap-1.5 rounded-md text-[12.5px] text-muted transition-colors hover:text-fg"
              >
                <Layers className="h-3.5 w-3.5 text-subtle" />
                {sources.length} {sources.length === 1 ? "passage" : "passages"} retrieved, none cited
                <span className="text-accent">Inspect</span>
              </button>
            ) : null}

            <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-2">
              <div className="-ml-1.5 flex items-center">
                <GhostIcon
                  label={copied ? "Copied" : "Copy answer"}
                  onClick={() => {
                    navigator.clipboard.writeText(m.content);
                    setCopied(true);
                    setTimeout(() => setCopied(false), 1500);
                  }}
                >
                  {copied ? <Check className="text-ok" /> : <Copy />}
                </GhostIcon>
                {onFeedback ? (
                  <>
                    <GhostIcon
                      label="Helpful"
                      active={rated === 1 && "text-ok hover:text-ok"}
                      onClick={() => {
                        setRated(1);
                        onFeedback(1);
                      }}
                    >
                      <ThumbsUp className={rated === 1 ? "fill-current" : undefined} />
                    </GhostIcon>
                    <GhostIcon
                      label="Not helpful"
                      active={rated === -1 && "text-danger hover:text-danger"}
                      onClick={() => {
                        setRated(-1);
                        onFeedback(-1);
                      }}
                    >
                      <ThumbsDown className={rated === -1 ? "fill-current" : undefined} />
                    </GhostIcon>
                  </>
                ) : null}
                {onRegenerate && last ? (
                  <GhostIcon label="Regenerate" onClick={onRegenerate}>
                    <RefreshCw />
                  </GhostIcon>
                ) : null}
                <TraceDetails m={m} sources={sources.length} />
              </div>

              <div className="flex min-w-0 flex-wrap items-center gap-1.5 text-[12px] text-subtle sm:ml-auto">
                {refused ? (
                  <Badge tone="warn">Insufficient evidence</Badge>
                ) : mode === "extractive" ? (
                  <Tip content="No language model answered; the answer quotes the best-matching source sentences.">
                    <span>
                      <Badge tone="info">Extractive</Badge>
                    </span>
                  </Tip>
                ) : m.model ? (
                  <Badge className="font-mono text-[11px]">{m.model}</Badge>
                ) : null}
                {typeof m.confidence === "number" && !refused ? (
                  <Tip content="Combines retrieval evidence and how well each cited sentence is supported by its source.">
                    <span>
                      <Badge tone={m.confidence >= 0.7 ? "ok" : m.confidence >= 0.45 ? "warn" : "danger"}>
                        {m.grounded ? <ShieldCheck className="h-3 w-3" /> : <ShieldAlert className="h-3 w-3" />}
                        <span className="num">{Math.round(m.confidence * 100)}%</span> confidence
                      </Badge>
                    </span>
                  </Tip>
                ) : null}
                {m.retrieval_ms != null || m.generation_ms ? (
                  <Tip content={m.generation_ms ? "Retrieval · generation time" : "Retrieval time"}>
                    <span className="num hidden cursor-default font-mono text-[11px] sm:inline">
                      {[formatDuration(m.retrieval_ms), m.generation_ms ? formatDuration(m.generation_ms) : null].filter(Boolean).join(" · ")}
                    </span>
                  </Tip>
                ) : null}
              </div>
            </div>

            {last && m.metadata?.follow_ups?.length && onFollowUp ? (
              <div className="mt-6 border-t pt-4">
                <div className="label mb-2">Related</div>
                <ul className="divide-y">
                  {m.metadata.follow_ups.map((q) => (
                    <li key={q}>
                      <button
                        type="button"
                        onClick={() => onFollowUp(q)}
                        className="group flex w-full items-center gap-2.5 py-2 text-left text-[13.5px] text-muted transition-colors hover:text-fg"
                      >
                        <CornerDownRight className="h-3.5 w-3.5 shrink-0 text-subtle transition-colors group-hover:text-accent" />
                        <span className="min-w-0 flex-1">{q}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </>
        ) : null}
      </div>
    </article>
  );
}

/* ------------------------------------------------------- stage tracker */
function StageTracker({ stage, found }: { stage?: string; found: number }) {
  const idx = Math.max(
    0,
    STAGES.findIndex((s) => s.key === (stage || "retrieving")),
  );
  return (
    <div className="py-1" role="status" aria-live="polite">
      <div className="text-[14px] font-medium">
        <span className={styles.shimmerText}>{STAGES[idx].label}</span>
        {found && idx === 0 ? <span className="ml-2 font-mono text-[11.5px] font-normal text-subtle">{found} found</span> : null}
      </div>
      <ol className="mt-3 flex items-center gap-1.5" aria-label="Progress">
        {STAGES.map((s, i) => (
          <li key={s.key} className="flex items-center gap-1.5">
            <span
              className={cn(
                "h-1 rounded-full transition-all duration-300",
                i < idx ? "w-6 bg-accent/50" : i === idx ? "w-10 bg-accent" : "w-6 bg-border-strong/60",
              )}
            />
            <span className="sr-only">
              {s.label} {i < idx ? "(done)" : i === idx ? "(in progress)" : "(pending)"}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

/* --------------------------------------------------------- citations */
function InlineCite({ label, c, onCite }: { label: string; c?: Citation; onCite: (c: Citation) => void }) {
  const chip = (
    <button
      type="button"
      data-cite-chip
      onClick={() => c && onCite(c)}
      className={cn(
        "relative -top-[0.4em] mx-[1px] inline-flex h-4 min-w-4 items-center justify-center rounded-[5px] px-[3px] align-baseline font-mono text-[10px] font-semibold leading-none no-underline ring-1 ring-inset transition-colors",
        c ? "bg-accent/10 text-accent ring-accent/25 hover:bg-accent hover:text-accent-fg hover:ring-accent" : "cursor-default bg-danger/5 text-danger ring-danger/30",
      )}
      aria-label={c ? `Source ${label}: ${c.title}` : `Unknown source ${label}`}
    >
      {label.slice(1)}
    </button>
  );
  if (!c) return <Tip content="This citation does not match any retrieved source">{chip}</Tip>;
  const loc = shortLocator(c);
  return (
    <Tip
      content={
        <span className="block">
          <span className="block truncate">{c.title}</span>
          {loc ? <span className="mt-0.5 block truncate font-mono text-[10.5px] font-normal opacity-70">{loc}</span> : null}
        </span>
      }
    >
      {chip}
    </Tip>
  );
}

function SourceCard({ c, onClick }: { c: Citation; onClick: () => void }) {
  const loc = shortLocator(c);
  return (
    <button
      type="button"
      data-cite-chip
      onClick={onClick}
      className={cn(
        "group flex max-w-full items-center gap-2 rounded-lg border bg-surface py-1.5 pl-1.5 pr-2.5 text-left shadow-xs transition-[border-color,background] hover:border-accent/35 hover:bg-accent/[0.04] sm:max-w-[300px]",
        c.accessible === false && "opacity-60",
      )}
    >
      <span className="grid h-5 min-w-5 place-items-center rounded-[5px] bg-accent/10 px-1 font-mono text-[10px] font-semibold text-accent">{c.label.slice(1)}</span>
      <SourceIcon c={c} className="shrink-0 text-subtle" />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[12.5px] font-medium leading-4">{c.title}</span>
        {loc ? <span className="block truncate font-mono text-[10.5px] leading-4 text-subtle">{loc}</span> : null}
      </span>
      {c.flagged ? <ShieldAlert className="h-3.5 w-3.5 shrink-0 text-warn" aria-label="Contains instruction-like text" /> : null}
    </button>
  );
}

/* ------------------------------------------------------- trace details */
function Row({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <>
      <dt className="label pt-[3px]">{k}</dt>
      <dd className="min-w-0 text-[12.5px] text-fg [overflow-wrap:anywhere]">{children}</dd>
    </>
  );
}

function TraceDetails({ m, sources }: { m: ChatMessage; sources: number }) {
  const t = m.metadata?.trace;
  const support = m.metadata?.support_ratio;
  const invalid = m.metadata?.invalid_citations;
  const has = Boolean(t || m.model || m.retrieval_ms != null || m.generation_ms);
  if (!has) return null;
  const stages: any[] = Array.isArray(t?.stages) ? t.stages : [];
  const candidates = stages.find((s) => s.stage === "candidates");
  const rerank = stages.find((s) => s.stage === "rerank");
  return (
    <Popover.Root>
      <Tip content="Answer details">
        <Popover.Trigger asChild>
          <button
            type="button"
            aria-label="Answer details"
            className="grid h-7 w-7 place-items-center rounded-md text-subtle transition-colors hover:bg-surface-2 hover:text-fg focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/30 data-[state=open]:bg-surface-2 data-[state=open]:text-fg"
          >
            <ListTree className="h-3.5 w-3.5" />
          </button>
        </Popover.Trigger>
      </Tip>
      <Popover.Portal>
        <Popover.Content side="top" align="start" sideOffset={8} collisionPadding={12} className="z-50 w-[min(360px,calc(100vw-24px))] rounded-xl border bg-surface shadow-float animate-scale-in focus:outline-none">
          <div className="flex items-center gap-2 border-b px-4 py-3">
            <Info className="h-3.5 w-3.5 text-subtle" />
            <span className="text-[13px] font-semibold tracking-[-0.01em]">How this answer was produced</span>
          </div>
          <dl className="grid grid-cols-[88px_1fr] gap-x-3 gap-y-2.5 px-4 py-3.5">
            <Row k="Answer">
              <span className="capitalize">{m.metadata?.mode === "llm" ? "Generated" : m.metadata?.mode || m.status}</span>
              {m.model ? <span className="ml-1.5 font-mono text-[11.5px] text-muted">{m.model}</span> : null}
              {m.provider ? <span className="ml-1 font-mono text-[11.5px] text-subtle">via {m.provider}</span> : null}
            </Row>
            {t?.mode ? (
              <Row k="Retrieval">
                <span className="font-mono text-[11.5px]">{t.mode}</span>
                {typeof t.returned === "number" ? <span className="text-muted"> · {t.returned} passages</span> : sources ? <span className="text-muted"> · {sources} passages</span> : null}
                {typeof t.evidence === "number" ? <span className="text-muted"> · evidence {Number(t.evidence).toFixed(2)}</span> : null}
              </Row>
            ) : null}
            {candidates ? (
              <Row k="Candidates">
                <span className="num font-mono text-[11.5px]">
                  {candidates.dense ?? 0} dense / {candidates.sparse ?? 0} sparse
                </span>
                {candidates.ms != null ? <span className="text-subtle"> · {formatDuration(candidates.ms)}</span> : null}
              </Row>
            ) : null}
            {rerank ? (
              <Row k="Rerank">
                <span className="font-mono text-[11.5px]">{rerank.reranker}</span>
                <span className="text-muted"> · {rerank.candidates} scored</span>
                {rerank.ms != null ? <span className="text-subtle"> · {formatDuration(rerank.ms)}</span> : null}
              </Row>
            ) : null}
            {t?.condensed_query ? <Row k="Rewritten">{t.condensed_query}</Row> : null}
            {Array.isArray(t?.terms) && t.terms.length ? (
              <Row k="Terms">
                <span className="flex flex-wrap gap-1">
                  {t.terms.slice(0, 12).map((x: string) => (
                    <span key={x} className="rounded bg-surface-2 px-1.5 py-px font-mono text-[11px] text-muted ring-1 ring-inset ring-border">
                      {x}
                    </span>
                  ))}
                </span>
              </Row>
            ) : null}
            {Array.isArray(t?.prf_terms) && t.prf_terms.length ? (
              <Row k="Expanded">
                <span className="font-mono text-[11.5px] text-muted">{t.prf_terms.slice(0, 8).join(", ")}</span>
              </Row>
            ) : null}
            {typeof support === "number" ? (
              <Row k="Supported">
                <span className="num">{Math.round(support * 100)}%</span> <span className="text-muted">of sentences match their sources</span>
              </Row>
            ) : null}
            {invalid?.length ? (
              <Row k="Invalid">
                <span className="font-mono text-[11.5px] text-danger">{invalid.join(", ")}</span>
              </Row>
            ) : null}
            {m.retrieval_ms != null || m.generation_ms ? (
              <Row k="Timing">
                <span className="num font-mono text-[11.5px]">
                  {m.retrieval_ms != null ? `retrieval ${formatDuration(m.retrieval_ms)}` : null}
                  {m.retrieval_ms != null && m.generation_ms ? " · " : null}
                  {m.generation_ms ? `generation ${formatDuration(m.generation_ms)}` : null}
                </span>
              </Row>
            ) : null}
          </dl>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
