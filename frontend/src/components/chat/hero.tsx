"use client";

import { ArrowRight, Library } from "lucide-react";

import { AssistantMark } from "@/components/chat/mark";
import type { KB } from "@/components/chat/types";
import { Skeleton } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

/**
 * The new-chat screen: editorial heading, the composer, and suggestions built
 * only from knowledge bases the user can actually search.
 */
export function ChatHero({ kbs, loading, composer, onAsk, canChat }: { kbs: KB[]; loading: boolean; composer: React.ReactNode; onAsk: (q: string) => void; canChat: boolean }) {
  const suggestions = kbs
    .filter((k) => (k.stats?.documents ?? 0) > 0)
    .sort((a, b) => (b.stats?.documents ?? 0) - (a.stats?.documents ?? 0))
    .slice(0, 4);
  const searchable = kbs.length;

  return (
    <div className="relative flex min-h-full flex-col">
      {/* faint dot field behind the heading */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[420px] bg-dots opacity-70 [mask-image:radial-gradient(ellipse_60%_55%_at_50%_30%,black,transparent)]" aria-hidden />

      <div className="relative mx-auto flex w-full max-w-[720px] flex-1 flex-col justify-center px-4 pb-10 pt-10 sm:px-8 sm:pt-14">
        <div className="stagger text-center">
          <div className="relative mx-auto mb-7 grid h-20 w-20 place-items-center">
            <span className="absolute inset-0 rounded-full border border-dashed border-border-strong/70" />
            <span className="absolute inset-[11px] rounded-full border border-border bg-bg" />
            <span className="relative rounded-[12px] shadow-card">
              <AssistantMark className="h-10 w-10" />
            </span>
          </div>
          <div className="label mb-3 flex items-center justify-center gap-2">
            <span className="h-1 w-3 rounded-full bg-accent" />
            {loading ? "Grounded answers" : searchable ? `Grounded in ${searchable} knowledge ${searchable === 1 ? "base" : "bases"}` : "Grounded answers"}
          </div>
          <h1 className="font-display text-[38px] font-normal leading-[1.04] tracking-[-0.015em] text-fg sm:text-[52px]">What would you like to know?</h1>
          <p className="mx-auto mt-3.5 max-w-[470px] text-[14.5px] leading-relaxed text-muted">
            Answers come only from the documents, datasets and databases you can access, and every claim links back to its source.
          </p>
        </div>

        <div className="mt-8 animate-fade-up [animation-delay:120ms]">{composer}</div>

        <div className="mt-7 animate-fade-up [animation-delay:180ms]">
          {loading ? (
            <div className="grid grid-cols-1 gap-px overflow-hidden rounded-xl border bg-border sm:grid-cols-2">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="bg-surface px-3.5 py-3">
                  <Skeleton className="h-4 w-4/5" />
                  <Skeleton className="mt-2 h-3 w-1/3" />
                </div>
              ))}
            </div>
          ) : suggestions.length ? (
            <>
              <div className="label mb-2.5 px-1">Start with a knowledge base</div>
              <div className="grid grid-cols-1 gap-px overflow-hidden rounded-xl border bg-border shadow-card sm:grid-cols-2">
                {suggestions.map((kb, i) => {
                  const q = `What does ${kb.name} cover?`;
                  return (
                    <button
                      type="button"
                      key={kb.id}
                      disabled={!canChat}
                      onClick={() => onAsk(q)}
                      className={cn(
                        "group flex items-start gap-3 bg-surface px-3.5 py-3 text-left transition-colors hover:bg-surface-2/70 focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-inset focus-visible:ring-ring/30 disabled:pointer-events-none disabled:opacity-60",
                        suggestions.length % 2 === 1 && i === suggestions.length - 1 && "sm:col-span-2",
                      )}
                    >
                      <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-accent/10 text-accent ring-1 ring-inset ring-accent/15">
                        <Library className="h-3.5 w-3.5" />
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block text-[13.5px] font-medium leading-5 text-fg">{q}</span>
                        <span className="mt-0.5 block truncate font-mono text-[11px] text-subtle">
                          {kb.stats.documents} {kb.stats.documents === 1 ? "document" : "documents"}
                          {kb.description ? <span className="font-sans"> · {kb.description}</span> : null}
                        </span>
                      </span>
                      <ArrowRight className="mt-1.5 h-3.5 w-3.5 shrink-0 -translate-x-1 text-subtle opacity-0 transition-all group-hover:translate-x-0 group-hover:text-accent group-hover:opacity-100" />
                    </button>
                  );
                })}
              </div>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
