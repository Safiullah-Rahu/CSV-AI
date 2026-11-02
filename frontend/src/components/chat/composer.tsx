"use client";

import { ArrowUp, Lock, Square } from "lucide-react";
import { useEffect, useLayoutEffect, useRef } from "react";

import { Kbd, Tip } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

const MAX_HEIGHT = 216;

/**
 * Floating question composer: auto-growing textarea, scope chips on the left of
 * the footer row, the send / stop control on the right.
 */
export function Composer({
  value,
  onChange,
  onSubmit,
  onStop,
  streaming,
  canChat,
  inputRef,
  scope,
  autoFocus,
  hero,
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  streaming: boolean;
  canChat: boolean;
  inputRef: React.MutableRefObject<HTMLTextAreaElement | null>;
  scope?: React.ReactNode;
  autoFocus?: boolean;
  hero?: boolean;
}) {
  const local = useRef<HTMLTextAreaElement | null>(null);
  const ready = canChat && Boolean(value.trim());

  // Grow with content up to MAX_HEIGHT, then scroll inside.
  useLayoutEffect(() => {
    const el = local.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
    el.style.overflowY = el.scrollHeight > MAX_HEIGHT ? "auto" : "hidden";
  }, [value]);

  // Focus on mount for mouse/trackpad users; on touch screens this would pop the keyboard.
  useEffect(() => {
    if (!autoFocus || !canChat) return;
    if (typeof window !== "undefined" && !window.matchMedia("(pointer: fine)").matches) return;
    const el = local.current;
    if (!el || document.activeElement === el) return;
    el.focus({ preventScroll: true });
    const end = el.value.length;
    el.setSelectionRange(end, end);
  }, [autoFocus, canChat]);

  return (
    <form
      className={cn(
        "group/composer relative rounded-2xl border bg-surface shadow-lift transition-[border-color,box-shadow] duration-200",
        "focus-within:border-accent/45 focus-within:shadow-[0_0_0_4px_hsl(var(--accent)/0.10),0_2px_4px_hsl(var(--shadow)/0.05),0_16px_32px_-12px_hsl(var(--shadow)/0.16)]",
        !canChat && "bg-surface-2/50",
      )}
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      <label htmlFor="chat-composer" className="sr-only">
        Message
      </label>
      <textarea
        id="chat-composer"
        ref={(el) => {
          local.current = el;
          inputRef.current = el;
        }}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            onSubmit();
          }
        }}
        rows={1}
        placeholder={canChat ? (hero ? "Ask about your documents, datasets and databases…" : "Ask a follow-up…") : "You don't have permission to chat"}
        disabled={!canChat}
        className={cn(
          "block w-full resize-none bg-transparent px-4 pt-3.5 text-[15px] leading-[1.55] text-fg outline-none placeholder:text-subtle disabled:cursor-not-allowed",
          hero ? "min-h-[76px]" : "min-h-[52px]",
        )}
        aria-label="Message"
      />
      <div className="flex items-center gap-1 px-2 pb-2 pt-1">
        <div className="flex min-w-0 flex-1 items-center gap-0.5">{scope}</div>
        <div className="hidden items-center gap-1.5 pr-1.5 text-[11.5px] text-subtle md:flex">
          {canChat ? (
            <>
              <Kbd className="h-[18px] px-1 text-[10px]">Enter</Kbd> send
              <span className="text-border-strong">·</span>
              <Kbd className="h-[18px] px-1 text-[10px]">Shift</Kbd>
              <Kbd className="h-[18px] px-1 text-[10px]">Enter</Kbd> new line
            </>
          ) : (
            <span className="flex items-center gap-1">
              <Lock className="h-3 w-3" /> Needs the chat permission
            </span>
          )}
        </div>
        {streaming ? (
          <Tip content="Stop generating">
            <button
              type="button"
              onClick={onStop}
              aria-label="Stop generating"
              className="grid h-8 w-8 shrink-0 place-items-center rounded-[10px] bg-fg text-bg shadow-[inset_0_1px_0_hsl(0_0%_100%/0.14),0_1px_2px_hsl(var(--shadow)/0.25)] transition-colors hover:bg-fg/85 focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/35"
            >
              <Square className="h-3 w-3 fill-current" />
            </button>
          </Tip>
        ) : (
          <button
            type="submit"
            disabled={!ready}
            aria-label="Send"
            className={cn(
              "grid h-8 w-8 shrink-0 place-items-center rounded-[10px] transition-[background,color,box-shadow,transform] duration-150 focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/35",
              ready
                ? "bg-accent text-accent-fg shadow-[inset_0_1px_0_hsl(0_0%_100%/0.2),0_1px_2px_hsl(var(--shadow)/0.18),0_0_0_1px_hsl(var(--accent)/0.9)] [background-image:linear-gradient(180deg,hsl(0_0%_100%/0.12),transparent)] hover:brightness-[1.06] active:scale-95"
                : "cursor-not-allowed bg-surface-2 text-subtle ring-1 ring-inset ring-border",
            )}
          >
            <ArrowUp className="h-4 w-4" strokeWidth={2.25} />
          </button>
        )}
      </div>
    </form>
  );
}
