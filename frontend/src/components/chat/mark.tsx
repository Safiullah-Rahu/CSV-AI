import styles from "@/components/chat/chat.module.css";
import { cn } from "@/lib/utils";

/**
 * The product mark used as the assistant's avatar (same geometry as the shell
 * LogoMark). While an answer is in flight the accent satellite orbits.
 */
export function AssistantMark({ className, live }: { className?: string; live?: boolean }) {
  return (
    <svg viewBox="0 0 32 32" className={cn("h-7 w-7 shrink-0", className)} aria-hidden>
      <rect width="32" height="32" rx="9" className="fill-fg" />
      <circle cx="16" cy="16" r="7.25" fill="none" className="stroke-bg" strokeWidth="1.6" opacity=".55" />
      <circle cx="16" cy="16" r="2.6" className="fill-bg" />
      <g className={live ? styles.orbit : undefined}>
        <circle cx="21.13" cy="10.87" r="2.3" className="fill-accent" />
      </g>
    </svg>
  );
}
