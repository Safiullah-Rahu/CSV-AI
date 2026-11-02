import { Loader2 } from "lucide-react";
import { forwardRef } from "react";

import { cn } from "@/lib/utils";

/*
 * primary   — the one main action on a surface (accent, with a top highlight)
 * ink       — high-emphasis neutral (fg on bg); hero CTAs and auth screens
 * secondary — bordered surface button; the default for everything else
 * outline   — alias of secondary kept for existing call sites
 * ghost     — toolbar / icon actions
 */
const variants = {
  primary:
    "bg-accent text-accent-fg shadow-[inset_0_1px_0_hsl(0_0%_100%/0.2),0_1px_2px_hsl(var(--shadow)/0.18),0_0_0_1px_hsl(var(--accent)/0.9)] [background-image:linear-gradient(180deg,hsl(0_0%_100%/0.12),transparent)] hover:brightness-[1.06] active:brightness-95",
  ink: "bg-fg text-bg shadow-[inset_0_1px_0_hsl(0_0%_100%/0.14),0_1px_2px_hsl(var(--shadow)/0.25)] hover:bg-fg/90 active:bg-fg/85",
  secondary: "border bg-surface text-fg shadow-xs hover:border-border-strong hover:bg-surface-2/60",
  outline: "border bg-surface text-fg shadow-xs hover:border-border-strong hover:bg-surface-2/60",
  ghost: "text-muted hover:bg-surface-2 hover:text-fg",
  danger: "bg-danger text-white shadow-[inset_0_1px_0_hsl(0_0%_100%/0.18),0_1px_2px_hsl(var(--shadow)/0.18)] hover:brightness-110",
  link: "text-accent underline-offset-4 hover:underline px-0",
} as const;

const sizes = {
  xs: "h-7 px-2.5 text-[12px] gap-1 rounded-md [&_svg]:h-3.5 [&_svg]:w-3.5",
  sm: "h-8 px-3 text-[13px] gap-1.5 rounded-lg [&_svg]:h-3.5 [&_svg]:w-3.5",
  md: "h-9 px-3.5 text-[13.5px] gap-2 rounded-lg [&_svg]:h-4 [&_svg]:w-4",
  lg: "h-11 px-5 text-[14.5px] gap-2 rounded-[11px] [&_svg]:h-4 [&_svg]:w-4",
  icon: "h-8 w-8 justify-center rounded-lg [&_svg]:h-4 [&_svg]:w-4",
} as const;

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: keyof typeof variants;
  size?: keyof typeof sizes;
  loading?: boolean;
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className, variant = "primary", size = "md", loading, disabled, children, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={cn(
        "inline-flex select-none items-center justify-center whitespace-nowrap font-medium tracking-[-0.005em] transition-[background,color,filter,box-shadow,border-color] duration-150 focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/35 disabled:pointer-events-none disabled:opacity-50",
        variants[variant],
        sizes[size],
        className,
      )}
      {...props}
    >
      {loading ? <Loader2 className="animate-spin" aria-hidden /> : null}
      {children}
    </button>
  );
});
