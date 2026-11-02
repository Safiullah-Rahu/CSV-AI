import type { Config } from "tailwindcss";

const token = (name: string) => `hsl(var(--${name}) / <alpha-value>)`;

const config: Config = {
  darkMode: ["class"],
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    container: { center: true, padding: "1rem" },
    extend: {
      colors: {
        frame: token("frame"),
        bg: token("bg"),
        surface: token("surface"),
        "surface-2": token("surface-2"),
        border: token("border"),
        "border-strong": token("border-strong"),
        fg: token("fg"),
        muted: token("muted"),
        subtle: token("subtle"),
        accent: token("accent"),
        "accent-fg": token("accent-fg"),
        ok: token("ok"),
        warn: token("warn"),
        danger: token("danger"),
        info: token("info"),
        ring: token("ring"),
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
        display: ["var(--font-display)", "ui-serif", "Georgia", "serif"],
      },
      fontSize: { "2xs": ["0.6875rem", { lineHeight: "1rem" }] },
      borderRadius: { md: "7px", lg: "9px", xl: "13px", "2xl": "18px", "3xl": "24px" },
      boxShadow: {
        xs: "0 1px 1.5px hsl(var(--shadow) / 0.05)",
        card: "0 1px 2px hsl(var(--shadow) / 0.04), 0 2px 8px -2px hsl(var(--shadow) / 0.04), inset 0 1px 0 hsl(var(--highlight))",
        lift: "0 2px 4px hsl(var(--shadow) / 0.05), 0 12px 24px -8px hsl(var(--shadow) / 0.12), inset 0 1px 0 hsl(var(--highlight))",
        panel: "0 0 0 1px hsl(var(--shadow) / 0.02), 0 1px 3px hsl(var(--shadow) / 0.05)",
        pop: "0 1px 2px hsl(var(--shadow) / 0.06), 0 16px 40px -12px hsl(var(--shadow) / 0.28)",
        float: "0 0 0 1px hsl(var(--shadow) / 0.04), 0 24px 64px -16px hsl(var(--shadow) / 0.35), 0 4px 12px -4px hsl(var(--shadow) / 0.12)",
        glow: "0 0 0 1px hsl(var(--accent) / 0.35), 0 0 24px -4px hsl(var(--accent) / 0.45)",
        "inset-hl": "inset 0 1px 0 hsl(0 0% 100% / 0.16)",
      },
      keyframes: {
        "fade-in": { from: { opacity: "0", transform: "translateY(4px)" }, to: { opacity: "1", transform: "none" } },
        "fade-up": { from: { opacity: "0", transform: "translateY(10px)" }, to: { opacity: "1", transform: "none" } },
        "scale-in": { from: { opacity: "0", transform: "scale(0.97)" }, to: { opacity: "1", transform: "none" } },
        shimmer: { "100%": { transform: "translateX(100%)" } },
        pulseRing: { "0%": { transform: "scale(0.9)", opacity: "0.7" }, "100%": { transform: "scale(1.6)", opacity: "0" } },
        orbit: { to: { transform: "rotate(360deg)" } },
      },
      animation: {
        "fade-in": "fade-in 180ms ease-out",
        "fade-up": "fade-up 420ms cubic-bezier(0.2, 0.7, 0.2, 1) both",
        "scale-in": "scale-in 160ms cubic-bezier(0.2, 0.7, 0.2, 1)",
        shimmer: "shimmer 1.6s infinite",
        "pulse-ring": "pulseRing 2.4s cubic-bezier(0.2,0.6,0.4,1) infinite",
        orbit: "orbit 40s linear infinite",
      },
    },
  },
  plugins: [],
};

export default config;
