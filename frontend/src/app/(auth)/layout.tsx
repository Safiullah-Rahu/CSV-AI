import { Logo } from "@/components/shell/sidebar";

const PRINCIPLES = [
  ["Isolated by design", "Postgres row-level security on every tenant table, enforced below the application."],
  ["Grounded answers", "Every claim cites a passage the asker is allowed to read, or the assistant declines."],
  ["Read-only SQL", "Generated queries are parsed, allow-listed, sandboxed and written to the audit log."],
] as const;

export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid min-h-screen bg-bg lg:grid-cols-[1.08fr_1fr]">
      {/* Brand panel: always graphite (the `dark` class scopes the dark tokens). */}
      <div className="dark relative hidden overflow-hidden bg-frame text-fg lg:block">
        <div className="absolute inset-0 bg-grid opacity-60 [mask-image:radial-gradient(ellipse_at_30%_40%,black,transparent_75%)]" />
        <div className="absolute -left-40 -top-40 h-[520px] w-[520px] rounded-full bg-accent/20 blur-[120px]" />
        <Orbit />
        <div className="relative flex h-full flex-col justify-between p-12">
          <Logo />
          <div className="max-w-[520px]">
            <div className="label mb-5 flex items-center gap-2">
              <span className="h-1 w-3 rounded-full bg-accent" /> Enterprise knowledge intelligence
            </div>
            <h2 className="font-display text-[56px] leading-[0.98] tracking-[-0.02em]">
              Every answer, <span className="italic text-accent">traced</span> to its source.
            </h2>
            <p className="mt-5 max-w-md text-[15px] leading-relaxed text-muted">
              Documents, databases and connected tools in one permission-aware workspace. People only ever get answers from what they are allowed to see.
            </p>
            <ol className="mt-10 divide-y divide-border border-y">
              {PRINCIPLES.map(([title, body], i) => (
                <li key={title} className="flex gap-5 py-4">
                  <span className="font-mono text-[11px] text-subtle">{String(i + 1).padStart(2, "0")}</span>
                  <div>
                    <div className="text-[14px] font-medium">{title}</div>
                    <div className="mt-0.5 text-[13px] leading-relaxed text-muted">{body}</div>
                  </div>
                </li>
              ))}
            </ol>
          </div>
          <p className="font-mono text-[11px] text-subtle">Knowledge Intelligence · v3.0</p>
        </div>
      </div>
      <div className="relative flex items-center justify-center px-5 py-14">
        <div className="absolute inset-0 bg-dots opacity-50 [mask-image:radial-gradient(ellipse_at_center,black,transparent_70%)] lg:hidden" />
        <div className="relative w-full max-w-[380px] animate-fade-up">
          <div className="mb-10 lg:hidden">
            <Logo />
          </div>
          {children}
        </div>
      </div>
    </div>
  );
}

/** Slow orbit illustration, anchored bottom-right so it never sits under the copy. */
function Orbit() {
  return (
    <div
      className="pointer-events-none absolute -bottom-[280px] -right-[280px] h-[720px] w-[720px] opacity-90 [mask-image:linear-gradient(115deg,transparent_28%,black_62%)]"
      aria-hidden
    >
      <div className="absolute inset-0 rounded-full border border-border" />
      <div className="absolute inset-[90px] rounded-full border border-dashed border-border-strong/60" />
      <div className="absolute inset-[180px] rounded-full border border-border" />
      <div className="absolute inset-[270px] rounded-full bg-accent/10 blur-2xl" />
      <div className="absolute inset-0 animate-orbit">
        <span className="absolute left-1/2 top-0 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent shadow-[0_0_24px_4px_hsl(var(--accent)/0.6)]" />
      </div>
      <div className="absolute inset-[90px] animate-orbit [animation-direction:reverse] [animation-duration:60s]">
        <span className="absolute bottom-0 left-1/2 h-2 w-2 -translate-x-1/2 translate-y-1/2 rounded-full bg-fg/70" />
      </div>
      <div className="absolute inset-[180px] animate-orbit [animation-duration:28s]">
        <span className="absolute left-0 top-1/2 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-ok" />
      </div>
      <div className="absolute inset-0 grid place-items-center">
        <span className="h-4 w-4 rounded-full bg-accent shadow-[0_0_0_10px_hsl(var(--accent)/0.12),0_0_60px_12px_hsl(var(--accent)/0.45)]" />
      </div>
    </div>
  );
}
