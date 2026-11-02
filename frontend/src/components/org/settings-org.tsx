"use client";

import { useMutation } from "@tanstack/react-query";
import { Building2, ImageOff, LayoutGrid, MessageSquareText, Sparkles } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { useTheme } from "@/app/providers";
import { Callout, GatedButton, permissionHint, SettingsSection } from "@/components/org/kit";
import type { ProviderKind } from "@/components/org/types";
import { OrgAvatar } from "@/components/shell/branding";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/dialog";
import { Field, Input } from "@/components/ui/form";
import { Badge, Card, Checkbox, Mono, StatusBadge, Switch } from "@/components/ui/primitives";
import { del, patch } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatDate } from "@/lib/utils";

type Settings = {
  security?: { mfa_required?: boolean; session_ttl_days?: number };
  retention?: { conversations_days?: number | null; audit_days?: number | null; deleted_documents_days?: number | null };
  ai?: { grounding_policy?: "strict" | "balanced" | "open"; allowed_providers?: string[] };
  branding?: { accent?: string | null; logo_url?: string | null };
};

function useSection(section: keyof Settings, label: string) {
  const { reload } = useAuth();
  return useMutation({
    mutationFn: (values: Record<string, unknown>) => patch("/orgs/current/settings", { section, values }),
    onSuccess: async () => {
      await reload();
      toast.success(`${label} saved`);
    },
  });
}

const intOrNull = (s: string) => (s.trim() === "" ? null : Number(s));
const badInt = (s: string, min: number, max: number, allowEmpty: boolean) => {
  if (s.trim() === "") return !allowEmpty;
  const n = Number(s);
  return !Number.isInteger(n) || n < min || n > max;
};

function SectionCard({ title, description, perm, children, onSave, saving, dirty, invalid }: { title: string; description: string; perm: string; children: React.ReactNode; onSave: () => void; saving: boolean; dirty: boolean; invalid?: boolean }) {
  const { can } = useAuth();
  const allowed = can(perm);
  return (
    <SettingsSection title={title} description={description} aside={!allowed ? <Badge>read-only</Badge> : null}>
      <Card>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (allowed && dirty && !invalid) onSave();
          }}
        >
          <fieldset disabled={!allowed} className="space-y-5 px-5 py-5 disabled:opacity-80">
            {children}
          </fieldset>
          <div className="flex flex-wrap items-center justify-end gap-3 rounded-b-xl border-t bg-surface-2/30 px-5 py-3">
            {!allowed ? <span className="mr-auto text-[12px] text-subtle">{permissionHint(perm)} to change this.</span> : dirty ? <span className="mr-auto text-[12px] text-subtle">You have unsaved changes.</span> : null}
            <GatedButton type="submit" size="sm" perm={perm} loading={saving} disabled={!dirty || invalid}>
              Save
            </GatedButton>
          </div>
        </form>
      </Card>
    </SettingsSection>
  );
}

/* ------------------------------------------------------- branding preview */
/** Platform default accent triplets (mirror of --accent in globals.css), used when no brand accent is set. */
const DEFAULT_ACCENT = { light: "244 74% 58%", dark: "244 95% 73%" };

function hexToHsl(hex: string): [number, number, number] {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l * 100];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  const h = max === r ? (g - b) / d + (g < b ? 6 : 0) : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [h * 60, s * 100, l * 100];
}

/** Same per-theme lightness clamp the app applies at runtime (shell/branding.tsx). */
function accentTriplet(hex: string, theme: "light" | "dark") {
  if (!/^#[0-9a-fA-F]{6}$/.test(hex)) return DEFAULT_ACCENT[theme];
  const [h, s, l] = hexToHsl(hex);
  const lightness = theme === "dark" ? Math.max(l, 62) : Math.min(l, 48);
  return `${h.toFixed(0)} ${s.toFixed(0)}% ${lightness.toFixed(0)}%`;
}

function BrandPreview({ name, accent, logo }: { name: string; accent: string; logo: string | null }) {
  const { resolved } = useTheme();
  const triplet = accentTriplet(accent.trim(), resolved);
  const vars = { "--accent": triplet, "--ring": triplet } as React.CSSProperties;
  return (
    <div className="overflow-hidden rounded-xl border" style={vars} aria-hidden>
      <div className="flex items-center gap-2 border-b bg-surface-2/50 px-3.5 py-2">
        <span className="label">Live preview</span>
        <span className="ml-auto font-mono text-[10.5px] text-subtle">{accent.trim() && /^#[0-9a-fA-F]{6}$/.test(accent.trim()) ? accent.trim().toLowerCase() : "platform default"} · {resolved}</span>
      </div>
      <div className="grid sm:grid-cols-[176px_minmax(0,1fr)]">
        <div className="space-y-0.5 border-b bg-frame p-2.5 sm:border-b-0 sm:border-r">
          <div className="mb-2 flex items-center gap-2 px-1.5 py-1">
            <OrgAvatar name={name || "Organization"} logo={logo} className="h-6 w-6 text-[9.5px]" />
            <span className="truncate text-[12.5px] font-semibold">{name || "Organization"}</span>
          </div>
          <span className="flex items-center gap-2 rounded-md px-2 py-1.5 text-[12px] text-muted">
            <LayoutGrid className="h-3.5 w-3.5" /> Overview
          </span>
          <span className="flex items-center gap-2 rounded-md bg-surface px-2 py-1.5 text-[12px] font-medium text-fg shadow-xs ring-1 ring-border">
            <MessageSquareText className="h-3.5 w-3.5 text-accent" /> AI Chat
          </span>
        </div>
        <div className="space-y-3 bg-bg p-4">
          <div>
            <div className="text-[13px] font-semibold tracking-[-0.01em]">Parental leave policy</div>
            <p className="mt-1 text-[12px] leading-relaxed text-muted">
              Answers cite <span className="font-medium text-accent underline underline-offset-2">3 sources</span> from your knowledge base.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-accent px-3 text-[12.5px] font-medium text-accent-fg shadow-[inset_0_1px_0_hsl(0_0%_100%/0.2),0_1px_2px_hsl(var(--shadow)/0.18)]">
              <Sparkles className="h-3.5 w-3.5" /> Ask a question
            </span>
            <Badge tone="accent">grounded</Badge>
            <span className="relative h-5 w-9 rounded-full bg-accent">
              <span className="absolute right-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow-[0_1px_3px_hsl(0_0%_0%/0.25)]" />
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export function OrganizationTab() {
  const { org, can, orgs, reload, switchOrg } = useAuth();
  const router = useRouter();
  const s = (org?.organization.settings ?? {}) as Settings;
  const [name, setName] = useState("");
  useEffect(() => setName(org?.organization.name ?? ""), [org?.organization.name]);

  const rename = useMutation({
    mutationFn: () => patch("/orgs/current", { name: name.trim() }),
    onSuccess: async () => {
      await reload();
      toast.success("Organization renamed");
    },
  });

  /* security */
  const [mfa, setMfa] = useState(false);
  const [ttl, setTtl] = useState("");
  /* retention */
  const [convDays, setConvDays] = useState("");
  const [auditDays, setAuditDays] = useState("");
  /* ai */
  const [policy, setPolicy] = useState<"strict" | "balanced" | "open">("strict");
  const [allowed, setAllowed] = useState<string[]>([]);
  /* branding */
  const [accent, setAccent] = useState("");
  const [logo, setLogo] = useState("");
  const [logoBroken, setLogoBroken] = useState(false);

  useEffect(() => {
    setMfa(Boolean(s.security?.mfa_required));
    setTtl(s.security?.session_ttl_days != null ? String(s.security.session_ttl_days) : "");
    setConvDays(s.retention?.conversations_days != null ? String(s.retention.conversations_days) : "");
    setAuditDays(s.retention?.audit_days != null ? String(s.retention.audit_days) : "");
    setPolicy(s.ai?.grounding_policy ?? "strict");
    setAllowed(s.ai?.allowed_providers ?? []);
    setAccent(s.branding?.accent ?? "");
    setLogo(s.branding?.logo_url ?? "");
  }, [org]); // eslint-disable-line react-hooks/exhaustive-deps

  const kinds = useApi<ProviderKind[]>("/ai/provider-kinds", undefined, { staleTime: 10 * 60_000 });
  const security = useSection("security", "Security policy");
  const retention = useSection("retention", "Retention policy");
  const ai = useSection("ai", "AI policy");
  const branding = useSection("branding", "Branding");

  const [deleting, setDeleting] = useState(false);
  const remove = useMutation({
    mutationFn: () => del<{ detail?: string }>("/orgs/current", { confirm_name: org!.organization.name }),
    onSuccess: async (r) => {
      toast.success(r.detail || "Organization deactivated");
      setDeleting(false);
      const other = orgs.find((o) => o.id !== org?.organization.id && o.status === "active");
      await reload();
      if (other) switchOrg(other.id);
      router.push("/overview");
    },
  });

  const ttlBad = badInt(ttl, 1, 365, false);
  const convBad = badInt(convDays, 1, 3650, true);
  const auditBad = badInt(auditDays, 30, 3650, false);
  const hexBad = accent.trim() !== "" && !/^#[0-9a-fA-F]{6}$/.test(accent.trim());
  let logoBad = false;
  if (logo.trim()) {
    try {
      logoBad = new URL(logo.trim()).protocol !== "https:";
    } catch {
      logoBad = true;
    }
  }
  useEffect(() => setLogoBroken(false), [logo]);

  const sortedAllowed = useMemo(() => [...allowed].sort().join(","), [allowed]);
  if (!org) return null;
  const o = org.organization;
  const details: [string, React.ReactNode][] = [
    ["Slug", <Mono key="slug" className="text-fg">{o.slug}</Mono>],
    ["Plan", org.plan ? <Badge key="plan" tone="accent">{org.plan}</Badge> : "—"],
    ["Active members", <span key="members" className="num font-mono">{org.member_count}</span>],
    ["Created", formatDate(o.created_at, false)],
    ["Your rank", <span key="rank" className="num font-mono">{org.permissions.max_rank}</span>],
    ["Organization id", <Mono key="id" className="block truncate text-[11px] text-subtle" >{o.id}</Mono>],
  ];

  return (
    <div>
      <SettingsSection title="Organization profile" description="The name members see in the organization switcher, emails and exports." aside={<StatusBadge status={o.status} />}>
        <Card className="overflow-hidden">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (name.trim().length >= 2 && name.trim() !== o.name) rename.mutate();
            }}
          >
            <fieldset disabled={!can("org.edit")} className="flex items-start gap-4 px-5 py-5">
              <OrgAvatar name={o.name} logo={s.branding?.logo_url} className="mt-6 h-10 w-10 rounded-xl text-[13px]" />
              <Field label="Organization name" htmlFor="org-name" hint={!can("org.edit") ? permissionHint("org.edit") : "2–200 characters."} className="min-w-0 flex-1">
                <Input id="org-name" value={name} onChange={(e) => setName(e.target.value)} maxLength={200} />
              </Field>
            </fieldset>
            <dl className="grid grid-cols-2 gap-px border-t bg-border sm:grid-cols-3">
              {details.map(([k, v]) => (
                <div key={k} className="min-w-0 bg-surface px-5 py-3">
                  <dt className="label">{k}</dt>
                  <dd className="mt-1.5 min-w-0 truncate text-[13px]">{v}</dd>
                </div>
              ))}
            </dl>
            <div className="flex justify-end border-t bg-surface-2/30 px-5 py-3">
              <GatedButton type="submit" size="sm" perm="org.edit" loading={rename.isPending} disabled={name.trim().length < 2 || name.trim() === o.name}>
                Rename
              </GatedButton>
            </div>
          </form>
        </Card>
      </SettingsSection>

      <SectionCard
        title="Security policy"
        description="Authentication requirements for members of this organization."
        perm="org.security"
        saving={security.isPending}
        dirty={mfa !== Boolean(s.security?.mfa_required) || ttl !== (s.security?.session_ttl_days != null ? String(s.security.session_ttl_days) : "")}
        invalid={ttlBad}
        onSave={() => security.mutate({ mfa_required: mfa, session_ttl_days: Number(ttl) })}
      >
        <label className="flex cursor-pointer items-start justify-between gap-6">
          <span>
            <span className="block text-[13.5px] font-medium">Require two-factor authentication</span>
            <span className="mt-0.5 block text-[12.5px] leading-relaxed text-muted">Members must enable an authenticator app on their account.</span>
          </span>
          <Switch checked={mfa} onCheckedChange={setMfa} label="Require two-factor authentication" />
        </label>
        <div className="border-t pt-5">
          <Field label="Session lifetime (days)" htmlFor="sec-ttl" error={ttlBad ? "Whole number between 1 and 365." : null} hint="How long a sign-in stays valid before members must sign in again.">
            <Input id="sec-ttl" inputMode="numeric" value={ttl} onChange={(e) => setTtl(e.target.value.replace(/[^\d]/g, ""))} className="block w-32 font-mono" />
          </Field>
        </div>
      </SectionCard>

      <SectionCard
        title="Data retention"
        description="How long conversation history and audit events are kept."
        perm="org.settings"
        saving={retention.isPending}
        dirty={convDays !== (s.retention?.conversations_days != null ? String(s.retention.conversations_days) : "") || auditDays !== (s.retention?.audit_days != null ? String(s.retention.audit_days) : "")}
        invalid={convBad || auditBad}
        onSave={() => retention.mutate({ conversations_days: intOrNull(convDays), audit_days: Number(auditDays) })}
      >
        <div className="grid gap-5 sm:grid-cols-2">
          <Field label="Conversations (days)" htmlFor="ret-conv" error={convBad ? "Leave empty or use a whole number from 1 to 3650." : null} hint="Leave empty to keep conversations until users delete them.">
            <Input id="ret-conv" inputMode="numeric" value={convDays} onChange={(e) => setConvDays(e.target.value.replace(/[^\d]/g, ""))} placeholder="Keep forever" className="block w-40 font-mono placeholder:font-sans" />
          </Field>
          <Field label="Audit events (days)" htmlFor="ret-audit" error={auditBad ? "Whole number from 30 to 3650." : null} hint="Minimum 30 days. Many compliance regimes expect 365 or more.">
            <Input id="ret-audit" inputMode="numeric" value={auditDays} onChange={(e) => setAuditDays(e.target.value.replace(/[^\d]/g, ""))} className="block w-40 font-mono" />
          </Field>
        </div>
      </SectionCard>

      <SectionCard
        title="AI policy"
        description="Default answer grounding and which model providers may be used."
        perm="admin.ai_providers"
        saving={ai.isPending}
        dirty={policy !== (s.ai?.grounding_policy ?? "strict") || sortedAllowed !== [...(s.ai?.allowed_providers ?? [])].sort().join(",")}
        onSave={() => ai.mutate({ grounding_policy: policy, allowed_providers: allowed })}
      >
        <div>
          <div className="mb-2 text-[13px] font-medium">Grounding</div>
          <div role="radiogroup" aria-label="Grounding policy" className="grid gap-2 sm:grid-cols-3">
            {(
              [
                ["strict", "Strict", "Answers use retrieved sources only — no general knowledge."],
                ["balanced", "Balanced", "Sourced answer plus clearly separated, uncited suggestions."],
                ["open", "Open", "General knowledge may fill gaps, labelled as such."],
              ] as const
            ).map(([value, label, hint]) => (
              <button
                key={value}
                type="button"
                role="radio"
                aria-checked={policy === value}
                onClick={() => setPolicy(value)}
                className={cn("group relative rounded-xl border p-3.5 text-left transition-all", policy === value ? "border-accent/60 bg-accent/[0.05] shadow-[0_0_0_3px_hsl(var(--accent)/0.12)]" : "hover:border-border-strong hover:bg-surface-2/50")}
              >
                <span className="flex items-center justify-between gap-2">
                  <span className="text-[13px] font-semibold">{label}</span>
                  <span className={cn("grid h-4 w-4 place-items-center rounded-full border transition-colors", policy === value ? "border-accent bg-accent" : "border-border-strong bg-surface")}>
                    {policy === value ? <span className="h-1.5 w-1.5 rounded-full bg-accent-fg" /> : null}
                  </span>
                </span>
                <span className="mt-1 block text-[12px] leading-relaxed text-muted">{hint}</span>
              </button>
            ))}
          </div>
          <p className="mt-2 text-[12px] text-subtle">Organization default. A knowledge base&apos;s own grounding setting, or a policy sent with a chat request, takes precedence.</p>
        </div>
        <div className="border-t pt-5">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <div className="text-[13px] font-medium">Allowed provider types</div>
            <span className="text-[12px] text-muted">{allowed.length ? `${allowed.length} selected — other provider types are blocked at request time.` : "None selected — every provider type is allowed."}</span>
          </div>
          <div className="mt-2.5 grid gap-1 sm:grid-cols-2">
            {(kinds.data ?? []).map((k) => (
              <label key={k.kind} className={cn("flex cursor-pointer items-center gap-2.5 rounded-lg border px-2.5 py-2 text-[13px] transition-colors", allowed.includes(k.kind) ? "border-accent/30 bg-accent/[0.04]" : "border-transparent hover:bg-surface-2/60")}>
                <Checkbox checked={allowed.includes(k.kind)} onCheckedChange={(v) => setAllowed((a) => (v ? [...a, k.kind] : a.filter((x) => x !== k.kind)))} label={k.label} />
                <span className="truncate">{k.label}</span>
              </label>
            ))}
          </div>
        </div>
      </SectionCard>

      <SectionCard
        title="Branding"
        description="Accent colour and logo members see across the app. The preview uses the same contrast adjustments the app applies in light and dark themes."
        perm="org.settings"
        saving={branding.isPending}
        dirty={accent.trim() !== (s.branding?.accent ?? "") || logo.trim() !== (s.branding?.logo_url ?? "")}
        invalid={hexBad || logoBad}
        onSave={() => branding.mutate({ accent: accent.trim() || null, logo_url: logo.trim() || null })}
      >
        <div className="grid gap-5 sm:grid-cols-[auto_minmax(0,1fr)]">
          <Field label="Accent colour" htmlFor="brand-accent" error={hexBad ? "Use a 6-digit hex colour like #2563eb." : null} hint="Leave empty to use the platform default.">
            <div className="flex items-center gap-2">
              <label className="relative h-9 w-9 shrink-0 cursor-pointer overflow-hidden rounded-lg border shadow-xs" style={{ background: /^#[0-9a-fA-F]{6}$/.test(accent) ? accent : "hsl(var(--accent))" }}>
                <input type="color" aria-label="Pick accent colour" value={/^#[0-9a-fA-F]{6}$/.test(accent) ? accent : "#2563eb"} onChange={(e) => setAccent(e.target.value)} className="absolute inset-0 h-full w-full cursor-pointer opacity-0" />
              </label>
              <Input id="brand-accent" value={accent} onChange={(e) => setAccent(e.target.value)} placeholder="#2563eb" className="w-32 font-mono text-[13px]" maxLength={7} />
              {accent ? (
                <Button type="button" variant="ghost" size="xs" onClick={() => setAccent("")}>
                  Reset
                </Button>
              ) : null}
            </div>
          </Field>
          <Field label="Logo URL" htmlFor="brand-logo" error={logoBad ? "Enter a full https:// URL." : null} hint="Shown in the sidebar for every member. A square PNG or SVG works best.">
            <Input id="brand-logo" value={logo} onChange={(e) => setLogo(e.target.value)} placeholder="https://cdn.example.com/logo.svg" className="font-mono text-[13px]" />
          </Field>
        </div>
        {logo.trim() && !logoBad ? (
          <div className="flex items-center gap-3 rounded-xl border bg-surface-2/40 p-3">
            {logoBroken ? (
              <span className="grid h-10 w-10 place-items-center rounded-lg border bg-surface text-subtle">
                <ImageOff className="h-4 w-4" />
              </span>
            ) : (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={logo.trim()} alt="Logo preview" className="h-10 w-10 rounded-lg border bg-surface object-contain" onError={() => setLogoBroken(true)} />
            )}
            <span className="text-[12px] text-muted">{logoBroken ? "Preview unavailable — the image could not be loaded here (it may be blocked by this app's content security policy)." : "Logo preview"}</span>
          </div>
        ) : null}
        <BrandPreview name={o.name} accent={hexBad ? "" : accent} logo={logo.trim() && !logoBad && !logoBroken ? logo.trim() : null} />
      </SectionCard>

      <SettingsSection title={<span className="text-danger">Danger zone</span>} description="Deactivates the organization immediately and schedules a permanent purge of all its knowledge, documents, conversations and settings.">
        <Card className="border-danger/25">
          <div className="space-y-4 px-5 py-5">
            <Callout tone="danger">Members lose access right away and the purge cannot be undone. Export anything you need first.</Callout>
            <div className="flex flex-wrap items-center gap-3">
              <div className="min-w-0 flex-1 basis-56">
                <div className="text-[14px] font-medium">Delete organization</div>
                <p className="mt-0.5 text-[12.5px] text-muted">Only an organization owner can do this. You&apos;ll be asked to type the organization name.</p>
              </div>
              <GatedButton variant="danger" size="sm" perm="org.billing" reason="Only an organization owner (org.billing) can delete the organization." onClick={() => setDeleting(true)}>
                <Building2 className="h-3.5 w-3.5" /> Delete organization
              </GatedButton>
            </div>
          </div>
        </Card>
      </SettingsSection>
      <ConfirmDialog
        open={deleting}
        onOpenChange={setDeleting}
        title={`Delete ${o.name}?`}
        description="The organization is deactivated now and all of its data is purged by a background job."
        confirmLabel="Delete organization"
        requireText={o.name}
        loading={remove.isPending}
        onConfirm={() => remove.mutate()}
      />
    </div>
  );
}
