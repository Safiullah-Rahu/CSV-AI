"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, ArrowRight, Check, Lock, PlugZap, Search, ShieldCheck, XCircle } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Select } from "@/components/ui/form";
import { Badge, Skeleton, Tip } from "@/components/ui/primitives";
import { post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn } from "@/lib/utils";

import { ACCESS_MODES, EngineTile, FormSection, GatedButton, Notice, ProviderStatusBadge, RadioCards, SYNC_INTERVALS, invalidatePrefix, useKnowledgeBases, writableKbs } from "./shared";
import type { AccessMode, DbConnection, DbProvider, TestResult } from "./types";

export const CATEGORY_LABELS: Record<string, string> = {
  database: "Databases",
  warehouse: "Data warehouses",
  nosql: "NoSQL",
  vector: "Vector stores",
};
const CATEGORY_ORDER = ["database", "warehouse", "nosql", "vector"];

export const SSL_MODES = [
  { value: "disable", label: "disable — no TLS" },
  { value: "allow", label: "allow — TLS if the server insists" },
  { value: "prefer", label: "prefer — TLS when available (default)" },
  { value: "require", label: "require — always TLS, no verification" },
  { value: "verify-ca", label: "verify-ca — TLS + verify the CA" },
  { value: "verify-full", label: "verify-full — TLS + verify CA and hostname" },
];

type FormState = {
  name: string;
  host: string;
  port: string;
  database: string;
  username: string;
  password: string;
  ssl_mode: string;
  access_mode: AccessMode;
  kb_id: string;
  sync_interval: number;
};

const EMPTY: FormState = { name: "", host: "", port: "", database: "", username: "", password: "", ssl_mode: "prefer", access_mode: "live", kb_id: "", sync_interval: 0 };

export function AddConnectionDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  const [step, setStep] = useState<"provider" | "details" | "done">("provider");
  const [provider, setProvider] = useState<DbProvider | null>(null);
  const [form, setForm] = useState<FormState>(EMPTY);
  const [created, setCreated] = useState<DbConnection | null>(null);
  const [pending, setPending] = useState(false);

  const reset = () => {
    setStep("provider");
    setProvider(null);
    setForm(EMPTY);
    setCreated(null);
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => {
        onOpenChange(v);
        if (!v) setTimeout(reset, 200);
      }}
    >
      <DialogContent
        wide
        title="Add database connection"
        description={<Stepper step={step} />}
        footer={
          step === "provider" ? (
            <>
              <Button variant="secondary" onClick={() => onOpenChange(false)}>
                Cancel
              </Button>
              <Button disabled={!provider} onClick={() => setStep("details")}>
                Continue <ArrowRight className="h-4 w-4" />
              </Button>
            </>
          ) : step === "details" ? (
            <DetailsFooter pending={pending} onBack={() => setStep("provider")} />
          ) : null
        }
      >
        {step === "provider" ? (
          <ProviderPicker
            selected={provider?.key ?? null}
            onSelect={setProvider}
            onConfirm={(p) => {
              setProvider(p);
              setStep("details");
            }}
          />
        ) : step === "details" && provider ? (
          <DetailsForm
            provider={provider}
            form={form}
            setForm={setForm}
            onPending={setPending}
            onCreated={(c) => {
              setCreated(c);
              setStep("done");
            }}
          />
        ) : created ? (
          <CreatedStep connection={created} onClose={() => onOpenChange(false)} />
        ) : null}
      </DialogContent>
    </Dialog>
  );
}

/* --------------------------------------------------------------- stepper */

const STEPS = [
  { key: "provider", label: "Engine" },
  { key: "details", label: "Details" },
  { key: "done", label: "Test" },
] as const;

function Stepper({ step }: { step: "provider" | "details" | "done" }) {
  const idx = STEPS.findIndex((s) => s.key === step);
  return (
    <span className="mt-3.5 flex w-[250px] items-center gap-2 sm:w-[400px]" aria-label={`Step ${idx + 1} of ${STEPS.length}: ${STEPS[idx].label}`}>
      {STEPS.map((s, i) => (
        <span key={s.key} className={cn("flex min-w-0 items-center gap-2", i < STEPS.length - 1 && "flex-1")}>
          <span
            className={cn(
              "grid h-5 w-5 shrink-0 place-items-center rounded-full font-mono text-[10.5px] font-semibold transition-colors",
              i < idx ? "bg-accent text-accent-fg" : i === idx ? "bg-fg text-bg" : "border border-border-strong bg-surface text-subtle",
            )}
            aria-hidden
          >
            {i < idx ? <Check className="h-3 w-3" strokeWidth={3} /> : i + 1}
          </span>
          <span className={cn("shrink-0 whitespace-nowrap text-[12.5px] font-medium", i === idx ? "text-fg" : i < idx ? "text-muted" : "text-subtle", i !== idx && "max-sm:hidden")}>{s.label}</span>
          {i < STEPS.length - 1 ? <span className={cn("h-px min-w-4 flex-1 rounded-full", i < idx ? "bg-accent/50" : "bg-border")} aria-hidden /> : null}
        </span>
      ))}
    </span>
  );
}

/* ---------------------------------------------------------------- step 1 */

function ProviderPicker({ selected, onSelect, onConfirm }: { selected: string | null; onSelect: (p: DbProvider) => void; onConfirm: (p: DbProvider) => void }) {
  const providers = useApi<DbProvider[]>("/database-connections/providers", undefined, { staleTime: 5 * 60_000 });
  const [q, setQ] = useState("");
  const groups = useMemo(() => {
    const list = (providers.data ?? []).filter((p) => !q || p.label.toLowerCase().includes(q.toLowerCase()) || p.key.includes(q.toLowerCase()));
    const by: Record<string, DbProvider[]> = {};
    list.forEach((p) => (by[p.category] ||= []).push(p));
    const rank = { tested: 0, beta: 1, planned: 2 } as Record<string, number>;
    Object.values(by).forEach((arr) => arr.sort((a, b) => (rank[a.status] ?? 3) - (rank[b.status] ?? 3) || a.label.localeCompare(b.label)));
    return [...CATEGORY_ORDER, ...Object.keys(by).filter((k) => !CATEGORY_ORDER.includes(k))].filter((k) => by[k]?.length).map((k) => ({ key: k, items: by[k] }));
  }, [providers.data, q]);

  if (providers.isLoading) {
    return (
      <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 9 }).map((_, i) => (
          <Skeleton key={i} className="h-[88px] rounded-xl" />
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="space-y-3">
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-subtle" />
          <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search engines…" className="h-9 pl-8" aria-label="Search providers" autoFocus />
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11.5px] text-subtle">
          <span className="inline-flex items-center gap-1.5"><ProviderStatusBadge status="tested" /> verified against a live server by the test suite</span>
          <span className="inline-flex items-center gap-1.5"><ProviderStatusBadge status="beta" /> implemented, not yet verified live</span>
          <span className="inline-flex items-center gap-1.5"><ProviderStatusBadge status="planned" /> not connectable yet</span>
        </div>
      </div>
      {groups.map((g) => (
        <section key={g.key} aria-labelledby={`cat-${g.key}`}>
          <h4 id={`cat-${g.key}`} className="label mb-2.5 flex items-center gap-2">
            {CATEGORY_LABELS[g.key] ?? g.key}
            <span className="h-px flex-1 bg-border" aria-hidden />
            <span className="num">{g.items.length}</span>
          </h4>
          <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
            {g.items.map((p) => {
              const planned = p.status === "planned";
              const active = selected === p.key;
              const card = (
                <button
                  type="button"
                  disabled={planned}
                  aria-pressed={active}
                  onClick={() => onSelect(p)}
                  onDoubleClick={() => !planned && onConfirm(p)}
                  className={cn(
                    "relative flex h-full w-full items-start gap-3 rounded-xl border p-3 text-left transition-[border-color,background,box-shadow] duration-150",
                    planned ? "cursor-not-allowed border-dashed bg-transparent" : "bg-surface shadow-xs hover:border-border-strong hover:shadow-card",
                    active && "border-accent/50 bg-accent/[0.04] shadow-[0_0_0_3px_hsl(var(--accent)/0.12)] hover:border-accent/60",
                  )}
                >
                  <EngineTile category={p.category} active={active} muted={planned} />
                  <span className="min-w-0 flex-1">
                    <span className={cn("block truncate pr-5 text-[13.5px] font-medium tracking-[-0.01em]", planned && "text-muted")} title={p.label}>{p.label}</span>
                    <span className="mt-0.5 block truncate font-mono text-[11px] text-subtle">
                      {p.family}
                      {p.port ? ` · :${p.port}` : ""}
                    </span>
                    <span className="mt-2 flex flex-wrap items-center gap-1.5">
                      <ProviderStatusBadge status={p.status} />
                      {planned ? (
                        <span className="inline-flex min-w-0 items-center gap-1 text-[11px] text-subtle">
                          <Lock className="h-3 w-3 shrink-0" />
                          <span className="truncate">{p.requires ? `requires ${p.requires}` : "not available yet"}</span>
                        </span>
                      ) : !p.driver_installed ? (
                        <span className="text-[11px] text-warn">Driver not installed on this server</span>
                      ) : null}
                    </span>
                  </span>
                  {active ? (
                    <span className="absolute right-2.5 top-2.5 grid h-4 w-4 place-items-center rounded-full bg-accent text-accent-fg" aria-hidden>
                      <Check className="h-2.5 w-2.5" strokeWidth={3.5} />
                    </span>
                  ) : null}
                </button>
              );
              return planned ? (
                <Tip key={p.key} content={`${p.label} is planned and cannot be connected yet${p.requires ? ` (requires ${p.requires})` : ""}.`}>
                  <span className="block">{card}</span>
                </Tip>
              ) : (
                <div key={p.key}>{card}</div>
              );
            })}
          </div>
        </section>
      ))}
      {!groups.length ? <p className="rounded-xl border border-dashed py-8 text-center text-[13px] text-muted">No engines match “{q}”.</p> : null}
    </div>
  );
}

/* ---------------------------------------------------------------- step 2 */

const FORM_ID = "add-connection-form";

function DetailsFooter({ onBack, pending }: { onBack: () => void; pending: boolean }) {
  return (
    <>
      <Button variant="ghost" onClick={onBack} className="mr-auto">
        <ArrowLeft className="h-4 w-4" /> Back
      </Button>
      <Button type="submit" form={FORM_ID} loading={pending}>
        Create connection
      </Button>
    </>
  );
}

function DetailsForm({
  provider,
  form,
  setForm,
  onCreated,
  onPending,
}: {
  provider: DbProvider;
  form: FormState;
  setForm: React.Dispatch<React.SetStateAction<FormState>>;
  onCreated: (c: DbConnection) => void;
  onPending: (v: boolean) => void;
}) {
  const qc = useQueryClient();
  const kbs = useKnowledgeBases();
  const writable = writableKbs(kbs.data);
  const [errors, setErrors] = useState<Partial<Record<keyof FormState, string>>>({});
  const sqlite = provider.key === "sqlite";
  const set = <K extends keyof FormState>(k: K, v: FormState[K]) => setForm((f) => ({ ...f, [k]: v }));

  const create = useMutation({
    mutationFn: () =>
      post<DbConnection>("/database-connections", {
        name: form.name.trim(),
        provider: provider.key,
        host: sqlite ? "" : form.host.trim(),
        port: form.port ? Number(form.port) : null,
        database: form.database.trim(),
        username: sqlite ? "" : form.username.trim(),
        password: !sqlite && form.password ? form.password : null,
        ssl_mode: form.ssl_mode,
        access_mode: form.access_mode,
        kb_id: form.access_mode !== "live" && form.kb_id ? form.kb_id : null,
        sync_interval_minutes: form.access_mode !== "live" && form.sync_interval ? form.sync_interval : null,
      }),
    onSuccess: (c) => {
      toast.success(`Connection “${c.name}” created`);
      invalidatePrefix(qc, "/database-connections", "/connections");
      setForm((f) => ({ ...f, password: "" }));
      onCreated(c);
    },
  });

  useEffect(() => onPending(create.isPending), [create.isPending, onPending]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const errs: typeof errors = {};
    if (form.name.trim().length < 2) errs.name = "Give the connection a name (at least 2 characters).";
    if (!sqlite && !form.host.trim()) errs.host = "Host is required.";
    if (sqlite && !form.database.trim()) errs.database = "Path to the SQLite file is required.";
    if (form.port && (!/^\d+$/.test(form.port) || Number(form.port) < 1 || Number(form.port) > 65535)) errs.port = "Port must be between 1 and 65535.";
    setErrors(errs);
    if (Object.keys(errs).length) return;
    create.mutate();
  };

  return (
    <form id={FORM_ID} onSubmit={submit} className="space-y-5" noValidate>
      <div className="flex items-center gap-3 rounded-xl border bg-surface-2/50 p-3">
        <EngineTile category={provider.category} active />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[13.5px] font-medium tracking-[-0.01em]">{provider.label}</span>
            <ProviderStatusBadge status={provider.status} />
          </div>
          <div className="mt-0.5 truncate text-[12px] text-muted">
            <span className="font-mono text-[11.5px]">{provider.family}</span>
            {provider.status === "beta" ? " · Beta — not yet verified against a live server." : " · Tested against a live server."}
          </div>
        </div>
      </div>
      {!provider.driver_installed ? (
        <Notice tone="warn" title="Driver not installed on this server">
          You can save this connection, but tests and queries will fail until an administrator installs the {provider.label} driver (the backend&apos;s <span className="font-mono">connectors</span> extra).
        </Notice>
      ) : null}

      <FormSection title="Identity" description="How this connection appears across the workspace.">
        <Field label="Connection name" htmlFor="c-name" error={errors.name}>
          <Input id="c-name" value={form.name} onChange={(e) => set("name", e.target.value)} placeholder={`e.g. ${provider.label} — analytics replica`} autoFocus maxLength={160} />
        </Field>
      </FormSection>

      <FormSection title={sqlite ? "Database file" : "Endpoint & credentials"} description={sqlite ? "The file is opened read-only." : "Use a read-only role. The password is encrypted and never shown again."}>
        {sqlite ? (
          <Field label="Database file path" htmlFor="c-db" error={errors.database} hint="Absolute path on the server. It must be inside a directory an administrator allowed (SQLITE_ALLOWED_DIRS); the file is opened read-only.">
            <Input id="c-db" value={form.database} onChange={(e) => set("database", e.target.value)} placeholder="/data/sqlite/sales.db" className="font-mono text-[13px]" />
          </Field>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Host" htmlFor="c-host" error={errors.host} hint="Private and loopback addresses are blocked unless an administrator allowed them.">
              <Input id="c-host" value={form.host} onChange={(e) => set("host", e.target.value)} placeholder="db.example.com" autoComplete="off" spellCheck={false} />
            </Field>
            <Field label="Port" htmlFor="c-port" error={errors.port} hint={provider.port ? `Default ${provider.port}` : undefined}>
              <Input id="c-port" inputMode="numeric" value={form.port} onChange={(e) => set("port", e.target.value.replace(/[^\d]/g, ""))} placeholder={provider.port ? String(provider.port) : ""} />
            </Field>
            <Field label={provider.family === "mongodb" ? "Database" : "Database name"} htmlFor="c-database">
              <Input id="c-database" value={form.database} onChange={(e) => set("database", e.target.value)} placeholder="analytics" autoComplete="off" spellCheck={false} />
            </Field>
            <Field label="Username" htmlFor="c-user" hint="Use a read-only role; every query also runs in a read-only transaction.">
              <Input id="c-user" value={form.username} onChange={(e) => set("username", e.target.value)} placeholder="readonly_user" autoComplete="off" spellCheck={false} />
            </Field>
            <Field label="Password" htmlFor="c-pass" hint="Encrypted at rest and never shown again.">
              <Input id="c-pass" type="password" value={form.password} onChange={(e) => set("password", e.target.value)} autoComplete="new-password" />
            </Field>
            <Field label={provider.family === "mongodb" ? "TLS" : "SSL mode"} htmlFor="c-ssl" hint={provider.family === "mongodb" ? "TLS is used for require and verify-full." : undefined}>
              <Select id="c-ssl" value={form.ssl_mode} onChange={(e) => set("ssl_mode", e.target.value)}>
                {SSL_MODES.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
        )}
      </FormSection>

      <FormSection title="Access mode" description="You can change this later from the connection's Edit dialog.">
        <RadioCards name="Access mode" layout="stack" value={form.access_mode} onChange={(v) => set("access_mode", v)} options={ACCESS_MODES} />
      </FormSection>

      {form.access_mode !== "live" ? (
        <FormSection title="Synchronisation" description="Where synchronised tables are indexed, and how often.">
          <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Target knowledge base" htmlFor="c-kb" hint={writable.length ? "Synchronised tables become documents in this knowledge base. Required before the first sync." : undefined}>
            {kbs.isLoading ? (
              <Skeleton className="h-9" />
            ) : writable.length ? (
              <Select id="c-kb" value={form.kb_id} onChange={(e) => set("kb_id", e.target.value)}>
                <option value="">Choose later</option>
                {writable.map((k) => (
                  <option key={k.id} value={k.id}>
                    {k.name}
                  </option>
                ))}
              </Select>
            ) : (
              <div className="rounded-lg border border-dashed px-3 py-2 text-xs text-muted">
                No knowledge base you can write to.{" "}
                <Link href="/knowledge-bases" className="font-medium text-accent hover:underline">
                  Create one
                </Link>{" "}
                first, or choose it later.
              </div>
            )}
          </Field>
          <Field label="Sync schedule" htmlFor="c-int" hint="Scheduled syncs are incremental when an incremental column is set.">
            <Select id="c-int" value={form.sync_interval} onChange={(e) => set("sync_interval", Number(e.target.value))}>
              {SYNC_INTERVALS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </Select>
          </Field>
          </div>
        </FormSection>
      ) : null}
    </form>
  );
}

/* ---------------------------------------------------------------- step 3 */

function CreatedStep({ connection, onClose }: { connection: DbConnection; onClose: () => void }) {
  const { can } = useAuth();
  const router = useRouter();
  const qc = useQueryClient();
  const [result, setResult] = useState<TestResult | null>(null);
  const test = useMutation({
    mutationFn: () => post<TestResult>(`/database-connections/${connection.id}/test`),
    onSuccess: (r) => {
      setResult(r);
      invalidatePrefix(qc, "/database-connections");
      if (r.ok) toast.success(`Connected in ${r.latency_ms} ms`);
    },
  });
  const open = () => {
    onClose();
    router.push(`/database-connections/${connection.id}${result?.ok ? "?tab=schema" : ""}`);
  };
  return (
    <div className="space-y-5">
      <div className="flex items-start gap-4">
        <div className="relative shrink-0">
          <EngineTile category={connection.category} size="lg" />
          <span className="absolute -bottom-1 -right-1 grid h-5 w-5 place-items-center rounded-full bg-ok text-white ring-2 ring-surface" aria-hidden>
            <Check className="h-3 w-3" strokeWidth={3} />
          </span>
        </div>
        <div className="min-w-0">
          <div className="text-[16px] font-semibold tracking-[-0.02em]">“{connection.name}” saved</div>
          <div className="mt-0.5 truncate font-mono text-[12px] text-muted">{connection.endpoint}</div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            <Badge>
              <span className="font-mono text-[11px]">{connection.provider_label}</span>
            </Badge>
            <Badge tone={connection.has_credentials ? "ok" : "neutral"}>
              {connection.has_credentials ? <ShieldCheck className="h-3 w-3" /> : null}
              {connection.has_credentials ? "Password stored" : "No password"}
            </Badge>
            <Badge tone="info">{ACCESS_MODES.find((m) => m.key === connection.access_mode)?.label}</Badge>
          </div>
        </div>
      </div>

      <div className="rounded-xl border bg-surface-2/40 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0 flex-1 basis-60">
            <div className="text-[13.5px] font-medium">Test the connection now</div>
            <p className="mt-0.5 text-[12.5px] leading-relaxed text-muted">Opens a read-only session and runs a trivial query. The result is recorded in the health history.</p>
          </div>
          <GatedButton allowed={can("db.test")} perm="db.test" onClick={() => test.mutate()} loading={test.isPending}>
            <PlugZap className="h-4 w-4" /> Test connection
          </GatedButton>
        </div>
        {result ? (
          <div className="mt-3">
            {result.ok ? (
              <Notice tone="ok" title="Connection succeeded">
                Reached the server in <span className="num font-medium text-fg">{result.latency_ms} ms</span>. Next, discover the schema and choose which tables to expose.
              </Notice>
            ) : (
              <Notice tone="danger" title={`Connection failed${result.code ? ` (${result.code})` : ""}`}>
                {result.error}
              </Notice>
            )}
          </div>
        ) : null}
      </div>

      <div className="flex flex-wrap justify-end gap-2 border-t pt-4">
        <Button variant="secondary" onClick={onClose}>
          Done
        </Button>
        <Button onClick={open}>
          {result?.ok ? "Discover schema" : "Open connection"} <ArrowRight className="h-4 w-4" />
        </Button>
      </div>
      {result && !result.ok ? (
        <p className="flex items-center gap-1.5 text-xs text-subtle">
          <XCircle className="h-3.5 w-3.5" /> You can fix the settings from the connection&apos;s Overview → Edit, then test again.
        </p>
      ) : null}
    </div>
  );
}
