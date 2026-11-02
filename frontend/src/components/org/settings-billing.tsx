"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Clock, LifeBuoy, ScrollText, ShieldAlert, XCircle } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { toast } from "sonner";

import { Callout, Forbidden, formatLimitValue, GatedButton, LIMIT_LABELS, permissionHint, PlanMeters, SettingsSection, usd, When } from "@/components/org/kit";
import type { Billing, SupportGrant } from "@/components/org/types";
import { Button } from "@/components/ui/button";
import { type Column, DataTable } from "@/components/ui/data-table";
import { ConfirmDialog } from "@/components/ui/dialog";
import { Field, Input, Select, Textarea } from "@/components/ui/form";
import { Badge, Card, CardBody, EmptyState, Mono, Skeleton, StatusBadge } from "@/components/ui/primitives";
import { del, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatDate } from "@/lib/utils";

/* ------------------------------------------------------------------ billing */
export function BillingTab() {
  const q = useApi<Billing>("/billing");
  const b = q.data;
  if (q.isLoading || !b)
    return (
      <div className="space-y-4">
        <Skeleton className="h-32 rounded-xl" />
        <Skeleton className="h-64 rounded-xl" />
      </div>
    );
  const limitKeys = Object.keys(LIMIT_LABELS);
  return (
    <div>
      <div className="mb-8">
      {!b.billing_enabled ? (
        <Callout tone="info" title="Billing is disabled in this deployment">
          This is a self-hosted installation, so there is no checkout or payment method. The plan and its limits are assigned by a platform administrator; contact yours to change them.
        </Callout>
      ) : (
        <Callout tone="neutral" title="Plan changes are handled by your platform administrator">
          Online checkout is not available in this deployment yet. Your current subscription and usage are shown below.
        </Callout>
      )}
      </div>
      <SettingsSection title="Current plan" description="The plan assigned to this organization and the limits it enforces.">
        <Card className="overflow-hidden">
          <div className="relative px-5 py-6">
            <div aria-hidden className="pointer-events-none absolute inset-0 bg-dots opacity-50 [mask-image:linear-gradient(to_left,black,transparent_60%)]" />
            <div className="relative flex flex-wrap items-end justify-between gap-4">
              <div className="min-w-0">
                <div className="label">Plan</div>
                <div className="mt-2 flex flex-wrap items-center gap-2.5">
                  <span className="text-[28px] font-semibold leading-none tracking-[-0.03em]">{b.plan_name ?? "No plan"}</span>
                  {b.status ? <StatusBadge status={b.status} /> : null}
                </div>
              </div>
              <Mono className="rounded-[5px] border bg-surface-2/70 px-1.5 py-0.5 text-muted">{b.plan ?? "—"}</Mono>
            </div>
          </div>
          <dl className="grid gap-px border-t bg-border sm:grid-cols-2">
            <div className="bg-surface px-5 py-3">
              <dt className="label">Billing provider</dt>
              <dd className="mt-1.5 text-[13px]">{b.provider ?? (b.billing_enabled ? "none configured" : "not applicable")}</dd>
            </div>
            <div className="bg-surface px-5 py-3">
              <dt className="label">Usage limits</dt>
              <dd className="mt-1.5 text-[13px]">{Object.keys(b.limits).length ? `${Object.keys(b.limits).length} usage limits apply` : "No usage limits apply"}</dd>
            </div>
          </dl>
        </Card>
      </SettingsSection>
      <SettingsSection title="Usage" description="Counts right now; tokens are this calendar month (UTC) and API calls today.">
        <Card>
          <CardBody className="py-5">
            <PlanMeters usage={b.usage} limits={b.limits} />
          </CardBody>
        </Card>
      </SettingsSection>
      {b.plans.length ? (
        <SettingsSection title="Available plans" description={b.billing_enabled ? "Plans offered in this deployment. Ask your platform administrator to change yours." : "Plans defined in this deployment. A platform administrator assigns them; there is no checkout."}>
          <div className="grid gap-4 sm:grid-cols-2">
            {b.plans.map((p) => {
              const current = p.key === b.plan;
              const all = p.features.includes("*");
              return (
                <Card key={p.key} className={cn("flex flex-col overflow-hidden", current && "border-accent/50 shadow-[0_0_0_3px_hsl(var(--accent)/0.1)]")}>
                  <div className="border-b px-5 py-4">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[14px] font-semibold">{p.name}</span>
                      {current ? <Badge tone="accent">current</Badge> : <Mono className="text-[11px] text-subtle">{p.key}</Mono>}
                    </div>
                    <div className="num mt-2 text-[24px] font-semibold leading-none tracking-[-0.03em]">
                      {p.price_monthly_usd === null ? "Custom" : p.price_monthly_usd === 0 ? "Free" : usd(p.price_monthly_usd, 0)}
                      {p.price_monthly_usd ? <span className="ml-1 text-[12px] font-normal tracking-normal text-muted">/ month</span> : null}
                    </div>
                  </div>
                  <ul className="flex-1 divide-y px-5 py-1.5 text-[12.5px]">
                    {limitKeys.map((k) => (
                      <li key={k} className="flex justify-between gap-2 py-1.5">
                        <span className="text-muted">{LIMIT_LABELS[k]}</span>
                        <span className={cn("num font-mono text-[12px]", k in p.limits ? "font-medium text-fg" : "text-subtle")}>{k in p.limits ? formatLimitValue(k, p.limits[k]) : "Unlimited"}</span>
                      </li>
                    ))}
                  </ul>
                  <div className="flex flex-wrap gap-1 border-t bg-surface-2/30 px-5 py-3">
                    {all ? (
                      <Badge tone="ok">
                        <Check className="h-3 w-3" /> every feature
                      </Badge>
                    ) : (
                      p.features.map((f) => <Badge key={f}>{f}</Badge>)
                    )}
                  </div>
                </Card>
              );
            })}
          </div>
        </SettingsSection>
      ) : null}
    </div>
  );
}

/* ----------------------------------------------------------- support access */
function grantState(g: SupportGrant) {
  if (g.revoked_at) return "revoked";
  if (g.expires_at && new Date(g.expires_at).getTime() <= Date.now()) return "expired";
  return "active";
}

export function SupportAccessTab() {
  const { can } = useAuth();
  const qc = useQueryClient();
  const allowed = can("org.security");
  const canGrant = can("org.security") && can("org.billing");
  const grants = useApi<SupportGrant[]>(allowed ? "/support-access" : null);
  const [email, setEmail] = useState("");
  const [hours, setHours] = useState("4");
  const [reason, setReason] = useState("");
  const [revoking, setRevoking] = useState<SupportGrant | null>(null);

  const grant = useMutation({
    mutationFn: () => post<{ id: string; expires_at: string }>("/support-access", { platform_admin_email: email.trim(), hours: Number(hours), reason: reason.trim() }),
    onSuccess: (r) => {
      toast.success("Support access granted", { description: `Read-only access until ${formatDate(r.expires_at)}.` });
      setEmail("");
      setReason("");
      qc.invalidateQueries({ queryKey: ["/support-access"] });
    },
  });
  const revoke = useMutation({
    mutationFn: (g: SupportGrant) => del(`/support-access/${g.id}`),
    onSuccess: () => {
      toast.success("Support access revoked");
      setRevoking(null);
      qc.invalidateQueries({ queryKey: ["/support-access"] });
    },
  });

  if (!allowed) return <Forbidden perm="org.security" title="Support access is managed by owners and security administrators" />;

  const validEmail = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim());
  const valid = validEmail && reason.trim().length >= 5;

  const columns: Column<SupportGrant>[] = [
    { key: "admin", header: "Platform administrator", cell: (g) => <Mono className="block max-w-[180px] truncate text-muted">{g.platform_user_id}</Mono> },
    { key: "reason", header: "Reason", className: "min-w-[200px] max-w-[320px]", cell: (g) => <span className="line-clamp-2 text-[12.5px]">{g.reason}</span> },
    { key: "granted", header: "Granted", sortValue: (g) => g.created_at, cell: (g) => <When iso={g.created_at} className="text-muted" /> },
    {
      key: "expires",
      header: "Window ends",
      sortValue: (g) => g.expires_at || "",
      cell: (g) => (
        <span title={formatDate(g.expires_at)} className="inline-flex items-center gap-1.5 whitespace-nowrap text-muted">
          <Clock className="h-3 w-3 text-subtle" aria-hidden />
          {formatDate(g.expires_at)}
        </span>
      ),
    },
    { key: "status", header: "Status", sortValue: (g) => grantState(g), cell: (g) => <StatusBadge status={grantState(g)} /> },
    {
      key: "actions",
      header: <span className="relative block"><span className="sr-only">Actions</span></span>,
      className: "text-right",
      cell: (g) =>
        grantState(g) === "active" ? (
          <Button size="xs" variant="ghost" onClick={() => setRevoking(g)}>
            <XCircle className="h-3.5 w-3.5" /> Revoke
          </Button>
        ) : null,
    },
  ];

  return (
    <div>
      <Callout tone="info" icon={<ShieldAlert className="h-4 w-4" />} title="Platform administrators cannot read your content by default" className="mb-8">
        They manage tenants and system health using metadata and counts only. To let one troubleshoot inside this organization, grant a time-boxed, read-only window here. Every request they make during it is recorded in your{" "}
        <Link href="/audit?action=support." className="font-medium text-accent hover:underline">
          audit log
        </Link>
        , and you can revoke it at any time.
      </Callout>
      <SettingsSection
        title="Grant support access"
        description="Owners only. Opens a read-only window for one platform administrator; it closes on its own when the time runs out."
      >
        <Card>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (valid && canGrant) grant.mutate();
            }}
          >
            <fieldset disabled={!canGrant} className="space-y-4 px-5 py-5">
              <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_180px]">
              <Field label="Platform administrator email" htmlFor="sa-email">
                <Input id="sa-email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="support@platform-operator.com" />
              </Field>
              <Field label="Duration" htmlFor="sa-hours">
                <Select id="sa-hours" value={hours} onChange={(e) => setHours(e.target.value)}>
                  {[1, 2, 4, 8, 24, 48, 72].map((h) => (
                    <option key={h} value={h}>
                      {h < 24 ? `${h} hour${h === 1 ? "" : "s"}` : `${h / 24} day${h === 24 ? "" : "s"}`}
                    </option>
                  ))}
                </Select>
              </Field>
              </div>
              <Field label="Reason" htmlFor="sa-reason" hint="Recorded in the audit log. Reference a ticket if you have one." error={reason.trim().length > 0 && reason.trim().length < 5 ? "At least 5 characters." : null}>
                <Textarea id="sa-reason" value={reason} onChange={(e) => setReason(e.target.value)} rows={3} maxLength={1000} placeholder="e.g. Ticket #4821 — ingestion stuck on Q3 contracts" />
              </Field>
            </fieldset>
            <div className="flex flex-wrap items-center justify-end gap-3 rounded-b-xl border-t bg-surface-2/30 px-5 py-3">
              {!canGrant ? <span className="mr-auto text-[12px] text-subtle">{permissionHint("org.billing")} (owner).</span> : <span className="mr-auto text-[12px] text-subtle">Read-only · audited · revocable</span>}
              <GatedButton type="submit" size="sm" allowed={canGrant} reason="Only organization owners can grant support access." loading={grant.isPending} disabled={!valid}>
                <LifeBuoy className="h-3.5 w-3.5" /> Grant access
              </GatedButton>
            </div>
          </form>
        </Card>
      </SettingsSection>
      <SettingsSection title="Access windows" description="Every grant with its time window and status. Revoking ends access immediately.">
          <DataTable
            rows={grants.data}
            columns={columns}
            rowKey={(g) => g.id}
            loading={grants.isLoading}
            searchable={false}
            toolbar={
              <Link href="/audit?action=support." className="inline-flex items-center gap-1.5 text-xs font-medium text-accent hover:underline">
                <ScrollText className="h-3.5 w-3.5" /> Support activity
              </Link>
            }
            empty={<EmptyState icon={<LifeBuoy className="h-5 w-5" />} title="No support access granted" description="Grants you create appear here with their time window and status." />}
          />
      </SettingsSection>
      <ConfirmDialog
        open={Boolean(revoking)}
        onOpenChange={(v) => !v && setRevoking(null)}
        title="Revoke support access?"
        description="The platform administrator loses access to this organization immediately."
        confirmLabel="Revoke access"
        loading={revoke.isPending}
        onConfirm={() => {
          if (revoking) revoke.mutate(revoking);
        }}
      />
    </div>
  );
}
