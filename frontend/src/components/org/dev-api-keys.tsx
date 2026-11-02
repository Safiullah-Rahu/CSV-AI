"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { KeyRound, Plus, XCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { memberLabel, useIndex, useMembers, usePermissionCatalog } from "@/components/org/data";
import { CancelButton, CopyField, Forbidden, PermissionPicker, SecretOnceDialog, When } from "@/components/org/kit";
import type { ApiKey } from "@/components/org/types";
import { Button } from "@/components/ui/button";
import { type Column, DataTable } from "@/components/ui/data-table";
import { ConfirmDialog, Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Select } from "@/components/ui/form";
import { Badge, EmptyState, Mono, Skeleton, StatusBadge, Tip } from "@/components/ui/primitives";
import { del, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatDate } from "@/lib/utils";

const PRESETS: { label: string; scopes: string[] }[] = [
  { label: "Chat API", scopes: ["kb.view", "kb.chat", "conv.create", "conv.view_own"] },
  { label: "Read-only", scopes: ["org.view", "workspace.view", "kb.view", "conv.view_own"] },
  { label: "Ingestion", scopes: ["kb.view", "kb.upload", "kb.reprocess"] },
];

function keyState(k: ApiKey) {
  if (k.revoked_at) return "revoked";
  if (k.expires_at && new Date(k.expires_at).getTime() <= Date.now()) return "expired";
  return "active";
}

export function ApiKeysTab({ baseUrl }: { baseUrl: string }) {
  const { can, user } = useAuth();
  const qc = useQueryClient();
  const allowed = can("admin.api_keys");
  const keys = useApi<ApiKey[]>(allowed ? "/api-keys" : null);
  const members = useMembers(allowed);
  const memberById = useIndex(members.data?.items, (m) => m.user_id);
  const [creating, setCreating] = useState(false);
  const [revoking, setRevoking] = useState<ApiKey | null>(null);
  const [created, setCreated] = useState<(ApiKey & { key: string; notice: string }) | null>(null);

  const revoke = useMutation({
    mutationFn: (k: ApiKey) => del(`/api-keys/${k.id}`),
    onSuccess: (_r, k) => {
      toast.success(`Key “${k.name}” revoked`, { description: "Requests using it are rejected immediately." });
      setRevoking(null);
      qc.invalidateQueries({ queryKey: ["/api-keys"] });
    },
  });

  if (!allowed) return <Forbidden perm="admin.api_keys" title="API keys are managed by developers and administrators" />;

  const canRevoke = (k: ApiKey) => k.created_by === user?.id || can("org.security");

  const columns: Column<ApiKey>[] = [
    {
      key: "name",
      header: "Key",
      sortValue: (k) => k.name.toLowerCase(),
      className: "min-w-[200px]",
      cell: (k) => (
        <div className="flex min-w-0 items-center gap-3">
          <span className={cn("grid h-8 w-8 shrink-0 place-items-center rounded-lg border", keyState(k) === "active" ? "bg-surface-2/60 text-muted" : "bg-surface-2/30 text-subtle opacity-70")}>
            <KeyRound className="h-3.5 w-3.5" />
          </span>
          <div className="min-w-0 leading-tight">
            <div className={cn("truncate font-medium", keyState(k) !== "active" && "text-muted")}>{k.name}</div>
            <Mono className="mt-1 inline-block rounded-[5px] border bg-surface-2/60 px-1.5 py-px text-[11px] text-muted">
              {k.prefix}
              <span className="tracking-[0.12em] text-subtle">.••••••••</span>
            </Mono>
          </div>
        </div>
      ),
    },
    {
      key: "scopes",
      header: "Scopes",
      sortValue: (k) => k.scopes.length,
      cell: (k) => (
        <Tip content={<span className="font-mono text-[11px]">{k.scopes.join(", ")}</span>}>
          <span tabIndex={0} className="inline-flex flex-wrap gap-1">
            {k.scopes.slice(0, 3).map((s) => (
              <Badge key={s} className="font-mono text-[11px]">
                {s}
              </Badge>
            ))}
            {k.scopes.length > 3 ? <Badge className="font-mono text-[11px]">+{k.scopes.length - 3}</Badge> : null}
          </span>
        </Tip>
      ),
    },
    { key: "owner", header: "Created by", cell: (k) => <span className="whitespace-nowrap text-muted">{k.created_by === user?.id ? "You" : memberLabel(memberById.get(k.created_by), "Former member")}</span> },
    { key: "rate", header: "Rate limit", sortValue: (k) => k.rate_limit_per_minute ?? 0, cell: (k) => (k.rate_limit_per_minute ? <span className="num whitespace-nowrap font-mono text-[12px] text-muted">{k.rate_limit_per_minute}/min</span> : <span className="text-subtle">Default</span>) },
    { key: "used", header: "Last used", sortValue: (k) => k.last_used_at || "", cell: (k) => <When iso={k.last_used_at} className="text-muted" /> },
    { key: "expires", header: "Expires", sortValue: (k) => k.expires_at || "9999", cell: (k) => (k.expires_at ? <span title={formatDate(k.expires_at)} className="whitespace-nowrap text-muted">{formatDate(k.expires_at, false)}</span> : <span className="text-warn">Never</span>) },
    { key: "status", header: "Status", sortValue: (k) => keyState(k), cell: (k) => <StatusBadge status={keyState(k)} /> },
    {
      key: "actions",
      header: <span className="relative block"><span className="sr-only">Actions</span></span>,
      className: "text-right",
      cell: (k) =>
        k.active ? (
          canRevoke(k) ? (
            <Button variant="ghost" size="xs" onClick={() => setRevoking(k)}>
              <XCircle className="h-3.5 w-3.5" /> Revoke
            </Button>
          ) : (
            <Tip content="Only the creator or someone with org.security can revoke this key.">
              <span tabIndex={0} className="inline-flex">
                <Button variant="ghost" size="xs" disabled>
                  Revoke
                </Button>
              </span>
            </Tip>
          )
        ) : null,
    },
  ];

  return (
    <>
      <DataTable
        rows={keys.data}
        columns={columns}
        rowKey={(k) => k.id}
        loading={keys.isLoading}
        filterText={(k) => `${k.name} ${k.prefix} ${k.scopes.join(" ")}`}
        searchPlaceholder="Search keys…"
        toolbar={
          <Button size="sm" variant={keys.data?.length ? "primary" : "secondary"} onClick={() => setCreating(true)}>
            <Plus className="h-3.5 w-3.5" /> Create API key
          </Button>
        }
        empty={<EmptyState icon={<KeyRound className="h-5 w-5" />} title="No API keys yet" description="API keys let services call this platform as you, limited to the scopes you choose." action={<Button onClick={() => setCreating(true)}>Create API key</Button>} />}
      />
      <p className="mt-3 max-w-3xl text-[12px] leading-relaxed text-subtle">
        Keys act as their creator, limited to their scopes, and are shown in full only once at creation.
        {!can("admin.audit") ? " You see the keys you created. Administrators with admin.audit see every key in the organization." : null}
      </p>
      <CreateKeyDialog
        open={creating}
        onOpenChange={setCreating}
        onCreated={(k) => {
          setCreating(false);
          setCreated(k);
        }}
      />
      <SecretOnceDialog open={Boolean(created)} onClose={() => setCreated(null)} title={`API key created · ${created?.name ?? ""}`} secret={created?.key ?? ""} secretLabel="API key" description={created?.notice}>
        {created ? <CopyField label="Test request" multiline value={`curl -s "${baseUrl}/knowledge-bases" \\\n  -H "Authorization: Bearer ${created.key}"`} /> : null}
      </SecretOnceDialog>
      <ConfirmDialog
        open={Boolean(revoking)}
        onOpenChange={(v) => !v && setRevoking(null)}
        title="Revoke API key?"
        description={revoking ? `Any integration using “${revoking.name}” (${revoking.prefix}…) will start receiving 401 errors immediately. This cannot be undone.` : undefined}
        confirmLabel="Revoke key"
        requireText={revoking?.name}
        loading={revoke.isPending}
        onConfirm={() => {
          if (revoking) revoke.mutate(revoking);
        }}
      />
    </>
  );
}

function CreateKeyDialog({ open, onOpenChange, onCreated }: { open: boolean; onOpenChange: (v: boolean) => void; onCreated: (k: ApiKey & { key: string; notice: string }) => void }) {
  const { org } = useAuth();
  const qc = useQueryClient();
  const catalog = usePermissionCatalog();
  const held = useMemo(() => new Set(org?.permissions.org ?? []), [org]);
  const [name, setName] = useState("");
  const [expiry, setExpiry] = useState("90");
  const [rate, setRate] = useState("");
  const [scopes, setScopes] = useState<string[]>([]);

  useEffect(() => {
    if (open) {
      setName("");
      setExpiry("90");
      setRate("");
      setScopes(PRESETS[0].scopes.filter((s) => held.has(s)));
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const m = useMutation({
    mutationFn: () =>
      post<ApiKey & { key: string; notice: string }>("/api-keys", {
        name: name.trim(),
        scopes,
        expires_in_days: expiry === "never" ? null : Number(expiry),
        rate_limit_per_minute: rate ? Number(rate) : null,
      }),
    onSuccess: (k) => {
      qc.invalidateQueries({ queryKey: ["/api-keys"] });
      onCreated(k);
    },
  });
  const rateNum = rate ? Number(rate) : null;
  const rateBad = rateNum !== null && (!Number.isInteger(rateNum) || rateNum < 1 || rateNum > 10000);
  const valid = name.trim().length >= 2 && scopes.length > 0 && !rateBad;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        wide
        title="Create API key"
        description="The key acts as you within this organization, limited to the scopes below. It can never exceed your own permissions."
        footer={
          <>
            <span className="mr-auto text-xs text-muted">{scopes.length} scopes</span>
            <CancelButton />
            <Button type="submit" form="key-form" loading={m.isPending} disabled={!valid}>
              Create key
            </Button>
          </>
        }
      >
        <form
          id="key-form"
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (valid) m.mutate();
          }}
        >
          <div className="grid gap-4 sm:grid-cols-3">
            <Field label="Name" htmlFor="key-name" className="sm:col-span-3">
              <Input id="key-name" value={name} onChange={(e) => setName(e.target.value)} maxLength={120} placeholder="e.g. Support portal backend" autoFocus />
            </Field>
            <Field label="Expires after" htmlFor="key-expiry">
              <Select id="key-expiry" value={expiry} onChange={(e) => setExpiry(e.target.value)}>
                <option value="30">30 days</option>
                <option value="90">90 days</option>
                <option value="180">180 days</option>
                <option value="365">1 year</option>
                <option value="730">2 years</option>
                <option value="never">Never (not recommended)</option>
              </Select>
            </Field>
            <Field label="Rate limit (requests / min)" htmlFor="key-rate" error={rateBad ? "Use a whole number from 1 to 10,000." : null} hint="Empty = deployment default" className="sm:col-span-2">
              <Input id="key-rate" inputMode="numeric" value={rate} onChange={(e) => setRate(e.target.value.replace(/[^\d]/g, ""))} placeholder="Default" />
            </Field>
          </div>
          <div>
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <span className="label">Scopes</span>
              <span className="ml-auto text-xs text-subtle">Presets:</span>
              {PRESETS.map((p) => (
                <Button key={p.label} type="button" size="xs" variant="secondary" onClick={() => setScopes(p.scopes.filter((s) => held.has(s)))}>
                  {p.label}
                </Button>
              ))}
              <Button type="button" size="xs" variant="ghost" onClick={() => setScopes([])}>
                Clear
              </Button>
            </div>
            {catalog.data ? <PermissionPicker catalog={catalog.data} value={scopes} onChange={setScopes} held={held} heldReason="Keys cannot carry permissions you do not hold." /> : <Skeleton className="h-64" />}
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
