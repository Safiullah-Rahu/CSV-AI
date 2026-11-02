"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { History, Plus, RefreshCw, Trash2, Webhook as WebhookIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { Callout, CancelButton, CopyField, Forbidden, RowMenuButton, SecretOnceDialog, When } from "@/components/org/kit";
import type { Webhook, WebhookDelivery } from "@/components/org/types";
import { Button } from "@/components/ui/button";
import { type Column, DataTable } from "@/components/ui/data-table";
import { ConfirmDialog, Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, Input } from "@/components/ui/form";
import { Badge, Checkbox, EmptyState, Menu, MenuContent, MenuItem, MenuSeparator, MenuTrigger, Mono, SkeletonRows, StatusBadge, Switch } from "@/components/ui/primitives";
import { del, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatDate } from "@/lib/utils";

type WebhookList = { available_events: string[]; items: Webhook[] };

const VERIFY_SNIPPET = `// Node.js — verify an incoming delivery
import crypto from "node:crypto";

function verify(req, rawBody, secret) {
  const ts = req.headers["x-rag-timestamp"];
  const expected = "sha256=" + crypto
    .createHmac("sha256", secret)
    .update(\`\${ts}.\${rawBody}\`)
    .digest("hex");
  const given = req.headers["x-rag-signature"] || "";
  // Reject stale timestamps (e.g. > 5 minutes) to prevent replays.
  return given.length === expected.length &&
    crypto.timingSafeEqual(Buffer.from(given), Buffer.from(expected));
}`;

export function WebhooksTab() {
  const { can } = useAuth();
  const qc = useQueryClient();
  const allowed = can("admin.platform_settings");
  const list = useApi<WebhookList>(allowed ? "/webhooks" : null);
  const [creating, setCreating] = useState(false);
  const [created, setCreated] = useState<{ secret: string; notice: string; url: string } | null>(null);
  const [deleting, setDeleting] = useState<Webhook | null>(null);
  const [inspecting, setInspecting] = useState<Webhook | null>(null);

  const remove = useMutation({
    mutationFn: (h: Webhook) => del(`/webhooks/${h.id}`),
    onSuccess: () => {
      toast.success("Webhook deleted");
      setDeleting(null);
      qc.invalidateQueries({ queryKey: ["/webhooks"] });
    },
  });

  if (!allowed) return <Forbidden perm="admin.platform_settings" title="Webhooks are managed by platform settings administrators" />;

  const columns: Column<Webhook>[] = [
    {
      key: "url",
      header: "Endpoint",
      sortValue: (h) => h.url,
      className: "min-w-[260px] max-w-[380px]",
      cell: (h) => (
        <div className="flex min-w-0 items-center gap-3">
          <span className={cn("grid h-8 w-8 shrink-0 place-items-center rounded-lg border", h.enabled ? "bg-surface-2/60 text-muted" : "bg-surface-2/30 text-subtle opacity-70")}>
            <WebhookIcon className="h-3.5 w-3.5" />
          </span>
          <div className="min-w-0 leading-tight">
            <div className="flex min-w-0 items-center gap-1.5">
              <span className="shrink-0 rounded-[4px] bg-accent/10 px-1 py-px font-mono text-[9.5px] font-semibold tracking-[0.06em] text-accent">POST</span>
              <Mono className="block truncate text-fg">{h.url}</Mono>
            </div>
            {h.description ? <div className="mt-1 truncate text-[12px] text-subtle">{h.description}</div> : null}
          </div>
        </div>
      ),
    },
    {
      key: "events",
      header: "Events",
      sortValue: (h) => h.events.length,
      cell: (h) => (
        <div className="flex max-w-[260px] flex-wrap gap-1">
          {h.events.slice(0, 3).map((e) => (
            <Badge key={e} className="font-mono text-[11px]">
              {e}
            </Badge>
          ))}
          {h.events.length > 3 ? <Badge className="font-mono text-[11px]">+{h.events.length - 3}</Badge> : null}
        </div>
      ),
    },
    { key: "status", header: "Status", sortValue: (h) => (h.enabled ? 1 : 0), cell: (h) => <StatusBadge status={h.enabled ? "active" : "disabled"} /> },
    {
      key: "failures",
      header: "Failures",
      sortValue: (h) => h.failure_count,
      cell: (h) => <span className={cn("num font-mono text-[12.5px]", h.failure_count > 0 ? "font-medium text-danger" : "text-subtle")}>{h.failure_count}</span>,
    },
    { key: "last", header: "Last delivery", sortValue: (h) => h.last_delivery_at || "", cell: (h) => <When iso={h.last_delivery_at} className="text-muted" /> },
    {
      key: "actions",
      header: <span className="relative block"><span className="sr-only">Actions</span></span>,
      className: "w-10 text-right",
      cell: (h) => (
        <Menu>
          <MenuTrigger asChild>
            <RowMenuButton label={`Actions for ${h.url}`} />
          </MenuTrigger>
          <MenuContent>
            <MenuItem onSelect={() => setInspecting(h)}>
              <History className="h-3.5 w-3.5" /> Recent deliveries
            </MenuItem>
            <MenuSeparator />
            <MenuItem danger onSelect={() => setDeleting(h)}>
              <Trash2 className="h-3.5 w-3.5" /> Delete webhook
            </MenuItem>
          </MenuContent>
        </Menu>
      ),
    },
  ];

  return (
    <div className="space-y-3">
      <DataTable
        rows={list.data?.items}
        columns={columns}
        rowKey={(h) => h.id}
        loading={list.isLoading}
        filterText={(h) => `${h.url} ${h.description} ${h.events.join(" ")}`}
        searchPlaceholder="Search webhooks…"
        toolbar={
          <Button size="sm" variant={list.data?.items.length ? "primary" : "secondary"} onClick={() => setCreating(true)}>
            <Plus className="h-3.5 w-3.5" /> Add endpoint
          </Button>
        }
        empty={<EmptyState icon={<WebhookIcon className="h-5 w-5" />} title="No webhooks yet" description="Receive signed HTTP callbacks when documents are ingested, syncs finish, connections fail and more." action={<Button onClick={() => setCreating(true)}>Add endpoint</Button>} />}
      />
      <p className="text-[12px] leading-relaxed text-subtle">Each delivery is signed with HMAC-SHA256. Endpoints are disabled automatically after 50 consecutive failed deliveries. Private and loopback addresses are blocked.</p>
      <CreateWebhookDialog
        open={creating}
        onOpenChange={setCreating}
        events={list.data?.available_events ?? []}
        onCreated={(r) => {
          setCreating(false);
          setCreated(r);
        }}
      />
      <SecretOnceDialog open={Boolean(created)} onClose={() => setCreated(null)} title="Webhook endpoint added" secret={created?.secret ?? ""} secretLabel="Signing secret" description={created?.url}>
        {created ? (
          <>
            <Callout tone="info" title="Verifying deliveries">
              {created.notice}
            </Callout>
            <CopyField label="Verification example" multiline value={VERIFY_SNIPPET} />
          </>
        ) : null}
      </SecretOnceDialog>
      <DeliveriesDialog webhook={inspecting} onClose={() => setInspecting(null)} />
      <ConfirmDialog
        open={Boolean(deleting)}
        onOpenChange={(v) => !v && setDeleting(null)}
        title="Delete webhook?"
        description={deleting ? `Deliveries to ${deleting.url} stop immediately and its delivery history is removed.` : undefined}
        confirmLabel="Delete webhook"
        loading={remove.isPending}
        onConfirm={() => {
          if (deleting) remove.mutate(deleting);
        }}
      />
    </div>
  );
}

function CreateWebhookDialog({ open, onOpenChange, events, onCreated }: { open: boolean; onOpenChange: (v: boolean) => void; events: string[]; onCreated: (r: { secret: string; notice: string; url: string }) => void }) {
  const qc = useQueryClient();
  const [url, setUrl] = useState("");
  const [description, setDescription] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [enabled, setEnabled] = useState(true);
  useEffect(() => {
    if (open) {
      setUrl("");
      setDescription("");
      setSelected([]);
      setEnabled(true);
    }
  }, [open]);
  const groups = useMemo(() => {
    const m = new Map<string, string[]>();
    for (const e of events) {
      const g = e.split(".")[0];
      m.set(g, [...(m.get(g) ?? []), e]);
    }
    return Array.from(m.entries());
  }, [events]);
  const m = useMutation({
    mutationFn: () => post<{ id: string; secret: string; notice: string }>("/webhooks", { url: url.trim(), description, events: selected, enabled }),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["/webhooks"] });
      onCreated({ secret: r.secret, notice: r.notice, url: url.trim() });
    },
  });
  let urlError: string | null = null;
  if (url.trim()) {
    try {
      const u = new URL(url.trim());
      if (!["http:", "https:"].includes(u.protocol)) urlError = "Use an http(s) URL.";
    } catch {
      urlError = "Enter a full URL, e.g. https://hooks.example.com/rag";
    }
  }
  const insecure = !urlError && url.trim().startsWith("http://");
  const valid = url.trim() && !urlError && selected.length > 0;
  const toggle = (e: string, on: boolean) => setSelected((s) => (on ? Array.from(new Set([...s, e])) : s.filter((x) => x !== e)));
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        wide
        title="Add webhook endpoint"
        description="We POST a JSON body signed with HMAC-SHA256 for each subscribed event."
        footer={
          <>
            <span className="mr-auto text-xs text-muted">{selected.length} events</span>
            <CancelButton />
            <Button type="submit" form="hook-form" loading={m.isPending} disabled={!valid}>
              Add endpoint
            </Button>
          </>
        }
      >
        <form
          id="hook-form"
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (valid) m.mutate();
          }}
        >
          <Field label="Endpoint URL" htmlFor="hook-url" error={urlError} hint={insecure ? "Plain HTTP sends payloads unencrypted — prefer HTTPS." : undefined}>
            <Input id="hook-url" value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://hooks.example.com/rag" className="font-mono text-[13px]" autoFocus />
          </Field>
          <Field label="Description (optional)" htmlFor="hook-desc">
            <Input id="hook-desc" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="e.g. Notify #data-ops on failed syncs" />
          </Field>
          <label className="flex items-center gap-3 text-[13px]">
            <Switch checked={enabled} onCheckedChange={setEnabled} label="Enabled" />
            Deliver events as soon as the endpoint is created
          </label>
          <div>
            <div className="mb-2 flex items-center gap-2">
              <span className="label">Events</span>
              <Button type="button" size="xs" variant="secondary" className="ml-auto" onClick={() => setSelected(events)}>
                Select all
              </Button>
              <Button type="button" size="xs" variant="ghost" onClick={() => setSelected([])}>
                Clear
              </Button>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {groups.map(([group, items]) => (
                <fieldset key={group} className="rounded-xl border p-2.5">
                  <legend className="label px-1 text-muted">{group}</legend>
                  <ul className="space-y-1">
                    {items.map((e) => (
                      <li key={e}>
                        <label className="flex cursor-pointer items-center gap-2 rounded px-1 py-1 hover:bg-surface-2">
                          <Checkbox checked={selected.includes(e)} onCheckedChange={(v) => toggle(e, v)} label={e} />
                          <span className="font-mono text-[11.5px]">{e}</span>
                        </label>
                      </li>
                    ))}
                  </ul>
                </fieldset>
              ))}
            </div>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function DeliveriesDialog({ webhook, onClose }: { webhook: Webhook | null; onClose: () => void }) {
  const q = useApi<WebhookDelivery[]>(webhook ? `/webhooks/${webhook.id}/deliveries` : null);
  return (
    <Dialog open={Boolean(webhook)} onOpenChange={(v) => !v && onClose()}>
      <DialogContent
        wide
        title="Recent deliveries"
        description={webhook ? <Mono>{webhook.url}</Mono> : undefined}
        footer={
          <>
            <Button variant="secondary" className="mr-auto" onClick={() => q.refetch()} loading={q.isFetching}>
              {!q.isFetching ? <RefreshCw className="h-3.5 w-3.5" /> : null} Refresh
            </Button>
            <CancelButton label="Close" />
          </>
        }
      >
        {q.isLoading ? (
          <SkeletonRows rows={5} />
        ) : !q.data?.length ? (
          <EmptyState icon={<History className="h-5 w-5" />} title="No deliveries yet" description="Deliveries appear here when a subscribed event happens. The last 100 are kept." className="py-8" />
        ) : (
          <div className="relative -mx-6 overflow-x-auto">
            <table className="w-full min-w-[640px] text-[12.5px]">
              <thead>
                <tr className="border-y bg-surface-2/50 text-left font-mono text-[10.5px] uppercase tracking-[0.1em] text-subtle">
                  <th className="h-9 px-6 font-medium">Event</th>
                  <th className="h-9 px-3 font-medium">Status</th>
                  <th className="h-9 px-3 text-right font-medium">Attempts</th>
                  <th className="h-9 px-3 font-medium">HTTP</th>
                  <th className="h-9 px-3 font-medium">Created</th>
                  <th className="h-9 px-6 font-medium">Delivered</th>
                </tr>
              </thead>
              <tbody>
                {q.data.map((d) => (
                  <tr key={d.id} className="border-b align-top last:border-0">
                    <td className="px-6 py-2.5">
                      <Mono className="text-fg">{d.event_type}</Mono>
                      {d.error ? <div className="mt-0.5 max-w-[260px] break-words font-mono text-[11px] text-danger">{d.error}</div> : null}
                    </td>
                    <td className="px-3 py-2.5">
                      <StatusBadge status={d.status} />
                    </td>
                    <td className="num px-3 py-2.5 text-right font-mono">{d.attempts}</td>
                    <td className="px-3 py-2.5">{d.response_status ? <Mono className={cn("font-medium", d.response_status >= 400 ? "text-danger" : "text-ok")}>{d.response_status}</Mono> : <span className="text-subtle">—</span>}</td>
                    <td className="whitespace-nowrap px-3 py-2.5 text-muted" title={formatDate(d.created_at)}>
                      <When iso={d.created_at} />
                    </td>
                    <td className="whitespace-nowrap px-6 py-2.5 text-muted">{d.delivered_at ? formatDate(d.delivered_at) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
