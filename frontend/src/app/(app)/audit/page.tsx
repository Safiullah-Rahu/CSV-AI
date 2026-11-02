"use client";

import { useMutation } from "@tanstack/react-query";
import { ChevronRight, Download, FilterX, KeyRound, RefreshCw, ScrollText, Server, UserRound } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { Fragment, Suspense, useState } from "react";
import { toast } from "sonner";

import { CopyButton, downloadFile, Forbidden, JsonBlock, KV, Pager, useDebounced, When } from "@/components/org/kit";
import type { AuditEvent, Page as PageOf } from "@/components/org/types";
import { Button } from "@/components/ui/button";
import { Field, Input, Select } from "@/components/ui/form";
import { Card, Dot, EmptyState, Mono, Page, PageHeader, Skeleton, SkeletonRows, StatusBadge } from "@/components/ui/primitives";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";
import { cn, formatNumber } from "@/lib/utils";

const LIMIT = 50;
const ACTION_PREFIXES = ["auth.", "member.", "role.", "grant.", "authz.denied", "api_key.", "webhook.", "conversation.", "org.", "support.", "ai_provider.", "workspace.", "team.", "department.", "audit.export"];
const RESOURCE_TYPES = ["user", "membership", "invitation", "role", "role_assignment", "team", "department", "workspace", "organization", "conversation", "knowledge_base", "document", "db_connection", "integration", "agent", "api_key", "webhook", "ai_provider", "support_access"];

/** `YYYY-MM-DD` (local) → ISO instant at the start / end of that local day. */
function dayBound(d: string, end: boolean) {
  if (!d) return undefined;
  const dt = new Date(`${d}T${end ? "23:59:59.999" : "00:00:00"}`);
  return Number.isNaN(dt.getTime()) ? undefined : dt.toISOString();
}

const pad = (n: number) => String(n).padStart(2, "0");
/** Fixed-width timestamp for scanning down a column: `Nov 02` + `14:03:27`. */
function stamp(iso: string) {
  const d = new Date(iso);
  return { day: d.toLocaleDateString(undefined, { month: "short", day: "2-digit" }), time: `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`, full: d.toLocaleString(undefined, { dateStyle: "full", timeStyle: "long" }) };
}

/** `role.assignment.created` → dim namespace, strong verb. */
function ActionName({ action }: { action: string }) {
  const i = action.indexOf(".");
  if (i < 0) return <Mono className="text-fg">{action}</Mono>;
  return (
    <Mono className="whitespace-nowrap">
      <span className="text-subtle">{action.slice(0, i + 1)}</span>
      <span className="text-fg">{action.slice(i + 1)}</span>
    </Mono>
  );
}

export default function AuditPage() {
  return (
    <Suspense fallback={<Page><Skeleton className="h-96 rounded-xl" /></Page>}>
      <AuditInner />
    </Suspense>
  );
}

function AuditInner() {
  const { can } = useAuth();
  const params = useSearchParams();
  const [action, setAction] = useState(params.get("action") ?? "");
  const [outcome, setOutcome] = useState(params.get("outcome") ?? "");
  const [resourceType, setResourceType] = useState(params.get("resource_type") ?? "");
  const [since, setSince] = useState("");
  const [until, setUntil] = useState("");
  const [q, setQ] = useState(params.get("q") ?? "");
  const [offset, setOffset] = useState(0);
  const [open, setOpen] = useState<Set<number>>(new Set());

  const dAction = useDebounced(action.trim(), 350);
  const dResource = useDebounced(resourceType.trim(), 350);
  const dq = useDebounced(q.trim(), 350);
  const range = { since: dayBound(since, false), until: dayBound(until, true) };
  const rangeInvalid = Boolean(since && until && since > until);

  const filters = { action: dAction || undefined, outcome: outcome || undefined, resource_type: dResource || undefined, q: dq || undefined, ...range };
  const list = useApi<PageOf<AuditEvent>>(can("admin.audit") && !rangeInvalid ? "/audit" : null, { ...filters, limit: LIMIT, offset }, { placeholderData: (prev) => prev });
  const active = Boolean(action || outcome || resourceType || since || until || q);

  const exporter = useMutation({
    mutationFn: () => downloadFile("/audit/export", { action: dAction || undefined, ...range }, "audit.csv"),
    onSuccess: (name) => toast.success(`Exported ${name}`, { description: "Exports include up to 50,000 rows and are themselves audited." }),
  });

  const reset = () => {
    setAction("");
    setOutcome("");
    setResourceType("");
    setSince("");
    setUntil("");
    setQ("");
    setOffset(0);
  };
  const change = <T,>(fn: (v: T) => void) => (v: T) => {
    fn(v);
    setOffset(0);
  };

  if (!can("admin.audit")) {
    return (
      <Page>
        <PageHeader eyebrow="Organization" title="Audit logs" />
        <Forbidden perm="admin.audit" />
      </Page>
    );
  }

  const rows = list.data?.items ?? [];
  const exportScoped = Boolean(outcome || resourceType || q);

  return (
    <Page>
      <PageHeader
        eyebrow="Organization"
        title="Audit logs"
        description="An append-only record of security-relevant activity: sign-ins, permission changes, data access, exports and denied requests. Sensitive values are redacted at write time."
        actions={
          <>
            <Button variant="secondary" size="sm" onClick={() => list.refetch()} loading={list.isFetching && !list.isLoading}>
              {!list.isFetching || list.isLoading ? <RefreshCw className="h-3.5 w-3.5" /> : null} Refresh
            </Button>
            <Button size="sm" onClick={() => exporter.mutate()} loading={exporter.isPending} disabled={rangeInvalid}>
              {!exporter.isPending ? <Download className="h-3.5 w-3.5" /> : null} Export CSV
            </Button>
          </>
        }
      />

      <Card className="mb-5 overflow-hidden">
        <div className="grid grid-cols-2 gap-x-3 gap-y-3.5 p-4 lg:grid-cols-[1.1fr_0.9fr_1fr_0.85fr_0.85fr_1.3fr]">
          <Field label="Action starts with" htmlFor="f-action">
            <Input id="f-action" list="audit-actions" value={action} onChange={(e) => change(setAction)(e.target.value)} placeholder="e.g. role." className="font-mono text-[12.5px]" />
            <datalist id="audit-actions">
              {ACTION_PREFIXES.map((a) => (
                <option key={a} value={a} />
              ))}
            </datalist>
          </Field>
          <Field label="Outcome" htmlFor="f-outcome">
            <Select id="f-outcome" value={outcome} onChange={(e) => change(setOutcome)(e.target.value)}>
              <option value="">Any outcome</option>
              <option value="success">Success</option>
              <option value="denied">Denied</option>
              <option value="failure">Failure</option>
            </Select>
          </Field>
          <Field label="Resource type" htmlFor="f-resource" className="col-span-2 lg:col-span-1">
            <Input id="f-resource" list="audit-resources" value={resourceType} onChange={(e) => change(setResourceType)(e.target.value)} placeholder="Any" className="font-mono text-[12.5px]" />
            <datalist id="audit-resources">
              {RESOURCE_TYPES.map((r) => (
                <option key={r} value={r} />
              ))}
            </datalist>
          </Field>
          <Field label="From" htmlFor="f-since">
            <Input id="f-since" type="date" value={since} max={until || undefined} onChange={(e) => change(setSince)(e.target.value)} className="font-mono text-[12.5px]" />
          </Field>
          <Field label="To" htmlFor="f-until" error={rangeInvalid ? "Must be after the start date." : null}>
            <Input id="f-until" type="date" value={until} min={since || undefined} onChange={(e) => change(setUntil)(e.target.value)} className="font-mono text-[12.5px]" />
          </Field>
          <Field label="Actor email or resource id" htmlFor="f-q" className="col-span-2 lg:col-span-1">
            <Input id="f-q" type="search" value={q} onChange={(e) => change(setQ)(e.target.value)} placeholder="Search…" />
          </Field>
        </div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-t bg-surface-2/30 px-4 py-2.5 text-[12px] text-muted">
          <span className="inline-flex items-center gap-2">
            <Dot tone={list.isFetching ? "info" : "ok"} live={list.isFetching} />
            {list.data ? (
              <span>
                <span className="num font-mono font-medium text-fg">{formatNumber(list.data.total)}</span> matching events
              </span>
            ) : (
              "Loading…"
            )}
          </span>
          <span aria-hidden className="hidden text-border-strong sm:inline">
            /
          </span>
          {exportScoped ? <span className="text-warn">CSV export applies the action and date filters only.</span> : <span className="text-subtle">CSV export uses the action and date filters.</span>}
          {active ? (
            <Button variant="ghost" size="xs" className="ml-auto" onClick={reset}>
              <FilterX className="h-3.5 w-3.5" /> Clear filters
            </Button>
          ) : null}
        </div>
      </Card>

      <Card className="overflow-hidden">
        {list.isLoading ? (
          <SkeletonRows rows={10} />
        ) : rows.length === 0 ? (
          <EmptyState icon={<ScrollText className="h-5 w-5" />} title={active ? "No events match these filters" : "No audit events yet"} description={active ? "Widen the date range or clear a filter." : undefined} action={active ? <Button variant="secondary" onClick={reset}>Clear filters</Button> : undefined} />
        ) : (
          <div className={cn("relative overflow-x-auto transition-opacity", list.isFetching && "opacity-70")}>
            <table className="w-full min-w-[980px] text-[12.5px]">
              <thead>
                <tr className="border-b bg-surface-2/50 text-left font-mono text-[10.5px] uppercase tracking-[0.1em] text-subtle">
                  <th className="h-9 w-10 px-2 font-medium">
                    <span className="sr-only">Expand</span>
                  </th>
                  <th className="h-9 px-3 font-medium">Time</th>
                  <th className="h-9 px-3 font-medium">Actor</th>
                  <th className="h-9 px-3 font-medium">Action</th>
                  <th className="h-9 px-3 font-medium">Outcome</th>
                  <th className="h-9 px-3 font-medium">Resource</th>
                  <th className="h-9 px-3 font-medium">IP</th>
                  <th className="h-9 px-3 pr-4 font-medium">Request</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((e) => {
                  const expanded = open.has(e.id);
                  const toggle = () =>
                    setOpen((s) => {
                      const n = new Set(s);
                      if (n.has(e.id)) n.delete(e.id);
                      else n.add(e.id);
                      return n;
                    });
                  const t = stamp(e.created_at);
                  const flagged = e.outcome === "denied" || e.outcome === "failure";
                  return (
                    <Fragment key={e.id}>
                      <tr
                        onClick={() => {
                          // Selecting an IP or id to copy shouldn't collapse the row.
                          if (!window.getSelection()?.toString()) toggle();
                        }}
                        className={cn(
                          "group/row cursor-pointer border-b transition-colors hover:bg-surface-2/50",
                          expanded && "border-b-0 bg-surface-2/40",
                          e.outcome === "denied" && "bg-danger/[0.025] shadow-[inset_2px_0_0_hsl(var(--danger)/0.7)]",
                          e.outcome === "failure" && "shadow-[inset_2px_0_0_hsl(var(--warn)/0.7)]",
                        )}
                      >
                        <td className="px-2 py-2">
                          <button
                            onClick={(ev) => {
                              ev.stopPropagation();
                              toggle();
                            }}
                            className="grid h-6 w-6 place-items-center rounded-md text-subtle transition-colors hover:bg-surface-2 hover:text-fg"
                            aria-expanded={expanded}
                            aria-label={expanded ? "Hide details" : "Show details"}
                          >
                            <ChevronRight className={cn("h-3.5 w-3.5 transition-transform duration-150", expanded && "rotate-90")} />
                          </button>
                        </td>
                        <td className="whitespace-nowrap px-3 py-2" title={t.full}>
                          <span className="font-mono text-[12px] text-muted">{t.day}</span> <span className="num font-mono text-[12px] text-fg">{t.time}</span>
                          <div className="mt-0.5 text-[11px] text-subtle">
                            <When iso={e.created_at} />
                          </div>
                        </td>
                        <td className="max-w-[230px] px-3 py-2">
                          {e.actor_api_key_id ? (
                            <span className="inline-flex max-w-full items-center gap-2">
                              <span className="grid h-5 w-5 shrink-0 place-items-center rounded-md bg-surface-2 text-muted ring-1 ring-inset ring-border">
                                <KeyRound className="h-3 w-3" />
                              </span>
                              <span className="truncate">{e.actor_email || "API key"}</span>
                            </span>
                          ) : e.actor_email || e.actor_user_id ? (
                            <span className="inline-flex max-w-full items-center gap-2">
                              <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-surface-2 text-muted ring-1 ring-inset ring-border">
                                <UserRound className="h-3 w-3" />
                              </span>
                              <span className="truncate">{e.actor_email || e.actor_user_id}</span>
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-2 text-subtle">
                              <span className="grid h-5 w-5 shrink-0 place-items-center rounded-md border border-dashed">
                                <Server className="h-3 w-3" />
                              </span>
                              System
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-2">
                          <ActionName action={e.action} />
                        </td>
                        <td className="px-3 py-2">
                          <StatusBadge status={e.outcome} />
                        </td>
                        <td className="max-w-[220px] px-3 py-2">
                          {e.resource_type ? (
                            <div className="min-w-0">
                              <div className={cn("text-muted", flagged && "text-fg")}>{e.resource_type.replace(/_/g, " ")}</div>
                              {e.resource_id ? <Mono className="block truncate text-[11px] text-subtle">{e.resource_id}</Mono> : null}
                            </div>
                          ) : (
                            <span className="text-subtle">—</span>
                          )}
                        </td>
                        <td className="whitespace-nowrap px-3 py-2">
                          <Mono className="text-muted">{e.ip || "—"}</Mono>
                        </td>
                        <td className="px-3 py-2 pr-4">{e.request_id ? <Mono className="text-subtle">{e.request_id.slice(0, 12)}</Mono> : <span className="text-subtle">—</span>}</td>
                      </tr>
                      {expanded ? (
                        <tr className={cn("border-b bg-surface-2/40", e.outcome === "denied" && "shadow-[inset_2px_0_0_hsl(var(--danger)/0.7)]", e.outcome === "failure" && "shadow-[inset_2px_0_0_hsl(var(--warn)/0.7)]")}>
                          <td />
                          <td colSpan={7} className="px-3 pb-4 pr-4 pt-1">
                            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
                              <div className="divide-y rounded-xl border bg-surface px-3.5 shadow-xs">
                                <KV label="Event id">
                                  <Mono>{e.id}</Mono>
                                </KV>
                                <KV label="Actor user id">{e.actor_user_id ? <Mono>{e.actor_user_id}</Mono> : "—"}</KV>
                                <KV label="API key id">{e.actor_api_key_id ? <Mono>{e.actor_api_key_id}</Mono> : "—"}</KV>
                                <KV label="Resource id">{e.resource_id ? <Mono>{e.resource_id}</Mono> : "—"}</KV>
                                <KV label="Request id">
                                  {e.request_id ? (
                                    <span className="inline-flex items-center gap-1">
                                      <Mono>{e.request_id}</Mono>
                                      <CopyButton value={e.request_id} label="Copy request id" iconOnly variant="ghost" />
                                    </span>
                                  ) : (
                                    "—"
                                  )}
                                </KV>
                                <KV label="User agent">
                                  <span className="whitespace-normal break-words text-[12px] text-muted">{e.user_agent || "—"}</span>
                                </KV>
                              </div>
                              <div className="min-w-0">
                                <div className="label mb-1.5">Metadata</div>
                                {e.metadata && Object.keys(e.metadata).length ? <JsonBlock value={e.metadata} className="bg-surface" /> : <p className="rounded-xl border border-dashed bg-surface/60 px-3 py-5 text-center text-[12px] text-muted">No metadata recorded.</p>}
                              </div>
                            </div>
                          </td>
                        </tr>
                      ) : null}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        {list.data ? <Pager total={list.data.total} limit={LIMIT} offset={offset} onChange={setOffset} /> : null}
      </Card>
    </Page>
  );
}
