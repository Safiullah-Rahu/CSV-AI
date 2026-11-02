"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { ConfirmDialog, Dialog, DialogClose, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Select } from "@/components/ui/form";
import { Checkbox, Switch } from "@/components/ui/primitives";
import { del, patch } from "@/lib/api";

import { SSL_MODES } from "./connection-wizard";
import { ACCESS_MODES, FormSection, Notice, RadioCards, intervalOptions, invalidatePrefix, useKnowledgeBases, writableKbs } from "./shared";
import type { AccessMode, DbConnection } from "./types";

/* ------------------------------------------------------------------ edit */

type EditState = {
  name: string;
  host: string;
  port: string;
  database: string;
  username: string;
  ssl_mode: string;
  access_mode: AccessMode;
  kb_id: string;
  sync_interval: number;
  enabled: boolean;
  max_rows: string;
  timeout_s: string;
  max_joins: string;
};

function initial(c: DbConnection): EditState {
  return {
    name: c.name,
    host: c.host ?? "",
    port: c.port ? String(c.port) : "",
    database: c.database ?? "",
    username: c.username ?? "",
    ssl_mode: c.ssl_mode || "prefer",
    access_mode: c.access_mode,
    kb_id: c.kb_id ?? "",
    sync_interval: c.sync_interval_minutes ?? 0,
    enabled: c.enabled,
    max_rows: String(c.query_limits?.max_rows ?? 500),
    timeout_s: String(c.query_limits?.timeout_s ?? 15),
    max_joins: String(c.query_limits?.max_joins ?? 4),
  };
}

export function EditConnectionDialog({ connection, open, onOpenChange }: { connection: DbConnection; open: boolean; onOpenChange: (v: boolean) => void }) {
  const qc = useQueryClient();
  const kbs = useKnowledgeBases();
  const [s, setS] = useState<EditState>(() => initial(connection));
  const [errors, setErrors] = useState<Record<string, string>>({});
  useEffect(() => {
    if (open) {
      setS(initial(connection));
      setErrors({});
    }
  }, [open, connection]);
  const set = <K extends keyof EditState>(k: K, v: EditState[K]) => setS((p) => ({ ...p, [k]: v }));
  const sqlite = connection.provider === "sqlite";
  const writable = writableKbs(kbs.data);
  const currentKbMissing = connection.kb_id && !writable.some((k) => k.id === connection.kb_id);

  /** Only changed fields are sent: the API resets health status whenever `enabled` is present, and re-checks KB write access whenever `kb_id` is. */
  const changes = () => {
    const base = initial(connection);
    const body: Record<string, unknown> = {};
    if (s.name.trim() !== base.name) body.name = s.name.trim();
    if (!sqlite) {
      if (s.host.trim() !== base.host) body.host = s.host.trim();
      if (s.port !== base.port) body.port = s.port ? Number(s.port) : null;
      if (s.username.trim() !== base.username) body.username = s.username.trim();
      if (s.ssl_mode !== base.ssl_mode) body.ssl_mode = s.ssl_mode;
    }
    if (s.database.trim() !== base.database) body.database = s.database.trim();
    if (s.access_mode !== base.access_mode) body.access_mode = s.access_mode;
    if (s.kb_id !== base.kb_id) body.kb_id = s.kb_id || null;
    if (s.sync_interval !== base.sync_interval) body.sync_interval_minutes = s.sync_interval;
    if (s.enabled !== base.enabled) body.enabled = s.enabled;
    if (s.max_rows !== base.max_rows || s.timeout_s !== base.timeout_s || s.max_joins !== base.max_joins) {
      body.query_limits = { max_rows: Number(s.max_rows), timeout_s: Number(s.timeout_s), max_joins: Number(s.max_joins) };
    }
    return body;
  };

  const save = useMutation({
    mutationFn: (body: Record<string, unknown>) => patch<DbConnection>(`/database-connections/${connection.id}`, body),
    onSuccess: (c) => {
      toast.success(`Saved “${c.name}”`);
      invalidatePrefix(qc, "/database-connections", "/connections");
      onOpenChange(false);
    },
  });

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const errs: Record<string, string> = {};
    if (s.name.trim().length < 2) errs.name = "At least 2 characters.";
    if (!sqlite && !s.host.trim()) errs.host = "Host is required.";
    if (s.port && (Number(s.port) < 1 || Number(s.port) > 65535)) errs.port = "1–65535.";
    const rows = Number(s.max_rows);
    if (!Number.isFinite(rows) || rows < 1 || rows > 10000) errs.max_rows = "1–10,000.";
    const t = Number(s.timeout_s);
    if (!Number.isFinite(t) || t < 1 || t > 300) errs.timeout_s = "1–300 seconds.";
    const j = Number(s.max_joins);
    if (!Number.isInteger(j) || j < 0 || j > 20) errs.max_joins = "0–20.";
    setErrors(errs);
    if (Object.keys(errs).length) return;
    const body = changes();
    if (!Object.keys(body).length) {
      onOpenChange(false);
      return;
    }
    save.mutate(body);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        wide
        title={`Edit ${connection.name}`}
        description="Changes apply to the next test, query and sync. The stored password is unchanged — use Rotate password to replace it."
        footer={
          <>
            <DialogClose asChild>
              <Button variant="secondary">Cancel</Button>
            </DialogClose>
            <Button type="submit" form="edit-connection" loading={save.isPending}>
              Save changes
            </Button>
          </>
        }
      >
        <form id="edit-connection" onSubmit={submit} className="space-y-5" noValidate>
          <FormSection title="Connection" description={sqlite ? "Name and the read-only database file." : "Name, endpoint and role. The password is managed separately."}>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Name" htmlFor="e-name" error={errors.name} className="sm:col-span-2">
              <Input id="e-name" value={s.name} onChange={(e) => set("name", e.target.value)} maxLength={160} />
            </Field>
            {sqlite ? (
              <Field label="Database file path" htmlFor="e-db" className="sm:col-span-2">
                <Input id="e-db" value={s.database} onChange={(e) => set("database", e.target.value)} className="font-mono text-[13px]" />
              </Field>
            ) : (
              <>
                <Field label="Host" htmlFor="e-host" error={errors.host}>
                  <Input id="e-host" value={s.host} onChange={(e) => set("host", e.target.value)} spellCheck={false} autoComplete="off" />
                </Field>
                <Field label="Port" htmlFor="e-port" error={errors.port}>
                  <Input id="e-port" inputMode="numeric" value={s.port} onChange={(e) => set("port", e.target.value.replace(/[^\d]/g, ""))} />
                </Field>
                <Field label="Database" htmlFor="e-database">
                  <Input id="e-database" value={s.database} onChange={(e) => set("database", e.target.value)} spellCheck={false} />
                </Field>
                <Field label="Username" htmlFor="e-user">
                  <Input id="e-user" value={s.username} onChange={(e) => set("username", e.target.value)} spellCheck={false} autoComplete="off" />
                </Field>
                <Field label="SSL mode" htmlFor="e-ssl" className="sm:col-span-2">
                  <Select id="e-ssl" value={s.ssl_mode} onChange={(e) => set("ssl_mode", e.target.value)}>
                    {SSL_MODES.map((m) => (
                      <option key={m.value} value={m.value}>
                        {m.label}
                      </option>
                    ))}
                  </Select>
                </Field>
              </>
            )}
          </div>
          </FormSection>

          <FormSection title="Access mode" description="Live querying, synchronisation, or both.">
            <RadioCards name="Access mode" layout="stack" value={s.access_mode} onChange={(v) => set("access_mode", v)} options={ACCESS_MODES} />
          </FormSection>

          <FormSection title="Synchronisation" description="Target knowledge base and schedule for ingest and hybrid modes.">
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Target knowledge base" htmlFor="e-kb" hint={s.access_mode === "live" ? "Only used in ingest and hybrid modes." : "Synchronised tables are indexed here."}>
              <Select id="e-kb" value={s.kb_id} onChange={(e) => set("kb_id", e.target.value)}>
                <option value="">None</option>
                {currentKbMissing ? <option value={connection.kb_id!}>Current (not writable by you)</option> : null}
                {writable.map((k) => (
                  <option key={k.id} value={k.id}>
                    {k.name}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Sync schedule" htmlFor="e-int">
              <Select id="e-int" value={s.sync_interval} onChange={(e) => set("sync_interval", Number(e.target.value))}>
                {intervalOptions(connection.sync_interval_minutes).map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
          </FormSection>

          <FormSection title="Live query limits" description="Guardrails applied to every live query on this connection.">
            <div className="grid gap-3 sm:grid-cols-3">
              <Field label="Max rows" htmlFor="e-rows" error={errors.max_rows} hint="Results are truncated above this.">
                <Input id="e-rows" inputMode="numeric" value={s.max_rows} onChange={(e) => set("max_rows", e.target.value.replace(/[^\d]/g, ""))} />
              </Field>
              <Field label="Timeout (s)" htmlFor="e-timeout" error={errors.timeout_s} hint="Statement timeout.">
                <Input id="e-timeout" inputMode="numeric" value={s.timeout_s} onChange={(e) => set("timeout_s", e.target.value.replace(/[^\d]/g, ""))} />
              </Field>
              <Field label="Max joins" htmlFor="e-joins" error={errors.max_joins} hint="Queries with more need approval.">
                <Input id="e-joins" inputMode="numeric" value={s.max_joins} onChange={(e) => set("max_joins", e.target.value.replace(/[^\d]/g, ""))} />
              </Field>
            </div>
          </FormSection>

          <div className="flex items-center justify-between gap-4 rounded-xl border bg-surface-2/40 px-3.5 py-3">
            <div>
              <div className="text-[13px] font-medium">Enabled</div>
              <p className="mt-0.5 text-[12px] text-muted">Disabled connections cannot be queried or synchronised and are skipped by the scheduler.</p>
            </div>
            <Switch checked={s.enabled} onCheckedChange={(v) => set("enabled", v)} label="Connection enabled" />
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}

/* --------------------------------------------------------------- rotate */

export function RotatePasswordDialog({ connection, open, onOpenChange }: { connection: DbConnection; open: boolean; onOpenChange: (v: boolean) => void }) {
  const qc = useQueryClient();
  const [pw, setPw] = useState("");
  const [confirm, setConfirm] = useState("");
  useEffect(() => {
    if (!open) {
      setPw("");
      setConfirm("");
    }
  }, [open]);
  const rotate = useMutation({
    mutationFn: () => patch<DbConnection>(`/database-connections/${connection.id}`, { password: pw }),
    onSuccess: () => {
      toast.success("Password rotated", { description: "Test the connection to confirm the new credentials." });
      invalidatePrefix(qc, "/database-connections");
      onOpenChange(false);
    },
  });
  const mismatch = confirm.length > 0 && confirm !== pw;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        title="Rotate password"
        description={`Replace the stored password for ${connection.name}. The current password is never displayed.`}
        footer={
          <>
            <DialogClose asChild>
              <Button variant="secondary">Cancel</Button>
            </DialogClose>
            <Button type="submit" form="rotate-password" disabled={!pw || pw !== confirm} loading={rotate.isPending}>
              Rotate password
            </Button>
          </>
        }
      >
        <form
          id="rotate-password"
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (pw && pw === confirm) rotate.mutate();
          }}
        >
          <Notice tone="info">{connection.has_credentials ? "A password is currently stored (encrypted)." : "No password is stored yet."}</Notice>
          <Field label="New password" htmlFor="r-pw">
            <Input id="r-pw" type="password" value={pw} onChange={(e) => setPw(e.target.value)} autoComplete="new-password" autoFocus />
          </Field>
          <Field label="Confirm new password" htmlFor="r-pw2" error={mismatch ? "Passwords do not match." : null}>
            <Input id="r-pw2" type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} autoComplete="new-password" />
          </Field>
        </form>
      </DialogContent>
    </Dialog>
  );
}

/* ----------------------------------------------------------- disconnect */

export function DisconnectDialog({ connection, open, onOpenChange }: { connection: DbConnection; open: boolean; onOpenChange: (v: boolean) => void }) {
  const qc = useQueryClient();
  const router = useRouter();
  const [deleteDocs, setDeleteDocs] = useState(false);
  useEffect(() => {
    if (!open) setDeleteDocs(false);
  }, [open]);
  const remove = useMutation({
    mutationFn: () => del(`/database-connections/${connection.id}`, { confirm_name: connection.name, delete_synced_documents: deleteDocs }),
    onSuccess: () => {
      toast.success(`Disconnected “${connection.name}”`, { description: deleteDocs ? "Synced documents were deleted." : "Stored credentials were deleted." });
      invalidatePrefix(qc, "/database-connections", "/connections", "/documents", "/knowledge-bases");
      onOpenChange(false);
      router.push("/database-connections");
    },
  });
  return (
    <ConfirmDialog
      open={open}
      onOpenChange={onOpenChange}
      title="Disconnect database"
      description="The connection, its schema selection and its encrypted credentials are deleted. Query logs and audit entries are kept."
      confirmLabel="Disconnect"
      requireText={connection.name}
      loading={remove.isPending}
      onConfirm={() => remove.mutate()}
    >
      <label className="mb-4 flex cursor-pointer items-start gap-2.5 rounded-xl border bg-surface-2/40 p-3.5 text-[13px] transition-colors hover:border-border-strong">
        <span className="pt-0.5">
          <Checkbox checked={deleteDocs} onCheckedChange={setDeleteDocs} label="Also delete synced documents" />
        </span>
        <span>
          <span className="font-medium">Also delete synced documents</span>
          <span className="mt-0.5 block text-[12px] leading-relaxed text-muted">Removes every document this connection synchronised into a knowledge base, with its chunks and embeddings. This cannot be undone.</span>
        </span>
      </label>
    </ConfirmDialog>
  );
}
