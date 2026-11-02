"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { usePermissionCatalog } from "@/components/org/data";
import { Callout, CancelButton, PermissionPicker } from "@/components/org/kit";
import type { Role } from "@/components/org/types";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Textarea } from "@/components/ui/form";
import { Badge, Mono, Skeleton } from "@/components/ui/primitives";
import { ApiError, errorMessage, patch, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";

export type RoleEditorState = { mode: "create" } | { mode: "edit"; role: Role } | { mode: "duplicate"; role: Role } | null;

function toKey(name: string) {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64);
}

export function RoleEditorDialog({ state, onClose }: { state: RoleEditorState; onClose: () => void }) {
  const { org } = useAuth();
  const qc = useQueryClient();
  const catalog = usePermissionCatalog();
  const held = useMemo(() => new Set(org?.permissions.org ?? []), [org]);
  const [name, setName] = useState("");
  const [key, setKey] = useState("");
  const [keyTouched, setKeyTouched] = useState(false);
  const [description, setDescription] = useState("");
  const [perms, setPerms] = useState<string[]>([]);
  const [error, setError] = useState<{ message: string; missing: string[] } | null>(null);

  useEffect(() => {
    setError(null);
    setKeyTouched(false);
    if (!state) return;
    if (state.mode === "create") {
      setName("");
      setKey("");
      setDescription("");
      setPerms(["org.view"].filter((p) => held.has(p)));
    } else if (state.mode === "edit") {
      setName(state.role.name);
      setKey(state.role.key);
      setDescription(state.role.description);
      setPerms(state.role.permissions);
    } else {
      setName(`${state.role.name} (custom)`);
      setKey("");
      setDescription(state.role.description);
      // Only carry over what the creator can grant; the rest would be rejected.
      setPerms(state.role.permissions.filter((p) => held.has(p)));
    }
  }, [state]); // eslint-disable-line react-hooks/exhaustive-deps

  const editing = state?.mode === "edit";
  const m = useMutation({
    meta: { silent: true },
    mutationFn: () => {
      if (state?.mode === "edit") return patch<Role>(`/roles/${state.role.id}`, { name: name.trim(), description, permissions: perms });
      return post<Role>("/roles", { key: (keyTouched ? key : toKey(name)) || undefined, name: name.trim(), description, permissions: perms });
    },
    onSuccess: (r) => {
      toast.success(editing ? `Role “${r.name}” updated` : `Role “${r.name}” created`);
      qc.invalidateQueries({ queryKey: ["/roles"] });
      qc.invalidateQueries({ queryKey: ["/members"] });
      onClose();
    },
    onError: (err) => {
      const missing = err instanceof ApiError && err.details && typeof err.details === "object" && Array.isArray((err.details as { missing?: unknown }).missing) ? ((err.details as { missing: string[] }).missing) : [];
      setError({ message: errorMessage(err), missing });
      toast.error(errorMessage(err), err instanceof ApiError && err.requestId ? { description: `Request ID ${err.requestId}` } : undefined);
    },
  });

  const dropped = state?.mode === "duplicate" ? state.role.permissions.filter((p) => !held.has(p)) : [];
  const valid = name.trim().length >= 2 && perms.length > 0;

  return (
    <Dialog open={Boolean(state)} onOpenChange={(v) => !v && onClose()}>
      <DialogContent
        wide
        title={state?.mode === "edit" ? `Edit role · ${state.role.name}` : state?.mode === "duplicate" ? `Duplicate “${state.role.name}” as a custom role` : "Create custom role"}
        description="Roles are bundles of capabilities. Which knowledge bases, documents and connections they reach is decided by grants."
        footer={
          <>
            <span className="mr-auto text-xs text-muted">{perms.length} permissions selected</span>
            <CancelButton />
            <Button type="submit" form="role-form" loading={m.isPending} disabled={!valid}>
              {editing ? "Save role" : "Create role"}
            </Button>
          </>
        }
      >
        <form
          id="role-form"
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            setError(null);
            if (valid) m.mutate();
          }}
        >
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Name" htmlFor="role-name">
              <Input id="role-name" value={name} onChange={(e) => setName(e.target.value)} maxLength={120} placeholder="e.g. Support Analyst" autoFocus />
            </Field>
            <Field label="Key" htmlFor="role-key" hint={editing ? "Keys cannot be changed." : "Stable identifier used by the API."}>
              <Input
                id="role-key"
                value={editing ? key : keyTouched ? key : toKey(name)}
                onChange={(e) => {
                  setKeyTouched(true);
                  setKey(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, "_"));
                }}
                disabled={editing}
                className="font-mono text-[13px]"
                maxLength={64}
              />
            </Field>
          </div>
          <Field label="Description" htmlFor="role-desc">
            <Textarea id="role-desc" value={description} onChange={(e) => setDescription(e.target.value)} rows={2} maxLength={2000} placeholder="Who should get this role and why" />
          </Field>
          {dropped.length ? (
            <Callout tone="warn" title={`${dropped.length} permissions were not copied`}>
              You don't hold {dropped.map((p) => p).join(", ")}, so a custom role you create cannot include them.
            </Callout>
          ) : null}
          {error ? (
            <Callout tone="danger" title={error.message}>
              {error.missing.length ? (
                <span>
                  Not granted to you:{" "}
                  {error.missing.map((p) => (
                    <Badge key={p} tone="danger" className="mr-1 font-mono">
                      {p}
                    </Badge>
                  ))}
                </span>
              ) : null}
            </Callout>
          ) : null}
          <div>
            <div className="label mb-2">Permissions</div>
            {catalog.data ? <PermissionPicker catalog={catalog.data} value={perms} onChange={setPerms} held={held} /> : <Skeleton className="h-64" />}
          </div>
          {state?.mode === "edit" ? (
            <p className="text-xs text-subtle">
              Rank <Mono>{state.role.rank}</Mono> is fixed. Changes apply immediately to everyone holding this role.
            </p>
          ) : (
            <p className="text-xs text-subtle">
              New custom roles get rank <Mono>{Math.min(org?.permissions.max_rank ?? 0, 85)}</Mono> (your rank, capped at 85), so only people at or above it can assign them.
            </p>
          )}
        </form>
      </DialogContent>
    </Dialog>
  );
}

export function RoleViewDialog({ role, onClose }: { role: Role | null; onClose: () => void }) {
  const catalog = usePermissionCatalog();
  return (
    <Dialog open={Boolean(role)} onOpenChange={(v) => !v && onClose()}>
      <DialogContent
        wide
        title={
          <span className="flex items-center gap-2">
            {role?.name}
            {role ? <Badge tone={role.is_system ? "neutral" : "accent"}>{role.is_system ? "system" : "custom"}</Badge> : null}
          </span>
        }
        description={role?.description}
        footer={<CancelButton label="Close" />}
      >
        {role ? (
          <div className="space-y-3">
            <dl className="grid grid-cols-3 gap-px overflow-hidden rounded-xl border bg-border text-[13px]">
              <div className="min-w-0 bg-surface px-3.5 py-2.5">
                <dt className="label">Key</dt>
                <dd className="mt-1 truncate">
                  <Mono className="text-fg">{role.key}</Mono>
                </dd>
              </div>
              <div className="bg-surface px-3.5 py-2.5">
                <dt className="label">Rank</dt>
                <dd className="num mt-1 font-mono font-medium">{role.rank}</dd>
              </div>
              <div className="bg-surface px-3.5 py-2.5">
                <dt className="label">Assignments</dt>
                <dd className="num mt-1 font-mono font-medium">{role.assignment_count}</dd>
              </div>
            </dl>
            {role.is_system ? <Callout tone="neutral">System roles are read-only. Duplicate one to create a custom variant.</Callout> : null}
            {catalog.data ? <PermissionPicker catalog={catalog.data} value={role.permissions} readOnly /> : <Skeleton className="h-64" />}
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
