"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Building2, Trash2, UserRound, Users } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { memberLabel, useIndex, useMembers, useTeams } from "@/components/org/data";
import { Callout, CancelButton, Segmented } from "@/components/org/kit";
import type { Conversation, ConversationShare } from "@/components/org/types";
import { Button } from "@/components/ui/button";
import { ConfirmDialog, Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Select } from "@/components/ui/form";
import { Skeleton } from "@/components/ui/primitives";
import { del, patch, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";

export function RenameConversationDialog({ conv, onClose }: { conv: Conversation | null; onClose: () => void }) {
  const qc = useQueryClient();
  const [title, setTitle] = useState("");
  useEffect(() => {
    if (conv) setTitle(conv.title);
  }, [conv]);
  const m = useMutation({
    mutationFn: () => patch(`/conversations/${conv!.id}`, { title: title.trim() }),
    onSuccess: () => {
      toast.success("Conversation renamed");
      qc.invalidateQueries({ queryKey: ["/conversations"] });
      onClose();
    },
  });
  return (
    <Dialog open={Boolean(conv)} onOpenChange={(v) => !v && onClose()}>
      <DialogContent
        title="Rename conversation"
        footer={
          <>
            <CancelButton />
            <Button type="submit" form="rename-conv" loading={m.isPending} disabled={!title.trim() || title.trim() === conv?.title}>
              Save
            </Button>
          </>
        }
      >
        <form
          id="rename-conv"
          onSubmit={(e) => {
            e.preventDefault();
            if (title.trim()) m.mutate();
          }}
        >
          <Field label="Title" htmlFor="conv-title">
            <Input id="conv-title" value={title} onChange={(e) => setTitle(e.target.value)} maxLength={300} autoFocus />
          </Field>
        </form>
      </DialogContent>
    </Dialog>
  );
}

type ConvDetail = { id: string; title: string; shares: ConversationShare[] };

export function ShareConversationDialog({ conv, onClose }: { conv: Conversation | null; onClose: () => void }) {
  const qc = useQueryClient();
  const { user } = useAuth();
  const detail = useApi<ConvDetail>(conv ? `/conversations/${conv.id}` : null);
  const members = useMembers(Boolean(conv));
  const teams = useTeams(Boolean(conv));
  const memberById = useIndex(members.data?.items, (m) => m.user_id);
  const teamById = useIndex(teams.data, (t) => t.id);
  const [type, setType] = useState<"user" | "team" | "org">("user");
  const [subject, setSubject] = useState("");
  const [removing, setRemoving] = useState<ConversationShare | null>(null);

  useEffect(() => {
    if (!conv) {
      setType("user");
      setSubject("");
    }
  }, [conv]);

  const shares = detail.data?.shares ?? [];
  const already = useMemo(() => new Set(shares.map((s) => `${s.subject_type}:${s.subject_id ?? ""}`)), [shares]);
  const duplicate = already.has(`${type}:${type === "org" ? "" : subject}`);

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: [`/conversations/${conv?.id}`] });
    qc.invalidateQueries({ queryKey: ["/conversations"] });
  };
  const share = useMutation({
    mutationFn: () => post<{ id: string; note: string }>(`/conversations/${conv!.id}/shares`, { subject_type: type, subject_id: type === "org" ? null : subject }),
    onSuccess: (r) => {
      toast.success("Conversation shared", { description: r.note });
      setSubject("");
      invalidate();
    },
  });
  const unshare = useMutation({
    mutationFn: (s: ConversationShare) => del(`/conversations/${conv!.id}/shares/${s.id}`),
    onSuccess: () => {
      toast.success("Share removed");
      setRemoving(null);
      invalidate();
    },
  });

  const labelFor = (s: ConversationShare) => (s.subject_type === "org" ? "Everyone in the organization" : s.subject_type === "team" ? teamById.get(s.subject_id || "")?.name ?? "Deleted team" : memberLabel(memberById.get(s.subject_id || ""), "Former member"));

  return (
    <>
      <Dialog open={Boolean(conv)} onOpenChange={(v) => !v && onClose()}>
        <DialogContent title="Share conversation" description={conv?.title} footer={<CancelButton label="Close" />}>
          <div className="space-y-5">
            <Callout tone="info">Recipients can read the conversation. Cited sources stay hidden from anyone who cannot access them.</Callout>
            <div className="space-y-3">
              <Segmented
                label="Share with"
                value={type}
                onChange={(v) => {
                  setType(v);
                  setSubject("");
                }}
                options={[
                  { value: "user", label: <><UserRound className="h-3.5 w-3.5" /> Person</> },
                  { value: "team", label: <><Users className="h-3.5 w-3.5" /> Team</> },
                  { value: "org", label: <><Building2 className="h-3.5 w-3.5" /> Organization</> },
                ]}
              />
              <form
                className="flex flex-col gap-2 sm:flex-row"
                onSubmit={(e) => {
                  e.preventDefault();
                  share.mutate();
                }}
              >
                {type === "user" ? (
                  <Select aria-label="Member" value={subject} onChange={(e) => setSubject(e.target.value)} className="flex-1">
                    <option value="">Choose a member…</option>
                    {(members.data?.items ?? [])
                      .filter((m) => m.user_id !== user?.id && m.status === "active")
                      .map((m) => (
                        <option key={m.user_id} value={m.user_id}>
                          {m.full_name ? `${m.full_name} (${m.email})` : m.email}
                        </option>
                      ))}
                  </Select>
                ) : type === "team" ? (
                  <Select aria-label="Team" value={subject} onChange={(e) => setSubject(e.target.value)} className="flex-1">
                    <option value="">Choose a team…</option>
                    {(teams.data ?? []).map((t) => (
                      <option key={t.id} value={t.id}>
                        {t.name} · {t.member_ids.length} members
                      </option>
                    ))}
                  </Select>
                ) : (
                  <p className="flex-1 self-center text-[13px] text-muted">Every member of the organization will be able to open this conversation.</p>
                )}
                <Button type="submit" loading={share.isPending} disabled={(type !== "org" && !subject) || duplicate}>
                  {duplicate ? "Already shared" : "Share"}
                </Button>
              </form>
            </div>
            <div>
              <div className="label mb-2">Shared with</div>
              {detail.isLoading ? (
                <Skeleton className="h-16" />
              ) : shares.length === 0 ? (
                <p className="rounded-xl border border-dashed px-3 py-5 text-center text-[13px] text-muted">Only you can see this conversation.</p>
              ) : (
                <ul className="divide-y rounded-xl border">
                  {shares.map((s) => (
                    <li key={s.id} className="flex items-center gap-3 px-3 py-2 text-[13px]">
                      <span className="grid h-7 w-7 place-items-center rounded-md border bg-surface-2/60 text-muted">
                        {s.subject_type === "org" ? <Building2 className="h-3.5 w-3.5" /> : s.subject_type === "team" ? <Users className="h-3.5 w-3.5" /> : <UserRound className="h-3.5 w-3.5" />}
                      </span>
                      <span className="min-w-0 flex-1 truncate">{labelFor(s)}</span>
                      <span className="label">{s.subject_type}</span>
                      <Button variant="ghost" size="icon" className="h-7 w-7" aria-label={`Stop sharing with ${labelFor(s)}`} onClick={() => setRemoving(s)}>
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>
      <ConfirmDialog
        open={Boolean(removing)}
        onOpenChange={(v) => !v && setRemoving(null)}
        title="Stop sharing?"
        description={removing ? `${labelFor(removing)} will lose access to this conversation.` : undefined}
        confirmLabel="Remove access"
        loading={unshare.isPending}
        onConfirm={() => { if (removing) unshare.mutate(removing); }}
      />
    </>
  );
}
