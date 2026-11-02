"use client";

import { Crown, Globe2, KeyRound, Lock, ShieldCheck, Users } from "lucide-react";
import type { ReactNode } from "react";

import { Avatar, Badge, Card, CardBody, CardHeader, EmptyState, Skeleton } from "@/components/ui/primitives";
import { useApi } from "@/lib/hooks";

import { GrantRow, GrantsManager, useSubjectDirectory } from "./grants-manager";
import { AccessBadge, kbAbilities, usePerms } from "./shared";
import type { KBAccessSummary, KnowledgeBase } from "./types";

export function KBAccessTab({ kb }: { kb: KnowledgeBase }) {
  const perms = usePerms();
  const ab = kbAbilities(kb, perms);
  const summary = useApi<KBAccessSummary>(`/knowledge-bases/${kb.id}/access`);
  const dir = useSubjectDirectory();
  const owner = kb.owner_id ? dir.members.get(kb.owner_id) : undefined;
  const fullAccessRoles = dir.roles.filter((r) => r.permissions.includes("kb.access_all"));
  const workspace = kb.workspace_id ? dir.workspaces.find((w) => w.id === kb.workspace_id) : undefined;
  const s = summary.data;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
      <Card className="h-fit">
        <CardHeader title="Who can access" description="How access to this knowledge base is decided, in order." action={s ? <span className="flex items-center gap-2 text-[12px] text-muted">You <AccessBadge access={s.your_access} /></span> : null} />
        {!s ? (
          <CardBody className="space-y-2">
            <Skeleton className="h-10" />
            <Skeleton className="h-10" />
            <Skeleton className="h-10" />
          </CardBody>
        ) : (
          <ol className="divide-y">
              <Rule step={1} icon={<Crown className="h-3.5 w-3.5" />} title="Owner — manage">
                {s.owner_id ? (
                  <span className="flex items-center gap-2">
                    <Avatar name={owner?.full_name} email={owner?.email} className="h-6 w-6 text-[10px]" />
                    <span className="truncate">{owner ? owner.full_name || owner.email : dir.label("user", s.owner_id)}</span>
                  </span>
                ) : (
                  <span className="text-muted">No owner recorded.</span>
                )}
              </Rule>
              <Rule step={2} icon={s.visibility === "org" ? <Globe2 className="h-3.5 w-3.5" /> : s.visibility === "workspace" ? <Users className="h-3.5 w-3.5" /> : <Lock className="h-3.5 w-3.5" />} title="Visibility — read">
                {s.visibility === "org"
                  ? "Everyone in the organization with the kb.view permission can read it."
                  : s.visibility === "workspace"
                    ? <>Members of the <span className="font-medium">{workspace?.name ?? "assigned"}</span> workspace can read it.</>
                    : "Restricted: nobody gets implicit read access."}
              </Rule>
              <Rule step={3} icon={<KeyRound className="h-3.5 w-3.5" />} title="Full-access roles — manage">
                {fullAccessRoles.length ? (
                  <span className="flex flex-wrap gap-1">
                    {fullAccessRoles.map((r) => (
                      <Badge key={r.id}>{r.name}</Badge>
                    ))}
                  </span>
                ) : (
                  <span className="text-muted">Roles holding kb.access_all reach every knowledge base unless denied.</span>
                )}
              </Rule>
              <Rule step={4} icon={<ShieldCheck className="h-3.5 w-3.5" />} title="Explicit grants">
                {s.grants.length ? (
                  <span className="num font-medium">
                    {s.grants.filter((g) => g.effect === "allow").length} allow · {s.grants.filter((g) => g.effect === "deny").length} deny
                  </span>
                ) : (
                  "None"
                )}
                <span className="mt-1 block text-[12px] leading-relaxed text-muted">Allows raise the level above; a deny caps it, whatever the source.</span>
              </Rule>
          </ol>
        )}
      </Card>

      {ab.permissions ? (
        <GrantsManager resourceType="knowledge_base" resourceId={kb.id} resourceName={kb.name} invalidate={[`/knowledge-bases/${kb.id}/access`]} />
      ) : (
        <Card className="h-fit">
          <CardHeader title="Explicit grants" description="Managing grants requires the kb.permissions permission and manage access to this knowledge base." />
          {!s ? (
            <div className="p-4">
              <Skeleton className="h-16" />
            </div>
          ) : s.grants.length ? (
            <ul className="divide-y">
              {s.grants.map((g, i) => (
                <GrantRow key={i} g={g} dir={dir} />
              ))}
            </ul>
          ) : (
            <EmptyState icon={<ShieldCheck className="h-5 w-5" />} title="No explicit grants" description="Access comes from ownership, visibility and roles." className="py-10" />
          )}
        </Card>
      )}
    </div>
  );
}

function Rule({ step, icon, title, children }: { step: number; icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <li className="flex gap-3.5 px-5 py-4">
      <span className="relative mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-lg border bg-gradient-to-b from-surface to-surface-2/80 text-muted shadow-xs">
        {icon}
        <span className="absolute -right-1.5 -top-1.5 grid h-4 min-w-4 place-items-center rounded-full border bg-surface px-1 font-mono text-[9.5px] font-semibold text-subtle">{step}</span>
      </span>
      <div className="min-w-0 flex-1 text-[13px]">
        <div className="label mb-1">{title}</div>
        {children}
      </div>
    </li>
  );
}
