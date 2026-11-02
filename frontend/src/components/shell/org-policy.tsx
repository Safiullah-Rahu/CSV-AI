"use client";

import { KeyRound, ShieldAlert } from "lucide-react";
import { useRouter } from "next/navigation";

import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/primitives";
import { type OrgBlock, useAuth } from "@/lib/auth";

function useBlockedOrgName(): string {
  const { orgs, orgId } = useAuth();
  return orgs.find((o) => o.id === orgId)?.name ?? "This organization";
}

function PolicyAction({ block, size = "md" }: { block: OrgBlock; size?: "sm" | "md" }) {
  const router = useRouter();
  const { signOut } = useAuth();
  if (block.reason === "mfa_required") {
    return (
      <Button size={size} onClick={() => router.push("/settings?tab=security")}>
        <KeyRound className="h-4 w-4" /> Set up two-factor authentication
      </Button>
    );
  }
  // Signing out returns to /login with this page as `next` (useRequireAuth).
  return (
    <Button size={size} onClick={() => void signOut()}>
      Sign in again
    </Button>
  );
}

/** Replaces the page while the organization's security policy blocks this member. */
export function OrgPolicyGate({ block }: { block: OrgBlock }) {
  const name = useBlockedOrgName();
  const mfa = block.reason === "mfa_required";
  return (
    <EmptyState
      className="min-h-full"
      icon={<ShieldAlert className="h-5 w-5" />}
      title={mfa ? "Two-factor authentication required" : "Please sign in again"}
      description={
        <>
          {mfa
            ? `${name} requires every member to protect their account with an authenticator app. Turn it on in your account settings and you'll be taken straight back in.`
            : `${block.message} Your other organizations are not affected.`}
          <span className="mt-2 block text-xs text-subtle">Belong to another organization? Switch to it from the menu at the top.</span>
        </>
      }
      action={<PolicyAction block={block} />}
    />
  );
}

/** Slim notice shown on Settings, where the member can resolve the block. */
export function OrgPolicyBanner({ block }: { block: OrgBlock }) {
  const name = useBlockedOrgName();
  return (
    <div role="status" className="flex flex-wrap items-center gap-3 border-b bg-warn/10 px-6 py-2.5 text-[13px]">
      <ShieldAlert className="h-4 w-4 shrink-0 text-warn" />
      <span className="min-w-0 flex-1">
        {block.reason === "mfa_required" ? `${name} requires two-factor authentication. Turn it on under Security to continue.` : block.message}
      </span>
      {block.reason === "reauth_required" ? <PolicyAction block={block} size="sm" /> : null}
    </div>
  );
}
