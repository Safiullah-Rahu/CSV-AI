"use client";

import * as DialogPrimitive from "@radix-ui/react-dialog";
import { usePathname } from "next/navigation";
import { useState } from "react";

import { useOrgBranding } from "@/components/shell/branding";
import { CreateFirstOrg } from "@/components/shell/create-first-org";
import { OrgPolicyBanner, OrgPolicyGate } from "@/components/shell/org-policy";
import { LogoMark, Sidebar } from "@/components/shell/sidebar";
import { Topbar } from "@/components/shell/topbar";
import { useRequireAuth } from "@/lib/auth";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const auth = useRequireAuth();
  const pathname = usePathname();
  const [mobileNav, setMobileNav] = useState(false);
  useOrgBranding();

  if (auth.status !== "authenticated") {
    return (
      <div className="grid h-screen place-items-center bg-frame">
        <div className="flex flex-col items-center gap-4" role="status">
          <LogoMark className="h-9 w-9 animate-pulse" />
          <span className="label">Loading workspace</span>
        </div>
      </div>
    );
  }
  if (!auth.orgId) {
    return <CreateFirstOrg />;
  }
  return (
    // Frame + raised panel: the sidebar sits on the frame, content on its own sheet.
    <div className="flex h-screen overflow-hidden bg-frame">
      <aside className="hidden w-[252px] shrink-0 lg:block">
        <Sidebar />
      </aside>
      <DialogPrimitive.Root open={mobileNav} onOpenChange={setMobileNav}>
        <DialogPrimitive.Portal>
          <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/40 backdrop-blur-[2px] animate-fade-in lg:hidden" />
          <DialogPrimitive.Content className="fixed inset-y-0 left-0 z-50 w-[284px] border-r bg-frame shadow-float animate-fade-in lg:hidden">
            <DialogPrimitive.Title className="sr-only">Navigation</DialogPrimitive.Title>
            <DialogPrimitive.Description className="sr-only">Main navigation</DialogPrimitive.Description>
            <Sidebar onNavigate={() => setMobileNav(false)} />
          </DialogPrimitive.Content>
        </DialogPrimitive.Portal>
      </DialogPrimitive.Root>
      <div className="flex min-w-0 flex-1 flex-col lg:py-2 lg:pr-2">
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden bg-bg lg:rounded-[14px] lg:border lg:shadow-panel">
          <Topbar onMenu={() => setMobileNav(true)} />
          <main className="relative min-h-0 flex-1 overflow-y-auto">
            {/* Account settings stay reachable so a blocked member can satisfy the policy. */}
            {auth.orgBlock && pathname !== "/settings" ? (
              <OrgPolicyGate block={auth.orgBlock} />
            ) : (
              <>
                {auth.orgBlock ? <OrgPolicyBanner block={auth.orgBlock} /> : null}
                {children}
              </>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}
