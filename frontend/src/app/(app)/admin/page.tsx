"use client";

import { Activity, Building2, ListRestart, ScrollText, ShieldAlert, Users } from "lucide-react";
import { Suspense } from "react";

import { JobsPanel, OrganizationsPanel, PlatformAuditPanel, SystemPanel, UsersPanel } from "@/components/org/admin-panels";
import { Callout, PlatformForbidden, TabBar, useUrlTab } from "@/components/org/kit";
import { Page, PageHeader, Skeleton, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { useAuth } from "@/lib/auth";

const TABS = ["organizations", "users", "system", "jobs", "audit"] as const;

export default function AdminPage() {
  return (
    <Suspense fallback={<Page><Skeleton className="h-96 rounded-xl" /></Page>}>
      <AdminInner />
    </Suspense>
  );
}

function AdminInner() {
  const { user } = useAuth();
  const [tab, setTab] = useUrlTab(TABS, "organizations");

  if (!user?.is_platform_admin) {
    return (
      <Page>
        <PageHeader eyebrow="Platform" title="Administration" />
        <div className="animate-fade-up">
          <PlatformForbidden />
        </div>
      </Page>
    );
  }

  return (
    <Page>
      <PageHeader eyebrow="Platform" title="Administration" description="Tenants, users, plans and system health for this whole deployment. Every change here is written to the platform audit log." />
      <Callout tone="warn" icon={<ShieldAlert className="h-4 w-4" />} title="No access to customer content" className="mb-7">
        This console shows metadata and counts only. Platform administrators cannot read an organization&apos;s documents, conversations or data unless one of its owners grants a time-boxed, read-only support access window — and every request made during that window is audited in the organization&apos;s own log.
      </Callout>
      <Tabs value={tab} onValueChange={setTab}>
        <TabBar>
          <TabsList variant="line">
            <TabsTrigger value="organizations">
              <Building2 className="h-3.5 w-3.5" /> Organizations
            </TabsTrigger>
            <TabsTrigger value="users">
              <Users className="h-3.5 w-3.5" /> Users
            </TabsTrigger>
            <TabsTrigger value="system">
              <Activity className="h-3.5 w-3.5" /> System
            </TabsTrigger>
            <TabsTrigger value="jobs">
              <ListRestart className="h-3.5 w-3.5" /> Jobs
            </TabsTrigger>
            <TabsTrigger value="audit">
              <ScrollText className="h-3.5 w-3.5" /> Platform audit
            </TabsTrigger>
          </TabsList>
        </TabBar>
        <TabsContent value="organizations" className="animate-fade-in focus-visible:outline-none">
          <OrganizationsPanel />
        </TabsContent>
        <TabsContent value="users" className="animate-fade-in focus-visible:outline-none">
          <UsersPanel />
        </TabsContent>
        <TabsContent value="system" className="animate-fade-in focus-visible:outline-none">
          <SystemPanel />
        </TabsContent>
        <TabsContent value="jobs" className="animate-fade-in focus-visible:outline-none">
          <JobsPanel />
        </TabsContent>
        <TabsContent value="audit" className="animate-fade-in focus-visible:outline-none">
          <PlatformAuditPanel />
        </TabsContent>
      </Tabs>
    </Page>
  );
}
