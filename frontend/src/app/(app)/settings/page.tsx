"use client";

import { Bot, Building2, CreditCard, LifeBuoy, Lock, UserRound } from "lucide-react";
import { Suspense } from "react";

import { TabBar, useUrlTab } from "@/components/org/kit";
import { ProfileTab, SecurityTab } from "@/components/org/settings-account";
import { AIProvidersTab } from "@/components/org/settings-ai";
import { BillingTab, SupportAccessTab } from "@/components/org/settings-billing";
import { OrganizationTab } from "@/components/org/settings-org";
import { Page, PageHeader, Skeleton, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { useAuth } from "@/lib/auth";

const TABS = ["profile", "security", "organization", "ai", "billing", "support"] as const;

export default function SettingsPage() {
  return (
    <Suspense fallback={<Page><Skeleton className="h-96 rounded-xl" /></Page>}>
      <SettingsInner />
    </Suspense>
  );
}

function SettingsInner() {
  const { can, org } = useAuth();
  const [tab, setTab] = useUrlTab(TABS, "profile");
  const isOrgTab = tab === "organization" || tab === "ai" || tab === "billing" || tab === "support";

  return (
    <Page>
      <PageHeader eyebrow={isOrgTab ? org?.organization.name : "Your account"} title="Settings" description={isOrgTab ? "Organization-wide configuration. Changes are audited." : "Your personal profile, sign-in security and sessions."} />
      <Tabs value={tab} onValueChange={setTab}>
        <TabBar>
          <TabsList variant="line" aria-label="Settings sections">
            <TabsTrigger value="profile">
              <UserRound className="h-3.5 w-3.5" /> Profile
            </TabsTrigger>
            <TabsTrigger value="security">
              <Lock className="h-3.5 w-3.5" /> Security
            </TabsTrigger>
            <span aria-hidden className="mx-1 h-4 w-px shrink-0 bg-border" />
            <TabsTrigger value="organization">
              <Building2 className="h-3.5 w-3.5" /> Organization
            </TabsTrigger>
            {can("org.view") ? (
              <TabsTrigger value="ai">
                <Bot className="h-3.5 w-3.5" /> AI providers
              </TabsTrigger>
            ) : null}
            {can("org.view") ? (
              <TabsTrigger value="billing">
                <CreditCard className="h-3.5 w-3.5" /> Plan & billing
              </TabsTrigger>
            ) : null}
            {can("org.security") ? (
              <TabsTrigger value="support">
                <LifeBuoy className="h-3.5 w-3.5" /> Support access
              </TabsTrigger>
            ) : null}
          </TabsList>
        </TabBar>
        <TabsContent value="profile" className="animate-fade-in focus-visible:outline-none">
          <ProfileTab />
        </TabsContent>
        <TabsContent value="security" className="animate-fade-in focus-visible:outline-none">
          <SecurityTab />
        </TabsContent>
        <TabsContent value="organization" className="animate-fade-in focus-visible:outline-none">
          <OrganizationTab />
        </TabsContent>
        <TabsContent value="ai" className="animate-fade-in focus-visible:outline-none">
          <AIProvidersTab />
        </TabsContent>
        <TabsContent value="billing" className="animate-fade-in focus-visible:outline-none">
          <BillingTab />
        </TabsContent>
        <TabsContent value="support" className="animate-fade-in focus-visible:outline-none">
          <SupportAccessTab />
        </TabsContent>
      </Tabs>
    </Page>
  );
}
