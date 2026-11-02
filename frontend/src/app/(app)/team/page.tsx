"use client";

import { Building, FolderKanban, MailPlus, UserRound, Users } from "lucide-react";
import { Suspense, useState } from "react";

import { useMembers } from "@/components/org/data";
import { GatedButton, permissionHint, TabBar, useUrlTab } from "@/components/org/kit";
import { InvitationsTab, InviteDialog } from "@/components/org/team-invitations";
import { MembersTab } from "@/components/org/team-members";
import { DepartmentsTab, TeamsTab, WorkspacesTab } from "@/components/org/team-structure";
import { Page, PageHeader, Skeleton, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { useAuth } from "@/lib/auth";
import { pluralize } from "@/lib/utils";

const TABS = ["members", "invitations", "teams", "departments", "workspaces"] as const;

export default function TeamPage() {
  return (
    <Suspense fallback={<Page><Skeleton className="h-96 rounded-xl" /></Page>}>
      <TeamInner />
    </Suspense>
  );
}

function TeamInner() {
  const { org, can } = useAuth();
  const [tab, setTab] = useUrlTab(TABS, "members");
  const [inviting, setInviting] = useState(false);
  const members = useMembers();
  const canInvite = can("org.users") && can("admin.permissions");

  return (
    <Page>
      <PageHeader
        eyebrow={org?.organization.name}
        title="Team & users"
        description={`${members.data ? pluralize(members.data.total, "member") : "Members"} · manage people, invitations, teams, departments and workspaces.`}
        actions={
          <GatedButton allowed={canInvite} reason={!can("org.users") ? permissionHint("org.users") : <>Invitations assign a role, which {permissionHint("admin.permissions")}</>} onClick={() => setInviting(true)}>
            <MailPlus className="h-4 w-4" /> Invite people
          </GatedButton>
        }
      />
      <Tabs value={tab} onValueChange={setTab}>
        <TabBar>
          <TabsList variant="line">
            <TabsTrigger value="members">
              <UserRound className="h-3.5 w-3.5" /> Members
            </TabsTrigger>
            {can("org.users") ? (
              <TabsTrigger value="invitations">
                <MailPlus className="h-3.5 w-3.5" /> Invitations
              </TabsTrigger>
            ) : null}
            <TabsTrigger value="teams">
              <Users className="h-3.5 w-3.5" /> Teams
            </TabsTrigger>
            <TabsTrigger value="departments">
              <Building className="h-3.5 w-3.5" /> Departments
            </TabsTrigger>
            {can("workspace.view") ? (
              <TabsTrigger value="workspaces">
                <FolderKanban className="h-3.5 w-3.5" /> Workspaces
              </TabsTrigger>
            ) : null}
          </TabsList>
        </TabBar>
        <TabsContent value="members" className="animate-fade-in focus-visible:outline-none">
          <MembersTab />
        </TabsContent>
        <TabsContent value="invitations" className="animate-fade-in focus-visible:outline-none">
          <InvitationsTab />
        </TabsContent>
        <TabsContent value="teams" className="animate-fade-in focus-visible:outline-none">
          <TeamsTab />
        </TabsContent>
        <TabsContent value="departments" className="animate-fade-in focus-visible:outline-none">
          <DepartmentsTab />
        </TabsContent>
        <TabsContent value="workspaces" className="animate-fade-in focus-visible:outline-none">
          <WorkspacesTab />
        </TabsContent>
      </Tabs>
      <InviteDialog open={inviting} onOpenChange={setInviting} />
    </Page>
  );
}
