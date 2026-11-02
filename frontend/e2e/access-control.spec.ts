import { expect, test } from "./fixtures";
import { acceptInvitation } from "./support/api";
import { sidebar, signInWithForm } from "./support/browser";
import { strongPassword } from "./support/env";
import { invitationToken } from "./support/invitations";
import { uniqueEmail, uniqueName } from "./support/state";

const ADMIN_ONLY = ["Roles & Permissions", "API & Developers", "Audit Logs", "Analytics & Usage", "Database Connections", "Integrations", "Administration"];

test("a standard member sees only what they are allowed to", async ({ page, ownerApi, run }) => {
  // Seed as the owner: one knowledge base for everyone, one restricted one nobody granted.
  const open = await ownerApi.createKnowledgeBase(uniqueName(run, "open-kb"), { visibility: "org" });
  const restricted = await ownerApi.createKnowledgeBase(uniqueName(run, "restricted-kb"), { visibility: "restricted" });

  // Invite a member through the API and accept the invitation as a new account.
  const member = { email: uniqueEmail(run, "member"), password: strongPassword(), fullName: "Morgan Member" };
  const invitationId = await ownerApi.invite(member.email, "member");
  await acceptInvitation(await invitationToken(invitationId, member.email), member);

  // The member signs in through the login form.
  await signInWithForm(page, member);
  await expect(page.getByRole("button", { name: "Switch organization" })).toContainText(run.orgName);

  const nav = sidebar(page);
  await expect(nav.getByRole("link", { name: "Knowledge Bases" })).toBeVisible();
  await expect(nav.getByRole("link", { name: "AI Chat" })).toBeVisible();
  for (const label of ADMIN_ONLY) await expect(nav.getByRole("link", { name: label }), `${label} is hidden`).toHaveCount(0);

  // The platform console answers with the 403 screen.
  await page.goto("/admin");
  await expect(page.getByRole("heading", { name: "Platform administrators only" })).toBeVisible();
  await expect(page.getByText("403", { exact: true })).toBeVisible();

  // Knowledge bases: the shared one is listed, the restricted one is not.
  await page.goto("/knowledge-bases");
  await expect(page.getByRole("heading", { level: 3, name: open.name })).toBeVisible();
  await expect(page.getByText(restricted.name)).toHaveCount(0);

  // Even by direct link the restricted knowledge base does not exist for this member.
  await page.goto(`/knowledge-bases/${restricted.id}`);
  await expect(page.getByRole("heading", { name: "Knowledge base not found" })).toBeVisible();
  await expect(page.getByText(restricted.name)).toHaveCount(0);
});
