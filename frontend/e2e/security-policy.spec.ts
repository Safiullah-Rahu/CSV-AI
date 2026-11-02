import { expect, test } from "./fixtures";
import { acceptInvitation } from "./support/api";
import { inputAfterLabel, signInWithForm } from "./support/browser";
import { BASE_URL, strongPassword } from "./support/env";
import { invitationToken } from "./support/invitations";
import { uniqueEmail, uniqueName } from "./support/state";
import { freshTotp } from "./support/totp";

/*
 * Runs in its own organization with its own owner, so requiring two-factor never
 * blocks the shared run organization or the cached owner account other tests use.
 */
test("an owner requires two-factor authentication and members are gated", async ({ page, browser, ownerApi, run }) => {
  test.setTimeout(180_000);

  // Seed: a separate organization with a new owner and a pending member invitation.
  const org = await ownerApi.createOrg(uniqueName(run, "sec"));
  const secOwner = { email: uniqueEmail(run, "sec-owner"), password: strongPassword(), fullName: "Sam Owner" };
  const ownerInvite = await ownerApi.invite(secOwner.email, "owner", org.id);
  await acceptInvitation(await invitationToken(ownerInvite, secOwner.email), secOwner);
  const member = { email: uniqueEmail(run, "sec-member"), password: strongPassword() };
  const memberInvite = await ownerApi.invite(member.email, "member", org.id);
  const memberToken = await invitationToken(memberInvite, member.email);

  // 1. The new owner turns on two-factor for their own account.
  await signInWithForm(page, secOwner);
  await expect(page.getByRole("button", { name: "Switch organization" })).toContainText(org.name);
  await page.goto("/settings?tab=security");
  await page.getByRole("button", { name: "Set up two-factor" }).click();
  const secret = await page.getByLabel("Setup key", { exact: true }).inputValue();
  expect(secret).toMatch(/^[A-Z2-7]{16,}$/);
  await page.getByLabel("6-digit code from the app", { exact: true }).fill(await freshTotp(secret));
  await page.getByRole("button", { name: "Verify & enable" }).click();
  await expect(page.getByText("Protected with an authenticator app")).toBeVisible();

  // 2. ...then requires it for everyone in the organization.
  await page.goto("/settings?tab=organization");
  const toggle = page.getByRole("switch", { name: "Require two-factor authentication" });
  await expect(toggle).toHaveAttribute("aria-checked", "false");
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-checked", "true");
  await page.locator("form").filter({ has: toggle }).getByRole("button", { name: "Save" }).click();
  await expect(page.getByText("Security policy saved")).toBeVisible();

  // 3. A member without two-factor joins (through the invitation page) and is gated.
  // A separate browser context: its own cookies, so its own session.
  const memberContext = await browser.newContext({ baseURL: BASE_URL, viewport: { width: 1440, height: 900 } });
  try {
    const memberPage = await memberContext.newPage();
    await memberPage.goto(`/invite?token=${encodeURIComponent(memberToken)}`);
    await expect(memberPage.getByRole("heading", { name: `Join ${org.name}` })).toBeVisible();
    await inputAfterLabel(memberPage, "Full name").fill("Casey Member");
    await inputAfterLabel(memberPage, "Choose a password").fill(member.password);
    await memberPage.getByRole("button", { name: "Accept invitation" }).click();
    await memberPage.waitForURL(/\/overview/);

    await expect(memberPage.getByRole("heading", { name: "Two-factor authentication required" })).toBeVisible();
    await expect(memberPage.getByRole("main")).toContainText(`${org.name} requires every member`);

    // Account settings stay reachable so the member can comply.
    await memberPage.getByRole("button", { name: "Set up two-factor authentication" }).click();
    await memberPage.waitForURL(/\/settings\?tab=security/);
    await expect(memberPage.getByRole("status").filter({ hasText: "requires two-factor authentication" })).toBeVisible();
    await expect(memberPage.getByRole("button", { name: "Set up two-factor" })).toBeVisible();
  } finally {
    await memberContext.close();
  }
});
