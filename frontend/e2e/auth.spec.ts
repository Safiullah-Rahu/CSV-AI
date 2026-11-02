import { expect, test } from "./fixtures";
import { expectAppShell, inputAfterLabel, signInWithForm, signOut } from "./support/browser";
import { strongPassword } from "./support/env";
import { uniqueEmail, uniqueName } from "./support/state";

// The only test that registers through the UI: one registration per run (rate limited).
test("register, create the first organization, sign out and sign in again", async ({ page, run }) => {
  const user = { email: uniqueEmail(run, "reg"), password: strongPassword() };
  const orgName = uniqueName(run, "first-org");

  await page.goto("/register");
  await expect(page.getByRole("heading", { name: "Create your account" })).toBeVisible();
  await inputAfterLabel(page, "Full name").fill("Robin Tester");
  await inputAfterLabel(page, "Work email").fill(user.email);
  await inputAfterLabel(page, "Password").fill(user.password);
  // Organization left empty: the app asks for one after sign-up.
  await page.getByRole("button", { name: "Create account" }).click();

  const firstOrg = page.getByRole("heading", { name: "Create your organization" });
  const formError = page.locator("form").getByRole("alert");
  await expect(firstOrg.or(formError)).toBeVisible({ timeout: 45_000 });
  if (await formError.isVisible()) throw new Error(`registration failed: ${await formError.textContent()}`);
  await expect(page.getByText(`Signed in as ${user.email}`)).toBeVisible();

  await inputAfterLabel(page, "Organization name").fill(orgName);
  await page.getByRole("button", { name: "Create organization" }).click();

  await page.waitForURL(/\/overview/);
  await expectAppShell(page);
  await expect(page.getByRole("heading", { level: 1, name: /Good (morning|afternoon|evening), Robin\./ })).toBeVisible();
  await expect(page.getByRole("button", { name: "Switch organization" })).toContainText(orgName);
  await expect(page.getByText("Documents indexed")).toBeVisible();

  await signOut(page);
  // Signed out for real: the app no longer boots into the workspace.
  await page.goto("/overview");
  await page.waitForURL(/\/login/);

  await signInWithForm(page, user);
  await expect(page.getByRole("heading", { level: 1, name: /Good (morning|afternoon|evening), Robin\./ })).toBeVisible();
  await expect(page.getByRole("button", { name: "Switch organization" })).toContainText(orgName);
});
