import { expect, test } from "./fixtures";
import { expectAppShell, sidebar } from "./support/browser";

test("the theme choice survives a reload", async ({ ownerPage: page }) => {
  const html = page.locator("html");
  await expect(html).not.toHaveClass(/\bdark\b/);

  await page.getByRole("button", { name: "Switch to dark theme" }).click();
  await expect(html).toHaveClass(/\bdark\b/);
  expect(await page.evaluate(() => localStorage.getItem("rag.theme"))).toBe("dark");

  await page.reload();
  await expectAppShell(page);
  await expect(html).toHaveClass(/\bdark\b/);
  await expect(page.getByRole("button", { name: "Switch to light theme" })).toBeVisible();

  await page.getByRole("button", { name: "Switch to light theme" }).click();
  await expect(html).not.toHaveClass(/\bdark\b/);
});

test("the command palette opens with Ctrl+K and navigates", async ({ ownerPage: page }) => {
  // The owner sees the administrative pages a member does not.
  await expect(sidebar(page).getByRole("link", { name: "Roles & Permissions" })).toBeVisible();
  await expect(sidebar(page).getByRole("link", { name: "Audit Logs" })).toBeVisible();

  await page.keyboard.press("Control+K");
  const palette = page.getByRole("dialog", { name: "Search" });
  await expect(palette).toBeVisible();
  await palette.getByRole("textbox", { name: "Search" }).fill("analytics");
  await palette.getByRole("button", { name: /^Analytics & Usage/ }).click();

  await page.waitForURL(/\/analytics$/);
  await expect(palette).toBeHidden();
  await expect(page.getByRole("navigation", { name: "Breadcrumb" })).toContainText("Analytics & Usage");
});
