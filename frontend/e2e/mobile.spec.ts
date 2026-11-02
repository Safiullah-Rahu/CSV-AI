import type { Page } from "@playwright/test";

import { expect, test } from "./fixtures";
import { expectAppShell } from "./support/browser";
import { uniqueName } from "./support/state";

// Runs in the 390 px "mobile-chromium" project only (see playwright.config.ts).

test("the navigation drawer opens and navigates", async ({ ownerPage: page }) => {
  // The desktop sidebar is not shown at this width.
  await expect(page.getByRole("navigation", { name: "Main navigation" })).toBeHidden();

  await page.getByRole("button", { name: "Open navigation" }).click();
  const drawer = page.getByRole("dialog", { name: "Navigation" });
  await expect(drawer).toBeVisible();
  await expect(drawer.getByRole("link", { name: "Overview" })).toBeVisible();

  await drawer.getByRole("link", { name: "Knowledge Bases" }).click();
  await page.waitForURL(/\/knowledge-bases$/);
  await expect(drawer).toBeHidden();
  await expect(page.getByRole("heading", { level: 1, name: "Knowledge bases" })).toBeVisible();
});

type PageCheck = { path: string; ready: (p: Page) => Promise<void> };

/**
 * Visit each page and collect any that scroll sideways: the document itself, or the
 * app's content panel (<main>, the scroll container). Tables that scroll inside
 * their own card are fine and do not count.
 */
async function horizontalOverflow(page: Page, pages: PageCheck[]): Promise<string[]> {
  const offenders: string[] = [];
  for (const { path, ready } of pages) {
    await page.goto(path);
    await expectAppShell(page);
    await ready(page);
    // Let entry animations and late layout (charts, fonts) settle.
    await page.waitForTimeout(600);
    const o = await page.evaluate(() => {
      const doc = document.documentElement;
      const main = document.querySelector("main");
      return { doc: [doc.scrollWidth, doc.clientWidth], main: main ? [main.scrollWidth, main.clientWidth] : null };
    });
    if (o.doc[0] > o.doc[1]) offenders.push(`${path}: document ${o.doc[0]} > ${o.doc[1]}`);
    if (o.main && o.main[0] > o.main[1]) offenders.push(`${path}: content panel ${o.main[0]} > ${o.main[1]}`);
  }
  return offenders;
}

test("key pages have no horizontal overflow at 390 px", async ({ ownerPage: page, ownerApi, run }) => {
  const kb = await ownerApi.createKnowledgeBase(uniqueName(run, "mobile-kb"), { description: "Layout check at phone width." });
  const offenders = await horizontalOverflow(page, [
    { path: "/knowledge-bases", ready: (p) => expect(p.getByRole("heading", { level: 3, name: kb.name })).toBeVisible() },
    { path: `/knowledge-bases/${kb.id}`, ready: (p) => expect(p.getByRole("heading", { level: 1, name: kb.name })).toBeVisible() },
    { path: "/documents", ready: (p) => expect(p.getByRole("heading", { level: 1 })).toBeVisible() },
    { path: "/chat", ready: (p) => expect(p.getByRole("textbox", { name: "Message" })).toBeVisible() },
    { path: "/connections", ready: (p) => expect(p.getByRole("heading", { level: 1, name: "Connections" })).toBeVisible() },
    { path: "/team", ready: (p) => expect(p.getByRole("heading", { level: 1 })).toBeVisible() },
    { path: "/settings", ready: (p) => expect(p.getByRole("heading", { level: 1, name: "Settings" })).toBeVisible() },
  ]);
  expect(offenders, "pages that scroll sideways at 390 px").toEqual([]);
});

// Regression guard: the overview chart once propped its grid column open at 390 px.
test("the overview has no horizontal overflow at 390 px", async ({ ownerPage: page }) => {
  const offenders = await horizontalOverflow(page, [{ path: "/overview", ready: (p) => expect(p.getByText("Documents indexed")).toBeVisible() }]);
  expect(offenders, "pages that scroll sideways at 390 px").toEqual([]);
});
