import path from "node:path";

import type { Locator } from "@playwright/test";

import { expect, test } from "./fixtures";
import { DATA_DIR } from "./support/env";
import { uniqueName } from "./support/state";

const FIXTURE = path.join(DATA_DIR, "larkspur-handbook.md");

/** The cell of `row` under the column whose header matches `header`. */
async function cell(table: Locator, row: Locator, header: RegExp): Promise<Locator> {
  const headers = (await table.getByRole("columnheader").allTextContents()).map((h) => h.trim());
  const index = headers.findIndex((h) => header.test(h));
  expect(index, `column ${header} in [${headers.join(" | ")}]`).toBeGreaterThanOrEqual(0);
  return row.getByRole("cell").nth(index);
}

test("create a knowledge base, upload a Markdown file and see it indexed", async ({ ownerPage: page, ownerApi, run }) => {
  const kbName = uniqueName(run, "kb");

  await page.goto("/knowledge-bases");
  await expect(page.getByRole("heading", { level: 1, name: "Knowledge bases" })).toBeVisible();
  await page.getByRole("button", { name: "New knowledge base" }).first().click();

  const dialog = page.getByRole("dialog", { name: "New knowledge base" });
  await dialog.getByLabel("Name", { exact: true }).fill(kbName);
  await dialog.getByLabel("Description", { exact: true }).fill("Upload and indexing check from the end-to-end suite.");
  await dialog.getByRole("button", { name: "Create knowledge base" }).click();

  await page.waitForURL(/\/knowledge-bases\/[0-9a-f-]{36}/);
  await expect(page.getByRole("heading", { level: 1, name: kbName })).toBeVisible();
  const kbId = page.url().match(/knowledge-bases\/([0-9a-f-]{36})/)![1];

  // Upload through the page's file picker.
  await page.getByLabel("Choose files to upload").setInputFiles(FIXTURE);
  await expect(page.getByText("1 file selected")).toBeVisible();
  await page.getByRole("button", { name: /Upload 1 file/ }).click();
  await expect(page.getByText("Upload results")).toBeVisible();
  await expect(page.getByText("Queued for indexing").first()).toBeVisible();

  // The documents table polls while ingestion runs; wait for the row to turn ready.
  const table = page.getByRole("table");
  const row = table.getByRole("row").filter({ hasText: "larkspur-handbook.md" });
  await expect(row).toBeVisible();
  await expect(await cell(table, row, /^Status/)).toContainText("ready", { timeout: 90_000 });

  // The chunk count shown is the one the backend stored, and it is not zero.
  const docs = await ownerApi.get<{ items: { filename: string; chunk_count: number; status: string }[] }>("/documents", { params: { kb_id: kbId } });
  const doc = docs.items.find((d) => d.filename === "larkspur-handbook.md");
  expect(doc?.status).toBe("ready");
  expect(doc!.chunk_count).toBeGreaterThan(0);
  await expect(await cell(table, row, /^Chunks/)).toHaveText(String(doc!.chunk_count));

  // The document tab badge counts it too.
  await expect(page.getByRole("tab", { name: /Documents/ })).toContainText("1");
});
