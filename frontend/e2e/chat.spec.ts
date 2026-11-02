import path from "node:path";

import { expect, test } from "./fixtures";
import { DATA_DIR } from "./support/env";
import { uniqueName } from "./support/state";

/*
 * One question per run: when the backend has a language model (the dev backend falls
 * back to the platform's Claude key), every question is a real, paid model call.
 * The assertions hold for both extractive answers (no model) and generated ones:
 * the fact from the fixture, a citation chip, and the source panel.
 */
const QUESTION = "How often is the Larkspur Observatory primary mirror recoated?";
const FACT = /47[\s-]*days?/i;

test("ask about an indexed document and inspect the cited source", async ({ ownerPage: page, ownerApi, run }) => {
  test.setTimeout(180_000);

  // Seed: a knowledge base with the fixture, indexed, through the API.
  const kb = await ownerApi.createKnowledgeBase(uniqueName(run, "chat-kb"));
  const docId = await ownerApi.uploadFile(kb.id, path.join(DATA_DIR, "larkspur-handbook.md"));
  await ownerApi.waitForDocument(docId);

  await page.goto("/chat");
  await expect(page.getByRole("heading", { name: "What would you like to know?" })).toBeVisible();

  // Scope the question to the seeded knowledge base.
  await page.getByRole("button", { name: /^Search scope:/ }).click();
  await page.getByRole("dialog").getByRole("button", { name: new RegExp(kb.name) }).click();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", { name: `Search scope: ${kb.name}` })).toBeVisible();

  const composer = page.getByRole("textbox", { name: "Message" });
  await composer.fill(QUESTION);
  await page.getByRole("button", { name: "Send" }).click();

  // The answer streams in; wait for it to settle (the feedback controls appear when done).
  const answer = page.getByRole("article").last();
  await expect(answer).toContainText(FACT, { timeout: 120_000 });
  await expect(answer.getByRole("button", { name: "Copy answer" })).toBeVisible({ timeout: 60_000 });

  const chip = answer.getByRole("button", { name: /^Source S\d+:/ }).first();
  await expect(chip).toBeVisible();
  await chip.click();

  const panel = page.getByRole("complementary", { name: "Sources" });
  await expect(panel).toBeVisible();
  await expect(panel).toContainText(/Larkspur/i);
  await expect(panel.getByText("Cited in answer")).toBeVisible();

  await panel.getByRole("button", { name: "Close sources" }).click();
  await expect(panel).toBeHidden();
});
