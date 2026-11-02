import { expect, test } from "./fixtures";

test("the connections map renders nodes and switches to the list view", async ({ ownerPage: page }) => {
  await page.goto("/connections");
  await expect(page.getByRole("heading", { level: 1, name: "Connections" })).toBeVisible();

  // React Flow marks every rendered node with aria-roledescription="node".
  const nodes = page.locator('[aria-roledescription="node"]');
  await expect(nodes.first()).toBeVisible({ timeout: 30_000 });
  expect(await nodes.count()).toBeGreaterThan(0);

  const view = page.getByRole("group", { name: "View" });
  await expect(view.getByRole("button", { name: "Live map" })).toHaveAttribute("aria-pressed", "true");
  await view.getByRole("button", { name: "List" }).click();
  await expect(view.getByRole("button", { name: "List" })).toHaveAttribute("aria-pressed", "true");

  // This run's organization has no databases or integrations yet, so either a table or its empty state.
  await expect(page.getByRole("columnheader", { name: "Resource" }).or(page.getByRole("heading", { name: "No sources connected yet" }))).toBeVisible();
  await expect(nodes).toHaveCount(0);
});
