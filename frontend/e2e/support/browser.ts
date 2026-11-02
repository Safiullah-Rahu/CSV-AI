import { expect, type Locator, type Page } from "@playwright/test";

import { withLoginRetry, type Credentials } from "./api";
import { API_PREFIX, sleep } from "./env";

/**
 * The input that follows a visible field label.
 *
 * Some forms (register, accept invitation, create organization) render a <label>
 * that is not associated with its input, so getByLabel cannot find them. This keeps
 * the selector tied to the label text the user sees.
 */
export function inputAfterLabel(page: Page, label: string): Locator {
  const literal = label.includes('"') ? `'${label}'` : `"${label}"`;
  return page.locator(`xpath=//label[normalize-space(.)=${literal}]/following-sibling::input[1]`);
}

/** The app shell is up: auth booted from the refresh cookie and the organization loaded. */
export async function expectAppShell(page: Page) {
  await expect(page.getByRole("main")).toBeVisible({ timeout: 45_000 });
}

/**
 * Sign this browser context in through the API with `page.request`, which shares the
 * context's cookie jar: the httpOnly refresh cookie lands in the browser and the app
 * boots from it, exactly as after a real sign-in. One login per context: refresh
 * tokens rotate, and replaying one from another context revokes the whole session.
 *
 * `orgId` pins the active organization (the app keeps it in localStorage).
 */
export async function signInWithApi(page: Page, user: Credentials, opts: { orgId?: string; path?: string } = {}) {
  if (opts.orgId) {
    await page.addInitScript((id) => {
      try {
        window.localStorage.setItem("rag.org", id);
      } catch {
        /* storage unavailable */
      }
    }, opts.orgId);
  }
  const res = await withLoginRetry(() => page.request.post(`${API_PREFIX}/auth/login`, { data: user }), user.email);
  expect(res.status(), `sign-in for ${user.email}: ${await res.text().catch(() => "")}`).toBe(200);
  await page.goto(opts.path ?? "/overview");
  await expectAppShell(page);
}

/** Sign in through the real login form. Retries while the login endpoint is rate limited. */
export async function signInWithForm(page: Page, user: Credentials) {
  await page.goto("/login");
  await expect(page.getByRole("heading", { name: "Welcome back" })).toBeVisible();
  await page.getByLabel("Email").fill(user.email);
  await page.getByLabel("Password").fill(user.password);
  for (let attempt = 0; ; attempt++) {
    await page.getByRole("button", { name: "Sign in" }).click();
    // Scoped to the form: Next.js also renders a (visually hidden) role=alert route announcer.
    const alert = page.locator("form").getByRole("alert");
    const outcome = await Promise.race([
      page
        .waitForURL(/\/overview/, { timeout: 45_000 })
        .then(() => "ok" as const)
        .catch(() => "timeout" as const),
      alert
        .waitFor({ state: "visible", timeout: 45_000 })
        .then(() => "alert" as const)
        .catch(() => "timeout" as const),
    ]);
    if (outcome === "ok") break;
    if (outcome === "timeout") throw new Error(`sign-in for ${user.email}: neither /overview nor an error appeared`);
    const message = (await alert.textContent()) ?? "";
    if (!/too many requests/i.test(message) || attempt >= 4) throw new Error(`sign-in for ${user.email} failed: ${message}`);
    console.warn(`[e2e] ${user.email}: login form rate limited, waiting 15 s`);
    await sleep(15_000);
  }
  await expectAppShell(page);
}

export async function signOut(page: Page) {
  await page.getByRole("button", { name: "Account menu" }).click();
  await page.getByRole("menuitem", { name: "Sign out" }).click();
  await page.waitForURL(/\/login/);
  await expect(page.getByRole("heading", { name: "Welcome back" })).toBeVisible();
}

/** The sidebar navigation (desktop) — the drawer copy on mobile lives in a dialog. */
export function sidebar(page: Page): Locator {
  return page.getByRole("navigation", { name: "Main navigation" }).first();
}
