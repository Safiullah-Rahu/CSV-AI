import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end tests against the real app: Next.js frontend + FastAPI backend (no mocked
 * APIs). The servers must already be running; see docs/TESTING.md.
 *
 *   E2E_BASE_URL   app origin (default http://localhost:3000)
 *   E2E_API_URL    origin used to seed data through /api (default: E2E_BASE_URL)
 */
const baseURL = process.env.E2E_BASE_URL || "http://localhost:3000";
const isCI = Boolean(process.env.CI);

export default defineConfig({
  testDir: "./e2e",
  outputDir: "./test-results",
  globalSetup: "./e2e/global-setup.ts",
  // Few workers: the backend rate-limits sign-ins per account and per IP, and runs
  // ingestion and chat on one embedded worker.
  workers: 2,
  fullyParallel: false,
  forbidOnly: isCI,
  retries: isCI ? 1 : 0,
  timeout: 120_000,
  expect: { timeout: 15_000 },
  reporter: isCI ? [["list"], ["github"], ["html", { open: "never", outputFolder: "playwright-report" }]] : [["list"], ["html", { open: "never", outputFolder: "playwright-report" }]],
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "off",
    actionTimeout: 15_000,
    navigationTimeout: 60_000,
  },
  projects: [
    {
      name: "desktop-chromium",
      testIgnore: /mobile\.spec\.ts/,
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } },
    },
    {
      name: "mobile-chromium",
      testMatch: /mobile\.spec\.ts/,
      use: { ...devices["Pixel 5"], viewport: { width: 390, height: 844 } },
    },
  ],
});
