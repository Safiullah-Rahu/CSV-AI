import { test as base, type Page } from "@playwright/test";

import { Api } from "./support/api";
import { signInWithApi } from "./support/browser";
import { readRunState, type RunState } from "./support/state";

type WorkerFixtures = {
  /** Owner credentials and this run's organization (from global-setup.ts). */
  run: RunState;
  /** The owner's API client, acting in this run's organization. Seed data through this. */
  ownerApi: Api;
};

type TestFixtures = {
  /** A page signed in as the owner in this run's organization (one login for this context). */
  ownerPage: Page;
};

export const test = base.extend<TestFixtures, WorkerFixtures>({
  run: [
    async ({}, use) => {
      await use(readRunState());
    },
    { scope: "worker" },
  ],
  ownerApi: [
    async ({ run }, use) => {
      const api = await Api.create({ token: run.owner.accessToken, orgId: run.orgId, creds: run.owner });
      await use(api);
      await api.dispose();
    },
    { scope: "worker" },
  ],
  ownerPage: async ({ page, run }, use) => {
    await signInWithApi(page, run.owner, { orgId: run.orgId });
    await use(page);
  },
});

export { expect } from "@playwright/test";
