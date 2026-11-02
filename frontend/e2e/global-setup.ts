import fs from "node:fs";

import { request } from "@playwright/test";

import { loginForToken } from "./support/api";
import { API_PREFIX, API_URL, OWNER_CACHE_FILE, RUN_STATE_FILE, STATE_DIR, strongPassword } from "./support/env";
import type { RunState } from "./support/state";

type CachedOwner = { email: string; password: string; fullName: string };

/**
 * One owner account per environment, one fresh organization per run.
 *
 * Registration is rate limited (REGISTER_RATE_LIMIT_PER_HOUR, default 10 per IP per
 * hour, in memory), so the owner is created once and its credentials are cached in
 * e2e/.state/owner.json for later local runs. Each run then gets its own organization,
 * so tests never see another run's data.
 */
export default async function globalSetup() {
  fs.mkdirSync(STATE_DIR, { recursive: true });
  const ctx = await request.newContext({ baseURL: API_URL });
  try {
    const health = await ctx.get(`${API_PREFIX}/health`).catch((err) => {
      throw new Error(`Backend not reachable at ${API_URL}${API_PREFIX}/health (${err.message}). Start the dev servers first (see docs/TESTING.md).`);
    });
    if (!health.ok()) throw new Error(`Backend health check returned HTTP ${health.status()} at ${API_URL}${API_PREFIX}/health`);

    const runId = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 5)}`;
    let owner: CachedOwner | null = null;
    let token: string | null = null;

    if (fs.existsSync(OWNER_CACHE_FILE)) {
      const cached = JSON.parse(fs.readFileSync(OWNER_CACHE_FILE, "utf8")) as CachedOwner;
      try {
        token = await loginForToken(ctx, cached);
        owner = cached;
      } catch (err) {
        console.warn(`[e2e] cached owner ${cached.email} could not sign in (${(err as Error).message.slice(0, 120)}); registering a new one`);
      }
    }

    if (!owner || !token) {
      owner = { email: `e2e-owner-${runId}@example.com`, password: strongPassword(), fullName: "E2E Owner" };
      const res = await ctx.post(`${API_PREFIX}/auth/register`, {
        params: { transport: "body" },
        data: { email: owner.email, password: owner.password, full_name: owner.fullName, organization_name: null },
      });
      if (res.status() === 429) throw new Error("Registration is rate limited (REGISTER_RATE_LIMIT_PER_HOUR). Wait, or raise the limit on the backend.");
      if (!res.ok()) throw new Error(`Registering the e2e owner failed: HTTP ${res.status()} ${await res.text()}`);
      token = ((await res.json()) as { access_token: string | null }).access_token;
      if (!token) throw new Error("Registration returned no access token (is EMAIL_VERIFICATION_REQUIRED on?). The suite needs it off.");
      fs.writeFileSync(OWNER_CACHE_FILE, JSON.stringify(owner, null, 2));
    }

    const auth = { Authorization: `Bearer ${token}` };
    const me = await ctx.get(`${API_PREFIX}/auth/me`, { headers: auth });
    if (!me.ok()) throw new Error(`GET /auth/me failed: HTTP ${me.status()}`);
    const userId = ((await me.json()) as { user: { id: string } }).user.id;

    const orgName = `e2e-run-${runId}`;
    const orgRes = await ctx.post(`${API_PREFIX}/orgs`, { headers: auth, data: { name: orgName } });
    if (!orgRes.ok()) throw new Error(`Creating the run organization failed: HTTP ${orgRes.status()} ${await orgRes.text()}`);
    const org = (await orgRes.json()) as { id: string };

    const state: RunState = { runId, owner: { ...owner, userId, accessToken: token }, orgId: org.id, orgName };
    fs.writeFileSync(RUN_STATE_FILE, JSON.stringify(state, null, 2));
    console.log(`[e2e] run ${runId}: owner ${owner.email}, organization ${orgName}`);
  } finally {
    await ctx.dispose();
  }
}
