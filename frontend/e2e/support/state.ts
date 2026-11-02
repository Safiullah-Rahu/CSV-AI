import fs from "node:fs";

import { RUN_STATE_FILE } from "./env";

/** Written by global-setup.ts, read by every worker. */
export type RunState = {
  runId: string;
  owner: { email: string; password: string; fullName: string; userId: string; accessToken: string };
  /** A fresh organization for this run, owned by `owner`. */
  orgId: string;
  orgName: string;
};

export function readRunState(): RunState {
  if (!fs.existsSync(RUN_STATE_FILE)) throw new Error(`${RUN_STATE_FILE} is missing: run the suite with \`npx playwright test\` so global-setup.ts runs first`);
  return JSON.parse(fs.readFileSync(RUN_STATE_FILE, "utf8")) as RunState;
}

/** Names and emails for data a test creates; always `e2e-` prefixed and unique per run. */
export function uniqueName(run: RunState, tag: string): string {
  return `e2e-${tag}-${run.runId}`;
}

export function uniqueEmail(run: RunState, tag: string): string {
  return `e2e-${tag}-${run.runId}-${Math.random().toString(36).slice(2, 6)}@example.com`;
}
