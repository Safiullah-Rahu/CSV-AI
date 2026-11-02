import { execFileSync } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { BACKEND_DIR, backendPython, sleep } from "./env";

/**
 * Get a usable token for an invitation the test just created.
 *
 * Invitation links are only ever emailed. The dev backend's `console` email
 * backend prints them to its stderr as
 *   [dev email] You're invited to <org> -> <email>: http://…/invite?token=…
 *
 * 1. If `E2E_BACKEND_LOG` points at that output (CI starts the backend with its
 *    output redirected to a file), read the real token from the newest line for
 *    this email.
 * 2. Otherwise (local runs against a dev server whose output goes elsewhere),
 *    choose a token and store its hash on this one invitation in the embedded
 *    dev database with `set_invite_token.py`.
 */
export async function invitationToken(invitationId: string, email: string): Promise<string> {
  const log = process.env.E2E_BACKEND_LOG;
  if (log) return tokenFromLog(log, email);
  return tokenViaDevDatabase(invitationId);
}

async function tokenFromLog(file: string, email: string): Promise<string> {
  const needle = ` -> ${email.toLowerCase()}: `;
  const deadline = Date.now() + 15_000;
  for (;;) {
    const text = fs.existsSync(file) ? fs.readFileSync(file, "utf8") : "";
    const line = text
      .split(/\r?\n/)
      .reverse()
      .find((l) => l.includes("[dev email]") && l.includes(needle) && l.includes("/invite?token="));
    if (line) {
      const url = line.slice(line.indexOf(needle) + needle.length).trim();
      const token = new URL(url).searchParams.get("token");
      if (token) return token;
    }
    if (Date.now() > deadline) throw new Error(`no invitation email for ${email} in ${file} (is the backend using EMAIL_BACKEND=console?)`);
    await sleep(250);
  }
}

function tokenViaDevDatabase(invitationId: string): string {
  const token = randomBytes(32).toString("base64url");
  const hash = createHash("sha256").update(token, "utf8").digest("hex");
  const script = path.join(__dirname, "set_invite_token.py");
  try {
    execFileSync(backendPython(), [script, invitationId, hash], {
      encoding: "utf8",
      timeout: 60_000,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, DEVDB_DIR: process.env.DEVDB_DIR || path.join(BACKEND_DIR, "var", "pgdata") },
    });
  } catch (err: any) {
    throw new Error(
      `Could not read the invitation token. Set E2E_BACKEND_LOG to the backend's output file, or E2E_BACKEND_PYTHON to the backend's Python ` +
        `so set_invite_token.py can reach the embedded dev database.\n${err?.stderr || err?.message || err}`,
    );
  }
  return token;
}
