import fs from "node:fs";
import path from "node:path";

/** Where the browser opens the app. The app proxies `/api/*` to the backend. */
export const BASE_URL = (process.env.E2E_BASE_URL || "http://localhost:3000").replace(/\/$/, "");

/** Where tests seed data through the API (defaults to the same origin as the browser). */
export const API_URL = (process.env.E2E_API_URL || BASE_URL).replace(/\/$/, "");

export const API_PREFIX = "/api/v1";

export const E2E_DIR = path.resolve(__dirname, "..");
export const STATE_DIR = path.join(E2E_DIR, ".state");
export const RUN_STATE_FILE = path.join(STATE_DIR, "run.json");
/** Survives between local runs so each run registers one account at most (registration is rate limited). */
export const OWNER_CACHE_FILE = path.join(STATE_DIR, "owner.json");
export const DATA_DIR = path.join(E2E_DIR, "data");

export const REPO_ROOT = path.resolve(E2E_DIR, "..", "..");
export const BACKEND_DIR = path.resolve(process.env.E2E_BACKEND_DIR || path.join(REPO_ROOT, "backend"));

/** Python interpreter that has the backend's dependencies (pgserver, psycopg). */
export function backendPython(): string {
  if (process.env.E2E_BACKEND_PYTHON) return process.env.E2E_BACKEND_PYTHON;
  const candidates = [path.join(BACKEND_DIR, ".venv", "Scripts", "python.exe"), path.join(BACKEND_DIR, ".venv", "bin", "python")];
  return candidates.find((p) => fs.existsSync(p)) ?? "python";
}

/** A password that satisfies the platform's strength rules. */
export function strongPassword(): string {
  return `E2e-${Math.random().toString(36).slice(2, 10)}-Pass9`;
}

export const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
