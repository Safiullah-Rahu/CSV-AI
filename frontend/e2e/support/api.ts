import fs from "node:fs";
import path from "node:path";

import { type APIRequestContext, type APIResponse, request } from "@playwright/test";

import { API_PREFIX, API_URL, sleep } from "./env";

export type Credentials = { email: string; password: string };

type Json = Record<string, any>;
type CallOptions = {
  /** Organization to act in; `null` sends no X-Org-Id (account-level endpoints such as POST /orgs). */
  org?: string | null;
  data?: unknown;
  multipart?: Record<string, string | { name: string; mimeType: string; buffer: Buffer }>;
  params?: Record<string, string | number | boolean>;
};

async function failure(res: APIResponse, what: string): Promise<Error> {
  let body = "";
  try {
    body = (await res.text()).slice(0, 600);
  } catch {
    /* no body */
  }
  return new Error(`${what} failed: HTTP ${res.status()} ${body}`);
}

/**
 * Login is rate limited per email (default 10/min) and per IP (30/min), in memory on
 * the server. Every browser context signs in on its own, so a busy run can touch the
 * limit; wait out the sliding window instead of failing.
 */
export async function withLoginRetry(attempt: () => Promise<APIResponse>, label: string): Promise<APIResponse> {
  for (let i = 0; ; i++) {
    const res = await attempt();
    if (res.status() !== 429 || i >= 5) return res;
    console.warn(`[e2e] ${label}: login rate limited, waiting 15 s (attempt ${i + 1}/5)`);
    await sleep(15_000);
  }
}

/** Sign in with the API and return an access token (refresh token in the body, not a cookie). */
export async function loginForToken(ctx: APIRequestContext, user: Credentials): Promise<string> {
  const res = await withLoginRetry(() => ctx.post(`${API_PREFIX}/auth/login`, { params: { transport: "body" }, data: user }), user.email);
  if (!res.ok()) throw await failure(res, `login ${user.email}`);
  const body = (await res.json()) as Json;
  if (body.mfa_required || !body.access_token) throw new Error(`login ${user.email}: two-factor is enabled on this account; the API helper does not handle it`);
  return body.access_token as string;
}

/** Thin authenticated client for seeding data, bound to one user and a default organization. */
export class Api {
  private constructor(
    readonly ctx: APIRequestContext,
    private token: string,
    readonly orgId: string | null,
    private readonly creds?: Credentials,
  ) {}

  static async create(opts: { token?: string; orgId: string | null; creds?: Credentials }): Promise<Api> {
    const ctx = await request.newContext({ baseURL: API_URL });
    const token = opts.token ?? (opts.creds ? await loginForToken(ctx, opts.creds) : "");
    if (!token) throw new Error("Api.create needs a token or credentials");
    return new Api(ctx, token, opts.orgId, opts.creds);
  }

  get accessToken() {
    return this.token;
  }

  async dispose() {
    await this.ctx.dispose();
  }

  private headers(org: string | null | undefined): Record<string, string> {
    const h: Record<string, string> = { Authorization: `Bearer ${this.token}` };
    const o = org === undefined ? this.orgId : org;
    if (o) h["X-Org-Id"] = o;
    return h;
  }

  async call<T = Json>(method: "GET" | "POST" | "PATCH" | "PUT" | "DELETE", p: string, opts: CallOptions = {}): Promise<T> {
    const send = () =>
      this.ctx.fetch(`${API_PREFIX}${p}`, {
        method,
        headers: this.headers(opts.org),
        params: opts.params,
        ...(opts.multipart ? { multipart: opts.multipart } : opts.data !== undefined ? { data: opts.data as any } : {}),
      });
    let res = await send();
    if (res.status() === 401 && this.creds) {
      // The access token outlived its 15 minutes; sign in again once.
      this.token = await loginForToken(this.ctx, this.creds);
      res = await send();
    }
    if (!res.ok()) throw await failure(res, `${method} ${p}`);
    if (res.status() === 204) return undefined as T;
    return (await res.json()) as T;
  }

  get<T = Json>(p: string, opts?: CallOptions) {
    return this.call<T>("GET", p, opts);
  }
  post<T = Json>(p: string, data?: unknown, opts: CallOptions = {}) {
    return this.call<T>("POST", p, { ...opts, data });
  }
  patch<T = Json>(p: string, data?: unknown, opts: CallOptions = {}) {
    return this.call<T>("PATCH", p, { ...opts, data });
  }

  // ------------------------------------------------------------------ domain helpers

  createOrg(name: string) {
    return this.post<{ id: string; name: string }>("/orgs", { name }, { org: null });
  }

  async roleId(key: string, org?: string): Promise<string> {
    const roles = await this.get<{ id: string; key: string }[]>("/roles", { org });
    const role = roles.find((r) => r.key === key);
    if (!role) throw new Error(`role ${key} not found (have: ${roles.map((r) => r.key).join(", ")})`);
    return role.id;
  }

  /** Invite an email address with a role; returns the invitation id. */
  async invite(email: string, roleKey: string, org?: string): Promise<string> {
    const roleId = await this.roleId(roleKey, org);
    const res = await this.post<{ invitation: { id: string } }>("/invitations", { email, role_id: roleId }, { org });
    return res.invitation.id;
  }

  createKnowledgeBase(name: string, opts: { visibility?: "restricted" | "workspace" | "org"; description?: string } = {}) {
    return this.post<{ id: string; name: string }>("/knowledge-bases", { name, description: opts.description ?? "Created by the end-to-end suite.", visibility: opts.visibility ?? "restricted" });
  }

  async uploadFile(kbId: string, filePath: string, mimeType = "text/markdown"): Promise<string> {
    const res = await this.call<{ results: { outcome: string; error?: string; document?: { id: string } }[] }>("POST", `/knowledge-bases/${kbId}/documents`, {
      multipart: { files: { name: path.basename(filePath), mimeType, buffer: fs.readFileSync(filePath) } },
    });
    const r = res.results[0];
    if (!r?.document || (r.outcome !== "created" && r.outcome !== "new_version")) throw new Error(`upload ${filePath}: ${r?.outcome} ${r?.error ?? ""}`);
    return r.document.id;
  }

  /** Poll until ingestion finishes; throws if the document fails. */
  async waitForDocument(docId: string, timeoutMs = 120_000): Promise<{ status: string; chunk_count: number; title: string; filename: string }> {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const doc = await this.get<{ status: string; chunk_count: number; title: string; filename: string; error?: string | null }>(`/documents/${docId}`);
      if (doc.status === "ready") return doc;
      if (doc.status === "failed") throw new Error(`document ${docId} failed to index: ${doc.error}`);
      if (Date.now() > deadline) throw new Error(`document ${docId} still ${doc.status} after ${timeoutMs} ms`);
      await sleep(1_000);
    }
  }
}

/** Accept an invitation as a new account through the API (what the /invite page does). */
export async function acceptInvitation(token: string, user: Credentials & { fullName: string }): Promise<void> {
  const ctx = await request.newContext({ baseURL: API_URL });
  try {
    const res = await ctx.post(`${API_PREFIX}/invitations/accept`, { data: { token, password: user.password, full_name: user.fullName } });
    if (!res.ok()) throw await failure(res, `accept invitation for ${user.email}`);
  } finally {
    await ctx.dispose();
  }
}
