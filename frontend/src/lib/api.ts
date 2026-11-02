/**
 * API client.
 *
 * - Access token lives in memory only (never localStorage).
 * - The refresh token is an httpOnly, SameSite=Strict cookie scoped to /api/v1/auth;
 *   refresh requests carry `X-Requested-With: fetch` (CSRF guard).
 * - The active organization id is sent as `X-Org-Id`.
 */

export type ApiErrorBody = { error?: { code?: string; message?: string; details?: unknown; request_id?: string } };

export class ApiError extends Error {
  status: number;
  code: string;
  details: unknown;
  requestId?: string;
  constructor(status: number, body: ApiErrorBody | null, fallback: string) {
    super(body?.error?.message || fallback);
    this.status = status;
    this.code = body?.error?.code || "http_error";
    this.details = body?.error?.details;
    this.requestId = body?.error?.request_id;
  }
}

const BASE = "/api/v1";
const ORG_KEY = "rag.org";

let accessToken: string | null = null;
let refreshing: Promise<string | null> | null = null;
let onAuthLost: (() => void) | null = null;

export function setAccessToken(token: string | null) {
  accessToken = token;
}

export function hasAccessToken() {
  return accessToken !== null;
}

export function setAuthLostHandler(fn: () => void) {
  onAuthLost = fn;
}

export function getOrgId(): string | null {
  try {
    return localStorage.getItem(ORG_KEY);
  } catch {
    return null;
  }
}

export function setOrgId(id: string | null) {
  try {
    if (id) localStorage.setItem(ORG_KEY, id);
    else localStorage.removeItem(ORG_KEY);
  } catch {
    /* storage unavailable: org selection is per page load */
  }
}

export async function refreshAccessToken(): Promise<string | null> {
  if (!refreshing) {
    refreshing = (async () => {
      try {
        const r = await fetch(`${BASE}/auth/refresh`, {
          method: "POST",
          credentials: "same-origin",
          headers: { "X-Requested-With": "fetch" },
        });
        if (!r.ok) return null;
        const data = await r.json();
        accessToken = data.access_token ?? null;
        return accessToken;
      } catch {
        return null;
      } finally {
        setTimeout(() => (refreshing = null), 0);
      }
    })();
  }
  return refreshing;
}

type Options = {
  method?: string;
  body?: unknown;
  query?: Record<string, string | number | boolean | undefined | null>;
  signal?: AbortSignal;
  noOrg?: boolean;
  headers?: Record<string, string>;
  raw?: boolean;
};

function buildUrl(path: string, query?: Options["query"]) {
  const url = new URL(BASE + path, window.location.origin);
  if (query) {
    for (const [k, v] of Object.entries(query)) {
      if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v));
    }
  }
  return url.pathname + url.search;
}

function headersFor(opts: Options, isForm: boolean): Record<string, string> {
  const h: Record<string, string> = { ...(opts.headers || {}) };
  if (accessToken) h.Authorization = `Bearer ${accessToken}`;
  const org = getOrgId();
  if (org && !opts.noOrg) h["X-Org-Id"] = org;
  if (!isForm && opts.body !== undefined) h["Content-Type"] = "application/json";
  return h;
}

export async function apiFetch(path: string, opts: Options = {}, retried = false): Promise<Response> {
  const isForm = typeof FormData !== "undefined" && opts.body instanceof FormData;
  const res = await fetch(buildUrl(path, opts.query), {
    method: opts.method || "GET",
    credentials: "same-origin",
    headers: headersFor(opts, isForm),
    body: opts.body === undefined ? undefined : isForm ? (opts.body as FormData) : JSON.stringify(opts.body),
    signal: opts.signal,
  });
  if (res.status === 401 && !retried && !path.startsWith("/auth/login")) {
    const token = await refreshAccessToken();
    if (token) return apiFetch(path, opts, true);
    onAuthLost?.();
  }
  return res;
}

export async function api<T = unknown>(path: string, opts: Options = {}): Promise<T> {
  const res = await apiFetch(path, opts);
  if (opts.raw) return res as unknown as T;
  if (!res.ok) {
    let body: ApiErrorBody | null = null;
    try {
      body = await res.json();
    } catch {
      body = null;
    }
    throw new ApiError(res.status, body, `Request failed (${res.status})`);
  }
  if (res.status === 204) return undefined as T;
  const ct = res.headers.get("content-type") || "";
  return (ct.includes("application/json") ? res.json() : res.text()) as Promise<T>;
}

export const get = <T,>(path: string, query?: Options["query"]) => api<T>(path, { query });
export const post = <T,>(path: string, body?: unknown, opts: Options = {}) => api<T>(path, { ...opts, method: "POST", body });
export const patch = <T,>(path: string, body?: unknown) => api<T>(path, { method: "PATCH", body });
export const put = <T,>(path: string, body?: unknown) => api<T>(path, { method: "PUT", body });
export const del = <T,>(path: string, body?: unknown) => api<T>(path, { method: "DELETE", body });

export type SSEEvent = { event: string; data: any; id?: string };

/** Server-Sent Events over fetch (so the Authorization header can be sent). */
export async function streamSSE(
  path: string,
  opts: Options & { onEvent: (ev: SSEEvent) => void; lastEventId?: string; onOpen?: () => void },
): Promise<void> {
  const headers: Record<string, string> = { Accept: "text/event-stream" };
  if (opts.lastEventId) headers["Last-Event-ID"] = opts.lastEventId;
  const res = await apiFetch(path, { ...opts, headers: { ...headers, ...(opts.headers || {}) } });
  if (!res.ok || !res.body) {
    let body: ApiErrorBody | null = null;
    try {
      body = await res.json();
    } catch {
      body = null;
    }
    throw new ApiError(res.status, body, `Stream failed (${res.status})`);
  }
  opts.onOpen?.();
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let event = "message";
      let id: string | undefined;
      const data: string[] = [];
      for (const line of chunk.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
        else if (line.startsWith("id:")) id = line.slice(3).trim();
      }
      if (!data.length) continue;
      let parsed: unknown = data.join("\n");
      try {
        parsed = JSON.parse(parsed as string);
      } catch {
        /* plain text event */
      }
      opts.onEvent({ event, data: parsed, id });
    }
  }
}

export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return "Something went wrong.";
}
