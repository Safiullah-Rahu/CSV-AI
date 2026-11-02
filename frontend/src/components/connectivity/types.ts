/** Response shapes for the connectivity API (database connections, integrations, agents). */

export type ProviderStatus = "tested" | "beta" | "planned";

export type DbProvider = {
  key: string;
  label: string;
  family: string;
  port?: number | null;
  dialect?: string;
  status: ProviderStatus;
  category: "database" | "warehouse" | "nosql" | "vector" | string;
  requires?: string;
  driver_installed: boolean;
};

export type AccessMode = "live" | "ingest" | "hybrid";

export type TableSelection = {
  columns: string[];
  excluded: string[];
  sensitive: string[];
  row_filter: string | null;
  incremental_column: string | null;
  primary_key: string[];
  requires_approval: boolean;
  ingest: boolean;
};

export type DbConnection = {
  id: string;
  name: string;
  provider: string;
  provider_label: string;
  provider_status: ProviderStatus;
  category: string | null;
  endpoint: string;
  host: string | null;
  port: number | null;
  database: string;
  username: string | null;
  ssl_mode: string;
  has_credentials: boolean;
  options: Record<string, unknown>;
  access_mode: AccessMode;
  enabled: boolean;
  status: string;
  last_tested_at: string | null;
  last_success_at: string | null;
  last_error: string | null;
  schema_refreshed_at: string | null;
  table_count: number;
  selected_tables: string[];
  selection: Record<string, TableSelection> | null;
  query_limits: { max_rows?: number; timeout_s?: number; max_joins?: number; [k: string]: unknown };
  sync_interval_minutes: number | null;
  sync_paused: boolean;
  next_sync_at: string | null;
  last_sync_at: string | null;
  kb_id: string | null;
  workspace_id: string | null;
  owner_id: string | null;
  access: "none" | "read" | "write" | "manage";
  created_at: string | null;
  updated_at: string | null;
};

export type TestResult = { ok: boolean; latency_ms?: number; error?: string; code?: string; [k: string]: unknown };
export type DiscoverResult = { tables: number; added: string[]; removed: string[]; changed: string[] };

export type HealthPoint = { ok: boolean; latency_ms: number | null; error: string | null; checked_at: string };

export type SchemaColumn = { name: string; type: string; nullable: boolean };
export type SchemaTable = {
  schema: string | null;
  name: string;
  kind: "table" | "view" | "collection" | string;
  columns: SchemaColumn[];
  primary_key: string[];
  foreign_keys: { columns: string[]; references: string; referred_columns: string[] }[];
  row_estimate: number | null;
};
export type SchemaResponse = { schemas: string[]; tables: Record<string, SchemaTable>; refreshed_at: string | null };

export type SyncRun = {
  id: string;
  mode: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  items_seen: number;
  items_added: number;
  items_updated: number;
  items_deleted: number;
  items_failed?: number;
  error: string | null;
};

export type ChartSpec = { type: "line" | "bar" | string; x: string; y: string[]; data: Record<string, unknown>[] };

export type QueryOutcome = {
  status: "executed" | "rejected" | "pending_approval" | "failed" | "cannot_answer" | "approved" | string;
  question: string;
  generated_sql: string;
  sql: string;
  columns: string[];
  rows: unknown[][];
  row_count: number;
  truncated: boolean;
  duration_ms: number;
  errors: string[];
  approval_reasons: string[];
  explanation: string;
  chart: ChartSpec | null;
  log_id: string | null;
  approval_id: string | null;
};

export type QueryLog = {
  id: string;
  source_type: string;
  source_id: string;
  user_id: string | null;
  question: string;
  generated_sql: string;
  executed_sql: string;
  status: string;
  row_count: number | null;
  duration_ms: number | null;
  error: string | null;
  reasons: string[] | null;
  created_at: string;
};

export type Paged<T> = { items: T[]; total: number; limit: number; offset: number };

export type Approval = {
  id: string;
  kind: "sql_query" | "agent_tool" | string;
  status: string;
  summary: string;
  payload: {
    sql?: string;
    question?: string;
    source_kind?: string;
    source_id?: string;
    log_id?: string;
    reasons?: string[];
    tool?: string;
    input?: unknown;
    agent_name?: string;
    [k: string]: unknown;
  };
  /** SQL approvals: the query outcome. Agent approvals: the resumed run (or `{status: "resuming"}`). */
  result: QueryOutcome | null;
  mine?: boolean;
  decided_at?: string | null;
  requested_by: string;
  decided_by: string | null;
  reason: string;
  created_at: string;
};

export type Grant = {
  id: string;
  resource_type: string;
  resource_id: string;
  subject_type: "user" | "team" | "department" | "role" | "workspace" | "org" | string;
  subject_id: string | null;
  privilege: "read" | "write" | "manage" | string;
  effect: "allow" | "deny" | string;
  expires_at: string | null;
  constraints: { tables?: string[]; [k: string]: unknown };
  note: string;
  created_at: string;
};

export type KnowledgeBase = { id: string; name: string; description: string; access: string; status: string; is_sample: boolean };

export type Member = { user_id: string; email: string; full_name: string; status: string };
export type Named = { id: string; name: string };
export type Role = { id: string; key: string; name: string };

export type Job = {
  id: string;
  kind: string;
  status: "queued" | "running" | "succeeded" | "failed" | "dead" | "cancelled" | string;
  progress: number;
  message: string | null;
  attempts: number;
  max_attempts: number;
  error: string | null;
  result: Record<string, unknown> | null;
  finished_at: string | null;
};

/* ---------------------------------------------------------------- integrations */

export type IntegrationAuthMode = "oauth" | "token" | "none";

/** Platform-level OAuth app for a vendor (client id/secret come from the deployment's env). */
export type IntegrationOAuthApp = {
  provider: string;
  vendor: string;
  scopes: string[];
  configured: boolean;
  redirect_uri: string;
  /** Env var names an operator sets (never values). */
  env: string[];
  console?: string;
};

export type IntegrationCatalogItem = {
  key: string;
  label: string;
  category: string;
  status: ProviderStatus;
  auth: string;
  fields?: string[];
  auth_modes?: IntegrationAuthMode[];
  oauth?: IntegrationOAuthApp;
  /** Missing env vars; present (possibly empty) for OAuth-capable providers. */
  needs_setup?: string[];
  /** How a `tested` provider was verified. */
  verified?: string;
};

/** Non-secret OAuth status of one integration. Tokens are never returned by the API. */
export type IntegrationOAuthState = {
  provider: string;
  vendor: string;
  connected: boolean;
  account: string | null;
  connected_at: string | null;
};

export type Integration = {
  id: string;
  name: string;
  provider: string;
  provider_label: string;
  provider_status: ProviderStatus | null;
  category: string | null;
  config: Record<string, unknown>;
  has_secret: boolean;
  status: string;
  enabled: boolean;
  kb_id: string | null;
  sync_interval_minutes: number | null;
  sync_paused: boolean;
  next_sync_at: string | null;
  last_sync_at: string | null;
  last_success_at: string | null;
  last_error: string | null;
  created_at: string | null;
  auth_mode?: IntegrationAuthMode;
  oauth?: IntegrationOAuthState | null;
};

/* ---------------------------------------------------------------------- agents */

export type Agent = {
  id: string;
  name: string;
  description: string;
  instructions: string | null;
  kb_ids: string[];
  connection_ids: string[];
  provider_id: string | null;
  model: string | null;
  tools: string[];
  approval_tools: string[];
  retrieval_config: Record<string, unknown>;
  limits: { max_steps?: number; [k: string]: unknown };
  enabled: boolean;
  workspace_id: string | null;
  access: "none" | "read" | "write" | "manage";
  created_at: string;
};

export type AgentTool = { key: string; description: string };

export type AIProvider = {
  id: string;
  name: string;
  kind: string;
  kind_label: string;
  has_credentials: boolean;
  models: { id?: string; [k: string]: unknown }[];
  default_model: string | null;
  enabled: boolean;
  is_default: boolean;
  status: string;
  suggested_models: string[];
};

export type ProviderKind = { kind: string; label: string; keyless: boolean; embedding_only: boolean };

export type AgentStep = { step: number; tool: string; input: unknown; result?: string; ms?: number };

export type Citation = {
  label: string;
  ordinal: number;
  source_type: string;
  title: string;
  filename: string;
  document_id: string | null;
  kb_id: string;
  locator_text: string;
  snippet: string;
  score: number;
  cited: boolean;
  flagged?: boolean;
};

export type AgentRunResult = {
  status: "complete" | "pending_approval" | "error" | "max_steps" | string;
  answer: string;
  steps: AgentStep[];
  approval_id: string | null;
  citations: Citation[];
  chart: ChartSpec | null;
  usage: { prompt_tokens: number; completion_tokens: number };
};
