/** Response shapes for the knowledge / documents / datasets API (mirrors backend/app/api/v1). */

export type Access = "none" | "read" | "write" | "manage";
export type Visibility = "restricted" | "workspace" | "org";
export type GroundingPolicy = "strict" | "balanced" | "open";

export type Page<T> = { items: T[]; total: number; limit: number; offset: number };
export type Ok = { ok: boolean; detail: string | null };

export type KBStats = {
  documents: number;
  ready: number;
  failed: number;
  processing: number;
  chunks: number;
  storage_bytes: number;
  records: number;
  last_indexed_at: string | null;
};

export type EmbeddingConfig = { provider?: string; model?: string; dim?: number; provider_id?: string | null; [k: string]: unknown };

export type ChunkingConfig = {
  strategy: string;
  chunk_size: number;
  chunk_overlap: number;
  separators?: string[];
  contextual_headers: boolean;
  parent_size: number;
  child_size: number;
  semantic_percentile: number;
};

export type RetrievalConfig = {
  mode: string;
  top_k: number;
  candidate_k: number;
  fusion: string;
  rrf_k: number;
  alpha: number;
  rerank: string;
  rerank_top_n: number;
  min_score: number;
  min_evidence: number;
  mmr: boolean;
  mmr_lambda: number;
  query_expansion: string;
  multi_query: number;
  compression: boolean;
  max_context_tokens: number;
  quarantine_suspicious: boolean;
};

export type GenerationConfig = { grounding_policy?: GroundingPolicy; [k: string]: unknown };

export type KnowledgeBase = {
  id: string;
  name: string;
  description: string;
  workspace_id: string | null;
  department_id: string | null;
  owner_id: string | null;
  tags: string[];
  category: string;
  visibility: Visibility;
  embedding_config: EmbeddingConfig;
  chunking_config: Partial<ChunkingConfig>;
  retrieval_config: Partial<RetrievalConfig>;
  generation_config: GenerationConfig;
  status: string;
  config_version: number;
  is_sample: boolean;
  created_at: string;
  updated_at: string;
  access: Access;
  stats: Partial<KBStats>;
};

export type FormatSupport = { ext: string; family: string; available: boolean; requires: string | null };

export type KBOptions = {
  chunking_strategies: string[];
  chunking_defaults: ChunkingConfig;
  retrieval_defaults: RetrievalConfig;
  embedding_providers: { key: string; label: string }[];
  formats: FormatSupport[];
  max_upload_mb: number;
};

export type KBUpdateResult = { knowledge_base: KnowledgeBase; reindex_scheduled: boolean };

export type KBRevision = { version: number; config: Record<string, unknown>; changed_by: string | null; created_at: string };

export type AccessGrantSummary = { subject_type: SubjectType; subject_id: string | null; privilege: Privilege; effect: Effect; expires_at: string | null };
export type KBAccessSummary = { visibility: Visibility; owner_id: string | null; grants: AccessGrantSummary[]; your_access: Access };

export type SubjectType = "user" | "team" | "department" | "role" | "workspace" | "org";
export type Privilege = "read" | "write" | "manage";
export type Effect = "allow" | "deny";

export type Grant = {
  id: string;
  resource_type: string;
  resource_id: string;
  subject_type: SubjectType;
  subject_id: string | null;
  privilege: Privilege;
  effect: Effect;
  expires_at: string | null;
  constraints: Record<string, unknown>;
  note: string;
  created_at: string;
};

export type DocumentStatus = "queued" | "processing" | "ready" | "failed" | "deleting" | string;

export type DocumentMeta = {
  auto_tags?: string[];
  warnings?: string[];
  source_metadata?: Record<string, unknown>;
  security?: { injection_suspect?: boolean; flagged_chunks?: number; findings?: string[] };
  embedding?: { model?: string; dim?: number };
  chunking?: { strategy?: string; chunk_size?: number };
  processing_ms?: number;
  last_job_id?: string;
  custom?: Record<string, unknown>;
  [k: string]: unknown;
};

export type DocumentRow = {
  id: string;
  kb_id: string;
  title: string;
  filename: string;
  ext: string;
  mime_type: string;
  size_bytes: number;
  owner_id: string | null;
  status: DocumentStatus;
  parser: string;
  page_count: number | null;
  char_count: number;
  chunk_count: number;
  token_count: number;
  language: string | null;
  tags: string[];
  meta: DocumentMeta;
  error: string | null;
  current_version: number;
  last_indexed_at: string | null;
  embedding_status: string;
  source_type: string;
  source_url: string | null;
  acl_mode: "inherit" | "restricted" | string;
  created_at: string;
  updated_at: string;
  kb_name: string | null;
};

export type UploadOutcome = "created" | "new_version" | "duplicate" | "unchanged" | "rejected" | "skipped";
export type UploadResult = { filename: string; outcome: UploadOutcome; error?: string; document?: DocumentRow; path?: string };
export type UploadResponse = { results: UploadResult[]; accepted: number; rejected: number };

export type ChunkLocator = {
  section?: string[];
  page?: number;
  page_end?: number;
  sheet?: string;
  slide?: number;
  row_start?: number;
  row_end?: number;
  line_start?: number;
  line_end?: number;
  table_index?: number;
  kind?: string;
  ocr?: boolean;
  language?: string;
  url?: string;
  chapter?: string;
  cell?: string | number;
  injection_suspect?: boolean;
  injection_findings?: string[];
  [k: string]: unknown;
};
export type Chunk = { id: string; ordinal: number; content: string; token_count: number; metadata: ChunkLocator; parent_id: string | null };

export type DocumentVersion = { version: number; filename: string; size_bytes: number; sha256: string; created_by: string | null; created_at: string; current: boolean };

export type DocumentActivity =
  | { kind: "event"; id: number; type: string; resource_type: string | null; resource_id: string | null; actor_user_id: string | null; severity: string; message: string; payload: Record<string, unknown>; created_at: string | null }
  | { kind: "job"; id: string; type: string; status: string; attempts: number; error: string | null; created_at: string; finished_at: string | null };

export type BulkResult = { succeeded: number; failed: { document_id: string; error: string }[] };

/* --------------------------------------------------------------- directory */
export type Member = {
  user_id: string;
  email: string;
  full_name: string;
  status: string;
  user_status: string;
  title: string;
  department_id: string | null;
  department: string | null;
  roles: Record<string, unknown>[];
  teams: string[];
  joined_at: string;
  last_login_at: string | null;
  mfa_enabled: boolean;
};
export type Team = { id: string; name: string; description: string; department_id: string | null; workspace_id: string | null; created_at: string; member_ids: string[] };
export type Department = { id: string; name: string; description: string; created_at: string; member_count: number };
export type Workspace = { id: string; name: string; description: string; created_at: string; member_count: number; kb_count: number; is_member: boolean };
export type Role = { id: string; key: string; name: string; description: string; is_system: boolean; rank: number; permissions: string[]; assignment_count: number };
export type AIProvider = { id: string; name: string; kind: string; kind_label: string; has_credentials: boolean; enabled: boolean; is_default: boolean; status: string; default_embedding_model: string | null };

/* ---------------------------------------------------------------- datasets */
export type DatasetTableRef = { name: string; rows: number; columns: string[] };
export type Dataset = {
  id: string;
  name: string;
  document_id: string;
  kb_id: string;
  kb_name: string;
  table_name: string;
  tables: DatasetTableRef[];
  row_count: number;
  column_count: number;
  source_type: string | null;
  filename: string | null;
  updated_at: string;
};

export type ColumnProfile = {
  name: string;
  dtype: string;
  missing_pct: number;
  unique: number;
  sample: unknown[];
  mean: number | null;
  std: number | null;
  min: unknown;
  max: unknown;
  median: number | null;
};
export type TableProfile = { rows: number; columns: ColumnProfile[]; duplicate_rows: number; memory_mb: number };
export type DatasetDetail = Dataset & { profile: { tables: (DatasetTableRef & { storage_key?: string; profile?: TableProfile })[] } | null };

export type DatasetPreview = { columns: string[]; rows: unknown[][]; total: number; offset: number };

export type DatasetAnalysis = {
  rows: number;
  columns: number;
  missing: { column: string; missing: number; pct: number }[];
  numeric: Record<string, number | string | null>[];
  top_values: Record<string, { value: string | null; count: number }[]>;
  correlation: { columns: string[]; matrix: number[][] } | null;
  duplicates: number;
};

export type DatasetSummary = { summary: string; mode: "deterministic" | "llm"; model?: string };

export type QueryChart = { type: "bar" | "line"; x: string; y: string[]; data: Record<string, unknown>[] };
export type QueryResult = {
  status: "executed" | "rejected" | "pending_approval" | "failed" | "cannot_answer" | string;
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
  chart: QueryChart | null;
  log_id: string | null;
  approval_id: string | null;
};
