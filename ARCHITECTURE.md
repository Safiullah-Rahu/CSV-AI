# Architecture

Enterprise RAG & Knowledge Intelligence Platform (CSV-AI v3). A multi-tenant, permission-aware retrieval-augmented generation platform: organisations centralise documents, datasets, databases and web/code sources, and ask questions answered with verifiable citations — only from content each user may access.

```
                ┌──────────────────────── Browser ────────────────────────┐
                │ Next.js 15 (App Router) · TanStack Query · React Flow    │
                │ access token in memory · refresh token httpOnly cookie   │
                └──────────────┬──────────────────────────────┬───────────┘
                    HTTPS (Caddy, TLS)                  SSE (events, chat)
                               │                              │
        ┌──────────────────────▼──────────────────────────────▼───────────┐
        │ FastAPI  /api/v1   (modular monolith, sync handlers)             │
        │  auth · orgs · structure · roles/grants · knowledge · chat ·     │
        │  databases/datasets · integrations · connections graph/events ·  │
        │  ai providers/agents · analytics · api keys · webhooks · admin   │
        ├──────────────────────────────────────────────────────────────────┤
        │ Services: RBAC authorizer · ingestion · RAG engine · Text-to-SQL │
        │ connectors · billing limits · audit/activity · storage           │
        └──────┬───────────────┬──────────────────┬───────────────┬────────┘
               │ RLS-bound SQL │ LISTEN/NOTIFY    │ pub/sub       │ S3 API
        ┌──────▼──────┐ ┌──────▼──────┐   ┌───────▼─────┐  ┌──────▼─────┐
        │ PostgreSQL  │ │ Job worker  │   │   Redis     │  │ MinIO / S3 │
        │ + pgvector  │ │ (same code) │   │ events, RL  │  │ originals, │
        │ RLS on 42   │ │ SKIP LOCKED │   │ (optional)  │  │ datasets   │
        │ tables      │ │ retries/DLQ │   └─────────────┘  └────────────┘
        └─────────────┘ └─────────────┘
                    External: LLM providers · embedding APIs · customer databases · web/GitHub/S3
```

## Repository layout

| Path | Contents |
| --- | --- |
| `backend/app/config` | `pydantic-settings` configuration (kept from CSV-AI, extended) |
| `backend/app/db` | Declarative base, mixins, engine/session with tenant binding |
| `backend/app/models` | 47 tables: tenancy, RBAC, knowledge, connections/jobs/events, chat, AI, platform |
| `backend/app/rbac` | Permission catalogue + 11 predefined roles; the authorization engine |
| `backend/app/api` | Dependencies (principal, org context), middleware, v1 routers |
| `backend/app/ingestion` | Type detection & validation, 30+ format parsers, normalisation, pipeline job |
| `backend/app/rag` | Chunking, embeddings, pgvector/BM25 index, retrieval pipeline, grounding, answer engine |
| `backend/app/text2sql` | SQL validator/rewriter (sqlglot) and NL→SQL engine |
| `backend/app/connectors` | Database connectors (SQLAlchemy, MongoDB) and integrations (web, sitemap, RSS, GitHub, S3) |
| `backend/app/llm` | Provider abstraction from CSV-AI, extended to 10 provider kinds; DB-driven registry |
| `backend/app/data`, `app/analysis` | CSV-AI's loader/profiler/sampler and Chat/Summary/Analysis services (retained) |
| `backend/app/jobs` | PostgreSQL job queue, worker, scheduler, maintenance jobs |
| `backend/app/events` | Organisation-scoped pub/sub (memory or Redis) |
| `backend/app/agents` | Tool-using agent runtime |
| `backend/alembic` | Migrations: `0001` schema, `0002` RLS policies, audit trigger, vector indexes, plans |
| `backend/tests` | 164 tests against real PostgreSQL + pgvector (embedded) |
| `backend/eval` | RAG evaluation harness and results |
| `frontend/src` | Next.js app: design system, shell, 17 sections, chat, Connections command center |
| `deploy/` | Postgres init (app role), Caddyfile, backup/restore |

## Multi-tenancy

* Every tenant row carries `org_id`. **42 tables have Row-Level Security enabled and forced** with the policy `org_id = current_setting('app.current_org')` (plus a system-bypass flag used only by workers/platform jobs). `organizations` and `memberships` have membership-aware policies.
* Each transaction publishes its tenant context through `set_config(..., is_local => true)` in a SQLAlchemy `after_begin` hook, so the context is re-applied after every commit and can never leak to a pooled connection.
* The application must connect as a **non-superuser, non-BYPASSRLS** role (`rag_app`); the API refuses to start in production otherwise, and a test asserts it.
* Services also filter by `org_id` explicitly (defence in depth). Tests prove that a query with *no* org filter still returns only the bound tenant's rows and that cross-tenant writes are rejected by PostgreSQL.
* Platform-level audit events (`org_id IS NULL`) are write-only from request contexts.

## Authorization model

Two layers, both enforced server-side on every request:

1. **Capabilities** (`kb.upload`, `db.query`, `admin.audit`, …, 55 total) come from roles assigned to a user directly or via a team, at organisation or workspace scope, with optional expiry. Predefined roles: Owner, Admin, Security Administrator, Workspace Admin, Knowledge Base Manager, Data Source Manager, Developer, Analyst, Standard Member, Read-Only User; custom roles cannot exceed the creator's permissions or rank (escalation prevention). API keys are capped to their scopes.
2. **Resource privileges** (`none < read < write < manage`) on knowledge bases, documents, database connections and agents: implicit rules (owner, `kb.access_all`, KB visibility `org`/`workspace`) combined with explicit grants to users, teams, departments, roles, workspaces or the whole organisation. **Deny grants cap the privilege and beat every allow, including admin access.** Grants can expire. Documents in `restricted` ACL mode additionally require a document grant. Database grants can be narrowed to specific tables.

`Authorizer.access_scope()` computes, once per request, the readable knowledge bases plus denied and restricted-allowed documents. That scope is compiled into the SQL of every retrieval query, so unauthorized chunks are never fetched, scored, cited, cached or exported. Past citations are re-authorized whenever a conversation is viewed, previewed, shared or exported.

Platform administrators have **no** tenant content access. An organisation owner can grant a time-boxed, read-only support window; every request made under it is audited.

## Ingestion pipeline

`upload → spool (size limit, SHA-256) → detect (extension + magic bytes, secrets/executables rejected, zip-bomb checks) → store (object key per tenant) → Document + DocumentVersion → job`

The `ingest_document` job: parse → normalise (NFKC, control/bidi characters, whitespace) → language detection → auto-tags → prompt-injection screening → chunk → embed in batches with progress events → atomically replace the document's chunks/embeddings → derive datasets (Parquet + profile) for tabular files → emit `document.ingested`, `index.updated`, `pipeline.completed`.

Parsers keep citation locators: PDF page, heading trail, Excel sheet + row range, slide, code line range, notebook cell, EPUB chapter, source URL. Duplicate uploads (same hash in a KB) are detected; re-uploading with `replace_document_id` creates a new version and re-indexes incrementally.

## RAG engine

Separate components: chunking (structure, markdown, fixed, recursive, token, sentence, semantic, table-aware, parent-child), embeddings (hashing baseline, OpenAI/Azure/compatible, Cohere, Voyage, Hugging Face/TEI, Ollama, local ONNX via fastembed, custom HTTP), index, retrieval, reranking, context assembly, generation, citation verification, evaluation.

* **Vectors** live in `chunk_embeddings(embedding vector)` without a fixed dimension; per-dimension partial HNSW indexes (`(embedding::vector(768)) WHERE dim = 768`) let each knowledge base choose its own model. Dimension mismatches are rejected.
* **Sparse retrieval** is real Okapi BM25: PostgreSQL full-text (GIN, prefix matching for morphology) selects candidates; document frequencies, corpus size and average length are computed exactly over the *authorized* scope, so scores never depend on hidden documents.
* **Hybrid** fusion: reciprocal-rank fusion (default) or weighted. **Reranking**: a feature reranker (dense similarity, normalised BM25, IDF-weighted term coverage, proximity) — or Cohere / cross-encoder when configured. **MMR** diversification, **pseudo-relevance-feedback** query expansion (offline), LLM multi-query, parent-child expansion and contextual compression.
* **Conversational RAG**: follow-ups are condensed into standalone queries (LLM, or term carry-over offline).
* **Structured retrieval**: analytic questions over KBs with datasets or live database connections are routed to Text-to-SQL; the query result becomes a citable source.
* **Grounding**: sources are wrapped as untrusted data, instruction-like spans are neutralised, the model must cite `[S#]`; invented labels are stripped and reported; each cited sentence is checked for lexical support; insufficient evidence (IDF-weighted coverage + calibrated dense similarity) produces an honest refusal without calling the model. Without a configured model the engine produces a clearly labelled extractive answer; if the model fails before producing output, it falls back to extractive.

## Text-to-SQL

LLM generation (with one self-correction round) → **validator** (single read-only statement; no DDL/DML/session/admin nodes anywhere in the tree; dangerous functions blocked incl. `pg_sleep`, file readers, `set_config`; table/column allow-lists from the admin selection ∩ the user's grant constraints; `SELECT *` expanded to allowed columns; admin row filters injected by wrapping tables in filtered subqueries; hard `LIMIT`; join limits) → approval queue for sensitive tables or cross joins (executed with the *requester's* permissions, never self-approved) → execution in a read-only transaction with a server-side timeout (live DBs) or an in-memory DuckDB sandbox with external access disabled and configuration locked (datasets) → explanation and chart suggestion → `sql_query_logs`.

## Background processing

A PostgreSQL-backed queue replaces Celery+Redis (decision: transactional enqueue with the data it processes, one less service). Workers claim jobs with `FOR UPDATE SKIP LOCKED`, wake on `NOTIFY rag_jobs`, run handlers inside **tenant sessions** (RLS stays on), retry with exponential backoff + jitter, dead-letter after `max_attempts` with a notification, and recover stale leases. Idempotency keys prevent duplicate work. A scheduler (guarded by an advisory lock, one per cluster) enqueues scheduled syncs. `JOB_BACKEND=embedded` runs the worker inside the API for single-node installs.

## Real-time events and the Connections command center

`activity_events` is the source of truth; events are published after commit via the in-memory bus or Redis. `/events/stream` is an SSE endpoint, resumable with `Last-Event-ID`, filtered per subscriber by permission (connection, KB and job visibility). `/connections/graph` builds the ecosystem graph only from rows the viewer can see (organisation → workspaces → knowledge bases; sources → ingestion pipeline → knowledge bases → embedding models → pgvector → RAG engine → AI models → agents; live-query databases → RAG engine), with node status derived from health checks, sync runs and jobs, grouping for large graphs, and summary/health aggregates. The frontend renders it with React Flow (custom nodes, animated edges whose particles reflect real states, reduced-motion support).

## Frontend

Next.js App Router, TypeScript strict, Tailwind with CSS-variable design tokens (light/dark), Radix primitives, TanStack Query. The access token is held in memory; the refresh token is an httpOnly SameSite=Strict cookie scoped to `/api/v1/auth`, refreshed with a custom header (CSRF guard). Navigation is derived from the user's effective permissions; every action is re-checked by the API.

## Key data relationships

* `organizations 1─* memberships *─1 users`; `roles 1─* role_permissions`; `role_assignments` → user | team, scope org | workspace.
* `knowledge_bases 1─* documents 1─* document_versions`; `documents 1─* chunks 1─1 chunk_embeddings`; `chunks.parent_id` for parent-child; `documents 1─1 datasets` for tabular files.
* `resource_grants (resource_type, resource_id, subject_type, subject_id, privilege, effect, expires_at)`.
* `database_connections`/`integrations` → `credentials` (encrypted), `sync_runs`, `health_checks`, target `knowledge_bases`.
* `conversations 1─* messages 1─* citations`; `conversation_shares`; `message_feedback`.
* `jobs`, `activity_events`, `audit_events` (append-only trigger), `usage_records`, `sql_query_logs`, `approvals`, `webhooks 1─* webhook_deliveries`, `plans`/`subscriptions`, `api_keys`, `support_access_grants`.
