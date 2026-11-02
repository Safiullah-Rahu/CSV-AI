# Changelog

## Unreleased

### Fixed (found while recording the demo against Claude Sonnet 5.5)
- Agents: a tool call written as function-call markup (`<invoke name="sql_query">…</invoke>`) instead of the requested JSON ended the run with the raw markup as its answer. The parser now reads both forms (first call only) and the prompt asks for JSON without markup.
- Agents: runs use an output limit of 4,096 tokens per model call (`limits.max_output_tokens` overrides it) instead of the 1,024-token chat default, which cut long answers mid-sentence. A reply that still stops at the limit is marked as cut off. Providers report this as `ChatResponse.truncated`.
- Agents: when the SQL itself needs a separate approval, the model is told not to retry (it used to file one approval request per retry).
- Text-to-SQL: a cross join against a one-row source (a date window or total in a CTE, `LIMIT 1`) no longer needs approval; cross joins that can multiply rows still do.
- Query results: the suggested chart skips id columns and charts a rate or percentage column on its own when there is one (it used to plot `carrier_id` beside the counts).
- Connections map: shows the platform model ("Platform model", e.g. anthropic · claude-sonnet-5-5) when the organisation has no provider of its own, instead of "No language model configured".

### Added
- README demo video and screenshots (`docs/media`).

## 3.0.0 — 2026-11-02 · Enterprise RAG & Knowledge Intelligence Platform

A ground-up transformation of CSV-AI v2 (single-user Streamlit app) into a multi-tenant platform. Git history of the reused modules is preserved (`git mv` into `backend/app`).

### Added (late in the 3.0.0 cycle)
- Ten SaaS integrations: Google Drive, OneDrive, SharePoint, Dropbox, Notion, Confluence Cloud, Slack, Microsoft Teams, GitLab and email (IMAP), with an OAuth 2.0 framework (authorization code + PKCE, signed single-use state, encrypted token storage, refresh with persisted rotation) and token alternatives for Notion, Confluence, Slack, GitLab and IMAP. GitLab is `tested` (live check against a public gitlab.com project); the others are `beta`, and OAuth-only vendors show "needs setup" until their `OAUTH_*` app is configured. Responses served with `Content-Encoding: gzip` no longer fail in the website/sitemap/RSS fetcher.
- Organisation security policy is now enforced: "require two-factor authentication" blocks members (and their API keys) without MFA, and "session lifetime" forces re-sign-in after the configured number of days. Admins must have MFA themselves before requiring it. Blocked members see an in-app gate that leads to the fix.
- Organisation settings are validated per section (unknown keys, ranges, https-only logo URLs).
- Branding is applied app-wide: the accent colour drives the `--accent` token (lightness clamped per theme) and the logo appears in the organisation switcher.
- "Porcelain / Graphite" design system (`docs/DESIGN.md`): Geist / Geist Mono / Instrument Serif type, warm light and neutral dark themes, raised content panel shell, org switcher and account menu in the sidebar, breadcrumb top bar, redesigned overview, sign-in, command palette and every page.
- Anthropic provider moved to API version `2026-10-21`: sampling parameters are no longer sent (they are rejected on every model), an optional `effort` setting maps to `output_config.effort`, and a `refusal` stop reason surfaces as a provider error so the engine falls back to an extractive answer.
- OpenAI provider works with current models: `max_completion_tokens` on OpenAI's own API (reasoning models reject `max_tokens`; compatible servers keep it), a one-time retry without `temperature` for models that accept only the default, and an updated suggested-model list.
- Refusal detection is one shared function (`grounding.is_refusal`) that recognises common phrasings ("The provided sources don't include …"). Previously an uncited decline in that form was withheld under the strict policy and replaced with unrelated passages.
- RAG evaluation harness runs hosted providers (keys from the environment or `--env-file`, never printed) and records answers and token usage.
- Live model discovery: each provider's model list comes from its API (OpenAI-compatible `/v1/models`, Anthropic Models API, Gemini `models.list`, Ollama), is stored as suggestions and refreshed daily, so new models appear without a release. Settings → AI providers fetches it for every provider type.
- Platform model fallback supports Anthropic as well as OpenAI (`DEFAULT_PROVIDER`, `ANTHROPIC_WORKSPACE_ID`), respects the organisation's provider allow-list, and is shown to organisations in AI settings. `scripts/dev.py --env-file` loads only the model-provider keys for local development.
- The demo seed no longer registers a local Ollama provider unless asked (`--ollama`).
- Agent runs no longer fail with HTTP 500 when a model sends malformed tool input; the error is fed back to the model.
- Deleting a database connection expires its pending SQL approvals.
- `backend/var/` (embedded PostgreSQL data, uploads, SQLite sources) and frontend build output are gitignored.
- Approved agent tools resume the paused run in the background with the requester's permissions (sources re-authorized), notify the requester, and update the run view live; requesters can withdraw pending approvals; `GET /approvals/{id}`.
- Playwright end-to-end suite (`frontend/e2e`, `npm run test:e2e`) covering sign-up, knowledge bases, chat with citations, access control, the MFA policy gate, the Connections map, the shell and mobile layouts; CI job using a pgvector service container. See `docs/TESTING.md`.
- CI backend and E2E jobs use a PostgreSQL 16 + pgvector service container; `devdb.py` derives connection URLs from parsed conninfo (Unix sockets on Linux/macOS); `dev.py` honours an explicit `DATABASE_URL`; `REGISTER_RATE_LIMIT_PER_HOUR` is configurable.
- Accessibility: every sign-up, invitation, password, two-factor and organization form label is linked to its input. The overview no longer scrolls sideways on phones.
- Renamed the project to **Knowledge Intelligence** (repository `knowledge-intelligence`, package names, compose project and images, crawler user agent).
- `IMPLEMENTATION_STATUS.md` and `docs/RAG_EVALUATION.md`.

### Added
- FastAPI backend (`backend/app`) with 47-table PostgreSQL schema, Alembic migrations, Row-Level Security on 42 tenant tables, append-only audit trigger, per-dimension pgvector HNSW indexes.
- Authentication: registration, login, rotating refresh tokens with reuse detection, lockout, email verification, password reset, TOTP MFA, sessions, invitations, multi-organisation membership and switching, deactivation/suspension.
- RBAC engine: permission catalogue, predefined + custom roles, workspace-scoped and expiring assignments, resource grants with deny-wins and expiry, restricted documents, table-scoped database grants, platform support-access windows.
- Ingestion of 30+ formats with locators, safety checks (magic bytes, secrets/executables, zip bombs, XXE), OCR hook, normalisation, language detection, auto-tags, prompt-injection screening, versioning, duplicate detection, background jobs with progress.
- RAG engine: 9 chunking strategies, 10 embedding provider kinds, dense + BM25 + hybrid retrieval, reranking, MMR, PRF expansion, multi-query, parent-child, compression, conversational rewriting, structured (SQL) evidence, grounding with citation verification, refusals, extractive fallback.
- Chat API with SSE streaming, conversations, regenerate, stop, feedback, sharing, export, citation previews re-authorized at view time.
- Database connectors (PostgreSQL, SQLite tested; MySQL/MariaDB/SQL Server/Oracle/Redshift/Supabase/RDS/MongoDB beta), schema discovery, selection with sensitive columns and row filters, full/incremental/scheduled sync with deletion propagation, health history.
- Text-to-SQL with sqlglot validation/rewriting, approvals, query logs; DuckDB sandbox for datasets.
- Integrations: website crawler (robots.txt, SSRF-checked redirects), sitemap, RSS/Atom, GitHub, S3 (beta); catalogue of planned OAuth connectors.
- Connections graph, health and data-flow endpoints; organisation event stream (SSE) with permission filtering; webhooks with HMAC signatures.
- Agents with a provider-agnostic JSON tool loop (KB search, SQL, document retrieval, summarise, extract, chart) and approval-gated tools.
- API keys, analytics, notifications, global search, plan limits, platform administration, onboarding with a sample knowledge base, demo seed and bootstrap-admin scripts.
- Next.js 15 frontend with design system, permission-aware navigation, all 17 sections, streaming chat with citations, and the animated Connections command center.
- PostgreSQL-backed job queue (SKIP LOCKED, LISTEN/NOTIFY, retries, dead-letter, stale-lease recovery, advisory-locked scheduler).
- Docker images (non-root), Compose stack (pgvector, Redis, MinIO, migrations, worker, Caddy TLS), backup/restore scripts.
- Backend tests on real PostgreSQL + pgvector; RAG evaluation harness and results.

### Changed
- LLM provider abstraction (from v2) kept and extended to OpenAI, Azure OpenAI, Anthropic, Gemini, Bedrock, Ollama, vLLM, OpenAI-compatible, Hugging Face and custom endpoints over `httpx`, with retries, usage accounting, health checks and fallbacks.
- CSV loader/profiler/sampler and Chat/Summary/Analysis services retained as the dataset analysis engine.

### Removed
- Streamlit UI, `streamlit_app.py`, Streamlit Docker files and `push-to-github.ps1` (which force-pushed after deleting `.git`). All remain in history at `1cabf1d`.
