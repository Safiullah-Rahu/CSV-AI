# Implementation status

An honest map of what this platform does today. Every row is in exactly one
state:

- **Done**: implemented and verified by automated tests and/or end-to-end use
  on PostgreSQL 16 + pgvector.
- **Beta**: implemented, but not verified against the real external system
  (needs credentials or infrastructure that wasn't available).
- **Partial**: works, with a stated gap.
- **Planned**: not implemented. Where a catalog entry exists, the UI and API
  label it "planned" and refuse to create it; nothing is faked.

Last updated 2026-11-02. Backend: see the test count in the final report /
CI; frontend: `tsc` and `next build` clean, all 27 routes checked in a browser.

## Foundation

| Capability | State | Notes |
|---|---|---|
| FastAPI backend, versioned API (`/api/v1`) | Done | OpenAPI at backend `/docs` (not proxied under `/api`) |
| PostgreSQL 16 + pgvector, Alembic migrations | Done | 2 migrations; per-dimension partial HNSW indexes |
| Row-Level Security on all tenant tables | Done | Forced RLS on 42 tables; non-superuser app role; tests assert both |
| Next.js 15 / TypeScript / Tailwind frontend | Done | Design system in `docs/DESIGN.md`; light + dark |
| Retained CSV-AI analysis (loading, profiling, provider factory) | Done | Moved into `app/data`, `app/analysis`; original tests kept |
| Docker Compose stack (Postgres, Redis, MinIO, API, worker, web, Caddy) | Beta | Written and reviewed, not run (no Docker on the build machine) |
| CI workflow (lint, tests, typecheck, build) | Beta | `.github/workflows/ci.yml`, not yet run on GitHub |

## Identity and access

| Capability | State | Notes |
|---|---|---|
| Registration, email verification, password reset | Done | Console/memory email backends tested; SMTP backend is Beta |
| Access + rotating refresh tokens, reuse detection | Done | httpOnly SameSite=Strict cookie + CSRF header |
| Sessions list/revoke, lockout after failures | Done | |
| TOTP MFA | Done | |
| Invitations, multi-organization membership, switching | Done | |
| Org security policy: require MFA, maximum sign-in age | Done | Enforced on every org-scoped request (incl. API keys for MFA); blocked members see an in-app gate |
| Org branding: accent colour, logo | Done | Applied app-wide; logo must be `https://` |
| SSO (SAML / OIDC), SCIM | Planned | |
| Predefined roles (11) and custom roles with rank limits | Done | |
| Grants on knowledge bases, documents, connections, agents | Done | Deny wins, expiry, team inheritance, workspace scope |
| Enforcement before retrieval (permission filter compiled into SQL) | Done | Cited sources re-authorized on every view |
| Connection `write` privilege | Partial | Accepted and stored; currently behaves like `read` |
| Platform admins without customer content access | Done | Owner-granted, time-boxed, audited support access |

## Knowledge and ingestion

| Capability | State | Notes |
|---|---|---|
| Knowledge bases (visibility, workspace/department, settings) | Done | |
| Upload, dedup by hash, versioning, reprocess, delete propagation | Done | Deletes scrub stale citations |
| Formats parsed in tests | Done | pdf, docx, xlsx, pptx, odt, ods, odp, epub, md, html, txt, rtf, xml, csv, tsv, json, jsonl, parquet, py, sql, yaml, ipynb |
| Other code/config extensions (~40) | Done | Parsed as text with language tagging |
| Images via OCR (png, jpg, tiff, …) | Beta | Needs Tesseract; not installed on the build machine |
| Legacy `.doc` / `.ppt` via LibreOffice | Beta | Needs LibreOffice in the image |
| Upload safety: magic-byte check, executables/secrets blocked, zip-bomb and XXE protection | Done | |
| Prompt-injection screening and neutralization | Done | Pattern-based; flagged in the UI |

## Retrieval and answers

| Capability | State | Notes |
|---|---|---|
| 9 chunking strategies with locators (page/sheet/row/slide/section) | Done | |
| Embeddings: hashing (offline), local fastembed (bge-small) | Done | Both used in the evaluation |
| Embeddings: OpenAI, Azure OpenAI, Cohere, Voyage, Hugging Face, Ollama, custom | Beta | HTTP clients implemented; no live keys used |
| Dense, BM25 and hybrid (RRF / weighted) retrieval | Done | |
| Lexical reranking, MMR, PRF query expansion, query condensation | Done | |
| Multi-query, parent-child expansion, contextual compression | Partial | Implemented and configurable; not covered by automated tests |
| Grounding: source tagging, citation checks, refusals, strict-mode enforcement | Done | Fabricated citations stripped; uncited answers withheld in strict mode |
| Model-failure fallback | Partial | Falls back to an extractive answer if the model fails before output; a mid-answer failure ends with an error |
| Multi-hop / graph retrieval | Planned | |
| Evaluation harness and results | Done | See `docs/RAG_EVALUATION.md`; LLM-generated answers not credibly measured locally |

## Language models

| Provider | State | Notes |
|---|---|---|
| Anthropic (Claude Sonnet 5.5, Opus 5.5) | Done | Live end-to-end evaluation (`docs/RAG_EVALUATION.md`); API version `2026-10-21`, optional `effort`, org-level workspace header |
| Ollama (local) | Beta | Selectable provider; opt-in in the demo seed (`--ollama`). Not used for evaluation: small models on a CPU were slow and produced garbled output |
| OpenAI (GPT-5.4 mini, GPT-4.1 mini) | Done | Live end-to-end evaluation; `max_completion_tokens` on OpenAI's API, automatic retry without `temperature` for default-sampling-only models |
| Gemini, Azure OpenAI, vLLM / OpenAI-compatible, Hugging Face TGI, custom | Beta | Factory and request formats unit-tested; no live keys used. Azure still sends `max_tokens` |
| AWS Bedrock | Beta | Converse API still sends `temperature`; current Claude models may reject it there (unverified) |
| Keys encrypted at rest, never returned by the API | Done | Tested |
| Per-org provider allow-list, fallback chain | Done | |
| Any proprietary or open-source model | Done | Hosted APIs, Azure/Bedrock, Ollama, vLLM, TGI and any OpenAI-compatible server; any model id can be typed; agents can override the model |
| Live model discovery | Done | Lists fetched from each provider's API (OpenAI-compatible `/v1/models`, Anthropic Models API, Gemini, Ollama), stored as suggestions and refreshed daily by the worker. Verified live against OpenAI and Anthropic; Gemini and Ollama parsing unit-tested. Azure and Bedrock take typed ids |
| Platform model fallback | Done | `PLATFORM_LLM_FALLBACK` with Anthropic and/or OpenAI keys for organisations without their own provider; shown in Settings → AI providers |

## Chat and collaboration

| Capability | State | Notes |
|---|---|---|
| Streaming chat (SSE) with citations and source previews | Done | |
| Conversations, sharing, export, feedback | Done | |
| Structured (SQL) evidence in answers | Done | Dataset and live-database sources |

## Databases and Text-to-SQL

| Capability | State | Notes |
|---|---|---|
| PostgreSQL, SQLite, PostgreSQL + pgvector | Done | Tested against a second live PostgreSQL |
| MySQL, MariaDB, SQL Server, Azure SQL, Oracle, Redshift, Supabase, RDS, MongoDB | Beta | Drivers and dialects wired; not tested against live servers |
| Snowflake, BigQuery, Databricks, Redis, Elasticsearch, OpenSearch, Cassandra | Planned | Listed in the catalog as planned; creation refused |
| Schema discovery, table/column selection, sensitive columns, row filters | Done | |
| Full / incremental / scheduled sync with deletion propagation | Done | |
| SQL validation (sqlglot), allow-lists, LIMIT, row-filter injection, approvals | Done | |
| Read-only execution, statement timeout, DuckDB sandbox for datasets | Done | |
| SSRF guard on every outbound connection | Done | DNS-rebinding remains a residual risk (resolve-then-connect) |

## Integrations

| Integration | State | Notes |
|---|---|---|
| Website crawler (robots.txt, host-locked), sitemap, RSS/Atom | Done | Fetches of gzip-encoded pages fixed (they failed to decode and were skipped) |
| GitLab (gitlab.com or self-managed; token optional for public projects) | Done | Live check 2026-11-02 against public `gitlab-org/project-templates/express` without a token (project, recursive tree, raw blobs). Token and self-managed paths: stubbed tests only |
| GitHub, S3-compatible storage | Beta | |
| OAuth 2.0 framework (authorization code + PKCE S256, signed single-use state, token refresh with persisted rotation) | Done | Google, Microsoft identity platform, Dropbox, Notion, Atlassian 3LO, Slack. Tested end to end against stubbed vendor endpoints; no vendor's real consent screen has been exercised |
| Google Drive, OneDrive, SharePoint, Dropbox, Microsoft Teams | Beta | OAuth only. Need the vendor's OAuth app (`OAUTH_*` env vars); shown as "needs setup" until configured. Implemented and unit-tested against stubbed APIs; not verified live |
| Notion, Confluence Cloud, Slack | Beta | OAuth, or a token without an OAuth app (Notion internal integration token; Confluence email + API token; Slack `xoxb-` bot token). Not verified live |
| Email (IMAP) | Beta | SSL/TLS or STARTTLS with username + (app) password; each message becomes a Markdown document. Tested against a fake IMAP server only. No IMAP OAuth (XOAUTH2), so Gmail and Microsoft 365 need app passwords |
| SaaS sync gaps | Partial | Slack thread replies are not fetched (parent messages show the reply count); Confluence Server/Data Center is not supported (Cloud API only); Dropbox team spaces outside the member folder are not listed; disconnecting deletes stored tokens but does not revoke the grant at the vendor |

## Operations and platform

| Capability | State | Notes |
|---|---|---|
| Connections command center (live map, list, data flow, SSE) | Done | Graph built only from real resources |
| Activity events, permission-filtered SSE stream | Done | In-memory bus tested; Redis bus is Beta |
| Postgres job queue: retries, backoff, dead-letter, stale-lease requeue, scheduler | Done | |
| Retention (conversations, audit events) | Done | Hourly |
| Agents with tools that inherit the caller's permissions | Done | Malformed tool input from a model is reported back to it instead of failing the run; live run verified on Claude |
| Agent tool approvals | Done | An approved tool runs in the background with the requester's permissions and the agent finishes its answer; the requester is notified and the run view follows the approval live. Sources gathered before the pause are re-authorized on resume |
| Approvals lifecycle | Done | Requesters can withdraw pending requests; pending approvals expire when their connection is deleted; nobody can decide their own request |
| Analytics, global search, notifications | Done | Notifications currently cover failed background jobs only |
| API keys (scoped, revocable), signed webhooks with test delivery | Done | |
| Audit log (append-only, redacted, CSV export) | Done | |
| Billing readiness (plans, limits, usage) | Partial | No payment provider; self-hosted shows "billing disabled" |
| Backups / restore scripts | Beta | `deploy/backup.sh`, `deploy/restore.sh`; not exercised |

## Testing gaps

- Browser end-to-end suite (Playwright, `frontend/e2e`, 11 scenarios incl. mobile) runs green locally; its CI job is written but has not run on GitHub yet. No frontend unit tests.
- No load, soak or concurrency testing.
- Datasets list is unpaginated; `/members` is capped at 500.
