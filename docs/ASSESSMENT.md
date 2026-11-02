# Phase 1 — Technical Assessment of CSV-AI v2

Date: 2026-11-02 · Branch inspected: `v2` (commit `1cabf1d`, an orphan rewrite of the v1 history that lives on `origin/main`).

## 1. Existing architecture

CSV-AI v2 is a single-process Streamlit application (~40 Python files, ~2.5 kLOC):

| Layer | Contents | Quality |
| --- | --- | --- |
| `app/config` | `pydantic-settings` settings + Streamlit secrets merge | Good, small |
| `app/llm` | `LLMProvider` ABC (`complete`/`stream`), OpenAI, Anthropic, Ollama adapters, factory + model catalogue | Good abstraction, sync-only, no tool use, no usage/cost accounting |
| `app/data` | Encoding/delimiter-sniffing CSV loader, column profiler, stratified sampler, markdown prompt-context builder | Good, deterministic, tested |
| `app/prompts` | System prompts for chat/analyze/summary | Reasonable grounding language |
| `app/services` | `ChatService`, `SummaryService`, `AnalysisService` (UI-free) | Good separation |
| `app/ui` | Streamlit pages/components/state/theme | Single-user, session-state only |
| `tests` | 13 pytest tests (loader, profiler, factory) | Pass-level unit coverage |
| Ops | Dockerfile + compose for Streamlit, deployment notes, `push-to-github.ps1` | Single container |

There is no HTTP API, no persistence, no users, no authentication, no database, no retrieval index (v2 removed v1's FAISS flow on purpose), no background processing and no multi-tenancy.

## 2. Reusable components (kept and moved, with history, into `backend/app`)

* `app/data/*` — CSV loader, profiler, sampler and context builder become the **structured-data analysis** engine behind Data Sources → Datasets (profiling, summaries, NL questions).
* `app/llm/*` — the provider abstraction is extended (Gemini, Azure OpenAI, Bedrock, OpenAI-compatible/vLLM/HF endpoints, custom), gains usage accounting, health checks and a JSON tool protocol, and keeps its original interface so the CSV services keep working.
* `app/prompts`, `app/services` — the dataset chat/summary/analysis services are reused by the dataset API.
* `app/utils` — error hierarchy, logger, token estimation.
* The design principles in the old ARCHITECTURE.md (deterministic numbers first, LLM narrative second; UI-free services) carry over.

## 3. Technical debt

* `app/data/sampler.py` and `app/data/sampling_impl.py` are duplicates.
* Provider clients are SDK-specific and synchronous; no timeouts or retry policy for OpenAI/Anthropic.
* Model catalogue hard-coded in the factory.
* Streamlit session state is the only "database"; nothing survives a restart.
* `push-to-github.ps1` deletes `.git` and force-pushes — unsafe to keep in a product repo.
* Docker image installs `build-essential` and runs as root.

## 4. Security gaps (all addressed by the rebuild)

No authentication or authorization; API keys typed into the UI live in process memory per session; no upload limits beyond Streamlit's; no audit trail; no rate limiting; LLM prompt contains raw CSV rows with no prompt-injection boundary; container runs as root.

## 5. Missing capabilities (vs. the target brief)

Everything enterprise: tenancy, RBAC, persistence, document formats beyond CSV, a retrieval index, citations, database connectors, Text-to-SQL, integrations, events, admin, analytics, billing, API keys, workers, migrations.

## 6. Target architecture (decision summary)

* **Backend:** FastAPI + Pydantic v2 + SQLAlchemy 2 + Alembic, one modular monolith (`backend/app`) split by domain package; Celery workers share the same code base. No microservices until load proves the need.
* **Database:** PostgreSQL 16 only (no SQLite fallback for platform data) with Row-Level Security on every tenant table, enforced through a transaction-scoped `app.current_org` setting, plus explicit `org_id` filters in services (defence in depth).
* **Vectors:** pgvector in a separate `chunk_embeddings` table (undimensioned `vector` + per-dimension partial HNSW indexes), so each knowledge base can use its own embedding model.
* **Sparse retrieval:** PostgreSQL full-text (`tsvector` + GIN) for candidate generation, scored with real Okapi BM25 using exact document frequencies over the *authorized* corpus.
* **Jobs:** a persisted job table with retry/backoff/dead-letter semantics; Celery+Redis in production, an in-process thread runner for single-node/dev, eager mode for tests.
* **Events:** `activity_events` table + pub/sub (in-memory or Redis) streamed to the browser over Server-Sent Events.
* **Storage:** pluggable local filesystem / S3-compatible (MinIO) object storage.
* **Frontend:** Next.js (App Router) + TypeScript + Tailwind, Radix primitives, React Flow for the Connections Map, TanStack Query for data.
* **Local dev without Docker:** tests and dev use an embedded PostgreSQL 16 + pgvector (`pgserver`), so RLS and vector search are exercised for real on this Windows machine.

## 7. Migration strategy

1. Work on branch `platform/enterprise-rag`; `git mv` reusable modules into `backend/app` so blame/history follow the files.
2. Replace the Streamlit UI with the Next.js frontend; the Streamlit app is removed (still in history at `1cabf1d`). Its three workflows (chat with data, summarize, analyze) are re-implemented as Dataset features on top of the retained services.
3. Build domains in dependency order: platform foundation → knowledge/ingestion → RAG → databases/Text-to-SQL → Connections command center → integrations → agents → admin/analytics → hardening → deployment.
4. Keep `IMPLEMENTATION_STATUS.md` honest: implemented-and-tested vs. implemented-untested vs. planned.
