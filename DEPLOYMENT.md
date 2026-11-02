# Deployment & Operations Guide

This guide lets an engineer install, operate, back up and upgrade the platform on a Linux server (Docker Compose), and run it locally for development without Docker.

## 1. Requirements

| | Minimum | Recommended |
| --- | --- | --- |
| OS | Linux x86_64 with Docker Engine 24+ and Compose v2 | Ubuntu 22.04/24.04 LTS |
| CPU / RAM | 2 vCPU / 4 GB | 4–8 vCPU / 16 GB (local embedding models and OCR are CPU-bound) |
| Disk | 20 GB | SSD sized for documents ×3 (originals, Parquet copies, index) |
| Network | Ports 80/443 open; DNS `A` record for your domain | |

Optional: an LLM endpoint (OpenAI, Anthropic, Azure OpenAI, Gemini, Bedrock, or self-hosted Ollama/vLLM). Without one, the platform answers extractively with citations.

## 2. Install (single server)

```bash
git clone https://github.com/Safiullah-Rahu/knowledge-intelligence.git /opt/knowledge-intelligence
cd /opt/knowledge-intelligence
cp .env.example .env
```

Generate secrets and fill in `.env`:

```bash
python3 -c "import secrets; print('JWT_SECRET=' + secrets.token_urlsafe(48))"
python3 -c "from cryptography.fernet import Fernet; print('ENCRYPTION_KEYS=' + Fernet.generate_key().decode())"
openssl rand -base64 24   # use for POSTGRES_SUPERUSER_PASSWORD, APP_DB_PASSWORD, MINIO_ROOT_PASSWORD
```

Set at least: `DOMAIN`, `PUBLIC_APP_URL`, `CORS_ORIGINS`, `JWT_SECRET`, `ENCRYPTION_KEYS`, `POSTGRES_SUPERUSER_PASSWORD`, `APP_DB_PASSWORD`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, SMTP settings, and `BOOTSTRAP_ADMIN_EMAIL` / `BOOTSTRAP_ADMIN_PASSWORD` for the first platform administrator.

Start:

```bash
docker compose up -d --build
docker compose ps            # migrate must show "exited (0)"; api/worker/web/caddy "healthy/running"
docker compose logs -f api
```

What happens on first start:

1. `postgres` runs `deploy/postgres/10-init.sh`, creating the non-superuser application role, the database (owned by that role) and the `vector` extension.
2. `minio-init` creates the private document bucket.
3. `migrate` applies Alembic migrations (schema, Row-Level Security policies, audit trigger, vector indexes, plan seeds) and creates the bootstrap platform administrator.
4. `api` starts (refuses to start in `ENVIRONMENT=prod` with insecure settings or a superuser DB role), `worker` starts processing jobs and the scheduler.
5. `caddy` obtains a TLS certificate for `DOMAIN` and routes `/api/*` and `/docs` to the API (streaming flushed immediately) and everything else to the web app.

Open `https://<DOMAIN>/register` to create the first organisation, or sign in as the bootstrap admin for platform administration.

## 3. Configuration reference

All settings are environment variables (see `.env.example`, `backend/app/config/settings.py`). Important groups:

* **Security** — `JWT_SECRET`, `ENCRYPTION_KEYS` (rotate by prepending a new key; old keys still decrypt), `COOKIE_SECURE`, `ACCESS_TOKEN_TTL_MINUTES`, `REFRESH_TOKEN_TTL_DAYS`, rate limits, `EMAIL_VERIFICATION_REQUIRED`.
* **Network egress** — `ALLOW_PRIVATE_NETWORK_TARGETS` (customer databases/AI endpoints on private networks), `ALLOW_PRIVATE_WEB_INGESTION`, `SSRF_ALLOWLIST`, `SQLITE_ALLOWED_DIRS`.
* **Storage** — `STORAGE_BACKEND=local|s3`, `S3_*`, `MAX_UPLOAD_MB`.
* **Jobs** — `JOB_BACKEND=external|embedded`, `WORKER_CONCURRENCY`, `JOB_MAX_ATTEMPTS`.
* **Events** — `EVENT_BACKEND=redis` with `REDIS_URL` for multi-process deployments.
* **AI** — organisations configure providers in the UI; `PLATFORM_LLM_FALLBACK=true` with `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` lets a self-hosted operator provide a shared default (`DEFAULT_PROVIDER` decides which is tried first; `ANTHROPIC_WORKSPACE_ID` for organisation-level Anthropic keys). Organisations see which model that is under Settings → AI providers.
* **Billing** — `BILLING_ENABLED=false` (default) gives every organisation the unlimited plan.
* **SaaS integrations** — `OAUTH_<VENDOR>_CLIENT_ID` / `_CLIENT_SECRET` per vendor and `OAUTH_MICROSOFT_TENANT`; see below. `ALLOW_PRIVATE_WEB_INGESTION` / `SSRF_ALLOWLIST` also govern self-managed GitLab, Confluence site URLs and IMAP hosts.

### OAuth apps for SaaS integrations

Google Drive, OneDrive, SharePoint, Dropbox and Microsoft Teams connect only through OAuth. Notion, Confluence Cloud and Slack can use OAuth or a token (internal integration token; email + API token; `xoxb-` bot token). GitLab and email (IMAP) use a token or password and need no app. An operator registers each OAuth app **once per deployment**; every organisation then authorizes its own account with **Connect with …** in Integrations. Until a vendor's variables are set, its card shows **Needs setup** (admins see the variable names and the redirect URI) and the flow cannot start.

**Redirect URI** — register exactly `{PUBLIC_APP_URL}/api/v1/integrations/oauth/callback` with every vendor, e.g. `https://rag.example.com/api/v1/integrations/oauth/callback` (locally `http://localhost:3000/api/v1/integrations/oauth/callback`; the web app proxies `/api/*` to the API). Set the variables on **both `api` and `worker`** (the worker refreshes tokens during scheduled syncs) and restart them. Tokens are stored encrypted with `ENCRYPTION_KEYS`.

| Vendor (integrations) | Variables | Register at | Scopes and settings |
| --- | --- | --- | --- |
| Google (Google Drive) | `OAUTH_GOOGLE_CLIENT_ID`, `OAUTH_GOOGLE_CLIENT_SECRET` | Google Cloud console → APIs & Services: enable the Google Drive API, configure the consent screen, then Credentials → OAuth client ID → *Web application* | `https://www.googleapis.com/auth/drive.readonly`, requested with `access_type=offline&prompt=consent` so a refresh token is issued. It is a restricted scope: apps for external users need Google's verification. Docs/Sheets/Slides are exported as docx/xlsx/pptx. |
| Microsoft identity platform (OneDrive, SharePoint, Teams) | `OAUTH_MICROSOFT_CLIENT_ID`, `OAUTH_MICROSOFT_CLIENT_SECRET`, `OAUTH_MICROSOFT_TENANT` (default `common`) | Microsoft Entra admin center → App registrations → New registration, platform *Web* with the redirect URI; Certificates & secrets → client secret | Delegated Microsoft Graph permissions `offline_access`, `User.Read`, plus `Files.Read.All` (OneDrive), `Sites.Read.All` (SharePoint), `Team.ReadBasic.All`, `Channel.ReadBasic.All`, `ChannelMessage.Read.All` (Teams; needs admin consent). Use a directory id or domain as the tenant for a single-tenant app; an integration can override it (`tenant`). |
| Dropbox (Dropbox) | `OAUTH_DROPBOX_CLIENT_ID` (App key), `OAUTH_DROPBOX_CLIENT_SECRET` (App secret) | Dropbox App Console → Create app → *Scoped access*, *Full Dropbox*; add the redirect URI | Permissions `account_info.read`, `files.metadata.read`, `files.content.read`; `token_access_type=offline` for a refresh token. |
| Notion (Notion) | `OAUTH_NOTION_CLIENT_ID`, `OAUTH_NOTION_CLIENT_SECRET` | notion.so/my-integrations → New integration → *Public*, with the redirect URI | *Read content* capability. The token exchange uses HTTP Basic with the client id/secret. Only pages the user shares during consent are synced. |
| Atlassian (Confluence Cloud) | `OAUTH_ATLASSIAN_CLIENT_ID`, `OAUTH_ATLASSIAN_CLIENT_SECRET` | developer.atlassian.com → Developer console → Create → *OAuth 2.0 integration*; Authorization → callback URL; Permissions → Confluence API | Granular scopes `read:page:confluence`, `read:space:confluence`, plus `offline_access`; `audience=api.atlassian.com`. The Confluence site comes from `accessible-resources` (the integration's `site_url` picks one when the account has several); API calls go to `https://api.atlassian.com/ex/confluence/{cloudid}/wiki`. |
| Slack (Slack) | `OAUTH_SLACK_CLIENT_ID`, `OAUTH_SLACK_CLIENT_SECRET` | api.slack.com/apps → Create app → OAuth & Permissions → Redirect URLs | Bot token scopes `channels:read`, `channels:history`, `groups:read`, `groups:history`, `users:read` (`oauth.v2.access`). Invite the bot to every channel it should read. |

Only GitLab has been checked against the live service (a public gitlab.com project); the other integrations are **beta** — implemented and tested against stubbed vendor APIs. Run **Test** after the first connection for each vendor before relying on scheduled syncs.

## 4. AI providers and local models

Configure providers per organisation in **Settings → AI providers**. Credentials are encrypted at rest. Any hosted or self-hosted model can be used: OpenAI, Anthropic, Google Gemini, Azure OpenAI, AWS Bedrock, Ollama, vLLM, Hugging Face TGI, or any OpenAI-compatible server (LM Studio, llama.cpp, LocalAI, gateways such as LiteLLM). Model lists are discovered live from each provider's API (OpenAI-compatible `/v1/models`, Anthropic Models API, Gemini `models.list`, Ollama `/api/tags`), stored as suggestions, and refreshed daily by the worker, so new models appear without a platform release; any model id can also be typed. Azure (deployment names) and Bedrock take typed ids. For a self-hosted Ollama on the Docker network, add a service (e.g. `ollama/ollama`) and configure the provider URL `http://ollama:11434`; for an Ollama on the host, add its hostname to `SSRF_ALLOWLIST`. Ollama prompts are sized automatically (`num_ctx`).

Embedding models are chosen per knowledge base: `hashing` (built-in, offline, lexical), `local` (ONNX `BAAI/bge-small-en-v1.5` via fastembed — downloaded on first use, ~130 MB), or an API provider. Changing a knowledge base's embedding model schedules a re-index.

## 5. Operations

### Health & monitoring
* Liveness `GET /api/v1/health`; readiness `GET /api/v1/health/ready` (DB latency and migration revision, storage, Redis, job counts). Docker health checks use them.
* Logs are structured JSON (`LOG_JSON=true`) with request IDs; every API response carries `X-Request-ID`. Secrets are redacted before logging.
* Platform admins see system health, job queues, dead jobs (with retry), AI provider status and usage in **Administration**. Organisation admins see connection health in **Connections** and usage in **Analytics**.

### Background jobs
Scale workers horizontally: `docker compose up -d --scale worker=3`. Jobs are claimed with `SKIP LOCKED`; stale leases are recovered by the scheduler. Failed jobs retry with exponential backoff; jobs that exhaust retries are marked `dead` and the creator is notified.

### Backups
```bash
./deploy/backup.sh            # → backups/db-<ts>.dump + backups/storage-<ts>.tar.gz
```
Schedule nightly with cron and copy `backups/` off-site. `BACKUP_RETENTION_DAYS` (default 14) prunes old archives.

### Restore
```bash
./deploy/restore.sh backups/db-<ts>.dump
docker compose stop minio && gunzip -c backups/storage-<ts>.tar.gz | docker compose run --rm -T --entrypoint tar minio -C /data -xf - && docker compose start minio
```
Test restores regularly on a staging host.

### Upgrades
```bash
git pull
docker compose build
./deploy/backup.sh
docker compose up -d          # migrate runs first; api/worker start after it succeeds
```
Migrations are forward-only in production; take a backup before upgrading. Rolling back means restoring the backup and checking out the previous release.

### Encryption key rotation
1. Prepend a new Fernet key to `ENCRYPTION_KEYS` and restart `api`/`worker` (new secrets use the new key; old ones still decrypt).
2. Re-save credentials (or rotate them) to re-encrypt with the new key, then remove the old key.

## 6. Production hardening checklist

- [ ] `ENVIRONMENT=prod`, unique secrets, `COOKIE_SECURE=true`, real SMTP.
- [ ] DNS + TLS working; only ports 80/443 exposed (Postgres, Redis, MinIO stay on the Docker network).
- [ ] `ALLOW_PRIVATE_WEB_INGESTION=false` for multi-tenant SaaS; review `ALLOW_PRIVATE_NETWORK_TARGETS`.
- [ ] Give the platform read-only accounts on customer databases.
- [ ] Nightly backups copied off-site; restore tested.
- [ ] Log shipping/alerting on `level=ERROR` and dead jobs.
- [ ] Bootstrap admin password rotated after first login; MFA enabled for administrators.

## 7. Cloud / Kubernetes notes

The images are stateless (state lives in PostgreSQL, Redis and S3), so they map directly to managed services: RDS/Cloud SQL/Azure Database for PostgreSQL with pgvector, ElastiCache/Memorystore, S3/GCS (S3 API)/Azure Blob via an S3 gateway. Run `api` as a Deployment behind an ingress with response buffering disabled for `/api/v1/chat/stream` and `/api/v1/events/stream`; run `worker` as a separate Deployment; run `alembic upgrade head` as a pre-install/pre-upgrade Job. Use the readiness endpoint for probes. (Helm charts are not included yet.)

## 8. Local development (no Docker)

Requires Python 3.11+ and Node 20+. PostgreSQL 16 + pgvector run embedded via `pgserver`.

```bash
cd backend
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,local-embeddings]"
python scripts/dev.py                                # API on http://127.0.0.1:8000 (docs at /docs)
python scripts/seed_demo.py                          # optional demo org/users/content (dev credentials in the script)
```

```bash
cd frontend
npm install
npm run dev                                          # http://localhost:3000 (proxies /api to :8000)
```

Tests and quality:

```bash
cd backend && pytest -q && ruff check app tests
cd frontend && npx tsc --noEmit && npm run build
python backend/eval/run_eval.py --embedder local      # RAG evaluation
```

## 9. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| API exits with "Refusing to start" | Production checks failed — read the message (JWT secret, encryption keys, cookies, email backend). |
| "Database role bypasses Row-Level Security" | `DATABASE_URL` uses a superuser. Use the `rag_app` role created by the init script. |
| Documents stay `queued` | No worker running (`JOB_BACKEND=external` needs the `worker` service) — check `docker compose logs worker`. |
| Chat answers are "Extractive" | No AI provider configured for the organisation, or it is disabled/failing — see Settings → AI providers → Test. |
| Database connection test returns `unsafe_request` | The host resolves to a blocked range (metadata, loopback, platform DB) or a private range while `ALLOW_PRIVATE_NETWORK_TARGETS=false`. |
| SSE/live map not updating behind a proxy | Disable response buffering for `/api/v1/events/stream` and `/api/v1/chat/stream` (Caddy config does this). |
| Scanned PDF pages not indexed | OCR requires the `tesseract-ocr` package (included in the image by default). |
| `.doc` / `.ppt` rejected | Build the backend image with `--build-arg WITH_LIBREOFFICE=1`, or convert to `.docx`/`.pptx`. |
| An integration card shows **Needs setup** | That vendor's `OAUTH_*_CLIENT_ID` / `_CLIENT_SECRET` are not set on the API — see §3 "OAuth apps", then restart `api` and `worker`. |
| Vendor says `redirect_uri_mismatch` (or similar) | The registered redirect URI must equal `{PUBLIC_APP_URL}/api/v1/integrations/oauth/callback` exactly (scheme, host, port, no trailing slash). Check `PUBLIC_APP_URL`. |
| Back from the vendor with "authorization link expired" or "already used" | The state is valid for 10 minutes and once; a newer **Connect** replaces older links. Start again from the integration's menu. |
| Self-managed GitLab / IMAP host returns `unsafe_request` | The host is on a private network: set `ALLOW_PRIVATE_WEB_INGESTION=true` or add it to `SSRF_ALLOWLIST`. |
