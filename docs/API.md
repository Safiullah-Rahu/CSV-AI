# API Guide

The REST API is versioned under `/api/v1` and fully described by OpenAPI at `/api/v1/openapi.json` (interactive docs at `/docs` and `/redoc` on the API origin).

## Authentication

| Client | How |
| --- | --- |
| Browser app | `POST /auth/login` → access token (15 min) in the body, refresh token in an httpOnly cookie. Refresh with `POST /auth/refresh` and header `X-Requested-With: fetch`. |
| Scripts / CLIs | `POST /auth/login?transport=body` returns both tokens in the body; refresh with `POST /auth/refresh {"refresh_token": "…"}`. |
| Integrations | Create a scoped API key in **API & Developers**; send `Authorization: Bearer rk_…`. Keys are bound to one organisation, carry only their scopes, expire, and are rate limited. |

Organisation context: user tokens must send `X-Org-Id: <organization id>` (list them with `GET /auth/me`). API keys are already bound to an organisation.

## Errors

Every error uses one shape and status code mapping:

```json
{ "error": { "code": "permission_denied", "message": "Missing permission: kb.upload", "request_id": "6f0c…", "details": {} } }
```

Codes: `validation_error` (422), `authentication_required` (401), `permission_denied` (403), `not_found` (404), `conflict` (409), `limit_exceeded` (402), `rate_limited` (429), `unsafe_request` (400), `not_available` (501), `connector_error`/`provider_error` (502), `internal_error` (500). Quote the `request_id` when reporting problems.

## Pagination

List endpoints that can grow return `{ "items": [...], "total": n, "limit": l, "offset": o }` and accept `limit`/`offset` (and endpoint-specific filters/sort parameters).

## Common workflows

```bash
API=https://rag.example.com/api/v1
KEY="Authorization: Bearer rk_xxxxxxxx.yyyyyyyy"

# Knowledge bases you can access
curl -s -H "$KEY" $API/knowledge-bases

# Upload documents (multipart, many files per request)
curl -s -H "$KEY" -F files=@handbook.pdf -F files=@prices.xlsx $API/knowledge-bases/$KB/documents

# Poll ingestion
curl -s -H "$KEY" "$API/documents?kb_id=$KB&status=processing"

# Ask (non-streaming) — answer, mode, confidence, citations with locators, timings
curl -s -H "$KEY" -H 'Content-Type: application/json' \
  -d '{"question":"Who approves expenses above 5,000 EUR?","kb_ids":["'$KB'"]}' $API/chat/ask

# Ask (streaming, Server-Sent Events: meta, status, sources, token…, done)
curl -N -H "$KEY" -H 'Content-Type: application/json' -d '{"question":"…"}' $API/chat/stream

# Natural language or SQL over a dataset (CSV/XLSX/Parquet upload)
curl -s -H "$KEY" -H 'Content-Type: application/json' -d '{"sql":"SELECT region, SUM(revenue_eur) FROM quarterly_sales GROUP BY 1"}' $API/datasets/$DS/query

# Live database query (validated, read-only, permission-checked)
curl -s -H "$KEY" -H 'Content-Type: application/json' -d '{"question":"Revenue by month for the last 12 months"}' $API/database-connections/$CONN/query

# Organisation event stream (SSE, resumable with Last-Event-ID)
curl -N -H "$KEY" $API/events/stream
```

## Endpoint map

| Area | Endpoints |
| --- | --- |
| Auth | `/auth/register`, `/auth/login`, `/auth/mfa/{setup,confirm,verify,disable}`, `/auth/refresh`, `/auth/logout[-all]`, `/auth/me`, `/auth/sessions`, `/auth/verify-email`, `/auth/password/{forgot,reset,change}`, `/auth/deactivate` |
| Organisations | `/orgs`, `/orgs/current`, `/orgs/current/settings`, `/members`, `/invitations`, `/invitations/{lookup,accept}`, `/support-access` |
| Structure | `/workspaces[/{id}/members]`, `/departments`, `/teams[/{id}/members]` |
| Access control | `/permissions`, `/me/permissions`, `/roles`, `/role-assignments`, `/grants` |
| Knowledge | `/knowledge-bases[/{id}]`, `/knowledge-bases/{id}/{documents,revisions,access}`, `/knowledge-bases/options`, `/documents[/{id}]`, `/documents/{id}/{chunks,versions,activity,download,reprocess}`, `/documents/bulk`, `/jobs[/{id}/{cancel,retry}]` |
| Chat | `/chat/ask`, `/chat/stream`, `/conversations[/{id}]`, `/conversations/{id}/{shares,export}`, `/messages/{id}/feedback`, `/citations/{id}/preview` |
| Data | `/datasets[/{id}]`, `/datasets/{id}/{preview,analysis,summary,query}`, `/database-connections[/{id}]`, `/database-connections/{id}/{test,discover,schema,selection,sync,pause,resume,health,runs,query}`, `/database-connections/providers`, `/query-logs`, `/approvals[/{id}/decide]` |
| Integrations | `/integrations/catalog`, `/integrations[/{id}]`, `/integrations/{id}/{test,sync,pause,resume,runs}` |
| Connections | `/connections/graph`, `/connections/health`, `/connections/node/{id}`, `/connections/data-flow`, `/events`, `/events/stream` |
| AI | `/ai/provider-kinds`, `/ai/providers[/{id}]`, `/ai/providers/{id}/{health,models}`, `/agents[/{id}]`, `/agents/tools`, `/agents/{id}/run` |
| Platform | `/api-keys`, `/webhooks[/{id}/deliveries]`, `/analytics/overview`, `/notifications[/read]`, `/search`, `/billing`, `/onboarding/{status,sample-kb}`, `/audit[/export]` |
| Platform admin | `/admin/{organizations,users,system,jobs,audit,plans}` |
| Health | `/health`, `/health/ready` |

## Webhooks

Subscribe to events (`connection.*`, `sync.*`, `document.*`, `index.updated`, `pipeline.*`, `kb.*`, `schema.changed`, `job.dead`). Each delivery is a `POST` with JSON `{id, type, data}` and headers `X-RAG-Event`, `X-RAG-Delivery`, `X-RAG-Timestamp`, `X-RAG-Signature: sha256=<hex>` where the HMAC-SHA256 is computed over `<timestamp>.<raw body>` with your signing secret. Failed deliveries retry with backoff; 4xx responses (except 408/429) are not retried.

```python
import hashlib, hmac
def verify(secret: str, timestamp: str, body: bytes, signature: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
```
