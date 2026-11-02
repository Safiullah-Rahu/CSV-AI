# Security

Security is a design constraint of this platform, not a feature list. This document describes the controls that exist in the code today, how they are tested, and the limitations you should know before deploying.

## Reporting a vulnerability

Please report suspected vulnerabilities privately to the maintainer (repository owner) rather than in public issues. Include reproduction steps and affected versions.

## Tenant isolation

| Control | Where | Tested by |
| --- | --- | --- |
| Row-Level Security **enabled and forced** on 42 tenant tables + membership-aware policies on `organizations`/`memberships` | `alembic/versions/0002_rls_policies.py` | `test_every_tenant_table_has_forced_rls`, `test_rls_hides_other_tenants_rows`, `test_rls_blocks_cross_tenant_writes` |
| Transaction-local tenant context re-applied after every commit | `app/db/session.py` | all integration tests |
| App connects as a non-superuser, non-BYPASSRLS role; production start refused otherwise | `app/main.py`, `deploy/postgres/10-init.sh` | `test_app_role_is_not_superuser` |
| Explicit `org_id` filters in services (defence in depth) | services/routers | cross-tenant API tests |
| Workers run job handlers inside tenant sessions | `app/jobs/queue.py` | ingestion/sync tests |

The system bypass flag (`app.rls_bypass`) is only set by `system_session()`, which is used by workers' claim step, the scheduler, invitation acceptance, API-key lookup, slug uniqueness and platform administration. It is a GUC, so a SQL-injection flaw in the application could in theory set it; parameterised queries everywhere and the absence of raw SQL from user input are the mitigation. A stronger deployment can run the API under a role whose policies do not honour the flag (planned).

## Authentication and sessions

* Argon2id password hashing with automatic re-hash; strength policy; timing-equalised unknown-user login; lockout after 5 failures (15 min); generic error messages; login/registration/reset rate limits.
* Short-lived JWT access tokens (15 min) bound to a server-side session; refresh tokens are opaque, stored as SHA-256 hashes, **rotated on every use**, and replaying a rotated token revokes the whole session family (theft detection).
* Refresh cookie: `httpOnly`, `SameSite=Strict`, `Secure` in production, path-scoped to `/api/v1/auth`; cookie-based refresh requires a custom `X-Requested-With` header (CSRF).
* Logout, logout-all, per-device session revocation, password change/reset revoke other sessions; account deactivation; organisation-level suspension; platform-level suspension revokes sessions.
* TOTP MFA (secret encrypted at rest) with a separate short-lived challenge token.
* Organisation security policy, enforced on every org-scoped request: *require two-factor authentication* blocks members without MFA (and API keys whose owner has none), and *session lifetime* rejects sessions whose original sign-in is older than the configured number of days. An admin can only require MFA after enabling it on their own account (`test_org_policy.py`).
* Email verification and password-reset tokens are single-use, hashed, and expire (24 h / 1 h); "forgot password" never reveals whether an account exists.
* API keys: `rk_` prefix + random body, stored as SHA-256, scoped to a subset of the creator's permissions, expiring, revocable, rate-limited, never able to create keys or act as platform admin.

## Authorization

See ARCHITECTURE.md → Authorization model. Highlights: deny beats allow (even for admins), expiring grants and role assignments, escalation prevention for custom roles and role assignment, owner protection (last owner cannot be removed or suspended), self-approval blocked for SQL approvals, approved queries execute with the requester's permissions, agents inherit the requesting user's permissions and can only narrow scope. Authorization denials are themselves audited.

## RAG-specific protections

| Threat | Control |
| --- | --- |
| Cross-tenant / unauthorized retrieval | Access scope compiled into every dense, sparse, keyword and hybrid query; BM25 statistics computed only over the authorized corpus; citations, previews, shares and exports re-authorized at view time (`test_rag_permissions.py`) |
| Prompt injection in documents / web pages | Injection screening at ingestion (flags chunks and documents), instruction-like spans neutralised before entering the prompt, sources wrapped as untrusted data with explicit system rules, optional quarantine of flagged chunks per KB (`test_prompt_injection_document_is_flagged_and_neutralized`) |
| Fabricated citations | Citations are built from retrieved chunks only; unknown `[S#]` markers are stripped and reported (`test_fabricated_citations_are_stripped`) |
| Unsupported claims | Sentence-level support check; unsupported statements are surfaced in the UI; strict policy appends a verification warning |
| Insufficient evidence | IDF-weighted evidence threshold → refusal without calling the model |
| Data exfiltration via markdown images | Injection patterns include image beacons; the chat renderer does not render raw HTML (react-markdown without rehype-raw) |
| Poisoned / malicious uploads | Extension + magic-byte agreement, executables and secret files (`.env`, keys, keystores) rejected, binary-in-text rejected, zip-bomb ratio/size limits, XML parsed with entity resolution and network disabled (XXE test), size limits enforced while streaming |
| Insecure agent tools | Tools run with the user's permissions; approval-required tools pause the run and create an approval request |

## Text-to-SQL

Validator-enforced read-only single statements, forbidden node and function lists (incl. `pg_sleep`, file readers, `dblink`, `set_config`, DuckDB `read_*`/`glob`), allow-listed tables/columns, sensitive/excluded columns removed from the model's schema and from `SELECT *`, admin row filters injected into the query, hard row limits, join limits, approvals, server-side statement timeouts, read-only transactions, DuckDB sandbox with external access disabled. 36 validator tests plus live PostgreSQL/SQLite tests (`test_text2sql_validator.py`, `test_databases.py`). Recommendation: also give the platform a read-only database account.

## Network egress (SSRF)

`app/core/netguard.py` resolves every user-supplied destination (database hosts, AI endpoints, webhooks, crawls) and blocks cloud metadata and link-local ranges always, loopback and **the platform's own database endpoint** always, and private ranges by policy (`ALLOW_PRIVATE_NETWORK_TARGETS`, `ALLOW_PRIVATE_WEB_INGESTION`). Web crawls re-validate every redirect hop, honour robots.txt, cap page size, and stay on the start host. SQLite connections are confined to administrator-approved directories and opened read-only. *Residual risk:* DNS rebinding between the check and the connection is not fully prevented for HTTP fetches.

## Secrets

* Database passwords, integration tokens, AI provider keys, webhook signing secrets and MFA secrets are encrypted with Fernet (`MultiFernet`, key rotation by appending keys) in the `credentials` table and never returned by the API (tests assert keys never appear in responses or audit logs).
* Logs pass through a redaction filter (passwords, tokens, bearer headers, connection-string credentials, `sk-`/`rk-` keys); connector errors are scrubbed before being stored or returned.
* Production start is refused with default JWT secret, missing encryption keys, insecure cookies or console email.

## SaaS integrations and OAuth

* **Authorization code + PKCE (S256)** for every OAuth vendor (Google, Microsoft identity platform, Dropbox, Notion, Atlassian, Slack). A fresh code verifier is generated per start, is never stored in the database, and travels only inside the signed state, **encrypted** with the platform Fernet key.
* **State** is a JWT signed with `JWT_SECRET` (type `oauth_state`, 10-minute expiry) carrying organization, user, integration and vendor. The callback is a browser redirect without a bearer token, so it authenticates through the state alone, then re-checks that the user is still an active member holding `org.integrations` and that the organization is active.
* **Single use without a migration:** starting a flow stores the SHA-256 of a random nonce in `integrations.cursor`; the callback compares it in constant time and clears it *before* contacting the vendor. Replays, older links (a newer start supersedes them) and vendor-side cancellations all consume or fail the state. Every outcome is audited (`integration.oauth_start`, `integration.oauth_connect` with success/failure/denied).
* **Token storage:** access token, refresh token, expiry and non-secret vendor extras (Atlassian cloud id and site, Slack team, Notion workspace) are encrypted JSON in the integration's `credentials` row. The API only ever returns `connected`, an account label and the connection time; tokens and the PKCE verifier never appear in responses, events or audit metadata (asserted in `test_integrations_saas.py`). The log redaction filter also masks Slack (`xox?-`), GitLab (`glpat-`), Notion (`ntn_`) and Google (`ya29.`) token formats and `code_verifier` / `refresh_token` / `access_token` / `client_secret` parameters. A secret cannot be supplied through the API for an OAuth-mode integration, and switching an integration between OAuth and token mode deletes the old credential.
* **OAuth app secrets** are deployment settings (`OAUTH_*_CLIENT_SECRET`), never stored per organization and never sent to the browser; the catalog only lists the *names* of missing variables. Starting a flow for an unconfigured vendor is refused (HTTP 501).
* **Refresh:** an expired access token (or a 401) triggers the refresh-token grant; rotated refresh tokens (Microsoft, Atlassian, Dropbox) are persisted immediately, from inside the sync job when needed. *Residual risk:* two concurrent syncs of one integration can race on a rotating refresh token; the loser fails until the next run or a reconnect.
* **Egress:** vendor APIs are fixed hosts. User-supplied hosts (GitLab `base_url`, Confluence `site_url`, IMAP `host`) pass the SSRF guard under the web-ingestion policy (`ALLOW_PRIVATE_WEB_INGESTION`, `SSRF_ALLOWLIST`). Redirects are followed by hand with every hop re-checked, and the bearer token is dropped whenever a redirect changes host (pre-signed download URLs from Graph). Responses are capped at 10 MB while streaming; `Retry-After` is honoured up to 60 s. IMAP requires SSL/TLS or STARTTLS with certificate verification; plaintext IMAP is refused.
* **Revocation:** deleting an integration deletes its tokens but does not revoke the grant at the vendor. Revoke the app there as well (Google account permissions, Entra enterprise apps, Slack app management, and so on).
* Synced content (emails, chat messages, wiki pages) is untrusted input: it goes through the same ingestion screening (prompt-injection flags, quarantine) as uploads.

## Web security

Security headers on API and frontend (CSP, `X-Frame-Options: DENY`, `nosniff`, Referrer-Policy, Permissions-Policy, HSTS in production), strict CORS allow-list, request IDs on every response, standardised error bodies without stack traces, Pydantic validation on every input.

The frontend CSP allows `img-src https:` so an organisation's branding logo (an admin-configured `https://` URL) can load. The image is requested with `referrerpolicy="no-referrer"`, but the logo host still sees members' IP addresses; organisations that consider this sensitive should host the logo on their own domain.

## Audit

Append-only `audit_events` (database trigger rejects UPDATE/DELETE outside system retention) covering authentication, membership, roles, grants, knowledge bases, documents, downloads, connections, credentials rotation, queries, approvals, exports, API keys, webhooks, support access and authorization denials, with actor, IP, user agent, request ID and sanitised metadata. CSV export for administrators.

## Data lifecycle

Document deletion removes the row, all versions, chunks, embeddings, dataset Parquet copies and stored objects, and scrubs snippets from past citations. Knowledge-base deletion cascades the same way. Credential deletion on connection removal. Organisation deletion deactivates immediately and a purge job removes every row and object. Integration syncs propagate source deletions; database syncs propagate deleted rows. Retention settings exist per organisation (conversation and audit retention values are stored; automated retention purge jobs are **planned**, see IMPLEMENTATION_STATUS.md).

## Dependency hygiene

Pinned frontend dependencies (Next.js 15.5.28, a patched release line); backend dependencies declared with minimum versions in `pyproject.toml`. Run `pip-audit` and `npm audit` in CI (CI pipeline not yet included).
