# Testing

Three layers, from fastest to slowest:

| Layer | Where | What it covers |
|---|---|---|
| Backend tests | `backend/tests/` (pytest) | API, auth, tenancy/RLS, RBAC, RAG permissions, parsers, text-to-SQL validation |
| RAG evaluation | `backend/eval/run_eval.py` | Retrieval and answer quality, grounding, latency (see [RAG_EVALUATION.md](RAG_EVALUATION.md)) |
| End-to-end (browser) | `frontend/e2e/` (Playwright) | Real user flows through the Next.js app against the real backend |

## Backend tests

```bash
cd backend
pip install -e ".[dev]"
ruff check app tests
pytest -q
```

`tests/conftest.py` provisions a fresh, migrated PostgreSQL 16 + pgvector database per
session with the embedded `pgserver` (no Docker), or uses `TEST_DATABASE_URL` if set. The
app connects as a non-superuser role so row-level security is really enforced. Email goes
to an in-memory outbox, jobs run eagerly, and the login rate limit is raised. Fixtures
`make_user`, `invite_member` and the `Api` client show how to register users, invite
members and act in an organization.

## RAG evaluation

```bash
cd backend
python eval/run_eval.py                       # hashing embedder, extractive answers (no keys, no network)
python eval/run_eval.py --embedder local      # fastembed BAAI/bge-small-en-v1.5 (if installed)
python eval/run_eval.py --llm ollama:qwen2.5:7b-instruct --limit 10
python eval/run_eval.py --embedder local --llm anthropic:claude-sonnet-5-5 \
  --configs hybrid+rerank --env-file ../.env.eval   # keys read from the file, never printed
```

It runs the application in-process against an embedded PostgreSQL, uploads the corpus
through the API, asks every question through `/chat/ask` and writes per-question results to
`backend/eval/results/<timestamp>.json`. Runs with a hosted `--llm` make one paid model call
per question.

## End-to-end tests (Playwright)

The suite drives Chromium against the running app. Nothing is mocked: tests create their
own data through the real API and the real UI.

### Run locally

1. Start the dev servers (two terminals):

   ```bash
   cd backend && python scripts/dev.py      # API on http://127.0.0.1:8000
   cd frontend && npm run dev               # app on http://localhost:3000 (proxies /api)
   ```

2. Install and run:

   ```bash
   cd frontend
   npm install
   npx playwright install chromium          # only if the browser is not installed yet
   npm run test:e2e                         # whole suite, headless
   npm run test:e2e:ui                      # Playwright UI mode
   npx playwright test e2e/chat.spec.ts     # one file
   npx playwright test --project=mobile-chromium
   npx playwright show-report               # HTML report of the last run
   ```

`@playwright/test` is pinned to 1.61.0 (the same version as the Python Playwright on the
dev machine), so it reuses the browsers in `%LOCALAPPDATA%\ms-playwright` / `~/.cache/ms-playwright`.

Projects: `desktop-chromium` (1440×900, every spec except `mobile.spec.ts`) and
`mobile-chromium` (Pixel 5 emulation at 390×844, `mobile.spec.ts` only). Two workers,
no retries locally (one on CI), trace and screenshot kept on failure in
`frontend/test-results/`, HTML report in `frontend/playwright-report/` (both git-ignored).

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `E2E_BASE_URL` | `http://localhost:3000` | Where the browser opens the app |
| `E2E_API_URL` | `E2E_BASE_URL` | Origin used to seed data through `/api/v1` |
| `E2E_BACKEND_LOG` | unset | File with the backend's stderr; invitation links are read from it |
| `E2E_BACKEND_PYTHON` | `backend/.venv` Python, else `python` | Interpreter for `e2e/support/set_invite_token.py` |
| `E2E_BACKEND_DIR` | `<repo>/backend` | Backend checkout (to find its `.venv`) |
| `DEVDB_DIR` / `E2E_DEVDB_NAME` | `backend/var/pgdata` / `rag` | Embedded dev database used by the token helper |

### Test data and rate limits

- `e2e/global-setup.ts` checks `GET /api/v1/health`, then signs in an owner account and
  creates a fresh organization `e2e-run-<runId>` for the run. Every name starts with `e2e-`
  and every email ends in `@example.com`. Data is left in place after the run.
- Registration is limited to `REGISTER_RATE_LIMIT_PER_HOUR` (default 10 per IP per hour,
  in memory). The owner is registered once and cached in `frontend/e2e/.state/owner.json`
  for later local runs, so a run registers one account (the sign-up test), or two on the
  first run. Invited members are created by accepting invitations, which is not rate limited.
  Delete `e2e/.state/` to start over with a new owner.
- Login is limited to 10/min per email and 30/min per IP. Each browser context signs in on
  its own (refresh tokens rotate on every refresh, and replaying one from another context
  revokes the whole session family), so a busy run can reach the per-email limit; the
  helpers wait out the window and retry. To run the suite many times in a row, start the
  backend with higher limits, for example
  `REGISTER_RATE_LIMIT_PER_HOUR=1000 LOGIN_RATE_LIMIT_PER_MINUTE=1000 python scripts/dev.py`.
  The limiter is in memory, so restarting the backend also resets it.
- Browsers are signed in either through the login form or with `page.request` (which shares
  the context's cookies), so the app boots from its httpOnly refresh cookie as it does after
  a real sign-in. Seeding uses an access token through the API.

### Invitation links

Invitations are only delivered by email. With the dev `console` email backend the backend
prints `[dev email] … -> <email>: http://…/invite?token=…` to its stderr.

- If `E2E_BACKEND_LOG` points at that output, the tests read the real token from it (CI
  does this).
- Otherwise, for example against a dev server whose output is not in a file, the test picks
  a token and `e2e/support/set_invite_token.py` stores its SHA-256 hash on that one pending
  invitation in the embedded dev database. The invitation is then accepted through the
  normal API or `/invite` page. This helper is for local dev databases only.

### Chat and model cost

The chat test asks one question per run. If the backend has a language model (the dev
backend falls back to the platform's Claude key when started with `--env-file`), that is one
real, paid model call; without a model the answer is extractive. The test only asserts the
fact from the fixture, a citation chip and the source panel, so it passes either way.

### Scenarios

| Spec | Covers |
|---|---|
| `auth.spec.ts` | Register, create the first organization, Overview renders, sign out, sign in again |
| `knowledge.spec.ts` | Create a knowledge base, upload a Markdown file, wait until indexed, document row and chunk count |
| `chat.spec.ts` | Ask about the fixture, answer has the fact and a citation chip, the Sources panel opens |
| `access-control.spec.ts` | Invite a member through the API and accept; the member signs in, admin-only navigation is hidden, `/admin` shows the 403 screen, a restricted knowledge base is absent (also by direct link) |
| `security-policy.spec.ts` | In a separate organization, an owner turns on two-factor (TOTP computed with Node `crypto`, RFC 6238) and requires it; a member joining through `/invite` sees "Two-factor authentication required" |
| `connections.spec.ts` | The Connections map renders nodes; the List view works |
| `shell.spec.ts` | Theme toggle persists across reload; Ctrl+K command palette navigates; owner sees admin navigation |
| `mobile.spec.ts` | 390 px: the navigation drawer opens and navigates; key pages have no horizontal overflow |

Known issue: `mobile.spec.ts` › "the overview has no horizontal overflow at 390 px" is
marked `test.fixme`. The Overview's analytics chart props its grid column open at phone
width, so the content panel scrolls sideways. Remove `.fixme` once that is fixed.

### Continuous integration

The `e2e` job in `.github/workflows/ci.yml` sets up Python 3.12 and Node 20, installs the
backend (`pip install -e ".[dev]"`), starts `python scripts/dev.py` in the background with
its output in `e2e-logs/backend.log` (embedded PostgreSQL via pgserver), waits for
`/api/v1/health`, builds the frontend and starts the standalone server, installs Chromium
with `npx playwright install --with-deps chromium`, and runs `npm run test:e2e`. It raises
`REGISTER_RATE_LIMIT_PER_HOUR` and `LOGIN_RATE_LIMIT_PER_MINUTE`, reads invitation links
from the backend log, and uploads the HTML report, traces and server logs when it fails.
No model keys are configured there, so chat answers are extractive.

The job has not been run yet: it was written without access to a GitHub runner. One thing
to check on the first run: on Linux, pgserver serves PostgreSQL on a Unix socket
(`postgresql://postgres:@/postgres?host=<dir>`), while `backend/scripts/devdb.py` builds its
connection URLs by splitting a TCP-style URI. If `scripts/dev.py` fails to connect there,
that helper is the cause, and it affects the backend test job too.
