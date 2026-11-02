# Knowledge Intelligence — backend

FastAPI service for the Knowledge Intelligence enterprise RAG platform: multi-tenant
API, ingestion, retrieval and answer generation, connectors, agents and background jobs,
on PostgreSQL 16 + pgvector.

See the repository [README](../README.md) for an overview, `ARCHITECTURE.md` for the
design, `DEPLOYMENT.md` for running it, and `docs/TESTING.md` for the test suites.

```bash
pip install -e ".[dev]"
python scripts/dev.py          # embedded PostgreSQL + migrations + API on :8000
pytest -q
```
