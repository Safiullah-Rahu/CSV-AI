"""RAG evaluation: retrieval quality, grounding and latency across configurations.

    python eval/run_eval.py                        # hashing embedder, extractive answers
    python eval/run_eval.py --embedder local       # fastembed bge-small (if installed)
    python eval/run_eval.py --llm ollama:qwen2.5:7b-instruct --limit 12
    python eval/run_eval.py --embedder local --llm openai:gpt-4.1-mini --env-file ../.env.eval
    python eval/run_eval.py --embedder local --llm anthropic:claude-sonnet-5-5 --env-file ../.env.eval

Hosted providers read OPENAI_API_KEY / ANTHROPIC_API_KEY (and, for organisation-level
Anthropic keys, ANTHROPIC_WORKSPACE_ID) from the environment or from --env-file.
Key values are never printed or written to the results.

Runs the real application in-process against an embedded PostgreSQL, uploads
the corpus through the API, asks every question through ``/chat/ask`` and
scores the ranked sources and answers. Results are written to
``eval/results/<timestamp>.json`` and summarised on stdout.

Metrics (document-level relevance): precision@k, recall@k, MRR, nDCG@k,
answer key-phrase recall ("answer relevance"), citation correctness (cited
sources ⊆ relevant documents), faithfulness (lexical support of cited
sentences, from the verifier), refusal accuracy on unanswerable questions,
retrieval and generation latency.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("EMAIL_BACKEND", "memory")
os.environ.setdefault("JOB_BACKEND", "eager")
os.environ.setdefault("SCHEDULER_ENABLED", "false")
os.environ.setdefault("JWT_SECRET", "eval-secret-eval-secret-eval-secret-01")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("LOGIN_RATE_LIMIT_PER_MINUTE", "1000")
os.environ.setdefault("SSRF_ALLOWLIST", "localhost,127.0.0.1")
os.environ.setdefault("ACCESS_TOKEN_TTL_MINUTES", "720")  # long CPU-bound LLM runs  # local Ollama for --llm runs
os.environ["STORAGE_LOCAL_ROOT"] = tempfile.mkdtemp(prefix="rag_eval_")

CONFIGS = {
    "hybrid+rerank (default)": {},
    "hybrid, no rerank": {"rerank": "none"},
    "dense only": {"mode": "dense"},
    "sparse (BM25) only": {"mode": "sparse"},
    "hybrid, no PRF expansion": {"query_expansion": "none"},
}


def dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def score_ranking(ranked_docs: list[str], relevant: set[str], k: int) -> dict[str, float]:
    top = ranked_docs[:k]
    rels = [1 if d in relevant else 0 for d in top]
    first = next((i for i, d in enumerate(ranked_docs) if d in relevant), None)
    ideal = dcg([1] * min(len(relevant), k))
    return {
        "precision": sum(rels) / k,
        "recall": len(set(top) & relevant) / len(relevant) if relevant else 0.0,
        "mrr": 1.0 / (first + 1) if first is not None else 0.0,
        "ndcg": dcg(rels) / ideal if ideal else 0.0,
        "hit": 1.0 if first is not None and first < k else 0.0,
    }


def _load_keys(path: Path) -> None:
    """Load provider credentials from a KEY=VALUE file without echoing them."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key = key.strip()
        if key in {"ANTHROPIC_API_KEY", "ANTHROPIC_WORKSPACE_ID", "OPENAI_API_KEY", "GEMINI_API_KEY"}:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="hashing", choices=["hashing", "local"])
    ap.add_argument("--llm", default=None, help="e.g. ollama:qwen2.5:7b-instruct")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0, help="evenly spaced subset incl. unanswerable questions")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--configs", default="all")
    ap.add_argument("--grep", default=None, help="only questions containing this text")
    ap.add_argument("--env-file", default=None, help="KEY=VALUE file with provider keys (only the model-provider keys are read)")
    args = ap.parse_args()
    if args.env_file:
        _load_keys(Path(args.env_file))

    import devdb

    db_name = f"rag_eval_{os.getpid()}"
    devdb.drop(db_name)
    os.environ["DATABASE_URL"] = devdb.ensure(db_name)
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
    command.upgrade(cfg, "head")

    from fastapi.testclient import TestClient

    from app.main import app
    from app.rag.grounding import is_refusal
    from eval.corpus import EXTRA_DOCS, QUESTIONS

    client = TestClient(app)
    email = f"eval-{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/v1/auth/register?transport=body", json={"email": email, "password": "Eval-Password-123", "full_name": "Eval", "organization_name": "Eval Org"})
    token = r.json()["access_token"]
    org_id = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).json()["organizations"][0]["id"]
    h = {"Authorization": f"Bearer {token}", "X-Org-Id": org_id}
    emb = {"provider": "local", "model": "BAAI/bge-small-en-v1.5"} if args.embedder == "local" else {"provider": "hashing", "dim": 768}
    kb = client.post("/api/v1/knowledge-bases", headers=h, json={"name": "Eval KB", "embedding_config": emb}).json()
    files = []
    samples = ROOT / "app" / "samples"
    for p in sorted(samples.glob("*.md")):
        files.append(("files", (p.name, p.read_bytes(), "text/markdown")))
    for name, text in EXTRA_DOCS.items():
        files.append(("files", (name, text.encode(), "text/markdown")))
    t0 = time.perf_counter()
    up = client.post(f"/api/v1/knowledge-bases/{kb['id']}/documents", headers=h, files=files).json()
    ingest_s = time.perf_counter() - t0
    assert all(x["outcome"] == "created" for x in up["results"]), up
    if args.llm:
        kind, model = args.llm.split(":", 1)
        provider: dict[str, object] = {"name": "Eval LLM", "kind": kind, "default_model": model, "is_default": True}
        if kind == "ollama":
            provider |= {"base_url": "http://localhost:11434", "settings": {"temperature": 0.1, "max_tokens": 300, "timeout_s": 900, "think": False, "repeat_penalty": 1.15}}
        else:
            key_var = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}.get(kind, f"{kind.upper()}_API_KEY")
            if not os.environ.get(key_var):
                raise SystemExit(f"--llm {kind}: set {key_var} (environment or --env-file)")
            settings: dict[str, object] = {"temperature": 0.1, "max_tokens": 4000, "timeout_s": 180}
            if kind == "anthropic":
                settings["effort"] = os.environ.get("EVAL_EFFORT", "low")
                if os.environ.get("ANTHROPIC_WORKSPACE_ID"):
                    settings["workspace_id"] = os.environ["ANTHROPIC_WORKSPACE_ID"]
            provider |= {"api_key": os.environ[key_var], "settings": settings}
        pr = client.post("/api/v1/ai/providers", headers=h, json=provider)
        assert pr.status_code == 201, f"provider rejected ({pr.status_code}): {pr.json().get('error', {}).get('message')}"
    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS
    if args.sample:
        step = max(1, len(QUESTIONS) // args.sample)
        questions = QUESTIONS[::step][: args.sample - 2] + [q for q in QUESTIONS if not q.get("answerable", True)][:2]
    if args.grep:
        questions = [q for q in questions if args.grep.lower() in str(q["q"]).lower()]
    configs = CONFIGS if args.configs == "all" else {k: v for k, v in CONFIGS.items() if k.startswith(args.configs)}
    report: dict[str, object] = {"embedder": emb, "llm": args.llm, "k": args.k, "documents": len(files), "ingest_seconds": round(ingest_s, 2), "configs": {}}
    for name, overrides in configs.items():
        rows = []
        for item in questions:
            res = client.post("/api/v1/chat/ask", headers=h, json={"question": item["q"], "kb_ids": [kb["id"]], "retrieval": {"top_k": max(args.k, 5), **overrides}}).json()
            ranked = []
            for c in res.get("citations", []):
                if c["filename"] not in ranked:
                    ranked.append(c["filename"])
            relevant = set(item["docs"])  # type: ignore[arg-type]
            answerable = item.get("answerable", True)
            usage = res.get("usage") or {}
            row: dict[str, object] = {"q": item["q"], "mode": res.get("mode"), "answerable": answerable, "retrieval_ms": res.get("timings", {}).get("retrieval_ms"), "generation_ms": res.get("timings", {}).get("generation_ms"),
                                      "answer": (res.get("content") or "")[:800], "rejected_model_answer": (res.get("trace") or {}).get("rejected_model_answer"), "tokens_in": usage.get("prompt_tokens") or 0, "tokens_out": usage.get("completion_tokens") or 0}
            if answerable:
                row.update(score_ranking(ranked, relevant, args.k))
                content = (res.get("content") or "").lower()
                phrases = [str(p).lower() for p in item["phrases"]]  # type: ignore[union-attr]
                row["answer_relevance"] = sum(1 for p in phrases if p in content) / len(phrases) if phrases else 0.0
                cited = {c["filename"] for c in res.get("citations", []) if c["cited"]}
                row["citation_correct"] = 1.0 if cited and cited <= relevant else 0.0
                row["cited_any"] = 1.0 if cited else 0.0
                row["refused"] = 1.0 if res.get("mode") == "refused" else 0.0
            else:
                row["correct_refusal"] = 1.0 if res.get("mode") == "refused" or is_refusal(res.get("content") or "") else 0.0
            msg = client.get(f"/api/v1/conversations/{res['conversation_id']}", headers=h).json()["messages"][-1]
            support = (msg.get("metadata") or {}).get("support_ratio")
            if answerable and support is not None:
                row["faithfulness"] = support
            rows.append(row)
        ans = [r for r in rows if r["answerable"]]
        un = [r for r in rows if not r["answerable"]]

        def mean(key: str, data: list[dict[str, object]]) -> float | None:
            vals = [float(r[key]) for r in data if r.get(key) is not None]  # type: ignore[arg-type]
            return round(statistics.mean(vals), 4) if vals else None

        summary = {
            f"precision@{args.k}": mean("precision", ans), f"recall@{args.k}": mean("recall", ans), "mrr": mean("mrr", ans),
            f"ndcg@{args.k}": mean("ndcg", ans), f"hit@{args.k}": mean("hit", ans), "answer_relevance": mean("answer_relevance", ans),
            "citation_correctness": mean("citation_correct", ans), "faithfulness": mean("faithfulness", ans), "false_refusals": mean("refused", ans),
            "refusal_accuracy_unanswerable": mean("correct_refusal", un), "retrieval_ms_avg": mean("retrieval_ms", rows),
            "retrieval_ms_p95": round(sorted(float(r["retrieval_ms"] or 0) for r in rows)[int(0.95 * (len(rows) - 1))], 1),
            "generation_ms_avg": mean("generation_ms", rows), "questions": len(rows),
            "tokens_in_total": sum(int(r.get("tokens_in") or 0) for r in rows), "tokens_out_total": sum(int(r.get("tokens_out") or 0) for r in rows),
        }
        report["configs"][name] = {"summary": summary, "rows": rows}  # type: ignore[index]
        print(f"\n== {name}  (embedder={emb['provider']}, llm={args.llm or 'extractive'})")
        for k, v in summary.items():
            print(f"  {k:32s} {v}")
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{emb['provider']}{'-' + args.llm.replace(':', '_') if args.llm else ''}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {path}")
    client.close()
    devdb.drop(db_name)


if __name__ == "__main__":
    _ = io
    main()
