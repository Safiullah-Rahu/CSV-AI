"""Seed a local DEVELOPMENT environment with demo accounts and content.

    python scripts/seed_demo.py            # against the running dev API (http://127.0.0.1:8000)
    python scripts/seed_demo.py --ollama   # also register a local Ollama provider

Creates (idempotently):
  * demo.admin@example.com   / Demo-Admin-2026!   — owner of "Northwind Demo"
  * demo.member@example.com  / Demo-Member-2026!  — standard member, granted only the HR knowledge base
  * the bundled sample knowledge base, an "HR Policies" knowledge base and a restricted "Finance" one
  * with --ollama only: a local Ollama provider (when Ollama answers on localhost:11434)

Answers come from the platform model fallback when the dev server was started with
provider keys (``scripts/dev.py --env-file ...``), otherwise from extractive mode.

These are test credentials for local development only. Never run this against production.
"""

from __future__ import annotations

import io
import os
import sys

import httpx

API = os.environ.get("API_URL", "http://127.0.0.1:8000") + "/api/v1"
ADMIN = ("demo.admin@example.com", "Demo-Admin-2026!", "Avery Admin")
MEMBER = ("demo.member@example.com", "Demo-Member-2026!", "Morgan Member")

HR = b"""# HR Policies

## Parental leave
Primary caregivers receive 20 weeks of fully paid parental leave. Secondary caregivers receive 6 weeks.

## Probation
The probation period for new employees is 3 months, with a review meeting at week 10.
"""
FINANCE = b"""# Finance Procedures (restricted)

## Budget cycle
Departmental budgets are frozen on 15 November. Re-forecasts happen quarterly.

## Corporate cards
Corporate cards have a monthly limit of 3,000 EUR. Cash withdrawals are not permitted.
"""


def token_for(c: httpx.Client, email: str, password: str, name: str, org: str | None) -> str:
    r = c.post(f"{API}/auth/login?transport=body", json={"email": email, "password": password})
    if r.status_code == 200:
        return r.json()["access_token"]
    r = c.post(f"{API}/auth/register?transport=body", json={"email": email, "password": password, "full_name": name, "organization_name": org})
    r.raise_for_status()
    return r.json()["access_token"]


def main() -> None:
    with httpx.Client(timeout=120) as c:
        admin = token_for(c, *ADMIN, org="Northwind Demo")
        me = c.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {admin}"}).json()
        org_id = next(o["id"] for o in me["organizations"] if o["name"] == "Northwind Demo")
        h = {"Authorization": f"Bearer {admin}", "X-Org-Id": org_id}
        c.post(f"{API}/onboarding/sample-kb", headers=h)
        kbs = {k["name"]: k for k in c.get(f"{API}/knowledge-bases", headers=h).json()}
        for name, body, desc in (("HR Policies", HR, "Leave, probation and people policies"), ("Finance", FINANCE, "Budget and card rules — finance team only")):
            if name not in kbs:
                kbs[name] = c.post(f"{API}/knowledge-bases", headers=h, json={"name": name, "description": desc, "visibility": "restricted"}).json()
                c.post(f"{API}/knowledge-bases/{kbs[name]['id']}/documents", headers=h, files=[("files", (f"{name.lower().replace(' ', '-')}.md", io.BytesIO(body), "text/markdown"))])
        members = {m["email"]: m for m in c.get(f"{API}/members", headers=h).json()["items"]}
        if MEMBER[0] not in members:
            roles = {r["key"]: r["id"] for r in c.get(f"{API}/roles", headers=h).json()}
            c.post(f"{API}/invitations", headers=h, json={"email": MEMBER[0], "role_id": roles["member"]})
            print("Invitation created for the member account. With EMAIL_BACKEND=console the accept link is printed in the API log;")
            print("open it in the browser and use the member password above.")
        else:
            uid = members[MEMBER[0]]["user_id"]
            grants = c.get(f"{API}/grants", headers=h, params={"resource_type": "knowledge_base", "resource_id": kbs["HR Policies"]["id"]}).json()
            if not any(g["subject_id"] == uid for g in grants):
                c.post(f"{API}/grants", headers=h, json={"resource_type": "knowledge_base", "resource_id": kbs["HR Policies"]["id"], "subject_type": "user", "subject_id": uid, "privilege": "read"})
        ok = False
        if "--ollama" in sys.argv[1:]:  # opt-in: CPU inference is slow and small local models are unreliable
            try:
                ok = httpx.get("http://localhost:11434/api/tags", timeout=3).status_code == 200
            except httpx.HTTPError:
                ok = False
        providers = c.get(f"{API}/ai/providers", headers=h).json()
        if ok and not providers:
            c.post(f"{API}/ai/providers", headers=h, json={"name": "Local Ollama", "kind": "ollama", "base_url": "http://localhost:11434", "default_model": "qwen2.5:7b-instruct", "is_default": True, "settings": {"temperature": 0.1, "max_tokens": 600, "timeout_s": 300, "think": False}})
            print("Configured local Ollama provider (qwen2.5:7b-instruct).")
        print(f"Seeded organization {org_id}. Sign in as {ADMIN[0]} (see this script for the dev password).")


if __name__ == "__main__":
    sys.exit(main())
