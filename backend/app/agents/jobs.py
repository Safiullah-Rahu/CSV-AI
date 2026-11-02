"""Background continuation of agent runs that paused for a tool approval."""

from __future__ import annotations

import uuid
from typing import Any

from app.agents.runner import resume_agent, run_payload
from app.jobs.queue import JobContext, handler
from app.llm.registry import resolve_llm
from app.models import Agent, Approval, Notification, Organization, User
from app.rbac.authorizer import Authorizer, Principal
from app.services.activity import emit
from app.utils.errors import CSVAIError


@handler("resume_agent_run")
def resume_agent_run(ctx: JobContext) -> dict[str, Any]:
    approval_id = uuid.UUID(ctx.payload["approval_id"])
    with ctx.session() as db:
        a = db.get(Approval, approval_id)
        if a is None or a.kind != "agent_tool" or a.status != "approved" or "resume" not in (a.payload or {}):
            return {"skipped": "approval missing, not approved, or not resumable"}
        agent = db.get(Agent, a.resource_id) if a.resource_id else None
        org = db.get(Organization, ctx.org_id)
        requester = db.get(User, a.requested_by)
        result: dict[str, Any]
        if agent is None or not agent.enabled or org is None or requester is None:
            result = {"status": "error", "answer": "The agent or the requesting user no longer exists, so the run could not continue.", "steps": [], "citations": []}
        else:
            # Same rule as approved SQL: the run continues with the REQUESTER's permissions.
            authz = Authorizer(db, Principal(user_id=requester.id, email=requester.email, org_id=org.id)).load()
            try:
                llm = resolve_llm(db, org, provider_id=agent.provider_id, model=agent.model)
                if llm is None:
                    result = {"status": "error", "answer": "No language model is configured, so the approved run could not continue.", "steps": [], "citations": []}
                else:
                    result = run_payload(resume_agent(db, authz, org, agent, llm, a.payload))
            except CSVAIError as exc:
                result = {"status": "error", "answer": f"The run could not continue: {exc.message}", "steps": a.payload["resume"].get("steps", []), "citations": []}
        a.payload = {**a.payload, "result": result}
        finished = result["status"] == "complete"
        db.add(Notification(org_id=ctx.org_id, user_id=a.requested_by, kind="agent_resumed" if finished else "agent_failed",
                            title=(f"Approved and finished: {a.summary}" if finished else f"Approved, but the run did not finish: {a.summary}")[:300],
                            body=(result.get("answer") or "")[:500], severity="info" if finished else "warning", link="/agents"))
        emit(db, ctx.org_id, "agent.resumed", resource_type="agent", resource_id=a.resource_id, actor_user_id=a.requested_by,
             severity="success" if finished else "warning", message=f"{a.summary}: {result['status']}", payload={"approval_id": str(a.id)})
    return {"status": result["status"]}
