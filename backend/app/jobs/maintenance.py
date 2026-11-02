"""Maintenance jobs: organisation purge."""

from __future__ import annotations

import uuid

from sqlalchemy import delete

from app.db.session import system_session
from app.jobs.queue import JobContext, handler
from app.models import Organization
from app.storage import get_storage


@handler("purge_organization")
def purge_organization(ctx: JobContext) -> dict:
    """Delete every row (FK cascades) and every stored object of a deactivated org."""
    org_id = uuid.UUID(str(ctx.org_id))
    with system_session() as db:
        org = db.get(Organization, org_id)
        if org is None:
            return {"purged": False}
        if org.status != "deactivated":
            return {"purged": False, "reason": f"organization is {org.status}"}
    removed = get_storage().delete_prefix(str(org_id))
    with system_session() as db:
        db.execute(delete(Organization).where(Organization.id == org_id))
    return {"purged": True, "objects_removed": removed}
