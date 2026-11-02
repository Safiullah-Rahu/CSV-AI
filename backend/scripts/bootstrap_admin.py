"""Create (or promote) the first platform administrator.

    BOOTSTRAP_ADMIN_EMAIL=ops@example.com BOOTSTRAP_ADMIN_PASSWORD='…' python scripts/bootstrap_admin.py

Idempotent. Platform administrators manage tenants and system health; they do
not see customer content unless an organisation owner grants time-boxed
support access.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.core.security import hash_password, validate_password_strength  # noqa: E402
from app.db.session import system_session  # noqa: E402
from app.models import User  # noqa: E402
from app.services.activity import audit  # noqa: E402


def main() -> int:
    email = (os.environ.get("BOOTSTRAP_ADMIN_EMAIL") or "").strip().lower()
    password = os.environ.get("BOOTSTRAP_ADMIN_PASSWORD") or ""
    if not email:
        print("BOOTSTRAP_ADMIN_EMAIL is not set; nothing to do.")
        return 0
    with system_session() as db:
        user = db.scalar(select(User).where(User.email == email))
        if user is None:
            problems = validate_password_strength(password)
            if problems:
                print("Refusing to create admin: " + " ".join(problems))
                return 1
            user = User(email=email, password_hash=hash_password(password), full_name="Platform Administrator", is_platform_admin=True)
            db.add(user)
            db.flush()
            audit(db, "platform.bootstrap_admin", actor_user_id=user.id, actor_email=email)
            print(f"Created platform administrator {email}.")
        elif not user.is_platform_admin:
            user.is_platform_admin = True
            audit(db, "platform.bootstrap_admin", actor_user_id=user.id, actor_email=email, meta={"promoted": True})
            print(f"Promoted {email} to platform administrator.")
        else:
            print(f"{email} is already a platform administrator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
