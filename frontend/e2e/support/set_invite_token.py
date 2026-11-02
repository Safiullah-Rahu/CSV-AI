"""Test-only helper for the end-to-end suite: give one pending invitation a known token.

The development backend uses the ``console`` email backend, which prints
invitation links to the server's own stderr. When the suite cannot read that
output (for example, the dev server was started by another tool and does not
log to a file), the tests choose a token themselves and this script stores its
SHA-256 hash on the invitation they just created, in the embedded development
database (pgserver, ``backend/var/pgdata``). The invitation is then accepted
through the normal API or /invite page.

    python set_invite_token.py <invitation-id> <sha256-hex-of-token>

Environment: ``DEVDB_DIR`` (default ``<repo>/backend/var/pgdata``),
``E2E_DEVDB_NAME`` (default ``rag``). Never point this at a real deployment.
"""

from __future__ import annotations

import os
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    invitation_id = str(uuid.UUID(sys.argv[1]))
    token_hash = sys.argv[2].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", token_hash):
        print("token hash must be 64 hex characters", file=sys.stderr)
        return 2

    import pgserver
    import psycopg
    from psycopg.conninfo import make_conninfo

    data_dir = Path(os.environ.get("DEVDB_DIR") or ROOT / "backend" / "var" / "pgdata")
    if not (data_dir / "PG_VERSION").exists():
        print(f"no embedded development database at {data_dir}", file=sys.stderr)
        return 1
    # cleanup_mode=None: attach to the running dev server and never stop it.
    server = pgserver.get_server(str(data_dir), cleanup_mode=None)
    dsn = make_conninfo(server.get_uri(), dbname=os.environ.get("E2E_DEVDB_NAME", "rag"))
    with psycopg.connect(dsn, autocommit=True) as conn:
        row = conn.execute(
            "UPDATE invitations SET token_hash = %s WHERE id = %s AND status = 'pending' RETURNING id",
            (token_hash, invitation_id),
        ).fetchone()
    if row is None:
        print(f"invitation {invitation_id} not found or not pending", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
