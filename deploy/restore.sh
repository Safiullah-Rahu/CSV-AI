#!/usr/bin/env bash
# Restore a database dump produced by backup.sh (stops api/worker during restore).
#   ./deploy/restore.sh backups/db-20261102T020000Z.dump
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
DUMP="$1"
docker compose stop api worker
docker compose exec -T postgres pg_restore -U postgres --clean --if-exists --no-owner --role="${APP_DB_USER:-rag_app}" -d "${APP_DB_NAME:-rag}" < "$DUMP"
docker compose start api worker
echo "Database restored from $DUMP."
echo "To restore files: docker compose stop minio && gunzip -c backups/storage-<ts>.tar.gz | docker compose run --rm -T --entrypoint tar minio -C /data -xf - && docker compose start minio"
