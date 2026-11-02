#!/usr/bin/env bash
# Backup: PostgreSQL (custom-format dump) + object storage (MinIO mirror).
#   ./deploy/backup.sh [backup-dir]      # default ./backups
# Schedule with cron, e.g.: 0 2 * * * cd /opt/knowledge-intelligence && ./deploy/backup.sh >> backups/backup.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
DEST="${1:-./backups}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$DEST"
docker compose exec -T postgres pg_dump -U postgres -Fc "${APP_DB_NAME:-rag}" > "$DEST/db-$TS.dump"
# Object storage: archive the MinIO data directory (restore into the same MinIO release).
docker compose exec -T minio tar -C /data -cf - . | gzip > "$DEST/storage-$TS.tar.gz"
# Retention: keep the most recent N days of backups.
find "$DEST" -maxdepth 1 \( -name 'db-*.dump' -o -name 'storage-*.tar.gz' \) -mtime +"${BACKUP_RETENTION_DAYS:-14}" -delete
echo "Backup written to $DEST (db-$TS.dump, storage-$TS.tar.gz)"
