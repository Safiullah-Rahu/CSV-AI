#!/bin/sh
# Creates the NON-superuser application role that owns the platform database.
# Superusers bypass Row-Level Security, so the application must never use one.
set -eu
psql -v ON_ERROR_STOP=1 --username postgres <<SQL
DO \$\$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${APP_DB_USER}') THEN
    CREATE ROLE ${APP_DB_USER} LOGIN PASSWORD '${APP_DB_PASSWORD}' NOSUPERUSER NOBYPASSRLS NOCREATEROLE NOCREATEDB;
  END IF;
END \$\$;
SELECT 'CREATE DATABASE ${APP_DB_NAME} OWNER ${APP_DB_USER}' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${APP_DB_NAME}')\gexec
\c ${APP_DB_NAME}
CREATE EXTENSION IF NOT EXISTS vector;
SQL
