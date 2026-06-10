#!/bin/bash
set -euo pipefail

DB_WAIT_FOR_DATABASE="${DB_WAIT_FOR_DATABASE:-true}"
MAX_RETRIES="${DB_READY_MAX_RETRIES:-30}"
SLEEP_SECONDS="${DB_READY_SLEEP_SECONDS:-2}"

DATABASE_URL_NORMALIZED="$(printf '%s' "${DATABASE_URL:-}" | tr '[:upper:]' '[:lower:]')"
IS_SQLITE=false
if [ -z "${DATABASE_URL_NORMALIZED}" ]; then
  IS_SQLITE=true
elif [[ "${DATABASE_URL_NORMALIZED}" == sqlite:* ]]; then
  IS_SQLITE=true
fi

if [ "${IS_SQLITE}" = "true" ]; then
  echo "❌ Invalid DATABASE_URL for Docker backend."
  echo "   This project requires PostgreSQL + pgvector for full migrations/features."
  echo "   Configure DATABASE_URL like:"
  echo "   postgres://postgres:postgres@postgres:5432/automated_sdr"
  echo "   SQLite is only for non-Docker lightweight local/testing."
  exit 1
fi

if [ "${DB_WAIT_FOR_DATABASE}" = "true" ]; then
  DB_HOST="${PGHOST:-${DB_HOST:-postgres}}"
  DB_PORT="${PGPORT:-${DB_PORT:-5432}}"
  DB_USER="${PGUSER:-${POSTGRES_USER:-${DB_USER:-postgres}}}"

  echo "⏳ Waiting for PostgreSQL at ${DB_HOST}:${DB_PORT}..."

  wait_for_postgres() {
    if command -v pg_isready > /dev/null 2>&1; then
      pg_isready -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -q
    else
      python - <<EOF
import socket, sys
try:
    s = socket.create_connection(("${DB_HOST}", int("${DB_PORT}")), timeout=5)
    s.close()
    sys.exit(0)
except OSError:
    sys.exit(1)
EOF
    fi
  }

  RETRY=0
  until wait_for_postgres; do
    RETRY=$((RETRY + 1))
    if [ "${RETRY}" -ge "${MAX_RETRIES}" ]; then
      echo "❌ PostgreSQL did not become ready after ${MAX_RETRIES} attempts. Aborting."
      exit 1
    fi
    echo "   PostgreSQL is not ready yet — attempt ${RETRY}/${MAX_RETRIES}, sleeping ${SLEEP_SECONDS} s"
    sleep "${SLEEP_SECONDS}"
  done

  echo "✅ PostgreSQL is ready."
fi

echo "🔄 Running database migrations..."
alembic upgrade head

echo "🌱 Seeding default database data..."
python -m sdr.core.seed

WORKERS="${WORKERS:-3}"
UVICORN_HOST="${UVICORN_HOST:-0.0.0.0}"
UVICORN_PORT="${UVICORN_PORT:-8000}"
UVICORN_LOG_LEVEL="${UVICORN_LOG_LEVEL:-info}"
UVICORN_PROXY_HEADERS="${UVICORN_PROXY_HEADERS:-true}"
UVICORN_FORWARDED_ALLOW_IPS="${UVICORN_FORWARDED_ALLOW_IPS:-127.0.0.1}"

echo "🚀 Starting Uvicorn with ${WORKERS} worker(s)..."
UVICORN_ARGS=(
  --host "${UVICORN_HOST}"
  --port "${UVICORN_PORT}"
  --workers "${WORKERS}"
  --forwarded-allow-ips "${UVICORN_FORWARDED_ALLOW_IPS}"
  --log-level "${UVICORN_LOG_LEVEL}"
)
if [ "${UVICORN_PROXY_HEADERS}" = "true" ]; then
  UVICORN_ARGS+=(--proxy-headers)
fi

exec uvicorn sdr.main:app \
    "${UVICORN_ARGS[@]}"
