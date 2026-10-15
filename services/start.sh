#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_FILE="$REPO_ROOT/services/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # This file is expected to contain shell-compatible KEY=value assignments.
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo "ERROR: $ENV_FILE not found. Copy services/.env.example to services/.env and configure credentials." >&2
  exit 2
fi

if [[ -z "${SAGE_PLAYGROUND_USERNAME:-}" || -z "${SAGE_PLAYGROUND_PASSWORD:-}" ]]; then
  echo "ERROR: set SAGE_PLAYGROUND_USERNAME and SAGE_PLAYGROUND_PASSWORD in services/.env" >&2
  exit 2
fi
if [[ "${SAGE_PLAYGROUND_PASSWORD}" == *REPLACE* ]]; then
  echo "ERROR: replace the placeholder SAGE_PLAYGROUND_PASSWORD in services/.env" >&2
  exit 2
fi
SECRET_KEY_VALUE="${SAGE_PLAYGROUND_SECRET_KEY:-}"
if [[ "$SECRET_KEY_VALUE" == *REPLACE* || ${#SECRET_KEY_VALUE} -lt 32 ]]; then
  echo "ERROR: SAGE_PLAYGROUND_SECRET_KEY must have at least 32 characters" >&2
  exit 2
fi

HOST="${SAGE_PLAYGROUND_HOST:-0.0.0.0}"
PORT="${SAGE_PLAYGROUND_PORT:-8765}"

# One worker is intentional: the runtime owns one active model/GPU lock in-process.
exec python -m uvicorn services.app:app --host "$HOST" --port "$PORT" --workers 1
