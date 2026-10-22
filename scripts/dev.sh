#!/usr/bin/env bash
# API on :8000, explorer on :5173.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run uvicorn catlab.api:app --port 8000 &
API=$!
trap 'kill $API' EXIT
cd web && npm run dev
