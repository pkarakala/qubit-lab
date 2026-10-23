#!/usr/bin/env bash
# API on :8000, explorer on :5173.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --all-groups -q
# macOS can flag the editable-install .pth file as hidden, and Python then skips it, so
# `import qubitlab` fails. Clear the flag.
if [ "$(uname)" = Darwin ]; then chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null || true; fi
uv run uvicorn qubitlab.api:app --port 8000 &
API=$!
trap 'kill $API' EXIT
cd web && npm run dev
