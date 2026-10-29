#!/usr/bin/env bash
# API on :8000, explorer on :5173.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --all-groups -q
# macOS can flag the editable-install .pth file as hidden, and Python then skips it, so
# `import qubitlab` fails. Clear the flag.
if [ "$(uname)" = Darwin ]; then chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null || true; fi
QUBITLAB_WARM="${QUBITLAB_WARM:-1}" uv run --no-sync uvicorn qubitlab.api:app --port 8000 &  # synced above
API=$!
trap 'kill $API' EXIT
# The API takes 10 to 20 s to import JAX and scqubits. Start the explorer once it answers, so
# the first page load does not hit a 502.
echo "Starting the API..."
for _ in $(seq 1 120); do
  curl -sf localhost:8000/api/health >/dev/null && break
  kill -0 $API 2>/dev/null || { echo "The API exited; see the error above."; exit 1; }
  sleep 1
done
echo "API ready."
cd web && npm run dev
