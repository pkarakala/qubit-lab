#!/usr/bin/env bash
# Install the built wheel in a clean venv and check that `qubitlab serve` answers
# with both the explorer and the API on one port.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=$(mktemp -d)/venv
PORT=${PORT:-8917}
uv venv -q "$VENV"
uv pip install -q --python "$VENV/bin/python" dist/*.whl
"$VENV/bin/qubitlab" serve --no-browser --port "$PORT" &
PID=$!
trap 'kill $PID' EXIT
for _ in $(seq 1 180); do curl -sf "localhost:$PORT/api/health" >/dev/null && break; sleep 1; done
curl -sf "localhost:$PORT/api/health"
curl -sf "localhost:$PORT/" | grep -q "<title>Qubit Lab</title>"
echo "wheel serves the explorer and the API on :$PORT"
