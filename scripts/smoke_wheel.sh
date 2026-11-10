#!/usr/bin/env bash
# Install the built wheel in a clean venv and check that `qubitlab serve` answers
# with both the explorer and the API on one port, using only what the wheel ships.
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
# Package data, not just code: the bib, the formula registry and one solve.
curl -sf "localhost:$PORT/api/references" | grep -q '"koch2007"'
curl -sf "localhost:$PORT/api/formulas" >/dev/null
curl -sf -X POST -H "Content-Type: application/json" \
  -d '{"t_final": 1, "n_frames": 3, "grid": 21}' "localhost:$PORT/api/cat/evolve" >/dev/null
echo "wheel serves the explorer, its citations and a solve on :$PORT"
