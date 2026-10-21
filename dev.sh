#!/usr/bin/env bash
# TENNIS ALPHA: start the Python API and the web app together. Both listen on this computer only.
#   ./dev.sh          then open http://localhost:3100      (API docs: http://127.0.0.1:8000/docs)
#   Ctrl+C            stops both.
set -euo pipefail
cd "$(dirname "$0")"

[ -x .venv/bin/python ] || { echo "Missing .venv. Create it first: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
[ -d frontend/node_modules ] || npm --prefix frontend install
[ -f artifacts/dashboard.pkl ] || .venv/bin/python -m src.build_artifacts   # trains the frozen models once (about 30 seconds)
[ -f artifacts/h2h.pkl ] || .venv/bin/python -m src.h2h                      # head-to-head index (a second)
[ -f artifacts/v2.pkl ] || .venv/bin/python -m src.v2.build_artifact          # experimental V2 model (about 25 seconds)

# Each service runs in its own process group (set -m), so stopping this script stops everything it started,
# including the helper processes that npm and Next.js spawn.
set -m
.venv/bin/python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 &
API_PID=$!
for _ in $(seq 1 60); do curl -sf http://127.0.0.1:8000/health > /dev/null && break; sleep 0.5; done   # wait for the model to load
npm --prefix frontend run dev -- --port 3100 --hostname localhost &
WEB_PID=$!
trap 'trap - EXIT INT TERM; kill -- -"$API_PID" -"$WEB_PID" 2>/dev/null || true' EXIT INT TERM

# Run until one of the two stops (or Ctrl+C); the trap then stops the other. (Portable: macOS bash 3.2 has no `wait -n`.)
while kill -0 "$API_PID" 2>/dev/null && kill -0 "$WEB_PID" 2>/dev/null; do sleep 1; done
