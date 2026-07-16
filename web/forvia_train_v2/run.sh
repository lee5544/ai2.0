#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"

HOST="${FORVIA_TRAIN_V2_HOST:-127.0.0.1}"
PORT="${FORVIA_TRAIN_V2_PORT:-8001}"

if command -v uvicorn >/dev/null 2>&1; then
  exec uvicorn forvia_train_v2.backend.main:app --reload --host "$HOST" --port "$PORT"
fi

if python -c "import uvicorn" >/dev/null 2>&1; then
  exec python -m uvicorn forvia_train_v2.backend.main:app --reload --host "$HOST" --port "$PORT"
fi

exec conda run -n fault --no-capture-output python -m uvicorn forvia_train_v2.backend.main:app --reload --host "$HOST" --port "$PORT"
