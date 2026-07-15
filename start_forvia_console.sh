#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/tmp/forvia_console_logs"
mkdir -p "$LOG_DIR"

cleanup() {
  echo
  echo "正在关闭 Forvia AI2.0 Console..."
  for pid_file in "$LOG_DIR/label_v2.pid" "$LOG_DIR/train_v2.pid"; do
    if [[ -f "$pid_file" ]]; then
      pid="$(cat "$pid_file" 2>/dev/null || true)"
      [[ -n "${pid:-}" ]] && kill "$pid" 2>/dev/null || true
      rm -f "$pid_file"
    fi
  done
  for port in 8012 8001; do
    for pid in $(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true); do
      kill "$pid" 2>/dev/null || true
    done
  done
  echo "已关闭。"
}

trap cleanup INT TERM EXIT

for port in 8012 8001; do
  for pid in $(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true); do
    kill "$pid" 2>/dev/null || true
  done
done

sleep 1

PYTHON="${FORVIA_PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "错误：未找到当前 Python 环境: $PYTHON" >&2
  exit 1
fi

"$PYTHON" -m uvicorn forvia_label_v2.backend.main:app --reload --port 8012 \
  > "$LOG_DIR/label_v2.log" 2>&1 &
echo $! > "$LOG_DIR/label_v2.pid"

"$PYTHON" -m uvicorn forvia_train_v2.backend.main:app --reload --port 8001 \
  > "$LOG_DIR/train_v2.log" 2>&1 &
echo $! > "$LOG_DIR/train_v2.pid"

HTML="$ROOT/forvia_console_preview.html"
if command -v open >/dev/null 2>&1; then
  open "$HTML"
else
  echo "打开入口文件: $HTML"
fi

echo "Forvia AI2.0 Console 已启动"
echo "Label v2: http://127.0.0.1:8012/"
echo "Train v2: http://127.0.0.1:8001/"
echo "日志目录: $LOG_DIR"
echo "按 Ctrl+C 关闭全部服务。"

while true; do
  sleep 3600
done
