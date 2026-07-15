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

if [[ -n "${FORVIA_PYTHON:-}" ]]; then
  PYTHON="$FORVIA_PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON="$CONDA_PREFIX/bin/python"
else
  PYTHON="python3"
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "错误：未找到当前 Python 环境: $PYTHON" >&2
  exit 1
fi
if ! "$PYTHON" -c 'import uvicorn' >/dev/null 2>&1; then
  echo "错误：当前 Python 环境没有安装 uvicorn: $($PYTHON -c 'import sys; print(sys.executable)' 2>/dev/null || echo "$PYTHON")" >&2
  echo "请先激活自己的 Conda 环境，然后执行: ./install_forvia_dependencies.sh" >&2
  exit 1
fi

"$PYTHON" -m uvicorn forvia_label_v2.backend.main:app --reload --port 8012 \
  > "$LOG_DIR/label_v2.log" 2>&1 &
echo $! > "$LOG_DIR/label_v2.pid"

"$PYTHON" -m uvicorn forvia_train_v2.backend.main:app --reload --port 8001 \
  > "$LOG_DIR/train_v2.log" 2>&1 &
echo $! > "$LOG_DIR/train_v2.pid"

sleep 2
for log in label_v2 train_v2; do
  pid_file="$LOG_DIR/$log.pid"
  if [[ ! -s "$pid_file" ]] || ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "错误：$log 服务启动失败，日志如下：" >&2
    tail -40 "$LOG_DIR/$log.log" >&2 || true
    exit 1
  fi
done

CONSOLE_URL="http://127.0.0.1:8012/console"
if command -v open >/dev/null 2>&1; then
  open "$CONSOLE_URL"
else
  echo "打开入口: $CONSOLE_URL"
fi

echo "Forvia AI2.0 Console 已启动"
echo "Label v2: http://127.0.0.1:8012/"
echo "Train v2: http://127.0.0.1:8001/"
echo "日志目录: $LOG_DIR"
echo "按 Ctrl+C 关闭全部服务。"

while true; do
  sleep 3600
done
