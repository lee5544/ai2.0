#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PYTHON="${FORVIA_PYTHON:-python3}"

run_python() { "$PYTHON" "$@"; }

run_pip() { "$PYTHON" -m pip "$@"; }

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "错误：未找到当前 Python 环境: $PYTHON"
  exit 1
fi
echo "使用当前 Python 环境: $($PYTHON -c 'import sys; print(sys.executable)')"

run_pip install --upgrade pip
run_pip install -r forvia_label_v2/requirements.txt -r forvia_train_v2/requirements.txt pyinstaller

run_python -c 'import fastapi, uvicorn, numpy, pandas, scipy, librosa, soundfile, nptdms, zstandard, plotly, pywt, sklearn, xgboost, lightgbm, matplotlib; print("Forvia AI2.0 依赖检查通过")'
echo "依赖安装完成。启动：./start_forvia_console.sh"
