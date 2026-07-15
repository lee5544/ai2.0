#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONDA_ENV="${FORVIA_CONDA_ENV:-fault}"

run_python() { conda run -n "$CONDA_ENV" --no-capture-output python "$@"; }

run_pip() { conda run -n "$CONDA_ENV" --no-capture-output python -m pip "$@"; }

if ! command -v conda >/dev/null 2>&1; then
  echo "错误：未找到 conda。请先安装 conda。"
  exit 1
fi
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "错误：conda 环境不存在: $CONDA_ENV"
  echo "可通过 FORVIA_CONDA_ENV 指定环境名。"
  exit 1
fi
echo "使用 conda 环境: $CONDA_ENV"

run_pip install --upgrade pip
run_pip install -r forvia_label_v2/requirements.txt -r forvia_train_v2/requirements.txt pyinstaller

run_python -c 'import fastapi, uvicorn, numpy, pandas, scipy, librosa, soundfile, nptdms, zstandard, plotly, pywt, sklearn, xgboost, lightgbm, matplotlib; print("Forvia AI2.0 依赖检查通过")'
echo "依赖安装完成。启动：./start_forvia_console.sh"
