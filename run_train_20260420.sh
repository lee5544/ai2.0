#!/usr/bin/env bash
# 在本机 fault 环境里跑 v4 训练
# 用法：
#   cd /path/to/2025-forvia-异常检测/AI-2.0
#   bash run_train_20260420.sh
set -euo pipefail

# 1) 激活 conda 环境
#    兼容 macOS (miniconda3 / anaconda3 在 $HOME 下) 与通用路径
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
if [[ -z "$CONDA_BASE" ]]; then
  for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/miniconda3" "/opt/anaconda3"; do
    if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
      CONDA_BASE="$candidate"
      break
    fi
  done
fi
[[ -z "$CONDA_BASE" ]] && { echo "[ERR] 找不到 conda，请先装好再跑"; exit 1; }

# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate fault

# 2) 跑训练
CFG="cfg/epump2_general_20260420.yaml"
LOG_DIR="results/epump2_general_20260420_xgb"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_${TS}.log"

echo "[info] python=$(python -V 2>&1)"
echo "[info] config=$CFG"
echo "[info] log=$LOG_FILE"

# 关键包快速自检
python -c "import xgboost, nptdms, pywt, librosa, scipy, pandas, yaml; print('[info] deps OK')"

# 3) 训练（tee 一份到 log，终端也能看到进度）
python -u -m ml.train --config "$CFG" --step train 2>&1 | tee "$LOG_FILE"

echo "[done] 结果目录: $LOG_DIR"
echo "[done] 日志文件: $LOG_FILE"
