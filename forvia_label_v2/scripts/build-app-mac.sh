#!/usr/bin/env bash
# 在你的 Mac 上跑一次，产出一个可双击的 App：dist/Forvia标注v2.app
# 接收者无需装 Python / Docker，双击即用，原生选择本地/外接盘文件夹。
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"   # forvia_label_v2/
REPO="$(cd "$HERE/.." && pwd)"             # 仓库根 AI-2.0/
cd "$REPO"

echo "→ [1/3] 准备 Python 环境与依赖（建议在干净的 venv 里跑）"
python3 -m pip install --upgrade pip >/dev/null
python3 -m pip install -r "$HERE/requirements.txt"
python3 -m pip install pyinstaller

echo "→ [2/3] 打包（首次较慢，会收集 librosa/numba 等二进制）"
rm -rf "$REPO/build/Forvia标注v2" "$REPO/dist/Forvia标注v2" "$REPO/dist/Forvia标注v2.app"
pyinstaller --clean --noconfirm "$HERE/forvia_label_v2.spec"

echo "→ [3/3] 完成"
echo "    App:      $REPO/dist/Forvia标注v2.app   ← 双击运行 / 发给别人"
echo "    去隔离:   xattr -dr com.apple.quarantine \"$REPO/dist/Forvia标注v2.app\""
echo "    打包分发: 把 .app 压成 zip 发送即可"
