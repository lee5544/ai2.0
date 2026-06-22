#!/usr/bin/env bash
# Forvia 标注 v2 · Mac 一键运行
# 作用：把复用的 v1 子集（data_manager / sample_view / cfg）暂存到 vendor/，
#       然后用 Docker 构建并启动。镜像自包含，运行后访问 http://localhost:8000
#
# 用法：
#   1) 编辑 .env，设置 FORVIA_DATA_DIR 指向你的数据所在目录（会挂到容器内 /data）
#   2) ./scripts/run-mac.sh            # 构建并前台启动（Ctrl+C 停止）
#      ./scripts/run-mac.sh -d         # 后台启动
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"     # forvia_label_v2/
REPO="$(cd "$HERE/.." && pwd)"               # 仓库根 AI-2.0/

echo "→ [1/3] 暂存 v1 依赖到 vendor/ ..."
rm -rf "$HERE/vendor"
mkdir -p "$HERE/vendor"
for d in data_manager sample_view cfg; do
  if [ ! -d "$REPO/$d" ]; then
    echo "  ✗ 找不到 $REPO/$d ，请确认在仓库内运行此脚本"; exit 1
  fi
  rsync -a --exclude '__pycache__' --exclude '*.pyc' "$REPO/$d/" "$HERE/vendor/$d/"
  echo "  ✓ vendor/$d"
done

mkdir -p "$HERE/_data"

if [ ! -f "$HERE/.env" ] && [ -f "$HERE/.env.example" ]; then
  cp "$HERE/.env.example" "$HERE/.env"
  echo "→ 已生成 .env（默认数据目录 ./data）。如需挂载外接盘，请编辑 .env 里的 FORVIA_DATA_DIR。"
fi

echo "→ [2/3] 构建镜像 ..."
cd "$HERE"
echo "→ [3/3] 启动（首次构建较慢，需联网装依赖）..."
docker compose up --build "$@"
