# Forvia 标注 v2 · 自包含镜像（Mac/Windows 用 Docker Desktop 运行）
#
# 构建上下文 = forvia_label_v2/。镜像里自带复用的 v1 子集（data_manager / sample_view / cfg），
# 这些由 scripts/run-mac.sh（或 run-win.ps1）在构建前暂存到 vendor/，因此镜像独立、不依赖宿主仓库。
FROM python:3.11-slim

# soundfile / librosa 运行所需系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Python 依赖（单独一层，便于缓存）
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 2) 复用的 v1 子集（由 run 脚本暂存到 vendor/）。保持目录结构，使 config_paths/cfg 解析正确。
COPY vendor/data_manager /app/vendor/data_manager
COPY vendor/sample_view  /app/vendor/sample_view
COPY vendor/cfg          /app/vendor/cfg

# 3) 应用本体
COPY backend  /app/backend
COPY frontend /app/frontend

# FORVIA_REPO_ROOT 指向 vendor → config.py 把 vendor 注入 sys.path，
# 并让 data_manager.config_paths 在 vendor/cfg 下找规则 yaml。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FORVIA_REPO_ROOT=/app/vendor \
    FORVIA_IN_DOCKER=1 \
    FORVIA_DATA_MOUNT=/data \
    FORVIA_TDMS_CACHE_DIR=/app/_data/tdms_cache

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
