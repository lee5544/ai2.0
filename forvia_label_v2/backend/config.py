"""全局配置 + 接入复用的仓库模块（data_manager / sample_view / cfg）。

通过 sys.path / FORVIA_REPO_ROOT 复用现网代码（read_tdms / Label 规则 等）。
绘图已全部由各卡片自包含，不再依赖 plot_utils。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# forvia_label_v2/backend/config.py -> 仓库根 AI-2.0
REPO_ROOT = Path(os.environ.get("FORVIA_REPO_ROOT", "")).expanduser() if os.environ.get(
    "FORVIA_REPO_ROOT"
) else Path(__file__).resolve().parents[2]

# 复用现有代码所需的两个 import 根
for p in (REPO_ROOT, REPO_ROOT / "forvia_label"):
    sp = str(p)
    if p.exists() and sp not in sys.path:
        sys.path.insert(0, sp)

def _disable_numba_cache_for_librosa() -> None:
    """关闭 librosa 内部 numba 的磁盘缓存（打包/只读环境下避免写缓存报错）。
    在任何卡片 import librosa 之前由本模块（最早被导入）执行。"""
    try:
        import numba
        from functools import wraps
    except Exception:
        return

    def _wrap(decorator):
        if getattr(decorator, "_forvia_cache_patched", False):
            return decorator

        @wraps(decorator)
        def _w(*a, **k):
            k["cache"] = False
            return decorator(*a, **k)

        _w._forvia_cache_patched = True
        return _w

    for name in ("jit", "njit", "vectorize", "guvectorize"):
        if hasattr(numba, name):
            setattr(numba, name, _wrap(getattr(numba, name)))


_disable_numba_cache_for_librosa()

TYPICAL_TAG = "[[典型异音]]"

# 与 data_manager/LabelStore.HEADER 严格一致
LABEL_HISTORY_COLUMNS = [
    "line", "sn", "sample_id", "timestamp", "source",
    "result_key", "result_id", "result_name",
    "reason_key", "reason_id", "reason_name", "reason_confidence",
    "label_version", "note",
]

# 启动时的默认数据（可由 /api/init 在运行期覆盖）
DEFAULT_SAMPLE_VIEW_PATH = os.environ.get("SAMPLE_VIEW_PATH", "")
DEFAULT_LABEL_RECORDS_DB_PATH = os.environ.get(
    "LABEL_RECORDS_DB_PATH",
    os.environ.get("SAMPLE_RECORDS_DB_PATH", ""),
)
DEFAULT_TDMS_ROOT = os.environ.get("TDMS_ROOT", "")
