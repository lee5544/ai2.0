from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable


def resolve_runtime_context(script_path: Path) -> tuple[type, Callable[..., dict[str, Any]], Path, bool]:
    """
    返回:
    - ChannelPredictor 类
    - read_tdms 函数
    - root_path: 仓库根目录或 runtime 导出包根目录
    - is_runtime_bundle: 是否在 runtime 导出包内执行
    """
    root_path = script_path.resolve().parents[1]
    runtime_dir = root_path / "runtime"

    if runtime_dir.exists():
        sys.path.insert(0, str(runtime_dir))
        from ChannelPredictor import ChannelPredictor  # noqa: E402
        from tdms_read import read_tdms  # noqa: E402

        return ChannelPredictor, read_tdms, root_path, True

    sys.path.insert(0, str(root_path))
    from ml.predictor import ChannelPredictor  # noqa: E402
    from data_manager.tdms_read import read_tdms  # noqa: E402

    return ChannelPredictor, read_tdms, root_path, False


def resolve_model_kwargs(
    model_dir: str | None,
    model_id: str | None,
    *,
    default_model_id: str | None = None,
    default_model_dir: str | Path | None = None,
) -> dict[str, Any]:
    if model_dir:
        return {"model_dir": str(Path(model_dir).expanduser().resolve())}
    if model_id:
        return {"model_id": model_id}
    if default_model_dir is not None:
        return {"model_dir": str(Path(default_model_dir).expanduser().resolve())}
    if default_model_id is None:
        raise ValueError("model_dir 与 model_id 不能同时为空")
    return {"model_id": default_model_id}


def apply_threshold_overrides(
    predictor: Any,
    *,
    threshold: float | None = None,
    threshold_dict_json: str | None = None,
) -> None:
    if threshold is not None:
        predictor.setThreshold(float(threshold))

    if threshold_dict_json:
        raw = json.loads(threshold_dict_json)
        predictor.setThresholdDict({int(k): float(v) for k, v in raw.items()})


def format_score_list(scores: dict[str, Any] | None) -> str:
    """Format all class probabilities for human-readable prediction logs."""
    parts = []
    for name, score in (scores or {}).items():
        try:
            value = f"{float(score):.6f}"
        except (TypeError, ValueError):
            value = str(score)
        parts.append(f"{name}：{value}")
    return f"[{'；'.join(parts)}]"
