"""推理运行时工具。

包含：
  predict_with_threshold_rule   — 阈值决策函数（原 inference/thresholds.py）
  build_threshold_scorer        — sklearn-compatible scorer
  build_uniform_threshold_dict  — 均匀阈值字典
  export_runtime_bundle         — 打包推理运行时到结果目录（原 inference/runtime.py）
"""

from __future__ import annotations

import shutil
import stat
import textwrap
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping


# ===========================================================================
# 阈值推理工具（原 inference/thresholds.py）
# ===========================================================================

DEFAULT_THRESHOLD = 0.5


def build_uniform_threshold_dict(
    labels,
    *,
    default_threshold: float = DEFAULT_THRESHOLD,
) -> dict[int, float]:
    import numpy as np
    threshold = float(default_threshold)
    return {int(label): threshold for label in labels}


def _ensure_probability_matrix(y_proba, model_classes):
    import numpy as np
    classes = np.asarray(model_classes, dtype=int)
    proba = np.asarray(y_proba, dtype=float)

    if proba.ndim == 1:
        if classes.size == 2:
            proba = np.column_stack([1.0 - proba, proba])
        else:
            proba = proba.reshape(-1, 1)

    if proba.ndim != 2:
        raise ValueError(f"predict_proba 输出维度异常: ndim={proba.ndim}")
    if proba.shape[1] != classes.size:
        raise ValueError(
            f"predict_proba 列数与 classes_ 不一致: proba={proba.shape}, classes={classes.size}"
        )
    return proba, classes


def predict_with_threshold_rule(
    y_proba,
    model_classes,
    threshold_by_label: Mapping[int, float],
    *,
    ok_mlabel: int = 0,
):
    import numpy as np
    proba, classes = _ensure_probability_matrix(y_proba, model_classes)
    if proba.shape[0] == 0:
        return np.empty((0,), dtype=int)

    class_to_idx = {int(label): idx for idx, label in enumerate(classes)}
    ok_idx = class_to_idx.get(int(ok_mlabel))

    disabled = np.array(
        [float(threshold_by_label.get(int(c), DEFAULT_THRESHOLD)) >= 1.0 for c in classes],
        dtype=bool,
    )
    if ok_idx is not None:
        disabled[ok_idx] = False
    proba_for_pick = proba
    if disabled.any():
        proba_for_pick = proba.copy()
        proba_for_pick[:, disabled] = -np.inf

    best_idx = np.argmax(proba_for_pick, axis=1)
    predicted = classes[best_idx].astype(int, copy=False)
    outputs = predicted.copy()

    if ok_idx is None:
        return outputs

    row_index = np.arange(proba.shape[0])
    pred_conf = proba[row_index, best_idx]
    pred_threshold = np.array(
        [float(threshold_by_label.get(int(label), DEFAULT_THRESHOLD)) for label in predicted],
        dtype=float,
    )
    fail_threshold_mask = pred_conf <= pred_threshold
    outputs[fail_threshold_mask] = int(ok_mlabel)
    return outputs


def score_with_threshold_rule(
    estimator: Any,
    X,
    y_true,
    *,
    metric_func: Callable[..., float],
    threshold_by_label: Mapping[int, float],
    ok_mlabel: int = 0,
    metric_kwargs: Mapping[str, Any] | None = None,
) -> float:
    import numpy as np
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        raise AttributeError("estimator 缺少 classes_，无法按部署规则评分。")
    y_proba = estimator.predict_proba(X)
    y_pred = predict_with_threshold_rule(
        y_proba, classes, threshold_by_label, ok_mlabel=ok_mlabel,
    )
    kwargs = dict(metric_kwargs or {})
    return float(metric_func(np.asarray(y_true), y_pred, **kwargs))


def build_threshold_scorer(
    metric_func: Callable[..., float],
    *,
    threshold_by_label: Mapping[int, float],
    ok_mlabel: int = 0,
    **metric_kwargs: Any,
):
    return partial(
        score_with_threshold_rule,
        metric_func=metric_func,
        threshold_by_label={int(k): float(v) for k, v in threshold_by_label.items()},
        ok_mlabel=int(ok_mlabel),
        metric_kwargs=dict(metric_kwargs),
    )


# ===========================================================================
# 运行时打包（原 inference/runtime.py）
# ===========================================================================

_INFER_REQUIREMENTS = """\
numpy
pandas
tqdm
scipy
PyWavelets
librosa
scikit-learn
xgboost
pyyaml
nptdms
zstandard
"""

_QUICK_TEST_SCRIPTS = (
    "_bootstrap.py",
    "predict_tdms.py",
    "predict_folder.py",
    "predict_csv.py",
)

_DEFAULT_FEATURE_VERSION = "v4"

_RUNTIME_FEATURES_INIT_TEMPLATE = """\
from __future__ import annotations

from typing import Any, Callable

from .{module_name} import {function_name}

DEFAULT_FEATURE_VERSION = "{feature_version}"

FEATURE_EXTRACTORS: dict[str, Callable[..., Any]] = {{
    "{feature_version}": {function_name},
}}


def resolve_feature_version(cfg: dict[str, Any] | None = None) -> str:
    _ = cfg
    return DEFAULT_FEATURE_VERSION


def get_feature_extractor(version: str | None = None) -> Callable[..., Any]:
    normalized = str(version or DEFAULT_FEATURE_VERSION).strip().lower() or DEFAULT_FEATURE_VERSION
    if normalized not in FEATURE_EXTRACTORS:
        supported = ", ".join(sorted(FEATURE_EXTRACTORS))
        raise ValueError(f"不支持的 feature_version: {{normalized}}，可选: {{supported}}")
    return FEATURE_EXTRACTORS[normalized]


__all__ = [
    "DEFAULT_FEATURE_VERSION",
    "FEATURE_EXTRACTORS",
    "{function_name}",
    "get_feature_extractor",
    "resolve_feature_version",
]
"""

_STANDALONE_MAIN_PREDICT = """\
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _format_score_list(scores):
    return "[" + "；".join(
        f"{name}：{float(score):.6f}" for name, score in (scores or {}).items()
    ) + "]"


def _parse_args():
    parser = argparse.ArgumentParser(description="Standalone TDMS predictor (from results bundle)")
    parser.add_argument("--tdms", required=True, help="TDMS file path")
    parser.add_argument("--line", default=None, help="line name (optional, e.g. epump4)")
    parser.add_argument("--threshold", type=float, default=None, help="global threshold override")
    parser.add_argument(
        "--threshold-dict",
        default=None,
        help="JSON dict override, e.g. '{\\\"0\\\": 0.8, \\\"1\\\": 0.7}'",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    base_dir = Path(__file__).resolve().parent
    runtime_dir = base_dir / "runtime"
    if not runtime_dir.exists():
        raise FileNotFoundError(f"runtime folder missing: {runtime_dir}")

    sys.path.insert(0, str(runtime_dir))

    from ChannelPredictor import ChannelPredictor  # noqa: E402
    from tdms_read import read_tdms  # noqa: E402

    predictor = ChannelPredictor(model_dir=base_dir)

    if args.threshold is not None:
        predictor.setThreshold(args.threshold)

    if args.threshold_dict:
        threshold_dict_raw = json.loads(args.threshold_dict)
        threshold_dict = {int(k): float(v) for k, v in threshold_dict_raw.items()}
        predictor.setThresholdDict(threshold_dict)

    tdms_data = read_tdms(args.tdms, line=args.line)
    sampling_rate = tdms_data.get("sampling_rate")
    if sampling_rate is None:
        raise ValueError(f"TDMS sampling rate is missing: {args.tdms}")

    up = predictor.predict_detail(tdms_data["up_data"], sampling_rate, "up")
    down = predictor.predict_detail(tdms_data["down_data"], sampling_rate, "down")

    print(f"up:   {up['result']}, {up['mtype']}, score={up['score']:.6f}")
    print(f"up 分数列表={_format_score_list(up['scores'])}")
    print(f"down: {down['result']}, {down['mtype']}, score={down['score']:.6f}")
    print(f"down 分数列表={_format_score_list(down['scores'])}")


if __name__ == "__main__":
    main()
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _mark_executable(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass


def _normalize_feature_version(feature_version: str | None = None) -> str:
    normalized = str(feature_version or _DEFAULT_FEATURE_VERSION).strip().lower() or _DEFAULT_FEATURE_VERSION
    if not normalized.startswith("v"):
        raise ValueError(f"feature_version 格式不正确: {normalized}")
    return normalized


def _feature_module_name(feature_version: str) -> str:
    return f"extract_features_{feature_version}"


def _build_runtime_features_init(feature_version: str) -> str:
    module_name = _feature_module_name(feature_version)
    return textwrap.dedent(
        _RUNTIME_FEATURES_INIT_TEMPLATE.format(
            feature_version=feature_version,
            module_name=module_name,
            function_name=module_name,
        )
    )


_MODEL_CONFIG_FILENAMES = (
    "model.pkl",
    "predict_config.yaml",
)


def export_runtime_bundle(
    results_root: str | Path,
    model_id: str,
    *,
    source_model_dir: str | Path | None = None,
    feature_version: str | None = None,
) -> Path:
    """
    在 results_root/<model_id>/ 下新建推理包：
    - model.pkl                   （来自 source_model_dir）
    - predict_config.yaml         （来自 source_model_dir）
    - runtime/ChannelPredictor.py （来自 ml/predictor.py）
    - runtime/prediction_logic.py （来自 ml/runtime.py 本文件）
    - runtime/features/__init__.py
    - runtime/features/extract_features_<version>.py
    - runtime/tdms_read.py
    - runtime/line_rules.py
    - main_predict.py
    - quick_test/*.py
    - requirements_infer.txt

    Args:
        results_root:    结果根目录（如 "results/"），bundle 写入其下的 <model_id>/ 子目录。
        model_id:        模型 ID，用作子目录名（如 "epump4_epump_xgb"）。
        source_model_dir: 训练产物所在目录（含 model.pkl / predict_config.yaml）。
                          默认为 results_root/<model_id>/。
        feature_version: 特征版本号，如 "v4"；None 则使用默认版本。
    """
    results_root = Path(results_root).expanduser().resolve()
    bundle_dir = results_root / model_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 确定训练产物来源目录
    src_model_dir = Path(source_model_dir).expanduser().resolve() if source_model_dir else bundle_dir

    # 复制 model.pkl / predict_config.yaml
    for fname in _MODEL_CONFIG_FILENAMES:
        src_cfg = src_model_dir / fname
        if src_cfg.exists():
            destination = bundle_dir / fname
            if src_cfg.resolve() == destination.resolve():
                print(f"[INFO] 配置文件已位于推理包目录，跳过复制: {src_cfg}")
            else:
                shutil.copy2(src_cfg, destination)
                print(f"[INFO] 复制配置文件: {src_cfg} → {destination}")
        else:
            print(f"[WARN] 配置文件不存在，跳过: {src_cfg}")

    results_dir = bundle_dir  # 后续逻辑统一使用 bundle_dir
    runtime_dir = results_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    features_dir = runtime_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    repo_root = _repo_root()
    resolved_feature_version = _normalize_feature_version(feature_version)
    feature_module_name = _feature_module_name(resolved_feature_version)
    feature_src = repo_root / "ml" / "features" / f"{feature_module_name}.py"
    if not feature_src.exists():
        raise FileNotFoundError(f"runtime feature source file missing: {feature_src}")

    for stale_path in runtime_dir.glob("extract_features_v*.py"):
        stale_path.unlink()
    for stale_path in features_dir.glob("extract_features_v*.py"):
        if stale_path.name != feature_src.name:
            stale_path.unlink()

    # ml/predictor.py → runtime/ChannelPredictor.py
    # ml/runtime.py (this file) → runtime/prediction_logic.py
    copy_map = {
        repo_root / "ml" / "predictor.py": runtime_dir / "ChannelPredictor.py",
        Path(__file__).resolve(): runtime_dir / "prediction_logic.py",
        repo_root / "data_manager" / "tdms_read.py": runtime_dir / "tdms_read.py",
        repo_root / "data_manager" / "line_rules.py": runtime_dir / "line_rules.py",
        feature_src: features_dir / feature_src.name,
    }
    selected_features_src = (
        repo_root / "ml" / "features" / f"{resolved_feature_version}_selected_features.py"
    )
    if selected_features_src.exists():
        copy_map[selected_features_src] = features_dir / selected_features_src.name
    for src, dst in copy_map.items():
        if not src.exists():
            raise FileNotFoundError(f"runtime source file missing: {src}")
        shutil.copy2(src, dst)

    (features_dir / "__init__.py").write_text(
        _build_runtime_features_init(resolved_feature_version),
        encoding="utf-8",
    )

    quick_test_dir = results_dir / "quick_test"
    quick_test_dir.mkdir(parents=True, exist_ok=True)
    for script_name in _QUICK_TEST_SCRIPTS:
        src = repo_root / "quick_test" / script_name
        dst = quick_test_dir / script_name
        if not src.exists():
            raise FileNotFoundError(f"quick_test source file missing: {src}")
        shutil.copy2(src, dst)

    launcher_path = results_dir / "main_predict.py"
    launcher_path.write_text(textwrap.dedent(_STANDALONE_MAIN_PREDICT), encoding="utf-8")

    req_path = results_dir / "requirements_infer.txt"
    req_path.write_text(_INFER_REQUIREMENTS, encoding="utf-8")

    _mark_executable(launcher_path)
    for script_name in _QUICK_TEST_SCRIPTS:
        _mark_executable(quick_test_dir / script_name)

    return launcher_path
