"""ML 训练入口。

包含：
  ModelSpec / 模型注册表  — 从 models/registry.py 合并
  run_training            — 训练协调（normal / cross / grid）
  train_from_config       — 完整训练流程公开入口
  main_dataset            — 特征提取/加载
  main_train              — 配置 → 训练
  main_predict            — 单信号推理
  run_pipeline            — 端到端流水线
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ===========================================================================
# 模型注册表（原 models/registry.py）
# ===========================================================================

SUPPORTED_TRAIN_MODES = ("normal", "grid", "cross")


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    module_name: str
    class_name: str
    display_name: str
    description: str = ""
    parameter_key: str = "model_params"
    parameters: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "description": self.description,
            "parameter_key": self.parameter_key,
            "parameters": [dict(p) for p in self.parameters],
        }


_MODEL_REGISTRY: dict[str, ModelSpec] = {
    "xgb": ModelSpec(
        model_type="xgb",
        module_name="ml.models.xgb",
        class_name="XGBModel",
        display_name="XGBoost",
        description="梯度提升树，适合当前默认训练流程。",
        parameter_key="xgb_params",
        parameters=(
            {"key": "n_estimators",      "label": "最大树数量",             "type": "int",    "default": 400,       "min": 1},
            {"key": "max_depth",         "label": "单棵树最大深度",         "type": "int",    "default": 6,         "min": 1},
            {"key": "learning_rate",     "label": "学习率",                 "type": "float",  "default": 0.05,      "min": 0.001},
            {"key": "min_child_weight",  "label": "子节点最小权重",         "type": "float",  "default": 5,         "min": 0},
            {"key": "gamma",             "label": "最小分裂损失 gamma",     "type": "float",  "default": 1,         "min": 0},
            {"key": "reg_lambda",        "label": "L2 正则 reg_lambda",    "type": "float",  "default": 5,         "min": 0},
            {"key": "reg_alpha",         "label": "L1 正则 reg_alpha",     "type": "float",  "default": 0.5,       "min": 0},
            {"key": "subsample",         "label": "样本采样比例",           "type": "float",  "default": 0.8,       "min": 0.01, "max": 1},
            {"key": "colsample_bytree",  "label": "特征采样比例",           "type": "float",  "default": 0.7,       "min": 0.01, "max": 1},
            {"key": "eval_metric",       "label": "评估指标",               "type": "select", "default": "mlogloss",
             "options": ["mlogloss", "logloss", "error", "merror", "auc", "aucpr"]},
            {"key": "tree_method",       "label": "树构建算法",             "type": "select", "default": "hist",
             "options": ["hist", "approx", "exact", "auto"]},
            {"key": "random_state",      "label": "模型随机种子",           "type": "int",    "default": 42},
            {"key": "n_jobs",            "label": "并行线程数（-1 为全部）","type": "int",    "default": -1},
        ),
    ),
    "lgb": ModelSpec(
        model_type="lgb",
        module_name="ml.models.lgb",
        class_name="LGBModel",
        display_name="LightGBM",
        description="LightGBM 梯度提升，训练速度快，适合大特征量场景。",
        parameter_key="lgb_params",
        parameters=(
            {"key": "n_estimators",      "label": "最大树数量",             "type": "int",    "default": 400,  "min": 1},
            {"key": "num_leaves",        "label": "叶子数",                 "type": "int",    "default": 63,   "min": 2},
            {"key": "max_depth",         "label": "树最大深度（-1不限）",   "type": "int",    "default": -1},
            {"key": "learning_rate",     "label": "学习率",                 "type": "float",  "default": 0.05, "min": 0.001},
            {"key": "min_child_samples", "label": "叶节点最小样本数",       "type": "int",    "default": 20,   "min": 1},
            {"key": "min_split_gain",    "label": "分裂最小增益",           "type": "float",  "default": 0.0,  "min": 0},
            {"key": "reg_lambda",        "label": "L2 正则",                "type": "float",  "default": 5.0,  "min": 0},
            {"key": "reg_alpha",         "label": "L1 正则",                "type": "float",  "default": 0.5,  "min": 0},
            {"key": "subsample",         "label": "行采样比例",             "type": "float",  "default": 0.8,  "min": 0.01, "max": 1},
            {"key": "subsample_freq",    "label": "行采样频率",             "type": "int",    "default": 1,    "min": 0},
            {"key": "feature_fraction",  "label": "列采样比例",             "type": "float",  "default": 0.7,  "min": 0.01, "max": 1},
            {"key": "random_state",      "label": "模型随机种子",           "type": "int",    "default": 42},
            {"key": "n_jobs",            "label": "并行线程数（-1全部）",   "type": "int",    "default": -1},
        ),
    ),
    "rf": ModelSpec(
        model_type="rf",
        module_name="ml.models.rf",
        class_name="RFModel",
        display_name="Random Forest",
        description="随机森林，训练稳定，参数较少。",
        parameter_key="rf_params",
        parameters=(
            {"key": "n_estimators",      "label": "树数量",               "type": "int",   "default": 100, "min": 1},
            {"key": "max_depth",         "label": "最大深度",             "type": "int",   "default": 8,   "min": 1},
            {"key": "min_samples_split", "label": "节点最小切分样本",     "type": "int",   "default": 2,   "min": 2},
            {"key": "min_samples_leaf",  "label": "叶节点最小样本",       "type": "int",   "default": 1,   "min": 1},
            {"key": "max_features",      "label": "特征采样比例",         "type": "float", "default": 0.8, "min": 0.01, "max": 1},
        ),
    ),
    "brf": ModelSpec(
        model_type="brf",
        module_name="ml.models.brf",
        class_name="BRFModel",
        display_name="Balanced RF",
        description="平衡随机森林，适合类别不均衡数据。",
        parameter_key="rf_params",
        parameters=(
            {"key": "n_estimators",      "label": "树数量",               "type": "int",   "default": 100, "min": 1},
            {"key": "max_depth",         "label": "最大深度",             "type": "int",   "default": 8,   "min": 1},
            {"key": "min_samples_split", "label": "节点最小切分样本",     "type": "int",   "default": 2,   "min": 2},
            {"key": "min_samples_leaf",  "label": "叶节点最小样本",       "type": "int",   "default": 1,   "min": 1},
            {"key": "max_features",      "label": "特征采样比例",         "type": "float", "default": 0.8, "min": 0.01, "max": 1},
        ),
    ),
}


def register_model(
    model_type: str,
    module_name: str,
    class_name: str,
    *,
    display_name: str = "",
    description: str = "",
    parameter_key: str = "model_params",
    parameters: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> ModelSpec:
    normalized = str(model_type or "").strip().lower()
    if not normalized:
        raise ValueError("model_type 不能为空")
    spec = ModelSpec(
        model_type=normalized,
        module_name=str(module_name).strip(),
        class_name=str(class_name).strip(),
        display_name=str(display_name or normalized).strip(),
        description=str(description or "").strip(),
        parameter_key=str(parameter_key or "model_params").strip(),
        parameters=tuple(dict(p) for p in parameters),
    )
    if not spec.module_name or not spec.class_name:
        raise ValueError(f"模型 {normalized} 必须配置 module_name 和 class_name")
    _MODEL_REGISTRY[normalized] = spec
    return spec


def _external_registry_path() -> Path:
    configured = os.environ.get("FORVIA_MODEL_REGISTRY", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / "cfg" / "core" / "model_registry.yaml"


def _registered_models(registry_path: str | Path | None = None) -> dict[str, ModelSpec]:
    specs = dict(_MODEL_REGISTRY)
    path = Path(registry_path).expanduser() if registry_path else _external_registry_path()
    if not path.is_file():
        return specs
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = loaded.get("models") if isinstance(loaded, dict) else {}
    if not isinstance(models, dict):
        raise ValueError(f"模型注册配置 models 必须是 dict: {path}")
    for model_type, raw in models.items():
        if not isinstance(raw, dict) or raw.get("enabled", True) is False:
            specs.pop(str(model_type).strip().lower(), None)
            continue
        normalized = str(model_type).strip().lower()
        existing = specs.get(normalized)
        specs[normalized] = ModelSpec(
            model_type=normalized,
            module_name=str(raw.get("module_name") or (existing.module_name if existing else "")).strip(),
            class_name=str(raw.get("class_name") or (existing.class_name if existing else "")).strip(),
            display_name=str(raw.get("display_name") or (existing.display_name if existing else normalized)).strip(),
            description=str(raw.get("description") or (existing.description if existing else "")).strip(),
            parameter_key=str(raw.get("parameter_key") or (existing.parameter_key if existing else "model_params")).strip(),
            parameters=tuple(
                dict(p)
                for p in (
                    raw["parameters"]
                    if isinstance(raw.get("parameters"), list)
                    else (existing.parameters if existing else ())
                )
            ),
        )
        if not specs[normalized].module_name or not specs[normalized].class_name:
            raise ValueError(f"模型 {normalized} 必须配置 module_name 和 class_name: {path}")
    return specs


def list_registered_model_types() -> list[str]:
    return list(_registered_models().keys())


def list_model_specs() -> dict[str, dict[str, Any]]:
    return {mt: spec.to_dict() for mt, spec in _registered_models().items()}


def list_supported_train_modes() -> list[str]:
    return list(SUPPORTED_TRAIN_MODES)


def model_catalog() -> dict[str, Any]:
    """一次性返回注册表快照，供训练应用使用。"""
    specs = _registered_models()
    return {
        "model_types": list(specs),
        "model_specs": {mt: spec.to_dict() for mt, spec in specs.items()},
        "train_modes": list(SUPPORTED_TRAIN_MODES),
    }


def get_model_spec(model_type: object) -> ModelSpec:
    normalized = str(model_type or "").strip().lower()
    spec = _registered_models().get(normalized)
    if spec is None:
        supported = ", ".join(list_registered_model_types())
        raise ValueError(f"不支持的 model_type: {model_type}（支持: {supported}）")
    return spec


def create_model_instance(model_type: object, config_path: str | Path) -> Any:
    spec = get_model_spec(model_type)
    try:
        module = import_module(spec.module_name)
    except ModuleNotFoundError:
        fallback = spec.module_name if spec.module_name.startswith("ml.") else f"ml.{spec.module_name}"
        module = import_module(fallback)
    model_cls = getattr(module, spec.class_name, None)
    if model_cls is None:
        raise AttributeError(f"模型注册异常: {spec.module_name}.{spec.class_name} 不存在")
    return model_cls(str(config_path))


# ===========================================================================
# 训练协调（原 training/api.py）
# ===========================================================================

DEFAULT_CONFIG_PATH = Path("cfg/epump4.yaml")


def load_training_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    """读取 YAML 配置并校验必要字段。"""
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"配置文件格式错误，顶层必须是 dict: {path}")
    train_cfg = config.get("train") if isinstance(config.get("train"), dict) else {}
    if not str(train_cfg.get("model_type") or "").strip():
        raise KeyError(f"配置缺少 train.model_type: {path}")
    if not str(train_cfg.get("train_mode") or "").strip():
        raise KeyError(f"配置缺少 train.train_mode: {path}")
    return path, config


def run_training(model: Any, train_mode: object) -> None:
    """根据 train_mode 协调训练流程（normal / cross / grid）。"""
    from ml.training.split import (
        prepare_cross_validation_for_model,
        prepare_grid_search_cv_from_split,
        prepare_train_val_test_for_model,
    )

    normalized_mode = str(train_mode or "").strip().lower()
    split_random_state = int(
        getattr(model, "split_random_state", None)
        or getattr(model, "random_state", None)
        or getattr(model, "base_random_state", 42)
        or 42
    )

    if normalized_mode == "normal":
        handler = getattr(model, "train_and_evaluate", None)
        if not callable(handler):
            raise AttributeError(f"模型 {type(model).__name__} 缺少训练入口: train_and_evaluate")
        split_data = prepare_train_val_test_for_model(
            model, random_state=split_random_state, rebalance_train=True,
        )
        handler(split_data)
        return

    if normalized_mode == "cross":
        handler = getattr(model, "train_and_cross_validate", None)
        if not callable(handler):
            raise AttributeError(f"模型 {type(model).__name__} 缺少训练入口: train_and_cross_validate")
        cv_data = prepare_cross_validation_for_model(
            model,
            requested_splits=5,
            sample_random_state=split_random_state,
            cv_random_state=split_random_state,
            allow_oversample=False,
            split_name="5-fold CV",
        )
        handler(cv_data)
        return

    if normalized_mode == "grid":
        handler = getattr(model, "train_and_evaluate_grid", None)
        if not callable(handler):
            raise AttributeError(f"模型 {type(model).__name__} 缺少训练入口: train_and_evaluate_grid")
        split_data = prepare_train_val_test_for_model(
            model, random_state=split_random_state, rebalance_train=False,
        )
        grid_cv_data = prepare_grid_search_cv_from_split(
            split_data.train_df,
            split_strategy=str(getattr(model, "split_strategy", "reference_out")),
            requested_splits=3,
            random_state=split_random_state,
            split_name="GridSearchCV",
        )
        handler(split_data, grid_cv_data)
        return

    supported = ", ".join(SUPPORTED_TRAIN_MODES)
    raise ValueError(f"不支持的 train_mode: {train_mode}（支持: {supported}）")


def _copy_config_to_results(config_path: Path, model: Any) -> None:
    results_path = getattr(model, "results_path", None)
    if not results_path:
        print("[WARN] 模型未提供 results_path，跳过配置文件拷贝。")
        return
    destination = Path(str(results_path)).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, destination / "train_config.yaml")
    print(f"[INFO] 已拷贝训练配置: {destination / 'train_config.yaml'}")


def _generate_label_audit_outputs(model: Any) -> None:
    if bool(getattr(model, "_label_audit_generated", False)):
        print("[INFO] 标签复核清单已在训练流程内生成，跳过重复执行。")
        return

    model_hook = getattr(model, "_generate_label_audit_outputs", None)
    if callable(model_hook):
        model_hook()
        return

    results_path = getattr(model, "results_path", None)
    if not results_path:
        print("[WARN] 模型未提供 results_path，跳过标签复核清单生成。")
        return

    results_dir = Path(str(results_path)).expanduser()
    if not results_dir.exists():
        print(f"[WARN] results_path 不存在，跳过标签复核清单生成：{results_dir}")
        return

    try:
        from ml.training.audit_labels import run as run_label_audit
        summary = run_label_audit(results_dir=results_dir, output_dir=results_dir / "confirm_label")
    except FileNotFoundError as exc:
        print(f"[WARN] 未生成标签复核清单：{exc}")
        return
    except Exception as exc:
        print(f"[WARN] 标签复核清单生成失败：{exc}")
        return

    print(
        "[INFO] 标签复核清单已生成："
        f" candidates={summary.get('candidates', 0)},"
        f" suspicious_sns={summary.get('suspicious_sns', 0)},"
        f" false_ok={summary.get('false_ok', 0)}"
    )


def train_from_config(config_path: str | Path) -> Any:
    """完整训练流程入口：配置 → 模型实例化 → 训练 → 后处理。"""
    path, config = load_training_config(config_path)
    train_cfg = config["train"]
    model = create_model_instance(train_cfg["model_type"], path)
    run_training(model, train_cfg["train_mode"])
    _copy_config_to_results(path, model)
    _generate_label_audit_outputs(model)
    return model


# ===========================================================================
# 流水线（原 pipeline.py）
# ===========================================================================

def main_dataset(
    config_path: str | Path,
    *,
    rebuild: bool = False,
    output_dir: str | Path | None = None,
) -> Any:
    """提取特征，返回 FeatureDataset。"""
    from ml.dataset.build import build_xy, load_xy

    config_file = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}

    if rebuild:
        print(f"[pipeline] 重新提取特征: {config_file}")
        dataset = build_xy(config_file, output_dir=output_dir)
    else:
        print(f"[pipeline] 加载已有特征: {config_file}")
        dataset = load_xy(config, output_dir=output_dir)

    print(
        f"[pipeline] dataset 就绪: X={dataset.X.shape}, "
        f"labels={np.unique(dataset.y).tolist()}, "
        f"features={len(dataset.feature_names)}, "
        f"version={dataset.feature_version}"
    )
    return dataset


def main_train(config_path: str | Path) -> Any:
    """读配置、训练模型、保存结果，返回训练后的模型实例。"""
    config_file = Path(config_path).expanduser().resolve()
    print(f"[pipeline] 开始训练: {config_file}")
    model = train_from_config(config_file)
    print(f"[pipeline] 训练完成: results → {model.results_path}")
    return model


def main_predict(
    model_dir: str | Path,
    signal: np.ndarray,
    sampling_rate: int,
) -> dict:
    """单信号推理。"""
    from ml.predictor import ChannelPredictor
    predictor = ChannelPredictor(model_dir=model_dir)
    return predictor.predict_signal(signal, sampling_rate)


def main_predict_tdms(
    model_dir: str | Path,
    tdms_path: str | Path,
    *,
    line: str | None = None,
) -> dict:
    """TDMS 文件推理（返回 up/down 两路结果）。"""
    from ml.predictor import ChannelPredictor
    predictor = ChannelPredictor(model_dir=model_dir)
    return predictor.predict_tdms(tdms_path, line=line)


def run_pipeline(
    config_path: str | Path,
    *,
    rebuild_dataset: bool = False,
) -> Any:
    """端到端流水线：dataset → train。"""
    config_file = Path(config_path).expanduser().resolve()
    print(f"\n{'='*60}")
    print(f"[pipeline] 启动全流程: {config_file}")
    print(f"{'='*60}\n")
    main_dataset(config_file, rebuild=rebuild_dataset)
    model = main_train(config_file)
    print(f"\n{'='*60}")
    print(f"[pipeline] 全流程完成。模型输出: {model.results_path}")
    print(f"[pipeline] 推理入口: ChannelPredictor(model_dir='{model.results_path}')")
    print(f"{'='*60}\n")
    return model


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ml train: dataset / train / all",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python -m ml.train --config cfg/epump4.yaml
  python -m ml.train --config cfg/epump4.yaml --step dataset --rebuild
  python -m ml.train --config cfg/epump4.yaml --step train
""",
    )
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument(
        "--step",
        choices=["all", "dataset", "train"],
        default="all",
        help="执行阶段（默认: all）",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="重新从 TDMS 提取特征（仅 dataset/all 步骤有效）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).expanduser()
    if args.step == "dataset":
        main_dataset(config_path, rebuild=args.rebuild)
    elif args.step == "train":
        main_train(config_path)
    else:
        run_pipeline(config_path, rebuild_dataset=args.rebuild)


if __name__ == "__main__":
    main()
