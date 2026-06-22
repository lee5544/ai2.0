"""DL 训练调度模块。

合并自：
  model_registry.py  — ModelSpec / _MODEL_REGISTRY / run_training / list_*
  dataset_prep.py    — DLSplit / prepare_train_val_test_for_dl
  main_train.py      — CLI main()
"""
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from dl import train_core as core


# ===========================================================================
# 数据切分（原 dataset_prep.py）
# ===========================================================================

@dataclass
class DLSplit:
    """与 ml.dataset.split.TrainValTestSplit 对应的 DL 版本。"""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    full_df: pd.DataFrame
    schema: Dict[str, Any]
    channel_names: List[str]
    columns_by_channel: Dict[str, List[str]]
    seq_len: int
    compact_feature_column: str
    label_sample_dict: Dict[int, int] = field(default_factory=dict)
    mlabel_mtype_dict: Dict[int, str] = field(default_factory=dict)
    label_to_mlabel_map: Dict[int, int] = field(default_factory=dict)
    mlabel_to_reason_ids: Dict[int, List[int]] = field(default_factory=dict)
    class_name_by_id: Dict[int, str] = field(default_factory=dict)


def _load_feature_dataframe(model: Any) -> pd.DataFrame:
    dataset_files = list(getattr(model, "dataset_files"))
    if not dataset_files:
        raise RuntimeError(
            "未找到 dl 特征批次文件，请先执行: "
            f"python dl/main_dataset.py --config {getattr(model, 'config_path', '<config>')}"
        )
    dfs = []
    for path in dataset_files:
        print(f"读取特征批次: {path}")
        dfs.append(core._read_feature_batch(path))
    return pd.concat(dfs, ignore_index=True)


def prepare_train_val_test_for_dl(
    model: Any,
    *,
    random_state: int | None = None,
    rebalance_train: bool = True,
) -> DLSplit:
    """加载 mel 批次 → 构建标签运行时 → 切分，返回 DLSplit。"""
    del rebalance_train  # DL 的训练集精调在切分流程内完成

    global_train_cfg = dict(getattr(model, "global_train_cfg", {}) or {})
    schema = dict(getattr(model, "schema", {}) or {})
    compact_feature_column = core._resolve_compact_feature_column(schema)

    seed = int(getattr(model, "random_state", 42) if random_state is None else random_state)
    test_size = float(getattr(model, "test_size", 0.1))
    val_size = float(getattr(model, "val_size", 0.125))
    group_split_trials = int(getattr(model, "group_split_trials", 32))
    group_sample_trials = int(getattr(model, "group_sample_trials", 24))

    df = _load_feature_dataframe(model)
    available_reasons = core._scan_available_reasons(df)
    (
        label_sample_dict,
        mlabel_mtype_dict,
        label_to_mlabel_map,
        mlabel_to_reason_ids,
    ) = core.build_label_runtime(
        train_cfg=global_train_cfg,
        available_reasons=available_reasons,
    )
    class_name_by_id = {int(k): str(v) for k, v in mlabel_mtype_dict.items()}

    train_df, val_df, test_df = core._load_and_split_dataset(
        df=df,
        train_cfg=global_train_cfg,
        label_sample_dict=label_sample_dict,
        label_to_mlabel_map=label_to_mlabel_map,
        random_state=seed,
        test_size=test_size,
        val_size=val_size,
        group_split_trials=group_split_trials,
        group_sample_trials=group_sample_trials,
    )

    channel_names, columns_by_channel, seq_len = core._resolve_feature_layout(df, schema)
    print(f"特征布局: channels={channel_names}, seq_len={seq_len}")

    return DLSplit(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        full_df=df,
        schema=schema,
        channel_names=channel_names,
        columns_by_channel=columns_by_channel,
        seq_len=int(seq_len),
        compact_feature_column=compact_feature_column,
        label_sample_dict={int(k): int(v) for k, v in label_sample_dict.items()},
        mlabel_mtype_dict={int(k): str(v) for k, v in mlabel_mtype_dict.items()},
        label_to_mlabel_map={int(k): int(v) for k, v in label_to_mlabel_map.items()},
        mlabel_to_reason_ids={int(k): [int(x) for x in v] for k, v in mlabel_to_reason_ids.items()},
        class_name_by_id=class_name_by_id,
    )


# ===========================================================================
# 模型注册表（原 model_registry.py）
# ===========================================================================

SUPPORTED_TRAIN_MODES = ("normal", "grid", "cross")


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    module_name: str
    class_name: str
    display_name: str


_MODEL_REGISTRY: dict[str, ModelSpec] = {
    # 所有架构共用 CNNModel 作为训练包装器，
    # 具体网络结构由 CNNModel 内部的 build_dl_model(arch=...) 分发
    "cnn":    ModelSpec("cnn",    "CNNModel", "CNNModel", "CNN (1D)"),
    "cnn1d":  ModelSpec("cnn1d",  "CNNModel", "CNNModel", "CNN (1D)"),
    "cnn2d":  ModelSpec("cnn2d",  "CNNModel", "CNNModel", "CNN (2D)"),
    "lstm":   ModelSpec("lstm",   "CNNModel", "CNNModel", "LSTM"),
    "resnet": ModelSpec("resnet", "CNNModel", "CNNModel", "ResNet (2D)"),
    "tcn":    ModelSpec("tcn",    "CNNModel", "CNNModel", "TCN (膨胀卷积)"),
}


def list_registered_model_types() -> list[str]:
    return list(_MODEL_REGISTRY.keys())


def list_supported_train_modes() -> list[str]:
    return list(SUPPORTED_TRAIN_MODES)


def get_model_spec(model_type: object) -> ModelSpec:
    normalized = str(model_type or "").strip().lower()
    spec = _MODEL_REGISTRY.get(normalized)
    if spec is None:
        supported = ", ".join(list_registered_model_types())
        raise ValueError(f"不支持的 dl model_type: {model_type}（支持: {supported}）")
    return spec


def create_model_instance(model_type: object, config_path: str | Path) -> Any:
    spec = get_model_spec(model_type)
    module = import_module(f"dl.{spec.module_name}")
    model_cls = getattr(module, spec.class_name, None)
    if model_cls is None:
        raise AttributeError(f"模型注册异常: dl.{spec.module_name}.{spec.class_name} 不存在")
    return model_cls(str(config_path))


def run_training(model: Any, train_mode: object) -> None:
    normalized_mode = str(train_mode or "").strip().lower() or "normal"

    if normalized_mode == "normal":
        handler = getattr(model, "train_and_evaluate", None)
        if not callable(handler):
            raise AttributeError(f"模型 {type(model).__name__} 缺少训练入口: train_and_evaluate")
        split_data = prepare_train_val_test_for_dl(
            model,
            random_state=int(getattr(model, "split_random_state", getattr(model, "random_state", 42))),
            rebalance_train=True,
        )
        handler(split_data)
        return

    if normalized_mode == "cross":
        handler = getattr(model, "train_and_cross_validate", None)
        if not callable(handler):
            raise AttributeError(f"模型 {type(model).__name__} 缺少训练入口: train_and_cross_validate")
        handler(None)
        return

    if normalized_mode == "grid":
        handler = getattr(model, "train_and_evaluate_grid", None)
        if not callable(handler):
            raise AttributeError(f"模型 {type(model).__name__} 缺少训练入口: train_and_evaluate_grid")
        handler(None, None)
        return

    supported = ", ".join(list_supported_train_modes())
    raise ValueError(f"不支持的 train_mode: {train_mode}（支持: {supported}）")


# ===========================================================================
# CLI（原 main_train.py）
# ===========================================================================

DEFAULT_CONFIG_PATH = Path("cfg/epump4.yaml")


def _parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 dl 模型")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=f"YAML 配置路径（默认: {DEFAULT_CONFIG_PATH}）")
    parser.add_argument("--model-type", default=None, help="DL 模型类型（cnn/cnn1d...），默认读 dl.model_type 或 cnn")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[WARN] 忽略未识别参数（请改用 config 配置）: {unknown}")
    return args


def _resolve_model_type(cfg: dict, cli_model_type: str | None) -> str:
    if cli_model_type:
        return str(cli_model_type).strip().lower()
    dl_cfg = cfg.get("dl") if isinstance(cfg.get("dl"), dict) else {}
    raw = dl_cfg.get("model_type")
    if not raw:
        dl_model = dl_cfg.get("model") if isinstance(dl_cfg.get("model"), dict) else {}
        raw = dl_model.get("arch") or dl_model.get("name") or dl_cfg.get("arch")
    return (str(raw).strip().lower() if raw else "cnn") or "cnn"


def _resolve_train_mode(cfg: dict) -> str:
    dl_cfg = cfg.get("dl") if isinstance(cfg.get("dl"), dict) else {}
    dl_train = dl_cfg.get("train") if isinstance(dl_cfg.get("train"), dict) else {}
    global_train = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    mode = dl_train.get("train_mode") or global_train.get("train_mode") or "normal"
    return str(mode).strip().lower() or "normal"


def _copy_config_to_results(config_path: Path, model: Any) -> None:
    results_path = getattr(model, "results_path", None)
    if not results_path:
        print("[WARN] 模型未提供 results_path，跳过配置文件拷贝。")
        return
    dst_dir = Path(str(results_path)).expanduser()
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, dst_dir / "train_config.yaml")
    print(f"[INFO] 已拷贝训练配置: {dst_dir / 'train_config.yaml'}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_train_args(argv)
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式错误，顶层必须是 dict: {config_path}")

    model_type = _resolve_model_type(cfg, args.model_type)
    train_mode = _resolve_train_mode(cfg)
    print(f"[INFO] DL 训练: model_type={model_type}, train_mode={train_mode}, config={config_path}")

    model = create_model_instance(model_type, config_path)
    run_training(model, train_mode)
    _copy_config_to_results(config_path, model)


if __name__ == "__main__":
    main()
