#!/usr/bin/env python3
"""
BRF 训练入口，复用 RFModel 的训练逻辑。
"""

import argparse
from pathlib import Path

try:
    from ml.models.rf import RFModel as _BaseRFModel
except ModuleNotFoundError:
    from RFModel import RFModel as _BaseRFModel


DEFAULT_CONFIG_PATH = Path("cfg/epump4.yaml")


class BRFModel(_BaseRFModel):
    DEFAULT_MODEL_TYPE = "brf"


def _parse_args():
    parser = argparse.ArgumentParser(description="使用配置文件训练 BRF 模型")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"YAML 配置路径（默认: {DEFAULT_CONFIG_PATH}）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config_path = Path(args.config).expanduser()
    model = BRFModel(str(config_path))
    train_mode = str(model.train_cfg.get("train_mode") or "").strip().lower()
    if not train_mode:
        raise KeyError(f"配置缺少 train.train_mode: {config_path}")
    try:
        from model_registry import run_training
    except ModuleNotFoundError:
        from ml.train import run_training
    run_training(model, train_mode)
