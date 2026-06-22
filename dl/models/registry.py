from __future__ import annotations

from typing import Any

from torch import nn

from .cnn1d_model import CNN1DClassifier, build_cnn1d_model, resolve_cnn1d_config
from .cnn2d_model import CNN2DClassifier, build_cnn2d_model, resolve_cnn2d_config
from .lstm_model import LSTMClassifier, build_lstm_model, resolve_lstm_config
from .resnet2d_model import ResNet2DClassifier, build_resnet_model, resolve_resnet_config
from .tcn_model import TCNClassifier, build_tcn_model, resolve_tcn_config


def normalize_model_arch(value: str | None) -> str:
    raw = str(value or "").strip().lower().replace("-", "").replace("_", "")
    alias_map = {
        "": "cnn1d",
        "cnn": "cnn1d",
        "cnn1d": "cnn1d",
        "1dcnn": "cnn1d",
        "conv1d": "cnn1d",
        "cnn2d": "cnn2d",
        "2dcnn": "cnn2d",
        "conv2d": "cnn2d",
        "resnet": "resnet",
        "resnet2d": "resnet",
        "lstm": "lstm",
        "tcn": "tcn",
        "temporalcnn": "tcn",
        "dilatedcnn": "tcn",
    }
    if raw not in alias_map:
        supported = ", ".join(sorted({x for x in alias_map.values()}))
        raise ValueError(f"不支持的 dl 模型类型: {value}，支持: {supported}")
    return alias_map[raw]


def build_dl_model(
    *,
    arch: str | None,
    in_channels: int,
    sequence_length: int,
    num_classes: int,
    model_cfg: dict[str, Any] | None = None,
    train_cfg: dict[str, Any] | None = None,
) -> tuple[str, nn.Module, dict[str, Any]]:
    model_arch = normalize_model_arch(arch)
    if model_arch == "cnn1d":
        resolved_config = resolve_cnn1d_config(model_cfg=model_cfg, train_cfg=train_cfg)
        model = build_cnn1d_model(
            in_channels=in_channels,
            sequence_length=sequence_length,
            num_classes=num_classes,
            config=resolved_config,
        )
    elif model_arch == "cnn2d":
        resolved_config = resolve_cnn2d_config(model_cfg=model_cfg, train_cfg=train_cfg)
        model = build_cnn2d_model(
            in_channels=in_channels,
            sequence_length=sequence_length,
            num_classes=num_classes,
            config=resolved_config,
        )
    elif model_arch == "resnet":
        resolved_config = resolve_resnet_config(model_cfg=model_cfg, train_cfg=train_cfg)
        model = build_resnet_model(
            in_channels=in_channels,
            sequence_length=sequence_length,
            num_classes=num_classes,
            config=resolved_config,
        )
    elif model_arch == "lstm":
        resolved_config = resolve_lstm_config(model_cfg=model_cfg, train_cfg=train_cfg)
        model = build_lstm_model(
            in_channels=in_channels,
            sequence_length=sequence_length,
            num_classes=num_classes,
            config=resolved_config,
        )
    elif model_arch == "tcn":
        resolved_config = resolve_tcn_config(model_cfg=model_cfg, train_cfg=train_cfg)
        model = build_tcn_model(
            in_channels=in_channels,
            sequence_length=sequence_length,
            num_classes=num_classes,
            config=resolved_config,
        )
    else:  # pragma: no cover
        raise AssertionError(f"未处理的模型类型: {model_arch}")
    return model_arch, model, resolved_config


CNNWindowClassifier = CNN1DClassifier
