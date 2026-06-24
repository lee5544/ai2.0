from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn

from .base_model import BaseModel
from .common import normalize_channel_list, pick_config_value, to_float, to_int, to_int_list


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = max(0, int(kernel_size) // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN2DClassifier(BaseModel):
    """
    输入张量形状:
    - [batch, channels, steps]
    内部会扩成 [batch, 1, mel_bins, steps]
    """

    arch_name = "cnn2d"

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        conv_channels: Iterable[int] = (32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
        hidden_dim: int = 128,
    ) -> None:
        conv_channels = normalize_channel_list(conv_channels, default=[32, 64, 128])
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            hyperparams={
                "conv_channels": list(conv_channels),
                "kernel_size": int(kernel_size),
                "dropout": float(dropout),
                "hidden_dim": int(hidden_dim),
            },
        )

        layers = []
        prev_channels = 1
        for out_channels in conv_channels:
            layers.append(
                Conv2dBlock(
                    prev_channels,
                    out_channels,
                    kernel_size=int(kernel_size),
                    dropout=float(dropout),
                )
            )
            prev_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_channels, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.pool(x)
        return self.classifier(x)


def resolve_cnn2d_config(*, model_cfg: Any, train_cfg: Any) -> dict[str, Any]:
    return {
        "conv_channels": to_int_list(
            pick_config_value(model_cfg, train_cfg, "conv_channels", default=[32, 64, 128]),
            default=[32, 64, 128],
        ),
        "kernel_size": max(3, to_int(pick_config_value(model_cfg, train_cfg, "kernel_size", default=3), 3)),
        "dropout": to_float(pick_config_value(model_cfg, train_cfg, "dropout", default=0.2), 0.2),
        "hidden_dim": max(8, to_int(pick_config_value(model_cfg, train_cfg, "hidden_dim", default=128), 128)),
    }


def build_cnn2d_model(
    *,
    in_channels: int,
    sequence_length: int,
    num_classes: int,
    config: dict[str, Any],
) -> nn.Module:
    del sequence_length
    return CNN2DClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        conv_channels=config["conv_channels"],
        kernel_size=int(config["kernel_size"]),
        dropout=float(config["dropout"]),
        hidden_dim=int(config["hidden_dim"]),
    )
