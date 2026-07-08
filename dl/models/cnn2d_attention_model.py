from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn

from .base_model import BaseModel
from .cnn2d_model import Conv2dBlock
from .common import normalize_channel_list, pick_config_value, to_float, to_int, to_int_list


class ChannelAttention2d(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 8) -> None:
        super().__init__()
        channels = int(channels)
        reduced = max(1, channels // max(1, int(reduction)))
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * weights


class SpatialAttention2d(nn.Module):
    def __init__(self, *, kernel_size: int = 7) -> None:
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = max(0, kernel_size // 2)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        weights = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * weights


class CNN2DAttentionClassifier(BaseModel):
    """
    Input tensor:
    - [batch, channels, steps]
    Internal layout:
    - [batch, 1, channels, steps]
    """

    arch_name = "cnn2d_attention"

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        conv_channels: Iterable[int] = (32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
        hidden_dim: int = 128,
        attention_reduction: int = 8,
        spatial_kernel_size: int = 7,
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
                "attention_reduction": int(attention_reduction),
                "spatial_kernel_size": int(spatial_kernel_size),
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
            prev_channels = int(out_channels)

        self.encoder = nn.Sequential(*layers)
        self.channel_attention = ChannelAttention2d(prev_channels, reduction=int(attention_reduction))
        self.spatial_attention = SpatialAttention2d(kernel_size=int(spatial_kernel_size))
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
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = self.pool(x)
        return self.classifier(x)


def resolve_cnn2d_attention_config(*, model_cfg: Any, train_cfg: Any) -> dict[str, Any]:
    return {
        "conv_channels": to_int_list(
            pick_config_value(model_cfg, train_cfg, "conv_channels", default=[32, 64, 128]),
            default=[32, 64, 128],
        ),
        "kernel_size": max(3, to_int(pick_config_value(model_cfg, train_cfg, "kernel_size", default=3), 3)),
        "dropout": to_float(pick_config_value(model_cfg, train_cfg, "dropout", default=0.2), 0.2),
        "hidden_dim": max(8, to_int(pick_config_value(model_cfg, train_cfg, "hidden_dim", default=128), 128)),
        "attention_reduction": max(
            1,
            to_int(pick_config_value(model_cfg, train_cfg, "attention_reduction", default=8), 8),
        ),
        "spatial_kernel_size": max(
            3,
            to_int(pick_config_value(model_cfg, train_cfg, "spatial_kernel_size", default=7), 7),
        ),
    }


def build_cnn2d_attention_model(
    *,
    in_channels: int,
    sequence_length: int,
    num_classes: int,
    config: dict[str, Any],
) -> nn.Module:
    del sequence_length
    return CNN2DAttentionClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        conv_channels=config["conv_channels"],
        kernel_size=int(config["kernel_size"]),
        dropout=float(config["dropout"]),
        hidden_dim=int(config["hidden_dim"]),
        attention_reduction=int(config["attention_reduction"]),
        spatial_kernel_size=int(config["spatial_kernel_size"]),
    )
