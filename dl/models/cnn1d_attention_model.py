from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn

from .base_model import BaseModel
from .cnn1d_model import Conv1dBlock
from .common import normalize_channel_list, pick_config_value, to_float, to_int, to_int_list


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 8) -> None:
        super().__init__()
        channels = int(channels)
        reduced = max(1, channels // max(1, int(reduction)))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Sequential(
            nn.Conv1d(channels, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.pool(x))


class TemporalAttentionPooling(nn.Module):
    def __init__(self, channels: int, *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(1, int(hidden_dim))
        self.score = nn.Sequential(
            nn.Conv1d(int(channels), hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Dropout(p=float(dropout)),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=-1)
        return torch.sum(x * weights, dim=-1)


class CNN1DAttentionClassifier(BaseModel):
    """
    Input tensor:
    - [batch, channels, steps]
    """

    arch_name = "cnn1d_attention"

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        conv_channels: Iterable[int] = (32, 64, 128),
        kernel_size: int = 5,
        dropout: float = 0.2,
        hidden_dim: int = 128,
        attention_hidden_dim: int = 64,
        attention_reduction: int = 8,
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
                "attention_hidden_dim": int(attention_hidden_dim),
                "attention_reduction": int(attention_reduction),
            },
        )

        layers = []
        prev_channels = int(in_channels)
        for out_channels in conv_channels:
            layers.append(
                Conv1dBlock(
                    prev_channels,
                    out_channels,
                    kernel_size=int(kernel_size),
                    dropout=float(dropout),
                )
            )
            prev_channels = int(out_channels)

        self.encoder = nn.Sequential(*layers)
        self.channel_attention = SqueezeExcite1d(prev_channels, reduction=int(attention_reduction))
        self.temporal_pool = TemporalAttentionPooling(
            prev_channels,
            hidden_dim=int(attention_hidden_dim),
            dropout=float(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(prev_channels, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.channel_attention(x)
        x = self.temporal_pool(x)
        return self.classifier(x)


def resolve_cnn1d_attention_config(*, model_cfg: Any, train_cfg: Any) -> dict[str, Any]:
    return {
        "conv_channels": to_int_list(
            pick_config_value(model_cfg, train_cfg, "conv_channels", default=[32, 64, 128]),
            default=[32, 64, 128],
        ),
        "kernel_size": max(1, to_int(pick_config_value(model_cfg, train_cfg, "kernel_size", default=5), 5)),
        "dropout": to_float(pick_config_value(model_cfg, train_cfg, "dropout", default=0.2), 0.2),
        "hidden_dim": max(8, to_int(pick_config_value(model_cfg, train_cfg, "hidden_dim", default=128), 128)),
        "attention_hidden_dim": max(
            1,
            to_int(pick_config_value(model_cfg, train_cfg, "attention_hidden_dim", default=64), 64),
        ),
        "attention_reduction": max(
            1,
            to_int(pick_config_value(model_cfg, train_cfg, "attention_reduction", default=8), 8),
        ),
    }


def build_cnn1d_attention_model(
    *,
    in_channels: int,
    sequence_length: int,
    num_classes: int,
    config: dict[str, Any],
) -> nn.Module:
    del sequence_length
    return CNN1DAttentionClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        conv_channels=config["conv_channels"],
        kernel_size=int(config["kernel_size"]),
        dropout=float(config["dropout"]),
        hidden_dim=int(config["hidden_dim"]),
        attention_hidden_dim=int(config["attention_hidden_dim"]),
        attention_reduction=int(config["attention_reduction"]),
    )
