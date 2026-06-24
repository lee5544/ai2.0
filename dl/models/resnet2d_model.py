from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn

from .base_model import BaseModel
from .common import normalize_channel_list, pick_config_value, to_float, to_int, to_int_list


class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=float(dropout))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if int(stride) != 1 or int(in_channels) != int(out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.relu(out)


class ResNet2DClassifier(BaseModel):
    """
    输入张量形状:
    - [batch, channels, steps]
    内部会扩成 [batch, 1, mel_bins, steps]
    """

    arch_name = "resnet"

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        conv_channels: Iterable[int] = (32, 64, 128),
        dropout: float = 0.2,
        hidden_dim: int = 128,
        blocks_per_stage: int = 2,
    ) -> None:
        stage_channels = normalize_channel_list(conv_channels, default=[32, 64, 128])
        blocks_per_stage = max(1, int(blocks_per_stage))
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            hyperparams={
                "conv_channels": list(stage_channels),
                "dropout": float(dropout),
                "hidden_dim": int(hidden_dim),
                "blocks_per_stage": int(blocks_per_stage),
            },
        )
        stem_channels = int(stage_channels[0])
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        stages = []
        prev_channels = stem_channels
        for stage_idx, out_channels in enumerate(stage_channels):
            blocks = []
            stride = 1 if stage_idx == 0 else 2
            blocks.append(
                ResidualBlock2d(
                    prev_channels,
                    int(out_channels),
                    stride=stride,
                    dropout=float(dropout),
                )
            )
            for _ in range(blocks_per_stage - 1):
                blocks.append(
                    ResidualBlock2d(
                        int(out_channels),
                        int(out_channels),
                        stride=1,
                        dropout=float(dropout),
                    )
                )
            stages.append(nn.Sequential(*blocks))
            prev_channels = int(out_channels)
        self.encoder = nn.Sequential(*stages)
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
        x = self.stem(x)
        x = self.encoder(x)
        x = self.pool(x)
        return self.classifier(x)


def resolve_resnet_config(*, model_cfg: Any, train_cfg: Any) -> dict[str, Any]:
    return {
        "conv_channels": to_int_list(
            pick_config_value(model_cfg, train_cfg, "conv_channels", default=[32, 64, 128]),
            default=[32, 64, 128],
        ),
        "dropout": to_float(pick_config_value(model_cfg, train_cfg, "dropout", default=0.2), 0.2),
        "hidden_dim": max(8, to_int(pick_config_value(model_cfg, train_cfg, "hidden_dim", default=128), 128)),
        "resnet_blocks_per_stage": max(
            1,
            to_int(
                pick_config_value(
                    model_cfg,
                    train_cfg,
                    "resnet_blocks_per_stage",
                    "blocks_per_stage",
                    default=2,
                ),
                2,
            ),
        ),
    }


def build_resnet_model(
    *,
    in_channels: int,
    sequence_length: int,
    num_classes: int,
    config: dict[str, Any],
) -> nn.Module:
    del sequence_length
    return ResNet2DClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        conv_channels=config["conv_channels"],
        dropout=float(config["dropout"]),
        hidden_dim=int(config["hidden_dim"]),
        blocks_per_stage=int(config["resnet_blocks_per_stage"]),
    )
