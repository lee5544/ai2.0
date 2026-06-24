from __future__ import annotations

from typing import Any, Iterable, Literal

import torch
from torch import nn

from .base_model import BaseModel
from .common import normalize_channel_list, pick_config_value, to_float, to_int, to_int_list


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        pool: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = max(0, int(kernel_size) // 2)

        layers: list[nn.Module] = [
            nn.Conv1d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                stride=int(stride),
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(int(out_channels)),
            nn.ReLU(inplace=True),
        ]

        if pool:
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        if dropout > 0:
            layers.append(nn.Dropout(p=float(dropout)))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WindowCNNEncoder(nn.Module):
    """
    对每个短窗/中窗共享使用的 1D-CNN 编码器。

    输入:
        [B * T, C, window_size]

    输出:
        [B * T, feature_dim]
    """

    def __init__(
        self,
        *,
        in_channels: int,
        conv_channels: Iterable[int],
        kernel_size: int,
        dropout: float,
        feature_dim: int,
    ) -> None:
        super().__init__()

        conv_channels = normalize_channel_list(conv_channels, default=[32, 64, 128])

        layers: list[nn.Module] = []
        prev_channels = int(in_channels)

        for out_channels in conv_channels:
            layers.append(
                Conv1dBlock(
                    prev_channels,
                    int(out_channels),
                    kernel_size=int(kernel_size),
                    stride=1,
                    pool=True,
                    dropout=float(dropout),
                )
            )
            prev_channels = int(out_channels)

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_channels, int(feature_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pool(x)
        x = self.proj(x)
        return x


class AttentionPooling(nn.Module):
    """
    对窗口序列做 attention pooling。

    输入:
        [B, T, D]

    输出:
        [B, D]
    """

    def __init__(self, feature_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or feature_dim)

        self.score = nn.Sequential(
            nn.Linear(int(feature_dim), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        scores = self.score(x).squeeze(-1)          # [B, T]
        weights = torch.softmax(scores, dim=1)      # [B, T]
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled


class TopKFeaturePooling(nn.Module):
    """
    对窗口序列做 top-k feature pooling。

    思路:
    - 先给每个窗口学一个重要性分数
    - 取分数最高的 k 个窗口
    - 对它们的 feature 求平均

    输入:
        [B, T, D]

    输出:
        [B, D]
    """

    def __init__(self, feature_dim: int, k: int = 5) -> None:
        super().__init__()
        self.k = max(1, int(k))
        self.score = nn.Linear(int(feature_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        bsz, steps, dim = x.shape
        k = min(self.k, steps)

        scores = self.score(x).squeeze(-1)          # [B, T]
        indices = torch.topk(scores, k=k, dim=1).indices  # [B, k]

        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, dim)
        topk_features = torch.gather(x, dim=1, index=expanded_indices)  # [B, k, D]

        return topk_features.mean(dim=1)


class TCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        padding = (int(kernel_size) - 1) * int(dilation) // 2

        self.net = nn.Sequential(
            nn.Conv1d(
                int(channels),
                int(channels),
                kernel_size=int(kernel_size),
                padding=padding,
                dilation=int(dilation),
                bias=False,
            ),
            nn.BatchNorm1d(int(channels)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Conv1d(
                int(channels),
                int(channels),
                kernel_size=int(kernel_size),
                padding=padding,
                dilation=int(dilation),
                bias=False,
            ),
            nn.BatchNorm1d(int(channels)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T]
        return x + self.net(x)


class TCNAggregator(nn.Module):
    """
    对窗口 feature 序列建模周期性/时序关系。

    输入:
        [B, T, D]

    输出:
        [B, D]
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        kernel_size: int = 3,
        dilations: Iterable[int] = (1, 2, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        blocks = []
        for dilation in dilations:
            blocks.append(
                TCNBlock(
                    int(feature_dim),
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    dropout=float(dropout),
                )
            )

        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return x


class WindowBranch(nn.Module):
    """
    短窗/中窗分支。

    输入:
        [B, C, L]

    处理:
        unfold -> [B, T, C, W]
        shared CNN -> [B, T, D]
        aggregator -> [B, D]
    """

    def __init__(
        self,
        *,
        in_channels: int,
        window_size: int,
        stride: int,
        conv_channels: Iterable[int],
        kernel_size: int,
        dropout: float,
        feature_dim: int,
        aggregator: Literal["attention", "topk", "tcn", "mean", "max"] = "attention",
        topk: int = 5,
        tcn_kernel_size: int = 3,
        tcn_dilations: Iterable[int] = (1, 2, 4),
    ) -> None:
        super().__init__()

        self.window_size = max(1, int(window_size))
        self.stride = max(1, int(stride))
        self.feature_dim = int(feature_dim)
        self.aggregator_name = str(aggregator).lower()

        self.encoder = WindowCNNEncoder(
            in_channels=int(in_channels),
            conv_channels=conv_channels,
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            feature_dim=int(feature_dim),
        )

        if self.aggregator_name == "attention":
            self.aggregator: nn.Module = AttentionPooling(feature_dim=int(feature_dim))
        elif self.aggregator_name == "topk":
            self.aggregator = TopKFeaturePooling(feature_dim=int(feature_dim), k=int(topk))
        elif self.aggregator_name == "tcn":
            self.aggregator = TCNAggregator(
                feature_dim=int(feature_dim),
                kernel_size=int(tcn_kernel_size),
                dilations=tcn_dilations,
                dropout=float(dropout),
            )
        elif self.aggregator_name in {"mean", "max"}:
            self.aggregator = nn.Identity()
        else:
            raise ValueError(
                f"Unsupported aggregator={aggregator!r}. "
                "Use one of: attention, topk, tcn, mean, max."
            )

    def _make_windows(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, C, L], got {tuple(x.shape)}")

        length = int(x.shape[-1])
        if length < self.window_size:
            raise ValueError(
                f"Input length {length} is shorter than window_size {self.window_size}."
            )

        # unfold output: [B, C, T, W]
        windows = x.unfold(dimension=-1, size=self.window_size, step=self.stride)

        # [B, C, T, W] -> [B, T, C, W]
        windows = windows.permute(0, 2, 1, 3).contiguous()
        return windows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows = self._make_windows(x)
        bsz, num_windows, channels, window_size = windows.shape

        # [B, T, C, W] -> [B*T, C, W]
        windows = windows.view(bsz * num_windows, channels, window_size)

        # [B*T, D]
        features = self.encoder(windows)

        # [B, T, D]
        features = features.view(bsz, num_windows, self.feature_dim)

        if self.aggregator_name == "mean":
            return features.mean(dim=1)

        if self.aggregator_name == "max":
            return features.max(dim=1).values

        return self.aggregator(features)


class GlobalCNNBranch(nn.Module):
    """
    全局分支，直接处理完整 5s 信号。

    输入:
        [B, C, 100000]

    输出:
        [B, feature_dim]
    """

    def __init__(
        self,
        *,
        in_channels: int,
        conv_channels: Iterable[int],
        kernel_size: int,
        dropout: float,
        feature_dim: int,
    ) -> None:
        super().__init__()

        conv_channels = normalize_channel_list(conv_channels, default=[32, 64, 128, 256])

        layers: list[nn.Module] = []
        prev_channels = int(in_channels)

        for out_channels in conv_channels:
            layers.append(
                Conv1dBlock(
                    prev_channels,
                    int(out_channels),
                    kernel_size=int(kernel_size),
                    stride=2,
                    pool=True,
                    dropout=float(dropout),
                )
            )
            prev_channels = int(out_channels)

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_channels, int(feature_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pool(x)
        x = self.proj(x)
        return x


class MultiScaleCNN1DClassifier(BaseModel):
    """
    多尺度多分支 1D-CNN 分类模型。

    适用任务:
    - 5s 信号多分类
    - 每个 5s 样本只有一个类别标签
    - 输入形状: [B, C, 100000]
    - 输出形状: [B, num_classes]
    - loss: CrossEntropyLoss

    三个分支:
    - short branch: 捕捉短暂冲击、周期性冲击
    - mid branch: 捕捉局部抖动
    - global branch: 捕捉全局抖动、整体频谱/能量变化
    """

    arch_name = "multiscale_cnn1d"

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        short_window_size: int = 1000,
        short_stride: int = 500,
        mid_window_size: int = 5000,
        mid_stride: int = 2500,
        short_conv_channels: Iterable[int] = (32, 64, 128),
        mid_conv_channels: Iterable[int] = (32, 64, 128),
        global_conv_channels: Iterable[int] = (32, 64, 128, 256),
        short_kernel_size: int = 7,
        mid_kernel_size: int = 9,
        global_kernel_size: int = 15,
        branch_feature_dim: int = 128,
        short_aggregator: Literal["attention", "topk", "tcn", "mean", "max"] = "tcn",
        mid_aggregator: Literal["attention", "topk", "tcn", "mean", "max"] = "topk",
        short_topk: int = 8,
        mid_topk: int = 4,
        tcn_kernel_size: int = 3,
        tcn_dilations: Iterable[int] = (1, 2, 4),
        dropout: float = 0.2,
        hidden_dim: int = 256,
    ) -> None:
        short_conv_channels = normalize_channel_list(short_conv_channels, default=[32, 64, 128])
        mid_conv_channels = normalize_channel_list(mid_conv_channels, default=[32, 64, 128])
        global_conv_channels = normalize_channel_list(global_conv_channels, default=[32, 64, 128, 256])
        tcn_dilations = to_int_list(tcn_dilations, default=[1, 2, 4])

        super().__init__(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            hyperparams={
                "short_window_size": int(short_window_size),
                "short_stride": int(short_stride),
                "mid_window_size": int(mid_window_size),
                "mid_stride": int(mid_stride),
                "short_conv_channels": list(short_conv_channels),
                "mid_conv_channels": list(mid_conv_channels),
                "global_conv_channels": list(global_conv_channels),
                "short_kernel_size": int(short_kernel_size),
                "mid_kernel_size": int(mid_kernel_size),
                "global_kernel_size": int(global_kernel_size),
                "branch_feature_dim": int(branch_feature_dim),
                "short_aggregator": str(short_aggregator),
                "mid_aggregator": str(mid_aggregator),
                "short_topk": int(short_topk),
                "mid_topk": int(mid_topk),
                "tcn_kernel_size": int(tcn_kernel_size),
                "tcn_dilations": list(tcn_dilations),
                "dropout": float(dropout),
                "hidden_dim": int(hidden_dim),
            },
        )

        self.short_branch = WindowBranch(
            in_channels=int(in_channels),
            window_size=int(short_window_size),
            stride=int(short_stride),
            conv_channels=short_conv_channels,
            kernel_size=int(short_kernel_size),
            dropout=float(dropout),
            feature_dim=int(branch_feature_dim),
            aggregator=short_aggregator,
            topk=int(short_topk),
            tcn_kernel_size=int(tcn_kernel_size),
            tcn_dilations=tcn_dilations,
        )

        self.mid_branch = WindowBranch(
            in_channels=int(in_channels),
            window_size=int(mid_window_size),
            stride=int(mid_stride),
            conv_channels=mid_conv_channels,
            kernel_size=int(mid_kernel_size),
            dropout=float(dropout),
            feature_dim=int(branch_feature_dim),
            aggregator=mid_aggregator,
            topk=int(mid_topk),
            tcn_kernel_size=int(tcn_kernel_size),
            tcn_dilations=tcn_dilations,
        )

        self.global_branch = GlobalCNNBranch(
            in_channels=int(in_channels),
            conv_channels=global_conv_channels,
            kernel_size=int(global_kernel_size),
            dropout=float(dropout),
            feature_dim=int(branch_feature_dim),
        )

        fusion_dim = int(branch_feature_dim) * 3

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        short_feature = self.short_branch(x)
        mid_feature = self.mid_branch(x)
        global_feature = self.global_branch(x)

        fused = torch.cat([short_feature, mid_feature, global_feature], dim=1)
        logits = self.classifier(fused)
        return logits


def resolve_multiscale_cnn1d_config(*, model_cfg: Any, train_cfg: Any) -> dict[str, Any]:
    return {
        "short_window_size": max(
            1,
            to_int(
                pick_config_value(model_cfg, train_cfg, "short_window_size", default=1000),
                1000,
            ),
        ),
        "short_stride": max(
            1,
            to_int(
                pick_config_value(model_cfg, train_cfg, "short_stride", default=500),
                500,
            ),
        ),
        "mid_window_size": max(
            1,
            to_int(
                pick_config_value(model_cfg, train_cfg, "mid_window_size", default=5000),
                5000,
            ),
        ),
        "mid_stride": max(
            1,
            to_int(
                pick_config_value(model_cfg, train_cfg, "mid_stride", default=2500),
                2500,
            ),
        ),
        "short_conv_channels": to_int_list(
            pick_config_value(model_cfg, train_cfg, "short_conv_channels", default=[32, 64, 128]),
            default=[32, 64, 128],
        ),
        "mid_conv_channels": to_int_list(
            pick_config_value(model_cfg, train_cfg, "mid_conv_channels", default=[32, 64, 128]),
            default=[32, 64, 128],
        ),
        "global_conv_channels": to_int_list(
            pick_config_value(model_cfg, train_cfg, "global_conv_channels", default=[32, 64, 128, 256]),
            default=[32, 64, 128, 256],
        ),
        "short_kernel_size": max(
            1,
            to_int(
                pick_config_value(model_cfg, train_cfg, "short_kernel_size", default=7),
                7,
            ),
        ),
        "mid_kernel_size": max(
            1,
            to_int(
                pick_config_value(model_cfg, train_cfg, "mid_kernel_size", default=9),
                9,
            ),
        ),
        "global_kernel_size": max(
            1,
            to_int(
                pick_config_value(model_cfg, train_cfg, "global_kernel_size", default=15),
                15,
            ),
        ),
        "branch_feature_dim": max(
            16,
            to_int(
                pick_config_value(model_cfg, train_cfg, "branch_feature_dim", default=128),
                128,
            ),
        ),
        "short_aggregator": str(
            pick_config_value(model_cfg, train_cfg, "short_aggregator", default="tcn")
        ),
        "mid_aggregator": str(
            pick_config_value(model_cfg, train_cfg, "mid_aggregator", default="topk")
        ),
        "short_topk": max(
            1,
            to_int(
                pick_config_value(model_cfg, train_cfg, "short_topk", default=8),
                8,
            ),
        ),
        "mid_topk": max(
            1,
            to_int(
                pick_config_value(model_cfg, train_cfg, "mid_topk", default=4),
                4,
            ),
        ),
        "tcn_kernel_size": max(
            1,
            to_int(
                pick_config_value(model_cfg, train_cfg, "tcn_kernel_size", default=3),
                3,
            ),
        ),
        "tcn_dilations": to_int_list(
            pick_config_value(model_cfg, train_cfg, "tcn_dilations", default=[1, 2, 4]),
            default=[1, 2, 4],
        ),
        "dropout": to_float(
            pick_config_value(model_cfg, train_cfg, "dropout", default=0.2),
            0.2,
        ),
        "hidden_dim": max(
            8,
            to_int(
                pick_config_value(model_cfg, train_cfg, "hidden_dim", default=256),
                256,
            ),
        ),
    }


def build_multiscale_cnn1d_model(
    *,
    in_channels: int,
    sequence_length: int,
    num_classes: int,
    config: dict[str, Any],
) -> nn.Module:
    # 当前模型内部使用 unfold，可接受任意 >= window_size 的 sequence_length。
    # 这里保留 sequence_length 参数，方便和你现有 build 接口兼容。
    del sequence_length

    return MultiScaleCNN1DClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        short_window_size=int(config["short_window_size"]),
        short_stride=int(config["short_stride"]),
        mid_window_size=int(config["mid_window_size"]),
        mid_stride=int(config["mid_stride"]),
        short_conv_channels=config["short_conv_channels"],
        mid_conv_channels=config["mid_conv_channels"],
        global_conv_channels=config["global_conv_channels"],
        short_kernel_size=int(config["short_kernel_size"]),
        mid_kernel_size=int(config["mid_kernel_size"]),
        global_kernel_size=int(config["global_kernel_size"]),
        branch_feature_dim=int(config["branch_feature_dim"]),
        short_aggregator=config["short_aggregator"],
        mid_aggregator=config["mid_aggregator"],
        short_topk=int(config["short_topk"]),
        mid_topk=int(config["mid_topk"]),
        tcn_kernel_size=int(config["tcn_kernel_size"]),
        tcn_dilations=config["tcn_dilations"],
        dropout=float(config["dropout"]),
        hidden_dim=int(config["hidden_dim"]),
    )