from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .common import normalize_channel_list, pick_config_value, to_bool, to_float, to_int, to_int_list


# ---------------------------------------------------------------------------
# 基础构件
# ---------------------------------------------------------------------------

class _DilatedBlock(nn.Module):
    """单个 TCN 膨胀残差块。

    结构（两层膨胀卷积 + 残差连接）::

        x ──┬── [pad] ─ Conv1d(d) ─ BN ─ ReLU ─ Dropout
            │   [pad] ─ Conv1d(d) ─ BN ─ ReLU ─ Dropout ──┬── out
            └────────── shortcut(1×1 Conv / Identity) ───────┘

    参数
    ----
    in_ch, out_ch : int
        输入/输出通道数。
    kernel_size : int
        卷积核大小（建议奇数：3 或 5）。
    dilation : int
        膨胀系数，通常取 2^层号（1, 2, 4, 8, …）。
    dropout : float
        Dropout 概率。
    causal : bool
        True = 因果模式（仅左侧填充，适合实时/序列生成）；
        False = 非因果模式（双侧对称填充，适合分类任务，默认）。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()
        self.causal = causal
        # 每层需要的总填充量 = (kernel_size - 1) * dilation（保持序列长度不变）
        total_pad = (kernel_size - 1) * dilation
        if causal:
            self.pad_l = total_pad
            self.pad_r = 0
        else:
            self.pad_l = total_pad // 2
            self.pad_r = total_pad - self.pad_l

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.act1  = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act2  = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(p=dropout)

        # 残差路径：通道数不同时用 1×1 卷积对齐
        if in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_l == 0 and self.pad_r == 0:
            return x
        return F.pad(x, (self.pad_l, self.pad_r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.drop1(self.act1(self.bn1(self.conv1(self._pad(x)))))
        out = self.drop2(self.act2(self.bn2(self.conv2(self._pad(out)))))

        return out + identity


# ---------------------------------------------------------------------------
# TCN 分类器
# ---------------------------------------------------------------------------

class TCNClassifier(nn.Module):
    """TCN（时序卷积网络）分类器，使用指数递增的膨胀卷积。

    输入张量形状: ``[batch, channels, steps]``

    架构::

        Input [B, C_in, T]
          │
          ▼
        DilatedBlock(dilation=1)
        DilatedBlock(dilation=2)
        DilatedBlock(dilation=4)
           …  (共 num_blocks 个，膨胀系数 2^i)
          │
          ▼
        GlobalAvgPool1d  →  [B, C_out]
          │
          ▼
        Linear(C_out → hidden_dim) → ReLU → Dropout
          │
          ▼
        Linear(hidden_dim → num_classes)

    感受野大小::

        RF = 1 + 2 * (kernel_size - 1) * (2^num_blocks - 1)

    参数
    ----
    in_channels : int
        输入特征通道数。
    num_classes : int
        分类类别数（≥ 2）。
    num_channels : list[int]
        每个 block 的输出通道数列表，长度即为 block 总数。
        例如 ``[64, 64, 64, 64]`` 表示 4 个 block，每个 64 通道。
    kernel_size : int
        膨胀卷积核大小，建议 3 或 5（默认 3）。
    dropout : float
        各 block 内的 Dropout 概率（默认 0.2）。
    hidden_dim : int
        分类头全连接隐层大小（默认 128）。
    causal : bool
        True = 因果模式；False = 非因果模式（分类任务推荐，默认 False）。
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_channels: Iterable[int] = (64, 64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
        hidden_dim: int = 128,
        causal: bool = False,
    ) -> None:
        super().__init__()
        num_channels = normalize_channel_list(num_channels, default=[64, 64, 64, 64])
        if int(in_channels) <= 0:
            raise ValueError(f"in_channels 必须 > 0，当前: {in_channels}")
        if int(num_classes) <= 1:
            raise ValueError(f"num_classes 必须 > 1，当前: {num_classes}")
        if int(kernel_size) < 2:
            raise ValueError(f"kernel_size 必须 ≥ 2，当前: {kernel_size}")

        blocks: list[nn.Module] = []
        prev_ch = int(in_channels)
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            blocks.append(
                _DilatedBlock(
                    prev_ch, out_ch,
                    kernel_size=int(kernel_size),
                    dilation=dilation,
                    dropout=float(dropout),
                    causal=bool(causal),
                )
            )
            prev_ch = out_ch

        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_ch, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

        # 记录超参，方便 repr / 调试
        self.num_blocks = len(num_channels)
        self.causal = bool(causal)
        rf = 1 + 2 * (int(kernel_size) - 1) * (2 ** self.num_blocks - 1)
        self._receptive_field: int = rf

    def receptive_field(self) -> int:
        """返回模型的理论感受野（时间步数）。"""
        return self._receptive_field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TCNClassifier 期望输入维度为 3，当前: {tuple(x.shape)}")
        x = self.tcn(x)         # [B, C, T]
        x = self.pool(x)        # [B, C, 1]
        return self.classifier(x)  # [B, num_classes]

    def extra_repr(self) -> str:
        return (
            f"num_blocks={self.num_blocks}, "
            f"causal={self.causal}, "
            f"receptive_field={self._receptive_field}"
        )


# ---------------------------------------------------------------------------
# 配置解析 & 工厂函数
# ---------------------------------------------------------------------------

def resolve_tcn_config(*, model_cfg: Any, train_cfg: Any) -> dict[str, Any]:
    """从 YAML 配置中提取 TCN 超参，并做类型规整。

    YAML 关键字段（置于 ``dl.model`` 或 ``dl.train`` 下均可）::

        tcn_num_channels: [64, 64, 128, 128]   # 每 block 输出通道（列表长度 = block 数）
        tcn_num_blocks: 4                       # 若不填 num_channels，则自动生成等宽列表
        kernel_size: 3
        dropout: 0.2
        hidden_dim: 128
        tcn_causal: false
    """
    # 优先读 tcn_num_channels（完整列表）
    raw_channels = pick_config_value(model_cfg, train_cfg, "tcn_num_channels", "num_channels")
    if raw_channels is not None and isinstance(raw_channels, (list, tuple)):
        num_channels = to_int_list(raw_channels, default=[64, 64, 64, 64])
    else:
        # 退而求其次：用 tcn_num_blocks 生成等宽列表
        num_blocks = max(
            1,
            to_int(pick_config_value(model_cfg, train_cfg, "tcn_num_blocks", "num_blocks", default=4), 4),
        )
        base_ch = max(
            8,
            to_int(pick_config_value(model_cfg, train_cfg, "tcn_base_channels", default=64), 64),
        )
        num_channels = [base_ch] * num_blocks

    return {
        "num_channels": num_channels,
        "kernel_size": max(
            2,
            to_int(pick_config_value(model_cfg, train_cfg, "kernel_size", default=3), 3),
        ),
        "dropout": to_float(
            pick_config_value(model_cfg, train_cfg, "dropout", default=0.2), 0.2
        ),
        "hidden_dim": max(
            8,
            to_int(pick_config_value(model_cfg, train_cfg, "hidden_dim", default=128), 128),
        ),
        "causal": to_bool(
            pick_config_value(model_cfg, train_cfg, "tcn_causal", "causal", default=False),
            default=False,
        ),
    }


def build_tcn_model(
    *,
    in_channels: int,
    sequence_length: int,
    num_classes: int,
    config: dict[str, Any],
) -> nn.Module:
    """工厂函数：根据解析后的配置字典构造 TCNClassifier。"""
    del sequence_length  # TCN 不依赖固定序列长度
    model = TCNClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        num_channels=config["num_channels"],
        kernel_size=int(config["kernel_size"]),
        dropout=float(config["dropout"]),
        hidden_dim=int(config["hidden_dim"]),
        causal=bool(config["causal"]),
    )
    rf = model.receptive_field()
    print(
        f"[TCN] blocks={model.num_blocks}, channels={config['num_channels']}, "
        f"kernel={config['kernel_size']}, causal={config['causal']}, "
        f"receptive_field={rf} steps"
    )
    return model
