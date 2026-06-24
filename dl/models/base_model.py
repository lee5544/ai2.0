"""深度学习模型公共基类。

设计目标
========
- 为 ``models/`` 下所有时序分类模型提供 **统一的公共接口**，消除各模型中
  重复的 ``in_channels`` / ``num_classes`` 合法性校验与 ``[B, C, T]`` 输入维度
  校验。
- 子类只需实现 :meth:`BaseModel._forward_impl`，无需再各自做形状检查。
- 内置 **训练历史记录** 与 **loss / 指标曲线绘制** 能力（``plot_loss`` /
  ``plot_history`` / :class:`LiveHistoryPlotter`），train_core 与各模型均可复用。

输入张量约定
============
所有模型统一接收 ``[batch, channels, steps]`` 形状的张量；需要其它布局
（如 LSTM 的 ``[B, T, C]`` 或 2D 卷积的 ``[B, 1, F, T]``）的模型在
``_forward_impl`` 内部自行转换。
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch import nn

# matplotlib / pandas / numpy 为可选依赖：仅在真正绘图时才需要。
# 推理 / 训练前向不依赖它们，缺失时也能正常构建模型。
try:  # pragma: no cover - 仅在绘图环境缺失时触发
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg") if not matplotlib.get_backend() else None  # type: ignore
    import matplotlib.pyplot as plt

    _PLOTTING_AVAILABLE = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    pd = None  # type: ignore
    plt = None  # type: ignore
    _PLOTTING_AVAILABLE = False


# ---------------------------------------------------------------------------
# 绘图规格（loss / accuracy / macro-f1 三联图）
# ---------------------------------------------------------------------------

# (子图标题, train 列名, val 列名, y 轴范围)
HISTORY_PLOT_SPECS: tuple[tuple[str, str, str, tuple[float, float] | None], ...] = (
    ("Loss", "train_loss", "val_loss", None),
    ("Accuracy", "train_accuracy", "val_accuracy", (0.0, 1.05)),
    ("Macro F1", "train_macro_f1", "val_macro_f1", (0.0, 1.05)),
)


def _require_plotting() -> None:
    if not _PLOTTING_AVAILABLE:
        raise RuntimeError(
            "绘图功能需要 matplotlib / pandas / numpy，但当前环境未安装。"
        )


def _as_dataframe(history: Any) -> "pd.DataFrame":
    """把多种历史表示统一成 DataFrame。"""
    _require_plotting()
    if isinstance(history, pd.DataFrame):
        return history
    if isinstance(history, Sequence):
        return pd.DataFrame(list(history))
    raise TypeError(f"无法解析的 history 类型: {type(history)!r}")


def draw_history_axes(axes: Any, history_df: "pd.DataFrame") -> None:
    """在给定的一组子图坐标轴上绘制 loss / accuracy / macro-f1 曲线。

    该函数被 :meth:`BaseModel.plot_history`、:class:`LiveHistoryPlotter`
    以及 train_core 的兼容封装共同复用，确保各处曲线风格完全一致。
    """
    _require_plotting()
    axes_arr = np.atleast_1d(axes).reshape(-1)
    epochs = history_df["epoch"] if "epoch" in history_df.columns else pd.Series(dtype=np.int64)

    for ax, (title, train_col, val_col, ylim) in zip(axes_arr, HISTORY_PLOT_SPECS):
        ax.clear()
        if train_col in history_df.columns:
            ax.plot(epochs, history_df[train_col], label="train", linewidth=2.0)
        if val_col in history_df.columns:
            ax.plot(epochs, history_df[val_col], label="val", linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend()


def save_history_plot(history: Any, out_path: str | Path) -> Path:
    """把完整训练历史（loss + accuracy + macro-f1）保存为一张三联图。"""
    _require_plotting()
    history_df = _as_dataframe(history)
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)
    draw_history_axes(np.asarray(axes), history_df)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_loss_plot(history: Any, out_path: str | Path) -> Path:
    """只绘制 train / val loss 单图。"""
    _require_plotting()
    history_df = _as_dataframe(history)
    fig, ax = plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True)
    epochs = history_df["epoch"] if "epoch" in history_df.columns else range(len(history_df))
    if "train_loss" in history_df.columns:
        ax.plot(epochs, history_df["train_loss"], label="train", linewidth=2.0)
    if "val_loss" in history_df.columns:
        ax.plot(epochs, history_df["val_loss"], label="val", linewidth=2.0)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


class LiveHistoryPlotter:
    """训练过程中实时刷新的曲线绘制器。

    每个 epoch 调用 :meth:`update` 即可增量重绘 loss / accuracy / macro-f1，
    并把最新图像写入 ``out_path``；若检测到图形界面则同时在窗口实时显示。
    """

    def __init__(self, *, out_path: str | Path) -> None:
        _require_plotting()
        import os

        self.out_path = Path(out_path)
        self.fig: Any = None
        self.axes: Any = None
        backend = str(plt.get_backend()).lower()
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        self.show_window = has_display and "agg" not in backend

    def _ensure_figure(self):
        if self.fig is None or self.axes is None:
            if self.show_window:
                plt.ion()
            self.fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)
            self.axes = np.atleast_1d(axes).reshape(-1)
            if self.show_window:
                self.fig.show()
        return self.fig, self.axes

    def update(
        self,
        history: Any,
        *,
        best_epoch: int | None = None,
        best_score: float | None = None,
    ) -> None:
        history_df = _as_dataframe(history)
        if history_df.empty:
            return
        fig, axes = self._ensure_figure()
        draw_history_axes(axes, history_df)
        title = f"Training Progress ({len(history_df)} epochs)"
        if best_epoch is not None and best_epoch > 0 and best_score is not None:
            title = f"{title} | best_epoch={best_epoch}, best_val_f1={best_score:.4f}"
        fig.suptitle(title)
        fig.savefig(self.out_path, dpi=160)
        if self.show_window:
            fig.canvas.draw_idle()
            plt.show(block=False)
            plt.pause(0.001)

    def close(self) -> None:
        if self.fig is None:
            return
        if self.show_window:
            plt.ioff()
        plt.close(self.fig)
        self.fig = None
        self.axes = None


# ---------------------------------------------------------------------------
# 模型基类
# ---------------------------------------------------------------------------

class BaseModel(nn.Module, abc.ABC):
    """所有时序分类模型的公共基类。

    子类约定
    --------
    1. 在 ``__init__`` 中先调用 ``super().__init__(arch_name=..., in_channels=...,
       num_classes=..., hyperparams=...)``，由基类完成合法性校验与元数据登记。
    2. 实现 :meth:`_forward_impl`，输入已保证为 3 维 ``[B, C, T]`` 张量；
       无需再做形状检查。

    公共接口
    --------
    - :meth:`forward`            —— 统一入口，做输入维度校验后转交 ``_forward_impl``。
    - :attr:`arch_name`          —— 架构名（如 ``"cnn1d"``）。
    - :meth:`hyperparameters`    —— 返回该模型的超参字典。
    - :meth:`num_parameters`     —— 可训练参数量。
    - :meth:`describe`           —— 结构概览字典，便于日志 / 落盘。
    - 训练历史：:meth:`record_epoch` / :meth:`history_frame` /
      :meth:`reset_history`。
    - 绘图：:meth:`plot_loss` / :meth:`plot_history` /
      :meth:`make_live_plotter`。
    """

    #: 子类可覆盖的默认架构名；亦可通过构造参数 arch_name 传入。
    arch_name: str = "base"

    def __init__(
        self,
        *,
        arch_name: str | None = None,
        in_channels: int,
        num_classes: int,
        hyperparams: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if arch_name is not None:
            self.arch_name = str(arch_name)
        in_channels = int(in_channels)
        num_classes = int(num_classes)
        if in_channels <= 0:
            raise ValueError(f"in_channels 必须 > 0，当前: {in_channels}")
        if num_classes <= 1:
            raise ValueError(f"num_classes 必须 > 1，当前: {num_classes}")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self._hyperparams: Dict[str, Any] = dict(hyperparams or {})
        self._history_rows: List[Dict[str, Any]] = []

    # -- 前向 ---------------------------------------------------------------
    @abc.abstractmethod
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """子类实现的真正前向逻辑；输入保证为 ``[B, C, T]``。"""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"{type(self).__name__} 期望输入维度为 3 [B, C, T]，当前: {tuple(x.shape)}"
            )
        return self._forward_impl(x)

    # -- 元数据 -------------------------------------------------------------
    def hyperparameters(self) -> Dict[str, Any]:
        return dict(self._hyperparams)

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return int(sum(p.numel() for p in params if p.requires_grad))
        return int(sum(p.numel() for p in params))

    def describe(self) -> Dict[str, Any]:
        return {
            "arch": self.arch_name,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "num_parameters": self.num_parameters(),
            "hyperparams": self.hyperparameters(),
        }

    # -- 训练历史 -----------------------------------------------------------
    def reset_history(self) -> None:
        """清空已记录的训练历史。"""
        self._history_rows = []

    def record_epoch(self, row: Mapping[str, Any]) -> None:
        """追加一个 epoch 的指标行（至少应含 ``epoch`` / ``train_loss`` / ``val_loss``）。"""
        self._history_rows.append(dict(row))

    @property
    def history_rows(self) -> List[Dict[str, Any]]:
        return list(self._history_rows)

    def history_frame(self) -> "pd.DataFrame":
        _require_plotting()
        return pd.DataFrame(self._history_rows)

    # -- 绘图 ---------------------------------------------------------------
    def plot_loss(self, out_path: str | Path, *, history: Any = None) -> Path:
        """绘制 train / val loss 单图。

        ``history`` 缺省时使用本模型内部 :meth:`record_epoch` 累积的历史。
        """
        return save_loss_plot(history if history is not None else self._history_rows, out_path)

    def plot_history(self, out_path: str | Path, *, history: Any = None) -> Path:
        """绘制 loss + accuracy + macro-f1 三联图。"""
        return save_history_plot(history if history is not None else self._history_rows, out_path)

    @staticmethod
    def make_live_plotter(out_path: str | Path) -> LiveHistoryPlotter:
        """创建一个实时曲线绘制器。"""
        return LiveHistoryPlotter(out_path=out_path)

    def extra_repr(self) -> str:
        return (
            f"arch={self.arch_name}, in_channels={self.in_channels}, "
            f"num_classes={self.num_classes}"
        )


__all__ = [
    "BaseModel",
    "LiveHistoryPlotter",
    "HISTORY_PLOT_SPECS",
    "draw_history_axes",
    "save_history_plot",
    "save_loss_plot",
]
