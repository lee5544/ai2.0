"""DL 训练结果与可视化（绘图集中在此，与训练流程解耦）。

职责
----
- 训练曲线绘制：loss / accuracy / macro-f1 三联图、实时刷新（复用
  ``dl.models.base_model`` 的底层绘制，保证与模型自带 ``plot_loss`` 风格一致）。
- 逐类别指标曲线（accuracy / recall）绘制与历史行装配。
- 中文字体配置。

``train.py`` 只负责训练流程，所有“结果落盘 + 画图”统一从本模块导入。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

# 训练曲线（loss/accuracy/macro-f1）与实时绘制器复用模型层基础实现，
# 此处统一对外导出，使 train.py 只需从 dl.results 取绘图能力。
from dl.models.base_model import (
    LiveHistoryPlotter,
    draw_history_axes,
    save_history_plot,
    save_loss_plot,
)

__all__ = [
    "LiveHistoryPlotter",
    "draw_history_axes",
    "save_history_plot",
    "save_loss_plot",
    "configure_matplotlib_for_chinese",
    "append_per_class_metric_history",
    "save_per_class_metric_plot",
    "save_confusion_matrix_plot",
    # 兼容下划线别名（与原 train.py 调用名一致）
    "_configure_matplotlib_for_chinese",
    "_append_per_class_metric_history",
    "_save_per_class_metric_plot",
]


def configure_matplotlib_for_chinese() -> str:
    plt.rcParams["axes.unicode_minus"] = False
    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "STHeiti",
        "Heiti SC",
    ]
    try:
        available = {font.name for font in font_manager.fontManager.ttflist}
    except Exception:
        available = set()
    selected = [name for name in candidates if name in available]
    if selected:
        current = plt.rcParams.get("font.sans-serif", [])
        plt.rcParams["font.family"] = ["sans-serif"]
        plt.rcParams["font.sans-serif"] = selected + [x for x in current if x not in selected]
        return selected[0]
    return ""


def append_per_class_metric_history(
    *,
    history_row: Dict[str, Any],
    prefix: str,
    metrics: Dict[str, Any],
    num_classes: int,
    metric_key: str,
    column_suffix: str,
) -> None:
    per_class = metrics.get("per_class") if isinstance(metrics.get("per_class"), list) else []
    metric_by_class = {
        int(item.get("class_id")): float(item.get(metric_key, 0.0))
        for item in per_class
        if isinstance(item, dict)
    }
    for class_id in range(int(num_classes)):
        history_row[f"{prefix}_{column_suffix}_{class_id}"] = float(metric_by_class.get(class_id, 0.0))


def save_per_class_metric_plot(
    *,
    history_df: pd.DataFrame,
    class_name_by_id: Dict[int, str],
    out_path: Path,
    column_suffix: str,
    ylabel: str,
) -> None:
    if history_df.empty or not class_name_by_id:
        return

    class_ids = sorted(int(k) for k in class_name_by_id.keys())
    n_classes = len(class_ids)
    ncols = 2 if n_classes > 1 else 1
    nrows = int(np.ceil(n_classes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for plot_idx, class_id in enumerate(class_ids):
        ax = axes_arr[plot_idx]
        train_col = f"train_{column_suffix}_{class_id}"
        val_col = f"val_{column_suffix}_{class_id}"
        if train_col in history_df.columns:
            ax.plot(history_df["epoch"], history_df[train_col], label="train")
        if val_col in history_df.columns:
            ax.plot(history_df["epoch"], history_df[val_col], label="val")
        ax.set_title(f"class {class_id}: {class_name_by_id[class_id]}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend()

    for idx in range(n_classes, len(axes_arr)):
        axes_arr[idx].axis("off")

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_confusion_matrix_plot(
    cm: Any,
    class_name_by_id: Dict[int, str],
    out_path: Path,
    *,
    title: str = "Confusion Matrix",
) -> None:
    """绘制并保存计数版混淆矩阵（行=真实类，列=预测类）。"""
    cm = np.asarray(cm, dtype=np.int64)
    n = int(cm.shape[0])
    if n == 0:
        return
    labels = [str(class_name_by_id.get(i, i)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(4.0, n * 0.9 + 2.0), max(3.5, n * 0.9 + 1.5)), constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=9,
            )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# 下划线别名：与原 train.py 内部命名保持一致，避免改动调用点。
_configure_matplotlib_for_chinese = configure_matplotlib_for_chinese
_append_per_class_metric_history = append_per_class_metric_history
_save_per_class_metric_plot = save_per_class_metric_plot
