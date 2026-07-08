from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError("运行 dl/main_train.py 前请先安装 PyTorch：pip install torch") from exc


from dl.models import build_dl_model
from dl.training.results import (
    LiveHistoryPlotter,
    save_confusion_matrix_plot,
    save_history_plot,
    _append_per_class_metric_history,
    _configure_matplotlib_for_chinese,
    _save_per_class_metric_plot,
)

from dl.training.loader import (
    DEFAULT_BATCH_FORMAT,
    DEFAULT_OUTPUT_FILE_PREFIX,
    SCHEMA_FILENAME,
    _load_feature_schema,
    _model_tag_from_arch,
    _normalize_batch_format,
    _read_yaml,
    _resolve_dataset_dir,
    _resolve_dataset_files,
    _resolve_dl_cfg,
    _resolve_model_arch,
    _resolve_results_root,
    _to_text,
)
from dl.training.split import (
    DLSplit,
    prepare_train_val_test_for_dl,
)


DEFAULT_CONFIG_PATH = Path("cfg/epump4.yaml")


class _ConsoleProgress:
    def __init__(self, *, total: int | None, desc: str, unit: str = "item") -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.count = 0
        self._last_print = 0.0
        self._closed = False

    def __enter__(self):
        print(f"{self.desc}: 开始")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def update(self, n: int = 1) -> None:
        self.count += int(n)
        now = time.time()
        should_print = (now - self._last_print) >= 0.8
        if self.total is not None and self.count >= self.total:
            should_print = True
        if not should_print:
            return
        self._last_print = now
        if self.total is None or self.total <= 0:
            print(f"{self.desc}: {self.count} {self.unit}", flush=True)
            return
        ratio = min(1.0, max(0.0, float(self.count) / float(self.total)))
        print(
            f"{self.desc}: {self.count}/{self.total} {self.unit} ({ratio * 100.0:.1f}%)",
            flush=True,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.total is None or self.total <= 0:
            print(f"{self.desc}: 完成 {self.count} {self.unit}", flush=True)
        else:
            print(f"{self.desc}: 完成 {self.count}/{self.total} {self.unit}", flush=True)

    def set_postfix(self, **kwargs) -> None:
        if not kwargs:
            return
        items = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"{self.desc}: {items}", flush=True)


def _create_progress(*, total: int | None, desc: str, unit: str = "item"):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)
    except Exception:
        return _ConsoleProgress(total=total, desc=desc, unit=unit)


class WindowFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def _resolve_device(requested: str | None) -> torch.device:
    request = _to_text(requested).lower() or "auto"
    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if request == "auto":
        if cuda_ok:
            return torch.device("cuda")
        if mps_ok:
            return torch.device("mps")
        return torch.device("cpu")
    if request == "cuda":
        if cuda_ok:
            return torch.device("cuda")
        print("[WARN] CUDA 不可用，回退到 CPU。")
        return torch.device("cpu")
    if request == "mps":
        if mps_ok:
            return torch.device("mps")
        print("[WARN] MPS 不可用，回退到 CPU。")
        return torch.device("cpu")
    return torch.device("cpu")


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fix_time_length(arr: np.ndarray, target_length: int) -> np.ndarray:
    """把 [C, T] 沿时间轴裁剪/补齐到固定 target_length：超出取前段，不足末尾补零。"""
    t = int(arr.shape[-1])
    if t == target_length:
        return arr
    if t > target_length:
        return arr[..., :target_length]
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_length - t)]
    return np.pad(arr, pad_width, mode="constant")


def _build_feature_tensor(
    df: pd.DataFrame,
    *,
    channel_names: List[str],
    columns_by_channel: Dict[str, List[str]],
    compact_feature_column: str,
    target_length: int | None = None,
) -> np.ndarray:
    if compact_feature_column in df.columns:
        tensors = [np.asarray(item, dtype=np.float32) for item in df[compact_feature_column].tolist()]
        if not tensors:
            return np.empty((0, len(channel_names), int(target_length or 0)), dtype=np.float32)
        n_ch = len(channel_names)
        fixed: List[np.ndarray] = []
        for idx, tensor in enumerate(tensors):
            if tensor.ndim != 2 or tensor.shape[0] != n_ch:
                raise RuntimeError(
                    f"紧凑特征张量形状不符: row={idx}, got={tensor.shape}, 期望通道数={n_ch}"
                )
            if target_length is not None:
                tensor = _fix_time_length(tensor, int(target_length))
            fixed.append(tensor)
        # 裁剪/补齐后各样本时间轴一致；未给 target_length 时要求本就一致
        return np.stack(fixed, axis=0).astype(np.float32, copy=False)

    channels = []
    for channel in channel_names:
        cols = columns_by_channel[channel]
        channel_df = df.reindex(columns=cols, fill_value=0.0).copy()
        for col in cols:
            if col.startswith("__missing__"):
                channel_df[col] = 0.0
        channel_arr = channel_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        channels.append(channel_arr[:, None, :])
    return np.concatenate(channels, axis=1)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    for true_label, pred_label in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= int(true_label) < num_classes and 0 <= int(pred_label) < num_classes:
            cm[int(true_label), int(pred_label)] += 1
    return cm


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, Any]:
    cm = _confusion_matrix(y_true, y_pred, num_classes=num_classes)
    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0
    precision_list: List[float] = []
    recall_list: List[float] = []
    f1_list: List[float] = []
    support_list: List[int] = []
    per_class: List[Dict[str, Any]] = []
    for idx in range(int(num_classes)):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        support = int(cm[idx, :].sum())
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        class_accuracy = recall
        f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        support_list.append(support)
        per_class.append(
            {
                "class_id": int(idx),
                "accuracy": class_accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision_list)) if precision_list else 0.0,
        "macro_recall": float(np.mean(recall_list)) if recall_list else 0.0,
        "macro_f1": float(np.mean(f1_list)) if f1_list else 0.0,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "support": support_list,
    }


def _epoch_metrics(y_true_parts, y_pred_parts, total_loss, batch_count, num_classes):
    y_true = np.concatenate(y_true_parts) if y_true_parts else np.empty(0, dtype=np.int64)
    y_pred = np.concatenate(y_pred_parts) if y_pred_parts else np.empty(0, dtype=np.int64)
    metrics = _compute_metrics(y_true, y_pred, num_classes=max(1, int(num_classes)))
    metrics["loss"] = float(total_loss / max(1, batch_count))
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    optimizer: "torch.optim.Optimizer",
    device: torch.device,
    num_classes: int,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
    desc: str = "train",
) -> Dict[str, Any]:
    """训练一个 epoch —— 教科书 5 步：zero_grad -> forward -> loss -> backward -> step。"""
    use_amp = scaler is not None and device.type == "cuda"
    model.train()
    total_loss = 0.0
    batch_count = 0
    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []
    with _create_progress(total=len(loader), desc=desc, unit="batch") as progress:
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)                    # 1. 清梯度
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(batch_x)                              # 2. 前向
                loss = criterion(logits, batch_y)                    # 3. 算损失
            if use_amp:
                scaler.scale(loss).backward()                        # 4. 反向
                scaler.step(optimizer)                               # 5. 更新
                scaler.update()
            else:
                loss.backward()                                      # 4. 反向
                optimizer.step()                                     # 5. 更新
            total_loss += float(loss.detach().item())
            batch_count += 1
            preds = torch.argmax(logits.detach(), dim=1)
            y_true_parts.append(batch_y.detach().cpu().numpy())
            y_pred_parts.append(preds.detach().cpu().numpy())
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(loss=f"{float(loss.detach().item()):.4f}")
            progress.update(1)
    return _epoch_metrics(y_true_parts, y_pred_parts, total_loss, batch_count, num_classes)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    desc: str = "eval",
) -> Dict[str, Any]:
    """评估一个 epoch —— 不更新参数，no_grad 不建计算图（省显存）。"""
    model.eval()
    total_loss = 0.0
    batch_count = 0
    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []
    with _create_progress(total=len(loader), desc=desc, unit="batch") as progress:
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.detach().item())
            batch_count += 1
            preds = torch.argmax(logits.detach(), dim=1)
            y_true_parts.append(batch_y.detach().cpu().numpy())
            y_pred_parts.append(preds.detach().cpu().numpy())
            progress.update(1)
    return _epoch_metrics(y_true_parts, y_pred_parts, total_loss, batch_count, num_classes)


# ===========================================================================
# 训练配置（纯数据容器，非编排器；参考 ml 的 train，去掉模型编排类）
# ===========================================================================

@dataclass
class TrainConfig:
    """由 cfg 解析出的训练配置 + 路径 + 超参。只持数据，不含训练逻辑。"""
    config_path: str
    cfg: Dict[str, Any]
    dl_cfg: Dict[str, Any]
    dl_train_cfg: Dict[str, Any]
    dl_model_cfg: Dict[str, Any]
    global_train_cfg: Dict[str, Any]
    model_arch: str
    model_tag: str
    results_root: Path
    dataset_dir: Path
    train_output_dir: Path
    results_path: Path
    schema: Dict[str, Any]
    output_file_prefix: str
    batch_format: str
    dataset_files: List[Path]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    use_amp: bool
    random_state: int
    split_random_state: int
    test_size: float
    val_size: float
    group_split_trials: int
    group_sample_trials: int
    seed_runs: int
    split_strategy: str
    scheduler_type: str
    early_stopping_patience: int
    loss_type: str
    label_smoothing: float
    focal_gamma: float
    length_percentile: float
    device: "torch.device"


def build_train_config(config_path: str | Path, *, cli_arch: str | None = None) -> TrainConfig:
    """解析 cfg → TrainConfig（原 CNNModel.__init__，去编排器）。"""
    config_path = str(config_path)
    cfg = _read_yaml(config_path)
    dl_cfg = _resolve_dl_cfg(cfg)
    dl_train_cfg = dl_cfg.get("train") if isinstance(dl_cfg.get("train"), dict) else {}
    dl_model_cfg = dl_cfg.get("model") if isinstance(dl_cfg.get("model"), dict) else {}
    global_train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}

    model_arch = _resolve_model_arch(cfg, cli_arch)
    model_tag = _model_tag_from_arch(model_arch)

    results_root = _resolve_results_root(cfg, None, model_arch)
    dataset_dir = _resolve_dataset_dir(cfg, None, model_arch)
    train_output_dir = results_root / "dl_train"
    train_output_dir.mkdir(parents=True, exist_ok=True)

    schema = _load_feature_schema(dataset_dir)
    extract_cfg = dl_cfg.get("extract") if isinstance(dl_cfg.get("extract"), dict) else {}
    output_file_prefix = (
        _to_text(schema.get("output_file_prefix"))
        or _to_text(extract_cfg.get("output_file_prefix"))
        or DEFAULT_OUTPUT_FILE_PREFIX
    )
    batch_format = _normalize_batch_format(
        schema.get("output_format")
        or extract_cfg.get("output_format")
        or dl_cfg.get("output_format")
        or DEFAULT_BATCH_FORMAT
    )
    dataset_files = _resolve_dataset_files(
        dataset_dir=dataset_dir,
        output_file_prefix=output_file_prefix,
        batch_format=batch_format,
    )

    epochs = int(dl_train_cfg.get("epochs") or 300)
    batch_size = int(dl_train_cfg.get("batch_size") or 256)
    if epochs <= 0:
        raise ValueError(f"epochs 必须 > 0，当前: {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须 > 0，当前: {batch_size}")
    random_state = int(dl_train_cfg.get("random_state") or global_train_cfg.get("random_state") or 42)
    scheduler_type = _to_text(dl_train_cfg.get("scheduler_type") or dl_train_cfg.get("scheduler")) or "none"
    es_cfg = dl_train_cfg.get("early_stopping")
    if isinstance(es_cfg, dict):
        early_stopping_patience = int(es_cfg.get("patience") or 0) if es_cfg.get("enable", True) else 0
    else:
        early_stopping_patience = int(dl_train_cfg.get("early_stopping_patience") or 0)
    loss_cfg = dl_train_cfg.get("loss")
    if isinstance(loss_cfg, dict):
        loss_type = _to_text(loss_cfg.get("type")) or "ce"
        label_smoothing = float(loss_cfg.get("label_smoothing") or 0.0)
        focal_gamma = float(loss_cfg.get("focal_gamma") or loss_cfg.get("gamma") or 2.0)
    else:
        loss_type = _to_text(dl_train_cfg.get("loss")) or "ce"
        label_smoothing = float(dl_train_cfg.get("label_smoothing") or 0.0)
        focal_gamma = float(dl_train_cfg.get("focal_gamma") or 2.0)

    return TrainConfig(
        config_path=config_path, cfg=cfg, dl_cfg=dl_cfg, dl_train_cfg=dl_train_cfg,
        dl_model_cfg=dl_model_cfg, global_train_cfg=global_train_cfg,
        model_arch=model_arch, model_tag=model_tag,
        results_root=results_root, dataset_dir=dataset_dir,
        train_output_dir=train_output_dir, results_path=train_output_dir,
        schema=schema, output_file_prefix=output_file_prefix, batch_format=batch_format,
        dataset_files=dataset_files,
        epochs=epochs, batch_size=batch_size,
        learning_rate=float(dl_train_cfg.get("learning_rate") or 1e-2),
        weight_decay=float(dl_train_cfg.get("weight_decay") or 1e-4),
        num_workers=int(dl_train_cfg.get("num_workers", 0)),
        use_amp=bool(dl_train_cfg.get("use_amp", True)),
        random_state=random_state, split_random_state=random_state,
        test_size=float(dl_train_cfg.get("test_size") or 0.1),
        val_size=float(dl_train_cfg.get("val_size") or 0.125),
        group_split_trials=int(dl_train_cfg.get("group_split_trials") or global_train_cfg.get("group_split_trials") or 32),
        group_sample_trials=int(dl_train_cfg.get("group_sample_trials") or global_train_cfg.get("group_sample_trials") or 24),
        seed_runs=max(1, int(dl_train_cfg.get("seed_runs") or 1)),
        split_strategy=_to_text(global_train_cfg.get("split_strategy")) or "reference_in",
        scheduler_type=scheduler_type,
        early_stopping_patience=early_stopping_patience,
        loss_type=loss_type,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        length_percentile=float(dl_train_cfg.get("length_percentile") or 0.95),
        device=_resolve_device(dl_train_cfg.get("device") or dl_cfg.get("device")),
    )


# ===========================================================================
# 训练流程函数（原 CNNModel 的方法，改为模块函数，tc 显式传入）
# ===========================================================================

# ===========================================================================
# 准备组件（循环外的小工厂，各管一件事）
# ===========================================================================

@dataclass
class DataBundle:
    """build_dataloaders 的产物：三个 loader + 训练所需元信息。"""
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    channel_names: List[str]
    columns_by_channel: Dict[str, List[str]]
    seq_len: int
    class_name_by_id: Dict[int, str]
    y_train: np.ndarray
    class_weights: "np.ndarray | None"
    eval_rest_loader: "DataLoader | None"


def _class_weights_from_split(split: DLSplit, num_classes: int) -> "np.ndarray | None":
    """从 label_mapping 各 group 的 weight 取出按 mlabel 排序的类别权重；未配置则 None。"""
    wd = getattr(split, "mlabel_weight_dict", None) or {}
    if not wd:
        return None
    return np.asarray([float(wd.get(i, 1.0)) for i in range(num_classes)], dtype=np.float32)


def _resolve_target_length(split: DLSplit, percentile: float) -> int | None:
    """按全体样本「时间轴长度的 percentile 分位」定固定读取长度（各种特征通用）。

    - 紧凑列（mel/pcen/raw 等 pickle）：取每条 [C, T] 的 T，算分位数。
      mel/pcen 本就定长 → 分位 = 该定长 → 后续裁剪/补齐为 no-op；raw 不定长 → 取 95% 覆盖长度。
    - 无紧凑列（CSV 定长特征）：返回 None，不做裁剪（本就定长）。
    """
    compact = split.compact_feature_column
    base_df = split.full_df if compact in getattr(split.full_df, "columns", []) else split.train_df
    if compact not in base_df.columns:
        return None
    lengths = [int(np.asarray(a).shape[-1]) for a in base_df[compact].tolist() if a is not None]
    if not lengths:
        return None
    target = int(np.percentile(np.asarray(lengths, dtype=np.int64), float(percentile) * 100.0))
    target = max(1, target)
    print(
        f"[INFO] 固定读取长度: target={target} (percentile={percentile:.2f}, "
        f"样本时长 min/median/max={min(lengths)}/{int(np.median(lengths))}/{max(lengths)})"
    )
    return target


def build_dataloaders(tc: TrainConfig, split: DLSplit) -> DataBundle:
    """DLSplit 的三个 DataFrame -> (X, y) 张量 -> DataLoader。"""
    channel_names = split.channel_names
    columns_by_channel = split.columns_by_channel

    target_length = _resolve_target_length(split, tc.length_percentile)
    seq_len = int(target_length) if target_length is not None else int(split.seq_len)

    def _x(df: pd.DataFrame) -> np.ndarray:
        return _build_feature_tensor(
            df, channel_names=channel_names, columns_by_channel=columns_by_channel,
            compact_feature_column=split.compact_feature_column,
            target_length=target_length,
        )

    x_train, x_val, x_test = _x(split.train_df), _x(split.val_df), _x(split.test_df)
    y_train = split.train_df["label"].to_numpy(dtype=np.int64)
    y_val = split.val_df["label"].to_numpy(dtype=np.int64)
    y_test = split.test_df["label"].to_numpy(dtype=np.int64)
    num_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    if num_classes < 2:
        raise RuntimeError(f"类别数不足 2，无法训练 DL 模型。当前 num_classes={num_classes}")

    _nw = max(0, tc.num_workers)
    _pin = tc.device.type == "cuda"
    _persistent = _nw > 0
    _prefetch = 2 if _nw > 0 else None

    def _loader(ds: "WindowFeatureDataset", bs: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle, num_workers=_nw, pin_memory=_pin,
            persistent_workers=_persistent, prefetch_factor=_prefetch,
        )

    # 评估集（val + 采样剩余）：仅用于画混淆矩阵，可能为空
    rest_df = getattr(split, "val_plus_rest_df", None)
    if rest_df is not None and not rest_df.empty:
        x_rest = _x(rest_df)
        y_rest = rest_df["label"].to_numpy(dtype=np.int64)
        eval_rest_loader = _loader(WindowFeatureDataset(x_rest, y_rest), tc.batch_size * 2, False)
    else:
        eval_rest_loader = None

    return DataBundle(
        train_loader=_loader(WindowFeatureDataset(x_train, y_train), tc.batch_size, True),
        val_loader=_loader(WindowFeatureDataset(x_val, y_val), tc.batch_size * 2, False),
        test_loader=_loader(WindowFeatureDataset(x_test, y_test), tc.batch_size * 2, False),
        num_classes=num_classes,
        channel_names=channel_names,
        columns_by_channel=columns_by_channel,
        seq_len=seq_len,
        class_name_by_id=split.class_name_by_id,
        y_train=y_train,
        class_weights=_class_weights_from_split(split, num_classes),
        eval_rest_loader=eval_rest_loader,
    )


class FocalLoss(nn.Module):
    """多分类 Focal Loss：FL = -alpha_t * (1 - p_t)^gamma * log(p_t)。

    - gamma：聚焦因子，越大越压低「已分对的容易样本」，常取 2。
    - weight：每类 alpha（与 nn.CrossEntropyLoss 的 weight 含义一致）。
    """

    def __init__(self, *, weight: "torch.Tensor | None" = None, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.weight = weight
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = nn.functional.log_softmax(logits, dim=1)
        logpt = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)   # 真实类的 log p_t
        pt = logpt.exp()
        loss = -((1.0 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight.gather(0, target)              # 乘上每样本对应类的 alpha
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def build_criterion(
    tc: TrainConfig, y_train: np.ndarray, num_classes: int, *, class_weights: "np.ndarray | None" = None,
) -> nn.Module:
    """按 cfg `dl.train.loss` 构建损失：`ce`（交叉熵 + label_smoothing）或 `focal`（Focal Loss）。

    两种损失都吃同一套类别权重：来自 `train.label_mapping.groups[].weight`（按 mlabel 对齐），
    未配置时回退为按类频率反比均衡加权（ce 的 weight / focal 的 alpha）。
    """
    if class_weights is not None:
        weights = np.asarray(class_weights, dtype=np.float32)
        if weights.shape[0] != num_classes:
            raise ValueError(f"class_weights 长度 {weights.shape[0]} != num_classes {num_classes}")
        source = "groups.weight"
    else:
        counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
        weights = np.ones(num_classes, dtype=np.float32)
        valid = counts > 0
        weights[valid] = float(len(y_train)) / (float(num_classes) * counts[valid])
        source = "balanced(auto)"
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=tc.device)

    loss_type = (tc.loss_type or "ce").strip().lower()
    if loss_type in ("ce", "crossentropy", "cross_entropy"):
        print(f"[INFO] loss=ce | label_smoothing={tc.label_smoothing} | weight({source})={np.round(weights, 4).tolist()}")
        return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=float(tc.label_smoothing))
    if loss_type in ("focal", "focalloss", "focal_loss"):
        print(f"[INFO] loss=focal | gamma={tc.focal_gamma} | alpha({source})={np.round(weights, 4).tolist()}")
        return FocalLoss(weight=weight_tensor, gamma=float(tc.focal_gamma))
    raise ValueError(f"不支持的 loss: {tc.loss_type}（支持 ce / focal）")


def build_scheduler(tc: TrainConfig, optimizer: "torch.optim.Optimizer"):
    """按 cfg 构建学习率调度器；默认 none。返回 (scheduler|None, needs_val_metric)。"""
    t = (tc.scheduler_type or "none").strip().lower()
    if t in ("", "none", "off"):
        return None, False
    if t in ("cosine", "cosineannealing", "cosineannealinglr"):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, tc.epochs)), False
    if t in ("step", "steplr"):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, tc.epochs // 3), gamma=0.1), False
    if t in ("plateau", "reduce", "reducelronplateau"):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5), True
    raise ValueError(f"不支持的 scheduler: {tc.scheduler_type}（可选 none/cosine/step/plateau）")


class EarlyStopper:
    """监控 val_loss（越小越好）：连续 patience 轮无改善则停。patience<=0 关闭。"""

    def __init__(self, patience: int, *, min_delta: float = 0.0) -> None:
        self.patience = int(patience)
        self.enabled = self.patience > 0
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.count = 0

    def step(self, value: float) -> bool:
        if not self.enabled:
            return False
        if value < self.best - self.min_delta:
            self.best = value
            self.count = 0
        else:
            self.count += 1
        return self.count >= self.patience


def _make_history_row(epoch: int, tr: Dict[str, Any], va: Dict[str, Any], num_classes: int) -> Dict[str, Any]:
    row = {
        "epoch": int(epoch),
        "train_loss": float(tr["loss"]), "val_loss": float(va["loss"]),
        "train_accuracy": float(tr["accuracy"]), "val_accuracy": float(va["accuracy"]),
        "train_macro_f1": float(tr["macro_f1"]), "val_macro_f1": float(va["macro_f1"]),
    }
    for prefix, metrics in (("train", tr), ("val", va)):
        _append_per_class_metric_history(history_row=row, prefix=prefix, metrics=metrics, num_classes=num_classes, metric_key="accuracy", column_suffix="class_acc")
        _append_per_class_metric_history(history_row=row, prefix=prefix, metrics=metrics, num_classes=num_classes, metric_key="recall", column_suffix="class_recall")
    return row


def _is_better(va: Dict[str, Any], best: Dict[str, Any]) -> bool:
    """优于当前最优：val macro-f1 更高；平局则 val loss 更低。"""
    f1, loss = float(va["macro_f1"]), float(va["loss"])
    if f1 > best["score"] + 1e-8:
        return True
    return abs(f1 - best["score"]) <= 1e-8 and loss < best["val_loss"]


# ===========================================================================
# 单次完整训练（教科书骨架：准备 -> epoch 循环 -> 返回最优）
# ===========================================================================

def fit_one_run(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: "torch.optim.Optimizer",
    scheduler: Any,
    sched_needs_metric: bool,
    scaler: "torch.cuda.amp.GradScaler | None",
    stopper: "EarlyStopper",
    plotter: "LiveHistoryPlotter | None",
    num_classes: int,
    epochs: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """纯 epoch 循环（完全依赖注入）：每轮 train + eval，做调度 / 早停 / 最优记录。

    所有组件（model/criterion/optimizer/scheduler/scaler/stopper/plotter）均由
    调用方 run_training 建好传入。返回最优结果（best_state / best_epoch /
    best_score / history_rows）；seed / model / model_arch 等由调用方附加。
    """
    best: Dict[str, Any] = {"state": None, "epoch": -1, "score": float("-inf"), "val_loss": float("inf")}
    history_rows: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        print(f"\n[seed {seed}] Epoch {epoch}/{epochs}")
        tr = train_one_epoch(model, train_loader, criterion=criterion, optimizer=optimizer, device=device, num_classes=num_classes, scaler=scaler, desc=f"train epoch {epoch}")
        va = evaluate(model, val_loader, criterion=criterion, device=device, num_classes=num_classes, desc=f"valid epoch {epoch}")
        if scheduler is not None:
            scheduler.step(va["loss"]) if sched_needs_metric else scheduler.step()

        history_rows.append(_make_history_row(epoch, tr, va, num_classes))
        print(f"epoch={epoch} | train_loss={tr['loss']:.4f}, train_f1={tr['macro_f1']:.4f} | val_loss={va['loss']:.4f}, val_f1={va['macro_f1']:.4f}")

        if _is_better(va, best):
            best.update(state={"model_state": model.state_dict(), "epoch": int(epoch)}, epoch=int(epoch), score=float(va["macro_f1"]), val_loss=float(va["loss"]))

        if plotter is not None:
            plotter.update(pd.DataFrame(history_rows), best_epoch=best["epoch"] if best["epoch"] > 0 else None, best_score=best["score"] if np.isfinite(best["score"]) else None)

        if stopper.step(float(va["loss"])):
            print(f"[INFO] 早停：val_loss 连续 {stopper.patience} 轮未改善，停在 epoch={epoch}")
            break

    if plotter is not None:
        plotter.close()
    if best["state"] is None:
        raise RuntimeError("训练未产生有效模型状态。")

    return {
        "best_state": best["state"], "best_epoch": int(best["epoch"]),
        "best_score": float(best["score"]), "history_rows": history_rows,
    }


def _save_curves(out_dir: Path, *, history_rows: List[Dict[str, Any]], class_name_by_id: Dict[int, str]) -> None:
    """落盘训练曲线：history.csv + loss/acc/f1 三联图 + 逐类别 acc/recall 曲线。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")
    save_history_plot(history_df, out_dir / "training_curve.png")
    _save_per_class_metric_plot(history_df=history_df, class_name_by_id=class_name_by_id, out_path=out_dir / "per_class_accuracy_curve.png", column_suffix="class_acc", ylabel="Accuracy")
    _save_per_class_metric_plot(history_df=history_df, class_name_by_id=class_name_by_id, out_path=out_dir / "per_class_recall_curve.png", column_suffix="class_recall", ylabel="Recall")


def _save_confusion_matrices(
    out_dir: Path,
    *,
    model: nn.Module,
    data: DataBundle,
    criterion: nn.Module,
    tc: TrainConfig,
    train_eval: Dict[str, Any] | None = None,
    test_eval: Dict[str, Any] | None = None,
    tag: str = "",
) -> None:
    """画三张计数版混淆矩阵：train / test / (val + 采样剩余)。已算过的 eval 可传入复用。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    te = train_eval if train_eval is not None else evaluate(model, data.train_loader, criterion=criterion, device=tc.device, num_classes=data.num_classes, desc=f"{tag}train cm")
    save_confusion_matrix_plot(te["confusion_matrix"], data.class_name_by_id, out_dir / "confusion_matrix_train.png", title="Confusion Matrix - Train")
    tt = test_eval if test_eval is not None else evaluate(model, data.test_loader, criterion=criterion, device=tc.device, num_classes=data.num_classes, desc=f"{tag}test cm")
    save_confusion_matrix_plot(tt["confusion_matrix"], data.class_name_by_id, out_dir / "confusion_matrix_test.png", title="Confusion Matrix - Test")
    if data.eval_rest_loader is not None:
        rest_eval = evaluate(model, data.eval_rest_loader, criterion=criterion, device=tc.device, num_classes=data.num_classes, desc=f"{tag}val+rest cm")
        save_confusion_matrix_plot(rest_eval["confusion_matrix"], data.class_name_by_id, out_dir / "confusion_matrix_val_plus_rest.png", title="Confusion Matrix - Val+Rest")
    else:
        print("[WARN] val+采样剩余集为空，跳过该混淆矩阵。")


# ===========================================================================
# 顶层训练编排：多 seed -> 选最优 -> test 评估 -> 落盘
# ===========================================================================

def run_training(tc: TrainConfig, split: DLSplit) -> Dict[str, Any]:
    wall_t0 = time.perf_counter()
    selected_font = _configure_matplotlib_for_chinese()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    data = build_dataloaders(tc, split)
    criterion = build_criterion(tc, data.y_train, data.num_classes, class_weights=data.class_weights)

    print(f"特征目录: {tc.dataset_dir}")
    print(f"训练输出目录: {tc.train_output_dir}")
    print(f"数据文件数: {len(tc.dataset_files)}")
    print(f"device: {tc.device}")
    print(f"model_arch: {tc.model_arch} ({tc.model_tag})")
    print("matplotlib 中文字体: " + (selected_font or "[未找到，中文可能显示为方框]"))
    print(f"epochs={tc.epochs}, batch_size={tc.batch_size}, lr={tc.learning_rate}, weight_decay={tc.weight_decay}, "
          f"seed_runs={tc.seed_runs}, scheduler={tc.scheduler_type}, early_stop_patience={tc.early_stopping_patience}")

    # ---------------- 多 seed 复跑，取验证集 macro_f1 最优 ----------------
    seeds = [tc.random_state + i for i in range(tc.seed_runs)]
    best_run: Dict[str, Any] | None = None
    seed_overview_rows: List[Dict[str, Any]] = []
    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n===== seed run {run_idx}/{tc.seed_runs} (seed={seed}) =====")
        # ---- 每个 seed 的一次性组件（必须各自新建：model 重新随机初始化、
        #      optimizer/scheduler 绑定该 model、scaler/stopper 有状态需重置）----
        _set_random_seed(int(seed))
        model_arch, model, resolved_model_config = build_dl_model(
            arch=tc.model_arch, in_channels=len(data.channel_names), sequence_length=data.seq_len,
            num_classes=data.num_classes, model_cfg=tc.dl_model_cfg, train_cfg=tc.dl_train_cfg,
        )
        model = model.to(tc.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay)
        scheduler, sched_needs_metric = build_scheduler(tc, optimizer)
        stopper = EarlyStopper(tc.early_stopping_patience)
        scaler = torch.cuda.amp.GradScaler() if tc.use_amp and tc.device.type == "cuda" else None
        if scaler is not None:
            print("[INFO] AMP (混合精度) 已启用")
        plotter = LiveHistoryPlotter(out_path=tc.train_output_dir / "training_curve.png") if tc.seed_runs == 1 else None

        run = fit_one_run(
            model,
            train_loader=data.train_loader, val_loader=data.val_loader,
            criterion=criterion, optimizer=optimizer,
            scheduler=scheduler, sched_needs_metric=sched_needs_metric,
            scaler=scaler, stopper=stopper, plotter=plotter,
            num_classes=data.num_classes, epochs=tc.epochs, device=tc.device, seed=seed,
        )
        result = {"seed": int(seed), "model": model, "model_arch": model_arch,
                  "resolved_model_config": resolved_model_config, **run}
        seed_overview_rows.append({"seed": int(seed), "best_epoch": int(result["best_epoch"]), "val_macro_f1": float(result["best_score"])})

        # 每个 seed 各自落盘（多 seed 时）：曲线 + 逐类别 + 混淆矩阵，存到 seed_{seed}/ 子目录
        if tc.seed_runs > 1:
            seed_dir = tc.train_output_dir / f"seed_{seed}"
            model.load_state_dict(run["best_state"]["model_state"])  # 载入该 seed 的最优权重
            _save_curves(seed_dir, history_rows=run["history_rows"], class_name_by_id=data.class_name_by_id)
            _save_confusion_matrices(seed_dir, model=model, data=data, criterion=criterion, tc=tc, tag=f"seed{seed} ")
            print(f"[INFO] seed={seed} 结果已保存: {seed_dir} (best_epoch={result['best_epoch']}, val_macro_f1={result['best_score']:.4f})")

        if best_run is None or result["best_score"] > best_run["best_score"] + 1e-8:
            best_run = result
    assert best_run is not None

    best_seed = int(best_run["seed"])
    best_epoch = int(best_run["best_epoch"])
    best_score = float(best_run["best_score"])
    model = best_run["model"]
    model.load_state_dict(best_run["best_state"]["model_state"])
    print(f"\n使用最佳模型评估: seed={best_seed}, epoch={best_epoch}, val_macro_f1={best_score:.4f}")
    train_eval = evaluate(model, data.train_loader, criterion=criterion, device=tc.device, num_classes=data.num_classes, desc="best train eval")
    val_eval = evaluate(model, data.val_loader, criterion=criterion, device=tc.device, num_classes=data.num_classes, desc="best valid eval")
    test_eval = evaluate(model, data.test_loader, criterion=criterion, device=tc.device, num_classes=data.num_classes, desc="best test eval")

    # 最优模型：三个集合的混淆矩阵（计数版），存训练输出根目录（复用已算的 train/test eval）
    _save_confusion_matrices(tc.train_output_dir, model=model, data=data, criterion=criterion, tc=tc,
                             train_eval=train_eval, test_eval=test_eval, tag="best ")

    _save_outputs(
        tc, split=split, model=model, history_rows=best_run["history_rows"],
        class_name_by_id=data.class_name_by_id, channel_names=data.channel_names,
        columns_by_channel=data.columns_by_channel, seq_len=data.seq_len, num_classes=data.num_classes,
        resolved_model_config=best_run["resolved_model_config"], best_epoch=best_epoch,
        train_eval=train_eval, val_eval=val_eval, test_eval=test_eval,
        seed_overview_rows=seed_overview_rows, best_seed=best_seed,
    )

    total_sec = float(time.perf_counter() - wall_t0)
    print(f"\nDL 训练完成: model_arch={tc.model_arch}, best_seed={best_seed}, best_epoch={best_epoch}")
    print(f"train_output_dir: {tc.train_output_dir}")
    print(f"test metrics: accuracy={test_eval['accuracy']:.4f}, macro_precision={test_eval['macro_precision']:.4f}, "
          f"macro_recall={test_eval['macro_recall']:.4f}, macro_f1={test_eval['macro_f1']:.4f}")
    print(f"总耗时: {total_sec:.3f}s")
    return {"test": test_eval, "validation": val_eval, "train": train_eval, "best_seed": best_seed}


def _save_outputs(
    tc,
    *,
    split: DLSplit,
    model: nn.Module,
    history_rows: List[Dict[str, Any]],
    class_name_by_id: Dict[int, str],
    channel_names: List[str],
    columns_by_channel: Dict[str, List[str]],
    seq_len: int,
    num_classes: int,
    resolved_model_config: Dict[str, Any],
    best_epoch: int,
    train_eval: Dict[str, Any],
    val_eval: Dict[str, Any],
    test_eval: Dict[str, Any],
    seed_overview_rows: List[Dict[str, Any]],
    best_seed: int,
) -> None:
    out_dir = tc.train_output_dir
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")
    save_history_plot(history_df, out_dir / "training_curve.png")
    _save_per_class_metric_plot(history_df=history_df, class_name_by_id=class_name_by_id, out_path=out_dir / "per_class_accuracy_curve.png", column_suffix="class_acc", ylabel="Accuracy")
    _save_per_class_metric_plot(history_df=history_df, class_name_by_id=class_name_by_id, out_path=out_dir / "per_class_recall_curve.png", column_suffix="class_recall", ylabel="Recall")

    if len(seed_overview_rows) > 1:
        pd.DataFrame(seed_overview_rows).to_csv(out_dir / "seed_overview.csv", index=False, encoding="utf-8-sig")

    model_config_payload = {"model_arch": tc.model_arch, "model_tag": tc.model_tag, **resolved_model_config}

    checkpoint = {
        "model_state": model.state_dict(),
        "model_arch": tc.model_arch,
        "model_config": model_config_payload,
        "channel_names": channel_names,
        "feature_columns": columns_by_channel,
        "sequence_length": int(seq_len),
        "num_classes": int(num_classes),
        "mlabel_mtype_dict": {int(k): str(v) for k, v in split.mlabel_mtype_dict.items()},
        "class_names": class_name_by_id,
        "mlabel_to_reason_ids": {int(k): [int(x) for x in v] for k, v in split.mlabel_to_reason_ids.items()},
        "device": str(tc.device),
        "best_epoch": int(best_epoch),
        "best_seed": int(best_seed),
    }
    torch.save(checkpoint, out_dir / "best_model.pt")

    shutil.copy2(Path(tc.config_path).expanduser(), out_dir / "train_config.yaml")
    schema_path = tc.dataset_dir / SCHEMA_FILENAME
    if schema_path.exists():
        shutil.copy2(schema_path, out_dir / SCHEMA_FILENAME)

    split_summary = {
        "train_rows": int(len(split.train_df)),
        "val_rows": int(len(split.val_df)),
        "test_rows": int(len(split.test_df)),
        "train_label_counts": {int(k): int(v) for k, v in split.train_df["label"].value_counts().sort_index().items()},
        "val_label_counts": {int(k): int(v) for k, v in split.val_df["label"].value_counts().sort_index().items()},
        "test_label_counts": {int(k): int(v) for k, v in split.test_df["label"].value_counts().sort_index().items()},
    }
    with (out_dir / "split_summary.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(split_summary, f, allow_unicode=True, sort_keys=False)

    runtime_meta = {
        "model_arch": tc.model_arch,
        "model_config": model_config_payload,
        "mlabel_mtype_dict": {int(k): str(v) for k, v in split.mlabel_mtype_dict.items()},
        "class_names": class_name_by_id,
        "label_to_mlabel_map": {int(k): int(v) for k, v in split.label_to_mlabel_map.items()},
        "label_sample_dict": {int(k): int(v) for k, v in split.label_sample_dict.items()},
        "mlabel_to_reason_ids": {int(k): [int(x) for x in v] for k, v in split.mlabel_to_reason_ids.items()},
    }
    with (out_dir / "label_runtime.json").open("w", encoding="utf-8") as f:
        json.dump(runtime_meta, f, ensure_ascii=False, indent=2)

    metrics_payload = {
        "device": str(tc.device),
        "best_epoch": int(best_epoch),
        "best_seed": int(best_seed),
        "model_arch": tc.model_arch,
        "model_config": model_config_payload,
        "class_names": class_name_by_id,
        "feature_channels": list(channel_names),
        "feature_sequence_length": int(seq_len),
        "train": train_eval,
        "validation": val_eval,
        "test": test_eval,
    }
    with (out_dir / "metrics.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(metrics_payload, f, allow_unicode=True, sort_keys=False)


# ===========================================================================
# 主流程 + CLI
# ===========================================================================

def train_from_config(config_path: str | Path, *, cli_arch: str | None = None) -> Dict[str, Any]:
    """DL 训练主流程：加载+划分数据集（调 training.split）→ 训练 → 落盘结果。"""
    tc = build_train_config(config_path, cli_arch=cli_arch)
    print(f"[INFO] DL 训练: model_arch={tc.model_arch} ({tc.model_tag}), config={tc.config_path}")
    split = prepare_train_val_test_for_dl(
        tc, random_state=tc.split_random_state, rebalance_train=True,
    )
    return run_training(tc, split)


def _parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 dl 模型")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=f"YAML 配置路径（默认: {DEFAULT_CONFIG_PATH}）")
    parser.add_argument("--model-type", default=None, help="DL 架构（cnn1d/lstm/tcn...），默认读 dl.model_type")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[WARN] 忽略未识别参数（请改用 config 配置）: {unknown}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_train_args(argv)
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    train_from_config(config_path, cli_arch=args.model_type)


if __name__ == "__main__":
    main()
