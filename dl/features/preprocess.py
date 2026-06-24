"""信号统一预处理（所有提取器共用）。

- ``trim_edges``：裁剪原始信号头尾各 8000 点（去掉启停瞬态）。
- ``downsample``：每隔 step 个点取一个（抽取降采样），缩短 raw 序列。
- ``standardize``：每条信号内部 z-score 标准化 (x-mean)/(std+eps)。

参数用本模块默认值，暂不从外部 cfg 改。
"""
from __future__ import annotations

import numpy as np

DEFAULT_TRIM_HEAD = 8000
DEFAULT_TRIM_TAIL = 8000
DEFAULT_DOWNSAMPLE_STEP = 5  # raw 降采样：每隔 5 个点取一个（20kHz -> 4kHz）
DEFAULT_STD_EPS = 1e-8


def trim_edges(
    signal: np.ndarray,
    head: int = DEFAULT_TRIM_HEAD,
    tail: int = DEFAULT_TRIM_TAIL,
) -> np.ndarray:
    """裁剪信号头尾各 head/tail 个点；信号太短（<= head+tail）则原样返回，避免变空。"""
    sig = np.asarray(signal).reshape(-1)
    h = max(0, int(head))
    t = max(0, int(tail))
    if sig.size > h + t:
        return sig[h: sig.size - t]
    return sig


def downsample(signal: np.ndarray, step: int = DEFAULT_DOWNSAMPLE_STEP) -> np.ndarray:
    """抽取降采样：每隔 step 个点取一个（signal[::step]）。step<=1 时原样返回。"""
    sig = np.asarray(signal).reshape(-1)
    s = int(step)
    if s <= 1:
        return sig
    return sig[::s]


def standardize(signal: np.ndarray, eps: float = DEFAULT_STD_EPS) -> np.ndarray:
    """每条信号 z-score 标准化：(x - mean) / (std + eps)。"""
    sig = np.asarray(signal, dtype=np.float32).reshape(-1)
    if sig.size == 0:
        return sig
    mean = float(sig.mean())
    std = float(sig.std())
    return ((sig - mean) / (std + float(eps))).astype(np.float32, copy=False)
