"""信号统一预处理（所有提取器共用）。

- ``trim_edges``：按采样率裁剪原始信号头尾各 0.5 秒（去掉启停瞬态）。
- ``standardize``：每条信号内部 z-score 标准化 (x-mean)/(std+eps)。

降采样是 raw 专属，放在 features/extract_raw.py，不在此共用。
参数用本模块默认值，暂不从外部 cfg 改。
"""
from __future__ import annotations

import numpy as np

DEFAULT_TRIM_SECONDS = 0.5
DEFAULT_STD_EPS = 1e-8


def trim_edges(
    signal: np.ndarray,
    sampling_rate: float = 20000,
    trim_seconds: float = DEFAULT_TRIM_SECONDS,
) -> np.ndarray:
    """按采样率裁剪信号头尾各 trim_seconds 秒；短信号保持原样。"""
    sig = np.asarray(signal).reshape(-1)
    h = t = max(0, int(round(float(sampling_rate) * float(trim_seconds))))
    if sig.size > h + t:
        return sig[h: sig.size - t]
    return sig


def standardize(signal: np.ndarray, eps: float = DEFAULT_STD_EPS) -> np.ndarray:
    """每条信号 z-score 标准化：(x - mean) / (std + eps)。"""
    sig = np.asarray(signal, dtype=np.float32).reshape(-1)
    if sig.size == 0:
        return sig
    mean = float(sig.mean())
    std = float(sig.std())
    return ((sig - mean) / (std + float(eps))).astype(np.float32, copy=False)
