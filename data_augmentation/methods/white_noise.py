from __future__ import annotations

import numpy as np


def white_noise(
    data: np.ndarray,
    rng: np.random.Generator,
    *,
    min_ratio: float = 0.001,
    max_ratio: float = 0.02,
) -> np.ndarray:
    """Add random Gaussian white noise relative to the signal standard deviation."""
    source = np.asarray(data)
    if source.size == 0:
        return source.copy()
    scale = float(np.std(source)) * float(rng.uniform(min_ratio, max_ratio))
    if not np.isfinite(scale) or scale <= 0:
        return source.copy()
    noise = rng.normal(0.0, scale, size=source.shape)
    return (source.astype(np.float64, copy=False) + noise).astype(source.dtype, copy=False)
