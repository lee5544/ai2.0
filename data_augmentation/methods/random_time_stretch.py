from __future__ import annotations

import numpy as np


def random_time_stretch(
    data: np.ndarray,
    rng: np.random.Generator,
    *,
    min_factor: float = 0.90,
    max_factor: float = 1.10,
) -> np.ndarray:
    """Randomly compress or stretch time while retaining the original array length."""
    source = np.asarray(data)
    size = int(source.size)
    if size < 4:
        return source.copy()
    factor = float(rng.uniform(min_factor, max_factor))
    query = np.arange(size, dtype=np.float64) * factor
    query = np.clip(query, 0.0, size - 1.0)
    return np.interp(query, np.arange(size, dtype=np.float64), source).astype(source.dtype, copy=False)
