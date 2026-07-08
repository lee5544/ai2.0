from __future__ import annotations

import numpy as np


def speed_perturb(
    data: np.ndarray,
    rng: np.random.Generator,
    *,
    min_factor: float = 0.90,
    max_factor: float = 1.10,
) -> np.ndarray:
    """Randomly change playback speed, then crop or pad back to the original length."""
    source = np.asarray(data)
    size = int(source.size)
    if size < 4:
        return source.copy()
    values = source.reshape(-1)
    factor = float(rng.uniform(min_factor, max_factor))
    new_size = max(2, int(round(size / factor)))
    resampled = np.interp(
        np.linspace(0.0, size - 1.0, num=new_size),
        np.arange(size, dtype=np.float64),
        values.astype(np.float64, copy=False),
    )
    if new_size >= size:
        start = int(rng.integers(0, new_size - size + 1))
        output = resampled[start : start + size]
    else:
        pad = size - new_size
        left = int(rng.integers(0, pad + 1))
        output = np.pad(resampled, (left, pad - left), mode="edge")
    return output.reshape(source.shape).astype(source.dtype, copy=False)
