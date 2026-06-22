from __future__ import annotations

import numpy as np


def random_crop(
    data: np.ndarray,
    rng: np.random.Generator,
    *,
    min_fraction: float = 0.01,
    max_fraction: float = 0.10,
) -> np.ndarray:
    """Remove a random segment, then interpolate back to the original length."""
    source = np.asarray(data)
    size = int(source.size)
    if size < 4:
        return source.copy()
    crop_size = max(1, min(size - 2, int(size * float(rng.uniform(min_fraction, max_fraction)))))
    start = int(rng.integers(0, size - crop_size + 1))
    cropped = np.concatenate((source[:start], source[start + crop_size :]))
    old_x = np.linspace(0.0, 1.0, num=cropped.size)
    new_x = np.linspace(0.0, 1.0, num=size)
    return np.interp(new_x, old_x, cropped).astype(source.dtype, copy=False)
