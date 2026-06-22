from __future__ import annotations

import numpy as np


def random_amplitude(
    data: np.ndarray,
    rng: np.random.Generator,
    *,
    min_scale: float = 0.85,
    max_scale: float = 1.15,
) -> np.ndarray:
    """Scale the full signal by a random amplitude factor."""
    source = np.asarray(data)
    factor = float(rng.uniform(min_scale, max_scale))
    return (source.astype(np.float64, copy=False) * factor).astype(source.dtype, copy=False)
