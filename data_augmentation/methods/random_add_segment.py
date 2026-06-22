from __future__ import annotations

import numpy as np


def random_add_segment(
    data: np.ndarray,
    rng: np.random.Generator,
    *,
    min_fraction: float = 0.01,
    max_fraction: float = 0.10,
) -> np.ndarray:
    """Copy a random signal segment into another location and retain original length."""
    source = np.asarray(data)
    size = int(source.size)
    if size < 4:
        return source.copy()
    segment_size = max(1, min(size - 1, int(size * float(rng.uniform(min_fraction, max_fraction)))))
    segment_start = int(rng.integers(0, size - segment_size + 1))
    insert_at = int(rng.integers(0, size + 1))
    segment = source[segment_start : segment_start + segment_size]
    expanded = np.concatenate((source[:insert_at], segment, source[insert_at:]))
    crop_start = int(rng.integers(0, expanded.size - size + 1))
    return expanded[crop_start : crop_start + size].astype(source.dtype, copy=False)
