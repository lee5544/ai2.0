from __future__ import annotations

import numpy as np


def add_noise(
    data: np.ndarray,
    rng: np.random.Generator,
    *,
    min_snr_db: float = 18.0,
    max_snr_db: float = 35.0,
) -> np.ndarray:
    """Add Gaussian noise using a random SNR."""
    source = np.asarray(data)
    if source.size == 0:
        return source.copy()
    values = source.astype(np.float64, copy=False)
    power = float(np.mean(values * values))
    if not np.isfinite(power) or power <= 0:
        return source.copy()
    snr_db = float(rng.uniform(min_snr_db, max_snr_db))
    noise_std = float(np.sqrt(power / (10.0 ** (snr_db / 10.0))))
    noise = rng.normal(0.0, noise_std, size=source.shape)
    return (values + noise).astype(source.dtype, copy=False)
