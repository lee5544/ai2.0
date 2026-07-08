from __future__ import annotations

import numpy as np


def reverberation(
    data: np.ndarray,
    rng: np.random.Generator,
    *,
    min_wet: float = 0.12,
    max_wet: float = 0.35,
) -> np.ndarray:
    """Blend the signal with a short synthetic decaying echo tail."""
    source = np.asarray(data)
    size = int(source.size)
    if size < 8:
        return source.copy()
    values = source.astype(np.float64, copy=False)
    impulse_size = min(size, int(rng.integers(16, 1025)))
    impulse = np.exp(-np.linspace(0.0, float(rng.uniform(3.0, 7.0)), impulse_size))
    impulse[0] = 1.0
    for _ in range(int(rng.integers(1, 4))):
        delay = int(rng.integers(1, impulse_size))
        impulse[delay] += float(rng.uniform(0.08, 0.35)) * impulse[delay]
    impulse /= max(float(np.sum(np.abs(impulse))), 1e-12)

    echoed = np.convolve(values.reshape(-1), impulse, mode="full")[:size].reshape(source.shape)
    src_rms = float(np.sqrt(np.mean(values * values)))
    echo_rms = float(np.sqrt(np.mean(echoed * echoed)))
    if echo_rms > 0:
        echoed *= src_rms / echo_rms
    wet = float(rng.uniform(min_wet, max_wet))
    return ((1.0 - wet) * values + wet * echoed).astype(source.dtype, copy=False)
