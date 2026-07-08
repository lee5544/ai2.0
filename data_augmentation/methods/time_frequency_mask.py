from __future__ import annotations

import numpy as np


def time_frequency_mask(
    data: np.ndarray,
    rng: np.random.Generator,
    *,
    max_time_fraction: float = 0.12,
    max_freq_fraction: float = 0.18,
) -> np.ndarray:
    """Mask random STFT time frames and frequency bins, then reconstruct the signal."""
    source = np.asarray(data)
    size = int(source.size)
    if size < 64:
        return source.copy()
    values = source.reshape(-1).astype(np.float64, copy=False)
    frame_size = min(1024, max(64, 2 ** int(np.floor(np.log2(max(64, min(size, 1024)))))))
    hop = max(1, frame_size // 4)
    frame_count = 1 + int(np.ceil(max(0, size - frame_size) / hop))
    padded_size = (frame_count - 1) * hop + frame_size
    padded = np.pad(values, (0, padded_size - size), mode="edge")
    window = np.hanning(frame_size)
    window = np.maximum(window, 1e-3)
    frames = np.empty((frame_count, frame_size), dtype=np.float64)
    for i in range(frame_count):
        start = i * hop
        frames[i] = padded[start : start + frame_size] * window

    spectrum = np.fft.rfft(frames, axis=1)
    attenuation = float(rng.uniform(0.0, 0.2))
    if frame_count > 1:
        width = max(1, int(frame_count * float(rng.uniform(0.02, max_time_fraction))))
        start = int(rng.integers(0, frame_count - width + 1))
        spectrum[start : start + width, :] *= attenuation
    freq_count = spectrum.shape[1]
    if freq_count > 2:
        width = max(1, int(freq_count * float(rng.uniform(0.03, max_freq_fraction))))
        start = int(rng.integers(1, max(2, freq_count - width + 1)))
        spectrum[:, start : start + width] *= attenuation

    rebuilt_frames = np.fft.irfft(spectrum, n=frame_size, axis=1)
    output = np.zeros(padded_size, dtype=np.float64)
    weights = np.zeros(padded_size, dtype=np.float64)
    for i in range(frame_count):
        start = i * hop
        output[start : start + frame_size] += rebuilt_frames[i] * window
        weights[start : start + frame_size] += window * window
    output = np.divide(output, weights, out=np.zeros_like(output), where=weights > 1e-12)[:size]
    return output.reshape(source.shape).astype(source.dtype, copy=False)
