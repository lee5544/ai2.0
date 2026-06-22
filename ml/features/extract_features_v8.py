"""Feature extraction v8: v7 plus weak_tick high-band persistence features."""
from __future__ import annotations

import time

import numpy as np
from scipy import signal

try:
    from .extract_features_v7 import extract_features_v7
    from .v8_selected_features import V8_SELECTED_FEATURES, V8_TICK_PERSISTENCE_FEATURES
except ImportError:  # pragma: no cover
    from extract_features_v7 import extract_features_v7
    from v8_selected_features import V8_SELECTED_FEATURES, V8_TICK_PERSISTENCE_FEATURES


_TICK_HI_BAND = (7000.0, 9500.0)
_TICK_TRIM_S = 0.6
_TICK_WIN_S = 1.0
_TICK_HOP_S = 0.5


def _envkurt(envelope: np.ndarray) -> float:
    std = float(np.std(envelope))
    if envelope.size < 8 or std < 1e-12:
        return 0.0
    normalized = (envelope - float(np.mean(envelope))) / std
    return float(np.mean(normalized**4) - 3.0)


def extract_v8_tick_persistence(data: np.ndarray, sr: float) -> dict[str, float]:
    out = {name: 0.0 for name in V8_TICK_PERSISTENCE_FEATURES}
    x = np.asarray(data, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    trim_n = int(round(_TICK_TRIM_S * float(sr)))
    if x.size > 2 * trim_n:
        x = x[trim_n:-trim_n]
    window_n = int(round(_TICK_WIN_S * float(sr)))
    hop_n = max(1, int(round(_TICK_HOP_S * float(sr))))
    if x.size < window_n:
        return out

    x = x - float(np.mean(x))
    nyquist = 0.5 * float(sr)
    hi = min(_TICK_HI_BAND[1], nyquist * 0.98)
    if hi <= _TICK_HI_BAND[0]:
        return out
    sos = signal.butter(
        4,
        [_TICK_HI_BAND[0] / nyquist, hi / nyquist],
        btype="band",
        output="sos",
    )
    envelope = np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))
    values = np.asarray(
        [
            _envkurt(envelope[start : start + window_n])
            for start in range(0, envelope.size - window_n + 1, hop_n)
        ],
        dtype=float,
    )
    if values.size:
        out["x_tick_hi_envkurt_mean"] = float(np.mean(values))
        out["x_tick_hi_envkurt_p75"] = float(np.percentile(values, 75))
    return out


def extract_features_v8(data: np.ndarray, sr: float, return_timing: bool = False):
    started = time.perf_counter()
    if return_timing:
        base, timing = extract_features_v7(data, sr, return_timing=True)
    else:
        base = extract_features_v7(data, sr)
        timing = {}
    persistence = extract_v8_tick_persistence(data, sr)
    merged = {**base, **persistence}
    features = {name: merged[name] for name in V8_SELECTED_FEATURES}
    if return_timing:
        timing["v8_tick_persistence_sec"] = float(time.perf_counter() - started) - timing.get(
            "total_sec", 0.0
        )
        timing["total_sec"] = float(time.perf_counter() - started)
        return features, timing
    return features
