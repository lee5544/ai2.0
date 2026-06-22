"""Feature extraction v9: per-Mel-bin Hilbert envelopes, no find_peaks.

Each Mel-spaced frequency interval is treated as a real band-pass subband:

    signal -> band-pass -> Hilbert envelope -> window statistics + autocorrelation

The extractor intentionally does not import or call any previous feature
extractor, so legacy Mel peak detection cannot run indirectly.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import signal

try:
    from .v9_selected_features import (
        V9_AGGREGATE_FEATURES,
        V9_N_MEL_BINS,
        V9_PER_BIN_METRICS,
        V9_SELECTED_FEATURES,
    )
except ImportError:  # pragma: no cover
    from v9_selected_features import (
        V9_AGGREGATE_FEATURES,
        V9_N_MEL_BINS,
        V9_PER_BIN_METRICS,
        V9_SELECTED_FEATURES,
    )


_EPS = 1e-12
_FMIN_HZ = 100.0
_FMAX_NYQUIST_RATIO = 0.95
_TRIM_S = 0.6
_WINDOW_S = 1.0
_HOP_S = 0.5
_ENVELOPE_RATE_HZ = 200.0
_PERIOD_MIN_S = 0.03
_PERIOD_MAX_S = 1.0


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=float) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=float) / 2595.0) - 1.0)


def mel_bin_edges(sr: float, n_bins: int = V9_N_MEL_BINS) -> np.ndarray:
    nyquist = 0.5 * float(sr)
    fmax = nyquist * _FMAX_NYQUIST_RATIO
    if fmax <= _FMIN_HZ:
        return np.linspace(max(1.0, 0.05 * nyquist), max(2.0, fmax), n_bins + 1)
    mel_edges = np.linspace(
        _hz_to_mel(np.asarray([_FMIN_HZ]))[0],
        _hz_to_mel(np.asarray([fmax]))[0],
        int(n_bins) + 1,
    )
    return _mel_to_hz(mel_edges)


def _band_envelope(x: np.ndarray, sr: float, lo_hz: float, hi_hz: float) -> np.ndarray:
    nyquist = 0.5 * float(sr)
    lo = max(1.0, float(lo_hz))
    hi = min(float(hi_hz), nyquist * 0.98)
    if hi <= lo or x.size < 32:
        return np.zeros_like(x)
    sos = signal.butter(4, [lo / nyquist, hi / nyquist], btype="band", output="sos")
    return np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))


def _excess_kurtosis(values: np.ndarray) -> float:
    std = float(np.std(values))
    if values.size < 8 or std <= _EPS:
        return 0.0
    normalized = (values - float(np.mean(values))) / std
    return float(np.mean(normalized**4) - 3.0)


def _window_values(envelope: np.ndarray, sr: float) -> np.ndarray:
    window_n = max(8, int(round(_WINDOW_S * float(sr))))
    hop_n = max(1, int(round(_HOP_S * float(sr))))
    if envelope.size < window_n:
        return np.asarray([_excess_kurtosis(envelope)]) if envelope.size else np.empty(0)
    return np.asarray(
        [
            _excess_kurtosis(envelope[start : start + window_n])
            for start in range(0, envelope.size - window_n + 1, hop_n)
        ],
        dtype=float,
    )


def _downsample_envelope(envelope: np.ndarray, sr: float) -> tuple[np.ndarray, float]:
    target_rate = min(float(sr), _ENVELOPE_RATE_HZ)
    step = max(1, int(round(float(sr) / target_rate)))
    smooth_n = max(1, step)
    smoothed = signal.sosfilt(
        signal.butter(2, min(0.99, 0.45 * target_rate / (0.5 * float(sr))), output="sos"),
        envelope,
    )
    return smoothed[::step], float(sr) / step


def _autocorrelation_periodicity(envelope: np.ndarray, sr: float) -> tuple[float, float]:
    downsampled, envelope_sr = _downsample_envelope(envelope, sr)
    if downsampled.size < 8:
        return 0.0, 0.0
    centered = downsampled - np.median(downsampled)
    scale = float(np.sqrt(np.sum(centered * centered)))
    if scale <= _EPS:
        return 0.0, 0.0
    autocorr = signal.fftconvolve(centered, centered[::-1], mode="full")[centered.size - 1 :]
    overlap = np.arange(centered.size, 0, -1, dtype=float)
    autocorr = autocorr / np.maximum(overlap, 1.0)
    autocorr /= max(float(autocorr[0]), _EPS)
    min_lag = max(1, int(round(_PERIOD_MIN_S * envelope_sr)))
    max_lag = min(autocorr.size - 1, int(round(_PERIOD_MAX_S * envelope_sr)))
    if max_lag <= min_lag:
        return 0.0, 0.0
    region = autocorr[min_lag : max_lag + 1]
    lag = int(np.argmax(region)) + min_lag
    score = float(np.clip(autocorr[lag], 0.0, 1.0))
    return score, 1000.0 * lag / envelope_sr


def _top_k_mean(values: np.ndarray, k: int = 3) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    count = min(max(1, int(k)), values.size)
    return float(np.mean(np.partition(values, -count)[-count:]))


def extract_v9_melbin_envelopes(data: np.ndarray, sr: float) -> dict[str, float]:
    out = {name: 0.0 for name in V9_SELECTED_FEATURES}
    x = np.asarray(data, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    trim_n = int(round(_TRIM_S * float(sr)))
    if x.size > 2 * trim_n:
        x = x[trim_n:-trim_n]
    if x.size < max(32, int(round(_WINDOW_S * float(sr)))):
        return out
    x = x - float(np.mean(x))

    edges = mel_bin_edges(sr)
    bin_stats: list[dict[str, float]] = []
    for bin_index, (lo_hz, hi_hz) in enumerate(zip(edges[:-1], edges[1:])):
        envelope = _band_envelope(x, sr, float(lo_hz), float(hi_hz))
        kurts = _window_values(envelope, sr)
        periodicity, period_ms = _autocorrelation_periodicity(envelope, sr)
        mean = float(np.mean(envelope))
        stats = {
            "envkurt_mean": float(np.mean(kurts)) if kurts.size else 0.0,
            "envkurt_p75": float(np.percentile(kurts, 75)) if kurts.size else 0.0,
            "envkurt_max": float(np.max(kurts)) if kurts.size else 0.0,
            "env_cv": float(np.std(envelope) / max(mean, _EPS)),
            "periodicity": periodicity,
            "period_ms": period_ms,
        }
        bin_stats.append(stats)
        for metric in V9_PER_BIN_METRICS:
            value = float(stats[metric])
            out[f"melbin_{bin_index:02d}_{metric}"] = value if np.isfinite(value) else 0.0

    centers = 0.5 * (edges[:-1] + edges[1:])
    high_indices = np.flatnonzero(centers >= 7000.0)
    all_indices = np.arange(len(bin_stats))

    def values(indices: np.ndarray, metric: str) -> np.ndarray:
        return np.asarray([bin_stats[int(index)][metric] for index in indices], dtype=float)

    high_mean = values(high_indices, "envkurt_mean")
    high_p75 = values(high_indices, "envkurt_p75")
    high_max = values(high_indices, "envkurt_max")
    high_periodicity = values(high_indices, "periodicity")
    high_env_cv = values(high_indices, "env_cv")
    all_mean = values(all_indices, "envkurt_mean")
    all_periodicity = values(all_indices, "periodicity")
    out["melbin_high_envkurt_mean_top3"] = _top_k_mean(high_mean)
    out["melbin_high_envkurt_p75_top3"] = _top_k_mean(high_p75)
    out["melbin_high_envkurt_max_top3"] = _top_k_mean(high_max)
    out["melbin_high_periodicity_top3"] = _top_k_mean(high_periodicity)
    out["melbin_high_periodic_impact_top3"] = _top_k_mean(
        high_mean * (0.5 + 0.5 * high_periodicity)
    )
    out["melbin_high_env_cv_mean"] = float(np.mean(high_env_cv)) if high_env_cv.size else 0.0
    out["melbin_high_env_cv_top2"] = _top_k_mean(high_env_cv, k=2)
    out["melbin_all_envkurt_mean_top3"] = _top_k_mean(all_mean)
    out["melbin_all_periodicity_top3"] = _top_k_mean(all_periodicity)
    out["melbin_high_to_all_envkurt_ratio"] = float(
        _top_k_mean(high_mean) / max(_top_k_mean(all_mean), _EPS)
    )
    return out


def extract_features_v9(data: np.ndarray, sr: float, return_timing: bool = False):
    started = time.perf_counter()
    features = extract_v9_melbin_envelopes(data, sr)
    if return_timing:
        elapsed = float(time.perf_counter() - started)
        return features, {"v9_melbin_envelope_sec": elapsed, "total_sec": elapsed}
    return features
