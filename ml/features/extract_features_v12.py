"""Feature extraction v11 (standalone).

v11 = v10 骨架(165) + 弱秒表 8 维 + 震颤调制 9 维 = 182 维。
本文件完全自包含，不 import 任何其他特征版本（v10 全部代码已内联）。

Feature groups (inherited from v10):
  1. Global time-domain (6)
  2. Mel-band basic statistics (20)
  3. Mel-band windowed impulse statistics (48)
  4. Mel shock peaks (37)
  5. DWT-band windowed statistics (54)

v11 additions:
  6. weak_tick 双带分窗冲击 (8): x_tick_*
     来源 v7/v8 实测（7–9.5 kHz 高频带分窗包络峭度 d≈1.5 / AUC≈0.87）
  7. 震颤调制结构 (9): env_mod_chatter_*, mel_mod_*, dwt_3/4 SK burst
     来源 v5 实测（dwt_3_sk_burst_frac importance #4；跨带调制同步）

预处理改动：裁瞬态统一为 trim_s=0.6 s（v10 为 cut_len=8000 点 ≈ 0.4 s @20 kHz）。

入口: extract_features_v12(data, sr, return_timing=False)  # 独立文件，不依赖 v11
"""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.ndimage import percentile_filter, median_filter
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

try:
    import pywt
except Exception:
    pywt = None


_EPS = 1e-12

_DEFAULT_N_MELS = 13
_DEFAULT_N_FFT = 256

# Mel shock local contrast parameters
_JND_DB = 4.0
_SCORE_THRESHOLD = 4.0
_SCORE_PROMINENCE = 2.0
_LOCAL_BG_SEC = 0.6

# DWT parameters
_DWT_WAVELET = "db4"
_DWT_LEVEL = 5


# ===================== Mel / STFT utilities =====================

def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _power_to_db(
    power: np.ndarray,
    amin: float = 1e-10,
    top_db: float | None = 80.0,
) -> np.ndarray:
    p = np.maximum(np.asarray(power, dtype=np.float64), amin)
    ref = max(float(np.max(p)), amin)
    db = 10.0 * np.log10(p) - 10.0 * np.log10(ref)

    if top_db is not None:
        db = np.maximum(db, float(np.max(db)) - float(top_db))

    return db


def _mel_filterbank(
    sr: float,
    n_fft: int,
    n_mels: int,
    fmax: float | None = None,
) -> np.ndarray:
    sr = float(sr)
    n_fft = int(n_fft)
    n_mels = int(n_mels)
    fmax = min(float(fmax if fmax is not None else sr / 2.0), sr / 2.0)

    mel_edges = np.linspace(_hz_to_mel(0.0), _hz_to_mel(fmax), n_mels + 2)
    hz_edges = _mel_to_hz(mel_edges)
    fft_freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)

    fb = np.zeros((n_mels, fft_freqs.size), dtype=np.float64)

    for i in range(n_mels):
        left = hz_edges[i]
        center = hz_edges[i + 1]
        right = hz_edges[i + 2]

        up = (fft_freqs - left) / max(center - left, _EPS)
        down = (right - fft_freqs) / max(right - center, _EPS)
        fb[i] = np.maximum(0.0, np.minimum(up, down))

    # Slaney-like area normalization.
    enorm = 2.0 / np.maximum(hz_edges[2:n_mels + 2] - hz_edges[:n_mels], _EPS)
    fb *= enorm[:, None]

    return fb


def _melspectrogram(
    y: np.ndarray,
    sr: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    x = np.asarray(y, dtype=np.float64).ravel()

    if x.size == 0:
        return np.zeros((int(n_mels), 0), dtype=np.float64)

    n_fft = int(n_fft)
    hop_length = int(hop_length)

    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size), mode="constant")

    _, _, zxx = signal.stft(
        x,
        fs=float(sr),
        window="hann",
        nperseg=n_fft,
        noverlap=max(0, n_fft - hop_length),
        nfft=n_fft,
        boundary=None,
        padded=False,
    )

    power = np.abs(zxx) ** 2
    mel_fb = _mel_filterbank(sr, n_fft, n_mels, fmax=float(sr) / 2.0)
    mel_power = mel_fb @ power

    return np.maximum(mel_power, 0.0)


# ===================== Preprocessing =====================

@lru_cache(maxsize=128)
def _butter_coeffs(sr: float, cutoff: float, btype: str, order: int):
    nyquist = 0.5 * float(sr)
    normal_cutoff = min(max(float(cutoff) / nyquist, 1e-6), 0.999999)
    return signal.butter(int(order), normal_cutoff, btype=btype, analog=False)


def butter_filter(
    data: np.ndarray,
    sr: float,
    cutoff: float,
    btype: str = "low",
    order: int = 4,
) -> np.ndarray:
    b, a = _butter_coeffs(float(sr), float(cutoff), str(btype), int(order))
    x = np.asarray(data, dtype=np.float64).ravel()

    if x.size < 3 * max(len(a), len(b)):
        return x.astype(np.float32)

    return signal.filtfilt(b, a, x).astype(np.float32)


def process_data(
    raw_data: np.ndarray,
    sr: float = 20000,
    cut_len: int = 8000,
    target_length: int = 0,
    cutoff_low: float | None = 20,
    cutoff_high: float | None = None,
) -> np.ndarray:
    x = np.asarray(raw_data, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.zeros(0, dtype=np.float32)

    # 20 Hz high-pass before Mel and DWT features.
    if cutoff_low is not None:
        x = butter_filter(x, sr, cutoff_low, btype="high")

    if cutoff_high is not None:
        x = butter_filter(x, sr, cutoff_high, btype="low")

    # Remove start/end transient if signal is long enough.
    if x.size > 2 * int(cut_len):
        x = x[int(cut_len): -int(cut_len)]

    # RMS normalization.
    rms = float(np.sqrt(np.mean(x * x)))
    if rms > _EPS:
        x = x * (0.1 / rms)

    if target_length and target_length > 0:
        target_length = int(target_length)
        if x.size < target_length:
            x = np.pad(x, (0, target_length - x.size), mode="constant")
        elif x.size > target_length:
            x = x[:target_length]

    return np.asarray(x, dtype=np.float32)


# ===================== Numeric helpers =====================

def _safe_skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()

    if x.size < 3 or float(np.std(x)) <= _EPS:
        return 0.0

    v = float(skew(x, bias=False))
    return v if np.isfinite(v) else 0.0


def _safe_kurtosis(x: np.ndarray) -> float:
    """Pearson kurtosis. A normal distribution is about 3."""
    x = np.asarray(x, dtype=np.float64).ravel()

    if x.size < 4 or float(np.std(x)) <= _EPS:
        return 0.0

    v = float(kurtosis(x, fisher=False, bias=False))
    return v if np.isfinite(v) else 0.0


def _zero_crossing_rate(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()

    if x.size < 2:
        return 0.0

    s = np.sign(x)
    crossings = (s[1:] != s[:-1]) & (s[1:] != 0) & (s[:-1] != 0)

    return float(np.mean(crossings))


def _entropy_from_nonnegative(values: np.ndarray) -> float:
    v = np.maximum(np.asarray(values, dtype=np.float64).ravel(), 0.0)
    total = float(np.sum(v))

    if v.size == 0 or total <= _EPS:
        return 0.0

    p = v / total
    p = p[p > 0]

    return float(-np.sum(p * np.log(p)))


def _aggregate_values(values: list[float] | np.ndarray, prefix: str) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]

    if v.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_max": 0.0,
        }

    return {
        f"{prefix}_mean": float(np.mean(v)),
        f"{prefix}_p75": float(np.percentile(v, 75)),
        f"{prefix}_p90": float(np.percentile(v, 90)),
        f"{prefix}_max": float(np.max(v)),
    }


def _aggregate_values_3(values: list[float] | np.ndarray, prefix: str) -> dict[str, float]:
    """Aggregate with mean / p90 / max only."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]

    if v.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_max": 0.0,
        }

    return {
        f"{prefix}_mean": float(np.mean(v)),
        f"{prefix}_p90": float(np.percentile(v, 90)),
        f"{prefix}_max": float(np.max(v)),
    }


def _mel_band_indices(n_mels: int) -> dict[str, np.ndarray]:
    n_mels = int(n_mels)
    one = max(1, n_mels // 3)
    two = max(one + 1, 2 * n_mels // 3)

    return {
        "low": np.arange(0, one),
        "mid": np.arange(one, two),
        "high": np.arange(two, n_mels),
        "all": np.arange(0, n_mels),
    }


# ===================== Mel-band basic features =====================

def melband_basic_features(
    mel_power: np.ndarray,
    prefix: str = "melband_basic",
) -> dict[str, float]:
    """Merged Mel-bin basic statistics into low / mid / high / all groups.

    For each group:
      std:      std of log-power sequence
      ratio:    group power / total power
      kurtosis: kurtosis of log-power sequence
      impulse:  max(power) / mean(power)
      entropy:  temporal energy entropy
    """
    power = np.asarray(mel_power, dtype=np.float64)

    if power.ndim != 2:
        raise ValueError("mel_power must be 2-D: [n_mels, n_frames]")

    n_mels = int(power.shape[0])
    bands = _mel_band_indices(n_mels)
    total_power = float(np.sum(np.maximum(power, 0.0))) + _EPS

    features: dict[str, float] = {}

    for band_name, idx in bands.items():
        idx = idx[(idx >= 0) & (idx < n_mels)]

        if idx.size == 0 or power.shape[1] == 0:
            band_power = np.zeros(power.shape[1], dtype=np.float64)
        else:
            band_power = np.sum(np.maximum(power[idx, :], 0.0), axis=0)

        band_db = _power_to_db(band_power[None, :]).ravel() if band_power.size else np.zeros(0)

        mean_power = float(np.mean(band_power)) if band_power.size else 0.0
        base = f"{prefix}_{band_name}"

        features[f"{base}_std"] = float(np.std(band_db)) if band_db.size else 0.0
        features[f"{base}_ratio"] = float(np.sum(band_power) / total_power)
        features[f"{base}_kurtosis"] = _safe_kurtosis(band_db)
        features[f"{base}_impulse"] = float(np.max(band_power) / (mean_power + _EPS)) if band_power.size else 0.0
        features[f"{base}_entropy"] = _entropy_from_nonnegative(band_power)

    return features


# ===================== Mel-band windowed impulse features =====================

def _windowed_impulse_stats_from_series(
    db_series: np.ndarray,
    power_series: np.ndarray,
    mel_frame_sr: float,
    *,
    win_sec: float = 1.0,
    hop_sec: float = 0.5,
) -> dict[str, dict[str, float]]:
    db = np.asarray(db_series, dtype=np.float64).ravel()
    p = np.maximum(np.asarray(power_series, dtype=np.float64).ravel(), 0.0)

    n = min(db.size, p.size)
    db = db[:n]
    p = p[:n]

    win = max(2, int(round(float(win_sec) * float(mel_frame_sr))))
    hop = max(1, int(round(float(hop_sec) * float(mel_frame_sr))))

    if n == 0:
        windows: list[tuple[int, int]] = []
    elif n < win:
        windows = [(0, n)]
    else:
        windows = [(i, i + win) for i in range(0, n - win + 1, hop)]
        if windows[-1][1] < n:
            windows.append((n - win, n))

    kurt_vals: list[float] = []
    crest_vals: list[float] = []
    impulse_vals: list[float] = []

    for left, right in windows:
        dseg = db[left:right]
        pseg = p[left:right]

        if dseg.size == 0 or pseg.size == 0:
            continue

        kurt_vals.append(_safe_kurtosis(dseg))

        rms = float(np.sqrt(np.mean(pseg * pseg)))
        crest_vals.append(float(np.max(pseg) / (rms + _EPS)))

        mean_abs = float(np.mean(np.abs(pseg)))
        impulse_vals.append(float(np.max(pseg) / (mean_abs + _EPS)))

    return {
        "kurtosis": _aggregate_values(kurt_vals, "kurtosis"),
        "crest": _aggregate_values(crest_vals, "crest"),
        "impulse": _aggregate_values(impulse_vals, "impulse"),
    }


def mel_band_window_impulse_features(
    mel_power: np.ndarray,
    sr: float,
    hop_length: int,
    prefix: str = "melband",
) -> dict[str, float]:
    power = np.asarray(mel_power, dtype=np.float64)

    if power.ndim != 2 or power.shape[1] == 0:
        return {}

    n_mels = int(power.shape[0])
    bands = _mel_band_indices(n_mels)
    mel_frame_sr = float(sr) / float(hop_length)

    features: dict[str, float] = {}

    for band_name, idx in bands.items():
        idx = idx[(idx >= 0) & (idx < n_mels)]

        if idx.size == 0:
            band_power = np.zeros(power.shape[1], dtype=np.float64)
        else:
            band_power = np.sum(np.maximum(power[idx, :], 0.0), axis=0)

        band_db = _power_to_db(band_power[None, :]).ravel()

        stats = _windowed_impulse_stats_from_series(
            band_db,
            band_power,
            mel_frame_sr,
            win_sec=1.0,
            hop_sec=0.5,
        )

        for metric, vals in stats.items():
            for agg, value in vals.items():
                agg_name = agg.split("_", 1)[1]
                features[f"{prefix}_{band_name}_win_{metric}_{agg_name}"] = float(value)

    return features


# ===================== Mel shock features =====================

def _row_peak_map(
    row: np.ndarray,
    distance: int = 5,
    local_win_frames: int = 50,
) -> dict[int, dict[str, float]]:
    """Detect local Mel shocks with fast robust local contrast.

    Preprocessing:
      baseline = rolling p30(row)
      residual_db = row - baseline
      robust_noise = rolling MAD(residual_db) / 0.6745
      score = residual_db / robust_noise

    Peak condition:
      score >= _SCORE_THRESHOLD
      residual_db >= _JND_DB
    """
    row = np.asarray(row, dtype=np.float64).ravel()

    if row.size < 4:
        return {}

    local_win_frames = max(5, int(local_win_frames))

    baseline = percentile_filter(
        row,
        percentile=30,
        size=local_win_frames,
        mode="nearest",
    )

    residual = row - baseline

    med_res = median_filter(
        residual,
        size=local_win_frames,
        mode="nearest",
    )

    mad = median_filter(
        np.abs(residual - med_res),
        size=local_win_frames,
        mode="nearest",
    )

    noise = np.maximum(mad / 0.6745, 1.0)
    score = residual / (noise + _EPS)

    candidates, _ = find_peaks(
        score,
        height=_SCORE_THRESHOLD,
        distance=max(1, int(distance)),
        prominence=_SCORE_PROMINENCE,
        wlen=local_win_frames,
        width=[1, local_win_frames],
    )

    peaks = [int(p) for p in candidates if residual[int(p)] >= _JND_DB]

    out: dict[int, dict[str, float]] = {}

    for p in peaks:
        left = max(0, p - 10)
        right = min(row.size, p + 11)

        out[p] = {
            "kurtosis": _safe_kurtosis(row[left:right]),
            "response_db": float(residual[p]),
            "score": float(score[p]),
            "baseline_db": float(baseline[p]),
            "db": float(row[p]),
        }

    return out


def _merge_peak_maps(
    row_peak_maps: list[dict[int, dict[str, float]]],
    merge_distance: int = 1,
) -> dict[int, dict[str, float]]:
    candidates = [(int(pos), stat) for m in row_peak_maps for pos, stat in m.items()]

    if not candidates:
        return {}

    candidates.sort(key=lambda x: x[0])

    clusters: list[list[tuple[int, dict[str, float]]]] = []
    cur = [candidates[0]]

    for item in candidates[1:]:
        if item[0] - cur[-1][0] <= merge_distance:
            cur.append(item)
        else:
            clusters.append(cur)
            cur = [item]

    clusters.append(cur)

    merged: dict[int, dict[str, float]] = {}

    for cluster in clusters:
        best_pos, best_stat = max(
            cluster,
            key=lambda x: (
                x[1]["response_db"],
                x[1]["db"],
                x[1]["kurtosis"],
            ),
        )

        merged[best_pos] = {
            "kurtosis": float(max(s["kurtosis"] for _, s in cluster)),
            "response_db": float(max(s["response_db"] for _, s in cluster)),
            "db": float(best_stat["db"]),
        }

    return merged


def _summarize_peaks(
    peaks: dict[int, dict[str, float]],
    kf_thr: float,
    response_thr: float,
    prefix: str,
    mel_frame_sr: float,
    n_frames: int,
) -> dict[str, float]:
    keys = [
        "count_per_sec",
        "kurtosis_mean",
        "kurtosis_q95",
        "kurtosis_q50",
        "response_db_mean",
        "response_db_std",
        "response_db_q95",
        "response_db_q50",
        "interval_cv",
    ]

    empty = {f"{prefix}_{k}": 0.0 for k in keys}

    filtered = {
        p: s
        for p, s in peaks.items()
        if (
            (s["kurtosis"] >= 5.0 or s["response_db"] >= 9.0)
            or (s["kurtosis"] >= kf_thr and s["response_db"] >= response_thr)
        )
    }

    if not filtered:
        return empty

    duration = float(n_frames) / max(float(mel_frame_sr), _EPS)

    pos = np.asarray(sorted(filtered), dtype=np.float64)
    kvals = np.asarray([s["kurtosis"] for s in filtered.values()], dtype=np.float64)
    rvals = np.asarray([s["response_db"] for s in filtered.values()], dtype=np.float64)

    if pos.size >= 2:
        intervals = np.diff(pos) / max(float(mel_frame_sr), _EPS)
        interval_cv = float(np.std(intervals) / (float(np.mean(intervals)) + _EPS))
    else:
        interval_cv = 0.0

    return {
        f"{prefix}_count_per_sec": float(len(filtered) / max(duration, _EPS)),
        f"{prefix}_kurtosis_mean": float(np.mean(kvals)),
        f"{prefix}_kurtosis_q95": float(np.percentile(kvals, 95)),
        f"{prefix}_kurtosis_q50": float(np.percentile(kvals, 50)),
        f"{prefix}_response_db_mean": float(np.mean(rvals)),
        f"{prefix}_response_db_std": float(np.std(rvals)),
        f"{prefix}_response_db_q95": float(np.percentile(rvals, 95)),
        f"{prefix}_response_db_q50": float(np.percentile(rvals, 50)),
        f"{prefix}_interval_cv": interval_cv,
    }


def mel_shock_features(
    mel_power: np.ndarray,
    sr: float,
    hop_length: int,
    prefix: str = "mel_shock",
) -> dict[str, float]:
    power = np.asarray(mel_power, dtype=np.float64)

    if power.ndim != 2 or power.shape[1] < 3:
        return {}

    # Remove first/last frame to avoid edge artifacts.
    mel_db = _power_to_db(power)[:, 1:-1]

    n_frames = int(mel_db.shape[1])
    mel_frame_sr = float(sr) / float(hop_length)

    # Local background/MAD window: about 0.6 sec.
    local_win_frames = max(
        5,
        int(round(_LOCAL_BG_SEC * float(sr) / float(hop_length))),
    )

    row_maps = [
        _row_peak_map(
            mel_db[i, :],
            distance=5,
            local_win_frames=local_win_frames,
        )
        for i in range(mel_db.shape[0])
    ]

    features: dict[str, float] = {}

    features.update(
        _summarize_peaks(
            _merge_peak_maps(row_maps[1:7]),
            4,
            6,
            f"{prefix}_low",
            mel_frame_sr,
            n_frames,
        )
    )

    features.update(
        _summarize_peaks(
            _merge_peak_maps(row_maps[7:9]),
            3,
            6,
            f"{prefix}_med",
            mel_frame_sr,
            n_frames,
        )
    )

    features.update(
        _summarize_peaks(
            _merge_peak_maps(row_maps[-6:]),
            3,
            5,
            f"{prefix}_high",
            mel_frame_sr,
            n_frames,
        )
    )

    features.update(
        _summarize_peaks(
            _merge_peak_maps(row_maps[:]),
            3,
            6,
            f"{prefix}_all",
            mel_frame_sr,
            n_frames,
        )
    )

    low_count = max(features.get(f"{prefix}_low_count_per_sec", 0.0), 0.0)
    high_count = max(features.get(f"{prefix}_high_count_per_sec", 0.0), 0.0)

    features[f"{prefix}_high_ratio"] = float(
        high_count / max(low_count + high_count, _EPS)
    )

    return features


# ===================== DWT-band window features =====================

_DB4_DEC_LO = np.asarray(
    [
        -0.010597401785069032,
        0.032883011666982945,
        0.030841381835986965,
        -0.18703481171888114,
        -0.02798376941698385,
        0.6308807679298587,
        0.7148465705529154,
        0.2303778133088965,
    ],
    dtype=np.float64,
)

_DB4_DEC_HI = np.asarray(
    [
        -0.2303778133088965,
        0.7148465705529154,
        -0.6308807679298587,
        -0.02798376941698385,
        0.18703481171888114,
        0.030841381835986965,
        -0.032883011666982945,
        -0.010597401785069032,
    ],
    dtype=np.float64,
)


def _dwt_decompose_fallback_db4(
    x: np.ndarray,
    level: int = _DWT_LEVEL,
) -> dict[str, tuple[np.ndarray, int]]:
    """Fallback DWT implementation if pywt is unavailable."""
    cur = np.asarray(x, dtype=np.float64).ravel()

    if cur.size == 0:
        return {
            name: (np.zeros(0, dtype=np.float64), 1)
            for name in ["a5", "d5", "d4", "d3", "d2", "d1"]
        }

    details: list[tuple[str, np.ndarray, int]] = []
    filt_len = int(_DB4_DEC_LO.size)

    for lev in range(1, int(level) + 1):
        if cur.size < filt_len:
            cur = np.pad(cur, (0, filt_len - cur.size), mode="edge")

        pad = filt_len - 1
        padded = np.pad(cur, (pad, pad), mode="symmetric")

        approx = signal.convolve(padded, _DB4_DEC_LO[::-1], mode="valid")[::2]
        detail = signal.convolve(padded, _DB4_DEC_HI[::-1], mode="valid")[::2]

        details.append((f"d{lev}", np.asarray(detail, dtype=np.float64), 2 ** lev))
        cur = np.asarray(approx, dtype=np.float64)

    bands: dict[str, tuple[np.ndarray, int]] = {
        f"a{level}": (cur, 2 ** int(level))
    }

    for name, coeff, decim in reversed(details):
        bands[name] = (coeff, decim)

    return bands


def _dwt_decompose(
    x: np.ndarray,
    level: int = _DWT_LEVEL,
    wavelet: str = _DWT_WAVELET,
) -> dict[str, tuple[np.ndarray, int]]:
    """DWT decomposition.

    Prefer pywt.wavedec for speed if PyWavelets is installed.
    Fallback to self-contained db4 implementation otherwise.
    """
    x = np.asarray(x, dtype=np.float64).ravel()

    if pywt is not None:
        coeffs = pywt.wavedec(x, wavelet, level=level, mode="symmetric")
        names = [f"a{level}"] + [f"d{i}" for i in range(level, 0, -1)]
        decims = [2 ** level] + [2 ** i for i in range(level, 0, -1)]

        return {
            name: (np.asarray(coeff, dtype=np.float64), decim)
            for name, coeff, decim in zip(names, coeffs, decims)
        }

    return _dwt_decompose_fallback_db4(x, level=level)


def _windowed_dwt_stats_from_signal(
    series: np.ndarray,
    frame_sr: float,
    *,
    win_sec: float = 1.0,
    hop_sec: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Windowed DWT statistics directly on input series.

    Important:
      This function does NOT remove the mean.
      This function does NOT standardize the window.
      This function does NOT smooth the window.

    It directly computes:
      kurtosis(seg)
      zero_crossing_rate(seg)
      std(seg)
    """
    z = np.asarray(series, dtype=np.float64).ravel()
    z = z[np.isfinite(z)]

    n = int(z.size)

    win = max(4, int(round(float(win_sec) * float(frame_sr))))
    hop = max(1, int(round(float(hop_sec) * float(frame_sr))))

    if n == 0:
        windows: list[tuple[int, int]] = []
    elif n < win:
        windows = [(0, n)]
    else:
        windows = [(i, i + win) for i in range(0, n - win + 1, hop)]
        if windows and windows[-1][1] < n:
            windows.append((n - win, n))

    kurt_vals: list[float] = []
    zcr_vals: list[float] = []
    std_vals: list[float] = []

    for left, right in windows:
        seg = z[left:right]

        if seg.size == 0:
            continue

        kurt_vals.append(_safe_kurtosis(seg))
        zcr_vals.append(_zero_crossing_rate(seg))
        std_vals.append(float(np.std(seg)))

    return {
        "kurtosis": _aggregate_values_3(kurt_vals, "kurtosis"),
        "zcr": _aggregate_values_3(zcr_vals, "zcr"),
        "std": _aggregate_values_3(std_vals, "std"),
    }


def dwt_band_window_features(
    x: np.ndarray,
    sr: float,
    prefix: str = "dwtband",
    *,
    level: int = _DWT_LEVEL,
    wavelet: str = _DWT_WAVELET,
    win_sec: float = 1.0,
    hop_sec: float = 0.5,
) -> dict[str, float]:
    """DWT windowed statistics.

    No DWT basic full-band statistics:
      no global DWT rms
      no global DWT ratio
      no global DWT entropy
      no global DWT q95

    DWT feature flow:
      preprocessed signal x
      -> DWT decomposition
      -> each subband coefficients
      -> direct window segmentation
      -> kurtosis / zcr / std
      -> mean / p90 / max
    """
    bands = _dwt_decompose(x, level=level, wavelet=wavelet)
    order = [f"a{level}"] + [f"d{i}" for i in range(level, 0, -1)]

    features: dict[str, float] = {}

    for name in order:
        coeff, decim = bands.get(name, (np.zeros(0, dtype=np.float64), 1))

        # DWT coefficients are downsampled by decimation factor.
        coeff_sr = float(sr) / float(max(int(decim), 1))

        stats = _windowed_dwt_stats_from_signal(
            coeff,
            coeff_sr,
            win_sec=win_sec,
            hop_sec=hop_sec,
        )

        for metric, vals in stats.items():
            for agg, value in vals.items():
                agg_name = agg.split("_", 1)[1]
                features[f"{prefix}_{name}_win_{metric}_{agg_name}"] = float(value)

    return features


# ===================== Main extractor =====================

def _extract_base_features(
    data: np.ndarray,
    sr: float,
    return_timing: bool = False,
    *,
    n_mels: int = _DEFAULT_N_MELS,
    n_fft: int = _DEFAULT_N_FFT,
    hop_length: int | None = None,
    cut_len: int = 8000,
) -> dict[str, float] | tuple[dict[str, float], dict[str, float]]:
    timing: dict[str, float] = {}
    t_total = time.perf_counter()

    x_raw = np.asarray(data, dtype=np.float64).ravel()
    x_raw = x_raw[np.isfinite(x_raw)]

    t0 = time.perf_counter()
    x = process_data(
        x_raw,
        sr=sr,
        cut_len=cut_len,
        cutoff_low=20,
        cutoff_high=None,
    )
    timing["process_data_sec"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    features: dict[str, float] = {
        "mean": float(np.mean(x_raw)) if x_raw.size else 0.0,
        "std": float(np.std(x)) if x.size else 0.0,
        "duration": float(x_raw.size / float(sr)) if sr else 0.0,
        "skewness": _safe_skew(x),
        "kurtosis": _safe_kurtosis(x),
        "zero_crossing_rate": _zero_crossing_rate(x),
    }
    timing["global_sec"] = float(time.perf_counter() - t0)

    hop = int(hop_length) if hop_length is not None else int(n_fft) // 2

    t0 = time.perf_counter()
    mel_power = _melspectrogram(
        x,
        sr=float(sr),
        n_mels=int(n_mels),
        n_fft=int(n_fft),
        hop_length=hop,
    )

    features.update(melband_basic_features(mel_power, prefix="melband_basic"))

    features.update(
        mel_band_window_impulse_features(
            mel_power,
            sr=sr,
            hop_length=hop,
            prefix="melband",
        )
    )

    timing["melband_basic_and_window_sec"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    features.update(
        mel_shock_features(
            mel_power,
            sr=sr,
            hop_length=hop,
            prefix="mel_shock",
        )
    )
    timing["mel_shock_sec"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    features.update(
        dwt_band_window_features(
            x,
            sr=sr,
            prefix="dwtband",
            level=_DWT_LEVEL,
            wavelet=_DWT_WAVELET,
            win_sec=1.0,
            hop_sec=0.5,
        )
    )
    timing["dwtband_window_sec"] = float(time.perf_counter() - t0)

    timing["total_sec"] = float(time.perf_counter() - t_total)

    return (features, timing) if return_timing else features



# ===================== v11 additions =====================
# 来源与实测依据（实现全部内联，本文件不 import 任何其他特征版本）：
#   弱秒表 8 维：v7/v8 验证特征（解决方案-weak_tick识别为秒表.md，
#     高频带 7–9.5 kHz 分窗包络峭度 d≈1.5 / AUC≈0.87）。
#   震颤调制 8 维：v5 验证特征（v5_10新_结果分析.md，
#     dwt_3_sk_burst_frac importance #4；env chatter 80–300 Hz；跨带同步）。

V11_TICK_FEATURES: tuple[str, ...] = (
    "x_tick_hi_envkurt_max",       # 高频带 7–9.5 kHz 分窗包络峭度最大（weak_tick 主力）
    "x_tick_hi_envkurt_p90",       # 高频带分窗包络峭度 90 分位（稳健）
    "x_tick_hi_envkurt_mean",      # 高频带分窗包络峭度均值（持续性, v8）
    "x_tick_hi_envkurt_p75",       # 高频带分窗包络峭度 75 分位（持续性, v8）
    "x_tick_hi_crest_max",         # 高频带分窗 crest factor 最大
    "x_tick_lomid_envkurt_max",    # 低/中带 0.5–5 kHz 分窗包络峭度最大（强秒表）
    "x_tick_lomid_crest_max",      # 低/中带分窗 crest factor 最大
    "x_tick_hi_to_lomid_envkurt",  # 高/低带峭度占比（弱秒表 vs 强秒表）
)

V11_CHATTER_FEATURES: tuple[str, ...] = (
    "env_mod_chatter_lo",            # Hilbert 包络调制 80–200 Hz 能量占比（震颤主落点）
    "env_mod_chatter_hi",            # Hilbert 包络调制 200–300 Hz 能量占比
    "mel_mod_mid_peak_prominence",   # 中频组调制谱主峰突出度 dB
    "mel_mod_high_peak_prominence",  # 高频组调制谱主峰突出度 dB
    "mel_mod_sync_global",           # 跨带调制主峰频率同步度（震颤跨带同步）
    "mel_mod_band_coherence",        # 相邻 mel 行调制谱平均皮尔森相关
    "dwt_3_sk_burst_frac",           # cD2(2.5–5 kHz) SK 滑窗超阈占比
    "dwt_4_sk_burst_frac",           # cD1(5–10 kHz) SK 滑窗超阈占比（5 硬漏件 z=+2.21）
    "dwt_4_sk_max_run_len",          # cD1(5–10 kHz) SK 滑窗最长连续超阈段（z=+2.72）
)

# 统一裁瞬态（秒）：v10 的 cut_len=8000(0.4s@20k) 偏短，启停瞬态残留会劫持峭度。
_TRIM_S = 0.6

# 弱秒表参数（实测最优，见解决方案文档）
_TICK_HI_BAND = (7000.0, 9500.0)
_TICK_LOMID_BAND = (500.0, 5000.0)
_TICK_WIN_S = 1.0
_TICK_HOP_S = 0.5

# 震颤调制参数（与 v5 一致）
_CHATTER_BAND_GROUP_HZ = (
    ("low", 100.0, 1500.0),
    ("mid", 1500.0, 4000.0),
    ("high", 4000.0, None),
)
_ENV_MOD_BANDS = (
    ("chatter_lo", 80.0, 200.0),
    ("chatter_hi", 200.0, 300.0),
)
_SK_WIN = 256
_SK_STEP = 64
_SK_THR = 4.0


def _safe_finite(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def _v11_bandpass(x: np.ndarray, sr: float, lo: float, hi: float) -> np.ndarray:
    ny = 0.5 * float(sr)
    hi = min(float(hi), ny * 0.98)
    if hi <= lo:
        return x
    b, a = signal.butter(4, [lo / ny, hi / ny], btype="band")
    return signal.filtfilt(b, a, x)


def _excess_envkurt(e: np.ndarray) -> float:
    s = float(e.std())
    if s < 1e-12:
        return 0.0
    return float(np.mean(((e - e.mean()) / s) ** 4) - 3.0)


def _crest_factor(seg: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(seg * seg)))
    return 0.0 if r < 1e-12 else float(np.max(np.abs(seg)) / r)


def _windowed_apply(sig: np.ndarray, sr: float, fn) -> list[float]:
    w = int(_TICK_WIN_S * sr)
    h = int(_TICK_HOP_S * sr)
    if w <= 0 or sig.size < w:
        return [fn(sig)] if sig.size else []
    return [fn(sig[s:s + w]) for s in range(0, sig.size - w + 1, max(1, h))]


def extract_v11_tick(data: np.ndarray, sr: float) -> dict[str, float]:
    """弱秒表双频带分窗冲击特征：裁瞬态 0.6s → 带通 → |hilbert| 包络分窗峭度/crest。
    在原始信号上计算（与 v7 验证条件一致），单方向。"""
    out = {k: 0.0 for k in V11_TICK_FEATURES}
    x = np.asarray(data, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = int(float(sr) * _TRIM_S)
    if x.size > 2 * n:
        x = x[n:-n]
    if x.size < sr:
        return out
    x = x - x.mean()

    def band_stats(band: tuple[float, float]) -> tuple[list[float], list[float]]:
        xb = _v11_bandpass(x, sr, band[0], band[1])
        env = np.abs(signal.hilbert(xb))
        return _windowed_apply(env, sr, _excess_envkurt), _windowed_apply(xb, sr, _crest_factor)

    hi_kurts, hi_crests = band_stats(_TICK_HI_BAND)
    lo_kurts, lo_crests = band_stats(_TICK_LOMID_BAND)

    hi_max = max(hi_kurts) if hi_kurts else 0.0
    lo_max = max(lo_kurts) if lo_kurts else 0.0

    out["x_tick_hi_envkurt_max"] = _safe_finite(hi_max)
    out["x_tick_hi_envkurt_p90"] = _safe_finite(float(np.percentile(hi_kurts, 90))) if hi_kurts else 0.0
    out["x_tick_hi_envkurt_mean"] = _safe_finite(float(np.mean(hi_kurts))) if hi_kurts else 0.0
    out["x_tick_hi_envkurt_p75"] = _safe_finite(float(np.percentile(hi_kurts, 75))) if hi_kurts else 0.0
    out["x_tick_hi_crest_max"] = _safe_finite(max(hi_crests)) if hi_crests else 0.0
    out["x_tick_lomid_envkurt_max"] = _safe_finite(lo_max)
    out["x_tick_lomid_crest_max"] = _safe_finite(max(lo_crests)) if lo_crests else 0.0
    out["x_tick_hi_to_lomid_envkurt"] = _safe_finite(hi_max / (abs(hi_max) + abs(lo_max) + 1e-9))
    return out


# ----- 震颤调制：env 通路（v5 Family C env） -----

def _env_chatter_features(x: np.ndarray, sr: float) -> dict[str, float]:
    out = {"env_mod_chatter_lo": 0.0, "env_mod_chatter_hi": 0.0}
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < int(sr):
        return out

    env = np.abs(signal.hilbert(x)).astype(np.float32)
    decim = max(1, int(round(float(sr) / 1000.0)))
    if decim > 1:
        env = signal.decimate(env, decim, ftype="iir", zero_phase=True)
    eff_sr = float(sr) / float(decim)
    if env.size < 16:
        return out

    e = env - env.mean()
    e = e * np.hanning(e.size)
    spec = np.abs(np.fft.rfft(e))
    freqs = np.fft.rfftfreq(e.size, d=1.0 / eff_sr)

    power = spec * spec
    total = float(power[freqs >= 1.0].sum()) + _EPS
    for name, lo, hi in _ENV_MOD_BANDS:
        m = (freqs >= lo) & (freqs < hi)
        out[f"env_mod_{name}"] = _safe_finite(float(power[m].sum() / total))
    return out


# ----- 震颤调制：mel 通路（v5 Family B 主峰突出度 + Family D 跨带同步） -----

def _mel_center_freqs(sr: float, n_mels: int) -> np.ndarray:
    mel_edges = np.linspace(_hz_to_mel(0.0), _hz_to_mel(float(sr) / 2.0), int(n_mels) + 2)
    return _mel_to_hz(mel_edges)[1:int(n_mels) + 1]


def _chatter_row_groups(freq_axis: np.ndarray) -> dict[str, np.ndarray]:
    fmax = float(freq_axis[-1])
    out: dict[str, np.ndarray] = {}
    for name, lo, hi in _CHATTER_BAND_GROUP_HZ:
        hi_ = fmax if hi is None else hi
        if hi is None:
            mask = (freq_axis >= lo) & (freq_axis <= hi_)
        else:
            mask = (freq_axis >= lo) & (freq_axis < hi_)
        idx = np.where(mask)[0]
        if idx.size == 0:
            idx = np.array([int(np.argmin(np.abs(freq_axis - lo)))], dtype=int)
        out[name] = idx
    return out


def _mel_modulation_features(
    mel_power: np.ndarray,
    sr: float,
    hop_length: int,
    n_mels: int,
) -> dict[str, float]:
    out = {
        "mel_mod_mid_peak_prominence": 0.0,
        "mel_mod_high_peak_prominence": 0.0,
        "mel_mod_sync_global": 0.0,
        "mel_mod_band_coherence": 0.0,
    }
    power = np.asarray(mel_power, dtype=np.float64)
    if power.ndim != 2 or power.shape[1] < 16:
        return out

    spec_db = _power_to_db(power)
    n_rows, n_frames = spec_db.shape
    mel_frame_sr = float(sr) / float(hop_length)
    f_hi_all = min(78.0, mel_frame_sr / 2.0)

    centers = _mel_center_freqs(sr, n_mels)[:n_rows]
    row_groups = _chatter_row_groups(centers)

    peak_freqs: list[float] = []
    for g, _, _ in _CHATTER_BAND_GROUP_HZ:
        series = np.mean(spec_db[row_groups[g], :], axis=0)
        s = series - series.mean()
        spec = np.abs(np.fft.rfft(s * np.hanning(s.size)))
        freqs = np.fft.rfftfreq(s.size, d=1.0 / mel_frame_sr)

        m = (freqs >= 1.0) & (freqs < f_hi_all)
        if not np.any(m):
            continue
        sub_f, sub_s = freqs[m], spec[m]
        idx = int(np.argmax(sub_s))
        peak_f = float(sub_f[idx])
        noise = float(np.median(spec[freqs >= 1.0]))
        prom_db = 20.0 * np.log10((float(sub_s[idx]) + _EPS) / (noise + _EPS))
        if peak_f > 0.0:
            peak_freqs.append(peak_f)
        if g in ("mid", "high"):
            out[f"mel_mod_{g}_peak_prominence"] = _safe_finite(prom_db)

    peaks = np.asarray(peak_freqs, dtype=np.float64)
    if peaks.size >= 2 and peaks.mean() > _EPS:
        cv = float(peaks.std() / (peaks.mean() + _EPS))
        out["mel_mod_sync_global"] = float(np.clip(1.0 - cv, 0.0, 1.0))

    # 相邻 mel 行调制谱（>=1 Hz）皮尔森相关
    if n_rows >= 2:
        win = np.hanning(n_frames)
        centered = spec_db - spec_db.mean(axis=1, keepdims=True)
        mod = np.abs(np.fft.rfft(centered * win, axis=1))
        freqs = np.fft.rfftfreq(n_frames, d=1.0 / mel_frame_sr)
        mask = freqs >= 1.0
        if np.any(mask):
            mod = mod[:, mask]
            coeffs: list[float] = []
            for i in range(n_rows - 1):
                a = mod[i] - mod[i].mean()
                b = mod[i + 1] - mod[i + 1].mean()
                denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
                if denom <= _EPS:
                    continue
                coeffs.append(float(np.dot(a, b) / denom))
            if coeffs:
                out["mel_mod_band_coherence"] = _safe_finite(float(np.mean(coeffs)))
    return out


# ----- 震颤 SK burst（v5 Family III，db22 level=4 与验证条件一致） -----

def _sliding_kurtosis(coef: np.ndarray, win: int, step: int) -> np.ndarray:
    """前缀和批量计算滑窗 Pearson 峰度（biased，与 scipy bias=True 一致）。"""
    x = np.asarray(coef, dtype=np.float64).ravel()
    n = x.size
    if n < win + step:
        return np.zeros(0, dtype=np.float64)
    starts = np.arange(0, n - win + 1, step, dtype=int)
    ends = starts + win

    x2 = x * x
    ps1 = np.concatenate(([0.0], np.cumsum(x)))
    ps2 = np.concatenate(([0.0], np.cumsum(x2)))
    ps3 = np.concatenate(([0.0], np.cumsum(x2 * x)))
    ps4 = np.concatenate(([0.0], np.cumsum(x2 * x2)))

    nv = float(win)
    s1 = ps1[ends] - ps1[starts]
    s2 = ps2[ends] - ps2[starts]
    s3 = ps3[ends] - ps3[starts]
    s4 = ps4[ends] - ps4[starts]

    mu = s1 / nv
    e2 = s2 / nv
    e3 = s3 / nv
    e4 = s4 / nv
    m2 = e2 - mu * mu
    m4 = e4 - 4.0 * mu * e3 + 6.0 * mu * mu * e2 - 3.0 * mu ** 4

    with np.errstate(divide="ignore", invalid="ignore"):
        sk = m4 / (m2 * m2)
    return sk[np.isfinite(sk)]


def _max_run_length(mask: np.ndarray) -> int:
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not m.any():
        return 0
    padded = np.concatenate(([0], m.astype(np.int8), [0]))
    d = np.diff(padded)
    runs = np.where(d == -1)[0] - np.where(d == 1)[0]
    return int(runs.max()) if runs.size else 0


def _dwt_sk_burst_features(x: np.ndarray, sr: float) -> dict[str, float]:
    """dwt_3_sk_burst_frac / dwt_4_sk_max_run_len。
    与 v5 验证条件一致：db22 level=4，coeffs[3]=cD2(2.5–5k)、coeffs[4]=cD1(5–10k)。
    pywt 不可用时退化为本文件 db4 level=5 的 d2/d1（频带等价）。"""
    out = {
        "dwt_3_sk_burst_frac": 0.0,
        "dwt_4_sk_burst_frac": 0.0,
        "dwt_4_sk_max_run_len": 0.0,
    }
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < _SK_WIN + _SK_STEP:
        return out

    if pywt is not None:
        coeffs = pywt.wavedec(x, "db22", level=4)
        c3, c4 = np.asarray(coeffs[3], dtype=np.float64), np.asarray(coeffs[4], dtype=np.float64)
    else:
        bands = _dwt_decompose_fallback_db4(x, level=5)
        c3, c4 = bands["d2"][0], bands["d1"][0]

    sk3 = _sliding_kurtosis(c3, _SK_WIN, _SK_STEP)
    sk4 = _sliding_kurtosis(c4, _SK_WIN, _SK_STEP)

    if sk3.size:
        out["dwt_3_sk_burst_frac"] = _safe_finite(float(np.mean(sk3 > _SK_THR)))
    if sk4.size:
        above4 = sk4 > _SK_THR
        out["dwt_4_sk_burst_frac"] = _safe_finite(float(np.mean(above4)))
        out["dwt_4_sk_max_run_len"] = float(_max_run_length(above4))
    return out


# ===================== v11 main entry =====================



# ===================== v12 additions: a5/低频带震颤特征 =====================
# 针对"听得见的低频震颤音"——训练 chatter 是高频冲击型，本组专打 20-312Hz 低频调制。
# 与本文件预处理一致(trim 0.6s + 20Hz 高通)；输入 x 为已预处理信号。
_LF_LO = 20.0
_LF_HI = 312.0
_LF_MOD_LO = 1.0
_LF_MOD_HI = 30.0
_LF_WIN_SEC = 1.0
_LF_HOP_SEC = 0.5
_LF_EPS = 1e-12
_LF_KEYS = (
    "lf_energy_ratio", "lf_env_mod_depth", "lf_env_mod_depth_p90", "lf_env_mod_depth_max",
    "lf_env_mod_peak_ratio", "lf_env_mod_freq", "lf_env_mod_spec_entropy",
    "lf_spec_centroid", "lf_zcr_cv", "lf_crest",
)


def _lf_bandpass(x: np.ndarray, sr: float, lo: float, hi: float) -> np.ndarray:
    nyq = 0.5 * float(sr)
    lo_n = max(lo / nyq, 1e-5)
    hi_n = min(hi / nyq, 0.999)
    if hi_n <= lo_n:
        return np.asarray(x, dtype=np.float64)
    sos = signal.butter(4, [lo_n, hi_n], btype="band", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def _lf_agg(vals, name):
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"{name}_p90": 0.0, f"{name}_max": 0.0}
    return {f"{name}_p90": float(np.percentile(a, 90)), f"{name}_max": float(np.max(a))}


def _extract_lowfreq_features(x_pre: np.ndarray, sr: float) -> dict:
    x = np.asarray(x_pre, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    out = {k: 0.0 for k in _LF_KEYS}
    if x.size < int(sr * 0.5):
        return out

    xb = _lf_bandpass(x, sr, _LF_LO, _LF_HI)
    e_low = float(np.sum(xb ** 2))
    e_all = float(np.sum(x ** 2)) + _LF_EPS
    out["lf_energy_ratio"] = e_low / e_all

    env = np.abs(signal.hilbert(xb))
    mu = float(np.mean(env)) + _LF_EPS
    out["lf_env_mod_depth"] = float(np.std(env)) / mu

    win = max(4, int(round(_LF_WIN_SEC * sr)))
    hop = max(1, int(round(_LF_HOP_SEC * sr)))
    depths = []
    for i in range(0, max(1, env.size - win + 1), hop):
        seg = env[i:i + win]
        if seg.size < 4:
            continue
        m = float(np.mean(seg)) + _LF_EPS
        depths.append(float(np.std(seg)) / m)
    agg = _lf_agg(depths, "lf_env_mod_depth")
    out["lf_env_mod_depth_p90"] = agg["lf_env_mod_depth_p90"]
    out["lf_env_mod_depth_max"] = agg["lf_env_mod_depth_max"]

    env0 = env - np.mean(env)
    n = env0.size
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    P = np.abs(np.fft.rfft(env0)) ** 2
    band = (freqs >= _LF_MOD_LO) & (freqs <= _LF_MOD_HI)
    if np.any(band):
        Pb = P[band]; fb = freqs[band]
        tot = float(np.sum(Pb)) + _LF_EPS
        pk = int(np.argmax(Pb))
        out["lf_env_mod_peak_ratio"] = float(Pb[pk]) / tot
        out["lf_env_mod_freq"] = float(fb[pk])
        p = Pb / tot
        out["lf_env_mod_spec_entropy"] = float(-np.sum(p * np.log(p + _LF_EPS)) / np.log(len(p) + _LF_EPS))

    F = np.fft.rfftfreq(x.size, d=1.0 / sr)
    PX = np.abs(np.fft.rfft(xb)) ** 2
    out["lf_spec_centroid"] = float(np.sum(F * PX) / (np.sum(PX) + _LF_EPS))

    zs = []
    for i in range(0, max(1, xb.size - win + 1), hop):
        seg = xb[i:i + win]
        if seg.size < 4:
            continue
        zs.append(float(np.mean(np.abs(np.diff(np.sign(seg))) > 0)))
    za = np.asarray(zs, dtype=np.float64)
    out["lf_zcr_cv"] = float(np.std(za) / (np.mean(za) + _LF_EPS)) if za.size else 0.0

    rms = float(np.sqrt(np.mean(xb ** 2))) + _LF_EPS
    out["lf_crest"] = float(np.max(np.abs(xb))) / rms
    return out
# =================== end v12 additions ===================

def extract_features_v12(
    data: np.ndarray,
    sr: float,
    return_timing: bool = False,
    *,
    n_mels: int = _DEFAULT_N_MELS,
    n_fft: int = _DEFAULT_N_FFT,
    hop_length: int | None = None,
    trim_s: float = _TRIM_S,
) -> dict[str, float] | tuple[dict[str, float], dict[str, float]]:
    """v12 = v11(182) + a5/低频带震颤特征 10 = 192 维。（独立文件，不 import v11）

    与 v10 的差异：
      1. 裁瞬态统一为时间制 trim_s=0.6s（v10 为 cut_len=8000 点 ≈ 0.4s@20k）。
      2. 回挂 v7/v8 实测有效的 weak_tick 双带分窗特征（8 维）。
      3. 回挂 v5 实测有效的震颤调制结构特征（9 维）。
    """
    cut_len = int(round(float(trim_s) * float(sr)))

    base = _extract_base_features(
        data,
        sr,
        return_timing=return_timing,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        cut_len=cut_len,
    )
    if return_timing:
        features, timing = base
    else:
        features, timing = base, {}

    t0 = time.perf_counter()
    features.update(extract_v11_tick(data, sr))
    timing["v11_tick_sec"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    x_raw = np.asarray(data, dtype=np.float64).ravel()
    x_raw = x_raw[np.isfinite(x_raw)]
    x = process_data(x_raw, sr=sr, cut_len=cut_len, cutoff_low=20, cutoff_high=None)
    x = np.asarray(x, dtype=np.float64)

    hop = int(hop_length) if hop_length is not None else int(n_fft) // 2
    mel_power = _melspectrogram(x, sr=float(sr), n_mels=int(n_mels), n_fft=int(n_fft), hop_length=hop)

    features.update(_env_chatter_features(x, sr))
    features.update(_mel_modulation_features(mel_power, sr, hop, int(n_mels)))
    features.update(_dwt_sk_burst_features(x, sr))
    features.update(_extract_lowfreq_features(x, sr))
    timing["v12_chatter_lf_sec"] = float(time.perf_counter() - t0)

    if return_timing:
        timing["total_sec"] = float(
            timing.get("total_sec", 0.0)
            + timing["v11_tick_sec"]
            + timing["v12_chatter_lf_sec"]
        )
        return features, timing
    return features


extract_features = extract_features_v12


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    demo = rng.standard_normal(100000).astype(np.float32)

    feats, timing = extract_features_v12(demo, 20000, return_timing=True)

    print(f"feature_count = {len(feats)} (期望 182)")
    print(f"timing = {timing}")
    print("\n弱秒表 8 维:")
    for k in V11_TICK_FEATURES:
        print(f"{k:>32s} = {feats[k]:.6g}")
    print("\n震颤调制 9 维:")
    for k in V11_CHATTER_FEATURES:
        print(f"{k:>32s} = {feats[k]:.6g}")
