"""Feature extraction v10.

Feature groups:
  1. Global time-domain:
     mean, std, duration, skewness, kurtosis, zero_crossing_rate

  2. Mel-band basic statistics:
     melband_basic_{low,mid,high,all}_{std,ratio,kurtosis,impulse,entropy}

  3. Mel-band windowed impulse statistics:
     melband_{low,mid,high,all}_win_{kurtosis,crest,impulse}_{mean,p75,p90,max}

  4. Mel shock peaks:
     mel_shock_*

  5. DWT-band windowed statistics:
     dwtband_{a5,d5,d4,d3,d2,d1}_win_{kurtosis,zcr,std}_{mean,p90,max}

Default feature count:
  global: 6
  melband_basic: 4 * 5 = 20
  melband window: 4 * 3 * 4 = 48
  mel_shock: 37
  DWT window: 6 * 3 * 3 = 54
  total: 165

Important:
  DWT window features are computed directly on DWT coefficients.
  No per-window de-meaning, no per-window standardization, no smoothing.
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

def extract_features_v10(
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


extract_features = extract_features_v10


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    demo = rng.standard_normal(100000).astype(np.float32)

    feats, timing = extract_features_v10(
        demo,
        20000,
        return_timing=True,
    )

    print(f"feature_count = {len(feats)}")
    print(f"timing = {timing}")

    print("\nFirst 20 features:")
    for k in list(feats)[:20]:
        print(f"{k:>42s} = {feats[k]:.6g}")

    print("\nMel-band basic examples:")
    for k in [key for key in feats if key.startswith("melband_basic_")][:20]:
        print(f"{k:>42s} = {feats[k]:.6g}")

    print("\nDWT examples:")
    for k in [key for key in feats if key.startswith("dwtband_")][:18]:
        print(f"{k:>42s} = {feats[k]:.6g}")