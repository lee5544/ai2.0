"""特征提取 v7 —— 自包含版(standalone)。

本文件不 import 任何同目录特征模块(extract_features_v6 / v6_selected_features /
v7_selected_features 全部内联),可单文件放入 runtime bundle 而无依赖链。
内容 = v6 全部特征代码 + v7 的 6 震颤 + 6 弱秒表特征; 输出 237 维, 顺序按
V7_SELECTED_FEATURES。入口: extract_features_v7(data, sr, return_timing=False)。
注: 与模块化版数值完全一致(构建时已逐键校验)。
"""

from __future__ import annotations

# ===== 内联冻结特征清单(替代 v6_selected_features / v7_selected_features) =====
V6_SELECTED_FEATURES: tuple[str, ...] = (
    'mean',
    'std',
    'duration',
    'skewness',
    'kurtosis',
    'zero_crossing_rate',
    'dwt_0_rms',
    'dwt_0_energy_ratio',
    'dwt_0_q95',
    'dwt_0_log_energy_entropy',
    'dwt_0_shock_count_rate',
    'dwt_0_shock_stalta_mean',
    'dwt_0_shock_stalta_q95',
    'dwt_0_shock_iat_mean_ms',
    'dwt_0_shock_iat_cv',
    'dwt_0_sk_max',
    'dwt_0_sk_mean',
    'dwt_0_sk_q95',
    'dwt_1_kurt',
    'dwt_1_log_energy_entropy',
    'dwt_1_shock_kurt_mean',
    'dwt_1_shock_stalta_mean',
    'dwt_1_shock_stalta_q95',
    'dwt_1_shock_iat_cv',
    'dwt_1_shock_iat_periodicity',
    'dwt_1_sk_time_to_max',
    'dwt_2_rms',
    'dwt_2_energy_ratio',
    'dwt_2_kurt',
    'dwt_2_q95',
    'dwt_2_shock_count_rate',
    'dwt_2_shock_stalta_mean',
    'dwt_2_shock_stalta_q95',
    'dwt_2_sk_mean',
    'dwt_2_sk_time_to_max',
    'dwt_3_rms',
    'dwt_3_energy_ratio',
    'dwt_3_kurt',
    'dwt_3_crest',
    'dwt_3_q95',
    'dwt_3_impulse',
    'dwt_3_log_energy_entropy',
    'dwt_3_shock_count_rate',
    'dwt_3_shock_kurt_mean',
    'dwt_3_shock_kurt_q95',
    'dwt_3_shock_stalta_mean',
    'dwt_3_shock_stalta_q95',
    'dwt_3_shock_iat_mean_ms',
    'dwt_3_shock_iat_periodicity',
    'dwt_3_sk_max',
    'dwt_3_sk_mean',
    'dwt_3_sk_q95',
    'dwt_3_sk_time_to_max',
    'dwt_4_rms',
    'dwt_4_crest',
    'dwt_4_impulse',
    'dwt_4_log_energy_entropy',
    'dwt_4_shock_count_rate',
    'dwt_4_shock_kurt_q95',
    'dwt_4_shock_stalta_q95',
    'dwt_4_shock_iat_mean_ms',
    'dwt_4_shock_iat_cv',
    'dwt_4_shock_iat_periodicity',
    'dwt_4_sk_max',
    'dwt_4_sk_q95',
    'dwt_4_sk_time_to_max',
    'dwt_4_energy_center',
    'dwt_4_env_chatter_lo',
    'dwt_4_env_peak_f',
    'dwt_4_env_kurt',
    'dwt_3_sk_burst_frac',
    'mel_band_all_frame_mean_std',
    'mel_band_low_db_std',
    'mel_band_low_db_ratio',
    'mel_band_mid_db_mean',
    'mel_band_mid_db_std',
    'mel_band_high_db_mean',
    'mel_band_wide_std',
    'mel_band_hml_balance',
    'mel_shock_low_kurtosis_q95',
    'mel_shock_low_kurtosis_q50',
    'mel_shock_low_response_db_mean',
    'mel_shock_low_response_db_std',
    'mel_shock_low_response_db_q95',
    'mel_shock_low_interval_cv',
    'mel_shock_med_count_per_sec',
    'mel_shock_med_kurtosis_q95',
    'mel_shock_med_kurtosis_q50',
    'mel_shock_med_response_db_std',
    'mel_shock_med_response_db_q95',
    'mel_shock_med_response_db_q50',
    'mel_shock_med_interval_cv',
    'mel_shock_high_count_per_sec',
    'mel_shock_high_kurtosis_mean',
    'mel_shock_high_kurtosis_q95',
    'mel_shock_high_kurtosis_q50',
    'mel_shock_high_response_db_mean',
    'mel_shock_high_response_db_std',
    'mel_shock_high_response_db_q95',
    'mel_shock_high_interval_cv',
    'mel_shock_all_count_per_sec',
    'mel_shock_all_kurtosis_mean',
    'mel_shock_all_kurtosis_q95',
    'mel_shock_all_kurtosis_q50',
    'mel_shock_all_response_db_mean',
    'mel_shock_all_response_db_std',
    'mel_shock_all_response_db_q95',
    'mel_shock_all_response_db_q50',
    'mel_shock_all_interval_cv',
    'mel_shock_high_ratio',
    'mel_loud_low_temporal_kurt',
    'mel_loud_low_temporal_acf1',
    'mel_loud_mid_dynamic_range_db',
    'mel_loud_mid_temporal_kurt',
    'mel_loud_mid_temporal_acf1',
    'mel_loud_high_dynamic_range_db',
    'mel_loud_high_temporal_acf1',
    'mel_mod_low_peak_f_hz',
    'mel_mod_low_shudder',
    'mel_mod_low_rough_lo',
    'mel_mod_low_rough_hi',
    'mel_mod_mid_peak_f_hz',
    'mel_mod_mid_peak_prominence',
    'mel_mod_mid_shudder',
    'mel_mod_mid_rough_hi',
    'mel_mod_high_peak_f_hz',
    'mel_mod_high_peak_prominence',
    'mel_mod_high_rough_hi',
    'mel_mod_sync_global',
    'mel_mod_band_coherence',
    'env_mod_shudder',
    'env_mod_rough',
    'env_mod_chatter_lo',
    'env_mod_chatter_hi',
    'mel13_band_all_frame_mean_std',
    'mel13_band_low_db_mean',
    'mel13_band_low_db_std',
    'mel13_band_low_db_ratio',
    'mel13_band_mid_db_std',
    'mel13_band_mid_db_ratio',
    'mel13_band_high_db_mean',
    'mel13_band_high_db_std',
    'mel13_band_high_db_ratio',
    'mel13_band_wide_std',
    'mel13_shock_low_response_db_q95',
    'mel13_shock_med_count_per_sec',
    'mel13_shock_med_kurtosis_q95',
    'mel13_shock_med_kurtosis_q50',
    'mel13_shock_med_response_db_std',
    'mel13_shock_med_response_db_q95',
    'mel13_shock_med_response_db_q50',
    'mel13_shock_high_count_per_sec',
    'mel13_shock_high_kurtosis_mean',
    'mel13_shock_high_kurtosis_q50',
    'mel13_shock_high_response_db_mean',
    'mel13_shock_high_response_db_q95',
    'mel13_shock_all_count_per_sec',
    'mel13_shock_all_kurtosis_mean',
    'mel13_shock_all_kurtosis_q95',
    'mel13_shock_all_kurtosis_q50',
    'mel13_shock_all_response_db_mean',
    'mel13_shock_all_response_db_std',
    'mel13_shock_all_response_db_q95',
    'mel13_shock_high_ratio',
    'mel13_loud_low_temporal_kurt',
    'mel13_loud_low_temporal_acf1',
    'mel13_loud_mid_dynamic_range_db',
    'mel13_loud_mid_temporal_kurt',
    'mel13_loud_mid_temporal_acf1',
    'mel13_loud_high_dynamic_range_db',
    'mel13_loud_high_temporal_kurt',
    'mel13_loud_high_temporal_acf1',
    'mel13_mod_low_peak_f_hz',
    'mel13_mod_low_peak_prominence',
    'mel13_mod_low_shudder',
    'mel13_mod_mid_shudder',
    'mel13_mod_mid_rough_hi',
    'mel13_mod_high_peak_prominence',
    'mel13_mod_high_shudder',
    'mel13_mod_high_rough_hi',
    'mel13_mod_band_coherence',
    'v6_spec_centroid_hz',
    'v6_spec_flatness',
    'v6_spec_low_ratio',
    'v6_spec_high_ratio',
    'v6_spec_high_low_ratio',
    'v6_env_peak_f_hz',
    'v6_env_pulse_rate',
    'v6_env_pulse_iat_cv',
    'v6_seg_rms_cv',
    'v6_seg_rms_max_min_ratio',
    'v6_seg_high_ratio_cv',
    'v6_tick_weak_count_per_sec',
    'v6_tick_weak_interval_median_ms',
    'v6_tick_weak_interval_cv',
    'v6_tick_weak_interval_mad_ratio',
    'v6_tick_weak_interval_inlier_ratio',
    'v6_tick_strong_count_per_sec',
    'v6_tick_strong_interval_median_ms',
    'v6_tick_strong_interval_cv',
    'v6_tick_strong_interval_mad_ratio',
    'v6_tick_strong_interval_inlier_ratio',
    'v6_tick_autocorr_peak',
    'v6_tick_autocorr_lag_ms',
    'v6_tick_period_snr',
    'v6_tick_window_periodicity_mean',
    'v6_tick_window_periodicity_max',
    'v6_tick_periodic_window_ratio',
    'v6_tick_xwin_period_consistency',
    'v6_tick_strong_to_weak_count_ratio',
    'v6_tick_teager_count_per_sec',
    'v6_tick_teager_inlier_ratio',
    'v6_tick_teager_autocorr_peak',
    'v6_rel_mel_low_kurt_to_all',
    'v6_rel_mel_med_count_to_all',
    'v6_rel_mel_med_response_to_all',
    'v6_rel_mel_high_kurt_to_all',
    'v6_rel_mel_high_response_to_all',
    'v6_rel_mel_high_response_to_dynamic_range',
    'v6_rel_mel_high_to_low_count',
    'v6_rel_mel13_med_response_to_all',
    'v6_rel_mel13_high_response_to_dynamic_range',
    'v6_rel_med_interval_cv_mel13_to_mel26',
    'v6_rel_high_kurtosis_q95_mel13_to_mel26',
    'v6_rel_all_response_db_q95_mel13_to_mel26',
)

_V7_EXTRA: tuple[str, ...] = (
    'x_lf_ratio_125_250',
    'x_lf_ratio_250_500',
    'x_env_lowband_ratio',
    'x_spec_centroid_hz',
    'x_env_spec_entropy',
    'x_env_mod_freq_hz',
)

_V7_TICK: tuple[str, ...] = (
    'x_tick_hi_envkurt_max',
    'x_tick_hi_envkurt_p90',
    'x_tick_hi_crest_max',
    'x_tick_lomid_envkurt_max',
    'x_tick_lomid_crest_max',
    'x_tick_hi_to_lomid_envkurt',
)

V7_SELECTED_FEATURES: tuple[str, ...] = tuple(V6_SELECTED_FEATURES) + _V7_EXTRA + _V7_TICK

# ===== v6 特征实现(内联) =====
import os
import sys
import time
from functools import lru_cache, wraps
from typing import Iterable

import numpy as np
import pywt
import scipy.signal as signal
from scipy.signal import find_peaks



def _disable_numba_cache_for_librosa():
    """Python 3.13 + librosa/numba 组合下 cache=True 会触发 'no locator available'。
    仅关闭装饰器缓存，不影响数值结果。"""
    try:
        import numba
    except Exception:
        return

    def _wrap_no_cache(decorator):
        if getattr(decorator, "_forvia_cache_patched", False):
            return decorator

        @wraps(decorator)
        def _wrapped(*args, **kwargs):
            kwargs["cache"] = False
            return decorator(*args, **kwargs)

        _wrapped._forvia_cache_patched = True
        return _wrapped

    numba.jit = _wrap_no_cache(numba.jit)
    numba.njit = _wrap_no_cache(numba.njit)
    numba.vectorize = _wrap_no_cache(numba.vectorize)
    numba.guvectorize = _wrap_no_cache(numba.guvectorize)


_disable_numba_cache_for_librosa()
import librosa  # noqa: E402  - 必须在 numba 补丁之后导入

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ===================== ACC 数据预处理 =====================
@lru_cache(maxsize=128)
def _butter_coeffs(sr: float, cutoff: float, btype: str, order: int):
    nyquist = sr / 2
    normal_cutoff = cutoff / nyquist
    return signal.butter(order, normal_cutoff, btype=btype, analog=False)


def butter_filter(data, sr, cutoff, btype="low", order=4):
    b, a = _butter_coeffs(float(sr), float(cutoff), str(btype), int(order))
    return signal.filtfilt(b, a, data).astype(np.float32)


def process_data(
    raw_data,
    sr=20000,
    cut_len=None,
    target_length=0,
    cutoff_low=20,
    cutoff_high=None,
):
    """高/低通滤波 + 首尾裁剪，与 v2 保持一致。"""
    if cutoff_low is not None:
        raw_data = butter_filter(raw_data, sr=sr, cutoff=cutoff_low, btype="high")
    if cutoff_high is not None:
        raw_data = butter_filter(raw_data, sr=sr, cutoff=cutoff_high, btype="low")

    cut_len = round(float(sr) * 0.5) if cut_len is None else max(0, int(cut_len))
    if len(raw_data) > 2 * cut_len:
        dat_cut = raw_data[cut_len:-cut_len]
    else:
        dat_cut = raw_data[:]

    if target_length and len(dat_cut) > target_length:
        dat_cut = dat_cut[:target_length]
    return dat_cut


# ===================== 基础统计 helpers =====================
_EPS = 1e-12


def _analytic_envelope(x: np.ndarray) -> np.ndarray:
    """Hilbert 解析包络 |hilbert(x)|（原长 FFT，数值与旧实现完全一致）。

    整段提取只需调用一次，供包络谱/包络主峰/秒表周期等多个消费者共享，
    避免对同一信号重复做 Hilbert。
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        return np.abs(x)
    return np.abs(signal.hilbert(x))


def _zero_crossing_rate(x: np.ndarray) -> float:
    """过零率：相邻样本符号变化数 / (N-1)。零样本不算过零。"""
    if x.size < 2:
        return 0.0
    s = np.sign(x)
    nz = s != 0
    if nz.sum() < 2:
        return 0.0
    s_nz = s[nz]
    return float(np.sum(s_nz[1:] != s_nz[:-1])) / (x.size - 1)


def _safe_skew(x: np.ndarray) -> float:
    """无偏不校正的 Pearson 偏度，标准差为 0 时返 0。"""
    if x.size < 3:
        return 0.0
    m = float(np.mean(x))
    sd = float(np.std(x))
    if sd <= 0:
        return 0.0
    z = (x - m) / sd
    return float(np.mean(z * z * z))


def _safe_kurt(x: np.ndarray) -> float:
    """Pearson 峰度 (normal=3)，标准差为 0 时返 0。"""
    if x.size < 4:
        return 0.0
    m = float(np.mean(x))
    sd = float(np.std(x))
    if sd <= 0:
        return 0.0
    z = (x - m) / sd
    z2 = z * z
    return float(np.mean(z2 * z2))


# ===================== 批量统计 helpers（mel 冲击检测用）=====================
def _prefix_sum_sq(x: np.ndarray) -> np.ndarray:
    """返回 x**2 的前缀和（含 0）。sum_sq([s, e)) = ps[e] - ps[s]"""
    x = np.asarray(x, dtype=float, order="C")
    ps = np.empty(x.size + 1, dtype=float)
    ps[0] = 0.0
    np.cumsum(x * x, out=ps[1:])
    return ps


def _prefix_sum_pow4(x: np.ndarray):
    """返回 x, x^2, x^3, x^4 的前缀和（含 0）。"""
    x = np.asarray(x, dtype=float, order="C")
    ps1 = np.empty(x.size + 1, dtype=float); ps1[0] = 0.0
    ps2 = np.empty(x.size + 1, dtype=float); ps2[0] = 0.0
    ps3 = np.empty(x.size + 1, dtype=float); ps3[0] = 0.0
    ps4 = np.empty(x.size + 1, dtype=float); ps4[0] = 0.0
    x2 = x * x
    np.cumsum(x, out=ps1[1:])
    np.cumsum(x2, out=ps2[1:])
    np.cumsum(x2 * x, out=ps3[1:])
    np.cumsum(x2 * x2, out=ps4[1:])
    return ps1, ps2, ps3, ps4


def _segment_kurtosis_from_prefix(
    ps1: np.ndarray,
    ps2: np.ndarray,
    ps3: np.ndarray,
    ps4: np.ndarray,
    lefts: np.ndarray,
    rights: np.ndarray,
    *,
    fisher: bool = False,
    bias: bool = True,
) -> np.ndarray:
    """用前缀和批量计算区间 Pearson kurtosis。与 scipy.stats.kurtosis 一致。"""
    l = np.asarray(lefts, dtype=int)
    r = np.asarray(rights, dtype=int)
    n = (r - l).astype(float)

    out = np.zeros_like(n, dtype=float)
    valid = n > 3
    if not np.any(valid):
        return out

    lv = l[valid]
    rv = r[valid]
    nv = n[valid]

    s1 = ps1[rv] - ps1[lv]
    s2 = ps2[rv] - ps2[lv]
    s3 = ps3[rv] - ps3[lv]
    s4 = ps4[rv] - ps4[lv]

    mu = s1 / nv
    e2 = s2 / nv
    e3 = s3 / nv
    e4 = s4 / nv

    m2 = e2 - mu * mu
    m4 = e4 - 4.0 * mu * e3 + 6.0 * mu * mu * e2 - 3.0 * mu * mu * mu * mu

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = m4 / (m2 * m2)
        if bias:
            kurt_v = ratio
        else:
            n2 = nv * nv
            kurt_v = (
                (n2 - 1.0) * ratio - 3.0 * (nv - 1.0) * (nv - 1.0)
            ) / ((nv - 2.0) * (nv - 3.0)) + 3.0
        if fisher:
            kurt_v = kurt_v - 3.0

    kurt_v = np.where(m2 <= 0.0, np.nan, kurt_v)
    out[valid] = kurt_v
    return out


def _rolling_mean_std_numpy(arr: np.ndarray, win: int):
    """等价 pandas rolling(win, center=True, min_periods=1).mean()/std().fillna(0)。"""
    x = np.asarray(arr, dtype=float, order="C")
    N = x.size
    if N == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    win = max(1, int(win))
    win_l = win // 2
    win_r = win - win_l

    cs = np.empty(N + 1, dtype=float); cs[0] = 0.0
    np.cumsum(x, out=cs[1:])
    cs2 = np.empty(N + 1, dtype=float); cs2[0] = 0.0
    np.cumsum(x * x, out=cs2[1:])

    idx = np.arange(N)
    starts = np.clip(idx - win_l, 0, N)
    ends = np.clip(idx + win_r, 0, N)
    ns = ends - starts

    sums = cs[ends] - cs[starts]
    means = sums / np.maximum(ns, 1)

    sumsq = cs2[ends] - cs2[starts]
    denom_n = np.maximum(ns, 1)
    var_num = sumsq - (sums * sums) / denom_n

    denom_var = np.maximum(ns - 1, 1)
    var = var_num / denom_var
    var[ns < 2] = 0.0
    var[var < 0] = 0.0

    stds = np.sqrt(var)
    return means, stds


def _iir_local_mean(
    row: np.ndarray,
    *,
    alpha: float | None = None,
    equivalent_window: int = 50,
) -> np.ndarray:
    """因果一阶 IIR 低通：y[n] = α x[n] + (1-α) y[n-1]。
    alpha 未指定时按 SMA↔EMA 等效系数 α = 2/(W+1) 推导。
    用 `signal.lfilter_zi` 把初始状态预热成首样本值，消除起始瞬态。
    """
    x = np.asarray(row, dtype=float, order="C")
    if x.size == 0:
        return x.copy()
    if alpha is None:
        alpha = 2.0 / (max(int(equivalent_window), 1) + 1)
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    b = np.array([alpha], dtype=float)
    a = np.array([1.0, -(1.0 - alpha)], dtype=float)
    zi = signal.lfilter_zi(b, a) * float(x[0])
    y, _ = signal.lfilter(b, a, x, zi=zi)
    return y.astype(x.dtype, copy=False)


# ===================== STA/LTA 与 DWT 冲击特征 =====================
def _zero_cross_prefix_sum(x: np.ndarray) -> np.ndarray:
    """累计过零次数前缀和。
    返回 cs，长度 N，cs[i] = x[:i+1] 中相邻样本符号变化数。
    `# crossings in x[a:b]` ≈ cs[min(b-1, N-1)] - cs[a]。
    """
    N = x.size
    cs = np.zeros(N, dtype=np.int32)
    if N < 2:
        return cs
    s = np.sign(x)
    sign_change = (s[1:] != s[:-1]) & (s[1:] != 0) & (s[:-1] != 0)
    cs[1:] = np.cumsum(sign_change.astype(np.int32))
    return cs


def _zero_cross_per_peak(
    cs: np.ndarray, peaks: np.ndarray, half_window: int, N: int
) -> np.ndarray:
    """每个 peak ±half_window 样本范围内的过零次数（向量化）。"""
    if peaks.size == 0:
        return np.array([], dtype=np.int32)
    starts = np.maximum(peaks - int(half_window), 0)
    ends = np.minimum(peaks + int(half_window), N - 1)
    return cs[ends] - cs[starts]



def sta_lta_ratio_batch(
    x: np.ndarray,
    centers: np.ndarray,
    sta_win: int,
    lta_win: int,
    ps2: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """向量化批量计算多个 center 的 STA/LTA 比值（与 v2 等价）。"""
    x = np.asarray(x, dtype=float, order="C")
    centers = np.asarray(centers, dtype=int)
    N = x.size

    sta_win = max(1, int(sta_win))
    lta_win = max(sta_win + 1, int(lta_win))

    if ps2 is None:
        ps2 = _prefix_sum_sq(x)

    def _ms_range(_ps2, s, e):
        length = np.maximum(e - s, 1)
        return (_ps2[e] - _ps2[s]) / length

    sta_l = sta_win // 2
    sta_r = sta_win - sta_l
    s1 = np.clip(centers - sta_l, 0, N)
    e1 = np.clip(centers + sta_r, 0, N)
    sta_ms = _ms_range(ps2, s1, e1)

    lta_l = lta_win // 2
    lta_r = lta_win - lta_l
    s2 = np.clip(centers - lta_l, 0, N)
    e2 = np.clip(centers + lta_r, 0, N)

    left_len = np.maximum(s1 - s2, 0)
    right_len = np.maximum(e2 - e1, 0)
    denom = left_len + right_len

    left_ms = np.zeros_like(sta_ms)
    right_ms = np.zeros_like(sta_ms)

    left_mask = left_len > 0
    if np.any(left_mask):
        left_ms[left_mask] = _ms_range(ps2, s2[left_mask], s1[left_mask])

    right_mask = right_len > 0
    if np.any(right_mask):
        right_ms[right_mask] = _ms_range(ps2, e1[right_mask], e2[right_mask])

    lta_ms = np.zeros_like(sta_ms)
    ok = denom > 0
    if np.any(ok):
        lta_ms[ok] = (left_ms[ok] * left_len[ok] + right_ms[ok] * right_len[ok]) / denom[ok]

    bad = ~ok
    if np.any(bad):
        lta_ms[bad] = _ms_range(ps2, s2[bad], e2[bad])

    return sta_ms / (lta_ms + eps)


def _get_dwt_shock_features(
    data,
    window: int,
    prefix: str,
    sr_eff: float | None = None,
) -> dict:
    """检测单层小波系数上的冲击事件，返回 8 个统计。

    输出键 (8)：
      count_rate          冲击数 / 信号秒数 (用层有效 sr 算时长)
      kurt_mean, kurt_q95 每峰 ±W/2 局部 Pearson 峰度的中心+极端
      stalta_mean, stalta_q95
                          每峰 STA/LTA 能量比 (sta=10, lta=50 样本) 的中心+极端
      iat_mean_ms         冲击间隔均值 (ms)，对应特征频率
      iat_cv              IAT 的变异系数：→0 周期，→1 随机，>>1 bursty
      iat_periodicity     IAT 落在中位数 ±20% 内的比例 ∈[0,1]
    """
    x = np.asarray(data, dtype=float, order="C")
    abs_x = np.abs(x)
    N = x.size

    # 阈值与 prominence
    W_thresh = 2000
    local_mean, local_std = _rolling_mean_std_numpy(abs_x, W_thresh)
    thresholds = local_mean + local_std
    prom_req = 2.0 * np.maximum(local_std, _EPS)

    # |x| 上找峰
    peaks, _props = find_peaks(
        abs_x,
        distance=100,
        height=thresholds,
        prominence=prom_req,
        wlen=512,
    )

    # 每峰 STA/LTA + 局部 kurtosis（在原始有符号 x 上算）
    half = max(1, window // 2)
    ps2 = _prefix_sum_sq(x)
    stalta_arr = sta_lta_ratio_batch(
        x, centers=peaks, sta_win=10, lta_win=50, ps2=ps2,
    )

    k_arr = np.array([], dtype=float)
    if peaks.size:
        lefts = np.maximum(peaks - half, 0)
        rights = np.minimum(peaks + half, N)
        ps1, ps2_m, ps3, ps4 = _prefix_sum_pow4(x)
        k_arr = _segment_kurtosis_from_prefix(
            ps1, ps2_m, ps3, ps4, lefts, rights, fisher=False, bias=True,
        )

    e_arr = np.asarray(stalta_arr, dtype=float)
    M = k_arr.size

    # count_rate：冲击数 / 子带物理时长（秒）；没 sr_eff 就退回样本时长
    duration_sec = (N / sr_eff) if (sr_eff and sr_eff > 0) else max(N, 1)
    count_rate = float(M / max(duration_sec, _EPS))

    # IAT
    if peaks.size >= 3:
        iat = np.diff(peaks).astype(float)
        iat_mean = float(iat.mean())
        iat_cv = float(iat.std() / (iat_mean + _EPS))
        iat_median = float(np.median(iat))
        if iat_median > 0:
            in_band = np.abs(iat - iat_median) <= 0.2 * iat_median
            iat_periodicity = float(in_band.mean())
        else:
            iat_periodicity = 0.0
        iat_mean_ms = (
            iat_mean / sr_eff * 1000.0 if sr_eff and sr_eff > 0 else iat_mean
        )
    else:
        iat_mean_ms = 0.0
        iat_cv = 0.0
        iat_periodicity = 0.0

    if M:
        k_mean = float(k_arr.mean())
        k_q95 = float(np.percentile(k_arr, 95))
        s_mean = float(e_arr.mean())
        s_q95 = float(np.percentile(e_arr, 95))
    else:
        k_mean = k_q95 = s_mean = s_q95 = 0.0

    return {
        f"{prefix}_count_rate": count_rate,
        f"{prefix}_kurt_mean": k_mean,
        f"{prefix}_kurt_q95": k_q95,
        f"{prefix}_stalta_mean": s_mean,
        f"{prefix}_stalta_q95": s_q95,
        f"{prefix}_iat_mean_ms": float(iat_mean_ms),
        f"{prefix}_iat_cv": float(iat_cv),
        f"{prefix}_iat_periodicity": float(iat_periodicity),
    }


# ===================== Mel 频带 & 冲击特征 =====================
def _get_bands_features(mel_spectrogram_db, freq_axis, sr, prefix):
    """低/中/高频带统计 + 两个聚合谱形特征。"""
    bands = [
        ("low", 100, 1500),
        ("mid", 1500, 4000),
        ("high", 4000, sr / 2),
    ]
    mel_spectrogram_db = np.asarray(mel_spectrogram_db, dtype=float, order="C")
    freq_axis = np.asarray(freq_axis, dtype=float, order="C")
    n_mels, _ = mel_spectrogram_db.shape
    assert freq_axis.size == n_mels, "freq_axis 长度必须与 mel 频带数一致"
    assert np.all(np.diff(freq_axis) >= 0), "freq_axis 应为单调非降"

    feats: dict = {}

    frame_mean_db = np.mean(mel_spectrogram_db, axis=0)
    feats[f"{prefix}_all_frame_mean_std"] = float(np.std(frame_mean_db))

    mel_power = np.power(10.0, mel_spectrogram_db / 10.0)
    total_power = float(np.sum(mel_power)) + 1e-12

    fmin, fmax = float(freq_axis[0]), float(freq_axis[-1])
    band_stats: dict[str, dict[str, float]] = {}
    for band_name, lo, hi in bands:
        lo_ = max(lo, fmin)
        hi_ = min(hi, fmax)
        i0 = int(np.searchsorted(freq_axis, lo_, side="left"))
        i1 = int(np.searchsorted(freq_axis, hi_, side="right"))
        if i1 - i0 < 1:
            i0 = max(i0 - 1, 0)
            i1 = min(i0 + 1, n_mels)

        band_db_series = np.mean(mel_spectrogram_db[i0:i1, :], axis=0)
        band_power_sum = float(np.sum(mel_power[i0:i1, :]))
        band_power_ratio = band_power_sum / total_power

        key = f"{prefix}_{band_name}"
        stats = {
            "db_mean": float(np.mean(band_db_series)),
            "db_std": float(np.std(band_db_series)),
            "db_ratio": band_power_ratio,
        }
        band_stats[band_name] = stats
        feats.update({
            f"{key}_db_mean": stats["db_mean"],
            f"{key}_db_std": stats["db_std"],
            f"{key}_db_ratio": stats["db_ratio"],
        })

    feats[f"{prefix}_wide_std"] = float(
        band_stats["mid"]["db_std"] + band_stats["high"]["db_std"]
    )
    feats[f"{prefix}_hml_balance"] = float(
        band_stats["high"]["db_ratio"] - band_stats["low"]["db_ratio"]
    )
    return feats


_JND_DB = 6.0  # 感知门槛：人耳大约 6 dB 才能"明显感到更响"


def _calc_mel_row_peak_map(row: np.ndarray, distance: int = 5) -> dict:
    """在 mel dB row 上检测冲击峰。

    v4 改动：
      - local_mean 用因果 IIR（不偷看未来，边界更稳）
      - find_peaks 跑在 residual = row − local_mean 上（dB 单位）
      - 每峰存 response_db = residual[peak]（"高出 IIR 均值多少 dB"），
        语义直接对应感知响度，取代原先的 prominences
    """
    row = np.asarray(row, dtype=float, order="C")
    W = 50

    local_mean = _iir_local_mean(row, equivalent_window=W)
    _, local_std = _rolling_mean_std_numpy(row, W)
    residual = row - local_mean

    # height：至少超过局部 std 且 ≥ 6 dB 感知门槛
    thresholds = np.maximum(local_std, _JND_DB)
    peaks, _props = find_peaks(
        residual,
        height=thresholds,
        distance=distance,
        prominence=_JND_DB,   # 仍要求"邻域抬升 ≥ 6 dB"，过滤斜坡顶上的伪峰
        wlen=50,
        width=[1, 50],
    )

    peak_map: dict = {}
    half = 10
    if peaks.size:
        lefts = np.maximum(peaks - half, 0)
        rights = np.minimum(peaks + half + 1, row.size)
        ps1, ps2_m, ps3, ps4 = _prefix_sum_pow4(row)
        kf_arr = _segment_kurtosis_from_prefix(
            ps1, ps2_m, ps3, ps4, lefts, rights, fisher=False, bias=False,
        )
        for i, p in enumerate(peaks):
            p_int = int(p)
            peak_map[p_int] = {
                "kurtosis": float(kf_arr[i]),
                "response_db": float(residual[p_int]),   # ← E_dB − M_dB at peak
                "db": float(row[p_int]),
            }
    return peak_map


def _merge_mel_row_peak_maps(row_peak_maps: list[dict], merge_distance: int = 1) -> dict:
    """合并多个 Mel row 上的峰。

    先把所有 row 的候选峰摊平，再按时间帧聚类。不同频带上的同一次冲击
    往往会差 1 帧左右，默认把相邻帧也合成一个事件；事件强度取各频带最大值。
    """
    candidates: list[tuple[int, dict]] = []
    for row_peaks in row_peak_maps:
        for p, s in row_peaks.items():
            candidates.append((int(p), s))
    if not candidates:
        return {}

    merge_distance = max(0, int(merge_distance))
    candidates.sort(key=lambda item: item[0])
    all_peaks: dict = {}

    def _score(s: dict) -> tuple[float, float, float]:
        return (
            float(s.get("response_db", 0.0)),
            float(s.get("db", 0.0)),
            float(s.get("kurtosis", 0.0)),
        )

    def _merge_into(target: dict, s: dict) -> None:
        target["kurtosis"] = max(target["kurtosis"], float(s["kurtosis"]))
        target["response_db"] = max(target["response_db"], float(s["response_db"]))
        target["db"] = max(target["db"], float(s["db"]))

    cluster_key: int | None = None
    cluster_stats: dict | None = None
    cluster_score: tuple[float, float, float] | None = None
    cluster_last_p: int | None = None

    def _flush_cluster() -> None:
        if cluster_key is None or cluster_stats is None:
            return
        if cluster_key in all_peaks:
            _merge_into(all_peaks[cluster_key], cluster_stats)
        else:
            all_peaks[cluster_key] = dict(cluster_stats)

    for p, s in candidates:
        if cluster_last_p is not None and p - cluster_last_p > merge_distance:
            _flush_cluster()
            cluster_key = None
            cluster_stats = None
            cluster_score = None

        if cluster_stats is None:
            cluster_key = p
            cluster_stats = {
                "kurtosis": float(s["kurtosis"]),
                "response_db": float(s["response_db"]),
                "db": float(s["db"]),
            }
            cluster_score = _score(s)
        else:
            _merge_into(cluster_stats, s)
            s_score = _score(s)
            if cluster_score is None or s_score > cluster_score:
                cluster_key = p
                cluster_score = s_score

        cluster_last_p = p

    _flush_cluster()
    return all_peaks


def _summarize_mel_peak_features(
    *,
    all_peaks: dict,
    kf_thr: float,
    response_thr: float,
    prefix: str,
    mel_frame_sr: float,
    n_frames: int,
):
    """峰过滤 + 聚合。`response_thr` 单位是 dB（= E−M 的门槛，常用 6 或 9）。"""
    features: dict = {}
    filted_peaks = {
        x: s for x, s in all_peaks.items()
        if (s["kurtosis"] >= 5.0 or s["response_db"] >= 9.0)
        or (s["kurtosis"] >= kf_thr and s["response_db"] >= response_thr)
    }

    duration_sec = float(n_frames) / float(mel_frame_sr) if mel_frame_sr > 0 else 0.0
    count_per_sec = len(filted_peaks) / max(duration_sec, _EPS)

    if filted_peaks:
        sorted_positions = np.fromiter((int(x) for x in filted_peaks), dtype=float)
        sorted_positions.sort()
        k_list = np.fromiter((v["kurtosis"] for v in filted_peaks.values()), dtype=float)
        r_list = np.fromiter((v["response_db"] for v in filted_peaks.values()), dtype=float)
        if sorted_positions.size >= 2:
            diffs = np.diff(sorted_positions) / float(mel_frame_sr)
            dt_mean = float(np.mean(diffs))
            interval_cv = float(np.std(diffs) / (dt_mean + _EPS))
        else:
            interval_cv = 0.0
        features[prefix + "_count_per_sec"] = float(count_per_sec)
        features[prefix + "_kurtosis_mean"] = float(k_list.mean())
        features[prefix + "_kurtosis_q95"] = float(np.percentile(k_list, 95))
        features[prefix + "_kurtosis_q50"] = float(np.percentile(k_list, 50))
        features[prefix + "_response_db_mean"] = float(r_list.mean())
        features[prefix + "_response_db_std"] = float(np.std(r_list))
        features[prefix + "_response_db_q95"] = float(np.percentile(r_list, 95))
        features[prefix + "_response_db_q50"] = float(np.percentile(r_list, 50))
        features[prefix + "_interval_cv"] = interval_cv
    else:
        features[prefix + "_count_per_sec"] = 0.0
        features[prefix + "_kurtosis_mean"] = 0.0
        features[prefix + "_kurtosis_q95"] = 0.0
        features[prefix + "_kurtosis_q50"] = 0.0
        features[prefix + "_response_db_mean"] = 0.0
        features[prefix + "_response_db_std"] = 0.0
        features[prefix + "_response_db_q95"] = 0.0
        features[prefix + "_response_db_q50"] = 0.0
        features[prefix + "_interval_cv"] = 0.0

    return features, filted_peaks


_MEL_N_FFT = 256
_MEL_N_MELS = 26
_MEL_HOP_LENGTH = _MEL_N_FFT // 2
_MEL_PEAK_DISTANCE = 5
_MEL_SHOCK_SPECS = (
    ("low", 4, 6),    # 咬齿
    ("med", 3, 6),    # 震颤 / 中频冲击
    ("high", 3, 5),   # 秒表
    ("all", 3, 6),    # 全频率
)


def _mel_context(data, sr):
    """Build the shared mel spectrogram state used by v4-style and v5 features."""
    mel_spectrogram = librosa.feature.melspectrogram(
        y=np.asarray(data, dtype=float, order="C"),
        sr=sr,
        n_mels=_MEL_N_MELS,
        fmax=sr // 2,
        n_fft=_MEL_N_FFT,
        hop_length=_MEL_HOP_LENGTH,
    )
    freq_axis = librosa.mel_frequencies(n_mels=_MEL_N_MELS, fmax=sr // 2)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)[:, 1:-1]
    row_peak_maps = [
        _calc_mel_row_peak_map(mel_spectrogram_db[i, :], distance=_MEL_PEAK_DISTANCE)
        for i in range(mel_spectrogram_db.shape[0])
    ]
    return {
        "spec_db": mel_spectrogram_db,
        "freq_axis": freq_axis,
        "frame_sr": float(sr) / float(_MEL_HOP_LENGTH),
        "n_frames": int(mel_spectrogram_db.shape[1]),
        "row_peak_maps": row_peak_maps,
        "row_groups": _band_row_indices(freq_axis),
    }


def _row_peak_maps_for_group(row_peak_maps: list[dict], row_groups: dict[str, np.ndarray], group: str):
    if group == "all":
        return row_peak_maps
    group_key = "mid" if group == "med" else group
    return [row_peak_maps[int(i)] for i in row_groups[group_key]]


def _add_mel_shock_features(features: dict, context: dict, prefix: str) -> None:
    row_peak_maps = context["row_peak_maps"]
    row_groups = context["row_groups"]
    for group, kf_thr, response_thr in _MEL_SHOCK_SPECS:
        dd, _ = _summarize_mel_peak_features(
            all_peaks=_merge_mel_row_peak_maps(
                _row_peak_maps_for_group(row_peak_maps, row_groups, group)
            ),
            kf_thr=kf_thr,
            response_thr=response_thr,
            prefix=f"{prefix}_shock_{group}",
            mel_frame_sr=context["frame_sr"],
            n_frames=context["n_frames"],
        )
        features |= dd

    low_count = max(features.get(prefix + "_shock_low_count_per_sec", 0.0), 0.0)
    high_count = max(features.get(prefix + "_shock_high_count_per_sec", 0.0), 0.0)
    features[prefix + "_shock_high_ratio"] = high_count / max(low_count + high_count, _EPS)


def _base_mel_features(data, sr, prefix: str = "mel") -> tuple[dict, dict]:
    context = _mel_context(data, sr)
    features = _get_bands_features(
        context["spec_db"],
        context["freq_axis"],
        sr,
        prefix + "_band",
    )
    _add_mel_shock_features(features, context, prefix)
    return features, context


def mel_features(data, sr) -> dict:
    """Mel 频带统计 + low/med/high/all 冲击特征。"""
    features, _ = _base_mel_features(data, sr)
    return features


# ===================== 子带特征 =====================
_SUBBAND_KEYS = (
    "rms",
    "energy_ratio",
    "kurt",
    "crest",
    "q95",
    "impulse",
    "log_energy_entropy",
)


def _empty_subband(prefix: str) -> dict:
    return {f"{prefix}_{k}": 0.0 for k in _SUBBAND_KEYS}


def _subband_features(coeff: np.ndarray, total_energy: float, prefix: str) -> dict:
    """单层小波系数 -> 7 个基础统计特征。
    （mean / skew / zcr 已删除：mean 在子带≈0、skew 与全局重复、zcr 与该层中心频率几乎一一对应）
    """
    c = np.asarray(coeff, dtype=float, order="C").ravel()
    n = c.size
    if n == 0:
        return _empty_subband(prefix)

    abs_c = np.abs(c)
    mean_abs = float(np.mean(abs_c))
    energy = float(np.dot(c, c))
    rms = float(np.sqrt(energy / n))
    peak = float(abs_c.max())

    energy_ratio = energy / (total_energy + _EPS)
    crest = peak / (rms + _EPS)
    impulse = peak / (mean_abs + _EPS)
    q95 = float(np.percentile(abs_c, 95))
    kurt = _safe_kurt(c)

    # 子带内能量分布的 Shannon 熵（衡量能量集中 vs 弥散）
    if energy > 0:
        p = (c * c) / energy
        nz = p > 0
        log_energy_entropy = float(-np.sum(p[nz] * np.log(p[nz])))
    else:
        log_energy_entropy = 0.0

    return {
        f"{prefix}_rms": rms,
        f"{prefix}_energy_ratio": energy_ratio,
        f"{prefix}_kurt": kurt,
        f"{prefix}_crest": crest,
        f"{prefix}_q95": q95,
        f"{prefix}_impulse": impulse,
        f"{prefix}_log_energy_entropy": log_energy_entropy,
    }


# ===================== Spectral Kurtosis (滑窗 kurtosis) =====================
_SK_KEYS = ("max", "mean", "q95", "time_to_max")


def _sk_features(coeff: np.ndarray, prefix: str, win: int = 256, step: int = 64) -> dict:
    """每层小波系数上的 spectral kurtosis (滑窗 Pearson 峰度) 4 件套。

    - sk_max         : 全程 SK 最大值（最 impulsive 的瞬间）
    - sk_mean        : 全程 SK 均值（整体非高斯程度）
    - sk_q95         : 95% 分位 SK
    - sk_time_to_max : SK 最大值出现的相对位置 ∈ [0, 1]
                       0 = 最早期、1 = 最末端，告诉你冲击发生在 5s 内的哪个位置
    """
    c = np.asarray(coeff, dtype=float, order="C")
    n = c.size
    empty = {f"{prefix}_{k}": 0.0 for k in _SK_KEYS}

    if n < win + step:
        # 信号太短退化为整段 kurtosis
        kf = _safe_kurt(c)
        return {
            f"{prefix}_max": float(kf),
            f"{prefix}_mean": float(kf),
            f"{prefix}_q95": float(kf),
            f"{prefix}_time_to_max": 0.0,
        }

    # 用前缀和批量算每个滑窗的 Pearson kurtosis（O(n / step)）
    starts = np.arange(0, n - win + 1, step, dtype=int)
    if starts.size == 0:
        return empty
    ends = starts + win
    ps1, ps2, ps3, ps4 = _prefix_sum_pow4(c)
    sk = _segment_kurtosis_from_prefix(
        ps1, ps2, ps3, ps4, starts, ends, fisher=False, bias=True,
    )
    sk = sk[np.isfinite(sk)]
    if sk.size == 0:
        return empty

    sk_max = float(sk.max())
    sk_mean = float(sk.mean())
    sk_q95 = float(np.percentile(sk, 95))
    # time_to_max: argmax 在窗口序列上的位置，归一到 [0,1]
    time_to_max = float(np.argmax(sk) / max(sk.size - 1, 1))

    return {
        f"{prefix}_max": sk_max,
        f"{prefix}_mean": sk_mean,
        f"{prefix}_q95": sk_q95,
        f"{prefix}_time_to_max": time_to_max,
    }


# ===================== DWT =====================
def _layer_effective_sr(level: int, coeff_index: int, sr: float) -> float:
    """coeffs[0]=cA{level} 与 coeffs[1]=cD{level} 长度都是 N/2**level，
    所以等效采样率 = sr / 2**level；其余 cD{level-i+1} 的等效采样率
    = sr / 2**(level - i + 1)。
    """
    if coeff_index <= 1:
        return float(sr) / float(2 ** level)
    return float(sr) / float(2 ** (level - coeff_index + 1))


def dwt_features(data, sr: float, wavelet: str = "db22", level: int = 4) -> dict:
    """每层小波系数 (cA{level}, cD{level}..cD1) 提取
    7 base + 8 shock + 4 SK = 19 个特征。
    """
    x = np.asarray(data, dtype=float, order="C")
    coeffs = pywt.wavedec(x, wavelet, level=level)
    # Parseval：正交小波下 Σ E_subband ≈ E(x)
    total_energy = float(sum(np.dot(c, c) for c in coeffs))
    W_shock = 256       # shock 局部 kurtosis 窗长
    SK_WIN = 256        # SK 滑窗长度
    SK_STEP = 64        # SK 滑窗步长（约 4× 重叠）
    feats: dict = {}
    for i, coeff in enumerate(coeffs):
        ci = np.asarray(coeff, dtype=float, order="C")
        sr_eff = _layer_effective_sr(level, i, sr)
        feats.update(
            _subband_features(ci, total_energy=total_energy, prefix=f"dwt_{i}")
        )
        feats.update(
            _get_dwt_shock_features(
                ci, window=W_shock, prefix=f"dwt_{i}_shock", sr_eff=sr_eff,
            )
        )
        feats.update(
            _sk_features(ci, prefix=f"dwt_{i}_sk", win=SK_WIN, step=SK_STEP)
        )
    return feats


# ===================== Bark/mel 子带组定义 =====================
# 与 v4 mel_band 的 low/mid/high 同标准，但这里挂的是行索引，
# 因为新特征都建立在 mel_spectrogram_db 的每一行上。
_BAND_GROUP_HZ = (
    ("low", 100.0, 1500.0),
    ("mid", 1500.0, 4000.0),
    ("high", 4000.0, None),    # None = up to Nyquist
)

# v5 调制谱在 mel 通路里分的三段（Hz）。
# 上限受 mel_frame_sr ≈ 156 Hz 限制，约 78 Hz。
_MOD_BANDS_MEL = (
    ("shudder",  1.0,  20.0),  # 抖动
    ("rough_lo", 20.0, 50.0),  # 低频粗糙
    ("rough_hi", 50.0, 78.0),  # 中频粗糙
)

# Hilbert 包络通路（补 mel 通路抓不到的高调制频段）
_MOD_BANDS_ENV = (
    ("shudder",     1.0,   20.0),
    ("rough",       20.0,  80.0),
    ("chatter_lo",  80.0,  200.0),   # ★ 震颤主要落点
    ("chatter_hi",  200.0, 300.0),
)

_V5_REDUNDANT_MEL_FEATURES = {
    "mel_shock_med_response_db_mean",   # 保留 q50
    "mel_shock_med_kurtosis_mean",      # 保留 q50
    "mel_shock_high_response_db_q50",   # mean 在当前模型中贡献更高
    "mel_shock_low_kurtosis_mean",      # 保留 q50
}


def _empty_loud_features(prefix: str) -> dict:
    feats: dict = {}
    for g, _, _ in _BAND_GROUP_HZ:
        feats[f"{prefix}_loud_{g}_dynamic_range_db"] = 0.0
        feats[f"{prefix}_loud_{g}_temporal_kurt"] = 0.0
        feats[f"{prefix}_loud_{g}_temporal_acf1"] = 0.0
    return feats


def _empty_mod_features(prefix: str) -> dict:
    feats: dict = {}
    for g, _, _ in _BAND_GROUP_HZ:
        feats[f"{prefix}_mod_{g}_peak_f_hz"] = 0.0
        feats[f"{prefix}_mod_{g}_peak_prominence"] = 0.0
        for band, _, _ in _MOD_BANDS_MEL:
            feats[f"{prefix}_mod_{g}_{band}"] = 0.0
    return feats


def _empty_env_features() -> dict:
    return {f"env_mod_{band}": 0.0 for band, _, _ in _MOD_BANDS_ENV}


def _empty_sync_features(prefix: str) -> dict:
    return {
        f"{prefix}_mod_sync_global": 0.0,
        f"{prefix}_mod_band_coherence": 0.0,
    }


# ===================== Family A: 子带 loudness 时间序列 =====================
def _band_row_indices(freq_axis: np.ndarray) -> dict[str, np.ndarray]:
    """按 _BAND_GROUP_HZ 把 mel 行索引分到 low/mid/high 三组。"""
    fmax = float(freq_axis[-1])
    out: dict[str, np.ndarray] = {}
    for name, lo, hi in _BAND_GROUP_HZ:
        hi_ = fmax if hi is None else hi
        if hi is None:
            mask = (freq_axis >= lo) & (freq_axis <= hi_)
        else:
            mask = (freq_axis >= lo) & (freq_axis < hi_)
        idx = np.where(mask)[0]
        if idx.size == 0:
            # 边界情况：把最近的一行兜底放进来，避免空组
            idx = np.array([int(np.argmin(np.abs(freq_axis - lo)))], dtype=int)
        out[name] = idx
    return out


def _lag1_autocorr(x: np.ndarray) -> float:
    """归一化 lag-1 自相关，∈ [-1, 1]。连续周期振动 → 接近 1。"""
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 3:
        return 0.0
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= _EPS:
        return 0.0
    return float(np.dot(x[:-1], x[1:]) / denom)


def _loudness_time_series_features(
    spec_db: np.ndarray,
    row_groups: dict[str, np.ndarray],
    prefix: str,
) -> dict:
    """Family A：每组子带的"按帧平均 dB"时间序列摘要。"""
    feats: dict = {}
    n_mels, n_frames = spec_db.shape
    if n_frames < 8:
        return _empty_loud_features(prefix)

    for g, _, _ in _BAND_GROUP_HZ:
        rows = row_groups[g]
        # 子带内各帧的平均 dB → specific loudness 的代理
        series = np.mean(spec_db[rows, :], axis=0)
        p5, p95 = np.percentile(series, [5, 95])
        feats[f"{prefix}_loud_{g}_dynamic_range_db"] = float(p95 - p5)
        feats[f"{prefix}_loud_{g}_temporal_kurt"] = _safe_kurt(series)
        feats[f"{prefix}_loud_{g}_temporal_acf1"] = _lag1_autocorr(series)
    return feats


# ===================== Family B + C(mel 通路): 子带调制谱 =====================
def _modulation_spectrum_per_group(
    series: np.ndarray,
    mel_frame_sr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """对一个 1D 时间序列做调制谱（去 DC + Hann + rFFT 幅值）。"""
    series = np.asarray(series, dtype=float).ravel()
    n = series.size
    if n < 16:
        return np.zeros(1), np.zeros(1)
    s = series - series.mean()
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(s * win))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(mel_frame_sr))
    return freqs, spec


def _band_energy_ratios(
    freqs: np.ndarray,
    spec: np.ndarray,
    bands: Iterable[tuple[str, float, float]],
) -> dict[str, float]:
    """把调制谱按给定频段切，返回各段能量占比（占总能量）。
    1 Hz 以下（DC 残留）排除在分母外。"""
    out: dict[str, float] = {}
    if spec.size == 0:
        return {b[0]: 0.0 for b in bands}
    power = spec * spec
    mask_total = freqs >= 1.0
    total = float(power[mask_total].sum()) + _EPS
    for name, lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        out[name] = float(power[m].sum() / total)
    return out


def _peak_in_band(
    freqs: np.ndarray, spec: np.ndarray, f_lo: float, f_hi: float
) -> tuple[float, float]:
    """返回 [f_lo, f_hi) 内调制谱主峰的 (频率, prominence)。
    prominence ≈ 20·log10(peak / median(noise_floor))，单位 dB。"""
    m = (freqs >= f_lo) & (freqs < f_hi)
    if not np.any(m):
        return 0.0, 0.0
    sub_freqs = freqs[m]
    sub_spec = spec[m]
    idx = int(np.argmax(sub_spec))
    peak_f = float(sub_freqs[idx])
    peak_v = float(sub_spec[idx])
    # noise floor：用整段调制谱（>=1 Hz）的中位数估
    noise_mask = freqs >= 1.0
    noise = float(np.median(spec[noise_mask])) if np.any(noise_mask) else 0.0
    prominence_db = 20.0 * np.log10((peak_v + _EPS) / (noise + _EPS))
    return peak_f, float(prominence_db)


def _modulation_features_mel(
    spec_db: np.ndarray,
    row_groups: dict[str, np.ndarray],
    mel_frame_sr: float,
    prefix: str,
) -> tuple[dict, dict[str, tuple[float, float]]]:
    """Family B + C(mel 部分)：每组 mel 子带的调制谱特征。

    返回 (features, mod_peak_per_group)，后者给 Family D 同步分析用。
    """
    feats: dict = {}
    _, n_frames = spec_db.shape
    if n_frames < 16:
        return _empty_mod_features(prefix), {}

    f_lo_all = _MOD_BANDS_MEL[0][1]    # 1 Hz
    f_hi_all = _MOD_BANDS_MEL[-1][2]   # 78 Hz
    peak_per_group: dict[str, tuple[float, float]] = {}

    for g, _, _ in _BAND_GROUP_HZ:
        rows = row_groups[g]
        series = np.mean(spec_db[rows, :], axis=0)
        freqs, spec = _modulation_spectrum_per_group(series, mel_frame_sr)

        # Family B: 主峰
        peak_f, prom = _peak_in_band(freqs, spec, f_lo_all, f_hi_all)
        feats[f"{prefix}_mod_{g}_peak_f_hz"] = peak_f
        feats[f"{prefix}_mod_{g}_peak_prominence"] = prom
        peak_per_group[g] = (peak_f, prom)

        # Family C(mel 通路): 三段能量比
        ratios = _band_energy_ratios(freqs, spec, _MOD_BANDS_MEL)
        for band, val in ratios.items():
            feats[f"{prefix}_mod_{g}_{band}"] = val

    return feats, peak_per_group


# ===================== Family C(env 通路): Hilbert 包络谱 =====================
def _hilbert_envelope_modulation_features(
    x: np.ndarray, sr: float, env_target_sr: float = 1000.0, envelope: np.ndarray | None = None
) -> dict:
    """对原始信号求 Hilbert 包络再 FFT，覆盖 80–300 Hz 高调制频段。

    x         : 预处理后的 1D 信号
    sr        : 采样率
    env_target_sr: 包络重采样目标，默认 1000 Hz（足够覆盖 0–500 Hz 调制）
    envelope  : 可选，外部已算好的 |hilbert(x)| 包络（共享以避免重复 Hilbert）
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < int(sr):     # 至少 1 秒
        return _empty_env_features()

    # 1) 高通去 DC（信号已经去过，但包络再去一次保险）
    # 2) Hilbert 包络（复用共享包络）
    base = _analytic_envelope(x) if envelope is None else np.asarray(envelope, dtype=float)
    env = base.astype(np.float32)

    # 3) 包络重采样到 env_target_sr，降低 FFT 成本
    decim = max(1, int(round(float(sr) / float(env_target_sr))))
    if decim > 1:
        env = signal.decimate(env, decim, ftype="iir", zero_phase=True)
    eff_sr = float(sr) / float(decim)

    if env.size < 16:
        return _empty_env_features()

    # 4) 去 DC + 加窗 + rFFT
    e = env - env.mean()
    e = e * np.hanning(e.size)
    spec = np.abs(np.fft.rfft(e))
    freqs = np.fft.rfftfreq(e.size, d=1.0 / eff_sr)

    ratios = _band_energy_ratios(freqs, spec, _MOD_BANDS_ENV)
    return {f"env_mod_{name}": val for name, val in ratios.items()}


# ===================== Family D: 跨带同步 =====================
def _cross_band_synchrony_features(
    spec_db: np.ndarray,
    mel_frame_sr: float,
    peak_per_group: dict[str, tuple[float, float]],
    prefix: str,
) -> dict:
    """Family D：相邻 mel 行调制谱的同步性。

    1) sync_global       三组主峰频率的相对离散度 → 1-CV
    2) band_coherence    相邻 mel 行调制谱（dB 标度）平均皮尔森相关
    """
    if not peak_per_group:
        return _empty_sync_features(prefix)

    feats: dict = {}
    peaks = np.array(
        [v[0] for v in peak_per_group.values() if v[0] > 0.0], dtype=float
    )
    if peaks.size >= 2 and peaks.mean() > _EPS:
        cv = float(peaks.std() / (peaks.mean() + _EPS))
        feats[f"{prefix}_mod_sync_global"] = float(np.clip(1.0 - cv, 0.0, 1.0))
    else:
        feats[f"{prefix}_mod_sync_global"] = 0.0

    # 相邻 mel 行调制谱（取 1 Hz 以上）相关系数
    n_mels, n_frames = spec_db.shape
    if n_frames < 16 or n_mels < 2:
        feats[f"{prefix}_mod_band_coherence"] = 0.0
        return feats

    # 一次性 FFT：对每行去 DC + Hann
    win = np.hanning(n_frames)
    centered = spec_db - spec_db.mean(axis=1, keepdims=True)
    mod = np.abs(np.fft.rfft(centered * win, axis=1))
    freqs = np.fft.rfftfreq(n_frames, d=1.0 / float(mel_frame_sr))
    mask = freqs >= 1.0
    if not np.any(mask):
        feats[f"{prefix}_mod_band_coherence"] = 0.0
        return feats
    mod = mod[:, mask]

    # 行对行的皮尔森相关，取相邻 (i, i+1)，避免 O(N²)
    coeffs = []
    for i in range(n_mels - 1):
        a, b = mod[i], mod[i + 1]
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
        if denom <= _EPS:
            continue
        coeffs.append(float(np.dot(a, b) / denom))
    feats[f"{prefix}_mod_band_coherence"] = float(np.mean(coeffs)) if coeffs else 0.0
    return feats


# ===================== v5 单方向震颤冲击特征（12 维，挂在 DWT 上）=====================
# 设计动机见 漏检震颤深度分析：v4 单方向特征 (q95, kurtosis, sk_max 等)
# 把 5 秒压成一个数，抹掉了"在哪段、有多稳"的信息。下面 4 个 family 专门补：
#   Family I    时段非平稳 (4) — 抓"前段冲击多、后段干净"型
#   Family II   带限包络谱 (3) — 抓"高频共振被低频调制"型（NVH 经典做法）
#   Family III  SK burst (3)  — 抓"多段都超阈"持续型
#   Family IV   自相关周期 (2) — 抓周期重复型


def _max_run_length(mask: np.ndarray) -> int:
    """布尔数组中 True 最长连续段长度。"""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not m.any():
        return 0
    # 给两端补 0 后差分找上升/下降沿
    padded = np.concatenate(([0], m.astype(np.int8), [0]))
    d = np.diff(padded)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    runs = ends - starts
    return int(runs.max()) if runs.size > 0 else 0


# ----- Family I: 时段非平稳 (4 维) -----
def _quartile_nonstationarity_features(x: np.ndarray, prefix: str) -> dict:
    """把信号切 4 段统计，抓"能量在时间上不均匀"。
    震颤的前段冲击 / 间歇性强劲段在这里会冒出来。
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 16:
        return {
            f"{prefix}_quartile_rms_ratio_max_min": 1.0,
            f"{prefix}_quartile_rms_cv": 0.0,
            f"{prefix}_energy_center": 1.5,
            f"{prefix}_quartile_kurt_max": 0.0,
        }
    parts = np.array_split(x, 4)
    rms = np.array([float(np.sqrt(np.mean(p * p) + _EPS)) for p in parts])
    kurt = np.array([_safe_kurt(p) for p in parts])
    energy = rms * rms
    total = float(energy.sum()) + _EPS
    return {
        # 能量最强段 / 最弱段比；正常≈1，震颤集中段会 >>1
        f"{prefix}_quartile_rms_ratio_max_min": float(rms.max() / (rms.min() + _EPS)),
        # 4 段 RMS 的 CV；非平稳的指标
        f"{prefix}_quartile_rms_cv": float(rms.std() / (rms.mean() + _EPS)),
        # 能量重心 ∈ [0, 3]；0=能量全在第 1 段，3=全在最后段，正常≈1.5
        f"{prefix}_energy_center": float(np.sum(np.arange(4) * energy) / total),
        # 4 段里最剧烈那段的 kurtosis（被全局 kurtosis 平均掉的局部峰）
        f"{prefix}_quartile_kurt_max": float(kurt.max()),
    }


# ----- Family II: 带限包络谱 (3 维) -----
def _band_envelope_modulation_features(
    coef: np.ndarray, sr_eff: float, prefix: str
) -> dict:
    """对一个频带的小波系数做 Hilbert 包络 → FFT。
    震颤 = 高频共振被低频调制 → 包络谱在 80-200 Hz 段有峰。
    NVH 轴承诊断金标准（带通 + 包络谱）的工程实现。
    """
    coef = np.asarray(coef, dtype=float).ravel()
    if coef.size < 64:
        return {
            f"{prefix}_env_chatter_lo": 0.0,
            f"{prefix}_env_peak_f": 0.0,
            f"{prefix}_env_kurt": 0.0,
        }
    env = _analytic_envelope(coef)
    # 包络抽样到 ~1 kHz 控制 FFT 大小，覆盖 0–500 Hz 调制
    target_sr = 1000.0
    decim = max(1, int(round(sr_eff / target_sr)))
    if decim > 1 and env.size > decim * 16:
        env_d = signal.decimate(env, decim, ftype="iir", zero_phase=True)
        eff_sr = sr_eff / decim
    else:
        env_d = env
        eff_sr = sr_eff
    if env_d.size < 32:
        return {
            f"{prefix}_env_chatter_lo": 0.0,
            f"{prefix}_env_peak_f": 0.0,
            f"{prefix}_env_kurt": 0.0,
        }
    e = (env_d - env_d.mean()) * np.hanning(env_d.size)
    spec = np.abs(np.fft.rfft(e))
    freqs = np.fft.rfftfreq(e.size, d=1.0 / float(eff_sr))
    power = spec * spec
    mask_pos = freqs >= 1.0
    total = float(power[mask_pos].sum()) + _EPS
    # 80-200 Hz 段能量占比（震颤主要落点）
    m_ch = (freqs >= 80.0) & (freqs < 200.0)
    chatter_lo = float(power[m_ch].sum() / total)
    # 主峰频率（≥1 Hz 内）
    if mask_pos.any():
        sub_f = freqs[mask_pos]
        sub_s = spec[mask_pos]
        peak_f = float(sub_f[int(np.argmax(sub_s))])
    else:
        peak_f = 0.0
    return {
        f"{prefix}_env_chatter_lo": chatter_lo,
        f"{prefix}_env_peak_f": peak_f,
        # 包络 kurtosis：震颤件包络起伏剧烈 → 偏大
        f"{prefix}_env_kurt": _safe_kurt(env),
    }


# ----- Family III: SK burst 时序 -----
def _sk_burst_features(
    coef: np.ndarray, prefix: str, win: int = 256, step: int = 64, thr: float = 4.0
) -> dict:
    """SK 滑窗超阈值率 + 最长连续超阈段长度。
    抓"持续多段 SK > 4"型震颤，弥补 sk_max 只看单点峰值的不足。
    """
    coef = np.asarray(coef, dtype=float).ravel()
    n = coef.size
    if n < win + step:
        return {
            f"{prefix}_sk_burst_frac": 0.0,
            f"{prefix}_sk_max_run_len": 0.0,
        }
    starts = np.arange(0, n - win + 1, step, dtype=int)
    if starts.size == 0:
        return {
            f"{prefix}_sk_burst_frac": 0.0,
            f"{prefix}_sk_max_run_len": 0.0,
        }
    ends = starts + win
    ps1, ps2, ps3, ps4 = _prefix_sum_pow4(coef)
    sk = _segment_kurtosis_from_prefix(
        ps1, ps2, ps3, ps4, starts, ends, fisher=False, bias=True,
    )
    sk = sk[np.isfinite(sk)]
    if sk.size == 0:
        return {
            f"{prefix}_sk_burst_frac": 0.0,
            f"{prefix}_sk_max_run_len": 0.0,
        }
    above = sk > thr
    return {
        # 超阈窗占比：正常件 ≈ 0.05，震颤件可能 0.2–0.4
        f"{prefix}_sk_burst_frac": float(above.mean()),
        # 最长连续超阈段（震颤持续性）
        f"{prefix}_sk_max_run_len": float(_max_run_length(above)),
    }


# ----- Family IV: 自相关周期性 -----
def _autocorr_periodicity_features(
    coef: np.ndarray, sr_eff: float, prefix: str,
    max_lag_ms: float = 50.0, short_lag_n: int = 10,
) -> dict:
    """绝对值信号的自相关：找震颤重复周期。
    正常宽带噪声 R(τ) → 0 快；震颤在某周期 τ₀ 有显著峰。
    """
    coef = np.asarray(coef, dtype=float).ravel()
    if coef.size < 200:
        return {
            f"{prefix}_ac_peak_height": 0.0,
            f"{prefix}_ac_short_mean": 0.0,
        }
    env = np.abs(coef)
    env = env - env.mean()
    var = float(np.dot(env, env)) + _EPS
    max_lag = max(short_lag_n + 5, int(sr_eff * max_lag_ms / 1000.0))
    max_lag = min(max_lag, env.size // 4)
    if max_lag < short_lag_n + 5:
        return {
            f"{prefix}_ac_peak_height": 0.0,
            f"{prefix}_ac_short_mean": 0.0,
        }
    # 直接 O(max_lag·N) 计算（max_lag 通常 <500，可接受）
    # 用 FFT-based correlate 更快，但这里直接算更直观且省内存
    ac = np.empty(max_lag + 1, dtype=np.float64)
    ac[0] = 1.0
    for l in range(1, max_lag + 1):
        ac[l] = float(np.dot(env[:-l], env[l:]) / var)
    # 主峰：跳过 lag<5 (短 lag 受残余自相关影响)
    start = 5
    if start >= ac.size:
        peak_height = 0.0
    else:
        peak_height = float(np.max(ac[start:]))
    short_mean = float(ac[1:1 + short_lag_n].mean())
    return {
        f"{prefix}_ac_peak_height": peak_height,
        f"{prefix}_ac_short_mean": short_mean,
    }


# ===================== DWT v5: v4 全部 + 12 维新增 =====================
def dwt_features_v5(x, sr: float, wavelet: str = "db22", level: int = 4) -> dict:
    """v4 dwt_features 全部 + 在 dwt_3 / dwt_4 上新增 12 维 chatter 特征。

    布局（level=4, db22, sr=20000）：
      coeffs[0] = cA4 (1250 Hz eff)  → dwt_0  低频近似
      coeffs[1] = cD4 (1250 Hz)      → dwt_1
      coeffs[2] = cD3 (2500 Hz)      → dwt_2
      coeffs[3] = cD2 (5000 Hz)      → dwt_3  2.5–5 kHz
      coeffs[4] = cD1 (10000 Hz)     → dwt_4  5–10 kHz  ★ 新增 11 维全在这里
    """
    x = np.asarray(x, dtype=float, order="C")
    coeffs = pywt.wavedec(x, wavelet, level=level)
    total_energy = float(sum(np.dot(c, c) for c in coeffs))

    W_shock = 256
    SK_WIN = 256
    SK_STEP = 64

    feats: dict = {}

    # v4 同款：每层 7 base + 8 shock + 4 SK
    for i, coef in enumerate(coeffs):
        ci = np.asarray(coef, dtype=float, order="C")
        sr_eff = _layer_effective_sr(level, i, sr)
        feats.update(_subband_features(ci, total_energy=total_energy, prefix=f"dwt_{i}"))
        feats.update(_get_dwt_shock_features(
            ci, window=W_shock, prefix=f"dwt_{i}_shock", sr_eff=sr_eff,
        ))
        feats.update(_sk_features(ci, prefix=f"dwt_{i}_sk", win=SK_WIN, step=SK_STEP))

    feats.pop("dwt_4_sk_mean", None)  # 与 dwt_4_shock_kurt_mean 高相关，保留后者。

    # v5 新增 12 维（全部单方向、单件，仅用 dwt_3 / dwt_4 系数）
    c_dwt4 = np.asarray(coeffs[4], dtype=float, order="C")
    c_dwt3 = np.asarray(coeffs[3], dtype=float, order="C")
    sr_dwt4 = _layer_effective_sr(level, 4, sr)
    sr_dwt3 = _layer_effective_sr(level, 3, sr)

    # Family I: 时段非平稳 4 维 (dwt_4)
    feats.update(_quartile_nonstationarity_features(c_dwt4, prefix="dwt_4"))

    # Family II: 带限包络谱 3 维 (dwt_4 = 5–10 kHz 带 → 包络 → 80–200 Hz 调制能量)
    feats.update(_band_envelope_modulation_features(c_dwt4, sr_dwt4, prefix="dwt_4"))

    # Family III: SK burst (dwt_4 2 维 + dwt_3 1 维 = 3 维)
    burst4 = _sk_burst_features(c_dwt4, prefix="dwt_4", win=SK_WIN, step=SK_STEP)
    feats.update(burst4)
    burst3 = _sk_burst_features(c_dwt3, prefix="dwt_3", win=SK_WIN, step=SK_STEP)
    # 只保留 dwt_3 的 burst_frac（max_run_len 与 dwt_4 高度相关，跳过）
    feats["dwt_3_sk_burst_frac"] = burst3["dwt_3_sk_burst_frac"]

    # # Family IV: 自相关周期性 2 维 (dwt_4)
    # feats.update(_autocorr_periodicity_features(c_dwt4, sr_dwt4, prefix="dwt_4"))

    return feats


# ===================== mel_features v5 =====================
def mel_features_v5(data: np.ndarray, sr: float) -> dict:
    """v4 mel 特征 + v5 新增 4 类心理声学/调制谱特征。"""
    prefix = "mel"
    features, context = _base_mel_features(data, sr, prefix)
    for name in _V5_REDUNDANT_MEL_FEATURES:
        features.pop(name, None)

    # ---------- v5 新增 ----------
    mel_spectrogram_db = context["spec_db"]
    mel_frame_sr = context["frame_sr"]
    row_groups = context["row_groups"]

    # Family A: 子带 specific loudness 时间序列摘要
    features |= _loudness_time_series_features(mel_spectrogram_db, row_groups, prefix)

    # Family B + C(mel 通路): 子带调制谱主峰 + 三段能量
    mod_feats, peak_per_group = _modulation_features_mel(
        mel_spectrogram_db, row_groups, mel_frame_sr, prefix
    )
    features |= mod_feats

    # Family D: 跨带同步
    features |= _cross_band_synchrony_features(
        mel_spectrogram_db, mel_frame_sr, peak_per_group, prefix
    )

    return features


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den + _EPS)


def _finite(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _relative_ratio(num: float, den: float) -> float:
    den = float(den)
    if abs(den) <= _EPS:
        return 0.0
    return _finite(float(num) / den)


def _robust_scale(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    median = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - median)))


def _mel_context_for_n_mels(data: np.ndarray, sr: float, *, n_mels: int) -> dict:
    mel_spectrogram = librosa.feature.melspectrogram(
        y=np.asarray(data, dtype=float, order="C"),
        sr=sr,
        n_mels=int(n_mels),
        fmax=float(sr) / 2.0,
        n_fft=_MEL_N_FFT,
        hop_length=_MEL_HOP_LENGTH,
    )
    freq_axis = librosa.mel_frequencies(n_mels=int(n_mels), fmax=float(sr) / 2.0)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)[:, 1:-1]
    row_peak_maps = [
        _calc_mel_row_peak_map(mel_spectrogram_db[i, :], distance=_MEL_PEAK_DISTANCE)
        for i in range(mel_spectrogram_db.shape[0])
    ]
    return {
        "spec_db": mel_spectrogram_db,
        "freq_axis": freq_axis,
        "frame_sr": float(sr) / float(_MEL_HOP_LENGTH),
        "n_frames": int(mel_spectrogram_db.shape[1]),
        "row_peak_maps": row_peak_maps,
        "row_groups": _band_row_indices(freq_axis),
    }


def _mel_features_for_n_mels(
    data: np.ndarray,
    sr: float,
    *,
    n_mels: int,
    prefix: str,
) -> dict[str, float]:
    """Extract the v5 Mel families at another frequency resolution."""
    context = _mel_context_for_n_mels(data, sr, n_mels=n_mels)
    features = _get_bands_features(
        context["spec_db"],
        context["freq_axis"],
        sr,
        prefix + "_band",
    )
    _add_mel_shock_features(features, context, prefix)

    for old_name in _V5_REDUNDANT_MEL_FEATURES:
        features.pop(old_name.replace("mel_", f"{prefix}_", 1), None)

    spec_db = context["spec_db"]
    frame_sr = context["frame_sr"]
    row_groups = context["row_groups"]
    features.update(_loudness_time_series_features(spec_db, row_groups, prefix))
    mod_features, peak_per_group = _modulation_features_mel(
        spec_db,
        row_groups,
        frame_sr,
        prefix,
    )
    features.update(mod_features)
    features.update(
        _cross_band_synchrony_features(
            spec_db,
            frame_sr,
            peak_per_group,
            prefix,
        )
    )
    return {name: _finite(value) for name, value in features.items()}


def _welch_psd(x: np.ndarray, sr: float) -> tuple[np.ndarray, np.ndarray]:
    if x.size < 16:
        return np.array([], dtype=float), np.array([], dtype=float)
    nperseg = int(min(2048, max(256, 2 ** int(np.floor(np.log2(max(16, x.size // 4)))))))
    freqs, psd = signal.welch(
        np.asarray(x, dtype=float),
        fs=float(sr),
        window="hann",
        nperseg=min(nperseg, x.size),
        noverlap=min(nperseg // 2, max(0, x.size - 1)),
        detrend="constant",
        scaling="density",
    )
    return freqs.astype(float), np.maximum(psd.astype(float), 0.0)


def _band_energy(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= float(lo)) & (freqs < float(hi))
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def _spectral_shape_features(x: np.ndarray, sr: float) -> dict[str, float]:
    freqs, psd = _welch_psd(x, sr)
    prefix = "v6_spec"
    if freqs.size == 0:
        return {
            f"{prefix}_centroid_hz": 0.0,
            f"{prefix}_bandwidth_hz": 0.0,
            f"{prefix}_rolloff85_hz": 0.0,
            f"{prefix}_flatness": 0.0,
            f"{prefix}_low_ratio": 0.0,
            f"{prefix}_mid_ratio": 0.0,
            f"{prefix}_high_ratio": 0.0,
            f"{prefix}_ultra_ratio": 0.0,
            f"{prefix}_high_low_ratio": 0.0,
        }

    total = float(np.trapezoid(psd, freqs)) + _EPS
    weights = psd + _EPS
    centroid = float(np.sum(freqs * weights) / np.sum(weights))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * weights) / np.sum(weights)))
    cumsum = np.cumsum(psd)
    rolloff_idx = int(np.searchsorted(cumsum, 0.85 * float(cumsum[-1]), side="left"))
    rolloff_idx = int(np.clip(rolloff_idx, 0, freqs.size - 1))
    flatness = float(np.exp(np.mean(np.log(weights))) / (np.mean(weights) + _EPS))

    low = _band_energy(freqs, psd, 20.0, 200.0)
    mid = _band_energy(freqs, psd, 200.0, 1000.0)
    high = _band_energy(freqs, psd, 1000.0, 4000.0)
    ultra = _band_energy(freqs, psd, 4000.0, min(float(sr) / 2.0, 9000.0))
    return {
        f"{prefix}_centroid_hz": centroid,
        f"{prefix}_bandwidth_hz": bandwidth,
        f"{prefix}_rolloff85_hz": float(freqs[rolloff_idx]),
        f"{prefix}_flatness": flatness,
        f"{prefix}_low_ratio": _safe_ratio(low, total),
        f"{prefix}_mid_ratio": _safe_ratio(mid, total),
        f"{prefix}_high_ratio": _safe_ratio(high, total),
        f"{prefix}_ultra_ratio": _safe_ratio(ultra, total),
        f"{prefix}_high_low_ratio": _safe_ratio(high + ultra, low),
    }


def _spectral_entropy(power: np.ndarray) -> float:
    p = np.asarray(power, dtype=float)
    p = np.maximum(p, 0.0)
    total = float(np.sum(p))
    if total <= 0.0:
        return 0.0
    p = p / total
    entropy = -float(np.sum(p * np.log2(p + _EPS)))
    return entropy / float(np.log2(max(2, p.size)))


def _envelope_peak_features(
    x: np.ndarray, sr: float, envelope: np.ndarray | None = None
) -> dict[str, float]:
    prefix = "v6_env"
    if x.size < 16:
        return {
            f"{prefix}_peak_f_hz": 0.0,
            f"{prefix}_peak_prominence": 0.0,
            f"{prefix}_mod_entropy": 0.0,
            f"{prefix}_pulse_rate": 0.0,
            f"{prefix}_pulse_iat_cv": 0.0,
        }

    base = _analytic_envelope(x) if envelope is None else np.asarray(envelope, dtype=float)
    env = signal.detrend(base, type="constant")
    n = env.size
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    spec = np.abs(np.fft.rfft(env)) ** 2
    mask = (freqs >= 1.0) & (freqs <= min(300.0, float(sr) / 2.0))
    if not np.any(mask):
        peak_f = peak_prom = mod_entropy = 0.0
    else:
        f = freqs[mask]
        p = spec[mask]
        peak_idx = int(np.argmax(p))
        peak_f = float(f[peak_idx])
        peak_prom = float(p[peak_idx] / (np.median(p) + _EPS))
        mod_entropy = _spectral_entropy(p)

    smooth_win = max(3, int(round(float(sr) * 0.002)))
    if smooth_win % 2 == 0:
        smooth_win += 1
    if env.size >= smooth_win:
        env_s = signal.savgol_filter(env, smooth_win, polyorder=2, mode="interp")
    else:
        env_s = env
    threshold = float(np.mean(env_s) + 2.5 * np.std(env_s))
    distance = max(1, int(round(float(sr) * 0.002)))
    peaks, _ = signal.find_peaks(env_s, height=threshold, distance=distance)
    duration = float(n) / float(sr)
    pulse_rate = float(peaks.size) / max(duration, _EPS)
    if peaks.size >= 3:
        iat = np.diff(peaks).astype(float) / float(sr)
        pulse_iat_cv = float(np.std(iat) / (np.mean(iat) + _EPS))
    else:
        pulse_iat_cv = 0.0

    return {
        f"{prefix}_peak_f_hz": peak_f,
        f"{prefix}_peak_prominence": peak_prom,
        f"{prefix}_mod_entropy": mod_entropy,
        f"{prefix}_pulse_rate": pulse_rate,
        f"{prefix}_pulse_iat_cv": pulse_iat_cv,
    }


def _smooth_envelope(x: np.ndarray, sr: float, envelope: np.ndarray | None = None) -> np.ndarray:
    env = _analytic_envelope(x) if envelope is None else np.asarray(envelope, dtype=float).copy()
    win = max(3, int(round(float(sr) * 0.0015)))
    if win % 2 == 0:
        win += 1
    if env.size >= win and win >= 5:
        env = signal.savgol_filter(env, win, polyorder=2, mode="interp")
    return np.maximum(np.asarray(env, dtype=float), 0.0)


def _teager_envelope(x: np.ndarray, sr: float) -> np.ndarray:
    """Teager-Kaiser 能量算子的平滑包络，用于弱冲击检测。

    ψ[n] = x[n]² − x[n−1]·x[n+1]，对瞬态(高频冲击)放大、对缓慢背景抑制，
    比 Hilbert 幅度包络更易暴露贴着噪声底的弱秒表敲击；后续按稳健尺度设阈，
    因此对绝对幅值依赖低（归一化谱通量是同类替代）。
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 3:
        return np.zeros_like(x)
    psi = np.zeros_like(x)
    psi[1:-1] = x[1:-1] * x[1:-1] - x[:-2] * x[2:]
    np.maximum(psi, 0.0, out=psi)
    win = max(3, int(round(float(sr) * 0.0015)))
    if win % 2 == 0:
        win += 1
    if psi.size >= win and win >= 5:
        psi = signal.savgol_filter(psi, win, polyorder=2, mode="interp")
    return np.maximum(np.asarray(psi, dtype=float), 0.0)


def _detect_envelope_events(
    envelope: np.ndarray,
    sr: float,
    *,
    threshold_scale: float,
    min_separation_s: float = 0.15,
) -> np.ndarray:
    median = float(np.median(envelope))
    robust_scale = _robust_scale(envelope)
    if robust_scale <= _EPS:
        robust_scale = float(np.std(envelope))
    threshold = median + float(threshold_scale) * robust_scale
    prominence = max(0.75 * robust_scale, _EPS)
    # 合并同一次冲击产生的多次振铃峰：最小间隔提高到 ~150 ms，远小于秒表
    # 周期(0.2–2 s)但足以把单次撞击的振铃并成一个事件，避免事件数被振铃灌水。
    distance = max(1, int(round(float(sr) * float(min_separation_s))))
    peaks, _ = signal.find_peaks(
        envelope,
        height=threshold,
        prominence=prominence,
        distance=distance,
    )
    return peaks.astype(int, copy=False)


def _interval_periodicity(peaks: np.ndarray, sr: float) -> dict[str, float]:
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size < 2:
        return {
            "count_per_sec": 0.0,
            "interval_median_ms": 0.0,
            "interval_cv": 0.0,
            "interval_mad_ratio": 0.0,
            "interval_inlier_ratio": 0.0,
        }

    intervals = np.diff(peaks).astype(float) / float(sr)
    median = float(np.median(intervals))
    mean = float(np.mean(intervals))
    mad = float(np.median(np.abs(intervals - median)))
    tolerance = max(0.15 * median, 0.004)
    return {
        "count_per_sec": 0.0,
        "interval_median_ms": 1000.0 * median,
        "interval_cv": float(np.std(intervals) / (mean + _EPS)),
        "interval_mad_ratio": float(mad / (median + _EPS)),
        "interval_inlier_ratio": float(np.mean(np.abs(intervals - median) <= tolerance)),
    }


def _event_periodicity_features(
    peaks: np.ndarray,
    *,
    sr: float,
    duration: float,
    prefix: str,
) -> dict[str, float]:
    summary = _interval_periodicity(peaks, sr)
    summary["count_per_sec"] = float(len(peaks)) / max(float(duration), _EPS)
    return {f"{prefix}_{name}": _finite(value) for name, value in summary.items()}


def _autocorr_in_band(
    sig: np.ndarray,
    sr: float,
    *,
    lag_lo_s: float = 0.2,
    lag_hi_s: float = 2.0,
) -> tuple[float, float, float]:
    """在 [lag_lo_s, lag_hi_s] 秒的滞后范围内做归一化自相关。

    只在长周期带(默认 0.2–2 s)取峰，避开 ~30 ms 的短周期峰(泵自身转频/高频
    振动)误导周期估计。返回 (peak, lag_ms, snr)，其中 snr = 峰值 / 该带内
    |自相关| 的中位数 = 周期信噪比（越大说明周期越干净、越突出于背景）。
    """
    sig = np.asarray(sig, dtype=float).ravel()
    if sig.size < 32:
        return 0.0, 0.0, 0.0

    stride = max(1, int(round(float(sr) / 1000.0)))
    env = np.asarray(sig[::stride], dtype=float)
    env = env - float(np.mean(env))
    if float(np.dot(env, env)) <= _EPS:
        return 0.0, 0.0, 0.0

    sample_rate = float(sr) / float(stride)
    nfft = 1 << int(np.ceil(np.log2(max(2, 2 * env.size - 1))))
    spectrum = np.fft.rfft(env, n=nfft)
    autocorr = np.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft)[:env.size]
    autocorr = autocorr / (autocorr[0] + _EPS)

    lag_min = max(1, int(round(float(lag_lo_s) * sample_rate)))
    lag_max = min(env.size - 1, int(round(float(lag_hi_s) * sample_rate)))
    if lag_max <= lag_min:
        return 0.0, 0.0, 0.0

    region = autocorr[lag_min:lag_max + 1]
    idx = int(np.argmax(region))
    peak = float(region[idx])
    lag = lag_min + idx
    baseline = float(np.median(np.abs(region))) + _EPS
    snr = peak / baseline
    return _finite(peak), _finite(1000.0 * lag / sample_rate), _finite(snr)


def _xwin_period_consistency(
    events: np.ndarray,
    *,
    n_samples: int,
    sr: float,
    n_windows: int = 4,
) -> float:
    """跨窗口周期一致性：把信号切 n_windows 段，比较每段合并事件的中位间隔
    与全局中位间隔，返回落在 ±20% 内的窗口占比（仅统计事件≥3 的窗口）。
    真秒表全程周期稳定→接近 1；偶发/无规律敲击→低。与绝对幅值无关。
    """
    events = np.asarray(events, dtype=int)
    if events.size < 3:
        return 0.0
    global_intervals = np.diff(events).astype(float) / float(sr)
    if global_intervals.size == 0:
        return 0.0
    global_med = float(np.median(global_intervals))
    if global_med <= _EPS:
        return 0.0

    edges = np.linspace(0, n_samples, n_windows + 1, dtype=int)
    agree = 0
    valid = 0
    for s, e in zip(edges[:-1], edges[1:]):
        local = events[(events >= s) & (events < e)]
        if local.size < 3:
            continue
        valid += 1
        local_med = float(np.median(np.diff(local).astype(float) / float(sr)))
        if abs(local_med - global_med) <= 0.2 * global_med:
            agree += 1
    if valid == 0:
        return 0.0
    return float(agree) / float(valid)


def _window_tick_periodicity(
    peaks: np.ndarray,
    *,
    n_samples: int,
    sr: float,
    n_windows: int = 8,
) -> dict[str, float]:
    scores: list[float] = []
    for start, end in zip(
        np.linspace(0, n_samples, n_windows + 1, dtype=int)[:-1],
        np.linspace(0, n_samples, n_windows + 1, dtype=int)[1:],
    ):
        local = peaks[(peaks >= start) & (peaks < end)] - start
        if local.size < 3:
            scores.append(0.0)
        else:
            scores.append(float(_interval_periodicity(local, sr)["interval_inlier_ratio"]))
    arr = np.asarray(scores, dtype=float)
    return {
        "v6_tick_window_periodicity_mean": _finite(np.mean(arr)),
        "v6_tick_window_periodicity_max": _finite(np.max(arr)),
        "v6_tick_periodic_window_ratio": _finite(np.mean(arr >= 0.6)),
    }


def _tick_periodicity_features(
    x: np.ndarray, sr: float, envelope: np.ndarray | None = None
) -> dict[str, float]:
    """Describe repeated impacts without requiring them to be high-amplitude.

    重设计要点：
      1. 事件检测最小间隔 150 ms，合并同一次冲击的振铃峰；
      2. 自相关只在 0.2–2 s 长周期带取峰，避开 ~30 ms 短周期；
      3. 新增周期信噪比、跨窗口周期一致性；
      4. 增加 Teager 能量通路检测弱冲击，降低对绝对幅值的依赖。

    envelope：可选，外部共享的 |hilbert(x)| 包络（避免重复 Hilbert）。
    """
    envelope_s = _smooth_envelope(x, sr, envelope=envelope)
    teager = _teager_envelope(x, sr)
    duration = float(x.size) / float(sr)

    weak_peaks = _detect_envelope_events(envelope_s, sr, threshold_scale=1.5, min_separation_s=0.15)
    strong_peaks = _detect_envelope_events(envelope_s, sr, threshold_scale=3.0, min_separation_s=0.15)
    teager_peaks = _detect_envelope_events(teager, sr, threshold_scale=1.0, min_separation_s=0.15)

    features: dict[str, float] = {}
    features.update(
        _event_periodicity_features(weak_peaks, sr=sr, duration=duration, prefix="v6_tick_weak")
    )
    features.update(
        _event_periodicity_features(strong_peaks, sr=sr, duration=duration, prefix="v6_tick_strong")
    )

    # 长周期自相关(0.2–2 s) + 周期信噪比
    ac_peak, ac_lag_ms, ac_snr = _autocorr_in_band(envelope_s, sr, lag_lo_s=0.2, lag_hi_s=2.0)
    features["v6_tick_autocorr_peak"] = ac_peak
    features["v6_tick_autocorr_lag_ms"] = ac_lag_ms
    features["v6_tick_period_snr"] = ac_snr

    features.update(_window_tick_periodicity(weak_peaks, n_samples=x.size, sr=sr))
    features["v6_tick_xwin_period_consistency"] = _finite(
        _xwin_period_consistency(weak_peaks, n_samples=x.size, sr=sr)
    )
    features["v6_tick_strong_to_weak_count_ratio"] = _safe_ratio(
        float(strong_peaks.size), float(weak_peaks.size)
    )

    # Teager 能量通路：对弱冲击更敏感、低幅值依赖
    t_peak, _t_lag, _t_snr = _autocorr_in_band(teager, sr, lag_lo_s=0.2, lag_hi_s=2.0)
    features["v6_tick_teager_count_per_sec"] = _finite(float(teager_peaks.size) / max(duration, _EPS))
    features["v6_tick_teager_inlier_ratio"] = _finite(
        _interval_periodicity(teager_peaks, sr)["interval_inlier_ratio"]
    )
    features["v6_tick_teager_autocorr_peak"] = t_peak

    return {name: _finite(value) for name, value in features.items()}


def _relative_shock_features(features: dict[str, float]) -> dict[str, float]:
    """Relative descriptors that are less sensitive to absolute impact strength."""
    out: dict[str, float] = {}
    for prefix in ("mel", "mel13"):
        all_count = float(features.get(f"{prefix}_shock_all_count_per_sec", 0.0))
        all_kurt = float(features.get(f"{prefix}_shock_all_kurtosis_q95", 0.0))
        all_response = float(features.get(f"{prefix}_shock_all_response_db_q95", 0.0))
        for band in ("low", "med", "high"):
            key = f"v6_rel_{prefix}_{band}"
            count = float(features.get(f"{prefix}_shock_{band}_count_per_sec", 0.0))
            kurt = float(features.get(f"{prefix}_shock_{band}_kurtosis_q95", 0.0))
            response = float(features.get(f"{prefix}_shock_{band}_response_db_q95", 0.0))
            dynamic_range = float(features.get(f"{prefix}_loud_{band}_dynamic_range_db", 0.0))
            out[f"{key}_count_to_all"] = _relative_ratio(count, all_count)
            out[f"{key}_kurt_to_all"] = _relative_ratio(kurt, all_kurt)
            out[f"{key}_response_to_all"] = _relative_ratio(response, all_response)
            out[f"{key}_response_to_dynamic_range"] = _relative_ratio(response, dynamic_range)

        out[f"v6_rel_{prefix}_high_to_low_count"] = _relative_ratio(
            float(features.get(f"{prefix}_shock_high_count_per_sec", 0.0)),
            float(features.get(f"{prefix}_shock_low_count_per_sec", 0.0)),
        )

    for band in ("low", "med", "high", "all"):
        for metric in ("count_per_sec", "kurtosis_q95", "response_db_q95", "interval_cv"):
            out[f"v6_rel_{band}_{metric}_mel13_to_mel26"] = _relative_ratio(
                float(features.get(f"mel13_shock_{band}_{metric}", 0.0)),
                float(features.get(f"mel_shock_{band}_{metric}", 0.0)),
            )
    return {name: _finite(value) for name, value in out.items()}


def _segment_stability_features(x: np.ndarray, sr: float, n_segments: int = 8) -> dict[str, float]:
    prefix = "v6_seg"
    if x.size < n_segments * 8:
        return {
            f"{prefix}_rms_cv": 0.0,
            f"{prefix}_rms_max_min_ratio": 0.0,
            f"{prefix}_kurt_cv": 0.0,
            f"{prefix}_high_ratio_cv": 0.0,
        }

    segments = [seg for seg in np.array_split(np.asarray(x, dtype=float), n_segments) if seg.size >= 8]
    rms = np.asarray([float(np.sqrt(np.mean(seg * seg))) for seg in segments], dtype=float)
    kurt = np.asarray([_safe_kurt(seg) for seg in segments], dtype=float)
    high_ratios = []
    for seg in segments:
        freqs, psd = _welch_psd(seg, sr)
        total = float(np.trapezoid(psd, freqs)) + _EPS if freqs.size else _EPS
        high = _band_energy(freqs, psd, 1000.0, min(float(sr) / 2.0, 9000.0)) if freqs.size else 0.0
        high_ratios.append(_safe_ratio(high, total))
    high_ratios_arr = np.asarray(high_ratios, dtype=float)

    return {
        f"{prefix}_rms_cv": float(np.std(rms) / (np.mean(rms) + _EPS)),
        f"{prefix}_rms_max_min_ratio": _safe_ratio(float(np.max(rms)), float(np.min(rms))),
        f"{prefix}_kurt_cv": float(np.std(kurt) / (np.mean(np.abs(kurt)) + _EPS)),
        f"{prefix}_high_ratio_cv": float(np.std(high_ratios_arr) / (np.mean(high_ratios_arr) + _EPS)),
    }


def extract_features_v6(data, sr, return_timing: bool = False):
    """Return the frozen v6 Top-220 feature schema."""
    timing: dict[str, float] = {}
    t0_total = time.perf_counter()

    x_raw = np.asarray(data, dtype=float, order="C")

    t0 = time.perf_counter()
    x = process_data(x_raw, sr=sr, cutoff_low=20, cutoff_high=None)
    x = np.asarray(x, dtype=float, order="C")
    # 全信号 Hilbert 包络只算一次，供包络谱/包络主峰/秒表周期等多处共享
    base_env = _analytic_envelope(x)
    timing["process_data_sec"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    features: dict = {
        "mean": float(np.mean(x_raw)),
        "std": float(np.std(x)),
        "duration": float(len(x_raw) / sr),
        "skewness": _safe_skew(x),
        "kurtosis": _safe_kurt(x),
        "zero_crossing_rate": _zero_crossing_rate(x),
    }
    timing["global_sec"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    features.update(dwt_features_v5(x, sr=sr, wavelet="db22", level=4))
    timing["dwt_sec"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    features.update(mel_features_v5(x, sr))
    timing["mel_sec"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    features.update(_hilbert_envelope_modulation_features(x, sr, envelope=base_env))
    timing["env_sec"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    features.update(_mel_features_for_n_mels(x, sr, n_mels=13, prefix="mel13"))
    timing["mel13_sec"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    features.update(_spectral_shape_features(x, sr))
    features.update(_envelope_peak_features(x, sr, envelope=base_env))
    features.update(_segment_stability_features(x, sr))
    features.update(_tick_periodicity_features(x, sr, envelope=base_env))
    features.update(_relative_shock_features(features))
    timing["v6_extra_sec"] = float(time.perf_counter() - t0)

    missing = [name for name in V6_SELECTED_FEATURES if name not in features]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise RuntimeError(f"v6 Top-220 特征缺失 {len(missing)} 个: {preview}{suffix}")
    features = {name: features[name] for name in V6_SELECTED_FEATURES}

    timing["total_sec"] = float(time.perf_counter() - t0_total)
    if return_timing:
        return features, timing
    return features


# ===== v7 追加特征(震颤 + 弱秒表) =====
import time
import numpy as np
from scipy import signal as _sps
from scipy.fft import rfft as _rfft, rfftfreq as _rfftfreq


# 追加特征名(顺序固定, 必须与 v7_selected_features 尾部一致)
V7_EXTRA_FEATURES: tuple[str, ...] = (
    "x_lf_ratio_125_250",
    "x_lf_ratio_250_500",
    "x_env_lowband_ratio",
    "x_spec_centroid_hz",
    "x_env_spec_entropy",
    "x_env_mod_freq_hz",
)

# 弱秒表(weak_tick)特征名(顺序固定, 必须与 v7_selected_features._V7_TICK 一致)
V7_TICK_FEATURES: tuple[str, ...] = (
    "x_tick_hi_envkurt_max",
    "x_tick_hi_envkurt_p90",
    "x_tick_hi_crest_max",
    "x_tick_lomid_envkurt_max",
    "x_tick_lomid_crest_max",
    "x_tick_hi_to_lomid_envkurt",
)

# 弱秒表预处理/分窗参数(实测最优, 见解决方案文档第五节)
_TICK_HI_BAND = (7000.0, 9500.0)     # 弱秒表所在高频带
_TICK_LOMID_BAND = (500.0, 5000.0)   # 强秒表所在低/中带
_TICK_TRIM_S = 0.5                   # 裁掉首尾启停瞬态
_TICK_WIN_S = 1.0                    # 分窗长度
_TICK_HOP_S = 0.5                    # 分窗步长


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def _bandpass(x: np.ndarray, sr: float, lo: float, hi: float) -> np.ndarray:
    ny = 0.5 * sr
    hi = min(hi, ny * 0.98)
    if hi <= lo:
        return x
    b, a = _sps.butter(4, [lo / ny, hi / ny], btype="band")
    return _sps.filtfilt(b, a, x)


def _envkurt(e: np.ndarray) -> float:
    s = e.std()
    return 0.0 if s < 1e-12 else float(np.mean(((e - e.mean()) / s) ** 4) - 3.0)


def _crest(seg: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(seg * seg)))
    return 0.0 if r < 1e-12 else float(np.max(np.abs(seg)) / r)


def _windowed(sig: np.ndarray, sr: float, fn) -> list[float]:
    w = int(_TICK_WIN_S * sr)
    h = int(_TICK_HOP_S * sr)
    if w <= 0 or sig.size < w:
        return [fn(sig)] if sig.size else []
    return [fn(sig[s:s + w]) for s in range(0, sig.size - w + 1, max(1, h))]


def _band_tick_stats(x: np.ndarray, sr: float, band: tuple[float, float]) -> dict[str, float]:
    """带通 → |hilbert|包络的分窗峭度(max/p90) + 分窗 crest(max)。"""
    xb = _bandpass(x, sr, band[0], band[1])
    env = np.abs(_sps.hilbert(xb))
    kurts = _windowed(env, sr, _envkurt)
    crests = _windowed(xb, sr, _crest)
    return {
        "envkurt_max": max(kurts) if kurts else 0.0,
        "envkurt_p90": float(np.percentile(kurts, 90)) if kurts else 0.0,
        "crest_max": max(crests) if crests else 0.0,
    }


def extract_v7_tick(data, sr: int) -> dict[str, float]:
    """弱秒表(weak_tick)双频带分窗冲击特征。裁瞬态 → 双带分窗包络峭度/crest。
    训练增广与推理共用此函数, 保证一致。对单方向信号计算。"""
    out = {k: 0.0 for k in V7_TICK_FEATURES}
    x = np.asarray(data, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = int(sr * _TICK_TRIM_S)
    if x.size > 2 * n:                # 裁掉首尾启停瞬态
        x = x[n:-n]
    if x.size < sr:                   # 不足 1s 无法分窗, 返回 0
        return out
    x = x - x.mean()

    hi = _band_tick_stats(x, sr, _TICK_HI_BAND)
    lo = _band_tick_stats(x, sr, _TICK_LOMID_BAND)
    out["x_tick_hi_envkurt_max"] = _safe(hi["envkurt_max"])
    out["x_tick_hi_envkurt_p90"] = _safe(hi["envkurt_p90"])
    out["x_tick_hi_crest_max"] = _safe(hi["crest_max"])
    out["x_tick_lomid_envkurt_max"] = _safe(lo["envkurt_max"])
    out["x_tick_lomid_crest_max"] = _safe(lo["crest_max"])
    denom = abs(hi["envkurt_max"]) + abs(lo["envkurt_max"]) + 1e-9
    out["x_tick_hi_to_lomid_envkurt"] = _safe(hi["envkurt_max"] / denom)
    return out


def extract_v7_extra(data, sr: int) -> dict[str, float]:
    """在原始信号上计算 6 个震颤主力特征。训练集增广与推理共用此函数, 保证一致。"""
    out = {k: 0.0 for k in V7_EXTRA_FEATURES}
    x = np.asarray(data, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < sr // 5:
        return out
    x = x - x.mean()

    # --- 低频频带占比 + 谱重心 (Welch PSD) ---
    nper = min(8192, x.size)
    f, pxx = _sps.welch(x, fs=sr, nperseg=nper)
    tot = pxx.sum() + 1e-20
    m1 = (f >= 125) & (f < 250)
    m2 = (f >= 250) & (f < 500)
    out["x_lf_ratio_125_250"] = _safe(pxx[m1].sum() / tot)
    out["x_lf_ratio_250_500"] = _safe(pxx[m2].sum() / tot)
    out["x_spec_centroid_hz"] = _safe((f * pxx).sum() / tot)

    # --- 包络解调谱 (Hilbert 包络 -> FFT) ---
    env = np.abs(_sps.hilbert(x))
    env = env - env.mean()
    if env.std() < 1e-12:
        return out
    win = np.hanning(env.size)
    spec = np.abs(_rfft(env * win))
    freqs = _rfftfreq(env.size, d=1.0 / sr)
    band = (freqs >= 1.0) & (freqs <= 200.0)
    if not band.any():
        return out
    fb, sb = freqs[band], spec[band]
    if sb.sum() <= 0:
        return out
    k = int(np.argmax(sb))
    out["x_env_mod_freq_hz"] = _safe(fb[k])
    low = (fb >= 1.0) & (fb <= 30.0)
    out["x_env_lowband_ratio"] = _safe(sb[low].sum() / sb.sum())
    p = sb / sb.sum()
    p = p[p > 0]
    out["x_env_spec_entropy"] = _safe(-(p * np.log(p)).sum() / np.log(len(p) + 1e-12))
    return out


def extract_features_v7(data, sr, return_timing: bool = False):
    """Return v6 (225) + 6 震颤 + 6 弱秒表 = 237 维, 顺序按 V7_SELECTED_FEATURES。"""
    t0 = time.perf_counter()
    if return_timing:
        feats6, timing = extract_features_v6(data, sr, return_timing=True)
    else:
        feats6 = extract_features_v6(data, sr)
        timing = {}
    extra = extract_v7_extra(data, sr)
    tick = extract_v7_tick(data, sr)
    merged = {**feats6, **extra, **tick}

    missing = [n for n in V7_SELECTED_FEATURES if n not in merged]
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"v7 特征缺失 {len(missing)} 个: {preview}")
    features = {name: merged[name] for name in V7_SELECTED_FEATURES}

    if return_timing:
        timing["v7_extra_sec"] = float(time.perf_counter() - t0) - timing.get("total_sec", 0.0)
        timing["total_sec"] = float(time.perf_counter() - t0)
        return features, timing
    return features


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sig = rng.normal(0, 1, 25000).astype(np.float32)
    f = extract_features_v7(sig, 25000)
    print(f"features={len(f)} (期望237)")
    print("震颤6:", {k: round(f[k], 4) for k in V7_EXTRA_FEATURES})
    print("弱秒表6:", {k: round(f[k], 4) for k in V7_TICK_FEATURES})

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sig = rng.normal(0, 1, 25000).astype(np.float32)
    f = extract_features_v7(sig, 25000)
    print(f"features={len(f)} (期望237, 自包含)")
