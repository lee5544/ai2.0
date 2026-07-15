"""特征提取 v4 — 在 v3 基础上把 Mel 峰检测改为 IIR 局部均值 + residual (E−M) 评分。

相对 v3 的关键改动（仅 Mel 峰检测部分）：
  1) `local_mean` 改成因果一阶 IIR 低通（与 forvia_label/utils/plot_utils.py
     里 _iir_local_mean 一致），不再"看到未来"，边界更稳。
  2) find_peaks 在 `residual = row − local_mean` 上做，
     `height` 与 `prominence` 都按 dB 单位（≥ 6 dB 感知门槛）。
  3) 每个峰存储的不再是 find_peaks 返回的 `prominences`（=邻域谷高度差），
     而是 `response_db = residual[peak]`（=该帧高出 IIR 均值多少 dB），
     更直接对应人耳感知（6 dB ≈ "明显更响"）。
  4) 下游 `_summarize_mel_peak_features` 输出键从 `*_prominences_*`
     重命名为 `*_response_db_*`，语义对齐。

其余特征（global / DWT / SK / mel_band）与 v3 完全相同。


包含三类特征：

1. 全局时域 (6)
   mean / std / duration / skewness / kurtosis / zero_crossing_rate

2. 小波每层系数统计 + 冲击 + SK (19 × 5 = 95)
   db22, level=4 → coeffs[0]=cA4, coeffs[1..4]=cD4..cD1
   每层 base (7):   rms / energy_ratio / kurt / crest / q95 / impulse /
                   log_energy_entropy
   每层 shock (8):  find_peaks(|x|, prominence=2σ) + 每峰 kurtosis + STA/LTA
                   → count_rate / kurt_(mean,q95) / stalta_(mean,q95)
                     / iat_mean_ms / iat_cv / iat_periodicity
   每层 SK (4):     滑窗 (256/64) Pearson 峰度
                   → max / mean / q95 / time_to_max ∈ [0,1]
                     （time_to_max 给出最 impulsive 时刻在 5s 内的位置）

3. Mel 频带 + 冲击特征 (80)
   - mel_band (10):  低/中/高频带 dB 均值/std/能量占比 + 整段帧均值 std
   - mel_shock × 5 组 (低/中/高/马达/马达2)，每组 14 个 (70)
     峰检测改为 IIR 局部均值 + residual，每峰存 response_db (=E−M)
     输出键：count / kurt_gt4 / kurtosis_(sum,std,mean,q50,q75,q95)
            / response_db_(sum,std,mean,q50,q75,q95)

合计 6 + 95 + 80 = 181 维。
"""

from __future__ import annotations

import os
import sys
import time
from functools import lru_cache, wraps

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


def mel_features(data, sr) -> dict:
    """Mel 频带 12 维 + 5 组冲击特征 45 维 + 高频冲击占比 1 维。"""
    features: dict = {}
    prefix = "mel"

    n_fft = 256
    n_mels = 13
    hop_length = n_fft // 2
    mel_spectrogram = librosa.feature.melspectrogram(
        y=np.asarray(data, dtype=float, order="C"),
        sr=sr, n_mels=n_mels, fmax=sr // 2, n_fft=n_fft, hop_length=hop_length,
    )
    freq_axis = librosa.mel_frequencies(n_mels=n_mels, fmax=sr // 2)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)[:, 1:-1]
    n_frames = int(mel_spectrogram_db.shape[1])
    mel_frame_sr = float(sr) / float(hop_length)

    row_peak_maps = [
        _calc_mel_row_peak_map(mel_spectrogram_db[i, :], distance=5)
        for i in range(mel_spectrogram_db.shape[0])
    ]

    features |= _get_bands_features(mel_spectrogram_db, freq_axis, sr, prefix + "_band")

    # 冲击检测，含义见注释
    dd, _ = _summarize_mel_peak_features(
        all_peaks=_merge_mel_row_peak_maps(row_peak_maps[1:7]),
        kf_thr=4, response_thr=6, prefix=prefix + "_shock_low",
        mel_frame_sr=mel_frame_sr, n_frames=n_frames,
    )
    features |= dd  # 咬齿

    dd, _ = _summarize_mel_peak_features(
        all_peaks=_merge_mel_row_peak_maps(row_peak_maps[7:9]),
        kf_thr=3, response_thr=6, prefix=prefix + "_shock_med",
        mel_frame_sr=mel_frame_sr, n_frames=n_frames,
    )
    features |= dd  # 震颤 / 中频冲击

    dd, _ = _summarize_mel_peak_features(
        all_peaks=_merge_mel_row_peak_maps(row_peak_maps[-6:]),
        kf_thr=3, response_thr=5, prefix=prefix + "_shock_high",
        mel_frame_sr=mel_frame_sr, n_frames=n_frames,
    )
    features |= dd  # 秒表

    dd, _ = _summarize_mel_peak_features(
        all_peaks=_merge_mel_row_peak_maps(row_peak_maps[:]),
        kf_thr=3, response_thr=6, prefix=prefix + "_shock_all",
        mel_frame_sr=mel_frame_sr, n_frames=n_frames,
    )
    features |= dd  # 全频率

    # dd, _ = _summarize_mel_peak_features(
    #     all_peaks=_merge_mel_row_peak_maps(row_peak_maps[:3]),
    #     kf_thr=1, response_thr=9, prefix=prefix + "_shock_mada",
    #     mel_frame_sr=mel_frame_sr, n_frames=n_frames,
    # )
    # features |= dd  # 低频马达

    # dd, _ = _summarize_mel_peak_features(
    #     all_peaks=_merge_mel_row_peak_maps(row_peak_maps[3:7]),
    #     kf_thr=1, response_thr=9, prefix=prefix + "_shock_mada2",
    #     mel_frame_sr=mel_frame_sr, n_frames=n_frames,
    # )
    # features |= dd  # 低频马达 2

    low_count = max(features.get(prefix + "_shock_low_count_per_sec", 0.0), 0.0)
    high_count = max(features.get(prefix + "_shock_high_count_per_sec", 0.0), 0.0)
    features[prefix + "_shock_high_ratio"] = high_count / max(low_count + high_count, _EPS)

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


# ===================== 主入口 =====================
def extract_features_v4(data, sr, return_timing: bool = False):
    """v4: 在 v3 基础上把 Mel 峰检测改成 IIR 局部均值 + residual (E−M) 评分。"""
    timing: dict = {}
    t0_total = time.perf_counter()

    x_raw = np.asarray(data, dtype=float, order="C")

    # 预处理：高通去 DC，按采样率裁剪首尾各 0.5 秒
    t0 = time.perf_counter()
    x = process_data(x_raw, sr=sr, cutoff_low=20, cutoff_high=None)
    x = np.asarray(x, dtype=float, order="C")
    timing["process_data_sec"] = float(time.perf_counter() - t0)

    # 全局特征：mean 用原始信号（保留 DC 偏置信息），
    # 其余统计在去除 DC 的处理后信号上计算。
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

    # DWT：5 层 × (7 base + 8 shock + 4 SK) = 95 维
    t0 = time.perf_counter()
    features.update(dwt_features(x, sr=sr, wavelet="db22", level=4))
    timing["dwt_sec"] = float(time.perf_counter() - t0)

    # Mel：58 维
    t0 = time.perf_counter()
    features.update(mel_features(x, sr))
    timing["mel_sec"] = float(time.perf_counter() - t0)

    timing["total_sec"] = float(time.perf_counter() - t0_total)

    if return_timing:
        return features, timing
    return features


# ===================== 自检 =====================
if __name__ == "__main__":
    np.random.seed(0)
    data = np.random.randn(100000).astype(np.float32)  # 5s @ 20kHz
    feats, timing = extract_features_v4(data, 20000, return_timing=True)

    print(f"feature_count = {len(feats)}")
    print(f"timing = {timing}")

    # 打印部分示例
    print("--- global ---")
    for k in ["mean", "std", "duration", "skewness", "kurtosis", "zero_crossing_rate"]:
        print(f"{k:>26s} = {feats[k]:.6g}")
    print("--- dwt_0 (cA4) ---")
    for k in _SUBBAND_KEYS:
        print(f"{'dwt_0_' + k:>26s} = {feats['dwt_0_' + k]:.6g}")
    print("--- mel (samples) ---")
    for k in [
        "mel_band_all_frame_mean_std",
        "mel_band_low_db_ratio",
        "mel_band_mid_db_ratio",
        "mel_band_high_db_ratio",
        "mel_band_wide_std",
        "mel_band_hml_balance",
        "mel_shock_low_count_per_sec",
        "mel_shock_med_count_per_sec",
        "mel_shock_high_count_per_sec",
        "mel_shock_high_ratio",
    ]:
        print(f"{k:>26s} = {feats[k]:.6g}")
