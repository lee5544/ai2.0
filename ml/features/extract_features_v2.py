import os
import sys
import time
from functools import lru_cache, wraps
import numpy as np
import pywt


def _disable_numba_cache_for_librosa():
    # Python 3.13 + current librosa/numba 组合下，cache=True 会在导入阶段触发
    # "no locator available"。这里仅关闭 numba 装饰器缓存，不影响特征计算结果。
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
import librosa
import scipy.signal as signal
from scipy.signal import find_peaks
from scipy.stats import kurtosis

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ACC数据预处理
@lru_cache(maxsize=128)
def _butter_coeffs(sr: float, cutoff: float, btype: str, order: int):
    nyquist = sr / 2  # 奈奎斯特频率
    normal_cutoff = cutoff / nyquist
    return signal.butter(order, normal_cutoff, btype=btype, analog=False)


def butter_filter(data, sr, cutoff, btype='low', order=4):
    b, a = _butter_coeffs(float(sr), float(cutoff), str(btype), int(order))
    return signal.filtfilt(b, a, data).astype(np.float32)

def process_data(raw_data, sr=20000, cut_len=None, target_length=0, cutoff_low=20, cutoff_high=8000):
    # 对原始数据使用高通滤波，滤除20Hz以下
    if cutoff_low != None:
        raw_data = butter_filter(raw_data, sr=sr, cutoff=cutoff_low, btype='high')
    if cutoff_high != None:
        raw_data = butter_filter(raw_data, sr=sr, cutoff=cutoff_high, btype='low')

    filtered_data = raw_data
    
    cut_len = round(float(sr) * 0.5) if cut_len is None else max(0, int(cut_len))
    # 裁剪：如果数据长度足够，则去除首尾各 0.5 秒；否则直接使用原始数据
    if len(filtered_data) > 2 * cut_len:
        dat_cut = filtered_data[cut_len:-cut_len]
    else:
        dat_cut = filtered_data[:]
    # dat_cut = filtered_data

    # def max_normalize(
    #     data: np.ndarray,
    #     *,
    #     percentile: float = 99.0,
    #     percentile_range: float = 0.4,
    # ) -> np.ndarray:
    #     """基于百分位数方法，将振动数据归一化到 [-1, 1] 区间"""
    #     try:
    #         quantile = np.percentile(np.abs(data), percentile)
    #         scale_factor = (quantile + 1e-10) / percentile_range
    #         data_normalized = np.clip(data / scale_factor, -1, 1)
    #     except Exception as e:
    #         raise RuntimeError(f"归一化时发生未知错误: {e}")
    #     return data_normalized
    # dat_cut = max_normalize(dat_cut)

    # 归一化
    def rms_normalization(signal, target_rms=0.1):
        rms = np.sqrt(np.mean(signal**2))
        return signal * (target_rms / rms)
    dat_cut = rms_normalization(dat_cut)

    # 对齐长度：如果不足 target_length，补零；如果过长，则裁剪到 target_length
    if target_length > 0:
        current_length = len(dat_cut)
        if current_length < target_length:
            pad_length = target_length - current_length
            dat_cut = np.pad(dat_cut, (0, pad_length), mode='constant')
        elif current_length > target_length:
            dat_cut = dat_cut[:target_length]

    return dat_cut


# ===================== 基础数值工具 =====================
def _prefix_sum_sq(x: np.ndarray) -> np.ndarray:
    """返回 x**2 的前缀和（含 0），sum[a:b] = ps[b] - ps[a]"""
    x = np.asarray(x, dtype=float, order="C")
    ps = np.empty(x.size + 1, dtype=float)
    ps[0] = 0.0
    np.cumsum(x * x, out=ps[1:])
    return ps


def _prefix_sum_pow4(x: np.ndarray):
    """返回 x, x^2, x^3, x^4 的前缀和（含 0）"""
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
    """
    用前缀和批量计算区间 Pearson kurtosis。
    - fisher=False 返回 Pearson 峰度（normal=3）
    - bias 对齐 scipy.stats.kurtosis 的 bias 参数
    """
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
            kurt_v = ((n2 - 1.0) * ratio - 3.0 * (nv - 1.0) * (nv - 1.0)) / ((nv - 2.0) * (nv - 3.0)) + 3.0
        if fisher:
            kurt_v = kurt_v - 3.0

    # 与 scipy 保持一致：零方差区间会得到 nan（而不是 0）
    kurt_v = np.where(m2 <= 0.0, np.nan, kurt_v)
    out[valid] = kurt_v
    return out

# ===================== DWT 冲击特征 =====================
def _rolling_mean_std_numpy(arr: np.ndarray, win: int):
    """
    等价替代：s.rolling(win, center=True, min_periods=1).mean()/std().fillna(0)
    - 中心窗：左半 win_l = win//2，右半 win_r = win - win_l
    - 边界自动裁剪：每个位置的实际 n = end - start 可能小于 win
    - std 使用 ddof=1；当 n<2 时按 pandas 行为返回 0（原来是 NaN 后 fillna(0)）
    """
    x = np.asarray(arr, dtype=float, order="C")
    N = x.size
    if N == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    win = max(1, int(win))
    win_l = win // 2
    win_r = win - win_l

    # 前缀和：sum 与 sumsq
    cs = np.empty(N + 1, dtype=float); cs[0] = 0.0
    np.cumsum(x, out=cs[1:])
    cs2 = np.empty(N + 1, dtype=float); cs2[0] = 0.0
    np.cumsum(x * x, out=cs2[1:])

    idx = np.arange(N)
    starts = np.clip(idx - win_l, 0, N)
    ends   = np.clip(idx + win_r, 0, N)
    ns = ends - starts  # 每个位置的有效样本数

    sums = cs[ends] - cs[starts]
    means = sums / np.maximum(ns, 1)

    # 样本方差（ddof=1）。当 ns<2 -> 0（与 pandas std().fillna(0) 一致）
    sumsq = cs2[ends] - cs2[starts]
    # var_num = sumsq - sums^2 / n
    # 注意避免除零：n=0 的位置不会出现，因为 min_periods=1 -> n>=1；这里还是保护一下
    denom_n = np.maximum(ns, 1)
    var_num = sumsq - (sums * sums) / denom_n

    # ddof=1：除以 (n-1)。对 n<2 的位置置 0
    denom_var = np.maximum(ns - 1, 1)  # 临时分母
    var = var_num / denom_var
    var[ns < 2] = 0.0
    # 数值防护：浮点误差导致极小负值时截为 0
    var[var < 0] = 0.0

    stds = np.sqrt(var)
    return means, stds

def sta_lta_ratio_batch(
    x: np.ndarray,
    centers: np.ndarray,
    sta_win: int,
    lta_win: int,
    ps2: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    向量化计算多个 center 的 STA/LTA 比值（与逐点计算等价）。
    """
    x = np.asarray(x, dtype=float, order="C")
    centers = np.asarray(centers, dtype=int)
    N = x.size

    sta_win = max(1, int(sta_win))
    lta_win = max(sta_win + 1, int(lta_win))

    if ps2 is None:
        ps2 = _prefix_sum_sq(x)

    # mean_square([s, e)) = (ps2[e] - ps2[s]) / max(e-s, 1)
    def _ms_range(_ps2, s, e):
        length = np.maximum(e - s, 1)
        return (_ps2[e] - _ps2[s]) / length

    # STA 区间（中心短窗）
    sta_l = sta_win // 2
    sta_r = sta_win - sta_l
    s1 = np.clip(centers - sta_l, 0, N)
    e1 = np.clip(centers + sta_r, 0, N)
    sta_ms = _ms_range(ps2, s1, e1)

    # LTA 区间（中心长窗，去掉 STA 段）
    lta_l = lta_win // 2
    lta_r = lta_win - lta_l
    s2 = np.clip(centers - lta_l, 0, N)
    e2 = np.clip(centers + lta_r, 0, N)

    left_len  = np.maximum(s1 - s2, 0)
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

    # 退化兜底：极端边界导致两段都为 0（很少见），直接用整段 [s2, e2)
    bad = ~ok
    if np.any(bad):
        lta_ms[bad] = _ms_range(ps2, s2[bad], e2[bad])

    return sta_ms / (lta_ms + eps)

def _get_dwt_shock_features(data, window, prefix):
    """
    检测冲击事件（保留 rolling 作为阈值估计；其它环节加速）
    与原逻辑等价：rolling 阈值 + find_peaks + 每峰 kurtosis + 批量 STA/LTA + 聚合分位数
    """
    feats = {}
    x = np.asarray(data, dtype=float, order="C")
    N = x.size

    # step0 阈值（保留 rolling）
    W_thresh = 2000
    # ——替换 pandas：用等价 NumPy 前缀和版本，完全复刻 center=True, min_periods=1, ddof=1——
    local_mean, local_std = _rolling_mean_std_numpy(x, W_thresh)
    thresholds = local_mean + local_std

    # step1 找峰
    peaks, props = find_peaks(
        x,
        distance=100,
        # distance = max(1, window // 2)
        height=thresholds,
        # height=np.mean(x) * np.std(x),
        # prominence / width 可按需解注
        # prominence=0.01,
        # width=5
    )

    # step2 峰的局部特征
    half = max(1, window // 2)
    ps2 = _prefix_sum_sq(x)   # 供 STA/LTA 使用

    # === 批量计算所有峰的 STA/LTA（保持参数不变）===
    stalta_arr = sta_lta_ratio_batch(
        x, centers=peaks, sta_win=10, lta_win=50, ps2=ps2
    )

    k_arr = np.array([], dtype=float)
    if peaks.size:
        # 预先向量化边界
        lefts = np.maximum(peaks - half, 0)
        rights = np.minimum(peaks + half, N)
        ps1, ps2_m, ps3, ps4 = _prefix_sum_pow4(x)
        k_arr = _segment_kurtosis_from_prefix(
            ps1,
            ps2_m,
            ps3,
            ps4,
            lefts,
            rights,
            fisher=False,
            bias=True,
        )

    # step3 聚合特征（一次性分位数）
    e_arr = np.asarray(stalta_arr, dtype=float)
    M = k_arr.size

    cnt = int(np.sum((k_arr > 4.0) & (e_arr > 9.0))) if M else 0
    ratio = (cnt / M) if M else 0.0

    if M:
        k_q50, k_q75, k_q95 = np.percentile(k_arr, [50, 75, 95])
        e_q50, e_q75, e_q95 = np.percentile(e_arr, [50, 75, 95])
        feats = {
            f"{prefix}_count": M,
            f"{prefix}_gt4": cnt,
            f"{prefix}_gt4_ratio": ratio,

            f"{prefix}_kurt_mean": float(k_arr.mean()),
            f"{prefix}_kurt_q50":  float(k_q50),
            f"{prefix}_kurt_q75":  float(k_q75),
            f"{prefix}_kurt_q95":  float(k_q95),

            f"{prefix}_energy_mean": float(e_arr.mean()),
            f"{prefix}_energy_q50":  float(e_q50),
            f"{prefix}_energy_q75":  float(e_q75),
            f"{prefix}_energy_q95":  float(e_q95),
        }
    else:
        feats = {
            f"{prefix}_count": 0,
            f"{prefix}_gt4": 0,
            f"{prefix}_gt4_ratio": 0.0,

            f"{prefix}_kurt_mean": 0.0,
            f"{prefix}_kurt_q50":  0.0,
            f"{prefix}_kurt_q75":  0.0,
            f"{prefix}_kurt_q95":  0.0,

            f"{prefix}_energy_mean": 0.0,
            f"{prefix}_energy_q50":  0.0,
            f"{prefix}_energy_q75":  0.0,
            f"{prefix}_energy_q95":  0.0,
        }

    return feats

def dwt_features(data, wavelet='db22', level=4):
    """
    检测冲击事件（保留 rolling 作为阈值估计；其它环节加速）
    """
    x = np.asarray(data, dtype=float, order="C")
    coeffs = pywt.wavedec(x, wavelet, level=level)

    features = {}
    prefix = "dwt"

    for i, coeff in enumerate(coeffs[:]): # 频率从低到高
        ci = np.asarray(coeff, dtype=float, order="C")

        kf = float(kurtosis(ci, fisher=False, bias=False)) if ci.size > 3 else 0.0
        q50, q75, q95 = np.percentile(ci, [50, 75, 95])
        base = {
            f"{prefix}_{i}_kurt": float(kf),
            f"{prefix}_{i}_std": float(ci.std()),
            f"{prefix}_{i}_n50": float(q50),
            f"{prefix}_{i}_n75": float(q75),
            f"{prefix}_{i}_n95": float(q95),
        }
        features.update(base)

        W = 256 # 计算峰度的窗口
        # W = 256 * (len(coeffs)-i)
        shock = _get_dwt_shock_features(ci, W, f"{prefix}_{i}_shock")
        features.update(shock)

    return features

# ===================== Mel 频带与冲击特征 =====================
def _get_bands_features(mel_spectrogram_db, freq_axis, sr, prefix):
    """
    频带特征（低/中/高）与整体帧均值std（可视为噪声起伏）。
    - mel_spectrogram_db: (n_mels, n_frames) 的 dB 值
    - freq_axis: 各 mel bin 的中心频率（Hz），长度 n_mels
    - 占比在“线性功率域”计算，更合理
    """
    # 使用稳定的语义列名，避免不同采样率下 mel bin 边界变化导致特征列名漂移。
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

    feats = {}

    # —— 帧均值的 std（衡量时域上帧间起伏）——
    frame_mean_db = np.mean(mel_spectrogram_db, axis=0)   # 每帧在dB域的均值
    feats[f"{prefix}_all_frame_mean_std"] = float(np.std(frame_mean_db))

    # —— dB -> 线性功率，再做能量占比 —— #
    mel_power = np.power(10.0, mel_spectrogram_db / 10.0)  # (n_mels, n_frames)
    total_power = float(np.sum(mel_power)) + 1e-12

    fmin, fmax = float(freq_axis[0]), float(freq_axis[-1])

    for band_name, lo, hi in bands:
        # 限制在可用范围内
        lo_ = max(lo, fmin)
        hi_ = min(hi, fmax)

        # 选择对应的 mel bin 区间
        i0 = int(np.searchsorted(freq_axis, lo_, side="left"))
        i1 = int(np.searchsorted(freq_axis, hi_, side="right"))

        # 防御：确保至少有一个bin
        if i1 - i0 < 1:
            i0 = max(i0 - 1, 0)
            i1 = min(i0 + 1, n_mels)

        # 该频带在 dB 域的帧序列（先对mel bin求均值，再对时间做统计）
        band_db_series = np.mean(mel_spectrogram_db[i0:i1, :], axis=0)
        # 该频带的总线性功率占比
        band_power_sum = float(np.sum(mel_power[i0:i1, :]))
        band_power_ratio = band_power_sum / total_power

        key = f"{prefix}_{band_name}"
        feats.update({
            f"{key}_db_mean": float(np.mean(band_db_series)),
            f"{key}_db_std":  float(np.std(band_db_series)),
            f"{key}_db_ratio": band_power_ratio,
        })

    return feats

def _get_mel_shock_features(bands, sr, distance=5, kf_thr=3, prom_thr=9, prefix="mel_shock"):
    """
    Mel 频带冲击特征提取
    """
    features = {}
    row_peak_maps = []
    for row in np.asarray(bands, dtype=float, order="C"):
        row_peak_maps.append(_calc_mel_row_peak_map(row, distance=distance))
    all_peaks = _merge_mel_row_peak_maps(row_peak_maps)
    features, filted_peaks = _summarize_mel_peak_features(
        all_peaks=all_peaks,
        kf_thr=kf_thr,
        prom_thr=prom_thr,
        prefix=prefix,
    )
    return features, filted_peaks


def _calc_mel_row_peak_map(row: np.ndarray, distance: int = 5) -> dict:
    row = np.asarray(row, dtype=float, order="C")
    W = 50
    local_mean, local_std = _rolling_mean_std_numpy(row, W)
    thresholds = local_mean + local_std
    peaks, props = find_peaks(
        row,
        height=thresholds,
        distance=distance,
        prominence=6,
        wlen=50,
        width=[1, 50],
    )

    peak_map = {}
    half = 10
    prom_arr = props.get("prominences", [])
    if peaks.size:
        lefts = np.maximum(peaks - half, 0)
        rights = np.minimum(peaks + half + 1, row.size)
        ps1, ps2_m, ps3, ps4 = _prefix_sum_pow4(row)
        kf_arr = _segment_kurtosis_from_prefix(
            ps1,
            ps2_m,
            ps3,
            ps4,
            lefts,
            rights,
            fisher=False,
            bias=False,
        )
        for i, p in enumerate(peaks):
            kf = float(kf_arr[i])
            prom = float(prom_arr[i]) if i < len(prom_arr) else 0.0
            peak_map[int(p)] = {
                "kurtosis": kf,
                "prominences": prom,
                "db": float(row[p]),
            }
    return peak_map


def _merge_mel_row_peak_maps(row_peak_maps: list[dict]) -> dict:
    all_peaks = {}
    for row_peaks in row_peak_maps:
        for p, s in row_peaks.items():
            if p not in all_peaks:
                all_peaks[p] = {
                    "kurtosis": s["kurtosis"],
                    "prominences": s["prominences"],
                    "db": s["db"],
                }
            else:
                all_peaks[p]["kurtosis"] = max(all_peaks[p]["kurtosis"], s["kurtosis"])
                all_peaks[p]["prominences"] = max(all_peaks[p]["prominences"], s["prominences"])
                all_peaks[p]["db"] = max(all_peaks[p]["db"], s["db"])
    return all_peaks


def _summarize_mel_peak_features(*, all_peaks: dict, kf_thr: float, prom_thr: float, prefix: str):
    features = {}

    # ---- step1: 过滤峰 ----
    filted_peaks = {
        x: s for x, s in all_peaks.items()
        if s["kurtosis"] >= kf_thr and s["prominences"] >= prom_thr
    }

    # ---- step2: 合并峰的特征 ----
    if filted_peaks:
        k_list = np.fromiter((v["kurtosis"] for v in filted_peaks.values()), dtype=float)
        p_list = np.fromiter((v["prominences"] for v in filted_peaks.values()), dtype=float)
        M = k_list.size

        features[prefix + "_count"] = int(M)
        features[prefix + "_kurt_gt4"] = int(np.sum((k_list > 4) & (p_list > 9)))
        for name, arr in (("kurtosis", k_list), ("prominences", p_list)):
            features[prefix + f"_{name}_sum"] = float(np.sum(arr))
            features[prefix + f"_{name}_std"] = float(np.std(arr))
            features[prefix + f"_{name}_mean"] = float(arr.mean())
            features[prefix + f"_{name}_q50"]  = float(np.percentile(arr, 50))
            features[prefix + f"_{name}_q75"]  = float(np.percentile(arr, 75))
            features[prefix + f"_{name}_q95"]  = float(np.percentile(arr, 95))
    else:
        features[prefix + "_count"] = 0
        features[prefix + "_kurt_gt4"] = 0
        for name in ["kurtosis", "prominences"]:
            features[prefix + f"_{name}_sum"] = 0.0
            features[prefix + f"_{name}_std"] = 0.0
            features[prefix + f"_{name}_mean"] = 0.0
            features[prefix + f"_{name}_q50"]  = 0.0
            features[prefix + f"_{name}_q75"]  = 0.0
            features[prefix + f"_{name}_q95"]  = 0.0

    return features, filted_peaks

def mel_features(data, sr):
    features = {}

    prefix = "mel"

    n_fft = 256
    n_mels = 13
    mel_spectrogram = librosa.feature.melspectrogram(
        y=np.asarray(data, dtype=float, order="C"),
        sr=sr, n_mels=n_mels, fmax=sr // 2, n_fft=n_fft, hop_length=n_fft//2
    )
    freq_axis = librosa.mel_frequencies(n_mels=n_mels, fmax=sr // 2) # 每个bin的中心频率
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)[:,1:-1] # np.max/median
    row_peak_maps = [
        _calc_mel_row_peak_map(mel_spectrogram_db[i, :], distance=5)
        for i in range(mel_spectrogram_db.shape[0])
    ]
    
    features |= _get_bands_features(mel_spectrogram_db, freq_axis, sr, prefix+"_band")

    # # 冲击检测：摩擦、秒表、杂音 --> 有用
    dd, _ = _summarize_mel_peak_features(
        all_peaks=_merge_mel_row_peak_maps(row_peak_maps[1:7]),
        kf_thr=4,
        prom_thr=9,
        prefix=prefix+"_shock"+"_low",
    )
    features |= dd ## 咬齿

    dd, _ = _summarize_mel_peak_features(
        all_peaks=_merge_mel_row_peak_maps(row_peak_maps[7:9]),
        kf_thr=3,
        prom_thr=9,
        prefix=prefix+"_shock"+"_med",
    )
    features |= dd ## 震颤 / 中频冲击

    dd, _ = _summarize_mel_peak_features(
        all_peaks=_merge_mel_row_peak_maps(row_peak_maps[-6:]),
        kf_thr=3,
        prom_thr=6,
        prefix=prefix+"_shock"+"_high",
    )
    features |= dd ## 秒表

    dd, _ = _summarize_mel_peak_features(
        all_peaks=_merge_mel_row_peak_maps(row_peak_maps[:3]),
        kf_thr=1,
        prom_thr=9,
        prefix=prefix+"_shock"+"_mada",
    )
    features |= dd ## 低频马达。没有峰度，但是响度变化大
    dd, _ = _summarize_mel_peak_features(
        all_peaks=_merge_mel_row_peak_maps(row_peak_maps[3:7]),
        kf_thr=1,
        prom_thr=9,
        prefix=prefix+"_shock"+"_mada2",
    )
    features |= dd ## 低频马达。没有峰度，但是响度变化大
    
    # 反秒表：目前只存在etilt的齿轮箱，不考虑
    # dd, _ = _get_mel_shock_features(-mel_spectrogram_db[-2:,:], sr=sr, kf_thr=3, prom_thr=6, prefix=prefix+"_shock"+"_-high")
    # features |= dd

    return features

def extract_features_v2(data, sr, return_timing=False):
    features = {}
    timing = {}
    t0_total = time.perf_counter()

    # # 全局特征
    t0 = time.perf_counter()
    x = np.asarray(data, dtype=float, order="C")
    features |= {
        "mean": np.mean(x),
        "time_std": np.std(x),
        "duration": len(x) / sr,
    }
    timing["global_sec"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    data_processed = process_data(x, sr=sr, cutoff_low=20, cutoff_high=None)
    timing["process_data_sec"] = float(time.perf_counter() - t0)

    # 频域特征
    t0 = time.perf_counter()
    features |= mel_features(data_processed, sr)
    timing["mel_sec"] = float(time.perf_counter() - t0)

    # 小波特征 --> 震颤/抖动
    t0 = time.perf_counter()
    features |= dwt_features(data_processed)
    timing["dwt_sec"] = float(time.perf_counter() - t0)
    timing["total_sec"] = float(time.perf_counter() - t0_total)

    if return_timing:
        return features, timing
    return features


# 测试
if __name__ == "__main__":
    data = np.random.randn(90000)  # 模拟数据
    features = extract_features_v2(data, 20000)
