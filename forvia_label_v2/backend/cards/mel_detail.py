"""Mel 详情卡片（自包含）：Mel 谱能量 + 各频带曲线 + IIR 均值 + 冲击峰标注。"""
import numpy as np
import librosa
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import signal
from scipy.signal import find_peaks
from ..card_api import Card, CardContext, register


# ---- 数值工具（自包含，不依赖其它卡片/模块）----
def _prefix_sum_pow4(x):
    x = np.asarray(x, dtype=float, order="C")
    ps = [np.empty(x.size + 1, dtype=float) for _ in range(4)]
    for p in ps:
        p[0] = 0.0
    x2 = x * x
    np.cumsum(x, out=ps[0][1:]); np.cumsum(x2, out=ps[1][1:])
    np.cumsum(x2 * x, out=ps[2][1:]); np.cumsum(x2 * x2, out=ps[3][1:])
    return ps[0], ps[1], ps[2], ps[3]


def _segment_kurtosis_from_prefix(ps1, ps2, ps3, ps4, lefts, rights):
    lefts = np.asarray(lefts, dtype=int); rights = np.asarray(rights, dtype=int)
    n = (rights - lefts).astype(float)
    out = np.zeros_like(n, dtype=float)
    valid = n > 3
    if not np.any(valid):
        return out
    lv, rv, nv = lefts[valid], rights[valid], n[valid]
    s1 = ps1[rv] - ps1[lv]; s2 = ps2[rv] - ps2[lv]; s3 = ps3[rv] - ps3[lv]; s4 = ps4[rv] - ps4[lv]
    mu = s1 / nv; e2 = s2 / nv; e3 = s3 / nv; e4 = s4 / nv
    m2 = e2 - mu * mu
    m4 = e4 - 4.0 * mu * e3 + 6.0 * mu * mu * e2 - 3.0 * mu ** 4
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = m4 / (m2 * m2)
        n2 = nv * nv
        kurt = ((n2 - 1.0) * ratio - 3.0 * (nv - 1.0) ** 2) / ((nv - 2.0) * (nv - 3.0)) + 3.0
    kurt = np.where(m2 <= 0.0, 0.0, kurt)
    out[valid] = np.nan_to_num(kurt, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _rolling_std(arr, win):
    x = np.asarray(arr, dtype=float, order="C"); N = x.size
    if N == 0:
        return np.array([], dtype=float)
    win = max(1, int(win)); win_l = win // 2; win_r = win - win_l
    cs = np.empty(N + 1); cs[0] = 0.0; np.cumsum(x, out=cs[1:])
    cs2 = np.empty(N + 1); cs2[0] = 0.0; np.cumsum(x * x, out=cs2[1:])
    idx = np.arange(N)
    starts = np.clip(idx - win_l, 0, N); ends = np.clip(idx + win_r, 0, N); ns = ends - starts
    sums = cs[ends] - cs[starts]
    var = (cs2[ends] - cs2[starts] - (sums * sums) / np.maximum(ns, 1)) / np.maximum(ns - 1, 1)
    var[ns < 2] = 0.0; var[var < 0] = 0.0
    return np.sqrt(var)


def _iir_local_mean(row, equivalent_window=50):
    x = np.asarray(row, dtype=float, order="C")
    if x.size == 0:
        return x.copy()
    alpha = float(np.clip(2.0 / (max(int(equivalent_window), 1) + 1), 1e-6, 1.0))
    b = np.array([alpha]); a = np.array([1.0, -(1.0 - alpha)])
    zi = signal.lfilter_zi(b, a) * float(x[0])
    y, _ = signal.lfilter(b, a, x, zi=zi)
    return y.astype(x.dtype, copy=False)


def _row_peak_map(row, distance=5):
    row = np.asarray(row, dtype=float, order="C")
    local_mean = _iir_local_mean(row, equivalent_window=50)
    local_std = _rolling_std(row, 50)
    residual = row - local_mean
    thr = np.maximum(local_std, 1e-9)
    peaks, props = find_peaks(residual, height=thr, distance=distance, prominence=6, wlen=50, width=[1, 50])
    out = {}
    half = 10
    prom = props.get("prominences", [])
    if peaks.size:
        lefts = np.maximum(peaks - half, 0); rights = np.minimum(peaks + half + 1, len(row))
        ps1, ps2, ps3, ps4 = _prefix_sum_pow4(residual)
        kf = _segment_kurtosis_from_prefix(ps1, ps2, ps3, ps4, lefts, rights)
        for i, p in enumerate(peaks):
            out[int(p)] = {"kurtosis": float(kf[i]),
                           "prominences": float(prom[i]) if len(prom) > i else 0.0,
                           "db": float(row[int(p)])}
    return out


def _merge_peak_maps(maps, merge_distance=1):
    cand = []
    for m in maps:
        for p, s in m.items():
            cand.append({"frame": int(p), **{k: float(s.get(k, 0.0)) for k in ("kurtosis", "prominences", "db")}})
    if not cand:
        return {}
    cand.sort(key=lambda c: c["frame"])
    clusters = [[cand[0]]]; last = cand[0]["frame"]
    for c in cand[1:]:
        if c["frame"] - last <= max(0, int(merge_distance)):
            clusters[-1].append(c)
        else:
            clusters.append([c])
        last = c["frame"]
    res = {}
    for cl in clusters:
        rep = max(cl, key=lambda x: (x["prominences"], x["kurtosis"], x["db"]))
        res[int(rep["frame"])] = {"kurtosis": max(x["kurtosis"] for x in cl),
                                  "prominences": max(x["prominences"] for x in cl),
                                  "db": max(x["db"] for x in cl)}
    return dict(sorted(res.items()))


def _filter_peaks(all_peaks, kf_thr, prom_thr):
    def keep(s):
        kf = float(s.get("kurtosis", 0.0)); pr = float(s.get("prominences", 0.0))
        return (kf >= 5.0 and pr >= 12.0) or (kf >= kf_thr and pr >= prom_thr)
    return {x: s for x, s in all_peaks.items() if keep(s)}


def _mel_detail_figure(data, sr, n_mels):
    if data is None:
        return None
    n_fft = 256; hop = n_fft // 2
    mel = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels, fmax=sr // 2, n_fft=n_fft, hop_length=hop)
    freq = librosa.mel_frequencies(n_mels=n_mels, fmax=sr // 2)
    db = librosa.power_to_db(mel, ref=np.max)[:, 1:-1]
    total_energy = np.mean(db, axis=0)
    row_maps = [_row_peak_map(db[i, :], distance=5) for i in range(db.shape[0])]
    filted = _filter_peaks(_merge_peak_maps(row_maps[-5:]), kf_thr=3.0, prom_thr=8.0)
    peaks = list(filted.keys())
    kf = [v["kurtosis"] for v in filted.values()]
    eng = [v["prominences"] for v in filted.values()]

    t = np.arange(db.shape[1])
    total_rows = 1 + n_mels
    titles = ["Spectral Energy"] + [f"{int(f)} Hz" for f in freq]
    vspace = min(0.01, 0.2 / (total_rows - 1))
    fig = make_subplots(rows=total_rows, cols=1, row_heights=[1] * total_rows,
                        shared_xaxes=True, vertical_spacing=vspace, subplot_titles=titles)
    fig.add_trace(go.Scatter(x=t, y=total_energy, mode="lines", name="Spectral Energy",
                             line=dict(color="red")), row=1, col=1)
    fig.add_trace(go.Scatter(x=peaks, y=total_energy[peaks] if peaks else [], mode="markers+text",
                             marker=dict(color="black", size=6),
                             text=[f"{k:.1f}/{e:.1f}" for k, e in zip(kf, eng)],
                             textposition="top center", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t[peaks] if peaks else [], y=total_energy[peaks] if peaks else [],
                             mode="markers", marker=dict(color="orange", size=8), name="Peaks"), row=1, col=1)
    for i, f in enumerate(freq):
        band = db[i]
        fig.add_trace(go.Scatter(x=t, y=band, mode="lines", name=f"{int(f)} Hz", showlegend=False), row=2 + i, col=1)
        fig.add_trace(go.Scatter(x=t, y=_iir_local_mean(band, equivalent_window=50), mode="lines",
                                 line=dict(color="orange", width=1, dash="dash"), showlegend=False,
                                 hovertemplate="frame=%{x}<br>IIR mean=%{y:.2f} dB<extra></extra>"), row=2 + i, col=1)
    fig.update_layout(height=400 * total_rows, margin=dict(l=40, r=20, t=40, b=20), showlegend=False,
                      title="Mel Details: Spectrogram, Energy, and Individual Bands")
    fig.update_xaxes(title_text="Time (s)", row=total_rows, col=1)
    fig.update_yaxes(title_text="Power (dB)", row=1, col=1)
    return fig


@register
class MelDetailCard(Card):
    id = "mel_detail"; title = "Mel 详情"; category = "spectrum"; default = False; order = 70
    params = [
        {"key": "n_mels", "label": "Mel 频带数（频率分辨率）", "type": "number", "default": 39, "min": 8, "max": 128, "step": 1},
    ]

    def build(self, ctx: CardContext, p: dict):
        if ctx.proc is None:
            return None
        return _mel_detail_figure(ctx.proc, ctx.sr, int(p["n_mels"]))
