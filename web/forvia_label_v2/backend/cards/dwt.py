"""小波分解 DWT 卡片（自包含）。"""
import numpy as np
import pywt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from scipy.stats import kurtosis
from ..card_api import Card, CardContext, register


def _sta_lta_ratio(x: np.ndarray, center: int, sta_win: int, lta_win: int) -> float:
    N = len(x)
    sta_win = max(1, int(sta_win))
    lta_win = max(sta_win + 1, int(lta_win))
    s1 = max(center - sta_win // 2, 0); e1 = min(center + (sta_win - sta_win // 2), N)
    s2 = max(center - lta_win // 2, 0); e2 = min(center + (lta_win - lta_win // 2), N)
    if s1 > s2 or e1 < e2:
        segs = []
        if s1 > s2:
            segs.append(x[s2:s1])
        if e1 < e2:
            segs.append(x[e1:e2])
        lta_seg = np.concatenate(segs) if segs else x[s2:e2]
    else:
        lta_seg = x[s2:e2]
    sta = float(np.mean(x[s1:e1] ** 2)) if e1 > s1 else 0.0
    lta = float(np.mean(lta_seg ** 2)) if lta_seg.size > 0 else (
        float(np.mean(x[s2:e2] ** 2)) if e2 > s2 else 0.0)
    return sta / (lta + 1e-12)


def _dwt_figure(data, wavelet, level, sr):
    if data is None:
        return None
    coeffs = pywt.wavedec(data, wavelet, level=level)
    n_rows = level + 1
    titles = []
    for idx in range(len(coeffs)):
        if idx == 0:
            k = level; lo, hi = 0.0, sr / (2 ** (k + 1)); name = f"A{k}"
        else:
            k = level - idx + 1; lo, hi = sr / (2 ** (k + 1)), sr / (2 ** k); name = f"D{k}"
        band = f"{lo:.1f}-{hi:.1f} Hz"
        titles.append(f"{name} Features ({band})")

    fig = make_subplots(rows=n_rows, cols=1, vertical_spacing=0.03, subplot_titles=titles)
    for idx, coeff in enumerate(coeffs):
        row = idx + 1
        fig.add_trace(go.Scatter(y=coeff, mode="lines", line=dict(color="blue", width=1), showlegend=False), row=row, col=1)
        peaks, _ = find_peaks(coeff, distance=100, height=float(np.mean(coeff) + np.std(coeff)))
        half = 128
        b_feas, b_eng = [], []
        for pk in peaks:
            local = coeff[max(pk - half, 0):min(pk + half, len(coeff))]
            b_feas.append(kurtosis(local, fisher=False) if local.size > 3 else 0.0)
            b_eng.append(_sta_lta_ratio(coeff, center=pk, sta_win=10, lta_win=20))
        fig.add_trace(go.Scatter(x=peaks, y=coeff[peaks], mode="markers+text",
                                 marker=dict(color="black", size=6),
                                 text=[f"{k:.1f}/{e:.1f}" for k, e in zip(b_feas, b_eng)],
                                 textposition="top center", showlegend=False), row=row, col=1)
    fig.update_layout(height=n_rows * 400, margin=dict(l=30, r=20, t=40, b=30), template="plotly_white")
    return fig


@register
class DwtCard(Card):
    id = "dwt"; title = "小波分解 DWT"; category = "spectrum"; default = False; order = 40
    params = [
        {"key": "wavelet", "label": "小波基", "type": "select", "default": "db22",
         "options": ["db4", "db8", "db22", "sym8", "coif5"]},
        {"key": "level", "label": "分解层数", "type": "number", "default": 4, "min": 1, "max": 8, "step": 1},
    ]

    def build(self, ctx: CardContext, p: dict):
        if ctx.proc is None:
            return None
        return _dwt_figure(ctx.proc, str(p["wavelet"]), int(p["level"]), ctx.sr)
