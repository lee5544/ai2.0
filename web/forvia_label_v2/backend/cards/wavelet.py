"""Wavelet（连续小波 CWT）详情卡片（自包含）。"""
import numpy as np
import pywt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from ..card_api import Card, CardContext, register


def _wavelet_figure(data, sr):
    if data is None:
        return None
    data = np.asarray(data, dtype=np.float32)
    cf = pywt.central_frequency("cmor1.5-1.0")
    target_freqs = np.linspace(100, 8000, 25)
    scales = (cf * sr) / target_freqs
    coeffs, freqs = pywt.cwt(data, scales, "cmor1.5-1.0", sampling_period=1.0 / sr)
    max_abs = np.max(np.abs(coeffs))
    mag_db = 20 * np.log10(np.abs(coeffs) / (max_abs + 1e-6))
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Heatmap(z=mag_db, x=np.arange(mag_db.shape[1]) / sr, y=freqs,
                             colorscale="Plasma", colorbar=dict(title="Power (dB)"), showscale=True),
                  row=1, col=1)
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Frequency (Hz)", height=400,
                      margin=dict(l=20, r=20, t=40, b=40), showlegend=False, autosize=True,
                      title="Wavelet Transform (Morlet)")
    return fig


@register
class WaveletCard(Card):
    id = "wavelet"; title = "Wavelet 详情"; category = "spectrum"; default = False; order = 80

    def build(self, ctx: CardContext, p: dict):
        if ctx.proc is None:
            return None
        return _wavelet_figure(ctx.proc, ctx.sr)
