"""MFCC 详情卡片（自包含）。"""
import numpy as np
import librosa
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from ..card_api import Card, CardContext, register


def _mfcc_figure(data, sr, n_mfcc):
    if data is None:
        return None
    n_fft = 256
    hop = n_fft // 2
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop, fmax=sr // 2)
    t = np.linspace(0, len(data) / sr, mfcc.shape[1])
    idx = np.arange(1, n_mfcc + 1)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Heatmap(z=mfcc, x=t, y=idx, colorscale="Viridis",
                             colorbar=dict(title="MFCC Coef"), showscale=True), row=1, col=1)
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="MFCC Index", height=400,
                      margin=dict(l=20, r=20, t=40, b=40), showlegend=False, autosize=True,
                      title="MFCC (Mel-frequency Cepstral Coefficients)")
    return fig


@register
class MfccCard(Card):
    id = "mfcc"; title = "MFCC 详情"; category = "spectrum"; default = False; order = 60
    params = [
        {"key": "n_mfcc", "label": "MFCC 系数数", "type": "number", "default": 13, "min": 4, "max": 40, "step": 1},
    ]

    def build(self, ctx: CardContext, p: dict):
        if ctx.proc is None:
            return None
        return _mfcc_figure(ctx.proc, ctx.sr, int(p["n_mfcc"]))
