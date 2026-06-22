"""梅尔频谱卡片（自包含）。"""
import numpy as np
import librosa
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from ..card_api import Card, CardContext, register


def _mel_figure(data, sr, n_mels):
    if data is None:
        return None
    n_fft = 256
    mel = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels, fmax=sr // 2,
                                         n_fft=n_fft, hop_length=n_fft // 2)
    freq = librosa.mel_frequencies(n_mels=n_mels, fmax=sr // 2)
    db = librosa.power_to_db(mel, ref=np.max)
    fig = make_subplots(rows=1, cols=1, vertical_spacing=0.05)
    fig.add_trace(go.Heatmap(z=db, x=np.linspace(0, len(data) / sr, db.shape[1]), y=freq,
                             colorscale="Plasma", colorbar=dict(title="Power (dB)"), showscale=True),
                  row=1, col=1)
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Frequency (Hz)", height=400,
                      margin=dict(l=20, r=20, t=40, b=40), showlegend=False, autosize=True,
                      title="Mel Spectrogram")
    return fig


@register
class MelCard(Card):
    id = "mel"; title = "梅尔频谱"; category = "spectrum"; default = True; order = 30
    params = [
        {"key": "n_mels", "label": "Mel 频带数", "type": "number", "default": 26, "min": 8, "max": 128, "step": 1},
    ]

    def build(self, ctx: CardContext, p: dict):
        if ctx.proc is None:
            return None
        return _mel_figure(ctx.proc, ctx.sr, int(p["n_mels"]))
