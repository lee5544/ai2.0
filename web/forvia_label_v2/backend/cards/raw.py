"""原始曲线卡片（自包含：下采样到 ~6000 点，时间轴用原始索引/采样率保持正确）。"""
import numpy as np
import plotly.graph_objs as go
from ..card_api import Card, CardContext, register

_MAX_PTS = 6000


def _waveform_figure(data, sample_rate):
    if data is None:
        return None
    arr = np.asarray(data, dtype=np.float32)
    n = int(arr.shape[0])
    idx = (np.linspace(0, n - 1, num=_MAX_PTS, dtype=np.int64) if n > _MAX_PTS
           else np.arange(n, dtype=np.int64))
    sr = float(sample_rate) if sample_rate else 20000.0
    t = idx.astype(np.float32) / sr
    # 用 SVG Scatter（非 WebGL Scattergl）：6000 点足够流畅，且避免 WebGL 上下文反复创建导致的浏览器内存泄漏
    fig = go.Figure(go.Scatter(x=t, y=arr[idx], mode="lines", line=dict(width=1, color="blue")))
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude", title="Time",
                      height=300, margin=dict(l=20, r=20, t=40, b=20), autosize=True)
    return fig


@register
class RawCard(Card):
    id = "raw"; title = "原始曲线"; category = "waveform"; default = True; order = 10

    def build(self, ctx: CardContext, p: dict):
        if ctx.raw is None:
            return None
        return _waveform_figure(ctx.raw, ctx.sr)
