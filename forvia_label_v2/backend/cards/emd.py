"""EMD 详情卡片（自包含，较慢）。"""
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from PyEMD import EMD
from ..card_api import Card, CardContext, register


def _emd_figure(data):
    if data is None:
        return None
    max_imf = 3
    imfs = EMD()(data, max_imf=max_imf)
    residual = data - np.sum(imfs, axis=0)
    if imfs.shape[0] < max_imf or len(residual) != len(data):
        return None
    fig = make_subplots(rows=max_imf + 1, cols=1, vertical_spacing=0.08)
    for i in range(max_imf):
        fig.add_trace(go.Scatter(y=imfs[i], mode="lines", line=dict(width=1), showlegend=False),
                      row=i + 1, col=1)
    fig.add_trace(go.Scatter(y=residual, mode="lines", line=dict(width=1, color="#FF7F0E"),
                             showlegend=False), row=max_imf + 1, col=1)
    fig.update_layout(height=800, margin=dict(l=30, r=20, t=40, b=30), template="plotly_white")
    return fig


@register
class EmdCard(Card):
    id = "emd"; title = "EMD 详情"; category = "spectrum"; default = False; order = 90

    def build(self, ctx: CardContext, p: dict):
        if ctx.proc is None:
            return None
        return _emd_figure(ctx.proc)
