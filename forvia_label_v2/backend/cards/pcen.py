"""PCEN 冲击增强卡片（高级，自包含）。"""
from ..card_api import Card, CardContext, register

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import librosa  # noqa: E402

def generate_mel_pcen_figure(
    data,
    n_mels: int = 13,
    sr: int = 20000,
    n_fft: int = 256,
    hop_length: int = 128,
    *,
    time_constant: float = 0.5,
    gain: float = 0.8,
    bias: float = 10.0,
    power: float = 0.25,
    eps: float = 1e-6,
):
    """Mel 谱 + PCEN (Per-Channel Energy Normalization) 冲击增强显示。

    PCEN 对每个频带做自适应 AGC：长期稳态被压缩、短时瞬态被放大，
    用于在 Mel 谱图上"压底噪、抬冲击"。底部的低频稳态条带会被压暗，
    冲击对应的纵向条纹会更突出。

    参数选择经验：
      - time_constant 越小 (0.04~0.1)：AGC 响应越快，冲击越尖锐；越大 → 越像普通 Mel
      - gain 越大：AGC 越强，对稳态压制越狠
      - bias 抬高噪声基线，抑制弱伪峰
      - power 越小：动态范围压缩越强，弱冲击越明显
    """
    if data is None:
        return None
    n_fft = int(n_fft)
    hop_length = int(hop_length)
    n_mels = int(n_mels)

    y = np.asarray(data, dtype=np.float32)

    # 用功率谱 (power=2.0)，与 PCEN 推荐输入一致
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, fmax=sr // 2,
        n_fft=n_fft, hop_length=hop_length, power=2.0,
    )
    freq_axis = librosa.mel_frequencies(n_mels=n_mels, fmax=sr // 2)

    # 标准 PCEN 配方：先乘 2**31 把功率拉到"音频量级"，再做 PCEN
    S_pcen = librosa.pcen(
        S * (2 ** 31),
        sr=sr,
        hop_length=hop_length,
        time_constant=time_constant,
        gain=gain,
        bias=bias,
        power=power,
        eps=eps,
    )

    # 进一步把 PCEN 输出做 99 分位裁剪 + 归一化到 [0, 1]，避免极少数极强冲击
    # 把整张图压成"几乎全黑+一两个亮点"——保证视觉上稳态被压、冲击突出但仍可见
    if np.isfinite(S_pcen).any():
        vmax = float(np.percentile(S_pcen, 99.5))
        if vmax <= 0:
            vmax = float(S_pcen.max()) if S_pcen.size else 1.0
        S_show = np.clip(S_pcen / max(vmax, eps), 0.0, 1.0)
    else:
        S_show = np.zeros_like(S_pcen)

    fig = make_subplots(rows=1, cols=1, vertical_spacing=0.05)
    fig.add_trace(
        go.Heatmap(
            z=S_show,
            x=np.linspace(0, len(y) / sr, S_show.shape[1]),
            y=freq_axis,
            colorscale="Plasma",
            colorbar=dict(title="PCEN (norm)"),
            showscale=True,
            zmin=0.0,
            zmax=1.0,
        ),
        row=1,
        col=1,
    )
    fig.update_layout(
        xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
        height=400, margin=dict(l=20, r=20, t=40, b=40),
        showlegend=False, autosize=True,
        title=f"Mel Spectrogram (PCEN 冲击增强, tc={time_constant}s)",
    )
    return fig



@register
class PcenCard(Card):
    id = "pcen"; title = "PCEN 冲击增强"; category = "spectrum"; default = False; order = 50
    params = [
        {"key": "n_fft", "label": "n_fft（频率分辨率）", "type": "select", "default": 256,
         "options": [128, 256, 512, 1024, 2048]},
        {"key": "hop_length", "label": "hop（时间分辨率）", "type": "number", "default": 128, "min": 16, "max": 1024, "step": 16},
        {"key": "n_mels", "label": "Mel 频带数", "type": "number", "default": 13, "min": 8, "max": 128, "step": 1},
        {"key": "time_constant", "label": "time_constant", "type": "number", "default": 0.06, "min": 0.01, "max": 1.0, "step": 0.01},
    ]
    def build(self, ctx: CardContext, p: dict):
        if ctx.proc is None: return None
        return generate_mel_pcen_figure(ctx.proc, sr=ctx.sr, n_fft=int(p["n_fft"]),
                                        hop_length=int(p["hop_length"]), n_mels=int(p["n_mels"]),
                                        time_constant=float(p["time_constant"]))
