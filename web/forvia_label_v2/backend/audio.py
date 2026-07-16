"""音频流生成：把信号重采样到 22.05kHz 并打包成 WAV（供 /api/sample/{i}/audio.wav）。"""
from __future__ import annotations

import io

import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample


def generate_wav_stream(data, sample_rate: int = 20000):
    if data is None or len(data) == 0:
        return None
    sample_rate = int(sample_rate) if sample_rate else 20000
    fs_audio = 22050
    n = max(1, int(len(data) * fs_audio / sample_rate))
    d = resample(np.asarray(data, dtype=np.float32), n)
    d = np.nan_to_num(d, copy=False)
    d = np.clip(d, -1.0, 1.0)
    d16 = (d * 32767).astype(np.int16, copy=False)
    bio = io.BytesIO()
    wav.write(bio, fs_audio, d16)
    bio.seek(0)
    return bio
