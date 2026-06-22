"""批量异常分数：对每条样本算 RMS / 峰度 / 峰值因子，作为简单异常分。

演示任务：替换 process() 内的计算即可接入真实模型推理。
"""
import numpy as np
from scipy.stats import kurtosis

from ..task_api import Task, register
from ..tdms_loader import load_sample


@register
class AnomalyScoreTask(Task):
    id = "anomaly_score"
    title = "批量异常分数（RMS/峰度）"
    params = []

    def process(self, session, index, p):
        sig = load_sample(session, index)
        x = sig.proc
        if x is None or len(x) == 0:
            return {}
        x = np.asarray(x, dtype=np.float64)
        rms = float(np.sqrt(np.mean(x ** 2)))
        kurt = float(kurtosis(x, fisher=True, bias=False)) if len(x) > 3 else 0.0
        peak = float(np.max(np.abs(x)))
        crest = float(peak / rms) if rms > 0 else 0.0
        return {"result": {"rms": round(rms, 5), "kurtosis": round(kurt, 3),
                           "crest": round(crest, 3)}}
