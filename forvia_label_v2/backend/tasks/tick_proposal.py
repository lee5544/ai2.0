"""启发式建议标签：峰度超过阈值则建议某 reason（演示"建议层"）。

产出 suggestion，不直接写入标签库；需人工在界面"采纳"才落到 label_history。
"""
import numpy as np
from scipy.stats import kurtosis

from ..task_api import Task, register
from ..tdms_loader import load_sample


@register
class TickProposalTask(Task):
    id = "tick_proposal"
    title = "建议标签（峰度启发式）"
    params = [
        {"key": "kurt_threshold", "label": "峰度阈值", "type": "number",
         "default": 5.0, "min": 0, "max": 50, "step": 0.5},
        {"key": "reason_key", "label": "命中时建议 reason", "type": "select",
         "default": "tick_tock", "options": ["tick_tock", "noise", "friction", "other"]},
    ]

    def process(self, session, index, p):
        sig = load_sample(session, index)
        x = sig.proc
        if x is None or len(x) <= 3:
            return {}
        kurt = float(kurtosis(np.asarray(x, dtype=np.float64), fisher=True, bias=False))
        result = {"kurtosis": round(kurt, 3)}
        if kurt >= float(p["kurt_threshold"]):
            conf = min(0.95, 0.5 + (kurt - float(p["kurt_threshold"])) * 0.05)
            return {"result": result,
                    "suggestion": {"reason_key": str(p["reason_key"]),
                                   "confidence": round(conf, 2),
                                   "note": f"task:tick_proposal kurt={kurt:.2f}"}}
        return {"result": result}
