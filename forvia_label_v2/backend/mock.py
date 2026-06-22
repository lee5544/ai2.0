"""演示数据：未配置 sample_view / label_history 时使用，保证开箱即跑。"""
from __future__ import annotations

import pandas as pd

from .config import LABEL_HISTORY_COLUMNS, TYPICAL_TAG


def mock_sample_view(n: int = 60) -> pd.DataFrame:
    lines = ["epump2", "epump3", "epump4", "etilt1"]
    rows = []
    for i in range(n):
        sn = f"SN{i:04d}"
        line = lines[i % len(lines)]
        for direction in ("up", "down"):
            rows.append({
                "sn": sn,
                "sample_id": f"{sn}_{direction}",
                "line": line,
                "relative_path": f"{line}/{sn}.tdms.zst",
            })
    return pd.DataFrame(rows)


def mock_label_history(sv: pd.DataFrame) -> pd.DataFrame:
    reasons = [("tick", 11, "tick"), ("weak_tick", 12, "weak_tick"),
               ("normal", 0, "normal"), ("other", 99, "other")]
    rows = []
    for i, (_, sv_row) in enumerate(sv.iterrows()):
        if i % 5 == 0:
            continue
        rkey, rid, rname = reasons[i % len(reasons)]
        note = TYPICAL_TAG + " 边界样本，复核确认" if (i % 7 == 0 and rname != "normal") else ""
        rows.append({
            "line": sv_row["line"], "sn": sv_row["sn"], "sample_id": sv_row["sample_id"],
            "timestamp": f"2026-06-1{i % 3} 1{i % 6}:00:00",
            "source": "model" if i % 3 == 0 else "human",
            "result_key": "ok" if rname == "normal" else "ng",
            "result_id": "0" if rname == "normal" else "1",
            "result_name": "OK" if rname == "normal" else "NG",
            "reason_key": rkey, "reason_id": str(rid), "reason_name": rname,
            "reason_confidence": f"{0.6 + (i % 4) * 0.1:.2f}",
            "label_version": "v4", "note": note,
        })
    return pd.DataFrame(rows, columns=LABEL_HISTORY_COLUMNS)
