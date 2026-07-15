#!/usr/bin/env python3
"""Static check for Train v2 training flow buttons."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "forvia_train_v2" / "frontend" / "index.html"


def main() -> None:
    text = HTML.read_text(encoding="utf-8")
    labels = ["配置数据集", "配置模型", "数据校验", "提取特征", "开始训练", "训练结果"]
    missing = [label for label in labels if label not in text]
    assert not missing, missing
    assert "train-step-console" in text
    assert "trainStepClass" in text
    assert "openTrainStep" in text
    assert "step.done" in text


if __name__ == "__main__":
    main()
