#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "forvia_train_v2" / "frontend" / "index.html"


def main():
    text = HTML.read_text(encoding="utf-8")
    required = [
        ".train-step-workspace .step-subsection",
        ".run-strip",
        "class=\"action-flow run-strip\"",
        "class=\"augment-toggle\"",
        "点击开始 提取特征",
        "点击开始 训练模型",
        "step.id===s.activeTrainStep",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"missing flat train-step layout hooks: {missing}"


if __name__ == "__main__":
    main()
