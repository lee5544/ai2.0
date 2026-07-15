#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "forvia_train_v2" / "frontend" / "index.html"


def main():
    text = HTML.read_text(encoding="utf-8")
    required = [
        ".train-step-btn.done.active",
        "trainStepCompletion",
        "function datasetConfigured()",
        "function modelConfigured()",
        "function markConfigStepsSaved()",
        "markConfigStepsSaved();if(!quiet)notify('训练配置已保存')",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"missing train step state flow hooks: {missing}"


if __name__ == "__main__":
    main()
