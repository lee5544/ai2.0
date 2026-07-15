#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "forvia_train_v2" / "frontend" / "index.html"


def main():
    text = HTML.read_text(encoding="utf-8")
    required = [
        "activeTrainStep",
        "train-step-workspace",
        "train-step-footer",
        "v-show=\"activeTrainStep==='dataset'\"",
        "v-show=\"activeTrainStep==='model'\"",
        "v-show=\"activeTrainStep==='validate'\"",
        "v-show=\"activeTrainStep==='features'\"",
        "v-show=\"activeTrainStep==='train'\"",
        "v-show=\"activeTrainStep==='results'\"",
        "saveCurrentTrainStep",
        "nextTrainStep",
        "prevTrainStep",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"missing single-step flow hooks: {missing}"


if __name__ == "__main__":
    main()
