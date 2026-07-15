#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "forvia_label_v2" / "frontend" / "index.html",
    ROOT / "forvia_train_v2" / "frontend" / "index.html",
]


def main():
    required = [
        "--bg:#f5f7fb",
        "--card:#fff",
        "--side-w:240px",
        "上海帆古自动化",
        "Industrial",
        "border-radius:12px",
        "box-shadow:0 2px 8px rgba(20,40,80,.035)",
        ".side-tab {",
        ".mini-btn.active",
    ]
    missing = {}
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        missed = [item for item in required if item not in text]
        if missed:
            missing[str(path.relative_to(ROOT))] = missed
    assert not missing, f"missing unified console visual hooks: {missing}"


if __name__ == "__main__":
    main()
