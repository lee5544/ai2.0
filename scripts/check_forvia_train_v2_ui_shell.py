#!/usr/bin/env python3
"""Static check for the Train v2 console shell."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "forvia_train_v2" / "frontend" / "index.html"


def main() -> None:
    text = HTML.read_text(encoding="utf-8")
    required = [
        "app-shell",
        "sidebar",
        "上海帆古自动化",
        "Industrial Model Training Console",
        "welcome-hero",
        "toggleSidebar",
        "setUiScale",
        "uiScale",
        "项目列表（模型名称）",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, missing


if __name__ == "__main__":
    main()
