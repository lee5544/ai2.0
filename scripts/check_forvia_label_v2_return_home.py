#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "forvia_label_v2" / "frontend" / "index.html"


def main():
    text = HTML.read_text(encoding="utf-8")
    required = [
        "← 返回开始",
        "window.location.href='/console'",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"missing label return-home hooks: {missing}"
    assert "file:///Users/liyong" not in text, "return-home must use the served /console route, not a file URL"


if __name__ == "__main__":
    main()
