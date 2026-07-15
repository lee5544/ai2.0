#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "forvia_label_v2" / "frontend" / "index.html"


def main():
    text = HTML.read_text(encoding="utf-8")
    required = [
        '@dblclick="openProtoSample(r)"',
        "function openProtoSample",
        "openProtoSample,",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"missing prototype navigation hooks: {missing}"


if __name__ == "__main__":
    main()
