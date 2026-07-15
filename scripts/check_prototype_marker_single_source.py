#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = (ROOT / "forvia_label_v2" / "backend" / "config.py").read_text(encoding="utf-8")
    frontend = (ROOT / "forvia_label_v2" / "frontend" / "index.html").read_text(encoding="utf-8")
    assert 'TYPICAL_TAG = "[[prototype]]"' in config
    assert "[[典型异音]]" not in frontend
    assert "[[typical]]" not in frontend


if __name__ == "__main__":
    main()
