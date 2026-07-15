#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "forvia_train_v2" / "backend" / "main.py"
LABEL = ROOT / "forvia_label_v2" / "backend" / "main.py"


def main():
    for path in (TRAIN, LABEL):
        text = path.read_text(encoding="utf-8")
        required = [
            '@app.get("/console")',
            "forvia_console_preview.html",
            '@app.get("/forvia_console_modules.js")',
            "forvia_console_modules.js",
        ]
        missing = [item for item in required if item not in text]
        assert not missing, f"{path} missing console routes: {missing}"


if __name__ == "__main__":
    main()
