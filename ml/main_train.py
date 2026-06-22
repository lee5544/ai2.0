#!/usr/bin/env python3
"""ML 训练入口（ml/ 包内）。

用法：
  python ml/main_train.py --config cfg/epump4.yaml
  python ml/main_train.py --config cfg/epump4.yaml --step dataset
  python ml/main_train.py --config cfg/epump4.yaml --step train
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.train import main  # noqa: E402

if __name__ == "__main__":
    main()
