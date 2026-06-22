#!/usr/bin/env python3
"""ML 推理预测入口（ml/ 包内）。

用法：
  python ml/main_predictor.py --tdms path/to/file.tdms
  python ml/main_predictor.py --tdms path/to/file.tdms --line epump4
  python ml/main_predictor.py --tdms path/to/file.tdms --threshold 0.8
  python ml/main_predictor.py --tdms path/to/file.tdms --threshold-dict '{"0": 0.8, "1": 0.7}'
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.predictor import main  # noqa: E402

if __name__ == "__main__":
    main()
