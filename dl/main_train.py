#!/usr/bin/env python3
"""DL 训练入口（dl/ 包内）。

用法：
  python dl/main_train.py --config cfg/epump2_general_tcn.yaml
  python dl/main_train.py --config cfg/... --model-type tcn
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dl.train import main  # noqa: E402

if __name__ == "__main__":
    main()
