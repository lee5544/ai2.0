#!/usr/bin/env python3
"""DL 数据集生成入口（dl/ 包内）。

完整流程（两步，输出均在 DL 专属目录）：
  Step 1  生成 sample_view.csv  → results/{line}_{model}/dl_dataset_csv/
  Step 2  提取 DL 特征          → 同一目录下的 dl_feature_batch_*.pkl

用法：
  python dl/main_dataset.py --config cfg/epump2_general_tcn.yaml
  python dl/main_dataset.py --config cfg/... --step sample-view
  python dl/main_dataset.py --config cfg/... --step extract
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dl.dataset.build import main  # noqa: E402

if __name__ == "__main__":
    main()
