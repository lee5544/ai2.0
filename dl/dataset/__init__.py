"""DL 数据集工具包（样本筛选 / 标签筛选 / 生成 / 切分）。

结构对齐 ml/dataset（sample_filter 复用 label_filter 的标准列/输出目录，单一来源）：
  - label_filter.py  训练标签筛选 + sample_view 标准化 + 输出目录（基础模块）
  - sample_filter.py 候选样本筛选（引用 label_filter）
  - build.py         生成 sample_view + mel 特征提取 CLI（引用上面两者）
（数据集加载与划分 split 已迁至 dl/training/split.py）
"""

from dl.dataset.label_filter import (
    STANDARD_SAMPLE_VIEW_COLUMNS,
    filter_sample_view_dataframe,
    resolve_output_dir,
    standardize_sample_view,
)
from dl.dataset.sample_filter import filter_samples
from dl.dataset.build import generate, main

__all__ = [
    "STANDARD_SAMPLE_VIEW_COLUMNS",
    "filter_sample_view_dataframe",
    "filter_samples",
    "generate",
    "resolve_output_dir",
    "standardize_sample_view",
    "main",
]
