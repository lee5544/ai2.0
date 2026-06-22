"""Generate the filtered training sample_view.csv from a project config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .build import generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 ML 规则生成并筛选 sample_view.csv")
    parser.add_argument("--config", required=True, help="项目 cfg YAML 路径")
    parser.add_argument("--output-folder", default="", help="可选输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    output_folder = Path(args.output_folder).expanduser() if args.output_folder else None
    output_path = generate(config, output_folder)
    print(f"[WRITE] {output_path}")


if __name__ == "__main__":
    main()
