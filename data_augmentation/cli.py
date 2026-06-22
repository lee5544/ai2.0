from __future__ import annotations

import argparse
import json

from .runner import AUGMENTATION_METHODS, run_augmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TDMS 原始振动信号数据增强")
    parser.add_argument(
        "--input-folder",
        action="append",
        required=True,
        help="递归扫描 TDMS / TDMS.ZST 的输入文件夹，可重复",
    )
    parser.add_argument(
        "--folder-count",
        action="append",
        type=int,
        default=[],
        help="对应输入文件夹中每个 TDMS 的生成数量，可重复",
    )
    parser.add_argument("--output-dir", required=True, help="增强 TDMS 与清单输出目录")
    parser.add_argument("--count", type=int, default=1, help="未提供 --folder-count 时的默认生成数量")
    parser.add_argument(
        "--methods",
        required=True,
        help=f"逗号分隔的方法，可选: {', '.join(AUGMENTATION_METHODS)}",
    )
    parser.add_argument("--line", default="", help="产线名称；无法从路径推断时使用")
    parser.add_argument("--seed", type=int, default=None, help="可选随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folder_count and len(args.folder_count) != len(args.input_folder):
        raise ValueError("--folder-count 数量必须与 --input-folder 一致")
    counts = args.folder_count or [args.count] * len(args.input_folder)
    result = run_augmentation(
        input_folders=list(zip(args.input_folder, counts)),
        output_dir=args.output_dir,
        methods=args.methods.split(","),
        line=args.line,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
