#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import (
    apply_threshold_overrides,
    format_score_list,
    resolve_model_kwargs,
    resolve_runtime_context,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单文件 TDMS 快速预测")
    parser.add_argument("--tdms", required=True, help="TDMS 文件路径")
    parser.add_argument("--line", default=None, help="产线名（可选，例如 epump4）")
    parser.add_argument("--model-id", default=None, help="results 下模型目录名")
    parser.add_argument("--model-dir", default=None, help="模型目录路径（优先于 --model-id）")
    parser.add_argument("--threshold", type=float, default=None, help="统一阈值覆盖")
    parser.add_argument(
        "--threshold-dict",
        default=None,
        help='按 mlabel 覆盖阈值，JSON 字符串，例如 {"0":0.9,"1":0.8}',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    channel_predictor_cls, read_tdms, root_path, is_runtime_bundle = resolve_runtime_context(Path(__file__))
    predictor = channel_predictor_cls(
        **resolve_model_kwargs(
            args.model_dir,
            args.model_id,
            default_model_id="epump4_general_xgb",
            default_model_dir=root_path if is_runtime_bundle else None,
        )
    )
    apply_threshold_overrides(
        predictor,
        threshold=args.threshold,
        threshold_dict_json=args.threshold_dict,
    )

    tdms_data = read_tdms(args.tdms, line=args.line)
    sampling_rate = tdms_data.get("sampling_rate")
    if sampling_rate is None:
        raise ValueError(f"TDMS sampling rate is missing: {args.tdms}")

    up = predictor.predict_detail(tdms_data["up_data"], sampling_rate, "up")
    down = predictor.predict_detail(tdms_data["down_data"], sampling_rate, "down")
    sn = str(tdms_data.get("sn") or "").strip()
    up_sample_id = f"{sn}_up" if sn else ""
    down_sample_id = f"{sn}_down" if sn else ""

    print(
        f"{args.tdms} | sn={sn or '-'} | Up sample_id={up_sample_id or '-'} "
        f"| Up={up['result']},{up['mtype']},{up['score']:.6f} "
        f"| Down sample_id={down_sample_id or '-'} "
        f"| Down={down['result']},{down['mtype']},{down['score']:.6f}"
    )
    print(f"Up 分数列表={format_score_list(up['scores'])}")
    print(f"Down 分数列表={format_score_list(down['scores'])}")


if __name__ == "__main__":
    main()
