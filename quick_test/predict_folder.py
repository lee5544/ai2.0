#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from _bootstrap import (
    apply_threshold_overrides,
    format_score_list,
    resolve_model_kwargs,
    resolve_runtime_context,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量预测文件夹内的 TDMS / TDMS.ZST 文件（宽表输出）")
    parser.add_argument("--folder", required=True, help="待预测 TDMS 目录")
    parser.add_argument("--line", default=None, help="产线名（可选，例如 epump4）")
    parser.add_argument("--pattern", default="*", help="文件匹配模式（默认递归扫描全部 TDMS / TDMS.ZST）")
    parser.add_argument("--output", default=None, help="输出 CSV 路径（默认: <folder>/model.csv）")
    parser.add_argument("--model-id", default=None, help="results 下模型目录名")
    parser.add_argument("--model-dir", default=None, help="模型目录路径（优先于 --model-id）")
    parser.add_argument("--threshold", type=float, default=None, help="统一阈值覆盖")
    parser.add_argument(
        "--threshold-dict",
        default=None,
        help='按 mlabel 覆盖阈值，JSON 字符串，例如 {"0":0.9,"1":0.8}',
    )
    return parser.parse_args()


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _is_tdms_file(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and (name.endswith(".tdms") or name.endswith(".tdms.zst"))


def _merge_result(up: str | None, down: str | None) -> str | None:
    if up is None and down is None:
        return None
    if up == "NOK" or down == "NOK":
        return "NOK"
    return "OK"


def _predict_detail(predictor, channel, sampling_rate: int, direction: str) -> dict:
    if hasattr(predictor, "predict_detail"):
        return predictor.predict_detail(channel, sampling_rate, direction)
    result, mtype = predictor.predict(channel, sampling_rate, direction)
    return {"result": result, "mtype": mtype, "score": None, "scores": {}}


def _sample_ids(sn: object) -> tuple[str | None, str | None]:
    normalized = str(sn or "").strip()
    if not normalized:
        return None, None
    return f"{normalized}_up", f"{normalized}_down"


def main() -> None:
    args = _parse_args()
    import pandas as pd

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

    tdms_folder = Path(args.folder).expanduser().resolve()
    if not tdms_folder.exists() or not tdms_folder.is_dir():
        raise NotADirectoryError(f"无效目录: {tdms_folder}")

    output_csv = Path(args.output).expanduser().resolve() if args.output else tdms_folder / "model.csv"

    tdms_files = [
        path
        for path in sorted(tdms_folder.rglob(args.pattern))
        if _is_tdms_file(path) and not _is_hidden(path)
    ]
    if not tdms_files:
        raise FileNotFoundError(f"目录中未找到 .tdms 或 .tdms.zst 文件: {tdms_folder}")
    print(f"发现 {len(tdms_files)} 个 TDMS / TDMS.ZST 文件")

    records: list[dict] = []
    for tdms_file in tdms_files:
        label_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        up_detail = down_detail = {}
        up_result = up_type = down_result = down_type = None
        error = None
        sn = reference = file_time = None

        try:
            tdms_data = read_tdms(tdms_file, line=args.line)
            sn = tdms_data.get("sn")
            reference = tdms_data.get("reference")
            file_time = tdms_data.get("time")
            sampling_rate = tdms_data.get("sampling_rate")
            if sampling_rate is None:
                raise ValueError("TDMS sampling rate is missing")

            up_detail = _predict_detail(predictor, tdms_data["up_data"], sampling_rate, "up")
            down_detail = _predict_detail(predictor, tdms_data["down_data"], sampling_rate, "down")
            up_result, up_type = up_detail["result"], up_detail["mtype"]
            down_result, down_type = down_detail["result"], down_detail["mtype"]
            print(
                f"{tdms_file.name} | Up={up_result},{up_type},{up_detail.get('score')} "
                f"| Down={down_result},{down_type},{down_detail.get('score')}"
            )
            print(f"  Up 分数列表={format_score_list(up_detail.get('scores'))}")
            print(f"  Down 分数列表={format_score_list(down_detail.get('scores'))}")
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            up_result = "ERROR"
            down_result = "ERROR"
            print(f"{tdms_file.name} | ERROR={error}")

        up_sample_id, down_sample_id = _sample_ids(sn)
        records.append(
            {
                "filename": tdms_file.name,
                "tdms_path": str(tdms_file),
                "sn": sn,
                "sample_id": json.dumps(
                    [value for value in (up_sample_id, down_sample_id) if value],
                    ensure_ascii=False,
                ),
                "up_sample_id": up_sample_id,
                "down_sample_id": down_sample_id,
                "reference": reference,
                "time": file_time,
                "up": up_result,
                "up_type": up_type,
                "up_score": up_detail.get("score"),
                "up_raw_type": up_detail.get("raw_mtype"),
                "up_raw_score": up_detail.get("raw_score"),
                "up_scores": json.dumps(up_detail.get("scores", {}), ensure_ascii=False),
                "down": down_result,
                "down_type": down_type,
                "down_score": down_detail.get("score"),
                "down_raw_type": down_detail.get("raw_mtype"),
                "down_raw_score": down_detail.get("raw_score"),
                "down_scores": json.dumps(down_detail.get("scores", {}), ensure_ascii=False),
                "result": _merge_result(up_result, down_result),
                "source": "model",
                "timestamp": label_time,
                "error": error,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("-" * 60)
    print(f"已保存 {len(records)} 条记录到: {output_csv}")


if __name__ == "__main__":
    main()
