#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from _bootstrap import apply_threshold_overrides, resolve_model_kwargs, resolve_runtime_context


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取 sample_view.csv 并批量预测，导出 sample_view_model.csv")
    parser.add_argument("--input-csv", required=True, help="输入 sample_view.csv 路径")
    parser.add_argument(
        "--tdms-root",
        default=None,
        help="TDMS 根目录（可选；当 CSV 里没有可直接定位 TDMS 的路径时必填）",
    )
    parser.add_argument("--sn-col", default="sn", help="SN 列名（默认: sn）")
    parser.add_argument("--sample-id-col", default="sample_id", help="sample_id 列名（默认: sample_id）")
    parser.add_argument("--line-col", default="line", help="line 列名（默认: line）")
    parser.add_argument(
        "--relative-path-col",
        default="relative_path",
        help="CSV 中 TDMS 相对路径列名（默认: relative_path）",
    )
    parser.add_argument("--line", default=None, help="产线名（可选，例如 epump4）")
    parser.add_argument(
        "--output",
        default=None,
        help="输出 CSV 路径（默认: 与输入同目录的 sample_view_model.csv）",
    )
    parser.add_argument("--model-id", default=None, help="results 下模型目录名")
    parser.add_argument("--model-dir", default=None, help="模型目录路径（优先于 --model-id）")
    parser.add_argument("--threshold", type=float, default=None, help="统一阈值覆盖")
    parser.add_argument(
        "--threshold-dict",
        default=None,
        help='按 mlabel 覆盖阈值，JSON 字符串，例如 {"0":0.9,"1":0.8}',
    )
    return parser.parse_args()


def _find_tdms_by_sn(sn: str, tdms_files: list[Path]) -> Path | None:
    matches = [p for p in tdms_files if sn in p.stem]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    # 多个匹配时优先文件名更短的，减少误命中概率
    matches.sort(key=lambda p: len(p.name))
    return matches[0]


def _norm_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _direction_from_sample_id(sample_id: str) -> str:
    sid = _norm_text(sample_id).lower()
    if sid.endswith("_up"):
        return "up"
    if sid.endswith("_down"):
        return "down"
    if sid.endswith("-up"):
        return "up"
    if sid.endswith("-down"):
        return "down"
    return ""


def _read_csv_with_fallback(path: Path):
    import pandas as pd

    last_exc: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise RuntimeError(f"CSV 读取失败: {path} | {type(last_exc).__name__}: {last_exc}") from last_exc


def _resolve_tdms_from_row(
    row_dict: dict,
    *,
    tdms_root: Path | None,
    input_csv_dir: Path,
    relative_path_col: str,
    sn_col: str,
    tdms_files: list[Path],
) -> tuple[Path | None, str | None]:
    tdms_path_candidates = (
        row_dict.get("tdms_path"),
        row_dict.get("m_tdms_path"),
        row_dict.get("file_path"),
        row_dict.get("path"),
        row_dict.get(relative_path_col),
    )

    for raw_path in tdms_path_candidates:
        text = _norm_text(raw_path)
        if not text:
            continue
        path_obj = Path(text).expanduser()
        if path_obj.is_absolute():
            if path_obj.exists() and path_obj.is_file():
                return path_obj.resolve(), None
        else:
            rel = Path(text)
            for base in (tdms_root, input_csv_dir):
                if base is None:
                    continue
                candidate = (base / rel).resolve()
                if candidate.exists() and candidate.is_file():
                    return candidate, None

    sn = _norm_text(row_dict.get(sn_col))
    if not sn:
        return None, "empty sn"
    if not tdms_files:
        return None, "tdms not found (missing --tdms-root and no valid path columns)"

    tdms_file = _find_tdms_by_sn(sn, tdms_files)
    if tdms_file is None:
        return None, "tdms not found by sn"
    return tdms_file, None


def _pick_row_label(
    *,
    sample_id: str,
    up_result: str | None,
    up_type: str | None,
    down_result: str | None,
    down_type: str | None,
) -> tuple[str | None, str | None]:
    direction = _direction_from_sample_id(sample_id)
    if direction == "up":
        return up_result, up_type
    if direction == "down":
        return down_result, down_type

    if up_result is None and down_result is None:
        return None, None
    if up_result == "NOK" or down_result == "NOK":
        # 未知方向时，结果从严按 NOK；类型优先返回可用的异常类型
        fault_type = down_type if down_result == "NOK" and down_type else up_type
        return "NOK", fault_type
    return "OK", up_type or down_type


def main() -> None:
    args = _parse_args()
    import pandas as pd
    from tqdm import tqdm

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

    input_csv = Path(args.input_csv).expanduser().resolve()
    tdms_root = Path(args.tdms_root).expanduser().resolve() if args.tdms_root else None
    output_csv = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_csv.with_name(f"{input_csv.stem}_model.csv")
    )

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV 不存在: {input_csv}")
    if tdms_root is not None and (not tdms_root.exists() or not tdms_root.is_dir()):
        raise NotADirectoryError(f"TDMS 根目录无效: {tdms_root}")

    df = _read_csv_with_fallback(input_csv)
    if args.sn_col not in df.columns:
        raise KeyError(f"未找到 SN 列: {args.sn_col}; 现有列: {list(df.columns)}")

    has_sample_id = args.sample_id_col in df.columns
    input_csv_dir = input_csv.parent

    tdms_files: list[Path] = []
    if tdms_root is not None:
        tdms_files = sorted(
            p for p in tdms_root.rglob("*.tdms") if not any(part.startswith(".") for part in p.parts)
        )
    print(f"已索引 TDMS 文件数: {len(tdms_files)}")

    records: list[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting", unit="row"):
        row_dict = row.to_dict()
        sn = _norm_text(row_dict.get(args.sn_col))
        sample_id = _norm_text(row_dict.get(args.sample_id_col)) if has_sample_id else ""
        row_line = _norm_text(row_dict.get(args.line_col)) or _norm_text(args.line)
        label_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tdms_path: str | None = None
        up_result = up_type = down_result = down_type = None
        error = None

        tdms_file, resolve_err = _resolve_tdms_from_row(
            row_dict,
            tdms_root=tdms_root,
            input_csv_dir=input_csv_dir,
            relative_path_col=args.relative_path_col,
            sn_col=args.sn_col,
            tdms_files=tdms_files,
        )
        if resolve_err:
            error = resolve_err
        elif tdms_file is None:
            error = "tdms not resolved"
        else:
            tdms_path = str(tdms_file)
            try:
                tdms_data = read_tdms(tdms_file, line=row_line or None)
                sampling_rate = tdms_data.get("sampling_rate")
                if sampling_rate is None:
                    raise ValueError("TDMS sampling rate is missing")

                up_result, up_type = predictor.predict(tdms_data["up_data"], sampling_rate, "up")
                down_result, down_type = predictor.predict(tdms_data["down_data"], sampling_rate, "down")
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

        # 汇总结果（SN 粒度）
        if up_result is None and down_result is None:
            overall_result = None
        elif up_result == "NOK" or down_result == "NOK":
            overall_result = "NOK"
        else:
            overall_result = "OK"

        if has_sample_id:
            # sample_view 含 sample_id 时，输出 forvia_label 长表可识别列，避免 up_/down_ 触发重复解析。
            row_result, row_type = _pick_row_label(
                sample_id=sample_id,
                up_result=up_result,
                up_type=up_type,
                down_result=down_result,
                down_type=down_type,
            )
            records.append(
                {
                    "model_tdms_path": tdms_path,
                    "model_up_result": up_result,
                    "model_up_type": up_type,
                    "model_down_result": down_result,
                    "model_down_type": down_type,
                    "result": row_result,
                    "reason_name": row_type,
                    "type": row_type,
                    "source": "model",
                    "timestamp": label_time,
                    "model_error": error,
                }
            )
        else:
            # sample_view 只有 sn 时，输出宽表列（up/down），可直接被 forvia_label 外部标签导入。
            records.append(
                {
                    "tdms_path": tdms_path,
                    "up": up_result,
                    "up_type": up_type,
                    "down": down_result,
                    "down_type": down_type,
                    "result": overall_result,
                    "source": "model",
                    "timestamp": label_time,
                    "error": error,
                }
            )

    df_result = pd.concat([df.reset_index(drop=True), pd.DataFrame.from_records(records)], axis=1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 已输出预测结果: {output_csv}")


if __name__ == "__main__":
    main()
