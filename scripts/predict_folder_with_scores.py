#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict TDMS files in a folder and write probabilities.")
    parser.add_argument("--folder", required=True, help="Folder containing TDMS files.")
    parser.add_argument("--model-dir", required=True, help="Model result directory containing runtime/.")
    parser.add_argument("--line", default="epump2", help="Line name for TDMS parsing.")
    parser.add_argument("--pattern", default="*.tdms", help="TDMS glob pattern.")
    parser.add_argument("--output", default="", help="Output CSV path.")
    return parser.parse_args()


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _safe_col(text: object) -> str:
    raw = str(text or "").strip()
    return re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", raw).strip("_") or "unknown"


def _merge_result(up: str | None, down: str | None) -> str | None:
    if up is None and down is None:
        return None
    if up == "NOK" or down == "NOK":
        return "NOK"
    return "OK"


def _channel_scores(predictor, predict_with_threshold_rule, channel_raw: np.ndarray, sr: int) -> dict[str, object]:
    if channel_raw is None:
        return {
            "result": "NOK",
            "type": "传感器错误",
            "mlabel": "",
            "score": "",
            "raw_best_mlabel": "",
            "raw_best_type": "",
            "raw_best_score": "",
        }

    if len(channel_raw) > sr * predictor.time_thr:
        return {
            "result": "NOK",
            "type": "传感器错误",
            "mlabel": "",
            "score": "",
            "raw_best_mlabel": "",
            "raw_best_type": "",
            "raw_best_score": "",
        }

    if float(np.mean(channel_raw)) > float(predictor.mean_thr):
        return {
            "result": "NOK",
            "type": "传感器错误",
            "mlabel": "",
            "score": "",
            "raw_best_mlabel": "",
            "raw_best_type": "",
            "raw_best_score": "",
        }

    features = predictor.extract_features(channel_raw, sr)
    x_features = predictor._build_feature_vector(features)
    proba = np.asarray(predictor.model.predict_proba(x_features)[0], dtype=float)
    classes = np.asarray(predictor.model.classes_, dtype=int)
    final_mlabel = int(
        predict_with_threshold_rule(
            proba.reshape(1, -1),
            classes,
            predictor.mlabel_threshold_dict,
            ok_mlabel=predictor.ok_mlabel,
        )[0]
    )
    class_to_idx = {int(label): idx for idx, label in enumerate(classes)}
    final_idx = class_to_idx.get(final_mlabel)
    raw_best_idx = int(np.argmax(proba))
    raw_best_mlabel = int(classes[raw_best_idx])
    final_score = float(proba[final_idx]) if final_idx is not None else float("nan")
    raw_best_score = float(proba[raw_best_idx])

    out: dict[str, object] = {
        "result": "OK" if final_mlabel == int(predictor.ok_mlabel) else "NOK",
        "type": predictor.mlabel_mtype_dict.get(final_mlabel, str(final_mlabel)),
        "mlabel": final_mlabel,
        "score": round(final_score, 6),
        "raw_best_mlabel": raw_best_mlabel,
        "raw_best_type": predictor.mlabel_mtype_dict.get(raw_best_mlabel, str(raw_best_mlabel)),
        "raw_best_score": round(raw_best_score, 6),
    }
    for label, probability in zip(classes, proba):
        out[f"proba_{int(label)}"] = round(float(probability), 6)
    return out


def _prefix(prefix: str, payload: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in payload.items()}


def main() -> None:
    args = _parse_args()
    folder = Path(args.folder).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_csv = Path(args.output).expanduser().resolve() if args.output else folder / f"{model_dir.name}_predictions_with_scores.csv"

    runtime_dir = model_dir / "runtime"
    sys.path.insert(0, str(runtime_dir))

    from ChannelPredictor import ChannelPredictor  # noqa: E402
    from prediction_logic import predict_with_threshold_rule  # noqa: E402
    from tdms_read import read_tdms  # noqa: E402

    predictor = ChannelPredictor(model_dir=model_dir)
    label_columns = {
        f"label_{int(label)}": predictor.mlabel_mtype_dict.get(int(label), str(label))
        for label in getattr(predictor.model, "classes_", [])
    }

    records: list[dict[str, object]] = []
    for tdms_file in sorted(folder.rglob(args.pattern)):
        if tdms_file.suffix.lower() != ".tdms" or _is_hidden(tdms_file):
            continue

        record: dict[str, object] = {
            "filename": tdms_file.name,
            "tdms_path": str(tdms_file),
            "source": "model",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **label_columns,
        }
        try:
            tdms_data = read_tdms(tdms_file, line=args.line)
            sr = tdms_data.get("sampling_rate")
            if sr is None:
                raise ValueError("TDMS sampling rate is missing")
            sr = int(sr)

            up_payload = _channel_scores(predictor, predict_with_threshold_rule, tdms_data["up_data"], sr)
            down_payload = _channel_scores(predictor, predict_with_threshold_rule, tdms_data["down_data"], sr)
            record.update(
                {
                    "sn": tdms_data.get("sn"),
                    "reference": tdms_data.get("reference"),
                    "time": tdms_data.get("time"),
                    "sampling_rate": sr,
                    **_prefix("up", up_payload),
                    **_prefix("down", down_payload),
                    "result": _merge_result(str(up_payload.get("result")), str(down_payload.get("result"))),
                    "error": "",
                }
            )
            print(
                f"{tdms_file.name} | "
                f"Up={up_payload.get('result')},{up_payload.get('type')},score={up_payload.get('score')} | "
                f"Down={down_payload.get('result')},{down_payload.get('type')},score={down_payload.get('score')}"
            )
        except Exception as exc:  # noqa: BLE001
            record.update(
                {
                    "sn": "",
                    "reference": "",
                    "time": "",
                    "sampling_rate": "",
                    "up_result": "ERROR",
                    "down_result": "ERROR",
                    "result": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"{tdms_file.name} | ERROR={record['error']}")
        records.append(record)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print("-" * 60)
    print(f"已保存 {len(records)} 条带分数预测记录到: {output_csv}")


if __name__ == "__main__":
    main()
