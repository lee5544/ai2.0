#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_manager.label_database import load_label_dataframe, load_sample_dataframe  # noqa: E402


DATA_ROOT = Path("/Volumes/18555440521/fault/data_root")
MODEL_ID = "epump2_general_20260509_top100_xgb"
MODEL_DIR = Path(__file__).resolve().parents[1] / "results" / MODEL_ID
OUTPUT_CSV = MODEL_DIR / "epump2_unlabeled_sn_predictions_label_history.csv"
ERROR_CSV = MODEL_DIR / "epump2_unlabeled_sn_prediction_errors.csv"

LABEL_HISTORY_COLUMNS = [
    "line",
    "sn",
    "sample_id",
    "timestamp",
    "source",
    "result_key",
    "result_id",
    "result_name",
    "reason_key",
    "reason_id",
    "reason_name",
    "reason_confidence",
    "label_version",
    "note",
]

MODEL_REASON_MAP = {
    "正常_干扰_边界": ("clean_normal", "0", "正常"),
    "杂音_咬齿_哒哒_咔咔": ("model_noise_gear_dada", "1001", "杂音_咬齿_哒哒_咔咔"),
    "震颤_马达": ("model_chatter_mada", "1002", "震颤_马达"),
    "传感器错误": ("sensor_error", "101", "传感器错误"),
    "秒表": ("tick_tock", "102", "秒表"),
    "摩擦acc": ("friction_acc", "104", "摩擦acc"),
}


def _norm(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _direction(sample_id: str) -> str:
    sid = sample_id.lower()
    if sid.endswith("_up"):
        return "up"
    if sid.endswith("_down"):
        return "down"
    return ""


def _load_predictor():
    runtime_dir = MODEL_DIR / "runtime"
    sys.path.insert(0, str(runtime_dir))
    from ChannelPredictor import ChannelPredictor
    from prediction_logic import predict_with_threshold_rule
    from tdms_read import read_tdms

    return ChannelPredictor(model_dir=MODEL_DIR), predict_with_threshold_rule, read_tdms


def _predict_channel(predictor, predict_with_threshold_rule, channel_raw, sr: int) -> dict[str, object]:
    if channel_raw is None:
        return {
            "result": "NOK",
            "mlabel": 3,
            "mtype": "传感器错误",
            "confidence": 1.0,
            "rule": "missing_channel",
            "proba": {},
        }
    if len(channel_raw) > sr * predictor.time_thr:
        return {
            "result": "NOK",
            "mlabel": 3,
            "mtype": "传感器错误",
            "confidence": 1.0,
            "rule": "length_gt_time_threshold",
            "proba": {},
        }
    if float(np.mean(channel_raw)) > predictor.mean_thr:
        return {
            "result": "NOK",
            "mlabel": 3,
            "mtype": "传感器错误",
            "confidence": 1.0,
            "rule": "mean_gt_threshold",
            "proba": {},
        }

    features = predictor.extract_features(channel_raw, sr)
    x_features = predictor._build_feature_vector(features)
    proba = np.asarray(predictor.model.predict_proba(x_features)[0], dtype=float)
    classes = np.asarray(predictor.model.classes_, dtype=int)
    mlabel = int(
        predict_with_threshold_rule(
            proba.reshape(1, -1),
            classes,
            predictor.mlabel_threshold_dict,
            ok_mlabel=predictor.ok_mlabel,
        )[0]
    )
    class_to_idx = {int(label): idx for idx, label in enumerate(classes)}
    conf = float(proba[class_to_idx[mlabel]]) if mlabel in class_to_idx else float(np.max(proba))
    mtype = predictor.mlabel_mtype_dict[mlabel]
    return {
        "result": "OK" if mlabel == predictor.ok_mlabel else "NOK",
        "mlabel": mlabel,
        "mtype": mtype,
        "confidence": conf,
        "rule": "model",
        "proba": {str(int(cls)): round(float(proba[idx]), 6) for idx, cls in enumerate(classes)},
    }


def _label_row(*, sample_row: dict, manifest_row: dict, prediction: dict, timestamp: str) -> dict[str, str]:
    result = _norm(prediction["result"])
    raw_mtype = _norm(prediction["mtype"])
    if result == "OK":
        result_key, result_id, result_name = "ok", "0", "正常"
        reason_key, reason_id, reason_name = MODEL_REASON_MAP["正常_干扰_边界"]
    else:
        result_key, result_id, result_name = "nok", "1", "异常"
        reason_key, reason_id, reason_name = MODEL_REASON_MAP.get(
            raw_mtype,
            ("model_other", "1099", raw_mtype or "模型未知"),
        )

    note_parts = [
        f"model_id={MODEL_ID}",
        f"model_mlabel={prediction.get('mlabel')}",
        f"model_mtype={raw_mtype}",
        f"model_rule={prediction.get('rule')}",
        f"reference={_norm(manifest_row.get('reference'))}",
        f"time={_norm(manifest_row.get('time'))}",
        f"relative_path={_norm(manifest_row.get('relative_path'))}",
    ]
    proba = prediction.get("proba") or {}
    if proba:
        note_parts.append("model_proba=" + ",".join(f"{k}:{v}" for k, v in sorted(proba.items())))

    return {
        "line": "epump2",
        "sn": _norm(sample_row.get("sn")),
        "sample_id": _norm(sample_row.get("sample_id")),
        "timestamp": timestamp,
        "source": MODEL_ID,
        "result_key": result_key,
        "result_id": result_id,
        "result_name": result_name,
        "reason_key": reason_key,
        "reason_id": reason_id,
        "reason_name": reason_name,
        "reason_confidence": f"{float(prediction.get('confidence') or 0.0):.6f}",
        "label_version": "model_prediction_v1",
        "note": " | ".join(note_parts),
    }


def main() -> None:
    metadata = DATA_ROOT / "metadata"
    label_records_db_path = metadata / "label_records.db"
    manifest_path = metadata / "tdms_manifest.csv"

    sample_index = load_sample_dataframe(label_records_db_path).fillna("").astype(str)
    sample_index = sample_index[sample_index["line"].map(_norm).eq("epump2")].copy()

    label_history = load_label_dataframe(label_records_db_path).fillna("").astype(str)
    labeled_sns = set(
        label_history[label_history["line"].map(_norm).eq("epump2")]["sn"].map(_norm)
    )

    target_samples = sample_index[~sample_index["sn"].map(_norm).isin(labeled_sns)].copy()
    target_sns = sorted(set(target_samples["sn"].map(_norm)) - {""})

    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype=str)
    manifest = manifest[
        manifest["line"].map(_norm).eq("epump2") & manifest["sn"].map(_norm).isin(target_sns)
    ].copy()
    manifest_by_sn = {row["sn"]: row for row in manifest.to_dict("records")}

    missing_manifest = [sn for sn in target_sns if sn not in manifest_by_sn]
    if missing_manifest:
        raise RuntimeError(f"{len(missing_manifest)} target SN missing in tdms_manifest, e.g. {missing_manifest[:10]}")

    predictor, predict_with_threshold_rule, read_tdms = _load_predictor()
    timestamp = datetime.now().replace(microsecond=0).isoformat()

    rows: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    samples_by_sn = {
        sn: group.to_dict("records")
        for sn, group in target_samples.groupby(target_samples["sn"].map(_norm), sort=True)
    }

    for idx, sn in enumerate(target_sns, start=1):
        manifest_row = manifest_by_sn[sn]
        tdms_path = DATA_ROOT / _norm(manifest_row["tdms_storage_root"]) / _norm(manifest_row["relative_path"])
        if idx == 1 or idx % 50 == 0 or idx == len(target_sns):
            print(f"[{idx}/{len(target_sns)}] {sn} {tdms_path.name}", flush=True)

        try:
            required_groups = [
                _norm(sample_row.get("group_name"))
                for sample_row in samples_by_sn.get(sn, [])
                if _norm(sample_row.get("group_name"))
            ]
            tdms_data = read_tdms(tdms_path, line="epump2", required_groups=required_groups)
            sr = tdms_data.get("sampling_rate")
            if sr is None:
                raise ValueError("TDMS sampling rate is missing")
            predictions = {
                "up": _predict_channel(predictor, predict_with_threshold_rule, tdms_data.get("up_data"), int(sr)),
                "down": _predict_channel(predictor, predict_with_threshold_rule, tdms_data.get("down_data"), int(sr)),
            }
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "line": "epump2",
                    "sn": sn,
                    "tdms_path": str(tdms_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        for sample_row in samples_by_sn.get(sn, []):
            direction = _direction(_norm(sample_row.get("sample_id")))
            if direction not in predictions:
                errors.append(
                    {
                        "line": "epump2",
                        "sn": sn,
                        "sample_id": _norm(sample_row.get("sample_id")),
                        "tdms_path": str(tdms_path),
                        "error": f"unknown sample direction: {sample_row.get('sample_id')}",
                    }
                )
                continue
            rows.append(
                _label_row(
                    sample_row=sample_row,
                    manifest_row=manifest_row,
                    prediction=predictions[direction],
                    timestamp=timestamp,
                )
            )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with ERROR_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["line", "sn", "sample_id", "tdms_path", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)

    print("OUTPUT", OUTPUT_CSV)
    print("ERRORS", ERROR_CSV)
    print("target_sns", len(target_sns))
    print("target_samples", len(target_samples))
    print("prediction_rows", len(rows))
    print("errors", len(errors))
    print("result_counts", Counter(row["result_name"] for row in rows))
    print("reason_counts", Counter(row["reason_name"] for row in rows))


if __name__ == "__main__":
    main()
