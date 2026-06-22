#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_manager.label_database import load_label_dataframe, load_sample_dataframe  # noqa: E402


DATA_ROOT = Path("/Volumes/18555440521/fault/data_root")
MODEL_ID = "epump3_general_20260506_xgb"
MODEL_DIR = Path(__file__).resolve().parents[1] / "results" / MODEL_ID
OUTPUT_DIR = MODEL_DIR / "epump3_all_model_history_compare"
MODEL_LABEL_CSV = OUTPUT_DIR / "epump3_all_model_label_history.csv"
ERROR_CSV = OUTPUT_DIR / "epump3_all_model_prediction_errors.csv"
MISMATCH_CSV = OUTPUT_DIR / "epump3_model_vs_history_consistent_mismatches.csv"
COMPARE_DETAIL_CSV = OUTPUT_DIR / "epump3_model_vs_history_consensus_detail.csv"
COMPARE_SUMMARY_CSV = OUTPUT_DIR / "epump3_model_vs_history_compare_summary.csv"
MISMATCH_SPLIT_DIR = OUTPUT_DIR / "mismatch_by_model_reason"

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
    0: ("model_normal_noise_boundary", "1000", "正常_干扰_边界"),
    1: ("model_friction_acc_noise_gear", "1001", "摩擦_摩擦acc_杂音_咬齿"),
    2: ("model_chatter_mada", "1002", "震颤_马达"),
    3: ("sensor_error", "101", "传感器错误"),
    4: ("tick_tock", "102", "秒表"),
}

HISTORY_TO_MODEL_CATEGORY = {
    "正常": "正常_干扰_边界",
    "干扰": "正常_干扰_边界",
    "边界": "正常_干扰_边界",
    "摩擦": "摩擦_摩擦acc_杂音_咬齿",
    "摩擦acc": "摩擦_摩擦acc_杂音_咬齿",
    "杂音": "摩擦_摩擦acc_杂音_咬齿",
    "咬齿": "摩擦_摩擦acc_杂音_咬齿",
    "哒哒_咔咔": "摩擦_摩擦acc_杂音_咬齿",
    "震颤": "震颤_马达",
    "马达": "震颤_马达",
    "传感器错误": "传感器错误",
    "秒表": "秒表",
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


def _group_for_direction(direction: str) -> str:
    if direction == "up":
        return "Vib Up_0"
    if direction == "down":
        return "Vib Down_0"
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
        mlabel = 3
        return _rule_prediction(predictor, mlabel, "missing_channel")
    if len(channel_raw) > sr * predictor.time_thr:
        mlabel = 3
        return _rule_prediction(predictor, mlabel, "length_gt_time_threshold")
    if float(np.mean(channel_raw)) > predictor.mean_thr:
        mlabel = 3
        return _rule_prediction(predictor, mlabel, "mean_gt_threshold")

    features = predictor.extract_features(channel_raw, sr)
    x_features = np.fromiter(features.values(), dtype=float).reshape(1, -1)
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
    return {
        "result": "OK" if mlabel == predictor.ok_mlabel else "NOK",
        "mlabel": mlabel,
        "mtype": predictor.mlabel_mtype_dict[mlabel],
        "confidence": conf,
        "rule": "model",
        "proba": {str(int(cls)): round(float(proba[idx]), 6) for idx, cls in enumerate(classes)},
    }


def _rule_prediction(predictor, mlabel: int, rule: str) -> dict[str, object]:
    return {
        "result": "OK" if mlabel == predictor.ok_mlabel else "NOK",
        "mlabel": mlabel,
        "mtype": predictor.mlabel_mtype_dict[mlabel],
        "confidence": 1.0,
        "rule": rule,
        "proba": {},
    }


def _model_label_row(*, sample_row: dict, manifest_row: dict, prediction: dict, timestamp: str) -> dict[str, str]:
    mlabel = int(prediction["mlabel"])
    reason_key, reason_id, reason_name = MODEL_REASON_MAP.get(
        mlabel,
        (f"model_mlabel_{mlabel}", str(1000 + mlabel), _norm(prediction.get("mtype")) or f"mlabel_{mlabel}"),
    )
    result = _norm(prediction["result"])
    if result == "OK":
        result_key, result_id, result_name = "ok", "0", "正常"
    else:
        result_key, result_id, result_name = "nok", "1", "异常"

    note_parts = [
        f"model_id={MODEL_ID}",
        f"model_mlabel={mlabel}",
        f"model_mtype={_norm(prediction.get('mtype'))}",
        f"model_rule={_norm(prediction.get('rule'))}",
        f"reference={_norm(manifest_row.get('reference'))}",
        f"time={_norm(manifest_row.get('time'))}",
        f"relative_path={_norm(manifest_row.get('relative_path'))}",
    ]
    proba = prediction.get("proba") or {}
    if proba:
        note_parts.append("model_proba=" + ",".join(f"{k}:{v}" for k, v in sorted(proba.items())))

    return {
        "line": "epump3",
        "sn": _norm(sample_row.get("sn")),
        "sample_id": _norm(sample_row.get("sample_id")),
        "timestamp": timestamp,
        "source": "model",
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


def _build_history_consensus(label_history: pd.DataFrame) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    consensus: dict[str, dict[str, str]] = {}
    detail_rows: list[dict[str, str]] = []

    for sample_id, group in label_history.groupby(label_history["sample_id"].map(_norm), sort=True):
        if not sample_id:
            continue
        rows = group.to_dict("records")
        expert_rows = [r for r in rows if _norm(r.get("source")) == "expert"]
        if expert_rows:
            chosen = expert_rows
            priority = "expert"
        else:
            chosen = [r for r in rows if _norm(r.get("source")).startswith("operator")]
            priority = "operator"

        if not chosen:
            status = "no_expert_or_operator"
            selected_reason = ""
            selected_category = ""
        else:
            reasons = {_norm(r.get("reason_name")) for r in chosen}
            reasons.discard("")
            if len(reasons) == 1:
                selected_reason = next(iter(reasons))
                selected_category = HISTORY_TO_MODEL_CATEGORY.get(selected_reason, selected_reason)
                status = "consistent"
                first = chosen[0]
                consensus[sample_id] = {
                    "history_priority": priority,
                    "history_reason_name": selected_reason,
                    "history_model_category": selected_category,
                    "history_result_name": _norm(first.get("result_name")),
                    "history_sources": ";".join(sorted({_norm(r.get("source")) for r in chosen})),
                    "history_rows_used": str(len(chosen)),
                    "history_total_rows": str(len(rows)),
                }
            else:
                selected_reason = ";".join(sorted(reasons))
                selected_category = ""
                status = "conflict"

        detail_rows.append(
            {
                "sample_id": sample_id,
                "history_status": status,
                "history_priority": priority if chosen else "",
                "history_reason_name": selected_reason,
                "history_model_category": selected_category,
                "history_sources": ";".join(sorted({_norm(r.get("source")) for r in chosen})) if chosen else "",
                "history_rows_used": str(len(chosen)),
                "history_total_rows": str(len(rows)),
            }
        )

    return consensus, detail_rows


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    return safe.strip("_") or "unknown"


def main() -> None:
    metadata = DATA_ROOT / "metadata"
    sample_index = load_sample_dataframe(metadata / "label_records.db").fillna("").astype(str)
    sample_index = sample_index[sample_index["line"].map(_norm).eq("epump3")].copy()

    manifest = pd.read_csv(metadata / "tdms_manifest.csv", encoding="utf-8-sig", dtype=str)
    manifest = manifest[manifest["line"].map(_norm).eq("epump3")].copy()
    manifest_by_sn = {row["sn"]: row for row in manifest.to_dict("records")}

    label_history = load_label_dataframe(metadata / "label_records.db").fillna("").astype(str)
    label_history = label_history[label_history["line"].map(_norm).eq("epump3")].copy()
    history_consensus, history_detail = _build_history_consensus(label_history)

    sample_rows_by_sn: dict[str, list[dict]] = {
        sn: group.to_dict("records")
        for sn, group in sample_index.groupby(sample_index["sn"].map(_norm), sort=True)
    }
    target_sns = sorted(set(manifest_by_sn))
    for sn in target_sns:
        if sn not in sample_rows_by_sn:
            sample_rows_by_sn[sn] = [
                {"line": "epump3", "sn": sn, "sample_id": f"{sn}_up", "group_name": "Vib Up_0"},
                {"line": "epump3", "sn": sn, "sample_id": f"{sn}_down", "group_name": "Vib Down_0"},
            ]

    predictor, predict_with_threshold_rule, read_tdms = _load_predictor()
    timestamp = datetime.now().replace(microsecond=0).isoformat()

    model_rows: list[dict[str, str]] = []
    mismatch_rows: list[dict[str, str]] = []
    compare_rows: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    for idx, sn in enumerate(target_sns, start=1):
        manifest_row = manifest_by_sn[sn]
        tdms_path = DATA_ROOT / _norm(manifest_row["tdms_storage_root"]) / _norm(manifest_row["relative_path"])
        samples = sample_rows_by_sn[sn]
        required_groups = [_norm(r.get("group_name")) for r in samples if _norm(r.get("group_name"))]
        if idx == 1 or idx % 100 == 0 or idx == len(target_sns):
            print(f"[{idx}/{len(target_sns)}] {sn} {tdms_path.name}", flush=True)

        try:
            tdms_data = read_tdms(tdms_path, line="epump3", required_groups=required_groups)
            sr = tdms_data.get("sampling_rate")
            if sr is None:
                raise ValueError("TDMS sampling rate is missing")
            predictions: dict[str, dict[str, object]] = {}
            if tdms_data.get("up_data") is not None:
                predictions["up"] = _predict_channel(predictor, predict_with_threshold_rule, tdms_data.get("up_data"), int(sr))
            if tdms_data.get("down_data") is not None:
                predictions["down"] = _predict_channel(predictor, predict_with_threshold_rule, tdms_data.get("down_data"), int(sr))
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "line": "epump3",
                    "sn": sn,
                    "sample_id": "",
                    "tdms_path": str(tdms_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        for sample_row in samples:
            sample_id = _norm(sample_row.get("sample_id"))
            direction = _direction(sample_id)
            if direction not in predictions:
                errors.append(
                    {
                        "line": "epump3",
                        "sn": sn,
                        "sample_id": sample_id,
                        "tdms_path": str(tdms_path),
                        "error": f"prediction missing for direction: {direction or sample_id}",
                    }
                )
                continue

            row = _model_label_row(
                sample_row=sample_row,
                manifest_row=manifest_row,
                prediction=predictions[direction],
                timestamp=timestamp,
            )
            model_rows.append(row)

            hist = history_consensus.get(sample_id)
            if hist is None:
                compare_status = "no_consistent_history"
                history_model_category = ""
            else:
                history_model_category = hist["history_model_category"]
                compare_status = "same" if row["reason_name"] == history_model_category else "different"

            compare_row = {
                **row,
                "model_reason_name": row["reason_name"],
                "history_status": "consistent" if hist else "no_consistent_history",
                "history_priority": hist["history_priority"] if hist else "",
                "history_reason_name": hist["history_reason_name"] if hist else "",
                "history_model_category": history_model_category,
                "history_result_name": hist["history_result_name"] if hist else "",
                "history_sources": hist["history_sources"] if hist else "",
                "history_rows_used": hist["history_rows_used"] if hist else "",
                "history_total_rows": hist["history_total_rows"] if hist else "",
                "compare_status": compare_status,
            }
            compare_rows.append(compare_row)
            if compare_status == "different":
                mismatch_rows.append(compare_row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MISMATCH_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    with MODEL_LABEL_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(model_rows)

    compare_fields = LABEL_HISTORY_COLUMNS + [
        "model_reason_name",
        "history_status",
        "history_priority",
        "history_reason_name",
        "history_model_category",
        "history_result_name",
        "history_sources",
        "history_rows_used",
        "history_total_rows",
        "compare_status",
    ]
    with COMPARE_DETAIL_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=compare_fields)
        writer.writeheader()
        writer.writerows(compare_rows)

    with MISMATCH_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=compare_fields)
        writer.writeheader()
        writer.writerows(mismatch_rows)

    by_reason: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in mismatch_rows:
        by_reason[row["model_reason_name"]].append(row)
    for reason, rows in sorted(by_reason.items()):
        out = MISMATCH_SPLIT_DIR / f"mismatch_pred_{_safe_filename(reason)}.csv"
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=compare_fields)
            writer.writeheader()
            writer.writerows(rows)

    with ERROR_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["line", "sn", "sample_id", "tdms_path", "error"])
        writer.writeheader()
        writer.writerows(errors)

    history_detail_path = OUTPUT_DIR / "epump3_history_consensus_detail.csv"
    with history_detail_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "sample_id",
            "history_status",
            "history_priority",
            "history_reason_name",
            "history_model_category",
            "history_sources",
            "history_rows_used",
            "history_total_rows",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history_detail)

    summary_rows = [
        {"metric": "manifest_sn", "value": len(target_sns)},
        {"metric": "sample_index_rows", "value": len(sample_index)},
        {"metric": "model_label_rows", "value": len(model_rows)},
        {"metric": "prediction_error_rows", "value": len(errors)},
        {"metric": "history_consistent_samples", "value": len(history_consensus)},
        {"metric": "compare_rows", "value": len(compare_rows)},
        {"metric": "mismatch_rows", "value": len(mismatch_rows)},
    ]
    for key, value in Counter(row["reason_name"] for row in model_rows).most_common():
        summary_rows.append({"metric": f"model_reason:{key}", "value": value})
    for key, value in Counter(row["model_reason_name"] for row in mismatch_rows).most_common():
        summary_rows.append({"metric": f"mismatch_by_model_reason:{key}", "value": value})
    for key, value in Counter(row["compare_status"] for row in compare_rows).most_common():
        summary_rows.append({"metric": f"compare_status:{key}", "value": value})
    with COMPARE_SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("OUTPUT_DIR", OUTPUT_DIR)
    print("MODEL_LABEL_CSV", MODEL_LABEL_CSV)
    print("MISMATCH_CSV", MISMATCH_CSV)
    print("MISMATCH_SPLIT_DIR", MISMATCH_SPLIT_DIR)
    print("ERROR_CSV", ERROR_CSV)
    print("manifest_sn", len(target_sns))
    print("sample_index_rows", len(sample_index))
    print("model_label_rows", len(model_rows))
    print("errors", len(errors))
    print("history_consistent_samples", len(history_consensus))
    print("compare_status", Counter(row["compare_status"] for row in compare_rows))
    print("model_reason_counts", Counter(row["reason_name"] for row in model_rows))
    print("mismatch_by_model_reason", Counter(row["model_reason_name"] for row in mismatch_rows))


if __name__ == "__main__":
    main()
