#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_manager.label_database import load_label_dataframe  # noqa: E402
from data_manager.tdms_read import read_tdms  # noqa: E402
from ml.dataset.label_filter import _build_label_row_maps, _pick_training_label  # noqa: E402
from ml.predictor import ChannelPredictor  # noqa: E402


DATA_ROOT = Path("/Volumes/18555440521/fault/data_root")
DB_PATH = DATA_ROOT / "metadata" / "label_records.db"
MODEL_ID = "epump2_general"
MODEL_DIR = REPO_ROOT / "results" / MODEL_ID
OUTPUT_DIR = MODEL_DIR / "requested_refs_prediction"
TARGET_DECISIONS = ("operator_single", "operator_conflict")
TARGET_SAMPLE_VIEW_NAME = "sample_view_operator_single_conflict_predictions.csv"
NO_LABEL_SAMPLE_VIEW_NAME = "sample_view_no_label_tdms_predictions.csv"
LABEL_FILTER_STATUSES = ("confirmed", "unconfirmed")
ALL_LABEL_STATUSES = ("unconfirmed", "pending", "confirmed", "rejected")

REFERENCES = [
    "277797801",
    "277797701",
    "180652206",
    "180652406",
    "413492601",
    "413492701",
    "238683601",
    "238683701",
    "238683801",
    "185479206",
    "185479304",
]

SAMPLE_VIEW_COLUMNS = [
    "view_name",
    "line",
    "sn",
    "sample_id",
    "group_name",
    "channel_name",
    "sampling_rate",
    "reference",
    "time",
    "tdms_storage_root",
    "relative_path",
    "tdms_path",
    "result_key",
    "result_id",
    "result_name",
    "reason_key",
    "reason_id",
    "reason_name",
    "label_version",
    "note",
    "label_timestamp",
    "label_source",
]


def _norm(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _tdms_key_from_frame(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["line"].map(_norm)
        + "|"
        + frame["sn"].map(_norm)
        + "|"
        + frame["reference"].map(_norm)
        + "|"
        + frame["relative_path"].map(_norm)
    )


def _sn_ref_key_from_frame(frame: pd.DataFrame) -> pd.Series:
    return frame["line"].map(_norm) + "|" + frame["sn"].map(_norm) + "|" + frame["reference"].map(_norm)


def _load_manifest() -> pd.DataFrame:
    manifest_path = DATA_ROOT / "metadata" / "tdms_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str, encoding="utf-8-sig").fillna("")
    relative_path = manifest["relative_path"].map(_norm)
    tdms_path = manifest["tdms_path"].map(_norm)
    path_text = (relative_path + "|" + tdms_path).str.lower()
    manifest = manifest[
        manifest["line"].map(_norm).eq("epump2")
        & manifest["reference"].map(_norm).isin(REFERENCES)
        & relative_path.ne("")
        & ~path_text.str.contains("prototype", regex=False)
    ].copy()

    if manifest.empty:
        return pd.DataFrame()

    manifest["tdms_key"] = _tdms_key_from_frame(manifest)
    manifest["sn_ref_key"] = _sn_ref_key_from_frame(manifest)
    return manifest


def _label_maps(
    statuses: tuple[str, ...],
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]]]:
    label_df = load_label_dataframe(DB_PATH, statuses=statuses)
    return _build_label_row_maps(label_df)


def _label_rows_for(
    *,
    line: str,
    sn: str,
    sample_id: str,
    by_triplet: dict[tuple[str, str, str], list[dict[str, Any]]],
    by_pair: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = by_triplet.get((line, sn, sample_id), [])
    if rows:
        return rows
    return by_pair.get((sn, sample_id), [])


def _label_debug_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    operator_rows = [row for row in rows if row.get("_source_category") == "operator"]
    expert_rows = [row for row in rows if row.get("_source_category") == "expert"]
    signatures = sorted({"|".join(str(part) for part in row.get("_label_signature") or ()) for row in operator_rows})
    return {
        "label_event_count": len(rows),
        "expert_label_count": len(expert_rows),
        "operator_label_count": len(operator_rows),
        "operator_sources": ";".join(sorted({_norm(row.get("source")) for row in operator_rows if _norm(row.get("source"))})),
        "operator_signatures": " || ".join(signatures),
    }


def _build_candidate_samples() -> pd.DataFrame:
    manifest = _load_manifest()
    if manifest.empty:
        return pd.DataFrame()

    confirmed_by_triplet, confirmed_by_pair = _label_maps(LABEL_FILTER_STATUSES)
    all_by_triplet, all_by_pair = _label_maps(ALL_LABEL_STATUSES)
    rows: list[dict[str, Any]] = []
    for row in manifest.to_dict("records"):
        for direction, group_name in (("down", "Vib Down_0"), ("up", "Vib Up_0")):
            line = _norm(row.get("line"))
            sn = _norm(row.get("sn"))
            sample_id = f"{sn}_{direction}"
            confirmed_label_rows = _label_rows_for(
                line=line,
                sn=sn,
                sample_id=sample_id,
                by_triplet=confirmed_by_triplet,
                by_pair=confirmed_by_pair,
            )
            all_label_rows = _label_rows_for(
                line=line,
                sn=sn,
                sample_id=sample_id,
                by_triplet=all_by_triplet,
                by_pair=all_by_pair,
            )
            _picked_label, decision = _pick_training_label(confirmed_label_rows)
            debug = _label_debug_fields(confirmed_label_rows)
            rows.append(
                {
                    "line": line,
                    "sn": sn,
                    "sample_id": sample_id,
                    "group_name": group_name,
                    "channel_name": "ACC",
                    "sampling_rate": "",
                    "reference": _norm(row.get("reference")),
                    "time": _norm(row.get("time")),
                    "tdms_storage_root": _norm(row.get("tdms_storage_root")),
                    "relative_path": _norm(row.get("relative_path")),
                    "tdms_path": _norm(row.get("tdms_path")),
                    "tdms_key": _norm(row.get("tdms_key")),
                    "sn_ref_key": _norm(row.get("sn_ref_key")),
                    "label_filter_decision": decision,
                    "any_label_event_count": len(all_label_rows),
                    **debug,
                }
            )
    return pd.DataFrame(rows).fillna("")


def _load_operator_single_conflict_samples(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    return candidates[candidates["label_filter_decision"].isin(TARGET_DECISIONS)].copy()


def _load_no_label_tdms_samples(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    event_counts = candidates.groupby("tdms_key")["any_label_event_count"].sum()
    no_label_keys = set(event_counts[event_counts.eq(0)].index)
    return candidates[candidates["tdms_key"].isin(no_label_keys)].copy()


def _tdms_path(row: pd.Series) -> Path:
    explicit = _norm(row.get("tdms_path"))
    if explicit:
        return Path(explicit).expanduser()
    return DATA_ROOT / _norm(row.get("tdms_storage_root")) / _norm(row.get("relative_path"))


def _direction(row: pd.Series) -> str:
    sample_id = _norm(row.get("sample_id")).lower()
    group_name = _norm(row.get("group_name")).lower()
    if sample_id.endswith("_up") or "up" in group_name:
        return "up"
    if sample_id.endswith("_down") or "down" in group_name:
        return "down"
    return ""


def _result_fields(detail: dict[str, Any]) -> dict[str, str]:
    mlabel = detail.get("mlabel")
    mtype = _norm(detail.get("mtype")) or "未知"
    result_name = _norm(detail.get("result")) or "NOK"
    if result_name == "OK":
        result_key, result_id, normalized_result_name = "ok", "0", "正常"
    else:
        result_key, result_id, normalized_result_name = "nok", "1", "异常"

    if mlabel is None:
        reason_key, reason_id = "sensor_error", "101"
    else:
        reason_key, reason_id = f"model_mlabel_{int(mlabel)}", str(1000 + int(mlabel))

    return {
        "result_key": result_key,
        "result_id": result_id,
        "result_name": normalized_result_name,
        "reason_key": reason_key,
        "reason_id": reason_id,
        "reason_name": mtype,
    }


def _prediction_note(*, detail: dict[str, Any], row: pd.Series) -> str:
    note = {
        "model_id": MODEL_ID,
        "mlabel": detail.get("mlabel"),
        "mtype": detail.get("mtype"),
        "score": detail.get("score"),
        "raw_mlabel": detail.get("raw_mlabel"),
        "raw_mtype": detail.get("raw_mtype"),
        "raw_score": detail.get("raw_score"),
        "scores": detail.get("scores"),
        "reference": _norm(row.get("reference")),
        "relative_path": _norm(row.get("relative_path")),
        "label_filter_decision": _norm(row.get("label_filter_decision")),
        "label_event_count": _norm(row.get("label_event_count")),
        "operator_label_count": _norm(row.get("operator_label_count")),
        "operator_sources": _norm(row.get("operator_sources")),
        "operator_signatures": _norm(row.get("operator_signatures")),
    }
    return json.dumps(note, ensure_ascii=False, separators=(",", ":"))


def _sample_view_row(*, row: pd.Series, detail: dict[str, Any], timestamp: str, view_name: str) -> dict[str, str]:
    out = {col: "" for col in SAMPLE_VIEW_COLUMNS}
    for col in (
        "line",
        "sn",
        "sample_id",
        "group_name",
        "channel_name",
        "sampling_rate",
        "reference",
        "time",
        "tdms_storage_root",
        "relative_path",
    ):
        out[col] = _norm(row.get(col))
    out["view_name"] = view_name
    out["tdms_path"] = str(_tdms_path(row))
    out.update(_result_fields(detail))
    out["label_version"] = f"{MODEL_ID}_prediction_v1"
    out["note"] = _prediction_note(detail=detail, row=row)
    out["label_timestamp"] = timestamp
    out["label_source"] = f"model_{MODEL_ID}"
    return out


def _write_sample_view(path: Path, rows: list[dict[str, str]]) -> None:
    df = pd.DataFrame(rows, columns=SAMPLE_VIEW_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _audit_rows(candidates: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not candidates.empty:
        tdms_event_counts = candidates.groupby("tdms_key")["any_label_event_count"].sum().to_dict()
    else:
        tdms_event_counts = {}
    for reference in REFERENCES:
        sub = candidates[candidates["reference"].map(_norm).eq(reference)] if not candidates.empty else pd.DataFrame()
        decision_counts = Counter(sub["label_filter_decision"].map(_norm)) if not sub.empty else Counter()
        target = sub[sub["label_filter_decision"].isin(TARGET_DECISIONS)] if not sub.empty else pd.DataFrame()
        no_label_tdms = {
            _norm(key)
            for key in sub["tdms_key"].unique()
            if int(tdms_event_counts.get(key, 0) or 0) == 0
        } if not sub.empty else set()
        rows.append(
            {
                "reference": reference,
                "manifest_tdms": int(sub["tdms_key"].nunique()) if not sub.empty else 0,
                "candidate_sample_rows": int(len(sub)),
                "expert_rows": int(decision_counts.get("expert", 0)),
                "operator_consistent_rows": int(decision_counts.get("operator_consistent", 0)),
                "operator_single_rows": int(decision_counts.get("operator_single", 0)),
                "operator_conflict_rows": int(decision_counts.get("operator_conflict", 0)),
                "no_manual_label_rows": int(decision_counts.get("no_manual_label", 0)),
                "target_sample_rows": int(len(target)),
                "target_tdms_count": int(target["tdms_key"].nunique()) if not target.empty else 0,
                "no_label_tdms_count": int(len(no_label_tdms)),
                "no_label_sample_rows": int(sub[sub["tdms_key"].isin(no_label_tdms)].shape[0]) if not sub.empty else 0,
            }
        )
    return rows


def _predict_samples(
    *,
    samples: pd.DataFrame,
    sample_view_name: str,
    view_name: str,
    detail_name: str,
    errors_name: str,
    summary_name: str,
) -> dict[str, Any]:
    if samples.empty:
        _write_sample_view(OUTPUT_DIR / sample_view_name, [])
        pd.DataFrame().to_csv(OUTPUT_DIR / detail_name, index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["line", "sn", "sample_id", "tdms_path", "error"]).to_csv(
            OUTPUT_DIR / errors_name, index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([{"metric": "prediction_rows", "value": 0}]).to_csv(
            OUTPUT_DIR / summary_name, index=False, encoding="utf-8-sig"
        )
        return {"sample_rows": 0, "tdms_count": 0, "prediction_rows": 0, "error_rows": 0}

    predictor = ChannelPredictor(model_dir=MODEL_DIR)
    timestamp = datetime.now().replace(microsecond=0).isoformat()
    all_rows: list[dict[str, str]] = []
    detail_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    grouped = list(samples.groupby("tdms_key", sort=True))
    for idx, (tdms_key, group) in enumerate(grouped, start=1):
        first = group.iloc[0]
        path = _tdms_path(first)
        if idx == 1 or idx % 10 == 0 or idx == len(grouped):
            print(f"[{idx}/{len(grouped)}] {first['reference']} {first['sn']} {path.name}", flush=True)
        try:
            tdms = read_tdms(path, line="epump2")
            sampling_rate = int(tdms.get("sampling_rate") or 0)
            if sampling_rate <= 0:
                raise ValueError("TDMS sampling rate is missing")
            predictions = {
                "up": predictor.predict_detail(tdms.get("up_data"), sampling_rate, "up"),
                "down": predictor.predict_detail(tdms.get("down_data"), sampling_rate, "down"),
            }
            sampling_rates = {"up": sampling_rate, "down": sampling_rate}
            direction_errors: dict[str, str] = {}
        except KeyError:
            predictions = {}
            sampling_rates = {}
            direction_errors = {}
            for direction, group_name in (("up", "Vib Up_0"), ("down", "Vib Down_0")):
                try:
                    tdms = read_tdms(path, line="epump2", required_groups=[group_name])
                    sampling_rate = int(tdms.get("sampling_rate") or 0)
                    if sampling_rate <= 0:
                        raise ValueError("TDMS sampling rate is missing")
                    signal = tdms.get(f"{direction}_data")
                    if signal is None:
                        raise ValueError(f"{direction} signal is missing")
                    predictions[direction] = predictor.predict_detail(signal, sampling_rate, direction)
                    sampling_rates[direction] = sampling_rate
                except Exception as exc:  # noqa: BLE001
                    direction_errors[direction] = f"{type(exc).__name__}: {exc}"
            group = group.copy()
            group["sampling_rate"] = str(next(iter(sampling_rates.values()), ""))
        except Exception as exc:  # noqa: BLE001
            for _, row in group.iterrows():
                errors.append(
                    {
                        "line": _norm(row.get("line")),
                        "sn": _norm(row.get("sn")),
                        "sample_id": _norm(row.get("sample_id")),
                        "tdms_path": str(path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue

        for _, row in group.iterrows():
            direction = _direction(row)
            if direction not in predictions:
                error = direction_errors.get(direction, f"unsupported sample direction: {direction or row.get('sample_id')}")
                errors.append(
                    {
                        "line": _norm(row.get("line")),
                        "sn": _norm(row.get("sn")),
                        "sample_id": _norm(row.get("sample_id")),
                        "tdms_path": str(path),
                        "error": error,
                    }
                )
                continue
            row = row.copy()
            row["sampling_rate"] = str(sampling_rates.get(direction, row.get("sampling_rate", "")))
            detail = predictions[direction]
            sample_view_row = _sample_view_row(
                row=row,
                detail=detail,
                timestamp=timestamp,
                view_name=view_name,
            )
            all_rows.append(sample_view_row)
            detail_rows.append(
                {
                    **sample_view_row,
                    "direction": direction,
                    "label_filter_decision": _norm(row.get("label_filter_decision")),
                    "any_label_event_count": int(row.get("any_label_event_count") or 0),
                    "label_event_count": int(row.get("label_event_count") or 0),
                    "expert_label_count": int(row.get("expert_label_count") or 0),
                    "operator_label_count": int(row.get("operator_label_count") or 0),
                    "operator_sources": _norm(row.get("operator_sources")),
                    "operator_signatures": _norm(row.get("operator_signatures")),
                    "model_result": detail.get("result"),
                    "model_mtype": detail.get("mtype"),
                    "model_mlabel": detail.get("mlabel"),
                    "model_score": detail.get("score"),
                    "model_raw_mtype": detail.get("raw_mtype"),
                    "model_raw_mlabel": detail.get("raw_mlabel"),
                    "model_raw_score": detail.get("raw_score"),
                }
            )

    _write_sample_view(OUTPUT_DIR / sample_view_name, all_rows)
    pd.DataFrame(detail_rows).to_csv(OUTPUT_DIR / detail_name, index=False, encoding="utf-8-sig")
    pd.DataFrame(errors, columns=["line", "sn", "sample_id", "tdms_path", "error"]).to_csv(
        OUTPUT_DIR / errors_name, index=False, encoding="utf-8-sig"
    )

    summary_rows = [
        {"metric": "requested_references", "value": len(REFERENCES)},
        {"metric": "db_sample_rows", "value": len(samples)},
        {"metric": "db_tdms_count", "value": samples["tdms_key"].nunique()},
        {"metric": "prediction_rows", "value": len(all_rows)},
        {"metric": "prediction_error_rows", "value": len(errors)},
    ]
    for key, count in Counter(samples["label_filter_decision"].map(_norm)).most_common():
        summary_rows.append({"metric": f"label_filter_decision:{key}", "value": count})
    for key, count in Counter(row["reason_name"] for row in all_rows).most_common():
        summary_rows.append({"metric": f"prediction_reason:{key}", "value": count})
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / summary_name, index=False, encoding="utf-8-sig")
    return {
        "sample_rows": len(samples),
        "tdms_count": samples["tdms_key"].nunique(),
        "prediction_rows": len(all_rows),
        "error_rows": len(errors),
        "reason_counts": dict(Counter(row["reason_name"] for row in all_rows)),
    }


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names",
        category=UserWarning,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = _build_candidate_samples()

    audit = _audit_rows(candidates)
    pd.DataFrame(audit).to_csv(OUTPUT_DIR / "requested_refs_db_audit.csv", index=False, encoding="utf-8-sig")

    no_label_samples = _load_no_label_tdms_samples(candidates)
    operator_target_samples = _load_operator_single_conflict_samples(candidates)

    no_label_result = _predict_samples(
        samples=no_label_samples,
        sample_view_name=NO_LABEL_SAMPLE_VIEW_NAME,
        view_name="epump2_general_no_label_tdms_prediction",
        detail_name="prediction_detail_no_label_tdms.csv",
        errors_name="prediction_errors_no_label_tdms.csv",
        summary_name="summary_no_label_tdms.csv",
    )
    operator_result = _predict_samples(
        samples=operator_target_samples,
        sample_view_name=TARGET_SAMPLE_VIEW_NAME,
        view_name="epump2_general_operator_single_conflict_prediction",
        detail_name="prediction_detail_operator_single_conflict.csv",
        errors_name="prediction_errors_operator_single_conflict.csv",
        summary_name="summary_operator_single_conflict.csv",
    )

    combined_rows = [
        {"metric": "no_label_sample_rows", "value": no_label_result["sample_rows"]},
        {"metric": "no_label_tdms_count", "value": no_label_result["tdms_count"]},
        {"metric": "no_label_prediction_rows", "value": no_label_result["prediction_rows"]},
        {"metric": "no_label_error_rows", "value": no_label_result["error_rows"]},
        {"metric": "operator_single_conflict_sample_rows", "value": operator_result["sample_rows"]},
        {"metric": "operator_single_conflict_tdms_count", "value": operator_result["tdms_count"]},
        {"metric": "operator_single_conflict_prediction_rows", "value": operator_result["prediction_rows"]},
        {"metric": "operator_single_conflict_error_rows", "value": operator_result["error_rows"]},
    ]
    pd.DataFrame(combined_rows).to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    print(f"[DONE] output_dir={OUTPUT_DIR}")
    print(
        "no_label:",
        f"sample_rows={no_label_result['sample_rows']}",
        f"tdms={no_label_result['tdms_count']}",
        f"predictions={no_label_result['prediction_rows']}",
        f"errors={no_label_result['error_rows']}",
    )
    print(
        "operator_single_conflict:",
        f"sample_rows={operator_result['sample_rows']}",
        f"tdms={operator_result['tdms_count']}",
        f"predictions={operator_result['prediction_rows']}",
        f"errors={operator_result['error_rows']}",
    )


if __name__ == "__main__":
    main()
