#!/usr/bin/env python3
"""Split epump2 relabel SNs into result-consistent and inconsistent sample views."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_manager.label_database import load_label_dataframe, load_sample_dataframe  # noqa: E402
SUMMARY_DIR = ROOT / "outputs" / "epump2_relabel_summary_20260607"
DATA_ROOT = Path("/Volumes/18555440521/fault/data_root")
METADATA_DIR = DATA_ROOT / "metadata"

OUTPUT_COLUMNS = [
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
    "sn_consistency_group",
    "sn_consistency_reason",
    "round1_sn_result",
    "round2_sn_result",
    "history_sn_result",
    "history_direction_count",
    "relabel_batch",
    "relabel_reference",
]


def norm_result(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"nok", "ng", "异常", "1"}:
        return "nok"
    if text in {"ok", "正常", "0"}:
        return "ok"
    if text in {"boundary", "边界", "不确定", "待确认", "2"}:
        return "boundary"
    return ""


def aggregate_history_result(results: pd.Series) -> str:
    values = {norm_result(value) for value in results}
    values.discard("")
    if "nok" in values:
        return "nok"
    if "boundary" in values:
        return "boundary"
    if values == {"ok"}:
        return "ok"
    return ""


def classify_sns() -> pd.DataFrame:
    raw = pd.read_csv(SUMMARY_DIR / "all_workbook_rows.csv", dtype=str).fillna("")
    valid = raw[raw["round"].isin(["1", "2"])].copy()
    valid["sn"] = valid["sn"].str.strip().str.upper()
    valid["sn_result"] = valid["result"].map(norm_result)
    rounds = valid.pivot_table(
        index="sn",
        columns="round",
        values="sn_result",
        aggfunc="last",
    ).rename(columns={"1": "round1_sn_result", "2": "round2_sn_result"})
    relabel_info = valid[valid["round"] == "2"].set_index("sn")[["batch", "reference"]].rename(
        columns={"batch": "relabel_batch", "reference": "relabel_reference"}
    )

    target_sns = set(rounds.index)
    history = load_label_dataframe(METADATA_DIR / "label_records.db").fillna("").astype(str)
    history["sn"] = history["sn"].str.strip().str.upper()
    history = history[(history["line"] == "epump2") & history["sn"].isin(target_sns)].copy()
    history["timestamp_sort"] = pd.to_datetime(history["timestamp"], errors="coerce")
    source_rank = {"expert": 3, "operator": 2, "model": 1}
    history["source_rank"] = history["source"].map(source_rank).fillna(0)
    latest = history.sort_values(["sample_id", "timestamp_sort", "source_rank"]).groupby(
        "sample_id", as_index=False
    ).tail(1)
    latest["direction"] = latest["sample_id"].str.rsplit("_", n=1).str[-1]
    latest = latest[latest["direction"].isin(["up", "down"])]
    history_sn = (
        latest.groupby("sn", as_index=True)
        .agg(
            history_sn_result=("result_key", aggregate_history_result),
            history_direction_count=("direction", "nunique"),
        )
    )

    classification = rounds.join(history_sn, how="left").join(relabel_info, how="left").reset_index()
    classification["history_direction_count"] = (
        classification["history_direction_count"].fillna(0).astype(int)
    )
    classification["rounds_same"] = (
        (classification["round1_sn_result"] != "")
        & (classification["round1_sn_result"] == classification["round2_sn_result"])
    )
    classification["history_complete"] = classification["history_direction_count"] == 2
    classification["history_same"] = (
        classification["history_complete"]
        & (classification["round2_sn_result"] == classification["history_sn_result"])
    )
    classification["sn_consistency_group"] = "inconsistent"
    classification.loc[
        classification["rounds_same"] & classification["history_same"], "sn_consistency_group"
    ] = "consistent"
    classification["sn_consistency_reason"] = "两轮结果一致，且与历史SN结果一致"
    classification.loc[~classification["rounds_same"], "sn_consistency_reason"] = "两轮结果不一致"
    classification.loc[
        classification["rounds_same"] & ~classification["history_complete"],
        "sn_consistency_reason",
    ] = "两轮结果一致，但历史缺少方向"
    classification.loc[
        classification["rounds_same"]
        & classification["history_complete"]
        & ~classification["history_same"],
        "sn_consistency_reason",
    ] = "两轮结果一致，但与历史SN结果不一致"
    return classification


def build_sample_view(classification: pd.DataFrame) -> pd.DataFrame:
    target_sns = set(classification["sn"])

    manifest = pd.read_csv(METADATA_DIR / "tdms_manifest.csv", dtype=str).fillna("")
    manifest["sn"] = manifest["sn"].str.strip().str.upper()
    manifest = manifest[(manifest["line"] == "epump2") & manifest["sn"].isin(target_sns)].copy()
    manifest["created_time_sort"] = pd.to_datetime(manifest["created_time"], errors="coerce")
    manifest = manifest.sort_values(["sn", "created_time_sort"]).groupby("sn", as_index=False).tail(1)
    manifest = manifest[
        ["line", "sn", "reference", "time", "tdms_storage_root", "relative_path"]
    ]

    sample_index = load_sample_dataframe(METADATA_DIR / "label_records.db").fillna("").astype(str)
    sample_index["sn"] = sample_index["sn"].str.strip().str.upper()
    sample_index = sample_index[
        (sample_index["line"] == "epump2") & sample_index["sn"].isin(target_sns)
    ].copy()

    history = load_label_dataframe(METADATA_DIR / "label_records.db").fillna("").astype(str)
    history["sn"] = history["sn"].str.strip().str.upper()
    history = history[(history["line"] == "epump2") & history["sn"].isin(target_sns)].copy()
    history["timestamp_sort"] = pd.to_datetime(history["timestamp"], errors="coerce")
    source_rank = {"expert": 3, "operator": 2, "model": 1}
    history["source_rank"] = history["source"].map(source_rank).fillna(0)
    latest = history.sort_values(["sample_id", "timestamp_sort", "source_rank"]).groupby(
        "sample_id", as_index=False
    ).tail(1)
    latest = latest.rename(
        columns={
            "timestamp": "label_timestamp",
            "source": "label_source",
        }
    )
    latest = latest[
        [
            "sample_id",
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
    ]

    view = sample_index.merge(manifest, on=["line", "sn"], how="left").merge(
        latest, on="sample_id", how="left"
    )
    view = view.merge(
        classification[
            [
                "sn",
                "sn_consistency_group",
                "sn_consistency_reason",
                "round1_sn_result",
                "round2_sn_result",
                "history_sn_result",
                "history_direction_count",
                "relabel_batch",
                "relabel_reference",
            ]
        ],
        on="sn",
        how="left",
    )
    view["view_name"] = view["sn_consistency_group"].map(
        {"consistent": "sn_result_consistent", "inconsistent": "sn_result_inconsistent"}
    )
    for column in OUTPUT_COLUMNS:
        if column not in view:
            view[column] = ""
    return view[OUTPUT_COLUMNS].sort_values(["sn", "sample_id"]).reset_index(drop=True)


def main() -> None:
    classification = classify_sns()
    sample_view = build_sample_view(classification)
    classification.to_csv(
        SUMMARY_DIR / "sn_result_consistency.csv", index=False, encoding="utf-8-sig"
    )
    for group in ("consistent", "inconsistent"):
        output_dir = SUMMARY_DIR / f"sn_result_{group}"
        output_dir.mkdir(parents=True, exist_ok=True)
        group_view = sample_view[sample_view["sn_consistency_group"] == group]
        group_view.to_csv(output_dir / "sample_view.csv", index=False, encoding="utf-8-sig")
        group_sns = classification[classification["sn_consistency_group"] == group]
        group_sns.to_csv(output_dir / "sn_summary.csv", index=False, encoding="utf-8-sig")
        print(f"{group}: {len(group_sns)} SN, {len(group_view)} sample rows")


if __name__ == "__main__":
    main()
