#!/usr/bin/env python3
"""Append every final-round epump2 relabel to label_records.db."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_manager.label_database import append_confirmed_labels, load_label_dataframe  # noqa: E402
SAMPLE_LABELS = ROOT / "outputs" / "epump2_relabel_summary_20260607" / "sample_labels.csv"
DEFAULT_HISTORY = Path("/Volumes/18555440521/fault/data_root/metadata/label_records.db")
AUDIT_DIR = ROOT / "outputs" / "epump2_relabel_summary_20260607" / "all_label_history_update"
ALL_MARKER = "epump2_relabel_final_round_all_20260607"
STRONG_MARKER = "epump2_relabel_strong_final_round_20260607"
HISTORY_COLUMNS = [
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


def iso_label_time(value: str) -> str:
    return datetime.strptime(value, "%Y%m%d_%H%M%S").strftime("%Y-%m-%dT%H:%M:%S")


def severity_and_confidence(raw_type: str) -> tuple[str, str]:
    if "强烈" in raw_type:
        return "强烈", "0.9"
    if "中等" in raw_type:
        return "中等", "0.8"
    if "轻微" in raw_type:
        return "轻微", "0.6"
    return "默认", "0.9"


def build_rows() -> pd.DataFrame:
    labels = pd.read_csv(SAMPLE_LABELS, dtype=str).fillna("")
    final = labels[labels["round"] == "2"].copy()
    severity_confidence = final["raw_type"].map(severity_and_confidence)
    final["severity"] = severity_confidence.map(lambda value: value[0])
    final["confidence"] = severity_confidence.map(lambda value: value[1])
    result_ids = {"ok": "0", "nok": "1", "boundary": "2"}
    result_names = {"ok": "正常", "nok": "异常", "boundary": "边界"}
    rows = pd.DataFrame(
        {
            "line": "epump2",
            "sn": final["sn"],
            "sample_id": final["sample_id"],
            "timestamp": final["label_time"].map(iso_label_time),
            "source": "operator_" + final["operator_id"],
            "result_key": final["result_key"],
            "result_id": final["result_key"].map(result_ids),
            "result_name": final["result_key"].map(result_names),
            "reason_key": final["reason_key"],
            "reason_id": final["reason_id"],
            "reason_name": final["reason_name"],
            "reason_confidence": final["confidence"],
            "label_version": "v1.1",
            "note": final.apply(
                lambda row: (
                    f"import_marker={ALL_MARKER} | relabel_round=2"
                    f" | severity={row['severity']} | raw_type={row['raw_type'] or '<empty>'}"
                    f" | confidence_rule={row['severity']}->{row['confidence']}"
                    f" | reference={row['reference']} | batch={row['batch']}"
                    f" | source_file={row['source_file']} | source_row={row['row_number']}"
                ),
                axis=1,
            ),
            "severity": final["severity"],
        }
    )
    return rows.sort_values(["timestamp", "sample_id"]).reset_index(drop=True)


def existing_registration_status(history: Path) -> tuple[set[str], set[str]]:
    all_marked: set[str] = set()
    strong_marked: set[str] = set()
    chunk = load_label_dataframe(history).fillna("").astype(str)
    notes = chunk["note"].fillna("")
    all_marked.update(chunk.loc[notes.str.contains(ALL_MARKER, regex=False), "sample_id"].dropna())
    strong_marked.update(chunk.loc[notes.str.contains(STRONG_MARKER, regex=False), "sample_id"].dropna())
    return all_marked, strong_marked


def append_atomically(history: Path, rows: pd.DataFrame, backup: Path) -> None:
    del backup
    append_confirmed_labels(history, rows[HISTORY_COLUMNS].to_dict("records"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    all_marked, strong_marked = existing_registration_status(args.history)
    rows["registration_status"] = "pending"
    rows.loc[rows["sample_id"].isin(all_marked), "registration_status"] = "existing_all_marker"
    rows.loc[
        (rows["severity"] == "强烈")
        & rows["sample_id"].isin(strong_marked)
        & (rows["registration_status"] == "pending"),
        "registration_status",
    ] = "existing_strong_marker"
    pending = rows[rows["registration_status"] == "pending"].copy()

    rows.to_csv(AUDIT_DIR / "final_round_all_audit.csv", index=False, encoding="utf-8-sig")
    pending.to_csv(AUDIT_DIR / "final_round_pending.csv", index=False, encoding="utf-8-sig")
    print(f"final-round rows: {len(rows)}")
    print(rows["registration_status"].value_counts().to_string())
    print("confidence distribution:")
    print(rows.groupby(["severity", "reason_confidence"]).size().to_string())
    if not args.execute:
        print("dry-run only; pass --execute to update label_records.db")
        return
    if pending.empty:
        print("no rows appended")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.history.with_name(f"label_history_before_all_relabels_{stamp}.csv")
    append_atomically(args.history, pending, backup)
    print(f"backup: {backup}")
    print(f"appended: {len(pending)}")


if __name__ == "__main__":
    main()
