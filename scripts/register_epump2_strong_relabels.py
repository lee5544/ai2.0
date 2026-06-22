#!/usr/bin/env python3
"""Append final-round epump2 relabels containing 强烈 to label_records.db."""

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
AUDIT_DIR = ROOT / "outputs" / "epump2_relabel_summary_20260607" / "strong_label_history_update"
MARKER = "epump2_relabel_strong_final_round_20260607"
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


def build_rows() -> pd.DataFrame:
    labels = pd.read_csv(SAMPLE_LABELS, dtype=str).fillna("")
    strong = labels[
        (labels["round"] == "2") & labels["raw_type"].str.contains("强烈", na=False)
    ].copy()
    rows = pd.DataFrame(
        {
            "line": "epump2",
            "sn": strong["sn"],
            "sample_id": strong["sample_id"],
            "timestamp": strong["label_time"].map(iso_label_time),
            "source": "operator_" + strong["operator_id"],
            "result_key": strong["result_key"],
            "result_id": "1",
            "result_name": "异常",
            "reason_key": strong["reason_key"],
            "reason_id": strong["reason_id"],
            "reason_name": strong["reason_name"],
            "reason_confidence": "0.9",
            "label_version": "v1.1",
            "note": strong.apply(
                lambda row: (
                    f"import_marker={MARKER} | relabel_round=2 | severity=强烈"
                    f" | raw_type={row['raw_type']} | reference={row['reference']}"
                    f" | batch={row['batch']} | source_file={row['source_file']}"
                    f" | source_row={row['row_number']}"
                ),
                axis=1,
            ),
        }
    )
    return rows[HISTORY_COLUMNS].sort_values(["timestamp", "sample_id"]).reset_index(drop=True)


def existing_marked_sample_ids(history: Path) -> set[str]:
    marked: set[str] = set()
    chunk = load_label_dataframe(history).fillna("").astype(str)
    note = chunk["note"].fillna("")
    marked.update(chunk.loc[note.str.contains(MARKER, regex=False), "sample_id"].dropna())
    return marked


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
    marked = existing_marked_sample_ids(args.history)
    pending = rows[~rows["sample_id"].isin(marked)].copy()
    rows.to_csv(AUDIT_DIR / "strong_final_round_all.csv", index=False, encoding="utf-8-sig")
    pending.to_csv(AUDIT_DIR / "strong_final_round_pending.csv", index=False, encoding="utf-8-sig")
    print(f"strong final-round rows: {len(rows)}")
    print(f"already registered by marker: {len(rows) - len(pending)}")
    print(f"pending append rows: {len(pending)}")
    if not args.execute:
        print("dry-run only; pass --execute to update label_records.db")
        return
    if pending.empty:
        print("no rows appended")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.history.with_name(f"label_history_before_strong_relabels_{stamp}.csv")
    append_atomically(args.history, pending, backup)
    print(f"backup: {backup}")
    print(f"appended: {len(pending)}")


if __name__ == "__main__":
    main()
