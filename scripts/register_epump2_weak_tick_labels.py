#!/usr/bin/env python3
"""Append ep2-漏判0529.xlsx Up/Down labels to label_records.db."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import sys


DATA_ROOT = Path("/Volumes/18555440521/fault/data_root")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_manager.label_database import append_confirmed_labels, load_label_dataframe  # noqa: E402

DEFAULT_XLSX = DATA_ROOT / "metadata" / "ep2-漏判0529.xlsx"
DEFAULT_HISTORY = DATA_ROOT / "metadata" / "label_records.db"
DEFAULT_TDMS_DIR = DATA_ROOT / "factory_raw" / "epump2" / "weak_tick"
AUDIT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "ep2_loupan0529_label_history_update"
MARKER = "ep2_loupan0529_weak_tick_20260608"
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
LABEL_MAP = {
    "OK": ("ok", "0", "正常", "clean_normal", "0", "正常"),
    "秒表": ("nok", "1", "异常", "tick_tock", "102", "秒表"),
    "短促秒表": ("nok", "1", "异常", "tick_tock", "102", "秒表"),
    "摩擦": ("nok", "1", "异常", "friction", "103", "摩擦"),
}


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _find_available_sns(tdms_dir: Path) -> set[str]:
    available: set[str] = set()
    for path in tdms_dir.glob("*.tdms.zst"):
        if path.name.startswith("._"):
            continue
        parts = path.name.split("_")
        if len(parts) > 1:
            available.add(parts[1].strip())
    return available


def build_rows(xlsx: Path, tdms_dir: Path, timestamp: str) -> tuple[pd.DataFrame, list[str]]:
    labels = pd.read_excel(xlsx, dtype=str)
    labels = labels[labels["sn"].notna() & labels["sn"].astype(str).str.strip().ne("")].copy()
    available_sns = _find_available_sns(tdms_dir)
    missing_sns = sorted(set(labels["sn"].map(_text)) - available_sns)
    labels = labels[labels["sn"].map(_text).isin(available_sns)].copy()

    rows: list[dict[str, str]] = []
    for row_number, row in labels.iterrows():
        sn = _text(row["sn"])
        note = _text(row.get("Unnamed: 13"))
        for direction, column in (("up", "m_up_type"), ("down", "m_down_type")):
            raw_label = _text(row.get(column))
            if raw_label not in LABEL_MAP:
                raise ValueError(f"不支持的标签: sn={sn}, column={column}, value={raw_label!r}")
            result_key, result_id, result_name, reason_key, reason_id, reason_name = LABEL_MAP[raw_label]
            rows.append(
                {
                    "line": "epump2",
                    "sn": sn,
                    "sample_id": f"{sn}_{direction}",
                    "timestamp": timestamp,
                    "source": "expert_ep2_loupan0529",
                    "result_key": result_key,
                    "result_id": result_id,
                    "result_name": result_name,
                    "reason_key": reason_key,
                    "reason_id": reason_id,
                    "reason_name": reason_name,
                    "reason_confidence": "0.9",
                    "label_version": "v1.1",
                    "note": (
                        f"import_marker={MARKER} | source_file={xlsx.name}"
                        f" | source_row={row_number + 2} | direction={direction}"
                        f" | raw_label={raw_label} | note={note or '<empty>'}"
                    ),
                }
            )
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS), missing_sns


def existing_marked_sample_ids(history: Path) -> set[str]:
    marked: set[str] = set()
    chunk = load_label_dataframe(history).fillna("").astype(str)
    notes = chunk["note"].fillna("")
    marked.update(chunk.loc[notes.str.contains(MARKER, regex=False), "sample_id"].dropna())
    return marked


def append_atomically(history: Path, rows: pd.DataFrame, backup: Path) -> None:
    del backup
    append_confirmed_labels(history, rows[HISTORY_COLUMNS].to_dict("records"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--tdms-dir", type=Path, default=DEFAULT_TDMS_DIR)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().isoformat(timespec="seconds")
    rows, missing_sns = build_rows(args.xlsx, args.tdms_dir, timestamp)
    marked = existing_marked_sample_ids(args.history)
    pending = rows[~rows["sample_id"].isin(marked)].copy()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(AUDIT_DIR / "all_rows.csv", index=False, encoding="utf-8-sig")
    pending.to_csv(AUDIT_DIR / "pending_rows.csv", index=False, encoding="utf-8-sig")
    print(f"matched SNs: {rows['sn'].nunique()}")
    print(f"missing TDMS SNs: {missing_sns}")
    print(f"label rows: {len(rows)}")
    print(f"already registered by marker: {len(rows) - len(pending)}")
    print(f"pending append rows: {len(pending)}")
    print(pending.groupby(["reason_name", "sample_id"]).size().groupby(level=0).size().to_string())

    if not args.execute:
        print("dry-run only; pass --execute to update label_records.db")
        return
    if pending.empty:
        print("no rows appended")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.history.with_name(f"label_history_before_ep2_loupan0529_{stamp}.csv")
    append_atomically(args.history, pending, backup)
    print(f"backup: {backup}")
    print(f"appended: {len(pending)}")


if __name__ == "__main__":
    main()
