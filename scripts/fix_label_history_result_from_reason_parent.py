#!/usr/bin/env python3
"""Fix label_history result fields from each known reason's configured parent."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_manager.label_database import load_label_dataframe, replace_confirmed_labels  # noqa: E402

DEFAULT_HISTORY = Path("/Volumes/18555440521/fault/data_root/metadata/label_records.db")
AUDIT_DIR = ROOT / "outputs" / "label_history_result_parent_fix_20260607"
RESULTS = {
    "ok": {"id": "0", "name": "正常"},
    "nok": {"id": "1", "name": "异常"},
    "boundary": {"id": "2", "name": "边界"},
}
REASON_PARENT = {
    "clean_normal": "ok",
    "noisy_normal": "ok",
    "boundary": "boundary",
    "sensor_error": "nok",
    "tick_tock": "nok",
    "friction": "nok",
    "friction_acc": "nok",
    "noise": "nok",
    "gear_chatter": "nok",
    "chatter": "nok",
    "mada": "nok",
    "dada": "nok",
    "other": "nok",
}


def build_preview(history: Path) -> pd.DataFrame:
    frame = load_label_dataframe(history).fillna("").astype(str)
    expected = frame["reason_key"].map(REASON_PARENT).fillna("")
    mask = expected.ne("") & frame["result_key"].ne(expected)
    selected = frame.loc[mask].copy()
    if selected.empty:
        return selected
    selected.insert(0, "old_result_key", selected["result_key"])
    selected.insert(1, "old_result_id", selected["result_id"])
    selected.insert(2, "old_result_name", selected["result_name"])
    selected.insert(3, "new_result_key", expected.loc[mask])
    selected.insert(4, "new_result_id", expected.loc[mask].map(lambda key: RESULTS[key]["id"]))
    selected.insert(5, "new_result_name", expected.loc[mask].map(lambda key: RESULTS[key]["name"]))
    return selected.reset_index(drop=True)


def rewrite(history: Path) -> int:
    rows = load_label_dataframe(history).fillna("").astype(str).to_dict("records")
    changed = 0
    for row in rows:
        expected = REASON_PARENT.get(row.get("reason_key", ""), "")
        if expected and row.get("result_key", "") != expected:
            row["result_key"] = expected
            row["result_id"] = RESULTS[expected]["id"]
            row["result_name"] = RESULTS[expected]["name"]
            changed += 1
    replace_confirmed_labels(history, rows)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    preview = build_preview(args.history)
    preview.to_csv(AUDIT_DIR / "result_parent_fix_preview.csv", index=False, encoding="utf-8-sig")
    print(f"rows requiring fix: {len(preview)}")
    if not preview.empty:
        print(preview.groupby(["line", "old_result_key", "new_result_key", "reason_key"]).size().to_string())
    if not args.execute:
        print("dry-run only; pass --execute to update label_records.db")
        return
    if preview.empty:
        print("no rows changed")
        return
    changed = rewrite(args.history)
    if changed != len(preview):
        raise RuntimeError(f"changed row count mismatch: preview={len(preview)}, actual={changed}")
    print(f"changed: {changed}")


if __name__ == "__main__":
    main()
