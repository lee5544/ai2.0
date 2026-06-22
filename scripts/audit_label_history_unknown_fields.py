#!/usr/bin/env python3
"""Audit label_records.db for unknown label values and empty label groups."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_manager.label_database import load_label_dataframe  # noqa: E402

HISTORY = Path("/Volumes/18555440521/fault/data_root/metadata/label_records.db")
OUTPUT_DIR = ROOT / "outputs" / "label_history_unknown_fields_20260607"

RESULTS = {
    "ok": {"id": "0", "name": "正常"},
    "nok": {"id": "1", "name": "异常"},
    "boundary": {"id": "2", "name": "边界"},
}
REASONS = {
    "clean_normal": {"id": "0", "name": "正常", "parent": "ok"},
    "noisy_normal": {"id": "-1", "name": "干扰", "parent": "ok"},
    "boundary": {"id": "2", "name": "边界", "parent": "boundary"},
    "sensor_error": {"id": "101", "name": "传感器错误", "parent": "nok"},
    "tick_tock": {"id": "102", "name": "秒表", "parent": "nok"},
    "friction": {"id": "103", "name": "摩擦", "parent": "nok"},
    "friction_acc": {"id": "104", "name": "摩擦acc", "parent": "nok"},
    "noise": {"id": "105", "name": "杂音", "parent": "nok"},
    "gear_chatter": {"id": "106", "name": "咬齿", "parent": "nok"},
    "chatter": {"id": "107", "name": "震颤", "parent": "nok"},
    "mada": {"id": "108", "name": "马达", "parent": "nok"},
    "dada": {"id": "109", "name": "哒哒_咔咔", "parent": "nok"},
    "other": {"id": "199", "name": "其它", "parent": "nok"},
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unknown_parts = []
    empty_parts = []
    value_counts: dict[str, dict[str, int]] = {}
    total = 0
    label_columns = [
        "result_key",
        "result_id",
        "result_name",
        "reason_key",
        "reason_id",
        "reason_name",
        "reason_confidence",
    ]

    for chunk in [load_label_dataframe(HISTORY).fillna("").astype(str)]:
        total += len(chunk)
        for column in label_columns + ["timestamp", "source", "label_version"]:
            counts = chunk[column].value_counts(dropna=False)
            target = value_counts.setdefault(column, {})
            for value, count in counts.items():
                target[str(value)] = target.get(str(value), 0) + int(count)

        result_key = chunk["result_key"]
        reason_key = chunk["reason_key"]
        result_known = result_key.isin(RESULTS)
        reason_known = reason_key.isin(REASONS)
        result_nonempty = result_key.ne("")
        reason_nonempty = reason_key.ne("")

        expected_result_id = result_key.map(lambda key: RESULTS.get(key, {}).get("id", ""))
        expected_result_name = result_key.map(lambda key: RESULTS.get(key, {}).get("name", ""))
        expected_reason_id = reason_key.map(lambda key: REASONS.get(key, {}).get("id", ""))
        expected_reason_name = reason_key.map(lambda key: REASONS.get(key, {}).get("name", ""))
        expected_parent = reason_key.map(lambda key: REASONS.get(key, {}).get("parent", ""))

        reasons = pd.Series("", index=chunk.index, dtype=str)
        def add_reason(mask: pd.Series, text: str) -> None:
            reasons.loc[mask] = reasons.loc[mask].map(lambda old: f"{old} | {text}".strip(" |"))

        add_reason(result_nonempty & ~result_known, "unknown_result_key")
        add_reason(reason_nonempty & ~reason_known, "unknown_reason_key")
        add_reason(result_known & chunk["result_id"].ne(expected_result_id), "result_id_mismatch")
        add_reason(result_known & chunk["result_name"].ne(expected_result_name), "result_name_mismatch")
        add_reason(reason_known & chunk["reason_id"].ne(expected_reason_id), "reason_id_mismatch")
        add_reason(reason_known & chunk["reason_name"].ne(expected_reason_name), "reason_name_mismatch")
        add_reason(result_known & reason_known & result_key.ne(expected_parent), "reason_parent_mismatch")
        confidence = pd.to_numeric(chunk["reason_confidence"], errors="coerce")
        add_reason(chunk["reason_confidence"].ne("") & confidence.isna(), "invalid_reason_confidence")
        add_reason(confidence.notna() & ((confidence < 0) | (confidence > 1)), "confidence_out_of_range")

        unknown = chunk.loc[reasons.ne("")].copy()
        if not unknown.empty:
            unknown.insert(0, "audit_reason", reasons.loc[unknown.index])
            unknown_parts.append(unknown)

        any_label_empty = chunk[label_columns].eq("").any(axis=1)
        empty = chunk.loc[any_label_empty].copy()
        if not empty.empty:
            empty.insert(
                0,
                "empty_fields",
                chunk.loc[empty.index, label_columns].apply(
                    lambda row: " | ".join(row.index[row.eq("")]), axis=1
                ),
            )
            empty_parts.append(empty)

    unknown_df = pd.concat(unknown_parts, ignore_index=True) if unknown_parts else pd.DataFrame()
    empty_df = pd.concat(empty_parts, ignore_index=True) if empty_parts else pd.DataFrame()
    unknown_df.to_csv(OUTPUT_DIR / "unknown_or_mismatched_labels.csv", index=False, encoding="utf-8-sig")
    empty_df.to_csv(OUTPUT_DIR / "empty_label_fields.csv", index=False, encoding="utf-8-sig")

    count_rows = []
    for column, counts in value_counts.items():
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            count_rows.append({"column": column, "value": value, "count": count})
    pd.DataFrame(count_rows).to_csv(
        OUTPUT_DIR / "field_value_counts.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "total_rows": total,
        "unknown_or_mismatched_rows": len(unknown_df),
        "empty_label_rows": len(empty_df),
        "unknown_reasons": (
            unknown_df["audit_reason"].value_counts().to_dict() if not unknown_df.empty else {}
        ),
        "empty_patterns": (
            empty_df["empty_fields"].value_counts().to_dict() if not empty_df.empty else {}
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
