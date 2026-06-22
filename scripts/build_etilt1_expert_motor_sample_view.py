#!/usr/bin/env python3
"""Build a sample view from etilt1 expert motor labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_manager.label_database import load_label_rows  # noqa: E402


DEFAULT_DATA_ROOT = Path("/Volumes/18555440521/fault/data_root")
DEFAULT_OUTPUT = Path("factory_raw/etilt1/专家-马达/sample_view.csv")
OUTPUT_COLUMNS = ["view_name", "line", "sn", "sample_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = data_root / output_path

    target_rows: dict[str, dict[str, str]] = {}
    for row in load_label_rows(data_root / "metadata" / "label_records.db"):
        if row.get("line") != "etilt1" or row.get("source") != "expert":
            continue
        if row.get("reason_key") != "mada" and row.get("reason_name") != "马达":
            continue

        sample_id = str(row.get("sample_id") or "").strip()
        sn = str(row.get("sn") or "").strip().upper()
        if sample_id and sn:
            target_rows[sample_id] = {
                "view_name": "label",
                "line": "etilt1",
                "sn": sn,
                "sample_id": sample_id,
            }

    direction_order = {"up": 0, "down": 1}
    rows = sorted(
        target_rows.values(),
        key=lambda row: (
            row["sn"],
            direction_order.get(row["sample_id"].rsplit("_", 1)[-1], 2),
            row["sample_id"],
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Output: {output_path}")
    print(f"Target SNs: {len({row['sn'] for row in rows})}")
    print(f"Sample rows: {len(rows)}")


if __name__ == "__main__":
    main()
