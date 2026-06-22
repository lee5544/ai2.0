#!/usr/bin/env python3
"""Build an etilt1 sample view for SNs labeled as strong motor noise."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_manager.label_database import load_sample_rows  # noqa: E402


DEFAULT_DATA_ROOT = Path("/Volumes/18555440521/fault/data_root")
DEFAULT_LINE = "etilt1"
DEFAULT_OUTPUT = Path("factory_raw/etilt1/马达-强烈/sample_view.csv")
TARGET_LABEL = "强烈-马达音"
OUTPUT_COLUMNS = ["view_name", "line", "sn", "sample_id"]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                return list(reader.fieldnames or []), list(reader)
        except (UnicodeDecodeError, csv.Error) as exc:
            last_error = exc
    raise RuntimeError(f"Cannot read CSV: {path}: {last_error}")


def collect_target_sns(line_root: Path) -> tuple[set[str], Counter[str]]:
    target_sns: set[str] = set()
    source_counts: Counter[str] = Counter()

    for path in sorted(line_root.rglob("*.csv")):
        if path.name.startswith("._"):
            continue

        fieldnames, rows = read_csv(path)
        if "sn" not in fieldnames:
            continue

        matched_in_file = 0
        for row in rows:
            if not any(TARGET_LABEL in str(value or "") for value in row.values()):
                continue
            sn = str(row.get("sn") or "").strip().upper()
            if sn:
                target_sns.add(sn)
                matched_in_file += 1

        if matched_in_file:
            source_counts[str(path)] = matched_in_file

    return target_sns, source_counts


def collect_sample_ids(label_records_db_path: Path, line: str, target_sns: set[str]) -> dict[str, list[str]]:
    rows = load_sample_rows(label_records_db_path)
    sample_ids: dict[str, list[str]] = {}
    for row in rows:
        sn = str(row.get("sn") or "").strip().upper()
        if row.get("line") != line or sn not in target_sns:
            continue
        sample_id = str(row.get("sample_id") or "").strip()
        if sample_id:
            sample_ids.setdefault(sn, []).append(sample_id)
    return sample_ids


def write_sample_view(output_path: Path, line: str, sample_ids: dict[str, list[str]]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for sn in sorted(sample_ids):
            direction_order = {"up": 0, "down": 1}
            ordered_sample_ids = sorted(
                set(sample_ids[sn]),
                key=lambda sample_id: (
                    direction_order.get(sample_id.rsplit("_", 1)[-1], 2),
                    sample_id,
                ),
            )
            for sample_id in ordered_sample_ids:
                writer.writerow(
                    {
                        "view_name": "label",
                        "line": line,
                        "sn": sn,
                        "sample_id": sample_id,
                    }
                )
                row_count += 1
    return row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--line", default=DEFAULT_LINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    line_root = data_root / "factory_raw" / args.line
    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = data_root / output_path

    target_sns, source_counts = collect_target_sns(line_root)
    sample_ids = collect_sample_ids(data_root / "metadata" / "label_records.db", args.line, target_sns)
    missing_sns = sorted(target_sns - set(sample_ids))
    if missing_sns:
        raise RuntimeError(f"{len(missing_sns)} target SNs missing from sample_index: {missing_sns}")

    row_count = write_sample_view(output_path, args.line, sample_ids)
    print(f"Output: {output_path}")
    print(f"Target SNs: {len(target_sns)}")
    print(f"Sample rows: {row_count}")
    for source, count in source_counts.items():
        print(f"Source: {source} ({count} matched rows)")


if __name__ == "__main__":
    main()
