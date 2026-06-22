"""Export label_records.db tables to stable CSV formats."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Mapping

from .label_database import LabelDatabase, require_database


SAMPLES_CSV_COLUMNS = [
    "line",
    "sn",
    "sample_id",
    "group_name",
    "channel_name",
    "sampling_rate",
    "sample_type",
    "sample_config",
    "origin",
    "is_active",
    "created_at",
    "updated_at",
]

LABEL_EVENTS_CSV_COLUMNS = [
    "line",
    "sn",
    "sample_id",
    "event_uuid",
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
    "status",
    "imported_at",
]

SAMPLE_LABELS_CSV_COLUMNS = [
    *SAMPLES_CSV_COLUMNS,
    *[column for column in LABEL_EVENTS_CSV_COLUMNS if column not in {"line", "sn", "sample_id"}],
]


def _write_csv(
    path: Path,
    columns: list[str],
    rows: Iterable[Mapping[str, object]],
) -> int:
    normalized_rows = [
        {column: row.get(column, "") for column in columns}
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(normalized_rows)
    return len(normalized_rows)


def export_label_records(
    database_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Export samples, all label events, and latest confirmed sample labels."""
    database = LabelDatabase(require_database(database_path), readonly=True)
    destination = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else database.db_path.parent / "csv"
    )
    samples = database.list_samples(active_only=False)
    events = database.list_label_events()

    latest_confirmed: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for event in events:
        if str(event.get("status") or "") == "confirmed":
            key = (
                str(event.get("line") or ""),
                str(event.get("sn") or ""),
                str(event.get("sample_id") or ""),
            )
            latest_confirmed[key] = event

    sample_labels = []
    for sample in samples:
        key = (
            str(sample.get("line") or ""),
            str(sample.get("sn") or ""),
            str(sample.get("sample_id") or ""),
        )
        sample_labels.append({**sample, **dict(latest_confirmed.get(key, {}))})

    outputs = {
        "samples": destination / "samples.csv",
        "label_events": destination / "label_events.csv",
        "sample_labels": destination / "sample_labels.csv",
    }
    counts = {
        "samples": _write_csv(outputs["samples"], SAMPLES_CSV_COLUMNS, samples),
        "label_events": _write_csv(
            outputs["label_events"], LABEL_EVENTS_CSV_COLUMNS, events
        ),
        "sample_labels": _write_csv(
            outputs["sample_labels"], SAMPLE_LABELS_CSV_COLUMNS, sample_labels
        ),
    }
    return {
        "database_path": str(database.db_path),
        "output_dir": str(destination),
        "files": {key: str(value) for key, value in outputs.items()},
        "counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export label_records.db to CSV")
    parser.add_argument("--database", required=True, help="label_records.db path")
    parser.add_argument(
        "--output-dir",
        default="",
        help="CSV output directory; defaults to <database_dir>/csv",
    )
    args = parser.parse_args()
    result = export_label_records(args.database, args.output_dir or None)
    for name, path in result["files"].items():
        print(f"{name}: {path} | rows={result['counts'][name]}")


if __name__ == "__main__":
    main()
