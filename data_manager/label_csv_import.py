"""Import one or more supported label CSV files into label_records.db."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Iterable

from .label_database import LabelDatabase
from .label_forvia_registry import register_forvia_labels


def _read_csv(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as stream:
                return [dict(row) for row in csv.DictReader(stream)]
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"无法识别 CSV 编码: {path}")


def import_label_csv(
    db_path: str | Path,
    csv_path: str | Path,
    *,
    line: str = "",
) -> dict[str, Any]:
    """Import sample_view rows or a supported Forvia wide table into one DB."""
    database_path = Path(db_path).expanduser().resolve()
    input_path = Path(csv_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"标签 CSV 不存在: {input_path}")

    rows = _read_csv(input_path)
    header = set(rows[0]) if rows else set()
    is_sample_view = {"sn", "sample_id"}.issubset(header) and bool(
        header & {"result_key", "result_name", "reason_key", "reason_name"}
    )
    if is_sample_view:
        store = LabelDatabase(database_path)
        imported = 0
        skipped = 0
        for row in rows:
            row_line = str(row.get("line") or line or "").strip()
            sn = str(row.get("sn") or "").strip()
            sample_id = str(row.get("sample_id") or "").strip()
            if not row_line or not sn or not sample_id:
                skipped += 1
                continue
            try:
                store.append_label(
                    line=row_line,
                    sn=sn,
                    sample_id=sample_id,
                    status=str(row.get("status") or "confirmed").strip(),
                    label={
                        "timestamp": row.get("label_timestamp") or row.get("timestamp"),
                        "source": row.get("label_source") or row.get("source") or "sample_view_import",
                        "result_key": row.get("result_key"),
                        "result_id": row.get("result_id"),
                        "result_name": row.get("result_name"),
                        "reason_key": row.get("reason_key"),
                        "reason_id": row.get("reason_id"),
                        "reason_name": row.get("reason_name"),
                        "reason_confidence": row.get("reason_confidence"),
                        "label_version": row.get("label_version"),
                        "note": row.get("note"),
                    },
                )
                imported += 1
            except KeyError:
                skipped += 1
        return {
            "format": "sample_view",
            "imported_labels": imported,
            "skipped_rows": skipped,
            "label_csv": str(input_path),
        }

    before = LabelDatabase(database_path, readonly=True).counts()["label_events"]
    registration = register_forvia_labels(
        input_csv=input_path,
        label_records_db=database_path,
        line=line,
    )
    after = LabelDatabase(database_path, readonly=True).counts()["label_events"]
    return {
        "format": "employee_operator_wide"
        if any(key.startswith("label_") for key in header)
        else "standard_wide",
        "imported_labels": max(0, int(after) - int(before)),
        "label_csv": str(input_path),
        "log": registration["log"],
    }


def import_label_csvs(
    db_path: str | Path,
    csv_paths: Iterable[str | Path],
    *,
    line: str = "",
    progress: Callable[[int, int, Path], None] | None = None,
) -> dict[str, Any]:
    """Import multiple CSV files in order and return an aggregate report."""
    paths = [Path(path).expanduser().resolve() for path in csv_paths]
    reports: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        if progress is not None:
            progress(index, len(paths), path)
        reports.append(import_label_csv(db_path, path, line=line))
    return {
        "files": reports,
        "csv_count": len(reports),
        "imported_labels": sum(int(item.get("imported_labels") or 0) for item in reports),
        "skipped_rows": sum(int(item.get("skipped_rows") or 0) for item in reports),
    }
