#!/usr/bin/env python3
"""Audit factory_raw/<line>/prototype TDMS registration and labels.

Checks:
- real TDMS / TDMS.ZST files under factory_raw/<line>/prototype recursively
- exact tdms_manifest.csv registration path
- samples rows for the inferred line/sn
- whether sample path/origin points to the prototype file
- latest confirmed label reason versus the prototype folder reason
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_manager.line_rules import LINE_RULES, normalize_line_time_value


LINES = ("epump2", "epump3", "epump4", "etilt1")
VALID_SUFFIXES = (".tdms.zst", ".tdms", ".tdms.szt")


@dataclass(frozen=True)
class ManifestRow:
    line: str
    sn: str
    reference: str
    time: str
    tdms_storage_root: str
    relative_path: str
    tdms_path: str


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _logical_stem(path: Path) -> str:
    name = path.name
    lower = name.lower()
    if lower.endswith(".tdms.zst"):
        return name[:-9]
    if lower.endswith(".tdms.szt"):
        return name[:-9]
    if lower.endswith(".tdms"):
        return name[:-5]
    return path.stem


def _parse_filename(path: Path, line: str) -> dict[str, str]:
    stem = _logical_stem(path)
    generated = re.match(r"^(?P<line>[^_]+)__(?P<sn>[^_]+)__(?P<reference>[^_]+)__", stem)
    if generated:
        return {
            "line": generated.group("line"),
            "sn": generated.group("sn"),
            "reference": generated.group("reference"),
            "time": "",
        }

    rule = (LINE_RULES.get(line) or {}).get("filename") or {}
    split = str(rule.get("split") or "_")
    parts = stem.split(split)

    def pick(spec: Any) -> str:
        if isinstance(spec, int):
            return parts[spec] if spec < len(parts) else ""
        if isinstance(spec, (list, tuple)):
            vals = [parts[i] for i in spec if isinstance(i, int) and i < len(parts)]
            return "_".join(vals)
        return ""

    raw_time = pick(rule.get("time_index"))
    return {
        "line": line,
        "sn": pick(rule.get("sn_index")),
        "reference": pick(rule.get("reference_index")),
        "time": normalize_line_time_value(raw_time, line=line, filename_rule=rule),
    }


def _is_tdms(path: Path) -> bool:
    if not path.is_file() or path.name.startswith("._"):
        return False
    return path.name.lower().endswith(VALID_SUFFIXES)


def _prototype_files(data_root: Path) -> list[Path]:
    factory_root = data_root / "factory_raw"
    out: list[Path] = []
    for line in LINES:
        prototype_root = factory_root / line / "prototype"
        if not prototype_root.is_dir():
            continue
        out.extend(path for path in sorted(prototype_root.rglob("*")) if _is_tdms(path))
    return out


def _load_manifest(manifest_path: Path) -> dict[tuple[str, str], ManifestRow]:
    out: dict[tuple[str, str], ManifestRow] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            item = ManifestRow(
                line=_text(row.get("line")),
                sn=_text(row.get("sn")),
                reference=_text(row.get("reference")),
                time=_text(row.get("time")),
                tdms_storage_root=_text(row.get("tdms_storage_root")),
                relative_path=_text(row.get("relative_path")),
                tdms_path=_text(row.get("tdms_path")),
            )
            out[(item.tdms_storage_root, item.relative_path)] = item
    return out


def _load_samples(db_path: Path) -> tuple[dict[tuple[str, str], list[sqlite3.Row]], dict[int, sqlite3.Row]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = list(
            con.execute(
                "SELECT id,line,sn,sample_id,group_name,channel_name,sampling_rate,"
                "reference,time,tdms_storage_root,relative_path,tdms_path,origin,is_active "
                "FROM samples"
            )
        )
    finally:
        con.close()
    by_key: dict[tuple[str, str], list[sqlite3.Row]] = {}
    by_id: dict[int, sqlite3.Row] = {}
    for row in rows:
        by_key.setdefault((_text(row["line"]), _text(row["sn"])), []).append(row)
        by_id[int(row["id"])] = row
    return by_key, by_id


def _load_confirmed_labels(db_path: Path) -> dict[int, list[sqlite3.Row]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = list(
            con.execute(
                """
                SELECT *
                FROM label_events
                WHERE status='confirmed'
                ORDER BY timestamp DESC, id DESC
                """
            )
        )
    finally:
        con.close()
    out: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        out.setdefault(int(row["sample_pk"]), []).append(row)
    return out


def _issue(row: dict[str, Any], issues: list[str], issue: str) -> None:
    issues.append(issue)
    row[issue] = "1"


def audit(data_root: Path, output_csv: Path) -> dict[str, Any]:
    manifest_path = data_root / "metadata" / "tdms_manifest.csv"
    db_path = data_root / "metadata" / "label_records.db"
    manifest = _load_manifest(manifest_path)
    samples_by_key, _samples_by_id = _load_samples(db_path)
    confirmed_labels = _load_confirmed_labels(db_path)
    factory_root = data_root / "factory_raw"
    files = _prototype_files(data_root)

    report_rows: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    line_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    for path in files:
        rel = path.relative_to(factory_root).as_posix()
        parts = Path(rel).parts
        line = parts[0] if len(parts) > 0 else ""
        reason_from_folder = parts[2] if len(parts) > 2 else ""
        parsed = _parse_filename(path, line)
        manifest_row = manifest.get(("factory_raw", rel))
        sn = _text(manifest_row.sn if manifest_row else "") or _text(parsed.get("sn"))
        key = (line, sn)
        sample_rows = samples_by_key.get(key, [])
        all_label_rows = [label for row in sample_rows for label in confirmed_labels.get(int(row["id"]), [])]
        path_label_rows = [
            label for label in all_label_rows
            if rel in _text(label["note"]) or _text(label["note"]) == ""
        ]
        labels_for_reason_check = path_label_rows or all_label_rows
        reason_set = sorted(set(_text(label["reason_name"]) for label in labels_for_reason_check if _text(label["reason_name"])))
        exact_path_samples = [
            row for row in sample_rows
            if _text(row["tdms_storage_root"]) == "factory_raw" and _text(row["relative_path"]) == rel
        ]
        prototype_origin_samples = [row for row in sample_rows if _text(row["origin"]) == "prototype"]

        issues: list[str] = []
        out = {
            "line": line,
            "reason_from_folder": reason_from_folder,
            "relative_path": rel,
            "parsed_sn": _text(parsed.get("sn")),
            "manifest_found": "1" if manifest_row else "0",
            "manifest_sn": _text(manifest_row.sn if manifest_row else ""),
            "manifest_reference": _text(manifest_row.reference if manifest_row else ""),
            "sample_count_line_sn": len(sample_rows),
            "sample_count_exact_path": len(exact_path_samples),
            "sample_count_origin_prototype": len(prototype_origin_samples),
            "latest_reason_names": " | ".join(reason_set),
            "issues": "",
        }

        if not manifest_row:
            _issue(out, issues, "missing_manifest")
        else:
            if manifest_row.line != line:
                _issue(out, issues, "manifest_line_mismatch")
            if _text(parsed.get("sn")) and manifest_row.sn != _text(parsed.get("sn")):
                _issue(out, issues, "manifest_sn_mismatch")

        if not sample_rows:
            _issue(out, issues, "missing_samples_for_sn")
        if len(sample_rows) < 2:
            _issue(out, issues, "sample_count_lt_2")
        if not exact_path_samples:
            _issue(out, issues, "sample_path_not_registered")
        if not prototype_origin_samples:
            _issue(out, issues, "sample_origin_not_prototype")
        if not reason_set:
            _issue(out, issues, "missing_latest_confirmed_label")
        elif reason_from_folder and reason_from_folder not in reason_set:
            _issue(out, issues, "latest_label_reason_mismatch")

        out["issues"] = " | ".join(issues)
        for issue in issues:
            issue_counts[issue] += 1
        line_counts[line] += 1
        reason_counts[reason_from_folder] += 1
        report_rows.append(out)

    fieldnames = [
        "line",
        "reason_from_folder",
        "relative_path",
        "parsed_sn",
        "manifest_found",
        "manifest_sn",
        "manifest_reference",
        "sample_count_line_sn",
        "sample_count_exact_path",
        "sample_count_origin_prototype",
        "latest_reason_names",
        "issues",
        "missing_manifest",
        "manifest_line_mismatch",
        "manifest_sn_mismatch",
        "missing_samples_for_sn",
        "sample_count_lt_2",
        "sample_path_not_registered",
        "sample_origin_not_prototype",
        "missing_latest_confirmed_label",
        "latest_label_reason_mismatch",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    return {
        "data_root": str(data_root),
        "output_csv": str(output_csv),
        "prototype_file_count": len(files),
        "line_counts": dict(sorted(line_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "clean_file_count": sum(1 for row in report_rows if not row["issues"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit prototype TDMS registration and labels")
    parser.add_argument("--data-root", type=Path, default=Path("/Volumes/18555440521/fault/data_root"))
    parser.add_argument("--output-csv", type=Path, default=Path("tmp/prototype_registration_audit.csv"))
    args = parser.parse_args()
    summary = audit(args.data_root.expanduser(), args.output_csv.expanduser())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
