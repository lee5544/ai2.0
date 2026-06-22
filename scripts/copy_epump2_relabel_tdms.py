#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys


DATA_ROOT = Path("/Volumes/18555440521/fault/data_root")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_manager.label_database import load_label_rows  # noqa: E402

LABEL_HISTORY = DATA_ROOT / "metadata" / "label_records.db"
TDMS_MANIFEST = DATA_ROOT / "metadata" / "tdms_manifest.csv"
DEST_ROOT = DATA_ROOT / "factory_raw" / "epump2" / "再标记"
REPORT_DIR = Path("results") / "epump2_relabel_copy"

GRADE_RANK = {
    "expert": 3,
    "operator一致": 2,
    "operator单个": 1,
    "operator不一致": 1,
}

GRADE_DIR_SUFFIX = {
    "expert": "expert",
    "operator一致": "一致",
    "operator单个": "不一致",
    "operator不一致": "不一致",
}

REASON_DIR_PREFIX = {
    "震颤": "震颤",
    "摩擦acc": "摩擦acc",
}


@dataclass(frozen=True)
class Assignment:
    sn: str
    reason_group: str
    dir_suffix: str
    source_grade: str
    source_sample_id: str
    source_reason_name: str
    source_result_name: str
    tdms_path: Path
    dest_path: Path
    all_selected_samples: str


def norm(value: object) -> str:
    text = ("" if value is None else str(value)).strip()
    return text if text else "(missing)"


def source_kind(source: object) -> str:
    text = norm(source)
    if text == "expert":
        return "expert"
    if text.startswith("operator"):
        return "operator"
    return "other"


def label_key(row: dict[str, str]) -> tuple[str, str]:
    return norm(row.get("result_name")), norm(row.get("reason_name"))


def timestamp_key(row: dict[str, str]) -> tuple[bool, str, int]:
    timestamp = norm(row.get("timestamp"))
    return timestamp != "(missing)", timestamp, int(row["_rownum"])


def reason_group(reason_name: str) -> str | None:
    if reason_name == "震颤":
        return "震颤"
    if reason_name in {"摩擦", "摩擦acc"}:
        return "摩擦acc"
    return None


def read_epump2_label_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rownum, row in enumerate(load_label_rows(path), start=1):
        if norm(row.get("line")).lower() == "epump2":
            row["_rownum"] = str(rownum)
            rows.append(row)
    return rows


def resolve_sample_labels(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        sample_id = norm(row.get("sample_id"))
        if sample_id == "(missing)":
            sample_id = f"__missing_sample_row_{row['_rownum']}"
        by_sample[sample_id].append(row)

    resolved: list[dict[str, str]] = []
    for group in by_sample.values():
        experts = [row for row in group if source_kind(row.get("source")) == "expert"]
        operators = [row for row in group if source_kind(row.get("source")) == "operator"]

        if experts:
            chosen = max(experts, key=timestamp_key)
            grade = "expert"
        elif operators:
            operator_label_counts = Counter(label_key(row) for row in operators)
            if len(operators) == 1:
                chosen = operators[0]
                grade = "operator单个"
            elif len(operator_label_counts) == 1:
                chosen = max(operators, key=timestamp_key)
                grade = "operator一致"
            else:
                max_count = max(operator_label_counts.values())
                candidates = {
                    label
                    for label, count in operator_label_counts.items()
                    if count == max_count
                }
                chosen = max(
                    [row for row in operators if label_key(row) in candidates],
                    key=timestamp_key,
                )
                grade = "operator不一致"
        else:
            chosen = max(group, key=timestamp_key)
            grade = "other"

        grouped_reason = reason_group(norm(chosen.get("reason_name")))
        if grouped_reason is None or grade not in GRADE_RANK:
            continue

        out = dict(chosen)
        out["_grade"] = grade
        out["_reason_group"] = grouped_reason
        resolved.append(out)
    return resolved


def read_tdms_manifest(path: Path) -> dict[str, Path]:
    sn_to_file: dict[str, Path] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if norm(row.get("line")).lower() != "epump2":
                continue
            relative_path = norm(row.get("relative_path"))
            storage_root = norm(row.get("tdms_storage_root"))
            if storage_root == "factory_raw":
                tdms_path = DATA_ROOT / "factory_raw" / relative_path
            elif storage_root == "(missing)":
                tdms_path = DATA_ROOT / relative_path
            else:
                tdms_path = DATA_ROOT / storage_root / relative_path
            sn_to_file[norm(row.get("sn"))] = tdms_path
    return sn_to_file


def summarize_rows(rows: Iterable[dict[str, str]]) -> str:
    parts: list[str] = []
    for row in rows:
        parts.append(
            ":".join(
                [
                    norm(row.get("sample_id")),
                    norm(row.get("_reason_group")),
                    norm(row.get("_grade")),
                    norm(row.get("reason_name")),
                ]
            )
        )
    return "; ".join(sorted(parts))


def build_assignments(
    sample_rows: list[dict[str, str]],
    sn_to_file: dict[str, Path],
    reason_tie: str,
) -> tuple[list[Assignment], list[dict[str, str]], list[dict[str, str]]]:
    by_sn: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sample_rows:
        by_sn[norm(row.get("sn"))].append(row)

    assignments: list[Assignment] = []
    conflicts: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    for sn, rows in sorted(by_sn.items()):
        max_rank = max(GRADE_RANK[row["_grade"]] for row in rows)
        top_rows = [row for row in rows if GRADE_RANK[row["_grade"]] == max_rank]
        top_reasons = {row["_reason_group"] for row in top_rows}

        if len(top_reasons) > 1:
            conflicts.append(
                {
                    "sn": sn,
                    "top_reasons": ";".join(sorted(top_reasons)),
                    "selected_samples": summarize_rows(rows),
                }
            )
            if reason_tie == "error":
                continue
            selected_reason = reason_tie
        else:
            selected_reason = next(iter(top_reasons))

        candidate_rows = [
            row for row in top_rows if row["_reason_group"] == selected_reason
        ]
        chosen = max(candidate_rows, key=timestamp_key)
        tdms_path = sn_to_file.get(sn)
        if tdms_path is None:
            missing.append(
                {
                    "sn": sn,
                    "reason_group": selected_reason,
                    "dir_suffix": GRADE_DIR_SUFFIX[chosen["_grade"]],
                    "missing_type": "manifest",
                    "selected_samples": summarize_rows(rows),
                }
            )
            continue
        if not tdms_path.exists():
            missing.append(
                {
                    "sn": sn,
                    "reason_group": selected_reason,
                    "dir_suffix": GRADE_DIR_SUFFIX[chosen["_grade"]],
                    "missing_type": "file",
                    "tdms_path": str(tdms_path),
                    "selected_samples": summarize_rows(rows),
                }
            )
            continue

        dir_name = REASON_DIR_PREFIX[selected_reason] + GRADE_DIR_SUFFIX[chosen["_grade"]]
        dest_path = DEST_ROOT / dir_name / tdms_path.name
        assignments.append(
            Assignment(
                sn=sn,
                reason_group=selected_reason,
                dir_suffix=GRADE_DIR_SUFFIX[chosen["_grade"]],
                source_grade=chosen["_grade"],
                source_sample_id=norm(chosen.get("sample_id")),
                source_reason_name=norm(chosen.get("reason_name")),
                source_result_name=norm(chosen.get("result_name")),
                tdms_path=tdms_path,
                dest_path=dest_path,
                all_selected_samples=summarize_rows(rows),
            )
        )

    return assignments, conflicts, missing


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    assignments: list[Assignment],
    conflicts: list[dict[str, str]],
    missing: list[dict[str, str]],
) -> None:
    assignment_rows: list[dict[str, object]] = []
    for item in assignments:
        assignment_rows.append(
            {
                "sn": item.sn,
                "reason_group": item.reason_group,
                "dir_suffix": item.dir_suffix,
                "source_grade": item.source_grade,
                "source_sample_id": item.source_sample_id,
                "source_reason_name": item.source_reason_name,
                "source_result_name": item.source_result_name,
                "tdms_path": str(item.tdms_path),
                "dest_path": str(item.dest_path),
                "size_bytes": item.tdms_path.stat().st_size,
                "all_selected_samples": item.all_selected_samples,
            }
        )

    write_csv(
        REPORT_DIR / "assignments.csv",
        assignment_rows,
        [
            "sn",
            "reason_group",
            "dir_suffix",
            "source_grade",
            "source_sample_id",
            "source_reason_name",
            "source_result_name",
            "tdms_path",
            "dest_path",
            "size_bytes",
            "all_selected_samples",
        ],
    )
    write_csv(
        REPORT_DIR / "reason_conflicts.csv",
        conflicts,
        ["sn", "top_reasons", "selected_samples"],
    )
    missing_fields = [
        "sn",
        "reason_group",
        "dir_suffix",
        "missing_type",
        "tdms_path",
        "selected_samples",
    ]
    write_csv(REPORT_DIR / "missing.csv", missing, missing_fields)


def copy_assignments(assignments: list[Assignment], overwrite: bool) -> None:
    for item in assignments:
        item.dest_path.parent.mkdir(parents=True, exist_ok=True)
        if item.dest_path.exists() and not overwrite:
            continue
        shutil.copy2(item.tdms_path, item.dest_path)


def print_summary(
    assignments: list[Assignment],
    conflicts: list[dict[str, str]],
    missing: list[dict[str, str]],
) -> None:
    print(f"assignments={len(assignments)}")
    print(f"reason_conflicts={len(conflicts)}")
    print(f"missing={len(missing)}")
    print(f"report_dir={REPORT_DIR}")

    counts = Counter((item.reason_group, item.dir_suffix) for item in assignments)
    sizes: dict[tuple[str, str], int] = defaultdict(int)
    for item in assignments:
        sizes[(item.reason_group, item.dir_suffix)] += item.tdms_path.stat().st_size

    print("| reason_group | target_suffix | files | size_gib | size_gb |")
    print("|---|---|---:|---:|---:|")
    for key in sorted(counts):
        bytes_count = sizes[key]
        print(
            f"| {key[0]} | {key[1]} | {counts[key]} | "
            f"{bytes_count / 1024**3:.2f} | {bytes_count / 1000**3:.2f} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy epump2 vibration/friction TDMS files into relabel folders."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually copy files. Without this flag only reports are written.",
    )
    parser.add_argument(
        "--reason-tie",
        choices=["error", "震颤", "摩擦acc"],
        default="error",
        help="How to resolve same-rank SNs that match both reason groups.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_epump2_label_rows(LABEL_HISTORY)
    sample_rows = resolve_sample_labels(rows)
    sn_to_file = read_tdms_manifest(TDMS_MANIFEST)
    assignments, conflicts, missing = build_assignments(
        sample_rows=sample_rows,
        sn_to_file=sn_to_file,
        reason_tie=args.reason_tie,
    )
    write_reports(assignments, conflicts, missing)
    print_summary(assignments, conflicts, missing)

    if args.execute:
        if conflicts and args.reason_tie == "error":
            raise SystemExit(
                "Refusing to copy: reason conflicts exist. "
                "Pass --reason-tie 震颤 or --reason-tie 摩擦acc."
            )
        copy_assignments(assignments, overwrite=args.overwrite)
        print("copy_done=1")
    else:
        print("copy_done=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
