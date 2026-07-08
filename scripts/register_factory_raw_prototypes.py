#!/usr/bin/env python3
"""Register factory_raw/<line>/prototype TDMS files into manifest, samples, and labels."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_manager.label_rules import Label
from data_manager.line_rules import LINE_RULES, normalize_line_time_value
from data_manager.sample_generate import SampleGenerator
from data_manager.tdms_read import open_tdms, tdms_logical_stem


LINES = ("epump2", "epump3", "epump4", "etilt1")
VALID_SUFFIXES = (".tdms.zst", ".tdms", ".tdms.szt")
SOURCE = "expert_prototype_registry"
LABEL_VERSION = "prototype_v1"


@dataclass(frozen=True)
class PrototypeFile:
    path: Path
    line: str
    reason_name: str
    relative_path: str
    sn: str
    reference: str
    time: str


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _is_tdms(path: Path) -> bool:
    return (
        path.is_file()
        and not path.name.startswith("._")
        and path.name.lower().endswith(VALID_SUFFIXES)
    )


def _parse_regular_filename(filename: str, line: str) -> dict[str, str]:
    rule = (LINE_RULES.get(line) or {}).get("filename") or {}
    stem = tdms_logical_stem(filename)
    parts = stem.split(str(rule.get("split") or "_"))

    def pick(spec: Any) -> str:
        if isinstance(spec, int):
            return parts[spec] if spec < len(parts) else ""
        if isinstance(spec, (list, tuple)):
            vals = [parts[i] for i in spec if isinstance(i, int) and i < len(parts)]
            return "_".join(vals)
        return ""

    raw_time = pick(rule.get("time_index"))
    return {
        "sn": pick(rule.get("sn_index")),
        "reference": pick(rule.get("reference_index")),
        "time": normalize_line_time_value(raw_time, line=line, filename_rule=rule),
    }


def _parse_prototype_file(path: Path, *, data_root: Path) -> PrototypeFile:
    factory_root = data_root / "factory_raw"
    relative_path = path.relative_to(factory_root).as_posix()
    parts = Path(relative_path).parts
    if len(parts) < 4 or parts[1] != "prototype":
        raise ValueError(f"not a factory_raw/<line>/prototype file: {path}")
    line = parts[0]
    reason_name = parts[2]
    stem = tdms_logical_stem(path.name)
    generated = stem.split("__", 3)
    if len(generated) == 4 and generated[0] in LINE_RULES:
        original = generated[3]
        original_parsed = _parse_regular_filename(original, line)
        sn = generated[1]
        reference = generated[2]
        time = original_parsed["time"]
    else:
        parsed = _parse_regular_filename(path.name, line)
        sn = parsed["sn"]
        reference = parsed["reference"]
        time = parsed["time"]
    return PrototypeFile(
        path=path,
        line=line,
        reason_name=reason_name,
        relative_path=relative_path,
        sn=sn,
        reference=reference,
        time=time,
    )


def _prototype_files(data_root: Path) -> list[PrototypeFile]:
    out: list[PrototypeFile] = []
    for line in LINES:
        prototype_root = data_root / "factory_raw" / line / "prototype"
        if not prototype_root.is_dir():
            continue
        for path in sorted(prototype_root.rglob("*")):
            if _is_tdms(path):
                out.append(_parse_prototype_file(path, data_root=data_root))
    return out


def _select_channel_rules(line: str, reference: str) -> list[dict[str, Any]]:
    rules = (LINE_RULES.get(line) or {}).get("channels") or {}
    if "conditional" in rules:
        return SampleGenerator._select_conditional_rules(rules["conditional"], reference=reference)
    return [rules]


def _channel_rate(channel: Any) -> int | None:
    wf_increment = channel.properties.get("wf_increment")
    if not wf_increment:
        return None
    return int(round(1.0 / float(wf_increment)))


def _metadata_for_samples(item: PrototypeFile) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rules = _select_channel_rules(item.line, item.reference)
    with open_tdms(item.path, mode="read_metadata") as tdms:
        for rule in rules:
            for direction, group_name in (("up", rule.get("up_group")), ("down", rule.get("down_group"))):
                channel_name = _text(rule.get("acc_channel"))
                group_name = _text(group_name)
                if not group_name or not channel_name:
                    continue
                if group_name not in tdms or channel_name not in tdms[group_name]:
                    continue
                rate = _channel_rate(tdms[group_name][channel_name])
                if rate is None:
                    for group in tdms.groups():
                        for channel in group.channels():
                            rate = _channel_rate(channel)
                            if rate is not None:
                                break
                        if rate is not None:
                            break
                rows.append(
                    {
                        "line": item.line,
                        "sn": item.sn,
                        "sample_id": f"{item.sn}_{direction}",
                        "group_name": group_name,
                        "channel_name": channel_name,
                        "sampling_rate": rate,
                        "reference": item.reference,
                        "time": item.time,
                        "tdms_storage_root": "factory_raw",
                        "relative_path": item.relative_path,
                        "tdms_path": str(item.path),
                        "sample_type": "channel",
                        "sample_config": json.dumps(
                            {
                                "prototype": True,
                                "reason_name": item.reason_name,
                                "direction": direction,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "origin": "prototype",
                        "is_active": 1,
                    }
                )
    return rows


def _load_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for col in ("line", "sn", "reference", "time", "created_time", "tdms_storage_root", "relative_path", "tdms_path"):
        if col not in fieldnames:
            fieldnames.append(col)
    return fieldnames, rows


def _write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in fieldnames})
    tmp.replace(path)


def _sync_manifest(manifest_path: Path, files: list[PrototypeFile], *, execute: bool) -> dict[str, int]:
    fieldnames, rows = _load_manifest(manifest_path)
    by_path = {(_text(row.get("tdms_storage_root")), _text(row.get("relative_path"))): row for row in rows}
    now = datetime.now().isoformat(timespec="seconds")
    stats = Counter()
    for item in files:
        key = ("factory_raw", item.relative_path)
        row = by_path.get(key)
        payload = {
            "line": item.line,
            "sn": item.sn,
            "reference": item.reference,
            "time": item.time,
            "tdms_storage_root": "factory_raw",
            "relative_path": item.relative_path,
            "tdms_path": str(item.path),
        }
        if row is None:
            row = {col: "" for col in fieldnames}
            row["created_time"] = now
            row.update(payload)
            rows.append(row)
            by_path[key] = row
            stats["manifest_added"] += 1
            continue
        changed = False
        for col, value in payload.items():
            if _text(row.get(col)) != _text(value):
                row[col] = str(value)
                changed = True
        if not _text(row.get("created_time")):
            row["created_time"] = now
            changed = True
        stats["manifest_updated" if changed else "manifest_unchanged"] += 1
    if execute:
        _write_manifest(manifest_path, fieldnames, rows)
    return dict(stats)


def _reason_payload(label_runtime: Label, reason_name: str) -> dict[str, Any]:
    reason = label_runtime.get_reason_by_name(reason_name)
    result = label_runtime.get_result_by_reason(reason["key"])
    return {
        "result_key": result["key"],
        "result_id": result["id"],
        "result_name": result["name"],
        "reason_key": reason["key"],
        "reason_id": reason["id"],
        "reason_name": reason["name"],
    }


def _backup(data_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = data_root / "metadata" / "backups" / f"register_factory_raw_prototypes_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(data_root / "metadata" / "tdms_manifest.csv", backup_dir / "tdms_manifest.csv")
    with sqlite3.connect(data_root / "metadata" / "label_records.db") as source, sqlite3.connect(backup_dir / "label_records.db") as target:
        source.backup(target)
    return backup_dir


def _sync_database(
    db_path: Path,
    sample_rows: list[dict[str, Any]],
    label_specs: list[dict[str, Any]],
    *,
    execute: bool,
) -> dict[str, int]:
    stats = Counter()
    if not execute:
        stats["samples_planned"] = len(sample_rows)
        stats["labels_planned"] = len(label_specs)
        return dict(stats)

    now = datetime.now().isoformat(timespec="seconds")
    con = sqlite3.connect(db_path)
    try:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute(
            """
            UPDATE samples
            SET is_active=0, updated_at=?
            WHERE origin='prototype'
              AND (tdms_storage_root IS NULL OR tdms_storage_root='')
              AND (relative_path IS NULL OR relative_path='')
            """,
            (now,),
        )
        stats["stale_prototype_samples_deactivated"] = int(con.execute("SELECT changes()").fetchone()[0])

        con.execute(
            """
            UPDATE samples
            SET is_active=0, updated_at=?
            WHERE origin='prototype'
              AND sample_id LIKE '%__prototype_%'
            """,
            (now,),
        )
        stats["hash_prototype_samples_deactivated"] = int(con.execute("SELECT changes()").fetchone()[0])

        con.executemany(
            """
            INSERT INTO samples(
                line, sn, sample_id, group_name, channel_name, sampling_rate,
                reference, time, tdms_storage_root, relative_path, tdms_path,
                sample_type, sample_config, origin, is_active, created_at, updated_at
            ) VALUES (
                :line, :sn, :sample_id, :group_name, :channel_name, :sampling_rate,
                :reference, :time, :tdms_storage_root, :relative_path, :tdms_path,
                :sample_type, :sample_config, :origin, :is_active, :created_at, :updated_at
            )
            ON CONFLICT(line, sn, sample_id) DO UPDATE SET
                group_name=excluded.group_name,
                channel_name=excluded.channel_name,
                sampling_rate=excluded.sampling_rate,
                reference=excluded.reference,
                time=excluded.time,
                tdms_storage_root=excluded.tdms_storage_root,
                relative_path=excluded.relative_path,
                tdms_path=excluded.tdms_path,
                sample_type=excluded.sample_type,
                sample_config=excluded.sample_config,
                origin=excluded.origin,
                is_active=excluded.is_active,
                updated_at=excluded.updated_at
            """,
            [{**row, "created_at": now, "updated_at": now} for row in sample_rows],
        )
        stats["samples_upserted"] = len(sample_rows)

        sample_lookup = {
            (row[1], row[2], row[3]): int(row[0])
            for row in con.execute("SELECT id,line,sn,sample_id FROM samples")
        }
        con.execute("DELETE FROM label_events WHERE source=?", (SOURCE,))
        stats["old_prototype_labels_deleted"] = int(con.execute("SELECT changes()").fetchone()[0])

        label_values = []
        for spec in label_specs:
            key = (spec["line"], spec["sn"], spec["sample_id"])
            reason = spec["reason"]
            sample_pk = sample_lookup[key]
            label_values.append(
                (
                    str(uuid4()),
                    sample_pk,
                    now,
                    SOURCE,
                    reason["result_key"],
                    int(reason["result_id"]),
                    reason["result_name"],
                    reason["reason_key"],
                    int(reason["reason_id"]),
                    reason["reason_name"],
                    1.0,
                    LABEL_VERSION,
                    f"prototype_relative_path={spec['relative_path']}",
                    "confirmed",
                    now,
                )
            )
        con.executemany(
            """
            INSERT INTO label_events(
                event_uuid, sample_pk, timestamp, source,
                result_key, result_id, result_name,
                reason_key, reason_id, reason_name, reason_confidence,
                label_version, note, status, imported_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            label_values,
        )
        stats["labels_inserted"] = len(label_values)
        con.commit()
    finally:
        con.close()
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register factory_raw/<line>/prototype TDMS files")
    parser.add_argument("--data-root", type=Path, default=Path("/Volumes/18555440521/fault/data_root"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    manifest_path = data_root / "metadata" / "tdms_manifest.csv"
    db_path = data_root / "metadata" / "label_records.db"
    files = _prototype_files(data_root)
    label_runtime = Label()
    sample_rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    label_specs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for item in files:
        try:
            reason = _reason_payload(label_runtime, item.reason_name)
            rows = _metadata_for_samples(item)
            if len(rows) != 2:
                errors.append({"relative_path": item.relative_path, "error": f"expected 2 samples, got {len(rows)}"})
                continue
            for row in rows:
                key = (row["line"], row["sn"], row["sample_id"])
                sample_rows_by_key[key] = row
                label_specs.append(
                    {
                        "line": row["line"],
                        "sn": row["sn"],
                        "sample_id": row["sample_id"],
                        "relative_path": row["relative_path"],
                        "reason": reason,
                    }
                )
        except Exception as exc:
            errors.append({"relative_path": item.relative_path, "error": f"{type(exc).__name__}: {exc}"})

    sample_rows = list(sample_rows_by_key.values())
    backup_dir = ""
    if args.execute:
        backup_dir = str(_backup(data_root))
    manifest_stats = _sync_manifest(manifest_path, files, execute=args.execute)
    db_stats = _sync_database(db_path, sample_rows, label_specs, execute=args.execute)

    line_counts = Counter(item.line for item in files)
    reason_counts = Counter(item.reason_name for item in files)
    summary = {
        "execute": bool(args.execute),
        "data_root": str(data_root),
        "backup_dir": backup_dir,
        "prototype_files": len(files),
        "line_counts": dict(sorted(line_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "sample_rows_ready": len(sample_rows),
        "label_events_ready": len(label_specs),
        "errors": errors[:20],
        "error_count": len(errors),
        "manifest": manifest_stats,
        "database": db_stats,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.execute:
        print("\nDRY-RUN: add --execute to write manifest/db changes.")


if __name__ == "__main__":
    main()
