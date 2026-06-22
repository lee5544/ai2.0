#!/usr/bin/env python3
"""Move data_root/prototype to data_root/factory_raw/prototype and update metadata."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
import shutil
import sqlite3


LINES = ("epump2", "epump3", "epump4", "etilt1")


def sn_from_name(path: Path, line: str) -> str:
    name = path.name
    generated = re.match(r"^[^_]+__([^_]+)__", name)
    if generated:
        return generated.group(1)
    stem = name.removesuffix(".zst").removesuffix(".tdms")
    parts = stem.split("_")
    if line in {"epump2", "epump3"}:
        return parts[1]
    if line == "epump4":
        return parts[0]
    if line == "etilt1":
        return parts[-1]
    raise ValueError(f"Unsupported line: {line}")


def real_tdms_files(root: Path):
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith("._")
        and (path.name.endswith(".tdms") or path.name.endswith(".tdms.zst"))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve()
    source = root / "prototype"
    target = root / "factory_raw" / "prototype"
    manifest_path = root / "metadata" / "tdms_manifest.csv"
    db_path = root / "metadata" / "label_records.db"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not source.is_dir():
        raise FileNotFoundError(f"Prototype source not found: {source}")
    if target.exists():
        raise FileExistsError(f"Prototype target already exists: {target}")

    with manifest_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        manifest_rows = list(reader)
    prototype_manifest_rows = [
        row for row in manifest_rows if row.get("tdms_storage_root") == "prototype"
    ]

    prototype_files: list[tuple[str, str, Path]] = []
    for line in LINES:
        line_root = source / line
        if not line_root.exists():
            continue
        prototype_files.extend(
            (line, sn_from_name(path, line), path) for path in real_tdms_files(line_root)
        )
    factory_files: list[tuple[str, str, Path]] = []
    for line in LINES:
        line_root = root / "factory_raw" / line
        factory_files.extend(
            (line, sn_from_name(path, line), path) for path in real_tdms_files(line_root)
        )

    prototype_by_key: dict[tuple[str, str], list[Path]] = {}
    factory_by_key: dict[tuple[str, str], list[Path]] = {}
    for line, sn, path in prototype_files:
        prototype_by_key.setdefault((line, sn), []).append(path)
    for line, sn, path in factory_files:
        factory_by_key.setdefault((line, sn), []).append(path)
    internal_duplicates = {key: paths for key, paths in prototype_by_key.items() if len(paths) > 1}
    overlaps = sorted(set(prototype_by_key) & set(factory_by_key))

    print(f"Prototype real TDMS: {len(prototype_files)}")
    print(f"Prototype manifest rows: {len(prototype_manifest_rows)}")
    print(f"Prototype internal duplicate SN: {len(internal_duplicates)}")
    print(f"Prototype vs factory_raw duplicate SN: {len(overlaps)}")
    if not args.execute:
        print("Dry run only. Pass --execute to move.")
        return

    backup_dir = root / "metadata" / "backups" / f"prototype_under_factory_raw_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest_path, backup_dir / "tdms_manifest.csv")
    with sqlite3.connect(db_path) as source_db, sqlite3.connect(backup_dir / "label_records.db") as backup_db:
        source_db.backup(backup_db)

    report_path = backup_dir / "prototype_sn_duplicates.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["type", "line", "sn", "prototype_paths", "factory_paths"])
        writer.writeheader()
        for (line, sn), paths in sorted(internal_duplicates.items()):
            writer.writerow(
                {
                    "type": "prototype_internal",
                    "line": line,
                    "sn": sn,
                    "prototype_paths": " | ".join(str(path.relative_to(root)) for path in paths),
                    "factory_paths": "",
                }
            )
        for line, sn in overlaps:
            writer.writerow(
                {
                    "type": "prototype_vs_factory_raw",
                    "line": line,
                    "sn": sn,
                    "prototype_paths": " | ".join(
                        str(path.relative_to(root)) for path in prototype_by_key[(line, sn)]
                    ),
                    "factory_paths": " | ".join(
                        str(path.relative_to(root)) for path in factory_by_key[(line, sn)]
                    ),
                }
            )

    shutil.move(str(source), str(target))

    for row in prototype_manifest_rows:
        row["tdms_storage_root"] = "factory_raw"
        row["relative_path"] = f"prototype/{row['relative_path'].lstrip('/')}"
    manifest_tmp = manifest_path.with_suffix(".csv.tmp")
    with manifest_tmp.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    manifest_tmp.replace(manifest_path)

    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE samples SET origin='prototype', is_active=1, updated_at=?
            WHERE origin='prototype'
            """,
            (now,),
        )

    missing = [
        row
        for row in prototype_manifest_rows
        if not (root / row["tdms_storage_root"] / row["relative_path"]).is_file()
    ]
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        active = connection.execute(
            "SELECT COUNT(*) FROM samples WHERE origin='prototype' AND is_active=1"
        ).fetchone()[0]
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    if missing or active != 240 or integrity != "ok" or foreign_keys:
        raise RuntimeError(
            f"Verification failed: missing={len(missing)}, active={active}, "
            f"integrity={integrity}, fk={len(foreign_keys)}"
        )

    print(f"Move complete: {target}")
    print(f"Duplicate report: {report_path}")
    print(f"Backup: {backup_dir}")


if __name__ == "__main__":
    main()
