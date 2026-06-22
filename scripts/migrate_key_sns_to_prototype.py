#!/usr/bin/env python3
"""Move factory_raw/<line>/key_sns TDMS files into prototype and update metadata."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
from pathlib import Path
import shutil
import sqlite3


LINES = ("epump2", "epump3", "epump4", "etilt1")
REASON_NAMES = {
    "tick_tock": "秒表",
    "chatter": "震颤",
    "friction": "摩擦",
    "gear_chatter": "咬齿",
    "mada": "马达",
    "other": "其它",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve()
    manifest_path = root / "metadata" / "tdms_manifest.csv"
    db_path = root / "metadata" / "label_records.db"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with manifest_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        manifest_rows = list(reader)

    manifest_by_path = {
        (row["tdms_storage_root"], row["relative_path"]): row for row in manifest_rows
    }
    moves: list[tuple[Path, Path, dict[str, str], str, str]] = []
    for line in LINES:
        source_root = root / "factory_raw" / line / "key_sns"
        for source in sorted(source_root.rglob("*")):
            if not source.is_file() or not (
                source.name.endswith(".tdms") or source.name.endswith(".tdms.zst")
            ):
                continue
            reason_key = source.parent.name
            reason_name = REASON_NAMES.get(reason_key)
            if not reason_name:
                raise ValueError(f"Unsupported key_sns reason folder: {source.parent}")
            old_relative = source.relative_to(root / "factory_raw").as_posix()
            row = manifest_by_path.get(("factory_raw", old_relative))
            if row is None:
                raise KeyError(f"Manifest row not found: factory_raw/{old_relative}")
            target = root / "prototype" / line / "source" / reason_name / source.name
            if target.exists():
                raise FileExistsError(f"Prototype target already exists: {target}")
            moves.append((source, target, row, line, reason_key))

    sample_keys = {(line, row["sn"]) for _, _, row, line, _ in moves}
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        samples = connection.execute(
            "SELECT id, line, sn, sample_id, origin, is_active FROM samples"
        ).fetchall()
        selected_samples = [row for row in samples if (row[1], row[2]) in sample_keys]
        selected_ids = {int(row[0]) for row in selected_samples}
        label_count = sum(
            1
            for (sample_pk,) in connection.execute("SELECT sample_pk FROM label_events")
            if int(sample_pk) in selected_ids
        )

    print(f"TDMS files: {len(moves)}")
    print(f"Samples: {len(selected_samples)}")
    print(f"Labels: {label_count}")
    if not args.execute:
        print("Dry run only. Pass --execute to migrate.")
        return

    backup_dir = root / "metadata" / "backups" / f"key_sns_to_prototype_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest_path, backup_dir / "tdms_manifest.csv")
    with sqlite3.connect(db_path) as source_db, sqlite3.connect(backup_dir / "label_records.db") as backup_db:
        source_db.backup(backup_db)

    for index, (source, target, _, _, _) in enumerate(moves, start=1):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if source.stat().st_size != target.stat().st_size or sha256(source) != sha256(target):
            raise RuntimeError(f"Copy verification failed: {source} -> {target}")
        if index % 20 == 0 or index == len(moves):
            print(f"Copied and verified: {index}/{len(moves)}")

    for _, target, row, line, _ in moves:
        row["tdms_storage_root"] = "prototype"
        row["relative_path"] = target.relative_to(root / "prototype").as_posix()
        row["line"] = line

    manifest_tmp = manifest_path.with_suffix(".csv.tmp")
    with manifest_tmp.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    manifest_tmp.replace(manifest_path)

    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executemany(
            """
            UPDATE samples
            SET origin='prototype', is_active=1, updated_at=?
            WHERE line=? AND sn=?
            """,
            [(now, line, sn) for line, sn in sorted(sample_keys)],
        )

    for _, target, _, _, _ in moves:
        if not target.is_file():
            raise FileNotFoundError(f"Migrated target missing: {target}")
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        active_prototype = connection.execute(
            """
            SELECT COUNT(*) FROM samples
            WHERE origin='prototype' AND is_active=1
            """
        ).fetchone()[0]
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_key_issues = connection.execute("PRAGMA foreign_key_check").fetchall()
    if active_prototype < len(selected_samples) or integrity != "ok" or foreign_key_issues:
        raise RuntimeError(
            f"Post-migration DB verification failed: active={active_prototype}, "
            f"integrity={integrity}, fk={len(foreign_key_issues)}"
        )

    for source_root in (root / "factory_raw" / line / "key_sns" for line in LINES):
        shutil.rmtree(source_root)

    print(f"Migration complete. Backup: {backup_dir}")
    print(f"Active prototype samples: {active_prototype}")


if __name__ == "__main__":
    main()
