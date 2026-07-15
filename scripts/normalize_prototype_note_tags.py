#!/usr/bin/env python3
"""Normalize prototype note tags in label_records.db.

Legacy notes may contain [[典型异音]] or [[typical]]. The canonical marker is
[[prototype]], and only one copy should remain.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


OLD_TAGS = ("[[典型异音]]", "[[typical]]")
CANONICAL_TAG = "[[prototype]]"
ALL_TAGS = (*OLD_TAGS, CANONICAL_TAG)


def normalize_note(note: str) -> str:
    text = str(note or "")
    has_marker = any(tag in text for tag in ALL_TAGS)
    for tag in ALL_TAGS:
        text = text.replace(tag, " ")
    parts = text.split()
    if has_marker:
        parts.insert(0, CANONICAL_TAG)
    return " ".join(parts).strip()


def backup_db(db_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if db_path.parent.name == "metadata":
        backup_dir = db_path.parent / "backups" / f"normalize_prototype_note_tags_{stamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        target = backup_dir / db_path.name
    else:
        target = db_path.with_name(f"{db_path.stem}.bak_normalize_prototype_note_tags_{stamp}{db_path.suffix}")
    shutil.copy2(db_path, target)
    return target


def normalize_db(db_path: Path, *, dry_run: bool = False) -> dict:
    db_path = db_path.expanduser()
    if not db_path.exists():
        return {"db": str(db_path), "exists": False, "updated": 0, "backup": ""}

    con = sqlite3.connect(db_path)
    rows = con.execute(
        """
        SELECT id, note
        FROM label_events
        WHERE note LIKE '%[[典型异音]]%'
           OR note LIKE '%[[typical]]%'
           OR note LIKE '%[[prototype]]%[[prototype]]%'
        """
    ).fetchall()
    changes = [(row_id, normalize_note(note)) for row_id, note in rows if normalize_note(note) != str(note or "")]
    backup = ""
    if changes and not dry_run:
        backup = str(backup_db(db_path))
        with con:
            con.executemany("UPDATE label_events SET note=? WHERE id=?", [(note, row_id) for row_id, note in changes])
    con.close()
    return {"db": str(db_path), "exists": True, "updated": len(changes), "backup": backup}


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize prototype note tags in label_records.db")
    parser.add_argument("db", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for db_path in args.db:
        result = normalize_db(db_path, dry_run=args.dry_run)
        print(result)


if __name__ == "__main__":
    main()
