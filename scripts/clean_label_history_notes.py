#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_manager.label_internal_registry import LabelStore  # noqa: E402

DEFAULT_DB = Path("/Volumes/18555440521/fault/data_root/metadata/label_records.db")
USELESS_NOTE_PATTERN = re.compile(
    r"merged_|import_marker|group|factory_source",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clear technical metadata from SQLite label event notes without deleting rows."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = args.db.expanduser()
    store = LabelStore(db_path)
    rows = store.list_all()
    changed = 0
    for row in rows:
        if USELESS_NOTE_PATTERN.search(str(row.get("note") or "")):
            row["note"] = ""
            changed += 1
    print(f"rows={len(rows)}")
    if not args.execute:
        print(f"matched_note_cells={changed}")
        print("dry-run only; pass --execute to update label_records.db")
        return
    store._write_all(rows)
    print(f"cleared_note_cells={changed}")
    print(f"updated={db_path}")


if __name__ == "__main__":
    main()
