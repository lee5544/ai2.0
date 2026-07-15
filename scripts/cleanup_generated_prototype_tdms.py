#!/usr/bin/env python3
"""Clean prototype TDMS files whose names were generated as line__sn__ref__original.

The desired layout keeps the original TDMS filename under:
  data_root/factory_raw/<line>/prototype/<reason>/<original>.tdms[.zst]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path


VALID_SUFFIXES = (".tdms.zst", ".tdms", ".tdms.szt")


def _is_tdms(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and not path.name.startswith("._") and name.endswith(VALID_SUFFIXES)


def _generated_original_name(path: Path) -> str:
    parts = path.name.split("__", 3)
    if len(parts) != 4:
        return ""
    if not parts[0] or not parts[1] or not parts[2] or not parts[3]:
        return ""
    return parts[3]


def _backup(data_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = data_root / "metadata" / "backups" / f"cleanup_generated_prototype_tdms_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    manifest = data_root / "metadata" / "tdms_manifest.csv"
    db = data_root / "metadata" / "label_records.db"
    if manifest.exists():
        shutil.copy2(manifest, backup_dir / "tdms_manifest.csv")
    if db.exists():
        with sqlite3.connect(db) as source, sqlite3.connect(backup_dir / "label_records.db") as target:
            source.backup(target)
    return backup_dir


def _prototype_generated_files(data_root: Path) -> list[Path]:
    root = data_root / "factory_raw"
    out = []
    for path in root.rglob("*"):
        if _is_tdms(path) and "prototype" in path.parts and _generated_original_name(path):
            out.append(path)
    return sorted(out)


def _clean_manifest(data_root: Path, generated_names: set[str], *, execute: bool) -> int:
    manifest = data_root / "metadata" / "tdms_manifest.csv"
    if not manifest.exists() or not generated_names:
        return 0
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    def is_generated_row(row: dict[str, str]) -> bool:
        for key in ("relative_path", "tdms_path"):
            name = Path(str(row.get(key, "") or "").replace("\\", "/")).name
            if name in generated_names:
                return True
        return False

    kept = [row for row in rows if not is_generated_row(row)]
    removed = len(rows) - len(kept)
    if execute and removed:
        tmp = manifest.with_suffix(manifest.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in kept:
                writer.writerow({c: row.get(c, "") for c in fieldnames})
        tmp.replace(manifest)
    return removed


def _move_files(data_root: Path, backup_dir: Path, *, execute: bool) -> dict:
    files = _prototype_generated_files(data_root)
    moved = []
    duplicates = []
    errors = []
    for source in files:
        original_name = _generated_original_name(source)
        target = source.with_name(original_name)
        try:
            if not execute:
                moved.append({"from": str(source), "to": str(target), "mode": "plan"})
            elif target.exists():
                rel = source.relative_to(data_root / "factory_raw")
                backup_target = backup_dir / "generated_files" / rel
                backup_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(backup_target))
                duplicates.append({"from": str(source), "kept": str(target), "moved_to_backup": str(backup_target)})
            else:
                source.rename(target)
                moved.append({"from": str(source), "to": str(target), "mode": "rename"})
        except Exception as exc:
            errors.append({"path": str(source), "error": f"{type(exc).__name__}: {exc}"})
    return {"found": len(files), "moved": moved, "duplicates": duplicates, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/Volumes/18555440521/fault/data_root"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-register", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    backup_dir = _backup(data_root) if args.execute else Path("")
    generated_files = _prototype_generated_files(data_root)
    generated_names = {p.name for p in generated_files}
    file_stats = _move_files(data_root, backup_dir, execute=args.execute)
    manifest_removed = _clean_manifest(data_root, generated_names, execute=args.execute)

    register = {"skipped": True}
    if args.execute and not args.skip_register:
        cmd = [sys.executable, "scripts/register_factory_raw_prototypes.py", "--data-root", str(data_root), "--execute"]
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
        register = {
            "skipped": False,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-4000:],
        }

    print(json.dumps({
        "execute": bool(args.execute),
        "data_root": str(data_root),
        "backup_dir": str(backup_dir),
        "generated_files_before": len(generated_files),
        "files": file_stats,
        "manifest_generated_rows_removed": manifest_removed,
        "register": register,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
