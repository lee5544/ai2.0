#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path


SOURCE_ROOT = Path("/Volumes/18555440521/fault/data_root/factory_raw/epump2/再标记")
FOLDERS = [
    "震颤expert",
    "震颤不一致",
    "摩擦accexpert",
    "摩擦acc不一致",
]
TWO_GB = 2_000_000_000


@dataclass(frozen=True)
class ZipPlan:
    folder: str
    part: int
    files: tuple[Path, ...]
    raw_size: int
    zip_path: Path


def iter_data_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and not path.name.startswith("._")
        and (path.name.endswith(".tdms.zst") or path.name.endswith(".tdms"))
    )


def plan_folder(folder_name: str, max_bytes: int) -> list[ZipPlan]:
    folder = SOURCE_ROOT / folder_name
    files = iter_data_files(folder)
    if not files:
        return [
            ZipPlan(
                folder=folder_name,
                part=1,
                files=tuple(),
                raw_size=0,
                zip_path=SOURCE_ROOT / f"{folder_name}.zip",
            )
        ]

    chunks: list[list[Path]] = [[]]
    chunk_size = 0
    for path in files:
        size = path.stat().st_size
        # Keep an oversized single source file copyable rather than failing.
        if chunks[-1] and chunk_size + size > max_bytes:
            chunks.append([])
            chunk_size = 0
        chunks[-1].append(path)
        chunk_size += size

    plans: list[ZipPlan] = []
    for idx, chunk in enumerate(chunks, start=1):
        zip_name = f"ep2-{folder_name}{idx}.zip"
        plans.append(
            ZipPlan(
                folder=folder_name,
                part=idx,
                files=tuple(chunk),
                raw_size=sum(path.stat().st_size for path in chunk),
                zip_path=SOURCE_ROOT / zip_name,
            )
        )
    return plans


def write_zip(plan: ZipPlan, overwrite: bool) -> None:
    if plan.zip_path.exists() and not overwrite:
        raise FileExistsError(f"exists: {plan.zip_path}")
    with zipfile.ZipFile(
        plan.zip_path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for path in plan.files:
            arcname = str(Path(plan.folder) / path.name)
            archive.write(path, arcname=arcname)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zip epump2 relabel folders.")
    parser.add_argument("--execute", action="store_true", help="Create zip files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing zip files.")
    parser.add_argument("--max-bytes", type=int, default=TWO_GB, help="Split zip parts at this raw-byte threshold.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_plans: list[ZipPlan] = []
    for folder in FOLDERS:
        all_plans.extend(plan_folder(folder, max_bytes=args.max_bytes))

    print("| folder | zip | files | raw_gb | raw_gib |")
    print("|---|---|---:|---:|---:|")
    for plan in all_plans:
        print(
            f"| {plan.folder} | {plan.zip_path.name} | {len(plan.files)} | "
            f"{plan.raw_size / 1000**3:.2f} | {plan.raw_size / 1024**3:.2f} |"
        )

    if args.execute:
        for plan in all_plans:
            write_zip(plan, overwrite=args.overwrite)
        print("zip_done=1")
    else:
        print("zip_done=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
