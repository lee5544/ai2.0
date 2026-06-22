from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_manager.tdms_read import (
    compress_tdms_file,
    is_compressed_tdms_path,
    is_uncompressed_tdms_path,
    iter_uncompressed_tdms_files,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress .tdms files to .tdms.zst")
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more .tdms files or directories to scan recursively",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=22,
        help="zstd compression level (default: 22)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Threads per compression job, 0 means auto",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of TDMS files to compress concurrently (default: 1)",
    )
    parser.add_argument(
        "--verify",
        choices=("none", "size", "sha1"),
        default="sha1",
        help="Verification mode before removing source (default: sha1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .tdms.zst target file",
    )
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="Delete original .tdms after successful compression",
    )
    return parser.parse_args()


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    return f"{value:.1f}{units[unit_idx]}"


def _iter_input_tdms(paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if path.is_file():
            if is_uncompressed_tdms_path(path):
                files.append(path)
            elif is_compressed_tdms_path(path):
                print(f"[SKIP] already compressed: {path}")
            else:
                print(f"[SKIP] unsupported file suffix: {path}")
            continue

        if path.is_dir():
            files.extend(iter_uncompressed_tdms_files(path))
            continue

        print(f"[SKIP] unsupported path type: {path}")
    files = sorted({p.resolve() for p in files})
    return files


class _CompressResult(NamedTuple):
    source_path: Path
    target_path: Path
    raw_size: int
    compressed_size: int


def _resolve_threads_per_job(threads: int, jobs: int) -> int:
    if int(threads) > 0:
        return int(threads)
    cpu_count = int(os.cpu_count() or 1)
    return max(1, cpu_count // max(1, int(jobs)))


def _compress_one(
    source_path: Path,
    *,
    level: int,
    threads: int,
    overwrite: bool,
    remove_source: bool,
    verify: str,
) -> _CompressResult:
    raw_size = int(source_path.stat().st_size)
    target_path = compress_tdms_file(
        source_path,
        level=level,
        threads=threads,
        overwrite=overwrite,
        remove_source=remove_source,
        verify=verify,
    )
    compressed_size = int(target_path.stat().st_size)
    return _CompressResult(
        source_path=source_path,
        target_path=target_path,
        raw_size=raw_size,
        compressed_size=compressed_size,
    )


def main() -> int:
    args = _parse_args()
    tdms_files = _iter_input_tdms(args.paths)
    total = len(tdms_files)
    if total == 0:
        print("No .tdms files found.")
        return 0

    jobs = max(1, int(args.jobs))
    threads_per_job = _resolve_threads_per_job(int(args.threads), jobs)

    print(f"TDMS files to compress : {total}")
    print(f"remove source          : {'yes' if args.remove_source else 'no'}")
    print(f"zstd level             : {args.level}")
    print(f"verify                 : {args.verify}")
    print(f"jobs                   : {jobs}")
    print(f"threads per job        : {threads_per_job}")

    ok_count = 0
    fail_count = 0
    raw_bytes_total = 0
    compressed_bytes_total = 0
    start_ts = time.perf_counter()

    if jobs == 1:
        for idx, source_path in enumerate(tdms_files, start=1):
            print(f"[{idx}/{total}] {source_path}")
            try:
                result = _compress_one(
                    source_path,
                    level=args.level,
                    threads=threads_per_job,
                    overwrite=args.overwrite,
                    remove_source=args.remove_source,
                    verify=args.verify,
                )
                raw_bytes_total += result.raw_size
                compressed_bytes_total += result.compressed_size
                ok_count += 1
                ratio = (result.raw_size / result.compressed_size) if result.compressed_size else 0.0
                saving = (1.0 - (result.compressed_size / result.raw_size)) * 100.0 if result.raw_size else 0.0
                print(
                    f"  -> {result.target_path.name} | {_format_bytes(result.raw_size)} -> {_format_bytes(result.compressed_size)} "
                    f"| {ratio:.2f}x | save {saving:.1f}%"
                )
            except Exception as exc:  # noqa: BLE001
                fail_count += 1
                print(f"  !! {type(exc).__name__}: {exc}")
    else:
        future_to_source: dict[Future[_CompressResult], Path] = {}
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            for source_path in tdms_files:
                future = executor.submit(
                    _compress_one,
                    source_path,
                    level=args.level,
                    threads=threads_per_job,
                    overwrite=args.overwrite,
                    remove_source=args.remove_source,
                    verify=args.verify,
                )
                future_to_source[future] = source_path

            for idx, future in enumerate(as_completed(future_to_source), start=1):
                source_path = future_to_source[future]
                print(f"[{idx}/{total}] {source_path}")
                try:
                    result = future.result()
                    raw_bytes_total += result.raw_size
                    compressed_bytes_total += result.compressed_size
                    ok_count += 1
                    ratio = (result.raw_size / result.compressed_size) if result.compressed_size else 0.0
                    saving = (1.0 - (result.compressed_size / result.raw_size)) * 100.0 if result.raw_size else 0.0
                    print(
                        f"  -> {result.target_path.name} | {_format_bytes(result.raw_size)} -> {_format_bytes(result.compressed_size)} "
                        f"| {ratio:.2f}x | save {saving:.1f}%"
                    )
                except Exception as exc:  # noqa: BLE001
                    fail_count += 1
                    print(f"  !! {type(exc).__name__}: {exc}")

    elapsed = float(time.perf_counter() - start_ts)
    print("-" * 60)
    print(f"success                : {ok_count}")
    print(f"failed                 : {fail_count}")
    print(f"raw bytes              : {_format_bytes(raw_bytes_total)}")
    print(f"compressed bytes       : {_format_bytes(compressed_bytes_total)}")
    if compressed_bytes_total > 0:
        print(f"overall ratio          : {raw_bytes_total / compressed_bytes_total:.2f}x")
        print(f"overall saving         : {(1.0 - compressed_bytes_total / raw_bytes_total) * 100.0:.1f}%")
    print(f"elapsed                : {elapsed:.1f}s")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
