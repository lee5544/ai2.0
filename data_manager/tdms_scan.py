"""Scan TDMS storage and synchronize manifest and generated samples."""


import csv
import argparse
import os
import shutil
import sys
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from data_manager.tdms_registry import TdmsRegistry
from data_manager.sample_generate import SampleGenerator
from data_manager.label_database import LabelDatabase
from data_manager.label_internal_registry import LabelStore
from data_manager.config import load_data_manager_config
from data_manager.line_rules import LINE_RULES, normalize_line_time_value
from data_manager.tdms_read import (
    is_compressed_tdms_path,
    is_tdms_path,
    iter_tdms_files,
    open_tdms,
    tdms_logical_stem,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

COPY_EXIT_NO_TDMS = 2
COPY_EXIT_DISK_INSUFFICIENT = 3
COPY_EXIT_MERGE_CONFIRM_REQUIRED = 4
COPY_EXIT_INVALID_TARGET = 5
NESTED_PROTOTYPE_DIR = "prototype"


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", str(text))


def _normalize_path_obj(path: Path) -> Path:
    return Path(_normalize_text(os.fsdecode(path)))


def _to_posix(path: Path) -> str:
    return _normalize_text(path.as_posix()).replace("\\", "/")


def _is_nested_prototype_relative_path(relative_path: Path | str) -> bool:
    parts = Path(str(relative_path)).parts
    return bool(parts and parts[0] == NESTED_PROTOTYPE_DIR)


def _resolve_runtime_settings(
    *,
    storage_root_arg: str | None,
) -> Tuple[Path, str]:
    dm_cfg = load_data_manager_config()
    data_root_raw = dm_cfg.get("data_root")
    if not data_root_raw:
        raise ValueError("Config missing required field: data_root (checked cfg/core/data_manager.yaml)")
    data_root = _normalize_path_obj(Path(data_root_raw)).expanduser()

    storage_root_raw = (
        storage_root_arg
        if storage_root_arg is not None
        else dm_cfg.get("tdms_storage_root")
        or "factory_raw"
    )
    storage_root = str(storage_root_raw).strip().strip("/")
    if not storage_root:
        storage_root = "factory_raw"

    return data_root, storage_root


def _load_csv_rows_if_exists(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _row_get(row: Dict[str, Any], key: str) -> str:
    """
    兼容 BOM 列名和前后空白列名差异：
    - BOM 列名：\ufeffline
    - 前后空白列名
    """
    if key in row:
        return str(row.get(key) or "")

    bom_key = f"\ufeff{key}"
    if bom_key in row:
        return str(row.get(bom_key) or "")

    for raw_k, raw_v in row.items():
        normalized_k = str(raw_k or "").lstrip("\ufeff").strip()
        if normalized_k == key:
            return str(raw_v or "")

    return ""


def _resolve_line_from_path(relative_path: Path, forced_line: str | None = None) -> str:
    """
    优先使用 CLI 指定 line；否则根据 factory_raw 下的一级目录自动推断。
    例如: epump2/0825/a.tdms -> epump2
    """
    if forced_line:
        return forced_line

    if not relative_path.parts:
        raise ValueError(f"Cannot infer line from empty relative path: {relative_path}")

    inferred_line = str(relative_path.parts[0]).strip()
    if inferred_line not in LINE_RULES:
        raise ValueError(
            f"Inferred line '{inferred_line}' is not defined in LINE_RULES "
            f"(relative_path={relative_path})"
        )
    return inferred_line


def _parse_created_time_sort_value(value: str) -> Tuple[int, str]:
    raw = str(value or "").strip()
    if not raw:
        return (1, "")
    try:
        return (0, datetime.fromisoformat(raw).isoformat())
    except ValueError:
        return (1, raw)


def _normalized_manifest_row(row: Dict[str, Any]) -> Dict[str, str]:
    line = _row_get(row, "line").strip()
    return {
        "line": line,
        "sn": _row_get(row, "sn").strip(),
        "reference": _row_get(row, "reference").strip(),
        "time": normalize_line_time_value(_row_get(row, "time").strip(), line=line),
        "created_time": _row_get(row, "created_time").strip(),
        "tdms_storage_root": _row_get(row, "tdms_storage_root").strip(),
        "relative_path": _normalize_text(_row_get(row, "relative_path").replace("\\", "/")).strip("/"),
    }


def _normalized_sample_row(row: Dict[str, Any]) -> Dict[str, str]:
    return {
        "line": _row_get(row, "line").strip(),
        "sn": _row_get(row, "sn").strip(),
        "sample_id": _row_get(row, "sample_id").strip(),
        "group_name": _row_get(row, "group_name").strip(),
        "channel_name": _row_get(row, "channel_name").strip(),
        "sampling_rate": _row_get(row, "sampling_rate").strip(),
    }


def _same_logical_relative_path(left: str, right: str) -> bool:
    left_path = Path(str(left or ""))
    right_path = Path(str(right or ""))
    return (
        left_path.parent.as_posix() == right_path.parent.as_posix()
        and tdms_logical_stem(left_path.name) == tdms_logical_stem(right_path.name)
    )


def _manifest_row_abs_path(data_root: Path, row: Dict[str, str]) -> Path:
    storage_root = str(row.get("tdms_storage_root", "") or "").strip()
    relative_path = str(row.get("relative_path", "") or "").strip().replace("\\", "/")
    return data_root / storage_root / relative_path


def _resolve_existing_manifest_row_path(
    data_root: Path,
    row: Dict[str, str],
) -> Tuple[bool, Dict[str, str]]:
    normalized = dict(row)
    abs_path = _manifest_row_abs_path(data_root, normalized)
    if abs_path.exists():
        return True, normalized

    relative_path = normalized.get("relative_path", "")
    if relative_path and not is_compressed_tdms_path(relative_path):
        upgraded_relative_path = f"{relative_path}.zst"
        upgraded_abs_path = data_root / normalized["tdms_storage_root"] / upgraded_relative_path
        if upgraded_abs_path.exists():
            normalized["relative_path"] = upgraded_relative_path
            return True, normalized

    return False, normalized


def _merge_existing_row_with_candidate(
    existing_row: Dict[str, str],
    candidate_row: Dict[str, str],
) -> Dict[str, str]:
    merged = dict(existing_row)
    for key in ("line", "sn", "reference", "time", "tdms_storage_root"):
        if candidate_row.get(key):
            merged[key] = candidate_row[key]
    merged["relative_path"] = candidate_row["relative_path"]
    if not merged.get("created_time"):
        merged["created_time"] = candidate_row.get("created_time", "")
    return merged


def _build_existing_sn_registry(
    *,
    data_root: Path,
    storage_root: str,
    rows: List[Dict[str, str]],
    line: str | None = None,
) -> Dict[str, Dict[str, str]]:
    rows_by_sn: Dict[str, List[Dict[str, str]]] = {}
    for raw_row in rows:
        row = _normalized_manifest_row(raw_row)
        if row["tdms_storage_root"] != storage_root:
            continue
        if storage_root == "factory_raw" and _is_nested_prototype_relative_path(row["relative_path"]):
            continue
        if line is not None and row["line"] != str(line).strip():
            continue
        if not row["sn"]:
            continue
        rows_by_sn.setdefault(row["sn"], []).append(row)

    preferred: Dict[str, Dict[str, str]] = {}
    for sn, candidates in rows_by_sn.items():
        sorted_candidates = sorted(
            candidates,
            key=lambda item: (
                _parse_created_time_sort_value(item.get("created_time", "")),
                item.get("relative_path", ""),
            ),
        )
        for candidate in sorted_candidates:
            exists, resolved = _resolve_existing_manifest_row_path(data_root, candidate)
            if exists:
                preferred[sn] = resolved
                break
    return preferred


def _scope_manifest_rows(
    rows: List[Dict[str, Any]],
    *,
    storage_root: str,
    lines: Set[str] | None = None,
) -> List[Dict[str, str]]:
    line_scope = {str(item).strip() for item in (lines or set()) if str(item).strip()}
    scoped_rows: List[Dict[str, str]] = []
    for raw_row in rows:
        row = _normalized_manifest_row(raw_row)
        if row["tdms_storage_root"] != storage_root:
            continue
        if line_scope and row["line"] not in line_scope:
            continue
        scoped_rows.append(row)
    return sorted(
        scoped_rows,
        key=lambda row: (
            row["line"],
            _parse_created_time_sort_value(row.get("created_time", "")),
            row["sn"],
            row["relative_path"],
        ),
    )


def _scope_sample_rows(
    rows: List[Dict[str, Any]],
    *,
    lines: Set[str] | None = None,
) -> List[Dict[str, str]]:
    line_scope = {str(item).strip() for item in (lines or set()) if str(item).strip()}
    scoped_rows: List[Dict[str, str]] = []
    for raw_row in rows:
        row = _normalized_sample_row(raw_row)
        if line_scope and row["line"] not in line_scope:
            continue
        scoped_rows.append(row)
    return sorted(
        scoped_rows,
        key=lambda row: (
            row["line"],
            row["sn"],
            row["sample_id"],
            row["group_name"],
            row["channel_name"],
        ),
    )


def _row_counter(
    rows: List[Dict[str, str]],
    *,
    header: List[str],
) -> Counter[Tuple[str, ...]]:
    return Counter(
        tuple(str(row.get(col, "") or "") for col in header)
        for row in rows
    )


def _count_scope_changes(
    existing_rows: List[Dict[str, str]],
    new_rows: List[Dict[str, str]],
    *,
    header: List[str],
) -> Dict[str, int | bool]:
    existing_counter = _row_counter(existing_rows, header=header)
    new_counter = _row_counter(new_rows, header=header)
    added = int(sum((new_counter - existing_counter).values()))
    removed = int(sum((existing_counter - new_counter).values()))
    return {
        "added": added,
        "removed": removed,
        "changed": bool(added or removed),
    }


def ensure_metadata_files(
    *,
    manifest_path: Path,
    label_records_db_path: Path,
    create_missing: bool = False,
) -> Dict[str, Any]:
    """
    检查 manifest 与 SQLite metadata 是否存在；可选创建缺失文件。
    """
    manifest_path = _normalize_path_obj(manifest_path).expanduser()
    label_records_db_path = _normalize_path_obj(label_records_db_path).expanduser()

    checks = [
        ("tdms_manifest.csv", manifest_path, TdmsRegistry),
        ("label_records.db", label_records_db_path, LabelDatabase),
    ]

    existed = 0
    missing = 0
    created = 0
    invalid = 0

    print("-" * 40)
    print("[META] checking metadata csv files...")
    for name, path, creator in checks:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.is_file():
            existed += 1
            print(f"[META] {name:<18} : exists  | {path}")
            continue

        if path.exists() and not path.is_file():
            invalid += 1
            print(f"[META] {name:<18} : invalid | {path} (not a file)")
            continue

        missing += 1
        if create_missing:
            creator(path)
            created += 1
            print(f"[META] {name:<18} : created | {path}")
        else:
            print(f"[META] {name:<18} : missing | {path}")

    ok = invalid == 0 and (missing == 0 or create_missing)
    print(f"[META] summary             : existed={existed}, missing={missing}, created={created}, invalid={invalid}")
    return {
        "ok": ok,
        "existed": existed,
        "missing": missing,
        "created": created,
        "invalid": invalid,
    }


def _format_bytes(num_bytes: int) -> str:
    value = float(max(0, int(num_bytes)))
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.1f} {units[idx]}"


def _scan_source_folder_stats(src_root: Path) -> Dict[str, int]:
    total_files = 0
    tdms_files = 0
    compressed_tdms_files = 0
    total_bytes = 0
    for p in src_root.rglob("*"):
        if any(part.startswith(".") for part in p.parts) or not p.is_file():
            continue
        total_files += 1
        if is_tdms_path(p):
            tdms_files += 1
            if str(p).lower().endswith(".tdms.zst"):
                compressed_tdms_files += 1
        try:
            total_bytes += int(p.stat().st_size)
        except Exception:
            continue
    return {
        "total_files": total_files,
        "tdms_files": tdms_files,
        "compressed_tdms_files": compressed_tdms_files,
        "total_bytes": total_bytes,
    }


def _available_memory_bytes() -> int | None:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def copy_folder_to_tdms_storage(
    *,
    data_root: Path,
    storage_root: str,
    line: str,
    source_folder: Path,
    allow_merge: bool = False,
    remove_source: bool = False,
) -> int:
    """
    将 source_folder 整包导入到 data_root/storage_root/line/source_folder.name 下。
    默认复制并保留源目录；当 remove_source=True 时，导入后删除源目录（优先移动）。
    要求源目录内至少包含一个 .tdms.zst 文件。
    """
    source_folder = _normalize_path_obj(source_folder).expanduser()
    if not source_folder.exists() or not source_folder.is_dir():
        raise FileNotFoundError(f"source folder does not exist: {source_folder}")
    if line not in LINE_RULES:
        raise ValueError(f"Line '{line}' is not defined in LINE_RULES")

    target_line_root = data_root / storage_root / line
    target_line_root.mkdir(parents=True, exist_ok=True)
    target_root = target_line_root / source_folder.name
    try:
        source_resolved = source_folder.resolve()
        target_resolved = target_root.resolve(strict=False)
        if (
            source_resolved == target_resolved
            or target_resolved in source_resolved.parents
            or source_resolved in target_resolved.parents
        ):
            print("[ERROR] source folder cannot contain target folder")
            return COPY_EXIT_INVALID_TARGET
    except Exception:
        pass

    if target_root.exists() and not target_root.is_dir():
        print(f"[ERROR] target path exists and is not a directory: {target_root}")
        return COPY_EXIT_INVALID_TARGET

    target_exists = target_root.exists()
    existing_tdms_files = 0
    if target_exists:
        existing_tdms_files = sum(
            1
            for p in iter_tdms_files(target_root)
        )

    stats = _scan_source_folder_stats(source_folder)
    total_files = int(stats["total_files"])
    tdms_files = int(stats["tdms_files"])
    compressed_tdms_files = int(stats["compressed_tdms_files"])
    total_bytes = int(stats["total_bytes"])

    print("-" * 40)
    print("[COPY] source_folder        :", source_folder)
    print("[COPY] target_root          :", target_root)
    print("[COPY] source total files   :", total_files)
    print("[COPY] source tdms files    :", tdms_files)
    print("[COPY] source tdms.zst      :", compressed_tdms_files)
    print("[COPY] source size          :", _format_bytes(total_bytes))
    print("[COPY] target exists        :", "yes" if target_exists else "no")
    print("[COPY] preserve source      :", "no" if remove_source else "yes")
    if target_exists:
        print("[COPY] target tdms files   :", existing_tdms_files)

    if compressed_tdms_files == 0:
        print("[ERROR] source folder has no .tdms.zst files")
        return COPY_EXIT_NO_TDMS

    if target_exists and not allow_merge:
        print("[COPY][WARN] found same-name target folder under tdms_storage_root.")
        print("[COPY][ACTION] re-run command with --allow-merge to merge into existing folder.")
        return COPY_EXIT_MERGE_CONFIRM_REQUIRED
    if target_exists and allow_merge:
        print("[COPY] merge mode          : ON (confirmed)")

    target_usage_root = target_line_root if target_line_root.exists() else target_line_root.parent
    same_device = False
    try:
        same_device = int(source_folder.stat().st_dev) == int(target_usage_root.stat().st_dev)
    except Exception:
        same_device = False

    if remove_source and same_device:
        print("[COPY] transfer mode        : move")
        print("[COPY] same filesystem      : yes")
        print("[COPY] disk check           : skipped (rename/move on same filesystem)")
    else:
        if remove_source:
            print("[COPY] transfer mode        : move")
            print("[COPY] same filesystem      : no / unknown")
        usage = shutil.disk_usage(target_usage_root)
        disk_free = int(usage.free)
        disk_total = int(usage.total)
        need_bytes = int(total_bytes * 1.05) + 50 * 1024 * 1024  # +5% and 50MB buffer
        print("[COPY] required size        :", _format_bytes(need_bytes))
        print("[COPY] disk free            :", _format_bytes(disk_free))
        print("[COPY] disk total           :", _format_bytes(disk_total))

        if disk_free < need_bytes:
            print("[ERROR] disk space insufficient for copy/move")
            return COPY_EXIT_DISK_INSUFFICIENT
        if disk_free < int(need_bytes * 1.3):
            print("[WARN] disk space is tight")

    mem_avail = _available_memory_bytes()
    if mem_avail is not None:
        print("[COPY] memory available     :", _format_bytes(mem_avail))
        if mem_avail < 2 * 1024 * 1024 * 1024:
            print("[WARN] memory is tight (< 2GB available)")
    else:
        print("[WARN] memory check skipped (psutil unavailable)")

    if remove_source and not target_exists:
        shutil.move(str(source_folder), str(target_root))
        print("[COPY] moved folder         :", source_folder, "->", target_root)
        print("[COPY] done")
        return 0

    def _cleanup_empty_dirs(root: Path) -> None:
        for path in sorted(
            (p for p in root.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            try:
                path.rmdir()
            except OSError:
                continue
        try:
            root.rmdir()
            print("[COPY] removed source root  :", root)
        except OSError:
            print("[COPY][WARN] source root not empty after move:", root)

    copied = 0
    overwritten = 0
    for path in source_folder.rglob("*"):
        if any(part.startswith(".") for part in path.parts):
            continue
        try:
            rel = path.relative_to(source_folder)
        except ValueError:
            rel = Path(path.name)
        target = target_root / rel

        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            overwritten += 1
        if remove_source:
            shutil.move(str(path), str(target))
        else:
            shutil.copy2(path, target)
        copied += 1

    if remove_source:
        _cleanup_empty_dirs(source_folder)
        print("[COPY] moved files          :", copied)
    else:
        print("[COPY] copied files         :", copied)
    print("[COPY] overwritten files    :", overwritten)
    print("[COPY] done")
    return 0


def _extract_sample_keys_from_tdms(
    *,
    tdms_path: Path,
    line: str,
    sn: str,
    reference: str,
) -> Set[Tuple[str, str, str, str, str]]:
    if line not in LINE_RULES:
        raise ValueError(f"Line '{line}' is not defined in LINE_RULES")

    channel_rules = LINE_RULES[line].get("channels")
    if not channel_rules:
        raise ValueError(f"Channel rules are not defined for line '{line}'")

    if "conditional" in channel_rules:
        rules = SampleGenerator._select_conditional_rules(
            channel_rules["conditional"],
            reference=reference,
        )
    else:
        rules = [channel_rules]

    sample_keys: Set[Tuple[str, str, str, str, str]] = set()
    with open_tdms(tdms_path, mode="read_metadata") as tdms:
        for rule in rules:
            up_group = rule["up_group"]
            down_group = rule["down_group"]
            acc_channel = rule["acc_channel"]
            for direction, group_name in (("up", up_group), ("down", down_group)):
                if group_name not in tdms:
                    continue
                group = tdms[group_name]
                if acc_channel not in group:
                    continue
                sample_id = f"{sn}_{direction}"
                sample_keys.add((line, sn, sample_id, group_name, acc_channel))
    return sample_keys


def _build_manifest_row(*, storage_root: str, relative_path: Path, line: str) -> Dict[str, str]:
    if line not in LINE_RULES:
        raise ValueError(f"Line '{line}' is not defined in LINE_RULES")

    filename_rule = LINE_RULES[line].get("filename")
    if not filename_rule:
        raise ValueError(f"Filename rule is not defined for line '{line}'")

    relative_path_str = _to_posix(_normalize_path_obj(relative_path))
    filename = relative_path.name
    delimiter = filename_rule.get("split", "_")
    parsed = TdmsRegistry.parse_filename(
        filename=filename,
        rule=filename_rule,
        delimiter=delimiter,
    )

    return {
        "line": line,
        "sn": parsed.get("sn", "UNKNOWN_SN"),
        "reference": parsed.get("reference", filename),
        "time": normalize_line_time_value(
            parsed.get("time", "UNKNOWN_TIME"),
            line=line,
            filename_rule=filename_rule,
        ),
        "created_time": datetime.now().isoformat(timespec="seconds"),
        "tdms_storage_root": storage_root,
        "relative_path": relative_path_str,
    }


def _build_sample_rows_from_tdms(
    *,
    tdms_path: Path,
    line: str,
    sn: str,
    reference: str,
    existing_sample_keys: Set[Tuple[str, str, str, str, str]],
) -> List[Dict[str, str]]:
    if line not in LINE_RULES:
        raise ValueError(f"Line '{line}' is not defined in LINE_RULES")

    channel_rules = LINE_RULES[line].get("channels")
    if not channel_rules:
        raise ValueError(f"Channel rules are not defined for line '{line}'")

    if "conditional" in channel_rules:
        rules = SampleGenerator._select_conditional_rules(
            channel_rules["conditional"],
            reference=reference,
        )
    else:
        rules = [channel_rules]

    out_rows: List[Dict[str, str]] = []
    with open_tdms(tdms_path, mode="read_metadata") as tdms:
        for rule in rules:
            up_group = rule["up_group"]
            down_group = rule["down_group"]
            acc_channel = rule["acc_channel"]

            for direction, group_name in (("up", up_group), ("down", down_group)):
                sample_id = f"{sn}_{direction}"
                sample_key = (line, sn, sample_id, group_name, acc_channel)
                if sample_key in existing_sample_keys:
                    continue

                if group_name not in tdms:
                    continue

                group = tdms[group_name]
                if acc_channel not in group:
                    continue

                channel = group[acc_channel]
                sampling_rate = channel.properties.get("wf_increment")
                if sampling_rate is not None:
                    sampling_rate = int(1 / sampling_rate)

                existing_sample_keys.add(sample_key)
                out_rows.append(
                    {
                        "line": line,
                        "sn": sn,
                        "sample_id": sample_id,
                        "group_name": group_name,
                        "channel_name": acc_channel,
                        "sampling_rate": sampling_rate,
                    }
                )
    return out_rows


def scan_factory_raw(
    data_root: Path,
    manifest_path: Path,
    label_records_db_path: Path,
    *,
    line: str | None = None,
    storage_root: str = "factory_raw",
):
    """
    扫描 tdms_storage_root 目录，并在当前 scope 内同步 metadata：
    1. 对比 line 目录下当前 TDMS 文件与 manifest / sample_index 的差异
    2. 仅当存在变化时重写 scope 内 metadata
    """

    data_root = _normalize_path_obj(data_root).expanduser()
    manifest_path = _normalize_path_obj(manifest_path).expanduser()
    label_records_db_path = _normalize_path_obj(label_records_db_path).expanduser()

    factory_dir = data_root / storage_root
    if not factory_dir.exists():
        raise FileNotFoundError(f"{factory_dir} does not exist")

    manifest = TdmsRegistry(manifest_path)
    indexer = SampleGenerator(label_records_db_path)

    print("[1/2] Counting TDMS files...")
    if line:
        line = str(line).strip()
        line_dir = factory_dir / line
        if not line_dir.exists():
            raise FileNotFoundError(f"{line_dir} does not exist")
        tdms_files = sorted(iter_tdms_files(line_dir), key=lambda p: p.as_posix())
    else:
        tdms_files = sorted(iter_tdms_files(factory_dir), key=lambda p: p.as_posix())
    if storage_root == "factory_raw":
        tdms_files = [
            path
            for path in tdms_files
            if not _is_nested_prototype_relative_path(path.relative_to(factory_dir))
        ]
    total = len(tdms_files)
    print(f"[1/2] TDMS files found: {total}")
    if total == 0:
        print("[2/2] No TDMS files found in current scope, metadata will be cleared if needed.")
    else:
        print("[2/2] Start scanning TDMS files...")

    existing_manifest_rows = manifest.list_all()
    existing_scope_lines = {
        _row_get(r, "line").strip()
        for r in existing_manifest_rows
        if _row_get(r, "tdms_storage_root").strip() == storage_root
        and _row_get(r, "line").strip()
        and (line is None or _row_get(r, "line").strip() == line)
    }
    existing_by_sn = _build_existing_sn_registry(
        data_root=data_root,
        storage_root=storage_root,
        rows=existing_manifest_rows,
        line=line,
    )

    kept_rows_by_sn: Dict[str, Dict[str, str]] = {}
    sample_batch_rows: List[Dict[str, str]] = []
    duplicate_messages: List[str] = []
    upgraded_existing = 0
    duplicate_skipped = 0
    newly_registered = 0
    current_scope_lines: Set[str] = {line} if line else set()

    use_tqdm = total > 0 and tqdm is not None
    if use_tqdm:
        progress_iter = tqdm(tdms_files, total=total, desc="Scanning TDMS", unit="file")
        log = progress_iter.write
    else:
        progress_iter = tdms_files
        log = print

    for idx, tdms_file in enumerate(progress_iter, start=1):
        if not use_tqdm and (idx == 1 or idx % 50 == 0 or idx == total):
            log(f"[PROGRESS] {idx}/{total}")

        relative_path = tdms_file.relative_to(factory_dir)
        current_line = _resolve_line_from_path(relative_path, forced_line=line)
        candidate_row = _build_manifest_row(
            storage_root=storage_root,
            relative_path=relative_path,
            line=current_line,
        )
        sn = candidate_row["sn"]
        current_scope_lines.add(candidate_row["line"])

        existing_row = existing_by_sn.get(sn)
        kept_row = kept_rows_by_sn.get(sn)
        selected_row = kept_row or existing_row

        if selected_row is None:
            kept_rows_by_sn[sn] = candidate_row
            existing_by_sn[sn] = candidate_row
            newly_registered += 1
            continue

        selected_rel = selected_row["relative_path"]
        candidate_rel = candidate_row["relative_path"]
        if selected_rel == candidate_rel:
            kept_rows_by_sn[sn] = selected_row
            continue

        if (
            _same_logical_relative_path(selected_rel, candidate_rel)
            and not is_compressed_tdms_path(selected_rel)
            and is_compressed_tdms_path(candidate_rel)
        ):
            merged_row = _merge_existing_row_with_candidate(selected_row, candidate_row)
            kept_rows_by_sn[sn] = merged_row
            existing_by_sn[sn] = merged_row
            upgraded_existing += 1
            duplicate_messages.append(
                f"[UPGRADE SN] {sn} : {selected_rel} -> {candidate_rel}"
            )
            continue

        duplicate_skipped += 1
        duplicate_messages.append(
            f"[SKIP SN] {sn} already registered, keep {selected_rel}, skip {candidate_rel}"
        )

    scanned_manifest_batch_rows = sorted(
        kept_rows_by_sn.values(),
        key=lambda row: (
            row["line"],
            _parse_created_time_sort_value(row.get("created_time", "")),
            row["sn"],
            row["relative_path"],
        ),
    )
    existing_sample_keys: Set[Tuple[str, str, str, str, str]] = set()
    for row in scanned_manifest_batch_rows:
        tdms_path = data_root / row["tdms_storage_root"] / row["relative_path"]
        new_sample_rows = _build_sample_rows_from_tdms(
            tdms_path=tdms_path,
            line=row["line"],
            sn=row["sn"],
            reference=row["reference"],
            existing_sample_keys=existing_sample_keys,
        )
        if new_sample_rows:
            sample_batch_rows.extend(new_sample_rows)

    protected_manifest_rows = [
        _normalized_manifest_row(row)
        for row in existing_manifest_rows
        if _row_get(row, "tdms_storage_root").strip() == "factory_raw"
        and _is_nested_prototype_relative_path(_row_get(row, "relative_path"))
        and (line is None or _row_get(row, "line").strip() == line)
    ]
    manifest_batch_rows = sorted(
        [*scanned_manifest_batch_rows, *protected_manifest_rows],
        key=lambda row: (
            row["line"],
            _parse_created_time_sort_value(row.get("created_time", "")),
            row["sn"],
            row["relative_path"],
        ),
    )

    replace_lines = ({line} if line else set()) | existing_scope_lines | current_scope_lines
    existing_manifest_scope_rows = _scope_manifest_rows(
        existing_manifest_rows,
        storage_root=storage_root,
        lines={line} if line else None,
    )
    existing_sample_scope_rows = _scope_sample_rows(
        [
            row
            for row in indexer.database.list_samples()
            if str(row.get("origin", "") or "") == "index"
        ],
        lines=replace_lines,
    )

    manifest_delta = _count_scope_changes(
        existing_manifest_scope_rows,
        manifest_batch_rows,
        header=TdmsRegistry.HEADER,
    )
    sample_delta = _count_scope_changes(
        existing_sample_scope_rows,
        sample_batch_rows,
        header=SampleGenerator.HEADER,
    )
    metadata_changed = bool(manifest_delta["changed"] or sample_delta["changed"])

    if metadata_changed:
        manifest.replace_scope(
            manifest_batch_rows,
            storage_root=storage_root,
            lines={line} if line else None,
        )
        indexer.replace_scope(
            sample_batch_rows,
            lines=replace_lines,
            origins={"index"},
        )

    print("-" * 40)
    print(f"Total TDMS files found : {total}")
    print(f"Registered files       : {len(manifest_batch_rows)}")
    print(f"Newly registered SN    : {newly_registered}")
    print(f"Upgraded to .tdms.zst  : {upgraded_existing}")
    print(f"Duplicate SN skipped   : {duplicate_skipped}")
    print(f"Indexed samples        : {len(sample_batch_rows)}")
    print(f"Manifest delta         : +{int(manifest_delta['added'])} / -{int(manifest_delta['removed'])}")
    print(f"Sample index delta     : +{int(sample_delta['added'])} / -{int(sample_delta['removed'])}")
    print(f"Metadata changed       : {'yes' if metadata_changed else 'no'}")
    print(f"Metadata synced scope  : {', '.join(sorted(replace_lines)) if replace_lines else '(empty)'}")
    print(f"Manifest path          : {manifest_path}")
    print(f"Label records DB      : {label_records_db_path}")
    if not metadata_changed:
        print("Metadata sync skipped  : already up to date")
    if duplicate_messages:
        print("[SN messages]")
        for item in duplicate_messages[:200]:
            print(f"  - {item}")


def check_unregistered_tdms(
    data_root: Path,
    manifest_path: Path,
    label_records_db_path: Path,
    *,
    line: str | None = None,
    storage_root: str = "factory_raw",
    max_show: int = 200,
) -> int:
    """
    检查 factory_raw 下 TDMS 是否缺失于 tdms_manifest.csv / label_records.db。
    返回 0 表示检查通过，返回 1 表示有缺失项。
    """

    data_root = _normalize_path_obj(data_root).expanduser()
    manifest_path = _normalize_path_obj(manifest_path).expanduser()
    label_records_db_path = _normalize_path_obj(label_records_db_path).expanduser()

    factory_dir = data_root / storage_root
    if not factory_dir.exists():
        raise FileNotFoundError(f"{factory_dir} does not exist")

    if line:
        line = str(line).strip()
        line_dir = factory_dir / line
        if not line_dir.exists():
            raise FileNotFoundError(f"{line_dir} does not exist")
        tdms_files = sorted(iter_tdms_files(line_dir), key=lambda p: p.as_posix())
    else:
        tdms_files = sorted(iter_tdms_files(factory_dir), key=lambda p: p.as_posix())
    if storage_root == "factory_raw":
        tdms_files = [
            path
            for path in tdms_files
            if not _is_nested_prototype_relative_path(path.relative_to(factory_dir))
        ]

    manifest_rows = _load_csv_rows_if_exists(manifest_path)
    sample_rows = SampleGenerator(label_records_db_path).list_all()

    manifest_relpaths = {
        _normalize_text(str(r.get("relative_path", "")).replace("\\", "/"))
        for r in manifest_rows
        if str(r.get("tdms_storage_root", "")).strip() == storage_root
    }
    manifest_sns = {
        str(r.get("sn", "")).strip()
        for r in manifest_rows
        if str(r.get("tdms_storage_root", "")).strip() == storage_root
        and str(r.get("sn", "")).strip()
    }
    sample_sns = {
        str(r.get("sn", "")).strip()
        for r in sample_rows
        if str(r.get("sn", "")).strip()
    }

    sample_keys = {
        (
            str(r.get("line", "")),
            str(r.get("sn", "")),
            str(r.get("sample_id", "")),
            str(r.get("group_name", "")),
            str(r.get("channel_name", "")),
        )
        for r in sample_rows
    }

    missing_manifest: List[str] = []
    missing_sample_index: List[str] = []
    sample_parse_failed: List[str] = []
    duplicate_registered: List[str] = []

    use_tqdm = tqdm is not None
    if use_tqdm:
        progress_iter = tqdm(tdms_files, total=len(tdms_files), desc="Checking TDMS", unit="file")
        log = progress_iter.write
    else:
        progress_iter = tdms_files
        log = print

    for tdms_file in progress_iter:
        relative_path = tdms_file.relative_to(factory_dir)
        relpath_str = _to_posix(_normalize_path_obj(relative_path))

        try:
            current_line = _resolve_line_from_path(relative_path, forced_line=line)
            manifest_row = _build_manifest_row(
                storage_root=storage_root,
                relative_path=relative_path,
                line=current_line,
            )
            expected_keys = _extract_sample_keys_from_tdms(
                tdms_path=tdms_file,
                line=manifest_row["line"],
                sn=manifest_row["sn"],
                reference=manifest_row["reference"],
            )
        except Exception as exc:
            sample_parse_failed.append(f"{relpath_str} | {type(exc).__name__}: {exc}")
            continue

        sn = manifest_row["sn"]
        is_duplicate_registered = relpath_str not in manifest_relpaths and sn in manifest_sns
        if is_duplicate_registered:
            duplicate_registered.append(
                f"{sn} | already registered, keep existing manifest row, skip {relpath_str}"
            )
        elif relpath_str not in manifest_relpaths:
            missing_manifest.append(relpath_str)

        if not expected_keys:
            if not is_duplicate_registered and sn not in sample_sns:
                missing_sample_index.append(relpath_str)
            continue

        # sample_index 不是文件级表，因此按“是否存在任一对应 sample_key”判定覆盖。
        if not any(key in sample_keys for key in expected_keys) and not (is_duplicate_registered and sn in sample_sns):
            missing_sample_index.append(relpath_str)

    both_missing = sorted(set(missing_manifest) & set(missing_sample_index))

    print("-" * 40)
    print(f"Check root             : {factory_dir}")
    print(f"TDMS total             : {len(tdms_files)}")
    print(f"Missing in manifest    : {len(missing_manifest)}")
    print(f"Missing in sample_index: {len(missing_sample_index)}")
    print(f"Missing in both        : {len(both_missing)}")
    print(f"Duplicate SN kept      : {len(duplicate_registered)}")
    print(f"Sample parse failed    : {len(sample_parse_failed)}")
    print(f"Manifest path          : {manifest_path}")
    print(f"Label records DB      : {label_records_db_path}")

    if missing_manifest:
        print(f"\n[MISSING manifest] showing up to {max_show}")
        for rel in sorted(missing_manifest)[:max_show]:
            print(f"  - {rel}")
    if missing_sample_index:
        print(f"\n[MISSING sample_index] showing up to {max_show}")
        for rel in sorted(missing_sample_index)[:max_show]:
            print(f"  - {rel}")
    if both_missing:
        print(f"\n[MISSING both] showing up to {max_show}")
        for rel in both_missing[:max_show]:
            print(f"  - {rel}")
    if duplicate_registered:
        print(f"\n[DUPLICATE SN kept existing] showing up to {max_show}")
        for item in duplicate_registered[:max_show]:
            print(f"  - {item}")
    if sample_parse_failed:
        print(f"\n[PARSE FAILED] showing up to {max_show}")
        for item in sample_parse_failed[:max_show]:
            log(f"  - {item}")

    if missing_manifest or missing_sample_index or sample_parse_failed:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize TDMS metadata by scanning a line folder or importing a folder first."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="兼容参数（已弃用）：registered_tdms 现从 cfg/core/data_manager.yaml 读取 data_root/tdms_storage_root。",
    )
    parser.add_argument(
        "--storage-root",
        type=str,
        default=None,
        help="TDMS storage root under data_root (default from cfg/core/data_manager.yaml, fallback factory_raw).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["scan", "import"],
        default=None,
        help="scan: sync tdms_storage_root/<line>/ metadata; import: copy/move a folder into tdms_storage_root/<line>/ then sync.",
    )
    parser.add_argument(
        "--line",
        type=str,
        default=None,
        help="Target line name under tdms_storage_root. Web mode will always pass this value.",
    )
    parser.add_argument(
        "--source-folder",
        type=Path,
        default=None,
        help="Folder to import into data_root/tdms_storage_root/<line>/ (requires at least one .tdms.zst).",
    )
    parser.add_argument(
        "--copy-folder",
        type=Path,
        default=None,
        help="Deprecated alias of --source-folder.",
    )
    parser.add_argument(
        "--allow-merge",
        action="store_true",
        help="Allow merging into existing same-name folder when using import mode.",
    )
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="Delete the original source folder after import; move when possible.",
    )
    parser.add_argument(
        "--zstd-level",
        type=int,
        default=None,
        help="Deprecated and ignored.",
    )
    parser.add_argument(
        "--zstd-threads",
        type=int,
        default=None,
        help="Deprecated and ignored.",
    )
    parser.add_argument(
        "--zstd-verify",
        type=str,
        choices=["none", "size", "sha1"],
        default=None,
        help="Deprecated and ignored.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to tdms_manifest.csv",
    )
    parser.add_argument(
        "--check-missing",
        action="store_true",
        help="Check whether TDMS files under data_root/tdms_storage_root are missing in manifest/label_records.db.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only run missing-file check; do not scan/write metadata.",
    )
    parser.add_argument(
        "--check-meta",
        action="store_true",
        help="Only check whether metadata stores exist under data_root/metadata.",
    )
    parser.add_argument(
        "--init-meta",
        action="store_true",
        help="Create missing metadata stores (tdms_manifest.csv/label_records.db).",
    )
    parser.add_argument(
        "--max-show",
        type=int,
        default=200,
        help="Maximum rows to print for each missing list (default: 200).",
    )
    args = parser.parse_args()
    if args.config is not None:
        print(
            f"[WARN] --config 已弃用且不会用于 data_root/tdms_storage_root 解析: {args.config}"
        )
    if args.zstd_level is not None or args.zstd_threads is not None or args.zstd_verify is not None:
        print("[WARN] --zstd-level/--zstd-threads/--zstd-verify 已弃用，当前版本会忽略这些参数。")

    data_root, storage_root = _resolve_runtime_settings(
        storage_root_arg=args.storage_root,
    )
    source_folder = args.source_folder if args.source_folder is not None else args.copy_folder
    requested_mode = str(args.mode or ("import" if source_folder is not None else "scan")).strip().lower()

    metadata_dir = data_root / "metadata"

    manifest_path = (
        args.manifest
        if args.manifest
        else metadata_dir / "tdms_manifest.csv"
    )

    label_records_db_path = metadata_dir / "label_records.db"

    print(f"data_root              : {data_root}")
    print(f"tdms_storage_root      : {storage_root}")
    print(f"manifest_path          : {manifest_path}")
    print(f"label_records_db_path : {label_records_db_path}")

    meta_status = ensure_metadata_files(
        manifest_path=manifest_path,
        label_records_db_path=label_records_db_path,
        create_missing=bool(args.init_meta),
    )
    if args.check_meta:
        raise SystemExit(0 if meta_status.get("ok") else 1)

    if args.check_only:
        exit_code = check_unregistered_tdms(
            data_root=data_root,
            manifest_path=manifest_path,
            label_records_db_path=label_records_db_path,
            line=args.line,
            storage_root=storage_root,
            max_show=max(1, int(args.max_show)),
        )
        raise SystemExit(exit_code)

    if requested_mode == "import":
        if source_folder is None:
            print("[ERROR] import mode requires --source-folder (or --copy-folder)")
            raise SystemExit(2)
        if not args.line:
            print("[ERROR] import mode requires --line")
            raise SystemExit(2)
        copy_exit = copy_folder_to_tdms_storage(
            data_root=data_root,
            storage_root=storage_root,
            line=str(args.line).strip(),
            source_folder=source_folder,
            allow_merge=bool(args.allow_merge),
            remove_source=bool(args.remove_source),
        )
        if copy_exit != 0:
            raise SystemExit(copy_exit)

    if requested_mode == "scan" and not args.line:
        print("[WARN] scan mode 未指定 --line，将扫描整个 tdms_storage_root 并同步该 scope 的 metadata。")

    scan_factory_raw(
        data_root=data_root,
        manifest_path=manifest_path,
        label_records_db_path=label_records_db_path,
        line=args.line,
        storage_root=storage_root,
    )

    if args.check_missing:
        exit_code = check_unregistered_tdms(
            data_root=data_root,
            manifest_path=manifest_path,
            label_records_db_path=label_records_db_path,
            line=args.line,
            storage_root=storage_root,
            max_show=max(1, int(args.max_show)),
        )
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
