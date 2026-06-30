#!/usr/bin/env python3
"""
统一检查 data_root:
1) 检查 TDMS 重复文件（默认按文件名，可选按内容）
2) 更新 metadata/tdms_manifest.csv 与 metadata/label_records.db
3) 检查 SQLite 标签是否不一致；若存在，输出当前目录 sample_view.csv

默认行为不删除文件；只有显式传 --dedup 才执行去重删除。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import os
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from data_manager.label_internal_registry import LabelStore
from data_manager.sample_generate import SampleGenerator
from data_manager.label_database import LabelDatabase
from data_manager.config import load_data_manager_config
from data_manager.label_internal_registry import label_source_rank
from data_manager.line_rules import normalize_line_time_value
from data_manager.tdms_read import (
    is_compressed_tdms_path,
    iter_tdms_files,
    open_tdms,
    tdms_logical_stem,
)


MANIFEST_HEADER = [
    "line",
    "sn",
    "reference",
    "time",
    "created_time",
    "tdms_storage_root",
    "relative_path",
]

SAMPLE_INDEX_HEADER = [
    "line",
    "sn",
    "sample_id",
    "group_name",
    "channel_name",
    "sampling_rate",
]

SOURCE_PRIORITY = {"expert": 0, "operator": 1, "model": 2}
DEFAULT_CONFIG_PATH = Path("cfg/epump4.yaml")
DEFAULT_SAMPLE_VIEW_OUT = Path("sample_view.csv")
_LINE_RULES_CACHE: Optional[Dict[str, Any]] = None
NESTED_PROTOTYPE_DIR = "prototype"


def _get_line_rules() -> Dict[str, Any]:
    global _LINE_RULES_CACHE
    if _LINE_RULES_CACHE is not None:
        return _LINE_RULES_CACHE
    try:
        from data_manager.line_rules import LINE_RULES as loaded
    except Exception as exc:
        raise RuntimeError(f"加载 line_rules 失败: {type(exc).__name__}: {exc}") from exc
    _LINE_RULES_CACHE = loaded
    return loaded


def _normalize_text(text: object) -> str:
    return unicodedata.normalize("NFC", str(text))


def _normalize_path(path: Path) -> Path:
    return Path(_normalize_text(os.fsdecode(path))).expanduser()


def _norm(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _parse_dt(value: object) -> datetime:
    s = _norm(value)
    if not s:
        return datetime.min
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.min


def _read_yaml_cfg(config_path: Path) -> Dict[str, Any]:
    if yaml is None:
        return {}
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        return {}
    return cfg


def _resolve_runtime_settings(args: argparse.Namespace, cfg: Dict[str, Any]) -> Tuple[Path, str, str]:
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
    try:
        dm_cfg = load_data_manager_config()
    except Exception as exc:
        print(f"[WARN] 读取 data_manager 配置失败，回退默认值: {type(exc).__name__}: {exc}")
        dm_cfg = {}

    data_root_raw = (
        args.data_root
        or dataset_cfg.get("data_root")
        or cfg.get("data_root")
        or dm_cfg.get("data_root")
        or "./data_root"
    )
    data_root = _normalize_path(Path(data_root_raw))

    storage_root_raw = (
        args.storage_root
        or dataset_cfg.get("tdms_storage_root")
        or cfg.get("tdms_storage_root")
        or dm_cfg.get("tdms_storage_root")
        or "factory_raw"
    )
    storage_root = _norm(storage_root_raw).strip("/")
    if not storage_root:
        storage_root = "factory_raw"

    line_override = _norm(args.line)
    return data_root, storage_root, line_override


def _iter_tdms_files(tdms_root: Path) -> List[Path]:
    return [
        path
        for path in iter_tdms_files(tdms_root)
        if not _is_nested_prototype_relative_path(path.relative_to(tdms_root))
    ]


def _is_nested_prototype_relative_path(relative_path: Path | str) -> bool:
    parts = Path(str(relative_path)).parts
    return bool(parts and parts[0] == NESTED_PROTOTYPE_DIR)


def _group_by_filename(tdms_files: Iterable[Path]) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    for p in tdms_files:
        grouped[_normalize_text(p.name)].append(p)
    return grouped


def _file_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _group_by_content(tdms_files: List[Path]) -> Dict[Tuple[int, str], List[Path]]:
    size_bucket: Dict[int, List[Path]] = defaultdict(list)
    for p in tdms_files:
        try:
            size_bucket[p.stat().st_size].append(p)
        except OSError:
            continue

    candidates = [paths for paths in size_bucket.values() if len(paths) > 1]
    out: Dict[Tuple[int, str], List[Path]] = defaultdict(list)
    if not candidates:
        return out

    iterable = candidates
    if tqdm is not None:
        iterable = tqdm(candidates, desc="Hashing duplicate-size groups", unit="group")

    for paths in iterable:
        for p in paths:
            try:
                digest = _file_sha1(p)
                size = p.stat().st_size
            except OSError:
                continue
            out[(size, digest)].append(p)
    return out


def _build_duplicate_groups(groups: Iterable[Tuple[str, List[Path]]]) -> List[Tuple[str, List[Path]]]:
    rows = [(k, sorted(v, key=lambda p: p.as_posix())) for k, v in groups if len(v) > 1]
    rows.sort(key=lambda kv: len(kv[1]), reverse=True)
    return rows


def _print_duplicate_groups(title: str, groups: List[Tuple[str, List[Path]]], top_n: int) -> int:
    print(f"\n{title}")
    print(f"重复组数量: {len(groups)}")
    if not groups:
        print("  未发现重复。")
        return 0

    for idx, (key, paths) in enumerate(groups[: max(1, top_n)], start=1):
        print(f"  [{idx}] key={key} count={len(paths)}")
        for p in paths:
            print(f"      - {p}")
    if len(groups) > top_n:
        print(f"  ... 其余 {len(groups) - top_n} 组未展示")
    return sum(len(paths) for _, paths in groups)


def _file_modify_ts(path: Path) -> float:
    try:
        st = path.stat()
    except OSError:
        # 无法读取时间时视为“更晚”，优先被删除。
        return float("inf")
    return float(st.st_mtime)


def _format_ts(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "N/A"


def _choose_keep_and_delete(paths: List[Path]) -> Tuple[Path, List[Path]]:
    sorted_paths = sorted(
        paths,
        key=lambda p: (_file_modify_ts(p), p.as_posix()),
    )
    return sorted_paths[0], sorted_paths[1:]


def _deduplicate_groups(groups: List[Tuple[str, List[Path]]], *, dry_run: bool) -> Tuple[int, int]:
    if not groups:
        print("\n[Dedup] 无重复文件，无需处理。")
        return 0, 0

    deleted = 0
    failed = 0
    mode_text = "DRY-RUN" if dry_run else "APPLY"
    print(f"\n[Dedup - {mode_text}] 保留文件修改时间最早的文件（删除修改时间更新的重复文件）")

    for idx, (key, paths) in enumerate(groups, start=1):
        keep, to_delete = _choose_keep_and_delete(paths)
        print(
            f"  group[{idx}] key={key} keep={keep} "
            f"(mtime={_format_ts(_file_modify_ts(keep))}) delete={len(to_delete)}"
        )
        for p in to_delete:
            if dry_run:
                print(f"    [DRY-RUN] delete: {p}")
                continue
            try:
                p.unlink()
                deleted += 1
                print(f"    [DELETE] {p}")
            except OSError as exc:
                failed += 1
                print(f"    [FAIL] {p} | {exc}")
    return deleted, failed


def _build_duplicate_sn_groups(
    *,
    tdms_files: List[Path],
    tdms_root: Path,
    data_root: Path,
    storage_root: str,
    line_override: str,
    old_manifest_rows: Iterable[Dict[str, object]],
) -> Tuple[List[Tuple[str, Path, List[Path]]], List[str]]:
    existing_by_sn = _build_existing_sn_registry(
        data_root=data_root,
        storage_root=storage_root,
        rows=old_manifest_rows,
    )

    kept_rows_by_sn: Dict[str, Dict[str, str]] = {}
    kept_paths_by_sn: Dict[str, Path] = {}
    delete_paths_by_sn: Dict[str, List[Path]] = defaultdict(list)
    parse_errors: List[str] = []

    for tdms_path in sorted(tdms_files, key=lambda p: p.as_posix()):
        relative_path = tdms_path.relative_to(tdms_root)
        try:
            line = _resolve_line_from_relative_path(relative_path, line_override)
            candidate_row = _build_manifest_row(
                tdms_path=tdms_path,
                relative_path=relative_path,
                storage_root=storage_root,
                line=line,
            )
            sn = str(candidate_row["sn"]).strip()
            if not sn:
                continue

            selected_row = kept_rows_by_sn.get(sn) or existing_by_sn.get(sn)
            if selected_row is None:
                kept_rows_by_sn[sn] = dict(candidate_row)
                kept_paths_by_sn[sn] = tdms_path
                continue

            keep_path = kept_paths_by_sn.get(sn)
            if keep_path is None:
                keep_path = _manifest_row_abs_path(data_root, selected_row)
                kept_paths_by_sn[sn] = keep_path

            if tdms_path == keep_path:
                kept_rows_by_sn[sn] = dict(selected_row)
                continue

            selected_rel = str(selected_row.get("relative_path", "") or "")
            candidate_rel = str(candidate_row.get("relative_path", "") or "")
            same_logical = _same_logical_relative_path(selected_rel, candidate_rel)
            candidate_is_compressed = is_compressed_tdms_path(candidate_rel)
            selected_is_compressed = is_compressed_tdms_path(selected_rel)

            if same_logical and candidate_is_compressed and not selected_is_compressed:
                if keep_path != tdms_path and keep_path.exists():
                    delete_paths_by_sn[sn].append(keep_path)
                kept_rows_by_sn[sn] = dict(candidate_row)
                kept_paths_by_sn[sn] = tdms_path
                continue

            if same_logical and selected_is_compressed and not candidate_is_compressed:
                delete_paths_by_sn[sn].append(tdms_path)
                continue

            delete_paths_by_sn[sn].append(tdms_path)
        except Exception as exc:
            parse_errors.append(f"{relative_path.as_posix()} | {type(exc).__name__}: {exc}")

    groups: List[Tuple[str, Path, List[Path]]] = []
    for sn, delete_paths in delete_paths_by_sn.items():
        uniq_delete_paths = sorted({p for p in delete_paths}, key=lambda p: p.as_posix())
        if not uniq_delete_paths:
            continue
        keep_path = kept_paths_by_sn.get(sn)
        if keep_path is None:
            continue
        groups.append((sn, keep_path, uniq_delete_paths))
    groups.sort(key=lambda item: (len(item[2]), item[0]), reverse=True)
    return groups, parse_errors


def _print_duplicate_sn_groups(
    title: str,
    groups: List[Tuple[str, Path, List[Path]]],
    top_n: int,
) -> int:
    print(f"\n{title}")
    print(f"重复 SN 数量: {len(groups)}")
    if not groups:
        print("  未发现重复 SN。")
        return 0

    for idx, (sn, keep_path, delete_paths) in enumerate(groups[: max(1, top_n)], start=1):
        print(f"  [{idx}] sn={sn} keep={keep_path} delete={len(delete_paths)}")
        for path in delete_paths:
            print(f"      - {path}")
    if len(groups) > top_n:
        print(f"  ... 其余 {len(groups) - top_n} 组未展示")
    return sum(len(delete_paths) for _, _, delete_paths in groups)


def _cleanup_empty_parent_dirs(paths: Iterable[Path], *, stop_root: Path) -> None:
    stop_root = stop_root.resolve()
    seen: Set[Path] = set()
    for path in paths:
        current = path.parent
        while True:
            try:
                resolved = current.resolve()
            except OSError:
                break
            if resolved == stop_root or resolved in seen:
                break
            seen.add(resolved)
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent


def _deduplicate_sn_groups(
    groups: List[Tuple[str, Path, List[Path]]],
    *,
    dry_run: bool,
    tdms_root: Path,
) -> Tuple[int, int]:
    if not groups:
        print("\n[Dedup by SN] 无重复 SN 文件，无需处理。")
        return 0, 0

    deleted = 0
    failed = 0
    deleted_paths: List[Path] = []
    mode_text = "DRY-RUN" if dry_run else "APPLY"
    print(f"\n[Dedup by SN - {mode_text}] 保留已注册/既有 SN 文件，删除其余重复物理文件")

    for idx, (sn, keep_path, delete_paths) in enumerate(groups, start=1):
        print(f"  group[{idx}] sn={sn} keep={keep_path} delete={len(delete_paths)}")
        for path in delete_paths:
            if dry_run:
                print(f"    [DRY-RUN] delete: {path}")
                continue
            try:
                path.unlink()
                deleted += 1
                deleted_paths.append(path)
                print(f"    [DELETE] {path}")
            except OSError as exc:
                failed += 1
                print(f"    [FAIL] {path} | {exc}")

    if not dry_run and deleted_paths:
        _cleanup_empty_parent_dirs(deleted_paths, stop_root=tdms_root)

    return deleted, failed


def _resolve_line_from_relative_path(relative_path: Path, forced_line: str) -> str:
    line_rules = _get_line_rules()
    if forced_line:
        if forced_line not in line_rules:
            raise ValueError(f"line='{forced_line}' 不在 LINE_RULES 中")
        return forced_line

    if not relative_path.parts:
        raise ValueError(f"无法从空 relative_path 推断 line: {relative_path}")
    inferred = _norm(relative_path.parts[0])
    if inferred not in line_rules:
        raise ValueError(f"推断 line='{inferred}' 不在 LINE_RULES 中: {relative_path}")
    return inferred


def _parse_filename(filename: str, *, rule: Dict[str, Any], delimiter: str = "_") -> Dict[str, str]:
    stem = tdms_logical_stem(filename)
    generated_parts = stem.split("__", 3)
    if len(generated_parts) == 4 and generated_parts[0] in _get_line_rules():
        original = generated_parts[3]
        original_parsed = _parse_filename(original, rule=rule, delimiter=delimiter)
        return {
            "sn": generated_parts[1],
            "reference": generated_parts[2],
            "time": original_parsed.get("time", ""),
        }
    parts = stem.split(delimiter)

    def _get(index_spec: object) -> str:
        if isinstance(index_spec, int):
            return parts[index_spec] if index_spec < len(parts) else "UNKNOWN"
        if isinstance(index_spec, (list, tuple)):
            vals = [parts[i] for i in index_spec if isinstance(i, int) and i < len(parts)]
            return "_".join(vals) if vals else "UNKNOWN"
        return "UNKNOWN"

    return {
        "sn": _get(rule.get("sn_index")),
        "reference": _get(rule.get("reference_index")),
        "time": _get(rule.get("time_index")),
    }


def _select_conditional_rules(rules: List[Dict[str, Any]], *, reference: str) -> List[Dict[str, Any]]:
    for rule in rules:
        cond = rule.get("when", {})
        if cond.get("reference") == reference:
            return [rule]
    for rule in rules:
        cond = rule.get("when", {})
        if cond.get("reference") == "*":
            return [rule]
    raise ValueError(f"找不到 reference='{reference}' 的 conditional 规则")


def _build_manifest_row(
    *,
    tdms_path: Path,
    relative_path: Path,
    storage_root: str,
    line: str,
) -> Dict[str, str]:
    line_rule = _get_line_rules().get(line) or {}
    filename_rule = line_rule.get("filename")
    if not filename_rule:
        raise ValueError(f"line='{line}' 未配置 filename 规则")

    parsed = _parse_filename(
        tdms_path.name,
        rule=filename_rule,
        delimiter=_norm(filename_rule.get("split")) or "_",
    )
    return {
        "line": line,
        "sn": parsed.get("sn", "UNKNOWN_SN"),
        "reference": parsed.get("reference", tdms_path.name),
        "time": normalize_line_time_value(
            parsed.get("time", "UNKNOWN_TIME"),
            line=line,
            filename_rule=filename_rule,
        ),
        "created_time": datetime.now().isoformat(timespec="seconds"),
        "tdms_storage_root": storage_root,
        "relative_path": relative_path.as_posix(),
    }


def _build_sample_rows(
    *,
    tdms_path: Path,
    line: str,
    sn: str,
    reference: str,
    sample_time: str = "",
    tdms_storage_root: str = "",
    relative_path: str = "",
    existing_sample_keys: Optional[Set[Tuple[str, str, str, str, str]]] = None,
) -> List[Dict[str, object]]:
    line_rule = _get_line_rules().get(line) or {}
    channels = line_rule.get("channels")
    if not channels:
        raise ValueError(f"line='{line}' 未配置 channels 规则")
    rules: List[Dict[str, Any]]
    if "conditional" in channels:
        rules = _select_conditional_rules(channels["conditional"], reference=reference)
    else:
        rules = [channels]

    with open_tdms(tdms_path, mode="read_metadata") as tdms:
        out: List[Dict[str, object]] = []
        for rule in rules:
            up_group = _norm(rule.get("up_group"))
            down_group = _norm(rule.get("down_group"))
            acc_channel = _norm(rule.get("acc_channel"))
            if not up_group or not down_group or not acc_channel:
                continue

            for direction, group_name in (("up", up_group), ("down", down_group)):
                if group_name not in tdms:
                    continue
                group = tdms[group_name]
                if acc_channel not in group:
                    continue

                sample_id = f"{sn}_{direction}"
                sample_key = (line, sn, sample_id, group_name, acc_channel)
                if existing_sample_keys is not None and sample_key in existing_sample_keys:
                    continue

                channel = group[acc_channel]
                sampling_rate: object = ""
                wf_increment = channel.properties.get("wf_increment")
                if wf_increment is not None:
                    try:
                        wf = float(wf_increment)
                        if wf > 0:
                            sampling_rate = int(round(1.0 / wf))
                    except Exception:
                        sampling_rate = ""

                if existing_sample_keys is not None:
                    existing_sample_keys.add(sample_key)
                out.append(
                    {
                        "line": line,
                        "sn": sn,
                        "sample_id": sample_id,
                        "group_name": group_name,
                        "channel_name": acc_channel,
                        "sampling_rate": sampling_rate,
                        "reference": reference,
                        "time": sample_time,
                        "tdms_storage_root": tdms_storage_root,
                        "relative_path": relative_path,
                        "tdms_path": str(tdms_path.resolve()),
                    }
                )
        return out


def _recommended_workers(workers: int) -> int:
    if workers > 0:
        return workers
    cpu = os.cpu_count() or 1
    return max(1, min(8, cpu))


def _write_csv(path: Path, header: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with path.open("r", newline="", encoding=enc, errors="replace") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    return []
                return [dict(r) for r in reader]
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"读取 CSV 失败: {path} | {type(last_err).__name__}: {last_err}")


def _created_time_sort_key(row: Dict[str, object]) -> Tuple[int, str]:
    raw = _norm(row.get("created_time"))
    if not raw:
        return (1, "")
    try:
        return (0, datetime.fromisoformat(raw).isoformat())
    except ValueError:
        return (1, raw)


def _normalized_manifest_row(row: Dict[str, object]) -> Dict[str, str]:
    line = _norm(row.get("line"))
    return {
        "line": line,
        "sn": _norm(row.get("sn")),
        "reference": _norm(row.get("reference")),
        "time": normalize_line_time_value(_norm(row.get("time")), line=line),
        "created_time": _norm(row.get("created_time")),
        "tdms_storage_root": _norm(row.get("tdms_storage_root")),
        "relative_path": _norm(row.get("relative_path")).replace("\\", "/").strip("/"),
    }


def _manifest_row_abs_path(data_root: Path, row: Dict[str, str]) -> Path:
    return data_root / row["tdms_storage_root"] / row["relative_path"]


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


def _same_logical_relative_path(left: str, right: str) -> bool:
    left_path = Path(str(left or ""))
    right_path = Path(str(right or ""))
    return (
        left_path.parent.as_posix() == right_path.parent.as_posix()
        and tdms_logical_stem(left_path.name) == tdms_logical_stem(right_path.name)
    )


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
    rows: Iterable[Dict[str, object]],
) -> Dict[str, Dict[str, str]]:
    rows_by_sn: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for raw_row in rows:
        row = _normalized_manifest_row(raw_row)
        if row["tdms_storage_root"] != storage_root:
            continue
        if storage_root == "factory_raw" and _is_nested_prototype_relative_path(row["relative_path"]):
            continue
        if not row["sn"]:
            continue
        rows_by_sn[row["sn"]].append(row)

    preferred: Dict[str, Dict[str, str]] = {}
    for sn, candidates in rows_by_sn.items():
        sorted_candidates = sorted(
            candidates,
            key=lambda item: (_created_time_sort_key(item), item.get("relative_path", "")),
        )
        for candidate in sorted_candidates:
            exists, resolved = _resolve_existing_manifest_row_path(data_root, candidate)
            if exists:
                preferred[sn] = resolved
                break
    return preferred


def _manifest_identity_key(row: Dict[str, object]) -> Tuple[str, str, str, str, str]:
    line = _norm(row.get("line"))
    relative_path = _norm(row.get("relative_path")).replace("\\", "/").strip("/")
    filename = Path(relative_path).name
    return (
        line,
        _norm(row.get("sn")),
        _norm(row.get("reference")),
        normalize_line_time_value(_norm(row.get("time")), line=line),
        filename,
    )


def _relative_parent_folder(relative_path: str) -> str:
    parent = Path(relative_path).parent.as_posix()
    return "" if parent == "." else parent


def _detect_folder_renames(
    old_manifest_rows: Iterable[Dict[str, object]],
    new_manifest_rows: Iterable[Dict[str, object]],
    *,
    storage_root: str,
) -> Dict[str, object]:
    old_map: Dict[Tuple[str, str, str, str, str], List[str]] = defaultdict(list)
    new_map: Dict[Tuple[str, str, str, str, str], List[str]] = defaultdict(list)

    for row in old_manifest_rows:
        if _norm(row.get("tdms_storage_root")) != storage_root:
            continue
        rel = _norm(row.get("relative_path")).replace("\\", "/").strip("/")
        if not rel:
            continue
        old_map[_manifest_identity_key(row)].append(rel)

    for row in new_manifest_rows:
        if _norm(row.get("tdms_storage_root")) != storage_root:
            continue
        rel = _norm(row.get("relative_path")).replace("\\", "/").strip("/")
        if not rel:
            continue
        new_map[_manifest_identity_key(row)].append(rel)

    file_moves: List[Dict[str, str]] = []
    for key in set(old_map.keys()) & set(new_map.keys()):
        old_paths = sorted(set(old_map[key]))
        new_paths = sorted(set(new_map[key]))
        # 仅处理一对一映射，避免重复文件场景误判。
        if len(old_paths) != 1 or len(new_paths) != 1:
            continue

        old_rel = old_paths[0]
        new_rel = new_paths[0]
        if old_rel == new_rel:
            continue
        if Path(old_rel).name != Path(new_rel).name:
            continue

        old_folder = _relative_parent_folder(old_rel)
        new_folder = _relative_parent_folder(new_rel)
        if old_folder == new_folder:
            continue

        file_moves.append(
            {
                "line": key[0],
                "sn": key[1],
                "reference": key[2],
                "time": key[3],
                "filename": key[4],
                "old_relative_path": old_rel,
                "new_relative_path": new_rel,
                "old_folder": old_folder,
                "new_folder": new_folder,
            }
        )

    folder_counter: Dict[Tuple[str, str], int] = defaultdict(int)
    for mv in file_moves:
        folder_counter[(mv["old_folder"], mv["new_folder"])] += 1

    folder_moves = [
        {"old_folder": old, "new_folder": new, "file_count": cnt}
        for (old, new), cnt in folder_counter.items()
    ]
    folder_moves.sort(key=lambda x: int(x["file_count"]), reverse=True)
    file_moves.sort(key=lambda x: (x["old_folder"], x["new_folder"], x["filename"]))
    return {"folder_moves": folder_moves, "file_moves": file_moves}


def _rebuild_metadata(
    *,
    data_root: Path,
    storage_root: str,
    line_override: str,
    dry_run: bool,
    workers: int = 0,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, object]:
    tdms_root = data_root / storage_root
    metadata_dir = data_root / "metadata"
    manifest_path = metadata_dir / "tdms_manifest.csv"
    label_records_db_path = metadata_dir / "label_records.db"
    old_manifest_rows: List[Dict[str, str]] = []

    tdms_files = sorted(_iter_tdms_files(tdms_root), key=lambda p: p.as_posix())
    if line_override:
        tdms_files = [
            p for p in tdms_files
            if p.relative_to(tdms_root).parts and p.relative_to(tdms_root).parts[0] == line_override
        ]
    manifest_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    errors: List[str] = []

    if manifest_path.exists():
        try:
            old_manifest_rows = _read_csv_rows(manifest_path)
        except Exception as exc:
            errors.append(f"读取旧 manifest 失败: {type(exc).__name__}: {exc}")

    protected_manifest_rows = [
        _normalized_manifest_row(row)
        for row in old_manifest_rows
        if _norm(row.get("tdms_storage_root")) == "factory_raw"
        and _is_nested_prototype_relative_path(_norm(row.get("relative_path")))
        and (not line_override or _norm(row.get("line")) == line_override)
    ]

    # 当指定了 --line 时，需要把"其他线"的旧 manifest / sample_index 行原样保留下来，
    # 否则只重建当前 line 的条目会把别的 line 的历史数据冲掉。
    other_line_manifest_rows: List[Dict[str, object]] = []
    other_line_sample_rows: List[Dict[str, object]] = []
    if line_override:
        for raw_row in old_manifest_rows:
            row_line = _norm(raw_row.get("line"))
            if row_line and row_line != line_override:
                other_line_manifest_rows.append(_normalized_manifest_row(raw_row))
        if label_records_db_path.exists():
            try:
                for raw_row in LabelDatabase(label_records_db_path).list_samples():
                    if _norm(raw_row.get("origin")) != "index":
                        continue
                    row_line = _norm(raw_row.get("line"))
                    if row_line and row_line != line_override:
                        other_line_sample_rows.append(dict(raw_row))
            except Exception as exc:
                errors.append(f"读取旧 sample_index 失败: {type(exc).__name__}: {exc}")

    # 当指定 line 时，只让"当前 line"的旧 manifest 行参与 SN 合并/升级，避免跨 line SN
    # 冲突误把另一条线的旧记录当作"已注册"。
    if line_override:
        existing_rows_for_scan = [
            row
            for row in old_manifest_rows
            if _norm(row.get("line")) == line_override
            and not _is_nested_prototype_relative_path(_norm(row.get("relative_path")))
        ]
    else:
        existing_rows_for_scan = [
            row
            for row in old_manifest_rows
            if not _is_nested_prototype_relative_path(_norm(row.get("relative_path")))
        ]

    existing_by_sn = _build_existing_sn_registry(
        data_root=data_root,
        storage_root=storage_root,
        rows=existing_rows_for_scan,
    )
    kept_rows_by_sn: Dict[str, Dict[str, str]] = {}
    duplicate_messages: List[str] = []
    duplicate_skipped = 0
    upgraded_existing = 0

    # 第一步：快速重建 manifest，并准备 sample_index 任务列表。
    manifest_iter: Iterable[Path] = tdms_files
    if tqdm is not None:
        manifest_iter = tqdm(tdms_files, total=len(tdms_files), desc="Rebuilding manifest", unit="file")

    for manifest_processed, tdms_path in enumerate(manifest_iter, start=1):
        relative_path = tdms_path.relative_to(tdms_root)
        try:
            line = _resolve_line_from_relative_path(relative_path, line_override)
            candidate_row = _build_manifest_row(
                tdms_path=tdms_path,
                relative_path=relative_path,
                storage_root=storage_root,
                line=line,
            )
            sn = str(candidate_row["sn"])
            selected_row = kept_rows_by_sn.get(sn) or existing_by_sn.get(sn)

            if selected_row is None:
                kept_rows_by_sn[sn] = dict(candidate_row)
                existing_by_sn[sn] = dict(candidate_row)
                continue

            selected_rel = str(selected_row["relative_path"])
            candidate_rel = str(candidate_row["relative_path"])
            if selected_rel == candidate_rel:
                kept_rows_by_sn[sn] = dict(selected_row)
                continue

            if (
                _same_logical_relative_path(selected_rel, candidate_rel)
                and not is_compressed_tdms_path(selected_rel)
                and is_compressed_tdms_path(candidate_rel)
            ):
                merged_row = _merge_existing_row_with_candidate(
                    dict(selected_row),
                    _normalized_manifest_row(candidate_row),
                )
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
        except Exception as exc:
            errors.append(f"{relative_path.as_posix()} | {type(exc).__name__}: {exc}")
        finally:
            if progress_callback is not None:
                progress_callback("manifest", manifest_processed, len(tdms_files))

    manifest_rows = sorted(
        kept_rows_by_sn.values(),
        key=lambda row: (
            str(row.get("line", "")),
            _created_time_sort_key(row),
            str(row.get("sn", "")),
            str(row.get("relative_path", "")),
        ),
    )
    sample_tasks: List[Tuple[int, Path, str, str, str, str, str, str]] = []
    for idx, row in enumerate(manifest_rows):
        tdms_path = data_root / str(row["tdms_storage_root"]) / str(row["relative_path"])
        sample_tasks.append(
            (
                idx,
                tdms_path,
                str(row["line"]),
                str(row["sn"]),
                str(row["reference"]),
                str(row.get("time") or ""),
                str(row.get("tdms_storage_root") or ""),
                str(row.get("relative_path") or ""),
            )
        )

    # 第二步：并发提取 sample rows，再按原始顺序合并去重，保持稳定输出。
    workers_n = _recommended_workers(int(workers))
    sample_rows_by_idx: Dict[int, List[Dict[str, object]]] = {}
    sample_errors: List[str] = []

    def _task_runner(task: Tuple[int, Path, str, str, str, str, str, str]) -> Tuple[int, List[Dict[str, object]], Optional[str]]:
        idx_i, tdms_path_i, line_i, sn_i, ref_i, time_i, storage_i, relative_i = task
        rel_i = tdms_path_i.relative_to(tdms_root).as_posix()
        try:
            rows_i = _build_sample_rows(
                tdms_path=tdms_path_i,
                line=line_i,
                sn=sn_i,
                reference=ref_i,
                sample_time=time_i,
                tdms_storage_root=storage_i,
                relative_path=relative_i,
                existing_sample_keys=None,
            )
            return idx_i, rows_i, None
        except Exception as exc:
            return idx_i, [], f"{rel_i} | {type(exc).__name__}: {exc}"

    if workers_n <= 1 or len(sample_tasks) <= 1:
        sample_iter: Iterable[Tuple[int, Path, str, str, str, str, str, str]] = sample_tasks
        if tqdm is not None:
            sample_iter = tqdm(sample_tasks, total=len(sample_tasks), desc="Rebuilding sample_index", unit="file")
        for sample_processed, task in enumerate(sample_iter, start=1):
            idx_i, rows_i, err_i = _task_runner(task)
            sample_rows_by_idx[idx_i] = rows_i
            if err_i:
                sample_errors.append(err_i)
            if progress_callback is not None:
                progress_callback("sample_index", sample_processed, len(sample_tasks))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers_n) as ex:
            futures = [ex.submit(_task_runner, task) for task in sample_tasks]
            done_iter: Iterable[concurrent.futures.Future[Tuple[int, List[Dict[str, object]], Optional[str]]]] = (
                concurrent.futures.as_completed(futures)
            )
            if tqdm is not None:
                done_iter = tqdm(done_iter, total=len(futures), desc=f"Rebuilding sample_index({workers_n}w)", unit="file")
            for sample_processed, fut in enumerate(done_iter, start=1):
                idx_i, rows_i, err_i = fut.result()
                sample_rows_by_idx[idx_i] = rows_i
                if err_i:
                    sample_errors.append(err_i)
                if progress_callback is not None:
                    progress_callback("sample_index", sample_processed, len(sample_tasks))

    sample_keys: Set[Tuple[str, str, str, str, str]] = set()
    for idx_i, *_ in sample_tasks:
        for row in sample_rows_by_idx.get(idx_i, []):
            sample_key = (
                _norm(row.get("line")),
                _norm(row.get("sn")),
                _norm(row.get("sample_id")),
                _norm(row.get("group_name")),
                _norm(row.get("channel_name")),
            )
            if sample_key in sample_keys:
                continue
            sample_keys.add(sample_key)
            sample_rows.append(row)

    errors.extend(sample_errors)

    manifest_rows.extend(protected_manifest_rows)
    manifest_rows = sorted(
        manifest_rows,
        key=lambda row: (
            str(row.get("line", "")),
            _created_time_sort_key(row),
            str(row.get("sn", "")),
            str(row.get("relative_path", "")),
        ),
    )

    # 合并"其他线"的旧 manifest / sample_index 行，保证只重建当前 line 时不冲掉别的线。
    if line_override and (other_line_manifest_rows or other_line_sample_rows):
        manifest_rows = list(manifest_rows) + list(other_line_manifest_rows)
        manifest_rows = sorted(
            manifest_rows,
            key=lambda row: (
                str(row.get("line", "")),
                _created_time_sort_key(row),
                str(row.get("sn", "")),
                str(row.get("relative_path", "")),
            ),
        )
        for row in other_line_sample_rows:
            sample_key = (
                _norm(row.get("line")),
                _norm(row.get("sn")),
                _norm(row.get("sample_id")),
                _norm(row.get("group_name")),
                _norm(row.get("channel_name")),
            )
            if sample_key in sample_keys:
                continue
            sample_keys.add(sample_key)
            sample_rows.append(row)
        print(
            f"[INFO] --line='{line_override}' 模式：保留其他 line 的 {len(other_line_manifest_rows)} 条 manifest 行 / "
            f"{len(other_line_sample_rows)} 条 sample_index 行原样写回。"
        )

    rename_report = _detect_folder_renames(
        old_manifest_rows=old_manifest_rows,
        new_manifest_rows=manifest_rows,
        storage_root=storage_root,
    )

    if dry_run:
        print("\n[Metadata - DRY-RUN]")
        print(f"将写入 manifest 行数: {len(manifest_rows)} -> {manifest_path}")
        print(f"将写入 samples 行数: {len(sample_rows)} -> {label_records_db_path}")
        print(f"重复 SN 跳过          : {duplicate_skipped}")
        print(f"升级为 .tdms.zst      : {upgraded_existing}")
        return {
            "manifest_path": manifest_path,
            "label_records_db_path": label_records_db_path,
            "manifest_rows": len(manifest_rows),
            "sample_rows": len(sample_rows),
            "errors": errors,
            "tdms_total": len(tdms_files),
            "rename_report": rename_report,
            "duplicate_sn_skipped": duplicate_skipped,
            "upgraded_existing": upgraded_existing,
            "duplicate_messages": duplicate_messages,
        }

    _write_csv(manifest_path, MANIFEST_HEADER, manifest_rows)
    SampleGenerator(label_records_db_path).replace_scope(sample_rows, origins={"index"})
    return {
        "manifest_path": manifest_path,
        "label_records_db_path": label_records_db_path,
        "manifest_rows": len(manifest_rows),
        "sample_rows": len(sample_rows),
        "errors": errors,
        "tdms_total": len(tdms_files),
        "rename_report": rename_report,
        "duplicate_sn_skipped": duplicate_skipped,
        "upgraded_existing": upgraded_existing,
        "duplicate_messages": duplicate_messages,
    }


def _collect_inconsistent_label_rows(label_rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    variants: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    latest_row: Dict[str, Dict[str, str]] = {}
    latest_key: Dict[str, Tuple[datetime, int]] = {}

    for row in label_rows:
        sample_id = _norm(row.get("sample_id"))
        if not sample_id:
            continue
        result_key = _norm(row.get("result_key"))
        reason_key = _norm(row.get("reason_key"))
        if not result_key and not reason_key:
            continue
        variants[sample_id].add((result_key, reason_key))

        dt = _parse_dt(row.get("timestamp"))
        source_rank = label_source_rank(row.get("source"), SOURCE_PRIORITY)
        # 时间优先；同一时间 expert(0) > operator(1) > model(2)
        key = (dt, -source_rank)
        old = latest_key.get(sample_id)
        if old is None or key > old:
            latest_key[sample_id] = key
            latest_row[sample_id] = row

    out: List[Dict[str, str]] = []
    for sample_id, pair_set in variants.items():
        if len(pair_set) <= 1:
            continue
        latest = latest_row.get(sample_id, {})
        variants_text = "; ".join(sorted(f"{rk}/{rrk}" for rk, rrk in pair_set))
        out.append(
            {
                "view_name": "label_inconsistent",
                "line": _norm(latest.get("line")),
                "sn": _norm(latest.get("sn")),
                "sample_id": sample_id,
                "variant_count": str(len(pair_set)),
                "label_variants": variants_text,
                "latest_result_key": _norm(latest.get("result_key")),
                "latest_reason_key": _norm(latest.get("reason_key")),
                "latest_reason_name": _norm(latest.get("reason_name")),
                "latest_timestamp": _norm(latest.get("timestamp")),
                "latest_source": _norm(latest.get("source")),
            }
        )

    out.sort(key=lambda r: (r["line"], r["sn"], r["sample_id"]))
    return out


def _write_inconsistent_sample_view(
    rows: List[Dict[str, str]],
    *,
    output_path: Path,
    dry_run: bool,
) -> bool:
    if not rows:
        print("\n[Label Consistency] 未发现标签不一致样本。")
        return False

    if dry_run:
        print(f"\n[Label Consistency - DRY-RUN] 检测到不一致样本 {len(rows)} 条")
        print(f"将输出: {output_path}")
        return True

    header = [
        "view_name",
        "line",
        "sn",
        "sample_id",
        "variant_count",
        "label_variants",
        "latest_result_key",
        "latest_reason_key",
        "latest_reason_name",
        "latest_timestamp",
        "latest_source",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[Label Consistency] 检测到不一致样本 {len(rows)} 条")
    print(f"sample_view.csv: {output_path}")
    return True


def _print_folder_rename_report(rename_report: Dict[str, object], *, max_show: int) -> None:
    folder_moves = list(rename_report.get("folder_moves") or [])
    file_moves = list(rename_report.get("file_moves") or [])

    if not folder_moves:
        print("\n[Folder Rename] 未发现文件夹改名。")
        return

    max_show = max(1, int(max_show))
    print(f"\n[Folder Rename] 检测到 {len(folder_moves)} 组目录改名映射")
    for idx, item in enumerate(folder_moves[:max_show], start=1):
        print(
            f"  [{idx}] {item['old_folder']}  ->  {item['new_folder']} "
            f"(files={item['file_count']})"
        )
    if len(folder_moves) > max_show:
        print(f"  ... 其余 {len(folder_moves) - max_show} 组未展示")

    print(f"[Folder Rename] 文件级改名条数: {len(file_moves)}")
    for idx, mv in enumerate(file_moves[:max_show], start=1):
        print(
            f"  file[{idx}] {mv['old_relative_path']}  ->  {mv['new_relative_path']}"
        )
    if len(file_moves) > max_show:
        print(f"  ... 其余 {len(file_moves) - max_show} 条未展示")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查重复 TDMS、更新 metadata，并输出标签不一致样本 sample_view.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML 配置路径（默认: {DEFAULT_CONFIG_PATH}）",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="data_root 路径（优先级高于配置）",
    )
    parser.add_argument(
        "--storage-root",
        type=str,
        default=None,
        help="tdms 根目录（默认 factory_raw，可被配置覆盖）",
    )
    parser.add_argument(
        "--line",
        type=str,
        default="",
        help="可选：强制 line（不传则从相对路径首段推断）",
    )
    parser.add_argument(
        "--sample-view-out",
        type=Path,
        default=DEFAULT_SAMPLE_VIEW_OUT,
        help="标签不一致样本输出路径（默认当前目录 sample_view.csv）",
    )
    parser.add_argument(
        "--by-content",
        action="store_true",
        help="额外按内容 hash 检查重复（较慢）",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="执行重复文件删除（默认不删除，仅检查）",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="重复组最多展示前 N 组（默认 20）",
    )
    parser.add_argument(
        "--max-show-errors",
        type=int,
        default=50,
        help="metadata 解析错误最多展示条数（默认 50）",
    )
    parser.add_argument(
        "--max-show-renames",
        type=int,
        default=50,
        help="文件夹改名结果最多展示条数（默认 50）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览，不写 metadata/sample_view，不删除文件",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="并发提取样本记录的线程数（默认 0=自动，建议 1~8）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _read_yaml_cfg(_normalize_path(Path(args.config)))
    data_root, storage_root, line_override = _resolve_runtime_settings(args, cfg)
    tdms_root = data_root / storage_root

    if not tdms_root.exists():
        raise FileNotFoundError(f"tdms 根目录不存在: {tdms_root}")

    print(f"data_root              : {data_root}")
    print(f"tdms_storage_root      : {storage_root}")
    print(f"tdms_root              : {tdms_root}")
    print(f"line_override          : {line_override or '(auto)'}")
    print(f"dry_run                : {args.dry_run}")
    print(f"workers                : {_recommended_workers(int(args.workers))}")

    def _scan_tdms_files() -> List[Path]:
        files = _iter_tdms_files(tdms_root)
        if line_override:
            files = [
                p for p in files
                if p.relative_to(tdms_root).parts and p.relative_to(tdms_root).parts[0] == line_override
            ]
        return files

    tdms_files = _iter_tdms_files(tdms_root)
    print(f"TDMS 文件总数          : {len(tdms_files)}")
    if line_override:
        line_scan_root = tdms_root / line_override
        if not line_scan_root.exists():
            print(f"[WARN] line='{line_override}' 对应目录不存在: {line_scan_root}")
        before = len(tdms_files)
        tdms_files = [
            p for p in tdms_files
            if p.relative_to(tdms_root).parts and p.relative_to(tdms_root).parts[0] == line_override
        ]
        print(f"按 --line='{line_override}' 过滤后 TDMS 文件数: {len(tdms_files)}（原 {before}）")
    manifest_path = data_root / "metadata" / "tdms_manifest.csv"
    old_manifest_rows = _read_csv_rows(manifest_path)

    by_name_groups = _build_duplicate_groups(_group_by_filename(tdms_files).items())
    _print_duplicate_groups("[By filename]", by_name_groups, top_n=max(1, args.top_n))

    by_content_groups: List[Tuple[str, List[Path]]] = []
    if args.by_content:
        by_content = _group_by_content(tdms_files)
        by_content_groups = _build_duplicate_groups(
            (f"{size}:{digest[:12]}", paths) for (size, digest), paths in by_content.items()
        )
        _print_duplicate_groups("[By content sha1]", by_content_groups, top_n=max(1, args.top_n))

    deleted_count = 0
    delete_failed = 0
    if args.dedup:
        target_groups = by_content_groups if args.by_content else by_name_groups
        deleted_count, delete_failed = _deduplicate_groups(target_groups, dry_run=args.dry_run)
        if not args.dry_run and deleted_count > 0:
            tdms_files = _scan_tdms_files()
            print(f"\n[Dedup] 删除后 TDMS 文件数: {len(tdms_files)}")
    else:
        print("\n[Dedup] 未启用 --dedup，仅检查重复，不删除文件。")

    duplicate_sn_groups, duplicate_sn_parse_errors = _build_duplicate_sn_groups(
        tdms_files=tdms_files,
        tdms_root=tdms_root,
        data_root=data_root,
        storage_root=storage_root,
        line_override=line_override,
        old_manifest_rows=old_manifest_rows,
    )
    _print_duplicate_sn_groups("[By SN]", duplicate_sn_groups, top_n=max(1, args.top_n))
    if duplicate_sn_parse_errors:
        max_show = max(1, int(args.max_show_errors))
        print(f"\n[Duplicate SN Parse Failed] {len(duplicate_sn_parse_errors)}")
        for item in duplicate_sn_parse_errors[:max_show]:
            print(f"  - {item}")
        if len(duplicate_sn_parse_errors) > max_show:
            print(f"  ... 其余 {len(duplicate_sn_parse_errors) - max_show} 条未展示")

    duplicate_sn_deleted = 0
    duplicate_sn_failed = 0
    if args.dedup:
        duplicate_sn_deleted, duplicate_sn_failed = _deduplicate_sn_groups(
            duplicate_sn_groups,
            dry_run=args.dry_run,
            tdms_root=tdms_root,
        )
        deleted_count += duplicate_sn_deleted
        delete_failed += duplicate_sn_failed
        if not args.dry_run and duplicate_sn_deleted > 0:
            tdms_files = _scan_tdms_files()
            print(f"\n[Dedup by SN] 删除后 TDMS 文件数: {len(tdms_files)}")

    metadata_summary = _rebuild_metadata(
        data_root=data_root,
        storage_root=storage_root,
        line_override=line_override,
        dry_run=args.dry_run,
        workers=int(args.workers),
    )

    errors = list(metadata_summary["errors"])
    if errors:
        max_show = max(1, int(args.max_show_errors))
        print(f"\n[Metadata] 解析失败: {len(errors)}")
        for item in errors[:max_show]:
            print(f"  - {item}")
        if len(errors) > max_show:
            print(f"  ... 其余 {len(errors) - max_show} 条未展示")
    else:
        print("\n[Metadata] 解析失败: 0")

    rename_report = dict(metadata_summary.get("rename_report") or {})
    _print_folder_rename_report(
        rename_report,
        max_show=max(1, int(args.max_show_renames)),
    )
    duplicate_messages = list(metadata_summary.get("duplicate_messages") or [])
    duplicate_sn_skipped = int(metadata_summary.get("duplicate_sn_skipped") or 0)
    upgraded_existing = int(metadata_summary.get("upgraded_existing") or 0)
    if duplicate_messages:
        max_show = max(1, int(args.max_show_errors))
        print(f"\n[Duplicate SN] 跳过/升级记录: {len(duplicate_messages)}")
        for item in duplicate_messages[:max_show]:
            print(f"  - {item}")
        if len(duplicate_messages) > max_show:
            print(f"  ... 其余 {len(duplicate_messages) - max_show} 条未展示")

    label_records_db_path = Path(metadata_summary["label_records_db_path"])
    store = LabelStore(label_records_db_path)
    label_rows = store.list_all()
    inconsistent_rows = _collect_inconsistent_label_rows(label_rows)

    sample_view_out = _normalize_path(Path(args.sample_view_out))
    if not sample_view_out.is_absolute():
        sample_view_out = Path.cwd() / sample_view_out
    wrote_sample_view = _write_inconsistent_sample_view(
        inconsistent_rows,
        output_path=sample_view_out,
        dry_run=args.dry_run,
    )

    print("\n[Summary]")
    print(f"重复组(by filename)      : {len(by_name_groups)}")
    if args.by_content:
        print(f"重复组(by content)       : {len(by_content_groups)}")
    print(f"重复组(by SN)            : {len(duplicate_sn_groups)}")
    print(f"去重删除成功            : {deleted_count}")
    print(f"去重删除失败            : {delete_failed}")
    print(f"manifest 行数           : {metadata_summary['manifest_rows']}")
    print(f"sample_index 行数       : {metadata_summary['sample_rows']}")
    print(f"重复 SN 跳过            : {duplicate_sn_skipped}")
    print(f".tdms -> .tdms.zst 升级 : {upgraded_existing}")
    print(f"metadata 解析失败       : {len(errors)}")
    print(f"文件夹改名映射组数      : {len(rename_report.get('folder_moves') or [])}")
    print(f"文件级改名条数          : {len(rename_report.get('file_moves') or [])}")
    print(f"标签总记录              : {len(label_rows)}")
    print(f"标签不一致样本数        : {len(inconsistent_rows)}")
    print(f"已输出 sample_view.csv  : {wrote_sample_view and (not args.dry_run)}")
    print(f"manifest_path           : {metadata_summary['manifest_path']}")
    print(f"label_records_db_path  : {label_records_db_path}")
    if inconsistent_rows:
        print(f"sample_view_path        : {sample_view_out}")


if __name__ == "__main__":
    main()
