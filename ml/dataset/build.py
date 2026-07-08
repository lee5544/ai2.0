from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# 确保项目根目录可 import。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from data_manager.label_database import load_label_dataframe, load_sample_dataframe
from data_manager.tdms_read import read_tdms
from data_manager.config import load_data_manager_config
from data_manager.label_internal_registry import label_source_rank
from ml.features import get_feature_extractor, resolve_feature_version
from ml.training.config import resolve_dataset_output_dir


REQUIRED_SAMPLE_VIEW_COLS = {"line", "sn", "sample_id"}
EXCLUDED_OUTPUT_COLS = {"sample_view_file", "sample_view_row_index", "view_name"}
DEFAULT_CONFIG_PATH = Path("cfg/epump4.yaml")
DM_CFG = load_data_manager_config()
SAMPLE_VIEW_OUTPUT_COLS = ["line", "sn", "sample_id"]
FEATURE_OUTPUT_METADATA_COLUMNS = [
    "line",
    "sn",
    "sample_id",
    "reference",
    "time",
    "tdms_storage_root",
    "relative_path",
    "tdms_path",
    "group_name",
    "channel_name",
    "sampling_rate",
    "seq_length",
    "num_features",
    "label",
    "label_source",
    "label_timestamp",
    "result_key",
    "result_id",
    "result_name",
    "reason_key",
    "reason_id",
    "reason_name",
    "label_version",
    "note",
]

OPTIONAL_SAMPLE_VIEW_OUTPUT_COLS = [
    "reference",
    "time",
    "tdms_storage_root",
    "relative_path",
]

LABEL_SIGNAL_COLS = (
    "result_key",
    "result_id",
    "result_name",
    "reason_key",
    "reason_id",
    "reason_name",
)


def _ordered_unique(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _resolve_output_folder(cfg: Dict[str, Any], cli_output_folder: str | None) -> str:
    if cli_output_folder:
        return cli_output_folder
    return str(resolve_dataset_output_dir(cfg))


def _normalize_path_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (str, int, float)):
        items = [raw]
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        return []

    out: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _yaml_key_sns_sample_views(cfg: Dict[str, Any]) -> List[str]:
    data_cfg = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    dataset_cfg = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}

    paths: List[str] = []
    for container in (data_cfg, dataset_cfg):
        for key in ("key_sns", "key_sns_sample_view", "key_sns_sample_views"):
            raw = container.get(key)
            if key == "key_sns" and isinstance(raw, list):
                paths.extend(
                    str(item.get("path") or item.get("folder") or item.get("dir") or "").strip()
                    if isinstance(item, dict) else str(item or "").strip()
                    for item in raw
                )
            else:
                paths.extend(_normalize_path_list(raw))
    return _ordered_unique(paths)


def _resolve_yaml_sample_view_path(path_text: str, *, data_root: Path, output_feature_folder: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        data_root / path,
        output_feature_folder / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_sample_view_paths(
    *,
    cfg: Dict[str, Any],
    args_sample_view: List[str],
    data_root: Path,
    output_feature_folder: Path,
) -> List[Path]:
    if args_sample_view:
        return [Path(p).expanduser() for p in args_sample_view]

    # 默认从统一生成器的输出目录读取 sample_view.csv。
    paths: List[Path] = []
    default_sample_view = output_feature_folder / "sample_view.csv"
    if default_sample_view.exists():
        paths.append(default_sample_view)

    for path_text in _yaml_key_sns_sample_views(cfg):
        path = _resolve_yaml_sample_view_path(
            path_text,
            data_root=data_root,
            output_feature_folder=output_feature_folder,
        )
        if path.exists() and path.is_dir():
            print(f"跳过 data.key_sns 文件夹（已在 ml/dataset/sample_filter.py 中纳入）: {path}")
            continue
        if path.suffix.lower() != ".csv":
            print(f"跳过 data.key_sns 非 CSV 路径: {path}")
            continue
        paths.append(path)

    deduped: List[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    if deduped:
        return deduped

    print(f"⚠️ 未找到 sample_view.csv: {default_sample_view}")
    print("请先执行 ml/dataset/generate.py 生成 sample_view.csv，或在 YAML 的 data.key_sns 中配置额外 sample_view CSV。")
    return []


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于一系列 sample_view.csv 提取信号/标签/特征并保存为特征 CSV"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"YAML 配置路径（默认: {DEFAULT_CONFIG_PATH}）",
    )
    parser.add_argument(
        "--sample-view",
        action="append",
        default=[],
        help="sample_view.csv 路径（可重复传入；不传则默认读取输出目录下 sample_view.csv）",
    )
    parser.add_argument("--data-root", default=None, help="data_root 路径")
    parser.add_argument("--manifest-path", default=None, help="tdms_manifest.csv 路径")
    parser.add_argument("--label-records-db-path", default=None, help="label_records.db 路径")
    parser.add_argument("--output-feature-folder", default=None, help="特征输出目录")
    parser.add_argument("--output-file-prefix", default="features", help="输出文件前缀（默认生成 features_batch_x.csv）")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="每个批次写出行数（默认 2000，可被 dataset.batch_size 覆盖）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 1))),
        help="并行 worker 数（默认: min(16, CPU核数)）",
    )
    parser.add_argument(
        "--label-target-sampling",
        action="store_true",
        help=(
            "按 train.label_mapping 目标数量在提取前下采样（默认关闭）。"
            "注意：标准流程的目标数量采样在 "
            "ml/dataset/label_filter.py 中完成"
            "（保证 sample_view 与 features 一一对应，通过训练前校验）；"
            "本开关仅用于跳过 filter 步骤单独跑提取时。"
        ),
    )
    return parser.parse_args()


def _read_yaml(config_path: str | None) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_batch_size(cfg: Dict[str, Any], cli_batch_size: int | None) -> int:
    if cli_batch_size is not None:
        batch_size = cli_batch_size
    else:
        dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
        batch_size = dataset_cfg.get("batch_size")
        if batch_size is None:
            batch_size = cfg.get("batch_size")
        if batch_size is None:
            batch_size = 2000

    try:
        batch_size_int = int(batch_size)
    except Exception as exc:
        raise ValueError(f"batch_size 非法: {batch_size}") from exc
    if batch_size_int <= 0:
        raise ValueError(f"batch_size 必须 > 0，当前: {batch_size_int}")
    return batch_size_int


def _to_key(v: Any) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def _to_int(v: Any):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def _first_present_value(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def _tuple_get(row: Tuple[Any, ...], col_idx: Dict[str, int], col: str) -> Any:
    idx = col_idx.get(col)
    if idx is None:
        return None
    return row[idx]


def _has_effective_label_fields(row: Dict[str, Any]) -> bool:
    for col in LABEL_SIGNAL_COLS:
        if _to_key(row.get(col)):
            return True
    return False


def _extract_label_from_sample_view_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    candidate: Dict[str, Any] = {
        "source": _to_key(row.get("label_source") or row.get("source")),
        "timestamp": _to_key(row.get("label_timestamp") or row.get("timestamp")),
        "result_key": _to_key(row.get("result_key")),
        "result_id": _first_present_value(row, "result_id"),
        "result_name": _to_key(row.get("result_name")),
        "reason_key": _to_key(row.get("reason_key")),
        "reason_id": _first_present_value(row, "reason_id"),
        "reason_name": _to_key(row.get("reason_name")),
        "label_version": _to_key(row.get("label_version")),
        "note": _to_key(row.get("note")),
    }
    if not _has_effective_label_fields(candidate):
        return None

    if not candidate["result_key"]:
        candidate["result_key"] = candidate["result_name"]
    if not candidate["result_name"]:
        candidate["result_name"] = candidate["result_key"]
    if not candidate["reason_key"]:
        candidate["reason_key"] = candidate["reason_name"]
    if not candidate["reason_name"]:
        candidate["reason_name"] = candidate["reason_key"]
    return candidate


def _label_target_map_from_cfg(cfg: Dict[str, Any]) -> Dict[str, int]:
    """从 train.label_mapping.groups 取每个 reason 的目标数量（仅 >=0 的才参与下采样）。"""
    train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    lm = train_cfg.get("label_mapping") if isinstance(train_cfg.get("label_mapping"), dict) else {}
    if not lm or not lm.get("enable", False):
        return {}

    targets: Dict[str, int] = {}
    for group in lm.get("groups") or []:
        if not isinstance(group, dict):
            continue
        for token in group.get("reasons") or []:
            if not isinstance(token, dict):
                continue
            name = _to_key(token.get("reason"))
            target = _to_int(token.get("sample"))
            if name and target is not None and target >= 0:
                targets[name] = int(target)
    return targets


def _apply_label_target_sampling(
    rows: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    *,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """提取特征前按目标数量下采样，避免对训练用不到的样本提特征。

    - 目标数量 >= 0 的 reason：超过目标时随机下采样到目标数量（固定 seed 可复现）。
    - 目标数量 = -1（全取）或不在 label_mapping 里的行：保持不变。
    - 上采样（目标 > 可用数）不在这里做，仍由训练侧 rebalance 随机复制补齐。
    """
    targets = _label_target_map_from_cfg(cfg)
    if not targets or not rows:
        return rows

    import random

    by_reason: Dict[str, List[int]] = {}
    passthrough: List[int] = []
    for idx, row in enumerate(rows):
        name = _to_key(row.get("reason_name"))
        if name in targets:
            by_reason.setdefault(name, []).append(idx)
        else:
            passthrough.append(idx)

    rng = random.Random(int(seed))
    chosen: set[int] = set(passthrough)
    for name in sorted(by_reason):
        idxs = by_reason[name]
        target = targets[name]
        if len(idxs) > target:
            picked = rng.sample(idxs, target)
            print(f"[采样] {name}: {len(idxs)} -> {target}（提取前按目标数量下采样）")
        else:
            picked = idxs
            print(f"[采样] {name}: {len(idxs)} <= 目标 {target}，全取（上采样留给训练侧）")
        chosen.update(picked)

    sampled = [rows[i] for i in sorted(chosen)]
    print(f"[采样] 提取行数: {len(rows)} -> {len(sampled)}")
    return sampled


def _collect_sample_view_columns(sample_view_files: List[Path]) -> List[str]:
    optional_columns: List[str] = []
    for path in sample_view_files:
        if not path.exists():
            raise FileNotFoundError(f"sample_view not found: {path}")
        header_df = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        missing = REQUIRED_SAMPLE_VIEW_COLS - set(header_df.columns)
        if missing:
            raise ValueError(f"{path} 缺少列: {sorted(missing)}")
        for col in OPTIONAL_SAMPLE_VIEW_OUTPUT_COLS:
            if col in header_df.columns:
                optional_columns.append(col)
    return _ordered_unique(list(SAMPLE_VIEW_OUTPUT_COLS) + optional_columns)


def _flush_batch(
    *,
    batch: List[Dict[str, Any]],
    batch_index: int,
    output_folder: Path,
    output_file_prefix: str,
    fixed_columns: List[str],
    feature_columns: List[str],
) -> int:
    if not batch:
        return batch_index

    df = pd.DataFrame(batch)
    all_columns = fixed_columns + feature_columns
    for c in all_columns:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[all_columns]

    out_path = output_folder / f"{output_file_prefix}_batch_{batch_index}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 批次 {batch_index}: 保存 {len(df)} 条 -> {out_path}")
    return batch_index + 1


def _build_sample_meta_lookup(label_records_db_path: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    sample_index_df = load_sample_dataframe(label_records_db_path)
    lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    col_idx = {c: i for i, c in enumerate(sample_index_df.columns)}
    for row in sample_index_df.itertuples(index=False, name=None):
        line = _to_key(_tuple_get(row, col_idx, "line"))
        sn = _to_key(_tuple_get(row, col_idx, "sn"))
        sample_id = _to_key(_tuple_get(row, col_idx, "sample_id"))
        if not line or not sn or not sample_id:
            continue
        lookup[(line, sn, sample_id)] = {
            "group_name": _to_key(_tuple_get(row, col_idx, "group_name")),
            "sampling_rate": _to_int(_tuple_get(row, col_idx, "sampling_rate")),
        }
    return lookup


def _build_manifest_lookup(manifest_path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    manifest_df = pd.read_csv(manifest_path, encoding="utf-8-sig")
    if manifest_df.empty:
        return {}

    if "created_time" in manifest_df.columns:
        manifest_df["_created_ts"] = pd.to_datetime(manifest_df["created_time"], errors="coerce")
    else:
        manifest_df["_created_ts"] = pd.NaT

    manifest_df = manifest_df.sort_values("_created_ts", ascending=True, na_position="last")
    latest_df = manifest_df.drop_duplicates(subset=["line", "sn"], keep="last")

    lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
    col_idx = {c: i for i, c in enumerate(latest_df.columns)}
    for row in latest_df.itertuples(index=False, name=None):
        line = _to_key(_tuple_get(row, col_idx, "line"))
        sn = _to_key(_tuple_get(row, col_idx, "sn"))
        if not line or not sn:
            continue
        storage_root_val = _to_key(_tuple_get(row, col_idx, "tdms_storage_root"))
        lookup[(line, sn)] = {
            "tdms_storage_root": storage_root_val,
            "relative_path": _to_key(_tuple_get(row, col_idx, "relative_path")),
        }
    return lookup


def _build_label_lookup(label_records_db_path: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    label_df = load_label_dataframe(label_records_db_path, statuses=("confirmed", "unconfirmed"))
    if label_df.empty:
        return {}

    required_cols = {"line", "sn", "sample_id"}
    missing_cols = required_cols - set(label_df.columns)
    if missing_cols:
        raise ValueError(f"label_history missing columns: {sorted(missing_cols)}")

    if "timestamp" in label_df.columns:
        label_df["_ts"] = pd.to_datetime(label_df["timestamp"], errors="coerce")
    else:
        label_df["_ts"] = pd.NaT

    source_priority = {"expert": 0, "operator": 1, "model": 2}
    if "source" in label_df.columns:
        label_df["_source_rank"] = label_df["source"].astype(str).map(
            lambda value: label_source_rank(value, source_priority)
        ).fillna(99)
    else:
        label_df["_source_rank"] = 99

    label_df = label_df.sort_values(
        by=["line", "sn", "sample_id", "_ts", "_source_rank"],
        ascending=[True, True, True, False, True],
        na_position="last",
    )
    latest_df = label_df.drop_duplicates(subset=["line", "sn", "sample_id"], keep="first")

    keep_cols = [
        "source",
        "timestamp",
        "result_key",
        "result_id",
        "result_name",
        "reason_key",
        "reason_id",
        "reason_name",
        "label_version",
        "note",
    ]

    lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    col_idx = {c: i for i, c in enumerate(latest_df.columns)}
    for row in latest_df.itertuples(index=False, name=None):
        line = _to_key(_tuple_get(row, col_idx, "line"))
        sn = _to_key(_tuple_get(row, col_idx, "sn"))
        sample_id = _to_key(_tuple_get(row, col_idx, "sample_id"))
        if not line or not sn or not sample_id:
            continue
        record = {k: _tuple_get(row, col_idx, k) for k in keep_cols}
        normalized = _extract_label_from_sample_view_row(record)
        if normalized is None:
            continue
        lookup[(line, sn, sample_id)] = normalized
    return lookup


def _pick_signal_by_sample(
    *,
    tdms_ret: Dict[str, Any],
    sample_group_name: str,
    sample_id: str,
) -> Tuple[Any, str]:
    if sample_group_name == tdms_ret["up_group"]:
        signal = tdms_ret.get("up_data")
        if signal is None:
            raise KeyError(f"signal not loaded for group: {tdms_ret['up_group']}")
        return signal, tdms_ret["up_group"]
    if sample_group_name == tdms_ret["down_group"]:
        signal = tdms_ret.get("down_data")
        if signal is None:
            raise KeyError(f"signal not loaded for group: {tdms_ret['down_group']}")
        return signal, tdms_ret["down_group"]

    sample_id_lower = str(sample_id).lower()
    if sample_id_lower.endswith("_up"):
        signal = tdms_ret.get("up_data")
        if signal is None:
            raise KeyError(f"signal not loaded for group: {tdms_ret['up_group']}")
        return signal, tdms_ret["up_group"]
    if sample_id_lower.endswith("_down"):
        signal = tdms_ret.get("down_data")
        if signal is None:
            raise KeyError(f"signal not loaded for group: {tdms_ret['down_group']}")
        return signal, tdms_ret["down_group"]

    raise KeyError(f"sample group cannot be mapped: {sample_group_name}, sample_id={sample_id}")


def _process_one_group(
    *,
    line: str,
    sn: str,
    feature_version: str,
    group_storage_root: str,
    group_relative_path: str,
    rows: List[Dict[str, Any]],
    sample_view_columns: List[str],
    data_root: Path,
    sample_meta_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    manifest_lookup: Dict[Tuple[str, str], Dict[str, str]],
    label_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    cacheable_tdms_paths: set[str],
    tdms_cache: Dict[Tuple[str, str, str], Dict[str, Any]],
    tdms_cache_lock: Lock,
    row_workers: int = 1,
) -> List[Dict[str, Any]]:
    group_t0 = time.perf_counter()
    extract_features = get_feature_extractor(feature_version)
    results: List[Dict[str, Any]] = []
    resolved_storage_root = _to_key(group_storage_root)
    resolved_relative_path = _to_key(group_relative_path)
    if not resolved_relative_path:
        manifest_meta = manifest_lookup.get((line, sn))
        if manifest_meta:
            resolved_storage_root = _to_key(manifest_meta.get("tdms_storage_root"))
            resolved_relative_path = _to_key(manifest_meta.get("relative_path"))
    else:
        manifest_meta = {
            "tdms_storage_root": resolved_storage_root,
            "relative_path": resolved_relative_path,
        }
        if not resolved_storage_root:
            fallback_manifest_meta = manifest_lookup.get((line, sn))
            if fallback_manifest_meta:
                resolved_storage_root = _to_key(fallback_manifest_meta.get("tdms_storage_root"))
                manifest_meta["tdms_storage_root"] = resolved_storage_root

    prepared_rows: List[Dict[str, Any]] = []
    required_groups: set[str] = set()
    has_unknown_group = False

    for row_map in rows:
        row_line = _to_key(row_map.get("line"))
        row_sn = _to_key(row_map.get("sn"))
        sample_id = _to_key(row_map.get("sample_id"))
        sv_file = _to_key(row_map.get("__sample_view_file"))
        sv_row_idx = row_map.get("__sample_view_row_index")

        if not row_line or not row_sn or not sample_id:
            results.append(
                {
                    "status": "fail",
                    "message": "line/sn/sample_id 为空",
                    "sv_file": sv_file,
                    "sv_row_idx": sv_row_idx,
                    "line": row_line,
                    "sn": row_sn,
                    "sample_id": sample_id,
                }
            )
            continue

        key = (row_line, row_sn, sample_id)
        label_ret = _extract_label_from_sample_view_row(row_map) or label_lookup.get(key)
        if not label_ret:
            results.append(
                {
                    "status": "skip",
                    "sv_file": sv_file,
                    "sv_row_idx": sv_row_idx,
                    "line": row_line,
                    "sn": row_sn,
                    "sample_id": sample_id,
                }
            )
            continue

        sample_meta = sample_meta_lookup.get(key) or {}
        sample_group_name = _to_key(row_map.get("group_name")) or _to_key(sample_meta.get("group_name"))
        sample_sampling_rate = _to_int(row_map.get("sampling_rate")) or _to_int(sample_meta.get("sampling_rate"))
        if not sample_group_name:
            results.append(
                {
                    "status": "fail",
                    "message": "sample group not found in sample_index/sample_view",
                    "sv_file": sv_file,
                    "sv_row_idx": sv_row_idx,
                    "line": row_line,
                    "sn": row_sn,
                    "sample_id": sample_id,
                }
            )
            continue

        if not resolved_storage_root or not resolved_relative_path:
            results.append(
                {
                    "status": "fail",
                    "message": "manifest path not found in sample_view/lookup",
                    "sv_file": sv_file,
                    "sv_row_idx": sv_row_idx,
                    "line": row_line,
                    "sn": row_sn,
                    "sample_id": sample_id,
                }
            )
            continue

        if sample_group_name:
            required_groups.add(sample_group_name)
        else:
            has_unknown_group = True

        prepared_rows.append(
            {
                "row_map": row_map,
                "sample_id": sample_id,
                "sample_group_name": sample_group_name,
                "sample_meta": {
                    **sample_meta,
                    "sampling_rate": sample_sampling_rate,
                },
                "label_ret": label_ret,
                "sv_file": sv_file,
                "sv_row_idx": sv_row_idx,
                "line": row_line,
                "sn": row_sn,
            }
        )

    if not prepared_rows or not manifest_meta:
        results.append(
            {
                "status": "metric",
                "metric_type": "group",
                "line": line,
                "sn": sn,
                "rows_total": int(len(rows)),
                "prepared_rows": int(len(prepared_rows)),
                "tdms_read_sec": 0.0,
                "extract_sec_sum": 0.0,
                "extract_rows": 0,
                "group_total_sec": float(time.perf_counter() - group_t0),
                "tdms_read_failed": 0,
                "tdms_cache_hit": 0,
            }
        )
        return results

    tdms_path = data_root / resolved_storage_root / resolved_relative_path
    tdms_required_groups = None if has_unknown_group else required_groups

    cache_key = (
        str(tdms_path),
        line,
        ",".join(sorted(tdms_required_groups)) if tdms_required_groups is not None else "*",
    )
    tdms_cache_hit = 0
    should_cache_tdms = str(tdms_path) in cacheable_tdms_paths
    if should_cache_tdms:
        with tdms_cache_lock:
            cached_tdms_ret = tdms_cache.get(cache_key)
    else:
        cached_tdms_ret = None

    tdms_t0 = time.perf_counter()
    try:
        if cached_tdms_ret is not None:
            tdms_ret = cached_tdms_ret
            tdms_cache_hit = 1
            tdms_read_sec = 0.0
        else:
            tdms_ret = read_tdms(
                tdms_path,
                line=line,
                required_groups=tdms_required_groups,
            )
            tdms_read_sec = float(time.perf_counter() - tdms_t0)
            if should_cache_tdms:
                with tdms_cache_lock:
                    tdms_cache.setdefault(cache_key, tdms_ret)
    except Exception as e:
        tdms_read_sec = float(time.perf_counter() - tdms_t0)
        tdms_error = f"{type(e).__name__}: {e}"
        for item in prepared_rows:
            results.append(
                {
                    "status": "fail",
                    "message": tdms_error,
                    "sv_file": item["sv_file"],
                    "sv_row_idx": item["sv_row_idx"],
                    "line": item["line"],
                    "sn": item["sn"],
                    "sample_id": item["sample_id"],
                }
            )
        results.append(
            {
                "status": "metric",
                "metric_type": "group",
                "line": line,
                "sn": sn,
                "rows_total": int(len(rows)),
                "prepared_rows": int(len(prepared_rows)),
                "tdms_read_sec": tdms_read_sec,
                "extract_sec_sum": 0.0,
                "extract_rows": 0,
                "group_total_sec": float(time.perf_counter() - group_t0),
                "tdms_read_failed": 1,
                "tdms_cache_hit": tdms_cache_hit,
            }
        )
        return results

    def _process_prepared_row(item: Dict[str, Any]) -> Dict[str, Any]:
        row_t0 = time.perf_counter()
        sample_id = item["sample_id"]
        sample_meta = item["sample_meta"]
        label_ret = item["label_ret"]
        row_map = item["row_map"]
        sv_file = item["sv_file"]
        sv_row_idx = item["sv_row_idx"]
        row_line = item["line"]
        row_sn = item["sn"]

        try:
            signal, mapped_group_name = _pick_signal_by_sample(
                tdms_ret=tdms_ret,
                sample_group_name=item["sample_group_name"],
                sample_id=sample_id,
            )
            sampling_rate = _to_int(sample_meta.get("sampling_rate")) or _to_int(tdms_ret.get("sampling_rate"))
            if not sampling_rate or sampling_rate <= 0:
                raise ValueError(f"invalid sampling_rate: {sampling_rate}")

            ext_t0 = time.perf_counter()
            extracted = extract_features(signal, sampling_rate, return_timing=True)
            extract_timing: Dict[str, float] = {}
            if isinstance(extracted, tuple) and len(extracted) == 2:
                features, extract_timing = extracted
            else:
                features = extracted
            extract_sec = float(extract_timing.get("total_sec") or (time.perf_counter() - ext_t0))
            extract_process_data_sec = float(extract_timing.get("process_data_sec") or 0.0)
            extract_mel_sec = float(extract_timing.get("mel_sec") or 0.0)
            extract_dwt_sec = float(extract_timing.get("dwt_sec") or 0.0)
            record: Dict[str, Any] = {
                "tdms_path": str(tdms_path),
                "group_name": mapped_group_name,
                "channel_name": tdms_ret.get("acc_channel"),
                "sampling_rate": sampling_rate,
                "seq_length": int(len(signal)),
                "num_features": int(len(features)),
            }

            for c in sample_view_columns:
                record[c] = row_map.get(c)

            result_key = label_ret.get("result_key")
            result_id = _to_int(label_ret.get("result_id"))
            result_name = label_ret.get("result_name")
            reason_key = label_ret.get("reason_key")
            reason_id = _to_int(label_ret.get("reason_id"))
            reason_name = label_ret.get("reason_name")

            record["label_source"] = label_ret.get("source")
            record["label_timestamp"] = label_ret.get("timestamp")
            record["result_key"] = result_key
            record["result_id"] = result_id
            record["result_name"] = result_name
            record["reason_key"] = reason_key
            record["reason_id"] = reason_id
            record["reason_name"] = reason_name
            record["label_version"] = label_ret.get("label_version")
            record["note"] = label_ret.get("note")
            record["label"] = result_id
            record.update(features)

            return {
                "status": "ok",
                "record": record,
                "feature_keys": list(features.keys()),
                "sv_file": sv_file,
                "sv_row_idx": sv_row_idx,
                "line": row_line,
                "sn": row_sn,
                "sample_id": sample_id,
                "_extract_sec": extract_sec,
                "_extract_process_data_sec": extract_process_data_sec,
                "_extract_mel_sec": extract_mel_sec,
                "_extract_dwt_sec": extract_dwt_sec,
                "_row_total_sec": float(time.perf_counter() - row_t0),
            }
        except Exception as e:
            return {
                "status": "fail",
                "message": f"{type(e).__name__}: {e}",
                "sv_file": sv_file,
                "sv_row_idx": sv_row_idx,
                "line": row_line,
                "sn": row_sn,
                "sample_id": sample_id,
                "_extract_sec": 0.0,
                "_extract_process_data_sec": 0.0,
                "_extract_mel_sec": 0.0,
                "_extract_dwt_sec": 0.0,
                "_row_total_sec": float(time.perf_counter() - row_t0),
            }

    row_workers = max(1, int(row_workers))
    inner_workers = min(row_workers, len(prepared_rows))
    if inner_workers <= 1:
        for item in prepared_rows:
            results.append(_process_prepared_row(item))
    else:
        with ThreadPoolExecutor(max_workers=inner_workers) as row_executor:
            row_futures = [row_executor.submit(_process_prepared_row, item) for item in prepared_rows]
            for item, fut in zip(prepared_rows, row_futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append(
                        {
                            "status": "fail",
                            "message": f"{type(e).__name__}: {e}",
                            "sv_file": item.get("sv_file"),
                            "sv_row_idx": item.get("sv_row_idx"),
                            "line": item.get("line"),
                            "sn": item.get("sn"),
                            "sample_id": item.get("sample_id"),
                            "_extract_sec": 0.0,
                            "_extract_process_data_sec": 0.0,
                            "_extract_mel_sec": 0.0,
                            "_extract_dwt_sec": 0.0,
                            "_row_total_sec": 0.0,
                        }
                    )

    group_extract_sec_sum = 0.0
    group_extract_rows = 0
    for ret in results:
        if ret.get("status") == "ok":
            ext_sec = float(ret.get("_extract_sec") or 0.0)
            if ext_sec > 0:
                group_extract_sec_sum += ext_sec
                group_extract_rows += 1
    results.append(
        {
            "status": "metric",
            "metric_type": "group",
            "line": line,
            "sn": sn,
            "rows_total": int(len(rows)),
            "prepared_rows": int(len(prepared_rows)),
            "tdms_read_sec": tdms_read_sec,
            "extract_sec_sum": float(group_extract_sec_sum),
            "extract_rows": int(group_extract_rows),
            "group_total_sec": float(time.perf_counter() - group_t0),
            "tdms_read_failed": 0,
            "tdms_cache_hit": tdms_cache_hit,
        }
    )

    return results


def main():
    wall_t0 = time.perf_counter()
    args = parse_args()
    args.num_workers = max(1, int(args.num_workers))
    cfg = _read_yaml(args.config)
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
    feature_version = resolve_feature_version(cfg)

    data_root_raw = (
        args.data_root
        or dataset_cfg.get("data_root")
        or cfg.get("data_root")
        or DM_CFG.get("data_root")
    )
    if not data_root_raw:
        raise ValueError("未提供 data_root：请用 --data-root 或在 YAML/cfg/core/data_manager.yaml 中配置")
    data_root = Path(data_root_raw).expanduser()

    manifest_path = Path(
        args.manifest_path
        or dataset_cfg.get("manifest_path")
        or cfg.get("manifest_path")
        or (data_root / "metadata" / "tdms_manifest.csv")
    ).expanduser()
    label_records_db_path = Path(
        args.label_records_db_path
        or dataset_cfg.get("label_records_db_path")
        or cfg.get("label_records_db_path")
        or (data_root / "metadata" / "label_records.db")
    ).expanduser()

    output_feature_folder = Path(_resolve_output_folder(cfg, args.output_feature_folder)).expanduser()
    output_feature_folder.mkdir(parents=True, exist_ok=True)

    sv_paths = _resolve_sample_view_paths(
        cfg=cfg,
        args_sample_view=args.sample_view,
        data_root=data_root,
        output_feature_folder=output_feature_folder,
    )
    if not sv_paths:
        return
    sample_view_columns = _collect_sample_view_columns(sv_paths)

    for meta_path, meta_name in (
        (manifest_path, "tdms_manifest.csv"),
        (label_records_db_path, "label_records.db"),
    ):
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_name} not found: {meta_path}")

    # 并发保护：同一输出目录同时只允许一个提取进程，否则分片互删会产生重复/缺失主键。
    lock_path = output_feature_folder / ".extract_features.lock"
    if lock_path.exists():
        try:
            other_pid = int((lock_path.read_text(encoding="utf-8").strip() or "0"))
        except Exception:
            other_pid = 0
        other_alive = False
        if other_pid > 0:
            try:
                os.kill(other_pid, 0)
                other_alive = True
            except OSError:
                other_alive = False
        if other_alive:
            raise RuntimeError(
                f"检测到另一个提取进程 (pid={other_pid}) 正在写出同一目录，"
                f"已中止以避免特征分片混写: {output_feature_folder}。"
                "请等待其结束（或停止它）后重试。"
            )
        print(f"[WARN] 发现残留提取锁（进程 {other_pid} 已不存在），继续执行。")
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    import atexit

    atexit.register(lambda: lock_path.unlink(missing_ok=True))

    # 输入校验通过后再清理旧 CSV，避免路径写错导致误删历史结果
    sv_path_set = {p.resolve() for p in sv_paths}
    preserved_names = {"features_batch_augmented.csv"}
    old_files = [
        f
        for f in output_feature_folder.glob("*.csv")
        if f.resolve() not in sv_path_set and f.name not in preserved_names
    ]
    if old_files:
        print(f"检测到 {len(old_files)} 个旧 CSV，正在删除...")
        for f in old_files:
            f.unlink()
            print(f"  删除: {f.name}")
    else:
        print("输出目录中未发现旧 CSV。")

    fixed_columns = list(FEATURE_OUTPUT_METADATA_COLUMNS)

    feature_columns: List[str] = []
    batch: List[Dict[str, Any]] = []
    batch_index = 1
    ok_rows = 0
    failed_rows = 0
    skipped_rows = 0
    failure_records: List[Dict[str, Any]] = []

    print(f"输出目录: {output_feature_folder}")
    print(f"特征版本: {feature_version}")
    print(f"sample_view 文件数: {len(sv_paths)}")
    print(f"并行 worker 数: {args.num_workers}")
    per_group_workers = 2 if args.num_workers > 1 else 1
    print(f"组内特征并行 worker 数: {per_group_workers}")
    flush_batch_size = _resolve_batch_size(cfg, args.batch_size)
    print(f"特征分片大小: {flush_batch_size} 条/CSV")

    print("预加载 metadata 索引...")
    t_meta0 = time.perf_counter()
    sample_meta_lookup = _build_sample_meta_lookup(label_records_db_path)
    manifest_lookup = _build_manifest_lookup(manifest_path)
    label_lookup = _build_label_lookup(label_records_db_path)
    t_meta_sec = float(time.perf_counter() - t_meta0)
    print(
        f"索引完成: sample_meta={len(sample_meta_lookup)}, "
        f"manifest={len(manifest_lookup)}, label={len(label_lookup)}"
    )
    print(f"metadata 索引耗时: {t_meta_sec:.3f}s")

    print("加载并按 TDMS 分组 sample_view...")
    t_grouping0 = time.perf_counter()
    grouped_rows: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    invalid_rows: List[Dict[str, Any]] = []
    total_rows = 0

    all_rows: List[Dict[str, Any]] = []
    for sv_path in sv_paths:
        print(f"  读取: {sv_path}")
        sv_df = pd.read_csv(sv_path, encoding="utf-8-sig")
        total_rows += len(sv_df)
        sv_columns = list(sv_df.columns)
        for row_idx, row in enumerate(sv_df.itertuples(index=False, name=None)):
            row_map = dict(zip(sv_columns, row))
            row_map["__sample_view_file"] = str(sv_path)
            row_map["__sample_view_row_index"] = int(row_idx)
            all_rows.append(row_map)

    # 提取前按 train.label_mapping 目标数量下采样（仅显式开启时；标准流程已在 filter 步骤采样）。
    if args.label_target_sampling:
        train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
        sampling_seed = _to_int(train_cfg.get("split_seed"))
        all_rows = _apply_label_target_sampling(
            all_rows,
            cfg,
            seed=42 if sampling_seed is None else sampling_seed,
        )

    for row_map in all_rows:
        line = _to_key(row_map.get("line"))
        sn = _to_key(row_map.get("sn"))
        storage_root = _to_key(row_map.get("tdms_storage_root"))
        relative_path = _to_key(row_map.get("relative_path"))
        if line and sn:
            grouped_rows.setdefault((line, sn, storage_root, relative_path), []).append(row_map)
        else:
            invalid_rows.append(row_map)

    print(
        f"分组完成: groups={len(grouped_rows)}, "
        f"grouped_rows={len(all_rows) - len(invalid_rows)}, invalid_rows={len(invalid_rows)}, "
        f"sample_view_total_rows={total_rows}"
    )
    t_grouping_sec = float(time.perf_counter() - t_grouping0)
    print(f"sample_view 分组耗时: {t_grouping_sec:.3f}s")

    tdms_path_ref_count: Dict[str, int] = {}
    for line, sn, storage_root, relative_path in grouped_rows.keys():
        if storage_root and relative_path:
            tdms_path_key = str(data_root / storage_root / relative_path)
        else:
            manifest_meta = manifest_lookup.get((line, sn))
            if not manifest_meta:
                continue
            tdms_path_key = str(
                data_root / str(manifest_meta["tdms_storage_root"]) / str(manifest_meta["relative_path"])
            )
        tdms_path_ref_count[tdms_path_key] = tdms_path_ref_count.get(tdms_path_key, 0) + 1
    cacheable_tdms_paths = {p for p, c in tdms_path_ref_count.items() if c > 1}
    if cacheable_tdms_paths:
        print(f"TDMS 复用缓存启用: {len(cacheable_tdms_paths)} 个路径被多次引用")
    else:
        print("TDMS 复用缓存: 当前 sample_view 无重复 TDMS 引用路径")

    max_pending = max(8, args.num_workers * 2)
    pending = set()
    pending_sizes: Dict[Any, int] = {}
    metric_group_count = 0
    metric_group_tdms_fail = 0
    metric_group_total_sec_sum = 0.0
    metric_group_total_sec_max = 0.0
    metric_tdms_read_sec_sum = 0.0
    metric_tdms_read_sec_max = 0.0
    metric_tdms_cache_hits = 0
    metric_extract_sec_sum = 0.0
    metric_extract_rows = 0
    metric_extract_sec_max = 0.0
    metric_extract_process_data_sec_sum = 0.0
    metric_extract_process_data_sec_max = 0.0
    metric_extract_mel_sec_sum = 0.0
    metric_extract_mel_sec_max = 0.0
    metric_extract_dwt_sec_sum = 0.0
    metric_extract_dwt_sec_max = 0.0
    metric_row_total_sec_sum = 0.0
    metric_row_total_sec_max = 0.0
    tdms_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    tdms_cache_lock = Lock()

    def _consume_result(ret: Dict[str, Any]) -> None:
        nonlocal feature_columns, batch, batch_index, ok_rows, failed_rows, skipped_rows
        nonlocal metric_group_count, metric_group_tdms_fail
        nonlocal metric_group_total_sec_sum, metric_group_total_sec_max
        nonlocal metric_tdms_read_sec_sum, metric_tdms_read_sec_max
        nonlocal metric_tdms_cache_hits
        nonlocal metric_extract_sec_sum, metric_extract_rows, metric_extract_sec_max
        nonlocal metric_extract_process_data_sec_sum, metric_extract_process_data_sec_max
        nonlocal metric_extract_mel_sec_sum, metric_extract_mel_sec_max
        nonlocal metric_extract_dwt_sec_sum, metric_extract_dwt_sec_max
        nonlocal metric_row_total_sec_sum, metric_row_total_sec_max
        status = ret.get("status")
        if status == "metric":
            metric_group_count += 1
            tdms_read_failed = int(ret.get("tdms_read_failed") or 0)
            if tdms_read_failed:
                metric_group_tdms_fail += 1
            group_total_sec = float(ret.get("group_total_sec") or 0.0)
            tdms_read_sec = float(ret.get("tdms_read_sec") or 0.0)
            tdms_cache_hit = int(ret.get("tdms_cache_hit") or 0)
            extract_sec_sum = float(ret.get("extract_sec_sum") or 0.0)
            extract_rows = int(ret.get("extract_rows") or 0)
            metric_group_total_sec_sum += group_total_sec
            metric_tdms_read_sec_sum += tdms_read_sec
            metric_tdms_cache_hits += tdms_cache_hit
            metric_extract_sec_sum += extract_sec_sum
            metric_extract_rows += extract_rows
            if group_total_sec > metric_group_total_sec_max:
                metric_group_total_sec_max = group_total_sec
            if tdms_read_sec > metric_tdms_read_sec_max:
                metric_tdms_read_sec_max = tdms_read_sec
            if extract_rows > 0:
                extract_avg = extract_sec_sum / max(1, extract_rows)
                if extract_avg > metric_extract_sec_max:
                    metric_extract_sec_max = extract_avg
            return
        if status == "ok":
            record = ret["record"]
            for k in ret.get("feature_keys", []):
                if k not in feature_columns:
                    feature_columns.append(k)

            batch.append(record)
            ok_rows += 1
            if len(batch) >= flush_batch_size:
                batch_index = _flush_batch(
                    batch=batch,
                    batch_index=batch_index,
                    output_folder=output_feature_folder,
                    output_file_prefix=args.output_file_prefix,
                    fixed_columns=fixed_columns,
                    feature_columns=feature_columns,
                )
                batch.clear()
            row_total_sec = float(ret.get("_row_total_sec") or 0.0)
            extract_sec = float(ret.get("_extract_sec") or 0.0)
            metric_row_total_sec_sum += row_total_sec
            if row_total_sec > metric_row_total_sec_max:
                metric_row_total_sec_max = row_total_sec
            if extract_sec > metric_extract_sec_max:
                metric_extract_sec_max = extract_sec
            extract_process_data_sec = float(ret.get("_extract_process_data_sec") or 0.0)
            metric_extract_process_data_sec_sum += extract_process_data_sec
            if extract_process_data_sec > metric_extract_process_data_sec_max:
                metric_extract_process_data_sec_max = extract_process_data_sec
            extract_mel_sec = float(ret.get("_extract_mel_sec") or 0.0)
            metric_extract_mel_sec_sum += extract_mel_sec
            if extract_mel_sec > metric_extract_mel_sec_max:
                metric_extract_mel_sec_max = extract_mel_sec
            extract_dwt_sec = float(ret.get("_extract_dwt_sec") or 0.0)
            metric_extract_dwt_sec_sum += extract_dwt_sec
            if extract_dwt_sec > metric_extract_dwt_sec_max:
                metric_extract_dwt_sec_max = extract_dwt_sec
        elif status == "skip":
            skipped_rows += 1
            failure_records.append(
                {
                    "status": "skip",
                    "sv_file": ret.get("sv_file"),
                    "sv_row_idx": ret.get("sv_row_idx"),
                    "line": ret.get("line"),
                    "sn": ret.get("sn"),
                    "sample_id": ret.get("sample_id"),
                    "message": ret.get("message") or "无标签（label 缺失）",
                }
            )
        else:
            failed_rows += 1
            row_total_sec = float(ret.get("_row_total_sec") or 0.0)
            if row_total_sec > 0:
                metric_row_total_sec_sum += row_total_sec
                if row_total_sec > metric_row_total_sec_max:
                    metric_row_total_sec_max = row_total_sec
            failure_records.append(
                {
                    "status": "fail",
                    "sv_file": ret.get("sv_file"),
                    "sv_row_idx": ret.get("sv_row_idx"),
                    "line": ret.get("line"),
                    "sn": ret.get("sn"),
                    "sample_id": ret.get("sample_id"),
                    "message": ret.get("message"),
                }
            )
            print(
                f"⚠️ 处理失败: {ret.get('sv_file')}#{ret.get('sv_row_idx')} "
                f"line={ret.get('line')} sn={ret.get('sn')} sample_id={ret.get('sample_id')} | "
                f"{ret.get('message')}"
            )

    def _consume_done(done_futures) -> int:
        processed_count = 0
        nonlocal failed_rows
        for fut in done_futures:
            group_size = pending_sizes.pop(fut, 1)
            try:
                ret_list = fut.result()
            except Exception as e:
                failed_rows += group_size
                failure_records.append(
                    {
                        "status": "fail",
                        "sv_file": "",
                        "sv_row_idx": "",
                        "line": "",
                        "sn": "",
                        "sample_id": "",
                        "message": f"worker 异常（整组 {group_size} 行失败）: {type(e).__name__}: {e}",
                    }
                )
                print(f"⚠️ worker 异常: {type(e).__name__}: {e}")
                processed_count += group_size
                continue
            if not isinstance(ret_list, list):
                ret_list = [ret_list]
            for ret in ret_list:
                _consume_result(ret)
                if ret.get("status") != "metric":
                    processed_count += 1
        return processed_count

    t_processing0 = time.perf_counter()
    rows_to_process = len(all_rows)
    processed_samples = 0
    progress_emit_step = max(1, rows_to_process // 200)
    last_emitted_progress = -progress_emit_step

    def _emit_feature_progress(*, force: bool = False) -> None:
        nonlocal last_emitted_progress
        if not force and processed_samples - last_emitted_progress < progress_emit_step:
            return
        last_emitted_progress = processed_samples
        print(
            f"[FEATURE_PROGRESS] processed={processed_samples} total={rows_to_process}",
            flush=True,
        )

    _emit_feature_progress(force=True)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor, tqdm(
        total=rows_to_process if rows_to_process > 0 else None,
        desc="总体进度",
        unit="row",
        dynamic_ncols=True,
    ) as pbar_all:
        for row_map in invalid_rows:
            _consume_result(
                {
                    "status": "fail",
                    "message": "line/sn/sample_id 为空",
                    "sv_file": _to_key(row_map.get("__sample_view_file")),
                    "sv_row_idx": row_map.get("__sample_view_row_index"),
                    "line": _to_key(row_map.get("line")),
                    "sn": _to_key(row_map.get("sn")),
                    "sample_id": _to_key(row_map.get("sample_id")),
                }
            )
            pbar_all.update(1)
            processed_samples += 1
            _emit_feature_progress()

        for (line, sn, storage_root, relative_path), rows in grouped_rows.items():
            fut = executor.submit(
                _process_one_group,
                line=line,
                sn=sn,
                feature_version=feature_version,
                group_storage_root=storage_root,
                group_relative_path=relative_path,
                rows=rows,
                sample_view_columns=sample_view_columns,
                data_root=data_root,
                sample_meta_lookup=sample_meta_lookup,
                manifest_lookup=manifest_lookup,
                label_lookup=label_lookup,
                cacheable_tdms_paths=cacheable_tdms_paths,
                tdms_cache=tdms_cache,
                tdms_cache_lock=tdms_cache_lock,
                row_workers=per_group_workers,
            )
            pending.add(fut)
            pending_sizes[fut] = len(rows)

            if len(pending) >= max_pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                completed = _consume_done(done)
                pbar_all.update(completed)
                processed_samples += completed
                _emit_feature_progress()

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            completed = _consume_done(done)
            pbar_all.update(completed)
            processed_samples += completed
            _emit_feature_progress()
    _emit_feature_progress(force=True)
    t_processing_sec = float(time.perf_counter() - t_processing0)

    t_flush0 = time.perf_counter()
    if batch:
        batch_index = _flush_batch(
            batch=batch,
            batch_index=batch_index,
            output_folder=output_feature_folder,
            output_file_prefix=args.output_file_prefix,
            fixed_columns=fixed_columns,
            feature_columns=feature_columns,
        )
        batch.clear()
    t_flush_sec = float(time.perf_counter() - t_flush0)
    wall_total_sec = float(time.perf_counter() - wall_t0)

    # 持久化失败/跳过行，便于训练前校验（training_dataset_guard）报错时排查
    # 注意：文件名不能含 "features"，否则会被 discover_feature_csvs 误识别为特征文件
    failed_rows_path = output_feature_folder / "failed_rows.csv"
    if failure_records:
        failure_columns = ["status", "sv_file", "sv_row_idx", "line", "sn", "sample_id", "message"]
        with failed_rows_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=failure_columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(failure_records)
        print(f"\n⚠️ 失败/跳过行明细已写入: {failed_rows_path}")

    print("\n处理完成")
    print(f"成功行数: {ok_rows}")
    print(f"跳过行数: {skipped_rows}")
    print(f"失败行数: {failed_rows}")
    print("------ 耗时统计 ------")
    print(f"总耗时                 : {wall_total_sec:.3f}s")
    print(f"metadata 索引耗时      : {t_meta_sec:.3f}s")
    print(f"sample_view 分组耗时   : {t_grouping_sec:.3f}s")
    print(f"并行处理耗时           : {t_processing_sec:.3f}s")
    print(f"批次写盘耗时           : {t_flush_sec:.3f}s")
    if metric_group_count > 0:
        print(f"group 数               : {metric_group_count}")
        print(f"group 总耗时(sum/max)  : {metric_group_total_sec_sum:.3f}s / {metric_group_total_sec_max:.3f}s")
        print(f"read_tdms(sum/max)     : {metric_tdms_read_sec_sum:.3f}s / {metric_tdms_read_sec_max:.3f}s")
        print(f"read_tdms cache 命中组 : {metric_tdms_cache_hits}")
        print(f"read_tdms 失败组数      : {metric_group_tdms_fail}")
    processed_rows = ok_rows + failed_rows
    if processed_rows > 0:
        print(f"单样本总耗时(sum/max)   : {metric_row_total_sec_sum:.3f}s / {metric_row_total_sec_max:.3f}s")
    if metric_extract_rows > 0:
        print(
            f"extract_features_{feature_version}: "
            f"rows={metric_extract_rows}, "
            f"sum={metric_extract_sec_sum:.3f}s, "
            f"avg={metric_extract_sec_sum / metric_extract_rows:.6f}s, "
            f"max={metric_extract_sec_max:.6f}s"
        )
        print(
            "  process_data(sum/avg/max): "
            f"{metric_extract_process_data_sec_sum:.3f}s / "
            f"{metric_extract_process_data_sec_sum / metric_extract_rows:.6f}s / "
            f"{metric_extract_process_data_sec_max:.6f}s"
        )
        print(
            "  mel_features(sum/avg/max): "
            f"{metric_extract_mel_sec_sum:.3f}s / "
            f"{metric_extract_mel_sec_sum / metric_extract_rows:.6f}s / "
            f"{metric_extract_mel_sec_max:.6f}s"
        )
        print(
            "  dwt_features(sum/avg/max): "
            f"{metric_extract_dwt_sec_sum:.3f}s / "
            f"{metric_extract_dwt_sec_sum / metric_extract_rows:.6f}s / "
            f"{metric_extract_dwt_sec_max:.6f}s"
        )
    if ok_rows == 0:
        print("⚠️ 没有成功写出任何特征数据，请检查 sample_view 与 metadata/label_history 的对应关系。")


# ===========================================================================
# Data contracts（原 types.py）
# ===========================================================================

@dataclass
class FeatureDataset:
    """Feature matrix, target labels, and leakage-control metadata."""

    X: pd.DataFrame
    y: np.ndarray
    metadata: pd.DataFrame
    feature_names: list[str]
    feature_version: str


@dataclass
class FeatureDatasetSplit:
    train: FeatureDataset
    validation: FeatureDataset
    test: FeatureDataset
    requested_strategy: str
    resolved_strategy: str
    report: dict


# ===========================================================================
# Sample-view / feature alignment guard（原 guard.py）
# ===========================================================================

BASE_KEY_COLUMNS = (
    "line",
    "sn",
    "sample_id",
    "group_name",
    "channel_name",
)

OPTIONAL_KEY_COLUMNS = (
    "tdms_storage_root",
    "relative_path",
)


def _to_key_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _format_key(key_columns: tuple[str, ...], key: tuple[str, ...]) -> str:
    parts = [f"{col}={value}" for col, value in zip(key_columns, key)]
    return ", ".join(parts)


def _discover_duplicates(counter: Counter[tuple[str, ...]]) -> list[tuple[tuple[str, ...], int]]:
    return [(key, count) for key, count in counter.items() if count > 1]


def _counter_preview(
    counter: Counter[tuple[str, ...]],
    *,
    key_columns: tuple[str, ...],
    limit: int = 3,
) -> list[str]:
    items = sorted(counter.items(), key=lambda item: item[0])[:limit]
    return [f"{_format_key(key_columns, key)} (count={count})" for key, count in items]


def _read_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return tuple(reader.fieldnames or ())


def _resolve_key_columns(sample_view_path: Path, feature_paths: list[Path]) -> tuple[str, ...]:
    sample_view_columns = set(_read_header(sample_view_path))
    feature_column_sets = [set(_read_header(path)) for path in feature_paths]
    required_missing = sorted(set(BASE_KEY_COLUMNS) - sample_view_columns)
    if required_missing:
        raise ValueError(f"{sample_view_path} 缺少主键列: {required_missing}")
    for path, columns in zip(feature_paths, feature_column_sets):
        missing = sorted(set(BASE_KEY_COLUMNS) - columns)
        if missing:
            raise ValueError(f"{path} 缺少主键列: {missing}")

    common_optional = [
        col
        for col in OPTIONAL_KEY_COLUMNS
        if col in sample_view_columns and all(col in columns for columns in feature_column_sets)
    ]
    return tuple(BASE_KEY_COLUMNS) + tuple(common_optional)


def _read_key_counter(
    path: Path,
    *,
    key_columns: tuple[str, ...],
    required_columns: tuple[str, ...] = BASE_KEY_COLUMNS,
) -> tuple[int, Counter[tuple[str, ...]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = tuple(reader.fieldnames or ())
        missing = sorted(set(required_columns) - set(columns))
        if missing:
            raise ValueError(f"{path} 缺少主键列: {missing}")

        total_rows = 0
        invalid_rows: list[int] = []
        counter: Counter[tuple[str, ...]] = Counter()
        for row_idx, row in enumerate(reader, start=2):
            total_rows += 1
            required_values = [_to_key_text(row.get(col)) for col in required_columns]
            if any(not value for value in required_values):
                if len(invalid_rows) < 5:
                    invalid_rows.append(row_idx)
                continue
            key = tuple(_to_key_text(row.get(col)) for col in key_columns)
            counter[key] += 1

    if invalid_rows:
        raise ValueError(
            f"{path} 存在主键必填列为空的行，示例行号: {invalid_rows}"
        )
    return total_rows, counter


def discover_feature_csvs(results_path: str | Path) -> list[str]:
    dataset_dir = Path(results_path).expanduser() / "dataset_csv"
    if not dataset_dir.exists():
        return []
    files = [
        str(path)
        for path in sorted(dataset_dir.glob("*.csv"))
        if path.is_file()
        and not path.name.startswith(".")
        and "features" in path.name.lower()
        and path.name != "features_batch_augmented.csv"
    ]
    return files


def discover_augmented_feature_csvs(results_path: str | Path) -> list[str]:
    augmented_path = (
        Path(results_path).expanduser()
        / "dataset_csv"
        / "features_batch_augmented.csv"
    )
    return [str(augmented_path)] if augmented_path.is_file() else []


def validate_sample_view_features_alignment(
    results_path: str | Path,
    feature_files: Iterable[str | Path],
) -> dict[str, int]:
    results_dir = Path(results_path).expanduser()
    dataset_dir = results_dir / "dataset_csv"
    sample_view_path = dataset_dir / "sample_view.csv"
    if not sample_view_path.exists():
        raise FileNotFoundError(f"训练前校验失败：sample_view 不存在: {sample_view_path}")

    feature_paths = [Path(path).expanduser() for path in feature_files]
    if not feature_paths:
        raise RuntimeError(f"训练前校验失败：未找到特征 CSV，目录: {dataset_dir}")

    key_columns = _resolve_key_columns(sample_view_path, feature_paths)
    sample_total, sample_counter = _read_key_counter(sample_view_path, key_columns=key_columns)
    feature_total = 0
    feature_counter: Counter[tuple[str, ...]] = Counter()
    for path in feature_paths:
        total, counter = _read_key_counter(path, key_columns=key_columns)
        feature_total += total
        feature_counter.update(counter)

    sample_duplicates = _discover_duplicates(sample_counter)
    if sample_duplicates:
        preview = [f"{_format_key(key_columns, key)} (count={count})" for key, count in sample_duplicates[:3]]
        raise RuntimeError(
            "训练前校验失败：sample_view 存在重复主键，"
            f"columns={list(key_columns)}, unique={len(sample_counter)}, total={sample_total}，示例: {preview}"
        )

    feature_duplicates = _discover_duplicates(feature_counter)
    if feature_duplicates:
        preview = [f"{_format_key(key_columns, key)} (count={count})" for key, count in feature_duplicates[:3]]
        raise RuntimeError(
            "训练前校验失败：features 存在重复主键，"
            f"columns={list(key_columns)}, unique={len(feature_counter)}, total={feature_total}，示例: {preview}。"
            "常见原因：dataset_csv 混入了多次/并发提取的特征分片（features_batch_*.csv 编号不连续即可确认）。"
            "处理：重新执行一次「提取特征」并等待其完整结束后再训练。"
        )

    if sample_total != feature_total:
        raise RuntimeError(
            "训练前校验失败：sample_view 与 features 行数不一致，"
            f"sample_view={sample_total}, features={feature_total}"
        )

    if sample_counter != feature_counter:
        missing_counter = sample_counter - feature_counter
        extra_counter = feature_counter - sample_counter
        details: list[str] = []
        if missing_counter:
            details.append(
                f"features 缺少样本: {_counter_preview(missing_counter, key_columns=key_columns)}"
            )
        if extra_counter:
            details.append(
                f"features 多出样本: {_counter_preview(extra_counter, key_columns=key_columns)}"
            )
        detail_text = "；".join(details) if details else "主键集合不一致"
        raise RuntimeError(
            f"训练前校验失败：sample_view 与 features 主键不一致（columns={list(key_columns)}）。{detail_text}"
        )

    return {
        "sample_view_rows": sample_total,
        "feature_rows": feature_total,
        "unique_keys": len(sample_counter),
        "feature_files": len(feature_paths),
        "key_columns": list(key_columns),
    }


# ===========================================================================
# High-level API（原 api.py）
# ===========================================================================

def _feature_files_in_dir(output_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(output_dir.glob("*.csv"))
        if path.is_file() and "features" in path.name.lower()
    ]


def load_xy(config: dict, output_dir: str | Path | None = None) -> FeatureDataset:
    """Load the generated feature batches into one typed dataset object."""
    from ml.training.config import resolve_feature_columns_by_schema

    dataset_dir = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else resolve_dataset_output_dir(config)
    )
    files = _feature_files_in_dir(dataset_dir)
    if not files:
        raise FileNotFoundError(f"No feature batches found: {dataset_dir}")
    frame = pd.concat(
        [pd.read_csv(path, encoding="utf-8-sig") for path in files],
        ignore_index=True,
    )
    if "label" not in frame.columns:
        raise ValueError("Feature dataset is missing the training label column: label")
    feature_names = resolve_feature_columns_by_schema(frame)
    X = frame[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(frame["label"], errors="raise").astype(int).to_numpy()
    metadata = frame.drop(columns=feature_names, errors="ignore").copy()
    return FeatureDataset(
        X=X,
        y=np.asarray(y, dtype=int),
        metadata=metadata,
        feature_names=feature_names,
        feature_version=resolve_feature_version(config),
    )


def build_xy(
    config_path: str | Path,
    *,
    sample_views: Sequence[str | Path] = (),
    output_dir: str | Path | None = None,
    num_workers: int | None = None,
) -> FeatureDataset:
    """Run TDMS feature extraction and return ``X``, ``y``, and metadata."""
    config_file = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    command = [sys.executable, "-m", "ml.dataset.build", "--config", str(config_file)]
    for sample_view in sample_views:
        command.extend(["--sample-view", str(Path(sample_view).expanduser())])
    if output_dir is not None:
        command.extend(["--output-feature-folder", str(Path(output_dir).expanduser())])
    if num_workers is not None:
        command.extend(["--num-workers", str(int(num_workers))])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    return load_xy(config, output_dir)


# ===========================================================================
# Sample-view generation（原 generate.py）
# ===========================================================================

def generate(cfg: dict, output_folder: str | Path | None = None) -> Path:
    """Build the final labeled sample_view.csv from filter_samples + label rules."""
    from data_manager.label_database import load_label_dataframe
    from ml.dataset.label_filter import filter_sample_view_dataframe
    from ml.dataset.sample_filter import filter_samples

    def _database_path(cfg: dict) -> Path:
        from data_manager.label_database import resolve_database_path
        database = cfg.get("database") if isinstance(cfg.get("database"), dict) else {}
        dataset = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
        raw = (
            database.get("label_records_db_path")
            or cfg.get("label_records_db_path")
            or dataset.get("label_records_db_path")
            or ""
        )
        return resolve_database_path(str(raw))

    candidate_path = filter_samples(cfg, output_folder)
    candidates = pd.read_csv(candidate_path, encoding="utf-8-sig")
    from ml.dataset.label_filter import LABEL_FILTER_STATUSES
    labels = load_label_dataframe(_database_path(cfg), statuses=LABEL_FILTER_STATUSES)
    filtered, _stats = filter_sample_view_dataframe(candidates, labels, cfg)
    filtered.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    print(
        f"[ML TRAINING DATA] {candidate_path} | "
        f"candidates={len(candidates)} | kept={len(filtered)}"
    )
    return candidate_path


if __name__ == "__main__":
    main()
