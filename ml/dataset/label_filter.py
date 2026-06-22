#!/usr/bin/env python3
"""ML training-label selection policy.

根据 label_history 对 sample_view.csv 重新筛选训练标签：

标签级别与规则：
1. 专家：只要存在 expert 标签，使用最新的 expert 标签。
2. 员工多个且一致：无 expert，至少 2 条员工标签且全部一致，使用最新标签。
3. 员工多个但不一致：不产生训练 y。
4. 员工单个：不产生训练 y。
5. 无专家/员工标注：不产生训练 y。

默认直接覆盖 ML 输出目录下的 sample_view.csv。
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_manager.label_database import load_label_dataframe
from data_manager.config import load_data_manager_config
from data_manager.label_internal_registry import normalize_label_source_category
from ml.training.config import resolve_dataset_output_dir

DM_CFG = load_data_manager_config()

LABEL_SIGNAL_COLS = (
    "result_key",
    "result_id",
    "result_name",
    "reason_key",
    "reason_id",
    "reason_name",
)

LABEL_SIGNATURE_COLS = (
    "result_key",
    "result_id",
    "result_name",
    "reason_key",
    "reason_id",
    "reason_name",
)

SAMPLE_VIEW_LABEL_COLS = [
    "result_key",
    "result_id",
    "result_name",
    "reason_key",
    "reason_id",
    "reason_name",
    "label_version",
    "note",
    "label_timestamp",
    "label_source",
]

# sample_view.csv 的唯一标准格式。DL、ML 和 sample_view 生成器共用此定义。
STANDARD_SAMPLE_VIEW_COLUMNS = [
    "view_name",
    "line",
    "sn",
    "sample_id",
    "group_name",
    "channel_name",
    "sampling_rate",
    "reference",
    "time",
    "tdms_storage_root",
    "relative_path",
    *SAMPLE_VIEW_LABEL_COLS,
]

REQUIRED_SAMPLE_VIEW_COLUMNS = ("line", "sn", "sample_id")
MIN_CONSISTENT_EMPLOYEE_LABELS = 2


def standardize_sample_view(df: pd.DataFrame) -> pd.DataFrame:
    """补齐并按标准顺序输出 sample_view。

    训练样本表只保留标准列，避免 DL/ML 因生成入口不同产生不同格式。
    """
    out = df.copy()
    missing_required = [col for col in REQUIRED_SAMPLE_VIEW_COLUMNS if col not in out.columns]
    if missing_required:
        raise ValueError(f"sample_view 缺少主键列: {missing_required}")
    for col in STANDARD_SAMPLE_VIEW_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[STANDARD_SAMPLE_VIEW_COLUMNS].copy()


def _norm(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _parse_dt(text: object) -> datetime:
    raw = _norm(text)
    if not raw:
        return datetime.min
    formats = (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
    )
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return datetime.min


def _read_yaml_cfg(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser()
    if not config_path.exists() and not config_path.is_absolute():
        cfg_candidate = Path("cfg") / config_path
        if cfg_candidate.exists():
            config_path = cfg_candidate
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件不是字典结构: {config_path}")
    return cfg


def _resolve_output_folder(cfg: dict[str, Any]) -> Path:
    return resolve_dataset_output_dir(cfg)


def _resolve_data_root(cfg: dict[str, Any]) -> Path:
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
    data_root_raw = cfg.get("data_root") or dataset_cfg.get("data_root") or DM_CFG.get("data_root")
    if not data_root_raw:
        raise ValueError("缺少 data_root，请在 YAML 或 cfg/core/data_manager.yaml 中配置")
    return Path(str(data_root_raw)).expanduser()


def _normalize_path_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (str, int, float)):
        items = [raw]
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _norm(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _key_sns_paths_from_cfg(cfg: dict[str, Any]) -> list[str]:
    data_cfg = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    dataset_cfg = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    paths: list[str] = []
    for container in (data_cfg, dataset_cfg):
        key_sns = container.get("key_sns")
        if isinstance(key_sns, list):
            paths.extend(
                str(item.get("path") or item.get("folder") or item.get("dir") or "").strip()
                if isinstance(item, dict) else str(item or "").strip()
                for item in key_sns
            )
        else:
            paths.extend(_normalize_path_list(key_sns))
        paths.extend(_normalize_path_list(container.get("key_sns_sample_view")))
        paths.extend(_normalize_path_list(container.get("key_sns_sample_views")))
    return _normalize_path_list(paths)


def _count_key_sns_rows(df: pd.DataFrame) -> int:
    if df.empty or "relative_path" not in df.columns:
        return 0
    rel = df["relative_path"].fillna("").astype(str).str.replace("\\", "/", regex=False).str.strip("/")
    return int(rel.str.contains(r"(?:^|/)key_sns(?:/|$)", regex=True).sum())


def _has_effective_label_fields(row: dict[str, Any]) -> bool:
    for col in LABEL_SIGNAL_COLS:
        if _norm(row.get(col)):
            return True
    return False


def _first_present_text(*values: object) -> str:
    for value in values:
        text = _norm(value)
        if text:
            return text
    return ""


def _normalized_label_payload(row: dict[str, Any]) -> dict[str, str]:
    result_key = _norm(row.get("result_key"))
    result_id = _first_present_text(row.get("result_id"))
    result_name = _norm(row.get("result_name"))
    reason_key = _norm(row.get("reason_key"))
    reason_id = _first_present_text(row.get("reason_id"))
    reason_name = _norm(row.get("reason_name"))

    if not result_key:
        result_key = result_name
    if not result_name:
        result_name = result_key
    if not reason_key:
        reason_key = reason_name
    if not reason_name:
        reason_name = reason_key

    return {
        "result_key": result_key,
        "result_id": result_id,
        "result_name": result_name,
        "reason_key": reason_key,
        "reason_id": reason_id,
        "reason_name": reason_name,
        "label_version": _norm(row.get("label_version")),
        "note": _norm(row.get("note")),
        "label_timestamp": _norm(row.get("timestamp") or row.get("label_timestamp")),
        "label_source": _norm(row.get("source") or row.get("label_source")),
    }


def _label_signature(row: dict[str, Any]) -> tuple[str, ...]:
    payload = _normalized_label_payload(row)
    return tuple(payload.get(col, "") for col in LABEL_SIGNATURE_COLS)


def _build_label_row_maps(
    label_df: pd.DataFrame,
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]]]:
    by_triplet: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}

    if label_df.empty:
        return by_triplet, by_pair

    for idx, row in enumerate(label_df.to_dict(orient="records")):
        line = _norm(row.get("line"))
        sn = _norm(row.get("sn"))
        sample_id = _norm(row.get("sample_id"))
        if not sn or not sample_id:
            continue
        if not _has_effective_label_fields(row):
            continue

        source_text = _norm(row.get("source"))
        row_map: dict[str, Any] = dict(row)
        row_map["_row_index"] = int(idx)
        row_map["_source_category"] = normalize_label_source_category(source_text)
        row_map["_timestamp_dt"] = _parse_dt(row.get("timestamp"))
        row_map["_label_signature"] = _label_signature(row_map)

        if line:
            by_triplet.setdefault((line, sn, sample_id), []).append(row_map)
        else:
            by_pair.setdefault((sn, sample_id), []).append(row_map)

    return by_triplet, by_pair


def _pick_latest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row.get("_timestamp_dt") or datetime.min,
            int(row.get("_row_index") or 0),
        ),
    )


def _pick_training_label(rows: list[dict[str, Any]]) -> tuple[dict[str, str] | None, str]:
    if not rows:
        return None, "no_manual_label"

    expert_rows = [row for row in rows if row.get("_source_category") == "expert"]
    if expert_rows:
        return _normalized_label_payload(_pick_latest(expert_rows)), "expert"

    employee_rows = [row for row in rows if row.get("_source_category") == "operator"]
    if not employee_rows:
        return None, "no_manual_label"
    if len(employee_rows) < MIN_CONSISTENT_EMPLOYEE_LABELS:
        return None, "operator_single"

    signatures = {tuple(row.get("_label_signature") or ()) for row in employee_rows}
    if len(signatures) != 1:
        return None, "operator_conflict"

    return _normalized_label_payload(_pick_latest(employee_rows)), "operator_consistent"


def _inc(counter: dict[str, int], key: str) -> None:
    counter[key] = int(counter.get(key, 0)) + 1


def _sorted_counts(counter: dict[str, int], key_name: str) -> list[dict[str, Any]]:
    items = sorted(counter.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return [{key_name: k, "count": int(v)} for k, v in items]


def _label_target_map_from_cfg(cfg: dict[str, Any]) -> dict[str, int]:
    """从 train.label_mapping.groups 取每个 reason 的目标数量（仅 >=0 的才参与下采样）。"""
    train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    lm = train_cfg.get("label_mapping") if isinstance(train_cfg.get("label_mapping"), dict) else {}
    if not lm or not lm.get("enable", False):
        return {}

    targets: dict[str, int] = {}
    for group in lm.get("groups") or []:
        if not isinstance(group, dict):
            continue
        for token in group.get("reasons") or []:
            if not isinstance(token, dict):
                continue
            name = _norm(token.get("reason"))
            try:
                target = int(token.get("sample"))
            except (TypeError, ValueError):
                continue
            if name and target >= 0:
                targets[name] = target
    return targets


def _extract_all_features_from_cfg(cfg: dict[str, Any]) -> bool:
    train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    mapping_cfg = train_cfg.get("label_mapping") if isinstance(train_cfg.get("label_mapping"), dict) else {}
    return bool(mapping_cfg.get("extract_all_features", False))


def _downsample_by_label_targets(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """筛选后按目标数量下采样，让 sample_view 与后续提取的 features 一一对应。

    - 目标数量 >= 0 且可用数超标的 reason：随机下采样到目标数量（固定 seed 可复现）。
    - 目标 -1（全取）、可用数不足目标、不在 label_mapping 的行：保持不变。
    - 上采样（目标 > 可用数）仍由训练侧 rebalance 随机复制补齐。
    """
    targets = _label_target_map_from_cfg(cfg)
    if _extract_all_features_from_cfg(cfg):
        print("[采样] 已启用全量提取特征：保留筛选后的全部样本，训练目标外的剩余样本由训练侧并入测试集")
        return df, [
            {"reason": name, "available": int((df["reason_name"].map(_norm) == name).sum()), "kept": int((df["reason_name"].map(_norm) == name).sum()), "target": target}
            for name, target in sorted(targets.items())
        ]
    if not targets or df.empty or "reason_name" not in df.columns:
        return df, []

    names = df["reason_name"].map(_norm)
    keep_mask = ~names.isin(targets.keys())
    parts = [df[keep_mask]]
    stats: list[dict[str, Any]] = []

    for name in sorted(targets):
        target = targets[name]
        sub = df[names == name]
        if len(sub) > target:
            picked = sub.sample(n=target, random_state=int(seed))
            print(f"[采样] {name}: {len(sub)} -> {target}（按目标数量下采样）")
        else:
            picked = sub
            if len(sub) < target:
                print(f"[采样] {name}: {len(sub)} <= 目标 {target}，全取（上采样留给训练侧）")
        parts.append(picked)
        stats.append({"reason": name, "available": int(len(sub)), "kept": int(len(picked)), "target": target})

    out = pd.concat(parts).sort_index()
    if len(out) != len(df):
        print(f"[采样] sample_view 行数: {len(df)} -> {len(out)}")
    return out, stats


def _build_distribution_payload(filtered_df: pd.DataFrame) -> dict[str, Any]:
    result_counter: dict[str, int] = {}
    reason_counter: dict[str, int] = {}
    line_counter: dict[str, int] = {}
    line_result_counter: dict[tuple[str, str], int] = {}

    result_meta: dict[str, dict[str, str]] = {}
    reason_meta: dict[str, dict[str, str]] = {}

    for row in filtered_df.to_dict(orient="records"):
        line = _norm(row.get("line")) or "<empty>"

        result_key = _norm(row.get("result_key") or row.get("result_name"))
        result_name = _norm(row.get("result_name") or result_key)
        result_id = _first_present_text(row.get("result_id"))
        if not result_key:
            result_key = "<empty>"
        if not result_name:
            result_name = result_key

        reason_key = _norm(row.get("reason_key") or row.get("reason_name"))
        reason_name = _norm(row.get("reason_name") or reason_key)
        reason_id = _first_present_text(row.get("reason_id"))
        if not reason_key:
            reason_key = "<empty>"
        if not reason_name:
            reason_name = reason_key

        result_meta.setdefault(
            result_key,
            {
                "result_key": result_key,
                "result_id": result_id,
                "result_name": result_name,
            },
        )
        reason_meta.setdefault(
            reason_key,
            {
                "reason_key": reason_key,
                "reason_id": reason_id,
                "reason_name": reason_name,
                "parent_result_key": result_key,
                "parent_result_name": result_name,
            },
        )

        _inc(result_counter, result_key)
        _inc(reason_counter, reason_key)
        _inc(line_counter, line)
        line_result_counter[(line, result_key)] = int(line_result_counter.get((line, result_key), 0)) + 1

    result_distribution = [
        {
            "result": meta["result_name"],
            "result_detail": f"{meta['result_name']}（{meta['result_key']}，{meta['result_id'] or '-'}）",
            "result_key": meta["result_key"],
            "result_id": meta["result_id"],
            "count": int(result_counter.get(key, 0)),
        }
        for key, meta in sorted(result_meta.items(), key=lambda kv: (-int(result_counter.get(kv[0], 0)), kv[0]))
    ]
    reason_distribution = [
        {
            "reason": meta["reason_name"],
            "reason_detail": f"{meta['reason_name']}（{meta['reason_key']}，{meta['reason_id'] or '-'}）",
            "parent_result_key": meta["parent_result_key"],
            "parent_result_name": meta["parent_result_name"],
            "reason_key": meta["reason_key"],
            "reason_id": meta["reason_id"],
            "count": int(reason_counter.get(key, 0)),
        }
        for key, meta in sorted(reason_meta.items(), key=lambda kv: (-int(reason_counter.get(kv[0], 0)), kv[0]))
    ]
    line_result_distribution = [
        {"line": line, "result": result_key, "count": int(count)}
        for (line, result_key), count in sorted(
            line_result_counter.items(),
            key=lambda kv: (-int(kv[1]), str(kv[0][0]), str(kv[0][1])),
        )
    ]

    return {
        "result_distribution": result_distribution,
        "reason_distribution": reason_distribution,
        "line_distribution": _sorted_counts(line_counter, "line"),
        "line_result_distribution": line_result_distribution,
    }


def filter_sample_view_dataframe(
    sample_df: pd.DataFrame,
    label_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """为 DL/ML 统一筛选训练标签，并返回标准格式 sample_view。"""
    for col in REQUIRED_SAMPLE_VIEW_COLUMNS:
        if col not in sample_df.columns:
            raise ValueError(f"sample_view 缺少列: {col}")
    for col in ("sn", "sample_id"):
        if col not in label_df.columns:
            raise ValueError(f"label_history 缺少列: {col}")

    by_triplet, by_pair = _build_label_row_maps(label_df)
    kept_rows: list[dict[str, Any]] = []
    decision_counter: Counter[str] = Counter()
    pair_fallback_rows = 0

    for row in sample_df.to_dict(orient="records"):
        key = (_norm(row.get("line")), _norm(row.get("sn")), _norm(row.get("sample_id")))
        candidates = by_triplet.get(key)
        if not candidates:
            candidates = by_pair.get((key[1], key[2]), [])
            if candidates:
                pair_fallback_rows += 1
        chosen_label, decision = _pick_training_label(candidates)
        decision_counter[decision] += 1
        if not chosen_label:
            continue
        row_out = dict(row)
        row_out.update({col: chosen_label.get(col, "") for col in SAMPLE_VIEW_LABEL_COLS})
        kept_rows.append(row_out)

    filtered_df = pd.DataFrame(kept_rows, columns=list(sample_df.columns) + [
        col for col in SAMPLE_VIEW_LABEL_COLS if col not in sample_df.columns
    ])
    available_distribution = _build_distribution_payload(filtered_df)
    filtered_rows_before_sampling = int(len(filtered_df))
    train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    try:
        sampling_seed = int(train_cfg.get("split_seed"))
    except (TypeError, ValueError):
        sampling_seed = 42
    filtered_df, sampling_stats = _downsample_by_label_targets(filtered_df, cfg, seed=sampling_seed)
    filtered_df = standardize_sample_view(filtered_df)

    stats: dict[str, Any] = {
        "raw_sample_rows": int(len(sample_df)),
        "filtered_rows_before_sampling": filtered_rows_before_sampling,
        "kept_rows": int(len(filtered_df)),
        "line_empty_fallback_rows": int(pair_fallback_rows),
        "decision_counter": decision_counter,
        "label_target_sampling": sampling_stats,
    }
    stats.update(available_distribution)
    return filtered_df, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按专家/员工多人一致规则筛选训练 y 标签")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    parser.add_argument("--sample-view", default="", help="输入 sample_view.csv 路径，默认按配置推断")
    parser.add_argument("--label-records-db", default="", help="可选：label_records.db 路径")
    parser.add_argument("--output-sample-view", default="", help="输出 sample_view.csv 路径，默认覆盖输入 sample_view.csv")
    parser.add_argument("--json", action="store_true", help="输出 JSON 汇总")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _read_yaml_cfg(Path(args.config))
    output_folder = _resolve_output_folder(cfg)
    data_root = _resolve_data_root(cfg)

    sample_view_path = Path(args.sample_view).expanduser() if _norm(args.sample_view) else (output_folder / "sample_view.csv")
    label_records_db_path = Path(args.label_records_db).expanduser() if _norm(args.label_records_db) else (data_root / "metadata" / "label_records.db")
    output_sample_view_path = (
        Path(args.output_sample_view).expanduser()
        if _norm(args.output_sample_view)
        else sample_view_path
    )

    if not sample_view_path.exists() or not sample_view_path.is_file():
        raise FileNotFoundError(f"sample_view.csv 不存在: {sample_view_path}")
    sample_df = pd.read_csv(sample_view_path, encoding="utf-8-sig")
    label_df = load_label_dataframe(label_records_db_path)
    key_sns_paths = _key_sns_paths_from_cfg(cfg)
    key_sns_folder_paths = [path for path in key_sns_paths if Path(path).expanduser().suffix.lower() != ".csv"]
    key_sns_csv_paths = [path for path in key_sns_paths if Path(path).expanduser().suffix.lower() == ".csv"]
    raw_key_sns_rows = _count_key_sns_rows(sample_df)

    total_rows = int(len(sample_df))
    filtered_df, filter_stats = filter_sample_view_dataframe(sample_df, label_df, cfg)
    decision_counter = filter_stats["decision_counter"]
    pair_fallback_rows = int(filter_stats["line_empty_fallback_rows"])
    filtered_rows_before_sampling = int(filter_stats["filtered_rows_before_sampling"])
    sampling_stats = filter_stats["label_target_sampling"]

    output_sample_view_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_sample_view_path, index=False, encoding="utf-8-sig")

    kept_count = int(len(filtered_df))
    dropped_count = max(0, total_rows - kept_count)
    kept_key_sns_rows = _count_key_sns_rows(filtered_df)
    payload: dict[str, Any] = {
        "sample_view_path": str(output_sample_view_path),
        "label_records_db_path": str(label_records_db_path),
        "key_sns_paths": key_sns_paths,
        "key_sns_folder_paths": key_sns_folder_paths,
        "key_sns_csv_paths": key_sns_csv_paths,
        "key_sns_path_count": int(len(key_sns_paths)),
        "key_sns_folder_count": int(len(key_sns_folder_paths)),
        "key_sns_csv_count": int(len(key_sns_csv_paths)),
        "raw_key_sns_rows": int(raw_key_sns_rows),
        "kept_key_sns_rows": int(kept_key_sns_rows),
        "dropped_key_sns_rows": max(0, int(raw_key_sns_rows) - int(kept_key_sns_rows)),
        "raw_sample_rows": total_rows,
        "kept_rows": kept_count,
        "dropped_rows": dropped_count,
        "sample_rows": total_rows,
        "matched_rows": kept_count,
        "unmatched_rows": dropped_count,
        "expert_kept_rows": int(decision_counter.get("expert", 0)),
        "operator_consistent_kept_rows": int(decision_counter.get("operator_consistent", 0)),
        "dropped_single_operator_rows": int(decision_counter.get("operator_single", 0)),
        "dropped_conflicting_operator_rows": int(decision_counter.get("operator_conflict", 0)),
        "dropped_no_manual_label_rows": int(decision_counter.get("no_manual_label", 0)),
        "line_empty_fallback_rows": int(pair_fallback_rows),
        "selection_distribution": [
            {"decision": "expert", "count": int(decision_counter.get("expert", 0))},
            {"decision": "operator_consistent", "count": int(decision_counter.get("operator_consistent", 0))},
            {"decision": "operator_single", "count": int(decision_counter.get("operator_single", 0))},
            {"decision": "operator_conflict", "count": int(decision_counter.get("operator_conflict", 0))},
            {"decision": "no_manual_label", "count": int(decision_counter.get("no_manual_label", 0))},
        ],
    }
    payload["filtered_rows_before_sampling"] = filtered_rows_before_sampling
    payload["label_target_sampling"] = sampling_stats
    payload.update({
        key: value for key, value in filter_stats.items()
        if key.endswith("_distribution") and key != "decision_counter"
    })

    print(f"sample_view 输入        : {sample_view_path}")
    print(f"label_records_db      : {label_records_db_path}")
    print(f"sample_view 输出       : {output_sample_view_path}")
    print(f"输入 sample_view 行数   : {total_rows}")
    print(f"保留样本数量            : {kept_count}")
    print(f"删除样本数量            : {dropped_count}")
    print(f"data.key_sns 路径数量   : {len(key_sns_paths)}")
    print(f"key_sns 原始样本行数    : {raw_key_sns_rows}")
    print(f"key_sns 保留样本行数    : {kept_key_sns_rows}")
    print(f"expert 保留数量         : {payload['expert_kept_rows']}")
    print(f"员工多标且一致保留 : {payload['operator_consistent_kept_rows']}")
    print(f"员工单标删除数量     : {payload['dropped_single_operator_rows']}")
    print(f"员工多标不一致删除 : {payload['dropped_conflicting_operator_rows']}")
    print(f"无专家/员工标注删除 : {payload['dropped_no_manual_label_rows']}")
    print(f"line 为空回退匹配数量   : {payload['line_empty_fallback_rows']}")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
