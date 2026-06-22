"""DL 数据集构建（路径推导 + 样本筛选 + 标签筛选 + 生成 + 特征提取 CLI）。

合并自：
  training_data/paths.py          — resolve_output_dir
  training_data/label_filter.py   — 标签筛选规则
  training_data/sample_filter.py  — 样本候选筛选
  training_data/generate.py       — generate()
  main_dataset.py                 — 两步 CLI main()
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from data_manager.config import load_data_manager_config
from data_manager.label_database import (
    load_label_dataframe,
    load_sample_dataframe,
    resolve_database_path,
)
from data_manager.label_internal_registry import normalize_label_source_category


# ===========================================================================
# 路径推导（原 training_data/paths.py）
# ===========================================================================

def resolve_output_dir(cfg: dict) -> Path:
    """DL 专属输出目录：results/{line}_{model}_{dl.model_type}/dl_dataset_csv/"""
    line = str(cfg.get("line_name") or "line").strip()
    model = str((cfg.get("model") or {}).get("model_name") or "model").strip()
    dl_cfg = cfg.get("dl") if isinstance(cfg.get("dl"), dict) else {}
    model_type = str(dl_cfg.get("model_type") or "cnn").strip()
    results = Path(str(cfg.get("results_path") or "./results")).expanduser()
    return results / f"{line}_{model}_{model_type}" / "dl_dataset_csv"


# ===========================================================================
# 标签筛选（原 training_data/label_filter.py）
# ===========================================================================

DM_CFG = load_data_manager_config()

LABEL_SIGNAL_COLS = (
    "result_key", "result_id", "result_name",
    "reason_key", "reason_id", "reason_name",
)
LABEL_SIGNATURE_COLS = LABEL_SIGNAL_COLS

SAMPLE_VIEW_LABEL_COLS = [
    "result_key", "result_id", "result_name",
    "reason_key", "reason_id", "reason_name",
    "label_version", "note", "label_timestamp", "label_source",
]

STANDARD_SAMPLE_VIEW_COLUMNS = [
    "view_name", "line", "sn", "sample_id", "group_name", "channel_name",
    "sampling_rate", "reference", "time", "tdms_storage_root", "relative_path",
    *SAMPLE_VIEW_LABEL_COLS,
]

REQUIRED_SAMPLE_VIEW_COLUMNS = ("line", "sn", "sample_id")
MIN_CONSISTENT_EMPLOYEE_LABELS = 2


def standardize_sample_view(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = [col for col in REQUIRED_SAMPLE_VIEW_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"sample_view 缺少主键列: {missing}")
    for col in STANDARD_SAMPLE_VIEW_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[STANDARD_SAMPLE_VIEW_COLUMNS].copy()


def _norm(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _parse_dt(text: object) -> datetime:
    raw = _norm(text)
    if not raw:
        return datetime.min
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d", "%Y/%m/%d"):
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


def _resolve_data_root(cfg: dict[str, Any]) -> Path:
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
    data_root_raw = cfg.get("data_root") or dataset_cfg.get("data_root") or DM_CFG.get("data_root")
    if not data_root_raw:
        raise ValueError("缺少 data_root，请在 YAML 或 cfg/core/data_manager.yaml 中配置")
    return Path(str(data_root_raw)).expanduser()


def _normalize_path_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    items = [raw] if isinstance(raw, (str, int, float)) else list(raw) if isinstance(raw, (list, tuple, set)) else []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _norm(item)
        if text and text not in seen:
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
    return any(_norm(row.get(col)) for col in LABEL_SIGNAL_COLS)


def _first_present_text(*values: object) -> str:
    for value in values:
        text = _norm(value)
        if text:
            return text
    return ""


def _normalized_label_payload(row: dict[str, Any]) -> dict[str, str]:
    result_key = _norm(row.get("result_key")) or _norm(row.get("result_name"))
    result_name = _norm(row.get("result_name")) or result_key
    reason_key = _norm(row.get("reason_key")) or _norm(row.get("reason_name"))
    reason_name = _norm(row.get("reason_name")) or reason_key
    return {
        "result_key": result_key,
        "result_id": _first_present_text(row.get("result_id")),
        "result_name": result_name,
        "reason_key": reason_key,
        "reason_id": _first_present_text(row.get("reason_id")),
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
        if not sn or not sample_id or not _has_effective_label_fields(row):
            continue
        row_map: dict[str, Any] = dict(row)
        row_map["_row_index"] = int(idx)
        row_map["_source_category"] = normalize_label_source_category(_norm(row.get("source")))
        row_map["_timestamp_dt"] = _parse_dt(row.get("timestamp"))
        row_map["_label_signature"] = _label_signature(row_map)
        if line:
            by_triplet.setdefault((line, sn, sample_id), []).append(row_map)
        else:
            by_pair.setdefault((sn, sample_id), []).append(row_map)
    return by_triplet, by_pair


def _pick_latest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: (row.get("_timestamp_dt") or datetime.min, int(row.get("_row_index") or 0)))


def _pick_training_label(rows: list[dict[str, Any]]) -> tuple[dict[str, str] | None, str]:
    if not rows:
        return None, "no_manual_label"
    expert_rows = [r for r in rows if r.get("_source_category") == "expert"]
    if expert_rows:
        return _normalized_label_payload(_pick_latest(expert_rows)), "expert"
    employee_rows = [r for r in rows if r.get("_source_category") == "operator"]
    if not employee_rows:
        return None, "no_manual_label"
    if len(employee_rows) < MIN_CONSISTENT_EMPLOYEE_LABELS:
        return None, "operator_single"
    if len({tuple(r.get("_label_signature") or ()) for r in employee_rows}) != 1:
        return None, "operator_conflict"
    return _normalized_label_payload(_pick_latest(employee_rows)), "operator_consistent"


def _inc(counter: dict[str, int], key: str) -> None:
    counter[key] = int(counter.get(key, 0)) + 1


def _sorted_counts(counter: dict[str, int], key_name: str) -> list[dict[str, Any]]:
    return [{key_name: k, "count": int(v)} for k, v in sorted(counter.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))]


def _label_target_map_from_cfg(cfg: dict[str, Any]) -> dict[str, int]:
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
    df: pd.DataFrame, cfg: dict[str, Any], *, seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    targets = _label_target_map_from_cfg(cfg)
    if _extract_all_features_from_cfg(cfg):
        print("[采样] 全量提取模式：保留全部样本")
        return df, [
            {"reason": n, "available": int((df["reason_name"].map(_norm) == n).sum()),
             "kept": int((df["reason_name"].map(_norm) == n).sum()), "target": t}
            for n, t in sorted(targets.items())
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
        picked = sub.sample(n=target, random_state=int(seed)) if len(sub) > target else sub
        if len(sub) > target:
            print(f"[采样] {name}: {len(sub)} -> {target}")
        elif len(sub) < target:
            print(f"[采样] {name}: {len(sub)} <= 目标 {target}，全取")
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
        result_key = _norm(row.get("result_key") or row.get("result_name")) or "<empty>"
        result_name = _norm(row.get("result_name") or result_key)
        result_id = _first_present_text(row.get("result_id"))
        reason_key = _norm(row.get("reason_key") or row.get("reason_name")) or "<empty>"
        reason_name = _norm(row.get("reason_name") or reason_key)
        reason_id = _first_present_text(row.get("reason_id"))
        result_meta.setdefault(result_key, {"result_key": result_key, "result_id": result_id, "result_name": result_name})
        reason_meta.setdefault(reason_key, {"reason_key": reason_key, "reason_id": reason_id, "reason_name": reason_name,
                                             "parent_result_key": result_key, "parent_result_name": result_name})
        _inc(result_counter, result_key)
        _inc(reason_counter, reason_key)
        _inc(line_counter, line)
        line_result_counter[(line, result_key)] = int(line_result_counter.get((line, result_key), 0)) + 1
    return {
        "result_distribution": [
            {"result": m["result_name"], "result_key": m["result_key"], "result_id": m["result_id"],
             "count": int(result_counter.get(k, 0))}
            for k, m in sorted(result_meta.items(), key=lambda kv: (-int(result_counter.get(kv[0], 0)), kv[0]))
        ],
        "reason_distribution": [
            {"reason": m["reason_name"], "parent_result_key": m["parent_result_key"],
             "reason_key": m["reason_key"], "reason_id": m["reason_id"], "count": int(reason_counter.get(k, 0))}
            for k, m in sorted(reason_meta.items(), key=lambda kv: (-int(reason_counter.get(kv[0], 0)), kv[0]))
        ],
        "line_distribution": _sorted_counts(line_counter, "line"),
        "line_result_distribution": [
            {"line": line, "result": result_key, "count": int(count)}
            for (line, result_key), count in sorted(line_result_counter.items(), key=lambda kv: (-int(kv[1]), kv[0]))
        ],
    }


def filter_sample_view_dataframe(
    sample_df: pd.DataFrame,
    label_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """为 DL 统一筛选训练标签，并返回标准格式 sample_view。"""
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


# ===========================================================================
# 样本筛选（原 training_data/sample_filter.py）
# ===========================================================================

OUTPUT_COLUMNS = STANDARD_SAMPLE_VIEW_COLUMNS


def _list(raw: object) -> list[str]:
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    return [str(x).strip() for x in (raw or []) if str(x).strip()]


def _key_items(raw: object, *, default_line: str) -> list[dict[str, str]]:
    values = raw if isinstance(raw, list) else ([raw] if raw else [])
    out: list[dict[str, str]] = []
    for item in values:
        if isinstance(item, dict):
            path = str(item.get("path") or item.get("folder") or item.get("dir") or "").strip()
            line = str(item.get("line") or default_line).strip()
        else:
            path = str(item or "").strip()
            line = ""
        if path:
            out.append({"path": path, "line": line})
    return out


def filter_samples(cfg: dict, output_folder: str | Path | None = None) -> Path:
    """从 manifest + sample registry 中筛选 DL 候选样本。"""
    database = cfg.get("database") if isinstance(cfg.get("database"), dict) else {}
    dataset = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    data = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}

    db_path = resolve_database_path(
        str(database.get("label_records_db_path") or cfg.get("label_records_db_path")
            or dataset.get("label_records_db_path") or "")
    )
    tdms_root = Path(str(database.get("tdms_root") or "")).expanduser()
    manifest_path = Path(str(
        database.get("manifest_path") or dataset.get("manifest_path")
        or db_path.parent / "tdms_manifest.csv"
    )).expanduser()
    for path, title in ((db_path, "label_records.db"), (manifest_path, "tdms_manifest.csv")):
        if not path.is_file():
            raise FileNotFoundError(f"{title} 不存在: {path}")

    samples = load_sample_dataframe(db_path, active_only=True)
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig").fillna("")
    if samples.empty or manifest.empty:
        merged = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        main_mask = pd.Series(False, index=manifest.index)
        selected_folders = _list(data.get("folders") or data.get("folder"))
        references: set[str] = set()
        line_references: set[tuple[str, str]] = set()
        for token in _list(data.get("reference")):
            if "/" in token:
                line, ref = token.split("/", 1)
                line_references.add((line.strip(), ref.strip()))
            else:
                references.add(token)
        lines = set(_list(data.get("line")))
        if not lines and not selected_folders and not references and not line_references:
            lines = set(_list(cfg.get("line_name")))
        if lines:
            main_mask |= manifest["line"].astype(str).isin(lines)
        if references or line_references:
            main_mask |= manifest.apply(
                lambda row: str(row.get("reference") or "").strip() in references
                or (str(row.get("line") or "").strip(), str(row.get("reference") or "").strip()) in line_references,
                axis=1,
            )

        def prefixes_for(values: list[str]) -> list[str]:
            out: list[str] = []
            for raw in values:
                p = Path(raw).expanduser()
                try:
                    out.append(p.resolve().relative_to(tdms_root.resolve()).as_posix().rstrip("/") + "/")
                except Exception:
                    out.append(str(raw).replace("\\", "/").strip("/") + "/")
            return out

        folder_prefixes = prefixes_for(selected_folders)
        key_items = _key_items(data.get("key_sns"), default_line=str(data.get("key_line") or cfg.get("line_name") or "").strip())
        rel = manifest["relative_path"].astype(str)
        if folder_prefixes:
            main_mask |= rel.map(lambda x: any(x.startswith(p) for p in folder_prefixes))
        if not lines and not references and not line_references and not folder_prefixes:
            main_mask |= True
        for item in key_items:
            prefix = prefixes_for([item["path"]])[0]
            key_mask = rel.str.startswith(prefix)
            item_line = item["line"]
            if not item_line:
                parts = prefix.strip("/").split("/")
                item_line = parts[1] if parts and parts[0] == "prototype" and len(parts) > 1 else parts[0] if parts else ""
            if item_line:
                key_mask &= manifest["line"].astype(str).eq(item_line)
            main_mask |= key_mask
        manifest = manifest[main_mask]

        manifest_cols = ["line", "sn", "reference", "time", "tdms_storage_root", "relative_path"]
        for col in manifest_cols:
            if col not in manifest.columns:
                manifest[col] = ""
        merged = samples.merge(manifest[manifest_cols].drop_duplicates(), on=["line", "sn"], how="inner")
        merged.insert(0, "view_name", "train")
        for col in OUTPUT_COLUMNS:
            if col not in merged.columns:
                merged[col] = ""
        merged = merged[OUTPUT_COLUMNS].drop_duplicates()

    merged = standardize_sample_view(merged)
    out = (Path(output_folder).expanduser() if output_folder is not None else resolve_output_dir(cfg)) / "sample_view.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {out} | rows={len(merged)}")
    return out


# ===========================================================================
# 数据集生成（原 training_data/generate.py）
# ===========================================================================

def _database_path(cfg: dict) -> Path:
    database = cfg.get("database") if isinstance(cfg.get("database"), dict) else {}
    dataset = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    raw = (database.get("label_records_db_path") or cfg.get("label_records_db_path")
           or dataset.get("label_records_db_path") or "")
    return resolve_database_path(str(raw))


def generate(cfg: dict, output_folder: str | Path | None = None) -> Path:
    """生成 DL 训练 sample_view.csv（样本筛选 + 标签筛选）。"""
    candidate_path = filter_samples(cfg, output_folder)
    candidates = pd.read_csv(candidate_path, encoding="utf-8-sig")
    labels = load_label_dataframe(_database_path(cfg))
    filtered, _stats = filter_sample_view_dataframe(candidates, labels, cfg)
    filtered.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    print(f"[DL TRAINING DATA] {candidate_path} | candidates={len(candidates)} | kept={len(filtered)}")
    return candidate_path


# ===========================================================================
# 两步数据集 CLI（原 main_dataset.py）
# ===========================================================================

def _to_text(value: Any) -> str:
    if value is None:
        return ""
    import math
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _resolve_dl_output_folder(cfg: dict[str, Any]) -> Path:
    return resolve_output_dir(cfg)


def _parse_dataset_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DL 数据集生成（sample_view + 特征提取）")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    parser.add_argument("--step", choices=["all", "sample-view", "extract"], default="all",
                        help="all=全流程（默认），sample-view=仅生成 sample_view，extract=仅提取特征")
    parser.add_argument("--output-folder", default=None, help="覆盖 DL 输出目录")
    parser.add_argument("--num-workers", type=int, default=None, help="并行 worker 数")
    parser.add_argument("--batch-size", type=int, default=None, help="特征分片大小")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[WARN] 忽略未识别参数: {unknown}")
    return args


def _run_sample_view(cfg: dict[str, Any], output_folder: Path) -> Path:
    print(f"\n{'='*60}\n[Step 1] 生成 sample_view.csv\n  输出目录: {output_folder}\n{'='*60}")
    output_folder.mkdir(parents=True, exist_ok=True)
    sv_path = generate(cfg, output_folder)
    print(f"[Step 1] 完成: {sv_path}\n")
    return sv_path


def _run_extract(config_path: Path, output_folder: Path, *, num_workers: int | None, batch_size: int | None) -> None:
    from dl.features import run as ewf_run
    print(f"\n{'='*60}\n[Step 2] 提取 mel 窗口特征\n  输出目录: {output_folder}\n{'='*60}")
    import argparse as _ap
    fake_args = _ap.Namespace(
        config=str(config_path),
        output_feature_folder=str(output_folder),
        sample_view=[str(output_folder / "sample_view.csv")],
        num_workers=num_workers if num_workers is not None else max(1, min(8, (os.cpu_count() or 1))),
        batch_size=batch_size,
        output_file_prefix=None, output_format=None, data_root=None,
        manifest_path=None, label_records_db_path=None,
        n_fft=None, hop_length=None, n_mels=None, max_frames=None, fmin=None, fmax=None,
    )
    ewf_run(fake_args)
    print(f"[Step 2] 完成\n")


def main(argv: list[str] | None = None) -> None:
    args = _parse_dataset_args(argv)
    config_path = Path(args.config).expanduser()
    cfg = _read_yaml_cfg(config_path)
    output_folder = Path(args.output_folder).expanduser() if args.output_folder else _resolve_dl_output_folder(cfg)

    print(f"[INFO] DL 数据集生成\n  config: {config_path}\n  step: {args.step}\n  output: {output_folder}")

    if args.step in ("all", "sample-view"):
        _run_sample_view(cfg, output_folder)

    if args.step in ("all", "extract"):
        sv_path = output_folder / "sample_view.csv"
        if not sv_path.exists():
            raise FileNotFoundError(f"sample_view.csv 不存在: {sv_path}")
        _run_extract(config_path, output_folder, num_workers=args.num_workers, batch_size=args.batch_size)

    print(f"[INFO] 全部完成。输出目录: {output_folder}")


if __name__ == "__main__":
    main()
