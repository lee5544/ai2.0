"""DL 训练标签筛选（对齐 ml/dataset/label_filter.py，自包含，不依赖 ml 或同包其它模块）。

根据 label_history 为 sample_view 选择训练标签 y：
  1. expert：取最新 expert 标签
  2. ≥2 个员工标签且一致：取最新
  3. ≥2 个员工但不一致 / 单个 / 无标注：不产生 y
另含 sample_view 标准化与 DL 输出目录解析。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from data_manager.config import load_data_manager_config
from data_manager.label_database import load_label_dataframe, resolve_database_path
from data_manager.label_internal_registry import normalize_label_source_category

def resolve_output_dir(cfg: dict) -> Path:
    """DL 专属数据集目录：results/{line}_{model}/dl_dataset_csv/。

    ``line_name + model.model_name`` 是一个 DL 实验的唯一识别码；特征与训练结果
    都必须落在这个唯一根目录下，避免不同模型架构自动共享或分叉目录。
    """
    line = str(cfg.get("line_name") or "line").strip()
    model = str((cfg.get("model") or {}).get("model_name") or "model").strip()
    results = Path(str(cfg.get("results_path") or "./results")).expanduser()
    return results / f"{line}_{model}" / "dl_dataset_csv"


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
LABEL_FILTER_STATUSES = ("confirmed", "unconfirmed")


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


def _sample_at_extract_from_cfg(cfg: dict[str, Any]) -> bool:
    """提取阶段是否按各类 sample 目标数下采样。

    **默认 False（全量提取）**，只有显式开启才在提取阶段采样。任一处为真即开启：
      - ``dl.extract.sample`` / ``dl.extract.extract_sample`` / ``dl.extract_sample``
      - ``train.label_mapping.extract_sample`` / ``train.extract_sample``

    向后兼容：旧 ``extract_all_tdms`` / ``extract_all_features`` 为真时强制全量（不采样）。
    注意：**训练阶段始终按目标数采样**（见 training/split.py），与本开关无关。
    """
    def _flag(d: Any, *keys: str) -> bool:
        return isinstance(d, dict) and any(bool(d.get(k)) for k in keys)

    train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    mapping_cfg = train_cfg.get("label_mapping") if isinstance(train_cfg.get("label_mapping"), dict) else {}
    dl_cfg = cfg.get("dl") if isinstance(cfg.get("dl"), dict) else {}
    extract_cfg = dl_cfg.get("extract") if isinstance(dl_cfg.get("extract"), dict) else {}

    # 旧「全量提取」开关为真 → 强制全量，不在提取阶段采样
    if (_flag(mapping_cfg, "extract_all_tdms", "extract_all_features")
            or _flag(train_cfg, "extract_all_tdms")
            or _flag(extract_cfg, "extract_all_tdms")
            or _flag(dl_cfg, "extract_all_tdms")):
        return False

    return (
        _flag(extract_cfg, "sample", "extract_sample")
        or _flag(mapping_cfg, "extract_sample")
        or _flag(train_cfg, "extract_sample")
        or _flag(dl_cfg, "extract_sample")
    )


def _downsample_by_label_targets(
    df: pd.DataFrame, cfg: dict[str, Any], *, seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    targets = _label_target_map_from_cfg(cfg)
    if not _sample_at_extract_from_cfg(cfg):
        print("[采样] 提取阶段：全量提取（不下采样）。如需提取时按目标数采样，设 dl.extract.sample: true")
        if df.empty or "reason_name" not in df.columns:
            return df, []
        names_all = df["reason_name"].map(_norm)
        return df, [
            {"reason": n, "available": int((names_all == n).sum()),
             "kept": int((names_all == n).sum()), "target": t}
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
# CLI：仅做标签筛选（读取已有 sample_view.csv → 重写训练标签 y）
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="按 DL 规则为 sample_view 重新筛选训练标签 y")
    parser.add_argument("--config", required=True, help="项目 YAML 配置")
    parser.add_argument("--sample-view", default=None, help="sample_view.csv 路径（默认取 DL 输出目录）")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).expanduser().read_text(encoding="utf-8")) or {}
    sv = Path(args.sample_view).expanduser() if args.sample_view else resolve_output_dir(cfg) / "sample_view.csv"
    db_raw = ((cfg.get("database") or {}).get("label_records_db_path")
              or cfg.get("label_records_db_path")
              or (cfg.get("dataset") or {}).get("label_records_db_path") or "")
    labels = load_label_dataframe(resolve_database_path(str(db_raw)), statuses=LABEL_FILTER_STATUSES)
    candidates = pd.read_csv(sv, encoding="utf-8-sig")
    filtered, _stats = filter_sample_view_dataframe(candidates, labels, cfg)
    filtered.to_csv(sv, index=False, encoding="utf-8-sig")
    print(f"[LABEL FILTER] {sv} | candidates={len(candidates)} | kept={len(filtered)}")


if __name__ == "__main__":
    main()
