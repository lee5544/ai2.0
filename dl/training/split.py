"""DL 训练数据集划分层（对应 ml/training/split.py）。

职责：由 label_rules 构建标签 runtime（原始标签→合并类别、每类目标数），
按 sn 分组切分 train/val/test（相同 sn 不跨集泄漏）+ 按目标数重采样。

数据加载与特征布局见同包 loader.py；训练过程见 dl/train.py。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from data_manager.label_rules import LABEL_RULES, Label

from dl.training.loader import (
    _load_feature_dataframe,
    _normalize_train_label,
    _resolve_compact_feature_column,
    _resolve_feature_layout,
    _scan_available_reasons,
    _to_int,
    _to_text,
)

def _default_auto_group_key(reason_id: int, reason_name: str) -> str:
    if reason_id in (-1, 0) or _to_text(reason_name) in {"干扰", "正常"}:
        return "ok_with_noise"
    return f"reason_{reason_id}"


def _build_runtime_from_entries(entries: List[Dict[str, Any]]):
    raw_to_model_mlabel: Dict[int, int] = {}
    for item in entries:
        raw_mlabel = int(item["mlabel_raw"])
        if raw_mlabel not in raw_to_model_mlabel:
            raw_to_model_mlabel[raw_mlabel] = len(raw_to_model_mlabel)

    label_sample_dict: Dict[int, int] = {}
    label_to_mlabel_map: Dict[int, int] = {}
    mlabel_mtype_dict: Dict[int, str] = {}
    mlabel_to_reason_ids: Dict[int, List[int]] = {}
    mlabel_weight_dict: Dict[int, float] = {}

    tmp_names: Dict[int, List[str]] = {}
    for item in entries:
        reason_id = int(item["reason_id"])
        reason_name = _to_text(item.get("reason_name")) or str(reason_id)
        sample = int(item.get("sample", -1))
        mlabel = raw_to_model_mlabel[int(item["mlabel_raw"])]
        label_sample_dict[reason_id] = sample
        label_to_mlabel_map[reason_id] = mlabel
        try:
            mlabel_weight_dict[mlabel] = float(item.get("weight", 1.0))
        except (TypeError, ValueError):
            mlabel_weight_dict[mlabel] = 1.0
        mlabel_to_reason_ids.setdefault(mlabel, [])
        if reason_id not in mlabel_to_reason_ids[mlabel]:
            mlabel_to_reason_ids[mlabel].append(reason_id)
        tmp_names.setdefault(mlabel, [])
        if reason_name not in tmp_names[mlabel]:
            tmp_names[mlabel].append(reason_name)

    for mlabel, names in tmp_names.items():
        mlabel_mtype_dict[int(mlabel)] = "_".join(names)
    return label_sample_dict, mlabel_mtype_dict, label_to_mlabel_map, mlabel_to_reason_ids, mlabel_weight_dict


def build_label_runtime(*, train_cfg: Dict[str, Any] | None, available_reasons: List[Dict[str, Any]]):
    train_cfg = train_cfg or {}
    reason_resolver = Label(LABEL_RULES).build_reason_resolver(available_reasons)
    entries: List[Dict[str, Any]] = []

    mapping_cfg = train_cfg.get("label_mapping")
    if not isinstance(mapping_cfg, dict):
        mapping_cfg = {}
    mapping_enable = bool(mapping_cfg.get("enable"))
    auto_mlabel = bool(mapping_cfg.get("auto_mlabel", True))
    mapping_groups = mapping_cfg.get("groups", [])

    normalized_groups: List[Dict[str, Any]] = []
    if isinstance(mapping_groups, list):
        for item in mapping_groups:
            if isinstance(item, dict):
                normalized_groups.append(item)
            else:
                normalized_groups.append({"reasons": item})
    elif isinstance(mapping_groups, dict):
        for value in mapping_groups.values():
            if isinstance(value, dict):
                normalized_groups.append(value)
            else:
                normalized_groups.append({"reasons": value})

    if mapping_enable and normalized_groups:
        for group_idx, group in enumerate(normalized_groups):
            if not bool(group.get("enable", True)):
                continue
            raw_mlabel = group_idx if auto_mlabel else (_to_int(group.get("mlabel")) or group_idx)
            group_sample = _to_int(group.get("sample"))
            if group_sample is None:
                group_sample = -1
            try:
                group_weight = float(group.get("weight", 1.0))
            except (TypeError, ValueError):
                group_weight = 1.0
            reason_tokens = group.get("reasons", [])
            if isinstance(reason_tokens, (tuple, set)):
                reason_tokens = list(reason_tokens)
            elif not isinstance(reason_tokens, list):
                reason_tokens = [reason_tokens]

            for token in reason_tokens:
                if isinstance(token, dict):
                    rid, rname = reason_resolver.resolve(
                        reason_id_value=token.get("reason_id", token.get("label")),
                        reason_name_value=token.get("reason", token.get("type", token.get("name"))),
                    )
                    sample = _to_int(token.get("sample"))
                    if sample is None:
                        sample = group_sample
                else:
                    rid, rname = reason_resolver.resolve(
                        reason_id_value=token,
                        reason_name_value=token,
                    )
                    sample = group_sample
                if rid is None:
                    raise ValueError(f"无法解析 reason（label_mapping.groups）: {token}")
                entries.append({"reason_id": rid, "reason_name": rname, "mlabel_raw": raw_mlabel, "sample": int(sample), "weight": group_weight})
    else:
        if not available_reasons:
            raise RuntimeError("未扫描到可用标签，无法构建 DL 训练映射。")
        group_to_raw_mlabel: Dict[str, int] = {}
        for item in available_reasons:
            rid = _to_int(item.get("reason_id"))
            if rid is None:
                continue
            rname = _to_text(item.get("reason_name")) or str(rid)
            group_key = _default_auto_group_key(rid, rname)
            if group_key not in group_to_raw_mlabel:
                group_to_raw_mlabel[group_key] = len(group_to_raw_mlabel)
            entries.append({"reason_id": rid, "reason_name": rname, "mlabel_raw": group_to_raw_mlabel[group_key], "sample": -1, "weight": 1.0})

    dedup: Dict[int, Dict[str, Any]] = {}
    for item in entries:
        dedup[int(item["reason_id"])] = item
    final_entries = list(dedup.values())

    available_reason_ids = {_to_int(item.get("reason_id")) for item in available_reasons}
    available_reason_ids.discard(None)
    filtered_entries = [item for item in final_entries if int(item["reason_id"]) in available_reason_ids]
    if not filtered_entries:
        raise RuntimeError("可用标签映射为空，请检查 train.label_mapping 配置。")

    label_sample_dict, mlabel_mtype_dict, label_to_mlabel_map, mlabel_to_reason_ids, mlabel_weight_dict = _build_runtime_from_entries(filtered_entries)
    print("[INFO] DL 训练标签映射（mlabel -> merged reasons）:")
    for mlabel in sorted(mlabel_to_reason_ids):
        reason_ids = sorted(int(x) for x in mlabel_to_reason_ids[mlabel])
        counts = {int(item["reason_id"]): int(item["reason_count"]) for item in available_reasons if _to_int(item.get("reason_id")) is not None}
        total_count = sum(counts.get(rid, 0) for rid in reason_ids)
        sample_values = sorted({int(label_sample_dict.get(rid, -1)) for rid in reason_ids})
        sample_text = str(sample_values[0]) if len(sample_values) == 1 else "/".join(str(v) for v in sample_values)
        print(
            f"  mlabel={mlabel:>2} | count={total_count:>5} | sample={sample_text:>6} | "
            f"weight={float(mlabel_weight_dict.get(mlabel, 1.0)):>4} | "
            f"reason_ids={reason_ids} | mtype={mlabel_mtype_dict[mlabel]}"
        )
    return label_sample_dict, mlabel_mtype_dict, label_to_mlabel_map, mlabel_to_reason_ids, mlabel_weight_dict


def _resolve_split_groups(df: pd.DataFrame, *, split_name: str) -> pd.Series | None:
    if "sn" not in df.columns:
        print(f"[WARN] {split_name} 缺少 sn 列，回退为按行切分。")
        return None
    groups = df["sn"].map(_to_text)
    empty_mask = groups.eq("")
    if empty_mask.any():
        groups = groups.copy()
        missing_indexes = list(groups[empty_mask].index)
        for idx in missing_indexes:
            groups.at[idx] = f"__missing_sn__{idx}"
        print(f"[WARN] {split_name} 检测到 {len(missing_indexes)} 条 sn 为空，已按独立组处理。")
    if int(groups.nunique()) < 2:
        print(f"[WARN] {split_name} 仅有 {int(groups.nunique())} 个 sn 组，回退为按行切分。")
        return None
    return groups.astype(str)


def _random_row_split(df: pd.DataFrame, *, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 2:
        raise RuntimeError("样本数不足 2，无法切分。")
    rng = np.random.default_rng(int(random_state))
    indexes = np.arange(len(df))
    rng.shuffle(indexes)
    n_test = int(round(len(df) * float(test_size)))
    n_test = max(1, min(len(df) - 1, n_test))
    test_idx = indexes[:n_test]
    train_idx = indexes[n_test:]
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def _stratified_row_split(df: pd.DataFrame, *, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = df.groupby("label").indices
    if len(grouped) < 2:
        raise RuntimeError("类别数不足 2，无法做分层切分。")
    rng = np.random.default_rng(int(random_state))
    train_idx: List[int] = []
    test_idx: List[int] = []
    for _, idx_array in grouped.items():
        indexes = np.array(list(idx_array), dtype=int)
        if len(indexes) < 2:
            raise RuntimeError("存在样本数不足 2 的类别，无法做分层切分。")
        rng.shuffle(indexes)
        n_test = int(round(len(indexes) * float(test_size)))
        n_test = max(1, min(len(indexes) - 1, n_test))
        test_idx.extend(indexes[:n_test].tolist())
        train_idx.extend(indexes[n_test:].tolist())
    if not train_idx or not test_idx:
        raise RuntimeError("分层切分失败：出现空集合。")
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def _safe_row_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df["label"].nunique() > 1:
        try:
            return _stratified_row_split(df, test_size=test_size, random_state=random_state)
        except Exception as exc:
            print(f"[WARN] {split_name} 分层切分失败，改为非分层切分: {exc}")
    return _random_row_split(df, test_size=test_size, random_state=random_state)


def _group_split_score(
    *,
    full_labels: pd.Series,
    train_labels: pd.Series,
    test_labels: pd.Series,
    target_test_rows: float,
) -> float:
    full_counts = full_labels.value_counts()
    train_counts = train_labels.value_counts()
    test_counts = test_labels.value_counts()
    total_rows = max(int(len(full_labels)), 1)
    score = abs(len(test_labels) - target_test_rows) / total_rows
    for label in sorted(full_counts.index.tolist()):
        full_ratio = float(full_counts.get(label, 0)) / total_rows
        train_ratio = float(train_counts.get(label, 0)) / max(int(len(train_labels)), 1)
        test_ratio = float(test_counts.get(label, 0)) / max(int(len(test_labels)), 1)
        expected_test = float(full_counts.get(label, 0)) * (float(target_test_rows) / total_rows)
        score += abs(train_ratio - full_ratio)
        score += abs(test_ratio - full_ratio)
        score += abs(float(test_counts.get(label, 0)) - expected_test) / total_rows
        if full_counts.get(label, 0) > 0 and train_counts.get(label, 0) == 0:
            score += 5.0
    return float(score)


def _safe_group_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    split_name: str,
    group_split_trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _resolve_split_groups(df, split_name=split_name)
    if groups is None:
        return _safe_row_train_test_split(df, test_size=test_size, random_state=random_state, split_name=split_name)

    unique_groups = groups.unique().tolist()
    if len(unique_groups) < 2:
        return _safe_row_train_test_split(df, test_size=test_size, random_state=random_state, split_name=split_name)

    target_test_rows = float(len(df)) * float(test_size)
    best_score: float | None = None
    best_mask: pd.Series | None = None

    for trial in range(max(8, int(group_split_trials))):
        rng = np.random.default_rng(int(random_state) + int(trial))
        choice = rng.random(len(unique_groups)) < float(test_size)
        if int(choice.sum()) <= 0:
            choice[int(rng.integers(0, len(unique_groups)))] = True
        if int(choice.sum()) >= len(unique_groups):
            choice[int(rng.integers(0, len(unique_groups)))] = False
        selected = {unique_groups[idx] for idx, picked in enumerate(choice.tolist()) if picked}
        test_mask = groups.isin(selected)
        if int(test_mask.sum()) <= 0 or int((~test_mask).sum()) <= 0:
            continue
        score = _group_split_score(
            full_labels=df["label"],
            train_labels=df.loc[~test_mask, "label"],
            test_labels=df.loc[test_mask, "label"],
            target_test_rows=target_test_rows,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_mask = test_mask.copy()

    if best_mask is None:
        print(f"[WARN] {split_name} 按 sn 分组切分失败，回退为按行切分。")
        return _safe_row_train_test_split(df, test_size=test_size, random_state=random_state, split_name=split_name)

    train_df = df.loc[~best_mask].copy()
    test_df = df.loc[best_mask].copy()
    print(
        f"[INFO] {split_name} 按 sn 分组切分: "
        f"train_groups={int(groups.loc[~best_mask].nunique())}, "
        f"test_groups={int(groups.loc[best_mask].nunique())}, "
        f"train_rows={len(train_df)}, test_rows={len(test_df)}"
    )
    return train_df, test_df


def _overall_train_fraction(*, test_size: float, val_size: float) -> float:
    keep_after_test = max(0.0, 1.0 - float(test_size))
    keep_for_train = max(0.0, 1.0 - float(val_size))
    fraction = keep_after_test * keep_for_train
    return fraction if fraction > 0 else 1.0


def _prepare_core_and_extra_before_split(
    df: pd.DataFrame,
    *,
    label_sample_dict: Dict[int, int],
    test_size: float,
    val_size: float,
    random_state: int,
    group_sample_trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _resolve_split_groups(df, split_name="预分池")
    if groups is None:
        print("[WARN] 预分池缺少稳定分组键，跳过剩余样本外置逻辑。")
        return df.copy(), df.iloc[0:0].copy()

    train_fraction = _overall_train_fraction(test_size=test_size, val_size=val_size)
    label_counts = df["label"].value_counts().to_dict()
    target_pool_by_label: Dict[int, int] = {}
    for label, target_train_n in label_sample_dict.items():
        cur_n = int(label_counts.get(label, 0))
        if cur_n <= 0:
            continue
        if int(target_train_n) < 0:
            target_pool_by_label[int(label)] = cur_n
            continue
        target_pool_n = int(np.ceil(float(target_train_n) / float(train_fraction)))
        target_pool_n = max(int(target_train_n), target_pool_n)
        target_pool_by_label[int(label)] = min(cur_n, target_pool_n)
    if not target_pool_by_label:
        return df.copy(), df.iloc[0:0].copy()

    group_label_counts = pd.crosstab(groups, df["label"]).astype(int)
    target_series = pd.Series(target_pool_by_label, dtype=int)
    tracked_labels = [int(label) for label in target_series.index.tolist() if int(label) in group_label_counts.columns]
    if not tracked_labels:
        return df.copy(), df.iloc[0:0].copy()
    target_series = target_series.reindex(tracked_labels).fillna(0).astype(int)

    full_take_labels = [label for label in tracked_labels if int(target_series.get(label, 0)) >= int(label_counts.get(label, 0))]
    mandatory_groups: set[str]
    if full_take_labels:
        mandatory_mask = group_label_counts[full_take_labels].sum(axis=1) > 0
        mandatory_groups = set(group_label_counts.index[mandatory_mask].tolist())
    else:
        mandatory_groups = set()

    base_counts = (
        group_label_counts.loc[list(mandatory_groups), tracked_labels].sum(axis=0)
        if mandatory_groups
        else pd.Series(0, index=tracked_labels, dtype=int)
    )
    candidate_groups = [g for g in group_label_counts.index.tolist() if g not in mandatory_groups]
    group_rows_map = {
        group: group_label_counts.loc[group, tracked_labels].reindex(tracked_labels, fill_value=0).astype(int)
        for group in candidate_groups
    }

    best_selected = set(mandatory_groups)
    best_counts = base_counts.copy()
    best_score: tuple[int, int] | None = None
    trial_count = max(8, min(int(group_sample_trials), 96))

    for trial in range(trial_count):
        rng = np.random.default_rng(int(random_state) + int(trial))
        order = list(candidate_groups)
        rng.shuffle(order)
        selected = set(mandatory_groups)
        selected_order: List[str] = []
        current_counts = base_counts.copy()

        for group in order:
            deficit = (target_series - current_counts).clip(lower=0)
            if int(deficit.sum()) <= 0:
                break
            contribution = group_rows_map[group]
            if int(contribution[deficit > 0].sum()) <= 0:
                continue
            selected.add(group)
            selected_order.append(group)
            current_counts = current_counts.add(contribution, fill_value=0).astype(int)

        for group in reversed(selected_order):
            contribution = group_rows_map[group]
            candidate_counts = current_counts.subtract(contribution, fill_value=0).astype(int)
            if bool((candidate_counts >= target_series).all()):
                selected.remove(group)
                current_counts = candidate_counts

        deficit_total = int((target_series - current_counts).clip(lower=0).sum())
        overshoot_total = int((current_counts - target_series).clip(lower=0).sum())
        score = (deficit_total, overshoot_total)
        if best_score is None or score < best_score:
            best_score = score
            best_selected = set(selected)
            best_counts = current_counts.copy()

    core_mask = groups.isin(best_selected)
    core_df = df.loc[core_mask].copy()
    extra_df = df.loc[~core_mask].copy()
    print("[INFO] 预分池完成：")
    for label in tracked_labels:
        cur_n = int(label_counts.get(label, 0))
        target_pool_n = int(target_series.get(label, 0))
        core_n = int(best_counts.get(label, 0))
        extra_n = max(cur_n - core_n, 0)
        if target_pool_n >= cur_n:
            print(f"  类别 {label}: 全取 {cur_n} 条")
        else:
            print(
                f"  类别 {label}: 训练目标={label_sample_dict.get(label, -1)} "
                f"-> 主样本池目标={target_pool_n}，实际={core_n}，剩余={extra_n}"
            )
    return core_df, extra_df


def _nonempty_key_set(df: pd.DataFrame, column: str) -> set[str] | None:
    if column not in df.columns:
        return None
    values = {_to_text(v) for v in df[column].tolist()}
    values.discard("")
    return values


def _print_split_overlap_report(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("[自检] train/val/test 交集检查:")
    split_pairs = [("train∩val", train_df, val_df), ("train∩test", train_df, test_df), ("val∩test", val_df, test_df)]
    for column in ("sample_id", "sn", "tdms_path"):
        if _nonempty_key_set(train_df, column) is None or _nonempty_key_set(val_df, column) is None or _nonempty_key_set(test_df, column) is None:
            print(f"  {column}: 缺少列，跳过")
            continue
        parts = []
        for pair_name, left_df, right_df in split_pairs:
            left_keys = _nonempty_key_set(left_df, column) or set()
            right_keys = _nonempty_key_set(right_df, column) or set()
            parts.append(f"{pair_name}={len(left_keys & right_keys)}")
        print(f"  {column}: " + ", ".join(parts))


def _load_and_split_dataset(
    *,
    df: pd.DataFrame,
    train_cfg: Dict[str, Any],
    label_sample_dict: Dict[int, int],
    label_to_mlabel_map: Dict[int, int],
    random_state: int,
    test_size: float,
    val_size: float,
    group_split_trials: int,
    group_sample_trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _normalize_train_label(df)
    valid_labels = list(label_sample_dict.keys())
    df = df[df["label"].isin(valid_labels)].copy()
    if df.empty:
        raise RuntimeError("过滤后数据为空，请检查 label_mapping 与特征文件中的标签是否一致。")

    core_df, extra_df = _prepare_core_and_extra_before_split(
        df,
        label_sample_dict=label_sample_dict,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        group_sample_trials=group_sample_trials,
    )
    if core_df.empty:
        raise RuntimeError("主样本池为空，无法继续切分。")

    tmp_df, test_df = _safe_group_train_test_split(
        core_df,
        test_size=test_size,
        random_state=random_state,
        split_name="train/test",
        group_split_trials=group_split_trials,
    )
    train_df, val_df = _safe_group_train_test_split(
        tmp_df,
        test_size=val_size,
        random_state=random_state,
        split_name="train/val",
        group_split_trials=group_split_trials,
    )
    if not extra_df.empty:
        test_df = pd.concat([test_df, extra_df], ignore_index=True)
        print(f"[INFO] 预分池剩余样本已并入 test: +{len(extra_df)} 条")

    train_parts = []
    train_counts = train_df["label"].value_counts().to_dict()
    for label, target_n in label_sample_dict.items():
        if label not in train_counts:
            print(f"[WARN] 训练集中不存在 label={label}，跳过扩充。")
            continue
        sub = train_df[train_df["label"] == label]
        cur_n = len(sub)
        if int(target_n) < 0:
            sampled = sub.copy()
            print(f"类别 {label}: 训练集已有 {cur_n} 条，配置要求全取 -> 保留 {len(sampled)} 条")
        elif cur_n > int(target_n):
            sampled = sub.sample(n=int(target_n), random_state=random_state)
            print(f"类别 {label}: 训练集已有 {cur_n} 条 -> 精调到 {target_n}")
        else:
            need_extra = int(target_n) - cur_n
            if need_extra > 0:
                extra = sub.sample(n=need_extra, replace=True, random_state=random_state)
                sampled = pd.concat([sub, extra], ignore_index=True)
                print(f"类别 {label}: 训练集已有 {cur_n} 条 -> 上采样到 {target_n}")
            else:
                sampled = sub.copy()
        train_parts.append(sampled)
    if train_parts:
        train_df = pd.concat(train_parts, ignore_index=True)

    _print_split_overlap_report(train_df, val_df, test_df)

    def _map_labels(df_in: pd.DataFrame) -> pd.DataFrame:
        out = df_in.copy()
        out["label_raw"] = out["label"].astype(int)
        out["label"] = out["label"].map(label_to_mlabel_map).fillna(-1).astype(int)
        return out[out["label"] >= 0].copy()

    train_df = _map_labels(train_df)
    val_df = _map_labels(val_df)
    test_df = _map_labels(test_df)
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("切分后出现空数据集，请检查 sample 数量和标签分布。")

    # 评估集：所有「有有效标签、既不在 train 也不在 test」的样本 = 验证集 + 采样剩余的全部样本
    def _row_keys(frame: pd.DataFrame) -> pd.Series:
        cols = [c for c in ("line", "sn", "sample_id") if c in frame.columns]
        if not cols:
            return frame.index.to_series().astype(str)
        return frame[cols].astype(str).agg("|".join, axis=1)

    full_mapped = _map_labels(df)
    used_keys = set(_row_keys(train_df)) | set(_row_keys(test_df))
    full_keys = _row_keys(full_mapped)
    val_plus_rest_df = full_mapped[~full_keys.isin(used_keys)].copy()
    print(
        f"[INFO] 评估集(val+采样剩余): {len(val_plus_rest_df)} 条 "
        f"(full={len(full_mapped)}, train={len(train_df)}, val={len(val_df)}, test={len(test_df)})"
    )
    return train_df, val_df, test_df, val_plus_rest_df


@dataclass
class DLSplit:
    """与 ml.training.split.TrainValTestSplit 对应的 DL 版本。"""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    full_df: pd.DataFrame
    val_plus_rest_df: pd.DataFrame
    schema: Dict[str, Any]
    channel_names: List[str]
    columns_by_channel: Dict[str, List[str]]
    seq_len: int
    compact_feature_column: str
    label_sample_dict: Dict[int, int] = field(default_factory=dict)
    mlabel_mtype_dict: Dict[int, str] = field(default_factory=dict)
    label_to_mlabel_map: Dict[int, int] = field(default_factory=dict)
    mlabel_to_reason_ids: Dict[int, List[int]] = field(default_factory=dict)
    class_name_by_id: Dict[int, str] = field(default_factory=dict)
    mlabel_weight_dict: Dict[int, float] = field(default_factory=dict)


def prepare_train_val_test_for_dl(
    model: Any,
    *,
    random_state: int | None = None,
    rebalance_train: bool = True,
) -> DLSplit:
    """加载 mel 批次 → 构建标签运行时 → 切分，返回 DLSplit。"""
    del rebalance_train  # DL 的训练集精调在切分流程内完成

    global_train_cfg = dict(getattr(model, "global_train_cfg", {}) or {})
    schema = dict(getattr(model, "schema", {}) or {})
    compact_feature_column = _resolve_compact_feature_column(schema)

    seed = int(getattr(model, "random_state", 42) if random_state is None else random_state)
    test_size = float(getattr(model, "test_size", 0.1))
    val_size = float(getattr(model, "val_size", 0.125))
    group_split_trials = int(getattr(model, "group_split_trials", 32))
    group_sample_trials = int(getattr(model, "group_sample_trials", 24))

    df = _load_feature_dataframe(model)
    available_reasons = _scan_available_reasons(df)
    (
        label_sample_dict,
        mlabel_mtype_dict,
        label_to_mlabel_map,
        mlabel_to_reason_ids,
        mlabel_weight_dict,
    ) = build_label_runtime(
        train_cfg=global_train_cfg,
        available_reasons=available_reasons,
    )
    class_name_by_id = {int(k): str(v) for k, v in mlabel_mtype_dict.items()}

    train_df, val_df, test_df, val_plus_rest_df = _load_and_split_dataset(
        df=df,
        train_cfg=global_train_cfg,
        label_sample_dict=label_sample_dict,
        label_to_mlabel_map=label_to_mlabel_map,
        random_state=seed,
        test_size=test_size,
        val_size=val_size,
        group_split_trials=group_split_trials,
        group_sample_trials=group_sample_trials,
    )

    channel_names, columns_by_channel, seq_len = _resolve_feature_layout(df, schema)
    print(f"特征布局: channels={channel_names}, seq_len={seq_len}")

    return DLSplit(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        full_df=df,
        val_plus_rest_df=val_plus_rest_df,
        schema=schema,
        channel_names=channel_names,
        columns_by_channel=columns_by_channel,
        seq_len=int(seq_len),
        compact_feature_column=compact_feature_column,
        label_sample_dict={int(k): int(v) for k, v in label_sample_dict.items()},
        mlabel_mtype_dict={int(k): str(v) for k, v in mlabel_mtype_dict.items()},
        label_to_mlabel_map={int(k): int(v) for k, v in label_to_mlabel_map.items()},
        mlabel_to_reason_ids={int(k): [int(x) for x in v] for k, v in mlabel_to_reason_ids.items()},
        class_name_by_id=class_name_by_id,
        mlabel_weight_dict={int(k): float(v) for k, v in mlabel_weight_dict.items()},
    )
