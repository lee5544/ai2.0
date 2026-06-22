"""训练数据加载与重采样工具。

职责：从特征 CSV 文件加载 DataFrame，规范化标签，按目标样本数重采样。
划分逻辑（train/val/test split / CV）见 ml.dataset.split。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from ml.dataset.build import BASE_KEY_COLUMNS, OPTIONAL_KEY_COLUMNS


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _to_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _ordered_unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _nonempty_key_set(df: pd.DataFrame, column: str) -> set[str]:
    values = {_to_text(v) for v in df[column].tolist()}
    values.discard("")
    return values


# ---------------------------------------------------------------------------
# 元数据补齐
# ---------------------------------------------------------------------------

def _enrich_training_metadata_from_sample_view(
    df: pd.DataFrame,
    dataset_files: Sequence[str | Path],
) -> pd.DataFrame:
    """从同级 sample_view.csv 补齐 line/reference/time 等元数据列。"""
    sample_view_paths = [
        Path(path_text).expanduser().parent / "sample_view.csv"
        for path_text in dataset_files
    ]
    existing_sample_view_paths = []
    seen_paths: set[Path] = set()
    for path in sample_view_paths:
        if path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)
        existing_sample_view_paths.append(path)

    if not existing_sample_view_paths:
        return df

    metadata_priority = ["line", "reference", "time", "tdms_storage_root", "relative_path"]
    join_key_priority = tuple(BASE_KEY_COLUMNS) + tuple(OPTIONAL_KEY_COLUMNS)

    sample_view_frames: list[pd.DataFrame] = []
    for sample_view_path in existing_sample_view_paths:
        try:
            header_df = pd.read_csv(sample_view_path, nrows=0, encoding="utf-8-sig")
        except Exception as exc:
            print(f"[WARN] 读取 sample_view 头失败（跳过元数据补齐）: {sample_view_path} | {exc}")
            continue

        available_columns = list(header_df.columns)
        join_keys = [col for col in join_key_priority if col in df.columns and col in available_columns]
        metadata_columns = [col for col in metadata_priority if col in available_columns and col not in join_keys]
        if not join_keys or not metadata_columns:
            continue

        use_columns = _ordered_unique(join_keys + metadata_columns)
        try:
            sample_view_frames.append(
                pd.read_csv(sample_view_path, usecols=use_columns, encoding="utf-8-sig")
            )
        except Exception as exc:
            print(f"[WARN] 读取 sample_view 失败（跳过元数据补齐）: {sample_view_path} | {exc}")

    if not sample_view_frames:
        return df

    sample_view_df = pd.concat(sample_view_frames, ignore_index=True)
    join_keys = [col for col in join_key_priority if col in df.columns and col in sample_view_df.columns]
    metadata_columns = [col for col in metadata_priority if col in sample_view_df.columns and col not in join_keys]
    if not join_keys or not metadata_columns:
        return df

    sample_view_df = sample_view_df[join_keys + metadata_columns].drop_duplicates(subset=join_keys, keep="last")
    merged_df = df.merge(sample_view_df, on=join_keys, how="left", suffixes=("", "__sv"))

    fill_reports: list[str] = []
    for col in metadata_columns:
        sv_col = f"{col}__sv"
        if sv_col not in merged_df.columns:
            continue
        if col not in merged_df.columns:
            merged_df[col] = merged_df[sv_col]
            filled_count = int(merged_df[col].map(_to_text).ne("").sum())
        else:
            current_values = merged_df[col].map(_to_text)
            merged_values = merged_df[sv_col].map(_to_text)
            fill_mask = current_values.eq("") & merged_values.ne("")
            filled_count = int(fill_mask.sum())
            if filled_count > 0:
                merged_df.loc[fill_mask, col] = merged_df.loc[fill_mask, sv_col]
        if filled_count > 0:
            fill_reports.append(f"{col}={filled_count}")
        merged_df = merged_df.drop(columns=[sv_col])

    if fill_reports:
        sample_view_text = ", ".join(str(path) for path in existing_sample_view_paths)
        print(
            "[INFO] 已从 sample_view 补齐训练元数据: "
            + ", ".join(fill_reports)
            + f" | source={sample_view_text}"
        )
    return merged_df


# ---------------------------------------------------------------------------
# 标签规范化
# ---------------------------------------------------------------------------

def normalize_training_label(df: pd.DataFrame) -> pd.DataFrame:
    """将 reason_id 列规范化为整数 label 列。"""
    if "reason_id" not in df.columns:
        raise RuntimeError("特征文件中缺少标签列：reason_id")
    out = df.copy()
    out["label"] = pd.to_numeric(out["reason_id"], errors="coerce")
    out = out[out["label"].notna()].copy()
    out["label"] = out["label"].astype(int)
    print("[INFO] 训练标签列: reason_id -> label（reason 语义）")
    return out


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

def load_training_dataframe(
    dataset_files: Sequence[str | Path],
    *,
    valid_labels: Iterable[int],
) -> pd.DataFrame:
    """读取特征 CSV，补齐元数据，规范化标签，过滤有效 label。"""
    dfs: list[pd.DataFrame] = []
    for path_text in dataset_files:
        path = Path(path_text)
        try:
            dfs.append(pd.read_csv(path))
        except Exception as exc:
            print(f"❌ 读取失败 {path}: {exc}")
    if not dfs:
        expected_dir = Path(dataset_files[0]).expanduser().parent if dataset_files else Path("dataset_csv")
        raise RuntimeError(f"无可用特征文件！请检查目录: {expected_dir}")

    df = pd.concat(dfs, ignore_index=True)
    df = _enrich_training_metadata_from_sample_view(df, dataset_files)
    df = normalize_training_label(df)

    valid_label_set = {int(x) for x in valid_labels}
    df = df[df["label"].isin(valid_label_set)].copy()
    if df.empty:
        raise RuntimeError("过滤后数据为空，请检查标签映射与特征文件中的标签是否一致！")
    return df


# ---------------------------------------------------------------------------
# 重采样
# ---------------------------------------------------------------------------

def rebalance_dataframe_by_label_targets(
    df: pd.DataFrame,
    *,
    label_sample_dict: dict[int, int],
    random_state: int,
    context: str,
    allow_oversample: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按各类别目标样本数下采样或上采样。

    Returns:
        (sampled_df, remaining_df)  remaining_df 仅在下采样时非空。
    """
    if df.empty:
        return df.copy(), df.copy()

    parts: list[pd.DataFrame] = []
    remaining_parts: list[pd.DataFrame] = []
    label_counts = df["label"].value_counts().to_dict()

    for label, target_n in label_sample_dict.items():
        if label not in label_counts:
            print(f"[提示] {context}不存在 label={label}，跳过。")
            continue

        sub = df[df["label"] == label]
        cur_n = len(sub)

        if target_n < 0:
            sampled = sub.copy()
            remaining = pd.DataFrame()
            print(f"类别 {label}: {context}已有 {cur_n} 条，配置要求全取 -> 保留 {len(sampled)} 条")
        elif cur_n > target_n:
            sampled = sub.sample(n=int(target_n), random_state=random_state)
            remaining = sub.drop(sampled.index)
            print(f"类别 {label}: {context}已有 {cur_n} 条 -> 下采样到 {target_n}，剩余 {len(remaining)} 条")
        elif cur_n < target_n:
            if allow_oversample:
                need_extra = int(target_n) - cur_n
                extra = sub.sample(n=need_extra, replace=True, random_state=random_state)
                sampled = pd.concat([sub, extra], ignore_index=True)
                remaining = pd.DataFrame()
                print(f"类别 {label}: {context}已有 {cur_n} 条 -> 上采样到 {target_n}")
            else:
                sampled = sub.copy()
                remaining = pd.DataFrame()
                print(
                    f"[WARN] 类别 {label}: {context}已有 {cur_n} 条，小于目标 {target_n}；"
                    "为避免交叉验证样本泄漏，当前流程不做全局上采样。"
                )
        else:
            sampled = sub.copy()
            remaining = pd.DataFrame()
            print(f"类别 {label}: {context}命中目标 {target_n} 条")

        parts.append(sampled)
        remaining_parts.append(remaining)

    if not parts:
        raise RuntimeError(f"{context}按 label_sample_dict 处理后为空。")
    sampled_df = pd.concat(parts, ignore_index=True)
    remaining_df = pd.concat(remaining_parts, ignore_index=True) if remaining_parts else pd.DataFrame()
    return sampled_df, remaining_df


def map_labels(df: pd.DataFrame, label_to_mlabel_map: dict[int, int]) -> pd.DataFrame:
    """将原始 label 映射为模型标签 mlabel，过滤未覆盖的行。"""
    out = df.copy()
    out["label"] = out["label"].map(label_to_mlabel_map).fillna(-1).astype(int)
    out = out[out["label"] >= 0].copy()
    return out


# ---------------------------------------------------------------------------
# 打印辅助
# ---------------------------------------------------------------------------

def print_label_distribution(df: pd.DataFrame, title: str) -> None:
    print(f"[{title}]")
    if df.empty:
        print("  (empty)")
        return
    for label, count in df["label"].value_counts().sort_index().items():
        print(f"  类别 {int(label)}: {int(count)} 条")


def print_split_overlap_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    print("[自检] train/val/test 交集检查:")
    split_pairs = [
        ("train∩val", train_df, val_df),
        ("train∩test", train_df, test_df),
        ("val∩test", val_df, test_df),
    ]
    for column in ("sample_id", "sn", "tdms_path"):
        if column not in train_df.columns or column not in val_df.columns or column not in test_df.columns:
            print(f"  {column}: 缺少列，跳过")
            continue
        parts: list[str] = []
        for pair_name, left_df, right_df in split_pairs:
            left_keys = _nonempty_key_set(left_df, column)
            right_keys = _nonempty_key_set(right_df, column)
            parts.append(f"{pair_name}={len(left_keys & right_keys)}")
        print(f"  {column}: " + ", ".join(parts))


# ---------------------------------------------------------------------------
# 高级加载（用于训练）
# ---------------------------------------------------------------------------

def load_sampled_training_dataframe(
    dataset_files: Sequence[str | Path],
    *,
    label_sample_dict: dict[int, int],
    label_to_mlabel_map: dict[int, int],
    random_state: int,
    allow_oversample: bool = False,
) -> pd.DataFrame:
    """加载 → 按目标数重采样 → 映射标签。用于交叉验证场景。"""
    raw_df = load_training_dataframe(dataset_files, valid_labels=label_sample_dict.keys())
    sampled_df, _ = rebalance_dataframe_by_label_targets(
        raw_df,
        label_sample_dict=label_sample_dict,
        random_state=random_state,
        context="全量训练集",
        allow_oversample=allow_oversample,
    )
    mapped_df = map_labels(sampled_df, label_to_mlabel_map)
    print_label_distribution(mapped_df, "映射后全量训练集分布")
    return mapped_df
