"""数据集划分工具：train/val/test split 与交叉验证。

数据加载与重采样见 ml.training.loader。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None

from ml.training.loader import (
    load_sampled_training_dataframe,
    load_training_dataframe,
    map_labels,
    print_label_distribution,
    print_split_overlap_report,
    rebalance_dataframe_by_label_targets,
)


DEFAULT_SPLIT_STRATEGY = "reference_out"
SPLIT_STRATEGY_SPECS: dict[str, dict[str, str]] = {
    "stratified": {
        "title": "Stratified",
        "description": "按 label 比例随机划分，不考虑 line/reference。",
    },
    "reference_in": {
        "title": "Reference-In",
        "description": "在每个 line/reference 内部分层切分，评估熟悉结构上的识别上限。",
    },
    "reference_out": {
        "title": "Reference-Out",
        "description": "按 line/reference 分组切分，同组不跨集合，作为主实验。",
    },
    "line_out": {
        "title": "Line-Out",
        "description": "按 line 分组切分，评估跨产线泛化能力。",
    },
}


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class TrainValTestSplit:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    requested_strategy: str
    resolved_strategy: str
    val_strategy: str
    report: dict[str, Any]


@dataclass
class PreparedCrossValidationData:
    df: pd.DataFrame
    cv: Any
    groups: np.ndarray | None
    requested_splits: int
    resolved_strategy: str
    split_name: str
    report: dict[str, Any]


# ---------------------------------------------------------------------------
# 策略名称解析
# ---------------------------------------------------------------------------

def list_supported_split_strategies() -> list[str]:
    preferred_order = ["reference_out", "line_out", "reference_in", "stratified"]
    return [name for name in preferred_order if name in SPLIT_STRATEGY_SPECS]


def resolve_split_strategy_name(value: object) -> str:
    text = _to_text(value).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "sample": "stratified",
        "sample_level": "stratified",
        "stratified_sample": "stratified",
        "referencein": "reference_in",
        "reference_out_main": "reference_out",
        "referenceout": "reference_out",
        "lineout": "line_out",
    }
    normalized = aliases.get(text, text)
    if normalized in SPLIT_STRATEGY_SPECS:
        return normalized
    return DEFAULT_SPLIT_STRATEGY


def format_split_strategy_label(strategy: str) -> str:
    normalized = resolve_split_strategy_name(strategy)
    spec = SPLIT_STRATEGY_SPECS.get(normalized, {})
    title = _to_text(spec.get("title")) or normalized
    description = _to_text(spec.get("description"))
    return f"{title} | {description}" if description else title


# ---------------------------------------------------------------------------
# 高级加载（带 split）——供 registry.run_training 调用
# ---------------------------------------------------------------------------

def load_split_training_dataframe(
    dataset_files: Sequence[str | Path],
    *,
    label_sample_dict: dict[int, int],
    label_to_mlabel_map: dict[int, int],
    split_strategy: str,
    random_state: int,
    test_size: float = 0.1,
    val_size: float = 0.125,
    group_split_trials: int = 32,
    rebalance_train: bool = True,
) -> TrainValTestSplit:
    from pathlib import Path as _Path
    raw_df = load_training_dataframe(dataset_files, valid_labels=label_sample_dict.keys())
    split_bundle = split_dataframe(
        raw_df,
        split_strategy=split_strategy,
        random_state=random_state,
        test_size=test_size,
        val_size=val_size,
        group_split_trials=group_split_trials,
    )

    print(
        f"[INFO] 分割完成: strategy={split_bundle.resolved_strategy}, "
        f"val_strategy={split_bundle.val_strategy}, "
        f"train={split_bundle.train_df.shape}, val={split_bundle.val_df.shape}, test={split_bundle.test_df.shape}"
    )
    print_label_distribution(split_bundle.train_df, "训练集分布 (原始label)")
    print_label_distribution(split_bundle.val_df, "验证集分布 (原始label)")
    print_label_distribution(split_bundle.test_df, "测试集分布 (原始label)")
    print_split_overlap_report(split_bundle.train_df, split_bundle.val_df, split_bundle.test_df)

    if rebalance_train:
        sampled_train_df, remaining_train_df = rebalance_dataframe_by_label_targets(
            split_bundle.train_df,
            label_sample_dict=label_sample_dict,
            random_state=random_state,
            context="训练集",
            allow_oversample=True,
        )
        train_df = sampled_train_df
        test_df = (
            pd.concat([split_bundle.test_df, remaining_train_df], ignore_index=True)
            if not remaining_train_df.empty
            else split_bundle.test_df
        )
        print_label_distribution(train_df, "训练集精调后分布 (原始label)")
        if not remaining_train_df.empty:
            print(f"[INFO] 将训练集重采样剩余 {len(remaining_train_df)} 条数据放入测试集")
    else:
        train_df = split_bundle.train_df.copy()
        test_df = split_bundle.test_df.copy()
        print("[INFO] 当前流程关闭训练集重采样，保留原始切分结果。")

    mapped_train_df = map_labels(train_df, label_to_mlabel_map)
    mapped_val_df = map_labels(split_bundle.val_df, label_to_mlabel_map)
    mapped_test_df = map_labels(test_df, label_to_mlabel_map)

    print_label_distribution(mapped_train_df, "映射后训练集分布")
    print_label_distribution(mapped_val_df, "映射后验证集分布")
    print_label_distribution(mapped_test_df, "映射后测试集分布")
    print(
        f"[INFO] 最终分割: train={mapped_train_df.shape}, val={mapped_val_df.shape}, test={mapped_test_df.shape}"
    )

    return TrainValTestSplit(
        train_df=mapped_train_df,
        val_df=mapped_val_df,
        test_df=mapped_test_df,
        requested_strategy=split_bundle.requested_strategy,
        resolved_strategy=split_bundle.resolved_strategy,
        val_strategy=split_bundle.val_strategy,
        report=dict(split_bundle.report),
    )


def prepare_train_val_test_for_model(
    model: Any,
    *,
    random_state: int,
    rebalance_train: bool = True,
    test_size: float = 0.1,
    val_size: float = 0.125,
) -> TrainValTestSplit:
    return load_split_training_dataframe(
        getattr(model, "dataset_files"),
        label_sample_dict=dict(getattr(model, "label_sample_dict")),
        label_to_mlabel_map=dict(getattr(model, "label_to_mlabel_map")),
        split_strategy=str(getattr(model, "split_strategy", DEFAULT_SPLIT_STRATEGY)),
        random_state=int(random_state),
        test_size=float(test_size),
        val_size=float(val_size),
        group_split_trials=int(getattr(model, "train_cfg", {}).get("group_split_trials") or 32),
        rebalance_train=bool(rebalance_train),
    )


def prepare_cross_validation_data(
    *,
    dataset_files: Sequence[str | Path],
    label_sample_dict: dict[int, int],
    label_to_mlabel_map: dict[int, int],
    split_strategy: str,
    sample_random_state: int,
    cv_random_state: int,
    requested_splits: int,
    allow_oversample: bool = False,
    split_name: str,
) -> PreparedCrossValidationData:
    df = load_sampled_training_dataframe(
        dataset_files,
        label_sample_dict=label_sample_dict,
        label_to_mlabel_map=label_to_mlabel_map,
        random_state=sample_random_state,
        allow_oversample=allow_oversample,
    )
    cv, groups, resolved_strategy = build_cv_splitter(
        df,
        split_strategy=split_strategy,
        requested_splits=int(requested_splits),
        random_state=int(cv_random_state),
        split_name=split_name,
    )
    actual_splits = int(cv.get_n_splits(df, df["label"].values, groups))
    return PreparedCrossValidationData(
        df=df,
        cv=cv,
        groups=groups,
        requested_splits=actual_splits,
        resolved_strategy=resolved_strategy,
        split_name=split_name,
        report={
            "requested_strategy": resolve_split_strategy_name(split_strategy),
            "resolved_strategy": resolved_strategy,
        },
    )


def prepare_cross_validation_for_model(
    model: Any,
    *,
    requested_splits: int,
    sample_random_state: int,
    cv_random_state: int,
    allow_oversample: bool,
    split_name: str,
) -> PreparedCrossValidationData:
    return prepare_cross_validation_data(
        dataset_files=getattr(model, "dataset_files"),
        label_sample_dict=dict(getattr(model, "label_sample_dict")),
        label_to_mlabel_map=dict(getattr(model, "label_to_mlabel_map")),
        split_strategy=str(getattr(model, "split_strategy", DEFAULT_SPLIT_STRATEGY)),
        sample_random_state=int(sample_random_state),
        cv_random_state=int(cv_random_state),
        requested_splits=int(requested_splits),
        allow_oversample=bool(allow_oversample),
        split_name=split_name,
    )


def prepare_grid_search_cv_from_split(
    train_df: pd.DataFrame,
    *,
    split_strategy: str,
    requested_splits: int,
    random_state: int,
    split_name: str,
) -> PreparedCrossValidationData:
    cv, groups, resolved_strategy = build_cv_splitter(
        train_df,
        split_strategy=split_strategy,
        requested_splits=int(requested_splits),
        random_state=int(random_state),
        split_name=split_name,
    )
    actual_splits = int(cv.get_n_splits(train_df, train_df["label"].values, groups))
    return PreparedCrossValidationData(
        df=train_df.copy(),
        cv=cv,
        groups=groups,
        requested_splits=actual_splits,
        resolved_strategy=resolved_strategy,
        split_name=split_name,
        report={
            "requested_strategy": resolve_split_strategy_name(split_strategy),
            "resolved_strategy": resolved_strategy,
        },
    )


# ---------------------------------------------------------------------------
# split_xy (公开 API)
# ---------------------------------------------------------------------------

def split_xy(
    dataset: Any,
    *,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    random_state: int = 42,
    test_size: float = 0.1,
    val_size: float = 0.125,
    group_split_trials: int = 32,
) -> Any:
    """将 FeatureDataset 按策略划分为 train/validation/test 三份。"""
    from ml.dataset.build import FeatureDataset, FeatureDatasetSplit

    if not isinstance(dataset, FeatureDataset):
        raise TypeError(f"dataset must be FeatureDataset, got {type(dataset).__name__}")
    if len(dataset.X) != len(dataset.y) or len(dataset.X) != len(dataset.metadata):
        raise ValueError("X, y, and metadata must contain the same number of rows")

    frame = dataset.metadata.reset_index(drop=True).copy()
    frame = frame.drop(columns=list(dataset.feature_names), errors="ignore")
    frame["label"] = np.asarray(dataset.y, dtype=int)
    for column in dataset.feature_names:
        frame[column] = dataset.X[column].reset_index(drop=True)

    bundle = split_dataframe(
        frame,
        split_strategy=split_strategy,
        random_state=random_state,
        test_size=test_size,
        val_size=val_size,
        group_split_trials=group_split_trials,
    )

    def convert(part: pd.DataFrame) -> FeatureDataset:
        return FeatureDataset(
            X=part[dataset.feature_names].reset_index(drop=True),
            y=part["label"].astype(int).to_numpy(),
            metadata=part.drop(columns=[*dataset.feature_names, "label"], errors="ignore").reset_index(drop=True),
            feature_names=list(dataset.feature_names),
            feature_version=dataset.feature_version,
        )

    return FeatureDatasetSplit(
        train=convert(bundle.train_df),
        validation=convert(bundle.val_df),
        test=convert(bundle.test_df),
        requested_strategy=bundle.requested_strategy,
        resolved_strategy=bundle.resolved_strategy,
        report=dict(bundle.report),
    )


# ---------------------------------------------------------------------------
# split_dataframe
# ---------------------------------------------------------------------------

def split_dataframe(
    df: pd.DataFrame,
    *,
    split_strategy: str,
    random_state: int,
    test_size: float,
    val_size: float,
    group_split_trials: int = 32,
) -> TrainValTestSplit:
    requested_strategy = resolve_split_strategy_name(split_strategy)
    resolved_strategy = _resolve_runtime_strategy(df, requested_strategy)
    report: dict[str, Any] = {
        "requested_strategy": requested_strategy,
        "resolved_strategy": resolved_strategy,
    }

    if resolved_strategy == "stratified":
        train_df, val_df, test_df = _split_stratified_three_way(
            df, test_size=test_size, val_size=val_size, random_state=random_state,
        )
        val_strategy = "stratified"

    elif resolved_strategy == "reference_in":
        train_df, val_df, test_df = _split_reference_in_three_way(
            df, test_size=test_size, val_size=val_size,
            random_state=random_state, group_split_trials=group_split_trials,
        )
        val_strategy = "reference_in"

    elif resolved_strategy == "reference_out":
        groups = _build_reference_group_series(df, split_name="reference_out")
        train_pool_df, test_df = _split_by_groups_best_effort(
            df, groups=groups, test_size=test_size, random_state=random_state,
            split_name="train/test", group_split_trials=group_split_trials,
            prefer_line_coverage=True, allow_row_fallback=False,
        )
        train_pool_groups = _build_reference_group_series(train_pool_df, split_name="reference_out train pool")
        if train_pool_groups is not None and int(train_pool_groups.nunique()) >= 2:
            train_df, val_df = _split_by_groups_best_effort(
                train_pool_df, groups=train_pool_groups, test_size=val_size,
                random_state=random_state, split_name="train/val",
                group_split_trials=group_split_trials,
                prefer_line_coverage=False, allow_row_fallback=False,
            )
            val_strategy = "reference_out"
        else:
            print("[WARN] train/val 可用 reference 组不足，已回退到按 sn 分组切分。")
            train_df, val_df, val_strategy = _split_with_sn_fallback(
                train_pool_df, random_state=random_state, split_name="train/val",
                test_size=val_size,
                val_strategy_name="sn_group_within_reference_out_train",
                key_columns=("line", "reference", "sn"),
                group_split_trials=group_split_trials,
            )

    elif resolved_strategy == "line_out":
        groups = _build_line_group_series(df, split_name="line_out")
        train_pool_df, test_df = _split_by_groups_best_effort(
            df, groups=groups, test_size=test_size, random_state=random_state,
            split_name="train/test", group_split_trials=group_split_trials,
            prefer_line_coverage=False, allow_row_fallback=False,
        )
        train_pool_groups = _build_line_group_series(train_pool_df, split_name="line_out train pool")
        if train_pool_groups is not None and int(train_pool_groups.nunique()) >= 2:
            train_df, val_df = _split_by_groups_best_effort(
                train_pool_df, groups=train_pool_groups, test_size=val_size,
                random_state=random_state, split_name="train/val",
                group_split_trials=group_split_trials,
                prefer_line_coverage=False, allow_row_fallback=False,
            )
            val_strategy = "line_out"
        else:
            print("[WARN] train/val 可用 line 组不足，已回退到按 sn 分组切分。")
            train_df, val_df, val_strategy = _split_with_sn_fallback(
                train_pool_df, random_state=random_state, split_name="train/val",
                test_size=val_size,
                val_strategy_name="sn_group_within_line_out_train",
                key_columns=("line", "sn"),
                group_split_trials=group_split_trials,
            )
    else:  # pragma: no cover
        raise RuntimeError(f"未知 split_strategy: {resolved_strategy}")

    return TrainValTestSplit(
        train_df=train_df.copy(),
        val_df=val_df.copy(),
        test_df=test_df.copy(),
        requested_strategy=requested_strategy,
        resolved_strategy=resolved_strategy,
        val_strategy=val_strategy,
        report=report,
    )


# ---------------------------------------------------------------------------
# build_cv_splitter
# ---------------------------------------------------------------------------

def build_cv_splitter(
    df: pd.DataFrame,
    *,
    split_strategy: str,
    requested_splits: int,
    random_state: int,
    split_name: str,
) -> tuple[Any, np.ndarray | None, str]:
    resolved_strategy = _resolve_runtime_strategy(df, resolve_split_strategy_name(split_strategy))

    if resolved_strategy == "stratified":
        n_splits = _resolve_requested_split_count(df["label"].values, requested_splits, split_name=split_name)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        print(f"[INFO] {split_name} 按 label 分层交叉验证: n_splits={n_splits}")
        return cv, None, resolved_strategy

    if resolved_strategy == "reference_in":
        n_splits = _resolve_requested_split_count(df["label"].values, requested_splits, split_name=split_name)
        reference_groups = _build_reference_group_series(df, split_name=split_name)
        sn_groups = _build_sn_group_series(
            df, split_name=split_name,
            key_columns=("line", "reference", "sn"), required_columns=("sn",),
        )
        if reference_groups is None or sn_groups is None:
            raise RuntimeError(f"{split_name} 无法构建 reference_in 所需分组，缺少 line/reference/sn。")
        sn_group_count = int(sn_groups.nunique())
        if sn_group_count < 2:
            raise RuntimeError(f"{split_name} 可用 sn 组不足 2，无法执行 reference_in 交叉验证。")
        if n_splits > sn_group_count:
            print(f"[WARN] {split_name} 请求 n_splits={n_splits} > sn 组数 {sn_group_count}，已下调。")
            n_splits = sn_group_count
        cv = ReferenceInFoldSplitter(
            reference_groups=reference_groups.to_numpy(),
            sample_groups=sn_groups.to_numpy(),
            y=df["label"].to_numpy(),
            n_splits=n_splits, random_state=random_state,
        )
        print(
            f"[INFO] {split_name} Reference-In 交叉验证: "
            f"n_splits={n_splits}, references={int(reference_groups.nunique())}, sn_groups={sn_group_count}"
        )
        return cv, None, resolved_strategy

    if resolved_strategy == "reference_out":
        groups = _build_reference_group_series(df, split_name=split_name)
        if groups is None:
            raise RuntimeError(f"{split_name} 缺少 reference 分组信息，无法执行 reference_out 交叉验证。")
        return _build_group_cv(df, groups=groups, requested_splits=requested_splits,
                               random_state=random_state, split_name=split_name,
                               strategy_name=resolved_strategy)

    if resolved_strategy == "line_out":
        groups = _build_line_group_series(df, split_name=split_name)
        if groups is None:
            raise RuntimeError(f"{split_name} 缺少 line 分组信息，无法执行 line_out 交叉验证。")
        return _build_group_cv(df, groups=groups, requested_splits=requested_splits,
                               random_state=random_state, split_name=split_name,
                               strategy_name=resolved_strategy)

    raise RuntimeError(f"未知 split_strategy: {resolved_strategy}")


# ---------------------------------------------------------------------------
# ReferenceInFoldSplitter
# ---------------------------------------------------------------------------

class ReferenceInFoldSplitter:
    def __init__(
        self,
        *,
        reference_groups: np.ndarray,
        sample_groups: np.ndarray,
        y: np.ndarray,
        n_splits: int,
        random_state: int,
    ) -> None:
        self.reference_groups = np.asarray(reference_groups)
        self.sample_groups = np.asarray(sample_groups)
        self.y = np.asarray(y)
        self.n_splits = max(2, int(n_splits))
        self.random_state = int(random_state)
        self.fold_assignments = self._build_fold_assignments()
        self.active_fold_ids = sorted(int(x) for x in np.unique(self.fold_assignments))

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self.active_fold_ids)

    def split(self, X, y=None, groups=None):
        indices = np.arange(len(self.y))
        for fold_id in self.active_fold_ids:
            test_idx = indices[self.fold_assignments == fold_id]
            train_idx = indices[self.fold_assignments != fold_id]
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            yield train_idx, test_idx

    def _build_fold_assignments(self) -> np.ndarray:
        assignments = np.full(len(self.y), fill_value=-1, dtype=int)
        sample_group_fold: dict[str, int] = {}
        unique_reference_groups = pd.unique(self.reference_groups)

        for ref_idx, reference_value in enumerate(unique_reference_groups):
            member_idx = np.flatnonzero(self.reference_groups == reference_value)
            if len(member_idx) == 0:
                continue

            sample_groups = pd.unique(self.sample_groups[member_idx])
            if len(sample_groups) == 0:
                continue

            group_rows: list[tuple[str, int, int]] = []
            for sample_group in sample_groups:
                sample_member_idx = member_idx[self.sample_groups[member_idx] == sample_group]
                if len(sample_member_idx) == 0:
                    continue
                labels = self.y[sample_member_idx]
                label_mode = int(pd.Series(labels).mode(dropna=False).iloc[0])
                group_rows.append((str(sample_group), label_mode, int(len(sample_member_idx))))

            if not group_rows:
                continue

            group_names = np.array([item[0] for item in group_rows], dtype=object)
            group_labels = np.array([item[1] for item in group_rows], dtype=int)
            group_sizes = np.array([item[2] for item in group_rows], dtype=int)
            min_label_count = int(pd.Series(group_labels).value_counts().min()) if len(group_labels) else 0
            local_seed = int(self.random_state) + int(ref_idx)
            fold_by_group: dict[str, int] = {}

            if len(group_names) >= self.n_splits and min_label_count >= self.n_splits and len(np.unique(group_labels)) > 1:
                cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=local_seed)
                for fold_id, (_, test_local_idx) in enumerate(cv.split(np.zeros(len(group_names)), group_labels)):
                    for test_pos in test_local_idx:
                        fold_by_group[str(group_names[int(test_pos)])] = int(fold_id)
            else:
                rng = np.random.default_rng(local_seed)
                order = np.arange(len(group_names))
                rng.shuffle(order)
                fold_label_counts: list[dict[int, int]] = [dict() for _ in range(self.n_splits)]
                fold_sizes = np.zeros(self.n_splits, dtype=int)
                for pos in order:
                    group_name = str(group_names[int(pos)])
                    group_label = int(group_labels[int(pos)])
                    group_size = int(group_sizes[int(pos)])
                    best_fold = min(
                        range(self.n_splits),
                        key=lambda fid: (
                            int(fold_label_counts[fid].get(group_label, 0)),
                            int(fold_sizes[fid]), int(fid),
                        ),
                    )
                    fold_by_group[group_name] = int(best_fold)
                    fold_label_counts[best_fold][group_label] = int(
                        fold_label_counts[best_fold].get(group_label, 0)
                    ) + group_size
                    fold_sizes[best_fold] += group_size

            sample_group_fold.update(fold_by_group)

        for row_idx, sample_group in enumerate(self.sample_groups):
            fold_id = sample_group_fold.get(str(sample_group))
            if fold_id is not None:
                assignments[int(row_idx)] = int(fold_id)

        missing_mask = assignments < 0
        if missing_mask.any():
            raise RuntimeError("reference_in 交叉验证存在未分配 fold 的样本。")

        fold_counts = np.bincount(assignments, minlength=self.n_splits)
        empty_folds = [idx for idx, count in enumerate(fold_counts) if int(count) == 0]
        for empty_fold in empty_folds:
            donor_fold = int(np.argmax(fold_counts))
            donor_group_names = [g for g, fid in sample_group_fold.items() if int(fid) == donor_fold]
            if len(donor_group_names) <= 1:
                continue
            moved_group = donor_group_names[-1]
            sample_group_fold[moved_group] = int(empty_fold)
            assignments[self.sample_groups == moved_group] = int(empty_fold)
            fold_counts = np.bincount(assignments, minlength=self.n_splits)
        return assignments


# ---------------------------------------------------------------------------
# 内部辅助（策略运行时解析）
# ---------------------------------------------------------------------------

def _resolve_runtime_strategy(df: pd.DataFrame, requested_strategy: str) -> str:
    strategy = resolve_split_strategy_name(requested_strategy)
    if strategy == "line_out":
        groups = _build_line_group_series(df, split_name="line_out")
        if groups is None or int(groups.nunique()) < 2:
            raise RuntimeError("line_out 至少需要 2 个不同的 line 才能切分。")
        return strategy
    if strategy == "reference_out":
        groups = _build_reference_group_series(df, split_name=strategy)
        if groups is None or int(groups.nunique()) < 2:
            raise RuntimeError("reference_out 至少需要 2 个不同的 line/reference 组才能切分。")
        return strategy
    if strategy == "reference_in":
        groups = _build_reference_group_series(df, split_name=strategy)
        if groups is None or int(groups.nunique()) < 1:
            raise RuntimeError("reference_in 缺少可用的 line/reference 分组。")
        return strategy
    return strategy


# ---------------------------------------------------------------------------
# 内部辅助（三段式 split）
# ---------------------------------------------------------------------------

def _split_stratified_three_way(
    df: pd.DataFrame, *, test_size: float, val_size: float, random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_pool_df, test_df = _safe_row_train_test_split(df, test_size=test_size, random_state=random_state, split_name="train/test")
    train_df, val_df = _safe_row_train_test_split(train_pool_df, test_size=val_size, random_state=random_state, split_name="train/val")
    return train_df, val_df, test_df


def _split_reference_in_three_way(
    df: pd.DataFrame, *, test_size: float, val_size: float, random_state: int, group_split_trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = _build_reference_group_series(df, split_name="reference_in")
    if groups is None:
        raise RuntimeError("reference_in 缺少可用的 line/reference 分组，无法完成切分。")

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for group_idx, group_value in enumerate(pd.unique(groups)):
        group_df = df.loc[groups == group_value].copy()
        group_seed = int(random_state) + int(group_idx)
        group_train, group_val, group_test = _split_reference_group_by_sn(
            group_df, test_size=test_size, val_size=val_size,
            random_state=group_seed,
            split_name=f"reference_in::{group_value}",
            group_split_trials=group_split_trials,
        )
        if not group_train.empty:
            train_parts.append(group_train)
        if not group_val.empty:
            val_parts.append(group_val)
        if not group_test.empty:
            test_parts.append(group_test)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else df.iloc[0:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0].copy()

    if val_df.empty or test_df.empty:
        raise RuntimeError("reference_in 无法在保持 sn 不泄漏的前提下生成非空 train/val/test。")
    return train_df, val_df, test_df


def _split_reference_group_by_sn(
    df: pd.DataFrame, *, test_size: float, val_size: float,
    random_state: int, split_name: str, group_split_trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sn_groups = _build_sn_group_series(
        df, split_name=split_name,
        key_columns=("line", "reference", "sn"), required_columns=("sn",),
    )
    if sn_groups is None:
        print(f"[WARN] {split_name} 缺少 sn 列，当前 reference 整组保留在 train。")
        return df.copy(), df.iloc[0:0].copy(), df.iloc[0:0].copy()
    if int(sn_groups.nunique()) < 2:
        print(f"[WARN] {split_name} 可用 sn 组不足 2，当前 reference 整组保留在 train。")
        return df.copy(), df.iloc[0:0].copy(), df.iloc[0:0].copy()

    train_pool_df, test_df = _split_by_groups_best_effort(
        df, groups=sn_groups, test_size=test_size, random_state=random_state,
        split_name=f"{split_name}/train/test", group_split_trials=group_split_trials,
        prefer_line_coverage=False, allow_row_fallback=False,
    )
    train_pool_groups = _build_sn_group_series(
        train_pool_df, split_name=f"{split_name}/train_pool",
        key_columns=("line", "reference", "sn"), required_columns=("sn",),
    )
    if train_pool_groups is None or int(train_pool_groups.nunique()) < 2:
        print(f"[WARN] {split_name} train_pool 可用 sn 组不足 2，当前 reference 保留 train/test 两份。")
        return train_pool_df, train_pool_df.iloc[0:0].copy(), test_df

    train_df, val_df = _split_by_groups_best_effort(
        train_pool_df, groups=train_pool_groups, test_size=val_size, random_state=random_state,
        split_name=f"{split_name}/train/val", group_split_trials=group_split_trials,
        prefer_line_coverage=False, allow_row_fallback=False,
    )
    return train_df, val_df, test_df


def _split_with_sn_fallback(
    df: pd.DataFrame, *, random_state: int, test_size: float, split_name: str,
    val_strategy_name: str, key_columns: tuple[str, ...], group_split_trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    sn_groups = _build_sn_group_series(
        df, split_name=split_name, key_columns=key_columns, required_columns=("sn",),
    )
    if sn_groups is None or int(sn_groups.nunique()) < 2:
        raise RuntimeError(f"{split_name} 无法在保持同一 sn 不泄漏的前提下继续切分。")
    train_df, val_df = _split_by_groups_best_effort(
        df, groups=sn_groups, test_size=test_size, random_state=random_state,
        split_name=split_name, group_split_trials=group_split_trials,
        prefer_line_coverage=False, allow_row_fallback=False,
    )
    return train_df, val_df, val_strategy_name


# ---------------------------------------------------------------------------
# 内部辅助（行级 split）
# ---------------------------------------------------------------------------

def _safe_row_train_test_split(
    df: pd.DataFrame, *, test_size: float | int, random_state: int, split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()
    if isinstance(test_size, (int, np.integer)):
        test_count = int(test_size)
        if test_count <= 0:
            return df.copy(), df.iloc[0:0].copy()
        test_count = min(test_count, len(df) - 1)
        if test_count <= 0:
            return df.copy(), df.iloc[0:0].copy()
        effective_test_size: float | int = test_count
    else:
        effective_test_size = float(test_size)

    if len(df) < 2:
        return df.copy(), df.iloc[0:0].copy()

    stratify = df["label"] if df["label"].nunique() > 1 else None
    try:
        return train_test_split(df, test_size=effective_test_size, stratify=stratify, random_state=random_state)
    except ValueError as exc:
        if stratify is None:
            raise
        print(f"[WARN] {split_name} 分层切分失败，改为非分层切分: {exc}")
        return train_test_split(df, test_size=effective_test_size, stratify=None, random_state=random_state)


# ---------------------------------------------------------------------------
# 内部辅助（分组 CV）
# ---------------------------------------------------------------------------

def _build_group_cv(
    df: pd.DataFrame, *, groups: pd.Series, requested_splits: int,
    random_state: int, split_name: str, strategy_name: str,
) -> tuple[Any, np.ndarray | None, str]:
    unique_groups = int(groups.nunique())
    if unique_groups < 2:
        raise RuntimeError(f"{split_name} 可用分组不足 2，无法执行 {strategy_name} 分组交叉验证。")

    label_group_counts = (
        pd.DataFrame({"label": df["label"].values, "__group__": groups.to_numpy()})
        .drop_duplicates(subset=["label", "__group__"])
        .groupby("label")["__group__"]
        .nunique()
    )
    min_label_groups = int(label_group_counts.min()) if not label_group_counts.empty else unique_groups
    n_splits = min(int(requested_splits), unique_groups)

    if StratifiedGroupKFold is not None and min_label_groups >= 2:
        if n_splits > min_label_groups:
            print(
                f"[WARN] {split_name} 某些类别仅出现在 {min_label_groups} 个组中，"
                f"折数已从 {n_splits} 下调到 {min_label_groups}。"
            )
            n_splits = min_label_groups
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        print(f"[INFO] {split_name} {strategy_name} 分组交叉验证: n_splits={n_splits}, groups={unique_groups}")
        return cv, groups.to_numpy(), strategy_name

    n_splits = max(n_splits, 2)
    cv = GroupKFold(n_splits=n_splits)
    print(f"[INFO] {split_name} {strategy_name} 分组交叉验证(GroupKFold): n_splits={n_splits}, groups={unique_groups}")
    return cv, groups.to_numpy(), strategy_name


def _resolve_requested_split_count(y: np.ndarray, requested: int, *, split_name: str) -> int:
    value_counts = pd.Series(y).value_counts()
    min_count = int(value_counts.min()) if not value_counts.empty else 0
    n_splits = min(int(requested), min_count) if min_count > 0 else 0
    if n_splits < 2:
        raise RuntimeError(f"{split_name} 失败：可用样本不足以做交叉验证（n_splits={n_splits}）。")
    return n_splits


# ---------------------------------------------------------------------------
# 内部辅助（最优分组切分）
# ---------------------------------------------------------------------------

def _split_by_groups_best_effort(
    df: pd.DataFrame, *, groups: pd.Series | None, test_size: float,
    random_state: int, split_name: str, group_split_trials: int,
    prefer_line_coverage: bool, allow_row_fallback: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if groups is None or int(groups.nunique()) < 2:
        if allow_row_fallback:
            print(f"[WARN] {split_name} 分组不足，已回退到按样本分层切分。")
            return _safe_row_train_test_split(df, test_size=test_size, random_state=random_state, split_name=split_name)
        raise RuntimeError(f"{split_name} 分组不足，无法按组切分。")

    trials = max(8, min(int(group_split_trials), 256))
    splitter = GroupShuffleSplit(n_splits=trials, test_size=test_size, random_state=random_state)
    best_score: float | None = None
    best_train_idx: np.ndarray | None = None
    best_test_idx: np.ndarray | None = None
    target_test_rows = float(len(df)) * float(test_size)
    line_series = _normalized_text_series(df["line"]) if "line" in df.columns else None
    group_line_counts = _count_groups_per_line(groups, line_series) if prefer_line_coverage and line_series is not None else {}

    for train_idx, test_idx in splitter.split(df, y=df["label"], groups=groups):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        score = _group_split_score(
            full_labels=df["label"],
            train_labels=df.iloc[train_idx]["label"],
            test_labels=df.iloc[test_idx]["label"],
            target_test_rows=target_test_rows,
            line_series=line_series,
            train_idx=train_idx, test_idx=test_idx,
            group_line_counts=group_line_counts,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_train_idx = train_idx
            best_test_idx = test_idx

    if best_train_idx is None or best_test_idx is None:
        if allow_row_fallback:
            print(f"[WARN] {split_name} 分组切分失败，已回退到按样本分层切分。")
            return _safe_row_train_test_split(df, test_size=test_size, random_state=random_state, split_name=split_name)
        raise RuntimeError(f"{split_name} 分组切分失败。")

    train_df = df.iloc[best_train_idx].copy()
    test_df = df.iloc[best_test_idx].copy()
    print(
        f"[INFO] {split_name} 分组切分: "
        f"train_groups={int(groups.iloc[best_train_idx].nunique())}, "
        f"test_groups={int(groups.iloc[best_test_idx].nunique())}, "
        f"train_rows={len(train_df)}, test_rows={len(test_df)}"
    )
    return train_df, test_df


def _group_split_score(
    *, full_labels: pd.Series, train_labels: pd.Series, test_labels: pd.Series,
    target_test_rows: float, line_series: pd.Series | None,
    train_idx: np.ndarray, test_idx: np.ndarray,
    group_line_counts: dict[str, int],
) -> float:
    full_counts = full_labels.value_counts()
    train_counts = train_labels.value_counts()
    test_counts = test_labels.value_counts()
    total_rows = max(int(len(full_labels)), 1)
    target_train_rows = float(total_rows) - float(target_test_rows)

    score = 0.0
    score += _label_distribution_penalty(
        full_counts=full_counts, split_counts=train_counts,
        split_rows=int(len(train_labels)), target_rows=target_train_rows, total_rows=total_rows,
    )
    score += _label_distribution_penalty(
        full_counts=full_counts, split_counts=test_counts,
        split_rows=int(len(test_labels)), target_rows=float(target_test_rows), total_rows=total_rows,
    )

    if line_series is not None and group_line_counts:
        train_lines = set(line_series.iloc[train_idx].tolist())
        test_lines = set(line_series.iloc[test_idx].tolist())
        for line_name, group_count in group_line_counts.items():
            if int(group_count) < 2:
                continue
            if line_name not in train_lines:
                score += 2.0
            if line_name not in test_lines:
                score += 2.0
    return float(score)


def _label_distribution_penalty(
    *, full_counts: pd.Series, split_counts: pd.Series,
    split_rows: int, target_rows: float, total_rows: int,
) -> float:
    if split_rows <= 0:
        return 1e9

    score = 2.0 * abs(float(split_rows) - float(target_rows)) / max(float(target_rows), 1.0)
    split_ratio_denominator = max(float(split_rows), 1.0)
    total_rows_float = max(float(total_rows), 1.0)

    for label in sorted(full_counts.index.tolist()):
        full_count = float(full_counts.get(label, 0))
        observed_count = float(split_counts.get(label, 0))
        full_ratio = full_count / total_rows_float
        observed_ratio = observed_count / split_ratio_denominator
        expected_count = full_count * (float(target_rows) / total_rows_float)

        score += 3.0 * abs(observed_ratio - full_ratio)
        score += 1.5 * abs(observed_count - expected_count) / max(expected_count, 1.0)

        if expected_count >= 0.75 and observed_count == 0:
            score += 8.0
        elif expected_count >= 1.5 and observed_count < 1:
            score += 4.0

    return float(score)


# ---------------------------------------------------------------------------
# 分组构建辅助
# ---------------------------------------------------------------------------

def _build_sn_group_series(
    df: pd.DataFrame, *, split_name: str,
    key_columns: tuple[str, ...], required_columns: tuple[str, ...] = ("sn",),
) -> pd.Series | None:
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        print(f"[WARN] {split_name} 缺少列: {missing_required}")
        return None

    normalized_columns: dict[str, pd.Series] = {}
    missing_total = 0
    for col in key_columns:
        if col not in df.columns:
            normalized = pd.Series([""] * len(df), index=df.index, dtype=object)
        else:
            normalized = _normalized_text_series(df[col])
        empty_mask = normalized.eq("")
        if empty_mask.any():
            normalized = normalized.copy()
            for idx in list(df.index[empty_mask]):
                normalized.at[idx] = f"__missing_{col}__{idx}"
            missing_total += int(empty_mask.sum())
        normalized_columns[col] = normalized.astype(str)

    # 增强特征使用 augmented sn，但必须继承原始 sample 的分组，避免
    # 原始样本在 train、增强样本在 val/test 造成数据泄漏。
    if "sn" in normalized_columns and "source_sn" in df.columns:
        source_sn = _normalized_text_series(df["source_sn"])
        normalized_columns["sn"] = source_sn.where(source_sn.ne(""), normalized_columns["sn"]).astype(str)

    if missing_total > 0:
        print(f"[WARN] {split_name} 检测到 {missing_total} 个空分组字段，已按独立组处理。")

    groups = normalized_columns[key_columns[0]].copy()
    for col in key_columns[1:]:
        groups = groups + "::" + normalized_columns[col]
    return groups.astype(str)


def _build_reference_group_series(df: pd.DataFrame, *, split_name: str) -> pd.Series | None:
    if "line" not in df.columns or "reference" not in df.columns:
        print(f"[WARN] {split_name} 缺少 line/reference 列。")
        return None
    line_series = _normalized_text_series(df["line"])
    ref_series = _normalized_text_series(df["reference"])
    missing_mask = line_series.eq("") | ref_series.eq("")
    if missing_mask.any():
        line_series = line_series.copy()
        ref_series = ref_series.copy()
        missing_idx = list(df.index[missing_mask])
        for idx in missing_idx:
            if line_series.at[idx] == "":
                line_series.at[idx] = f"__missing_line__{idx}"
            if ref_series.at[idx] == "":
                ref_series.at[idx] = f"__missing_reference__{idx}"
        print(f"[WARN] {split_name} 检测到 {len(missing_idx)} 条 line/reference 为空，已按独立组处理。")
    return (line_series + "::" + ref_series).astype(str)


def _build_line_group_series(df: pd.DataFrame, *, split_name: str) -> pd.Series | None:
    if "line" not in df.columns:
        print(f"[WARN] {split_name} 缺少 line 列。")
        return None
    groups = _normalized_text_series(df["line"])
    missing_mask = groups.eq("")
    if missing_mask.any():
        groups = groups.copy()
        for idx in list(df.index[missing_mask]):
            groups.at[idx] = f"__missing_line__{idx}"
        print(f"[WARN] {split_name} 检测到 {int(missing_mask.sum())} 条 line 为空，已按独立组处理。")
    return groups.astype(str)


def _count_groups_per_line(groups: pd.Series, line_series: pd.Series | None) -> dict[str, int]:
    if line_series is None:
        return {}
    df = pd.DataFrame({"line": line_series.to_numpy(), "__group__": groups.to_numpy()})
    df = df[df["line"] != ""].drop_duplicates(subset=["line", "__group__"])
    if df.empty:
        return {}
    return {str(k): int(v) for k, v in df.groupby("line")["__group__"].nunique().items()}


def _normalized_text_series(series: pd.Series) -> pd.Series:
    return series.map(_to_text).astype(str)


def _to_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text
