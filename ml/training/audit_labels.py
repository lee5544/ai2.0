#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = Path("results/epump2_general_20260407_xgb")


KEY_COLUMNS = [
    "line",
    "sn",
    "sample_id",
    "reference",
    "time",
    "relative_path",
    "group_name",
]

REVIEW_COLUMNS = [
    "audit_priority",
    "issue_type",
    "review_status",
    "proposed_action",
    "proposed_label",
    "reviewer",
    "review_note",
    "error_seed_count",
    "suspect_score",
    "mean_pred_confidence",
    "max_pred_confidence",
    "min_true_label_proba",
    "max_score_gap",
    "true_mlabel",
    "true_mtype",
    "pred_mlabels_seen",
    "pred_mtypes_seen",
    "result_key",
    "reason_key",
    "reason_name",
    "label_source",
    "label_timestamp",
    "label_version",
    "note",
    "line",
    "sn",
    "sample_id",
    "reference",
    "time",
    "relative_path",
    "tdms_path",
    "group_name",
    "channel_name",
    "sampling_rate",
    "seq_length",
    "num_features",
    "model_seeds_seen",
]

SAMPLE_VIEW_COLUMNS = [
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
    "tdms_path",
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

SAMPLE_VIEW_AUDIT_COLUMNS = [
    "audit_priority",
    "issue_type",
    "review_status",
    "proposed_action",
    "proposed_label",
    "reviewer",
    "review_note",
    "suspect_score",
    "error_seed_count",
    "mean_pred_confidence",
    "max_pred_confidence",
    "min_true_label_proba",
    "max_score_gap",
    "true_mlabel",
    "true_mtype",
    "pred_mlabels_seen",
    "pred_mtypes_seen",
    "model_seeds_seen",
]


SN_REVIEW_COLUMNS = [
    "audit_priority_top",
    "has_false_ok",
    "false_ok_sample_count",
    "sample_count",
    "max_suspect_score",
    "mean_suspect_score",
    "max_error_seed_count",
    "review_status",
    "proposed_action",
    "reviewer",
    "review_note",
    "line",
    "sn",
    "issue_types",
    "audit_priorities",
    "reason_names",
    "sample_ids",
    "relative_paths",
    "tdms_paths",
    "model_seeds_seen",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _norm_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _join_unique(values: pd.Series) -> str:
    seen: list[str] = []
    for value in values:
        text = _norm_text(value)
        if text and text not in seen:
            seen.append(text)
    return "|".join(seen)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _sample_key(df: pd.DataFrame) -> pd.Series:
    usable = [col for col in KEY_COLUMNS if col in df.columns]
    if not usable:
        return df.index.astype(str).to_series(index=df.index)
    return df[usable].fillna("").astype(str).agg("||".join, axis=1)


def _issue_type(true_label: Any, pred_label: Any) -> str:
    true_int = int(true_label)
    pred_int = int(pred_label)
    if true_int != 0 and pred_int == 0:
        return "false_ok_NOK_as_OK"
    if true_int == 0 and pred_int != 0:
        return "false_reject_OK_as_NOK"
    if true_int != 0 and pred_int != 0 and true_int != pred_int:
        return "nok_type_confusion"
    return "other_mismatch"


def _priority(issue_type: str, seed_count: int, mean_conf: float) -> str:
    if issue_type == "false_ok_NOK_as_OK":
        if seed_count >= 2 or mean_conf >= 0.9:
            return "P0_critical_false_ok"
        return "P1_false_ok"
    if issue_type == "false_reject_OK_as_NOK":
        if seed_count >= 2 or mean_conf >= 0.9:
            return "P2_false_reject"
        return "P3_false_reject_low_conf"
    if issue_type == "nok_type_confusion":
        return "P3_nok_type_confusion"
    return "P4_other"


def _suspect_score(row: pd.Series) -> int:
    issue_type = _norm_text(row.get("issue_type"))
    base = {
        "false_ok_NOK_as_OK": 100,
        "false_reject_OK_as_NOK": 65,
        "nok_type_confusion": 40,
    }.get(issue_type, 20)

    seed_bonus = min(max(int(row.get("error_seed_count", 1)) - 1, 0) * 10, 40)
    conf_bonus = 10 if _safe_float(row.get("mean_pred_confidence"), 0.0) >= 0.9 else 0
    true_prob_bonus = 10 if _safe_float(row.get("min_true_label_proba"), 1.0) <= 0.05 else 0
    source_text = _norm_text(row.get("label_source")).lower()
    source_bonus = 10 if source_text and source_text != "expert" else 0
    note_text = _norm_text(row.get("note"))
    fuzzy_bonus = 10 if "fuzzy=" in note_text or "中等-" in note_text else 0
    return int(base + seed_bonus + conf_bonus + true_prob_bonus + source_bonus + fuzzy_bonus)


def find_misclassified_files(results_dir: Path) -> list[Path]:
    base = results_dir / "eval" / "best_model_eval"
    if not base.exists():
        return []
    test_files = sorted(base.glob("seed_*/test_misclassified_sample_view.csv"))
    if test_files:
        return test_files
    return sorted(base.glob("seed_*/validation_test_misclassified_sample_view.csv"))


def load_error_events(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = _read_csv(path)
        if df.empty:
            continue
        df["source_file"] = str(path)
        if "model_seed" not in df.columns:
            try:
                df["model_seed"] = int(path.parent.name.replace("seed_", ""))
            except Exception:
                df["model_seed"] = path.parent.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    events = pd.concat(frames, ignore_index=True)
    required = {"true_mlabel", "pred_mlabel"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"错分文件缺少必要列: {missing}")

    events["sample_key"] = _sample_key(events)
    events["issue_type"] = [
        _issue_type(t, p) for t, p in zip(events["true_mlabel"], events["pred_mlabel"])
    ]
    events["is_false_ok"] = events["issue_type"].eq("false_ok_NOK_as_OK")
    events["is_false_reject"] = events["issue_type"].eq("false_reject_OK_as_NOK")
    return events


def aggregate_candidates(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    first_cols = [
        col
        for col in [
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
            "true_mlabel",
            "true_mtype",
            "result_key",
            "result_id",
            "result_name",
            "reason_key",
            "reason_id",
            "reason_name",
            "label_source",
            "label_timestamp",
            "label_version",
            "note",
            "label",
        ]
        if col in events.columns
    ]

    rows: list[dict[str, Any]] = []
    for sample_key, group in events.groupby("sample_key", sort=False):
        first = group.iloc[0]
        issue_counts = group["issue_type"].value_counts()
        dominant_issue = str(issue_counts.index[0])
        mean_conf = float(pd.to_numeric(group.get("pred_confidence"), errors="coerce").mean())
        row: dict[str, Any] = {col: first.get(col, "") for col in first_cols}
        row.update(
            {
                "sample_key": sample_key,
                "issue_type": dominant_issue,
                "error_seed_count": int(group["model_seed"].nunique()),
                "event_count": int(len(group)),
                "mean_pred_confidence": mean_conf,
                "max_pred_confidence": float(pd.to_numeric(group.get("pred_confidence"), errors="coerce").max()),
                "min_true_label_proba": float(pd.to_numeric(group.get("true_label_proba"), errors="coerce").min()),
                "max_score_gap": float(pd.to_numeric(group.get("score_gap"), errors="coerce").max()),
                "pred_mlabels_seen": _join_unique(group.get("pred_mlabel", pd.Series(dtype=object)).astype(str)),
                "pred_mtypes_seen": _join_unique(group.get("pred_mtype", pd.Series(dtype=object))),
                "model_seeds_seen": _join_unique(group.get("model_seed", pd.Series(dtype=object)).astype(str)),
                "source_files_seen": _join_unique(group.get("source_file", pd.Series(dtype=object))),
                "review_status": "todo",
                "proposed_action": "",
                "proposed_label": "",
                "reviewer": "",
                "review_note": "",
            }
        )
        row["audit_priority"] = _priority(dominant_issue, int(row["error_seed_count"]), mean_conf)
        row["suspect_score"] = _suspect_score(pd.Series(row))
        rows.append(row)

    candidates = pd.DataFrame(rows)
    priority_order = {
        "P0_critical_false_ok": 0,
        "P1_false_ok": 1,
        "P2_false_reject": 2,
        "P3_false_reject_low_conf": 3,
        "P3_nok_type_confusion": 4,
        "P4_other": 5,
    }
    candidates["_priority_rank"] = candidates["audit_priority"].map(priority_order).fillna(99)
    candidates = candidates.sort_values(
        by=["_priority_rank", "suspect_score", "error_seed_count", "max_score_gap"],
        ascending=[True, False, False, False],
    ).drop(columns=["_priority_rank"])

    ordered_cols = [col for col in REVIEW_COLUMNS if col in candidates.columns]
    rest_cols = [col for col in candidates.columns if col not in ordered_cols]
    return candidates[ordered_cols + rest_cols]


def _to_sample_view(df: pd.DataFrame, *, view_name: str) -> pd.DataFrame:
    out = df.copy()
    out["view_name"] = view_name

    ordered_cols = [col for col in SAMPLE_VIEW_COLUMNS if col in out.columns]
    audit_cols = [col for col in SAMPLE_VIEW_AUDIT_COLUMNS if col in out.columns and col not in ordered_cols]
    rest_cols = [col for col in out.columns if col not in ordered_cols and col not in audit_cols]
    return out[ordered_cols + audit_cols + rest_cols]


def _write_sample_view_pair(
    *,
    df: pd.DataFrame,
    view_name: str,
    audit_dir: Path,
    filename: str,
) -> None:
    sample_view_df = _to_sample_view(df, view_name=view_name)
    sample_view_df.to_csv(audit_dir / filename, index=False, encoding="utf-8-sig")


FINAL_SAMPLE_VIEW_FILENAMES = (
    "sample_view_label_audit.csv",
    "sample_view_false_ok_review.csv",
)


def cleanup_confirm_label_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()


def build_label_consistency_tables(results_dir: Path, output_dir: Path) -> None:
    sample_view_path = results_dir / "dataset_csv" / "sample_view.csv"
    if not sample_view_path.exists():
        return

    sample_df = _read_csv(sample_view_path)
    if sample_df.empty:
        return

    if "label_source" in sample_df.columns:
        stats_cols = [col for col in ["label_source", "result_key", "reason_name"] if col in sample_df.columns]
        label_source_stats = (
            sample_df.groupby(stats_cols, dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        label_source_stats.to_csv(output_dir / "label_source_stats.csv", index=False, encoding="utf-8-sig")

    exact_keys = [col for col in KEY_COLUMNS if col in sample_df.columns]
    if exact_keys and "result_key" in sample_df.columns:
        exact_conflicts = (
            sample_df.groupby(exact_keys, dropna=False)
            .agg(
                row_count=("result_key", "size"),
                result_keys=("result_key", _join_unique),
                reason_names=("reason_name", _join_unique) if "reason_name" in sample_df.columns else ("result_key", _join_unique),
                label_sources=("label_source", _join_unique) if "label_source" in sample_df.columns else ("result_key", _join_unique),
            )
            .reset_index()
        )
        exact_conflicts = exact_conflicts[
            (exact_conflicts["row_count"] > 1) & exact_conflicts["result_keys"].str.contains("\\|", regex=True, na=False)
        ]
        exact_conflicts.to_csv(output_dir / "exact_key_label_conflicts.csv", index=False, encoding="utf-8-sig")

    group_keys = [col for col in ["line", "sn", "reference", "time", "relative_path"] if col in sample_df.columns]
    if group_keys and "result_key" in sample_df.columns:
        mixed = (
            sample_df.groupby(group_keys, dropna=False)
            .agg(
                channel_count=("group_name", "nunique") if "group_name" in sample_df.columns else ("result_key", "size"),
                row_count=("result_key", "size"),
                result_keys=("result_key", _join_unique),
                reason_names=("reason_name", _join_unique) if "reason_name" in sample_df.columns else ("result_key", _join_unique),
                sample_ids=("sample_id", _join_unique) if "sample_id" in sample_df.columns else ("result_key", _join_unique),
                label_sources=("label_source", _join_unique) if "label_source" in sample_df.columns else ("result_key", _join_unique),
            )
            .reset_index()
        )
        mixed = mixed[mixed["result_keys"].str.contains("\\|", regex=True, na=False)]
        mixed.to_csv(output_dir / "same_tdms_mixed_channel_labels.csv", index=False, encoding="utf-8-sig")


def build_suspicious_sn_table(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or "sn" not in candidates.columns:
        return pd.DataFrame()

    working = candidates.copy()
    for col in [
        "suspect_score",
        "error_seed_count",
    ]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    priority_order = {
        "P0_critical_false_ok": 0,
        "P1_false_ok": 1,
        "P2_false_reject": 2,
        "P3_false_reject_low_conf": 3,
        "P3_nok_type_confusion": 4,
        "P4_other": 5,
    }
    working["_priority_rank"] = working.get("audit_priority", pd.Series(index=working.index, dtype=object)).map(priority_order).fillna(99)
    working["_is_false_ok"] = working.get("issue_type", pd.Series(index=working.index, dtype=object)).eq("false_ok_NOK_as_OK")

    rows: list[dict[str, Any]] = []
    group_cols = [col for col in ["line", "sn"] if col in working.columns]
    for group_key, group in working.groupby(group_cols, sort=False, dropna=False):
        first = group.sort_values(
            by=["_priority_rank", "suspect_score", "error_seed_count"],
            ascending=[True, False, False],
            na_position="last",
        ).iloc[0]
        if isinstance(group_key, tuple):
            line_value = group_key[0] if len(group_key) >= 1 else first.get("line", "")
            sn_value = group_key[1] if len(group_key) >= 2 else first.get("sn", "")
        else:
            line_value = first.get("line", "")
            sn_value = group_key

        row = {
            "line": line_value,
            "sn": sn_value,
            "audit_priority_top": first.get("audit_priority", ""),
            "has_false_ok": bool(group["_is_false_ok"].any()),
            "false_ok_sample_count": int(group["_is_false_ok"].sum()),
            "sample_count": int(len(group)),
            "max_suspect_score": float(pd.to_numeric(group.get("suspect_score"), errors="coerce").max()),
            "mean_suspect_score": float(pd.to_numeric(group.get("suspect_score"), errors="coerce").mean()),
            "max_error_seed_count": int(pd.to_numeric(group.get("error_seed_count"), errors="coerce").max()),
            "review_status": "todo",
            "proposed_action": "",
            "reviewer": "",
            "review_note": "",
            "issue_types": _join_unique(group.get("issue_type", pd.Series(dtype=object))),
            "audit_priorities": _join_unique(group.get("audit_priority", pd.Series(dtype=object))),
            "reason_names": _join_unique(group.get("reason_name", pd.Series(dtype=object))),
            "sample_ids": _join_unique(group.get("sample_id", pd.Series(dtype=object))),
            "relative_paths": _join_unique(group.get("relative_path", pd.Series(dtype=object))),
            "tdms_paths": _join_unique(group.get("tdms_path", pd.Series(dtype=object))),
            "model_seeds_seen": _join_unique(group.get("model_seeds_seen", pd.Series(dtype=object))),
        }
        rows.append(row)

    sn_df = pd.DataFrame(rows)
    if sn_df.empty:
        return sn_df

    sn_df["_priority_rank"] = sn_df["audit_priority_top"].map(priority_order).fillna(99)
    sn_df = sn_df.sort_values(
        by=["_priority_rank", "has_false_ok", "max_suspect_score", "false_ok_sample_count", "sample_count", "sn"],
        ascending=[True, False, False, False, False, True],
        na_position="last",
    ).drop(columns=["_priority_rank"])
    ordered_cols = [col for col in SN_REVIEW_COLUMNS if col in sn_df.columns]
    rest_cols = [col for col in sn_df.columns if col not in ordered_cols]
    return sn_df[ordered_cols + rest_cols]


def write_readme(
    output_dir: Path,
    candidates: pd.DataFrame,
    suspicious_sn_df: pd.DataFrame,
    events: pd.DataFrame,
    source_files: list[Path],
) -> None:
    false_ok_count = int((candidates["issue_type"] == "false_ok_NOK_as_OK").sum()) if not candidates.empty else 0
    false_reject_count = int((candidates["issue_type"] == "false_reject_OK_as_NOK").sum()) if not candidates.empty else 0
    nok_type_count = int((candidates["issue_type"] == "nok_type_confusion").sum()) if not candidates.empty else 0
    text = f"""# 标签审计输出

本目录由 `AI-2.0/ml/training/audit_labels.py` 生成。脚本只生成复核清单，不会修改原始标签。

## 输入

- 错分样本文件数：{len(source_files)}
- 错分事件数：{len(events)}

## 输出文件

- `label_audit_candidates.csv`：按样本聚合后的总复核清单，优先处理这个文件。
- `suspicious_sn_review.csv`：按 `sn` 聚合的待标注/待复核名单，便于先从整件维度排优先级。
- `false_ok_review.csv`：实际 NOK 被预测为 OK，出厂漏检风险最高，优先人工复核。
- `false_reject_review.csv`：实际 OK 被预测为 NOK，主要影响误杀率。
- `nok_type_confusion_review.csv`：NOK 类型之间混淆，用于优化异常细分类和论文分析。
- `low_confidence_error_review.csv`：错分样本中的低置信/小分差样本，可视为第一批边界样本；预测正确但低置信的边界样本需要导出全量预测后补充。
- `label_audit_events.csv`：逐 seed 错分事件明细。
- `label_source_stats.csv`：不同标注来源与标签类型统计。
- `same_tdms_mixed_channel_labels.csv`：同一 TDMS/序列下 Up/Down 等通道标签不一致的样本，需结合业务判断，不一定是错标。
- `exact_key_label_conflicts.csv`：完全相同样本主键下标签冲突；如果非空，需要优先处理。

## forvia_label 对接文件

以下文件保存在 `confirm_label/` 目录下，可直接在 forvia_label 的 sample_view 路径输入框中加载：

- `sample_view_label_audit.csv`：全部复核候选。
- `sample_view_false_ok_review.csv`：NOK 被预测为 OK，优先复核。
- `sample_view_low_confidence_error_review.csv`：错分中的低置信/小分差样本。

## 当前聚合结果

- 复核候选样本数：{len(candidates)}
- 待复核 SN 数：{len(suspicious_sn_df)}
- false OK 候选：{false_ok_count}
- false reject 候选：{false_reject_count}
- NOK 类型混淆候选：{nok_type_count}

## 人工复核建议

1. 先处理 `false_ok_review.csv` 里的 `P0_critical_false_ok`。
2. 对每行填写 `review_status`、`proposed_action`、`proposed_label`、`reviewer`、`review_note`。
3. `proposed_action` 建议只使用：`keep_label`、`change_to_OK`、`change_to_NOK`、`change_reason`、`exclude_borderline`、`exclude_bad_signal`。
4. 第一轮不要直接覆盖训练标签；先保存复核结果，再用复核结果生成一份 cleaned sample_view 重新训练对比。
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run(results_dir: Path, output_dir: Path) -> dict[str, Any]:
    results_dir = results_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_confirm_label_dir(output_dir)

    source_files = find_misclassified_files(results_dir)
    if not source_files:
        raise FileNotFoundError(
            "未找到错分样本文件: "
            f"{results_dir}/eval/best_model_eval/seed_*/"
            "{test,validation_test}_misclassified_sample_view.csv"
        )

    events = load_error_events(source_files)
    candidates = aggregate_candidates(events)
    false_ok_df = candidates[candidates["issue_type"].eq("false_ok_NOK_as_OK")].copy()

    _write_sample_view_pair(
        df=candidates,
        view_name="label_audit",
        audit_dir=output_dir,
        filename="sample_view_label_audit.csv",
    )
    _write_sample_view_pair(
        df=false_ok_df,
        view_name="label_audit_false_ok",
        audit_dir=output_dir,
        filename="sample_view_false_ok_review.csv",
    )

    return {
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "source_files": len(source_files),
        "events": len(events),
        "candidates": len(candidates),
        "exported_files": len(FINAL_SAMPLE_VIEW_FILENAMES),
        "false_ok": int((candidates["issue_type"] == "false_ok_NOK_as_OK").sum()) if not candidates.empty else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="聚合模型错分样本，生成标签复核清单")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="训练结果目录")
    parser.add_argument("--output-dir", default="", help="输出目录，默认 results-dir/confirm_label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "confirm_label"
    summary = run(results_dir=results_dir, output_dir=output_dir)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
