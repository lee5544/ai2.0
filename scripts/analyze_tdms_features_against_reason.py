#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TDMS features with train-set reason distributions.")
    parser.add_argument("--tdms", required=True, help="TDMS path to analyze.")
    parser.add_argument("--model-dir", required=True, help="Model result directory.")
    parser.add_argument("--line", default="epump2", help="Line for TDMS parsing.")
    parser.add_argument("--output-dir", default="", help="Output directory.")
    parser.add_argument("--target-reason", default="秒表", help="Target reason name.")
    return parser.parse_args()


def _percentile(sorted_values: np.ndarray, value: float) -> float:
    if sorted_values.size == 0 or not np.isfinite(value):
        return float("nan")
    return float(np.searchsorted(sorted_values, value, side="right") / sorted_values.size)


def _safe_iqr(values: pd.Series) -> float:
    q75 = float(values.quantile(0.75))
    q25 = float(values.quantile(0.25))
    iqr = q75 - q25
    if np.isfinite(iqr) and iqr > 1e-12:
        return iqr
    std = float(values.std(ddof=0))
    if np.isfinite(std) and std > 1e-12:
        return std
    return 1.0


def _reason_group(reason: object) -> str:
    text = str(reason or "").strip()
    if text in {"正常", "干扰", "边界"}:
        return "normal_group"
    return text or "<empty>"


def _load_training_features(model_dir: Path) -> pd.DataFrame:
    files = sorted((model_dir / "dataset_csv").glob("features_batch_*.csv"))
    if not files:
        raise FileNotFoundError(f"No features_batch_*.csv under {model_dir / 'dataset_csv'}")
    return pd.concat((pd.read_csv(path, encoding="utf-8-sig") for path in files), ignore_index=True, sort=False)


def _analyze_one_channel(
    *,
    channel: str,
    raw_signal: np.ndarray,
    sr: int,
    predictor,
    train_df: pd.DataFrame,
    target_reason: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    features = predictor.extract_features(raw_signal, sr)
    feature_columns = list(predictor.feature_columns)
    importances = np.asarray(getattr(predictor.model, "feature_importances_", np.zeros(len(feature_columns))), dtype=float)
    if importances.size != len(feature_columns):
        importances = np.zeros(len(feature_columns), dtype=float)

    work = train_df.copy()
    work["_reason_group"] = work["reason_name"].map(_reason_group)
    normal_df = work[work["_reason_group"].eq("normal_group")].copy()
    target_df = work[work["reason_name"].astype(str).str.strip().eq(target_reason)].copy()
    if normal_df.empty:
        raise ValueError("No normal_group rows in training features.")
    if target_df.empty:
        raise ValueError(f"No target reason rows in training features: {target_reason}")

    rows: list[dict[str, object]] = []
    normal_like_count = 0
    target_like_count = 0
    weighted_normal_votes = 0.0
    weighted_target_votes = 0.0
    sep_scores: list[float] = []

    for idx, feature in enumerate(feature_columns):
        if feature not in train_df.columns:
            continue
        value = float(features.get(feature, np.nan))
        normal_values = pd.to_numeric(normal_df[feature], errors="coerce").dropna()
        target_values = pd.to_numeric(target_df[feature], errors="coerce").dropna()
        if normal_values.empty or target_values.empty:
            continue

        normal_median = float(normal_values.median())
        target_median = float(target_values.median())
        normal_iqr = _safe_iqr(normal_values)
        target_iqr = _safe_iqr(target_values)
        pooled_scale = max(1e-12, (normal_iqr + target_iqr) / 2.0)
        dist_normal = abs(value - normal_median) / max(normal_iqr, 1e-12)
        dist_target = abs(value - target_median) / max(target_iqr, 1e-12)
        closer = "target" if dist_target < dist_normal else "normal"
        importance = float(importances[idx])
        sep = abs(target_median - normal_median) / pooled_scale
        sep_score = importance * sep
        sep_scores.append(sep_score)
        if closer == "target":
            target_like_count += 1
            weighted_target_votes += importance
        else:
            normal_like_count += 1
            weighted_normal_votes += importance

        rows.append(
            {
                "channel": channel,
                "feature": feature,
                "value": value,
                "importance": importance,
                "normal_median": normal_median,
                "target_median": target_median,
                "normal_iqr": normal_iqr,
                "target_iqr": target_iqr,
                "normal_percentile": _percentile(np.sort(normal_values.to_numpy(dtype=float)), value),
                "target_percentile": _percentile(np.sort(target_values.to_numpy(dtype=float)), value),
                "dist_to_normal_iqr": dist_normal,
                "dist_to_target_iqr": dist_target,
                "closer_to": closer,
                "target_vs_normal_separation": sep,
                "importance_x_separation": sep_score,
            }
        )

    result_df = pd.DataFrame(rows)
    summary = {
        "channel": channel,
        "features_compared": int(len(result_df)),
        "normal_like_feature_count": int(normal_like_count),
        "target_like_feature_count": int(target_like_count),
        "weighted_normal_like_importance": round(float(weighted_normal_votes), 6),
        "weighted_target_like_importance": round(float(weighted_target_votes), 6),
        "target_reason": target_reason,
        "normal_rows": int(len(normal_df)),
        "target_rows": int(len(target_df)),
    }
    return result_df, summary


def main() -> None:
    args = _parse_args()
    tdms_path = Path(args.tdms).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else tdms_path.parent / f"feature_analysis_{model_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_dir = model_dir / "runtime"
    sys.path.insert(0, str(runtime_dir))

    from ChannelPredictor import ChannelPredictor  # noqa: E402
    from tdms_read import read_tdms  # noqa: E402

    predictor = ChannelPredictor(model_dir=model_dir)
    train_df = _load_training_features(model_dir)
    data = read_tdms(tdms_path, line=args.line)
    sr = int(data.get("sampling_rate") or 20000)

    all_dfs: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for channel, raw_signal in (("up", data["up_data"]), ("down", data["down_data"])):
        df, summary = _analyze_one_channel(
            channel=channel,
            raw_signal=raw_signal,
            sr=sr,
            predictor=predictor,
            train_df=train_df,
            target_reason=args.target_reason,
        )
        all_dfs.append(df)
        summaries.append(summary)

    analysis_df = pd.concat(all_dfs, ignore_index=True, sort=False)
    analysis_path = output_dir / f"{tdms_path.stem}_feature_analysis_vs_{args.target_reason}.csv"
    analysis_df.to_csv(analysis_path, index=False, encoding="utf-8-sig")

    key_df = analysis_df.sort_values(
        by=["importance_x_separation", "importance"],
        ascending=[False, False],
    ).head(60)
    key_path = output_dir / f"{tdms_path.stem}_key_features_vs_{args.target_reason}.csv"
    key_df.to_csv(key_path, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / f"{tdms_path.stem}_summary_vs_{args.target_reason}.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"analysis_csv={analysis_path}")
    print(f"key_features_csv={key_path}")
    print(f"summary_csv={summary_path}")
    print(summary_df.to_string(index=False))
    print("\nTop key features:")
    show_cols = [
        "channel",
        "feature",
        "value",
        "normal_median",
        "target_median",
        "normal_percentile",
        "target_percentile",
        "closer_to",
        "importance",
        "importance_x_separation",
    ]
    print(key_df[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
