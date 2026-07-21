"""ML 分析可视化工具。

包含：
  main_tsne()            — t-SNE 特征可视化（原 analysis/plot_tsne.py）
  main_select_features() — 特征重要性筛选（原 analysis/select_features.py）

CLI 用法：
  python -m ml.plot tsne       --config cfg/epump4.yaml
  python -m ml.plot features   --config cfg/epump4.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


try:
    from ml.training.loader import load_sampled_training_dataframe, load_training_dataframe
    from ml.training.config import (
        DEFAULT_CONFIG_PATH,
        NON_FEATURE_COLUMNS,
        FeatureSchemaError,
        _looks_like_numeric_feature,
        _to_int,
        build_label_runtime,
        build_training_bootstrap,
        resolve_feature_columns_by_schema,
    )
except ModuleNotFoundError:  # pragma: no cover — standalone bundle
    from dataset_loader import load_sampled_training_dataframe, load_training_dataframe  # type: ignore[import]
    from model_support import (  # type: ignore[import]
        DEFAULT_CONFIG_PATH,
        NON_FEATURE_COLUMNS,
        FeatureSchemaError,
        _looks_like_numeric_feature,
        _to_int,
        build_label_runtime,
        build_training_bootstrap,
        resolve_feature_columns_by_schema,
    )


# ===========================================================================
# t-SNE（原 analysis/plot_tsne.py）
# ===========================================================================

def _tsne_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制特征数据的 t-SNE")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=f"YAML 配置路径（默认: {DEFAULT_CONFIG_PATH}）")
    parser.add_argument("--output-dir", default="", help="输出目录，默认写到 results/<model_id>/tsne/")
    parser.add_argument("--output-prefix", default="feature_tsne", help="输出文件前缀")
    parser.add_argument("--color-by", default="reason_name",
                        choices=["reason_name", "reason_id", "result_name", "line", "reference", "channel_name"],
                        help="按哪一列着色")
    parser.add_argument("--all-reasons", action="store_true", help="忽略 train.label_mapping，使用全部 reason_id")
    parser.add_argument("--max-points", type=int, default=-1, help="最多参与 t-SNE 的样本数，默认全量（<=0 表示全量）")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--pca-components", type=int, default=50, help="t-SNE 前的 PCA 降维维度（<=0 跳过）")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    return parser.parse_args(argv)


def _to_text_tsne(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _configure_matplotlib_for_chinese() -> str:
    plt.rcParams["axes.unicode_minus"] = False
    candidates = [
        "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei",
        "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
        "Arial Unicode MS", "STHeiti", "Heiti SC",
    ]
    try:
        available = {font.name for font in font_manager.fontManager.ttflist}
    except Exception:
        available = set()
    selected = [name for name in candidates if name in available]
    if not selected:
        return ""
    current = plt.rcParams.get("font.sans-serif", [])
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = selected + [name for name in current if name not in selected]
    return selected[0]


def _resolve_feature_columns_tsne(df: pd.DataFrame) -> list[str]:
    return resolve_feature_columns_by_schema(df)


def _resolve_output_dir_tsne(args: argparse.Namespace, results_path: str) -> Path:
    if _to_text_tsne(args.output_dir):
        output_dir = Path(args.output_dir).expanduser()
    else:
        output_dir = Path(results_path).expanduser() / "tsne"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_valid_labels(bootstrap: Any, all_reasons: bool) -> list[int]:
    if all_reasons:
        return sorted(
            int(item["reason_id"])
            for item in bootstrap.available_reasons
            if item.get("reason_id") is not None
        )
    _, label_sample_dict, _, _ = build_label_runtime(
        train_cfg=bootstrap.train_cfg,
        available_reasons=bootstrap.available_reasons,
    )
    return sorted(int(label) for label in label_sample_dict.keys())


def _sample_balanced(df: pd.DataFrame, *, label_col: str, max_points: int, random_state: int) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df.copy()

    rng = np.random.default_rng(random_state)
    groups = []
    for _, sub_df in df.groupby(label_col, sort=True, dropna=False):
        shuffled_idx = rng.permutation(sub_df.index.to_numpy())
        groups.append(shuffled_idx.tolist())

    picked_indices: list[int] = []
    quota = max(1, max_points // max(1, len(groups)))
    remainders: list[int] = []

    for group_indices in groups:
        picked_indices.extend(group_indices[:quota])
        remainders.extend(group_indices[quota:])

    if len(picked_indices) > max_points:
        picked_indices = rng.choice(np.array(picked_indices), size=max_points, replace=False).tolist()
    elif len(picked_indices) < max_points and remainders:
        need = min(max_points - len(picked_indices), len(remainders))
        picked_indices.extend(rng.choice(np.array(remainders), size=need, replace=False).tolist())

    sampled_df = df.loc[sorted(picked_indices)].copy()
    print(f"[INFO] t-SNE 采样: {len(df)} -> {len(sampled_df)}")
    return sampled_df


def _resolve_perplexity(requested: float, n_samples: int) -> float:
    if n_samples < 3:
        raise RuntimeError(f"t-SNE 样本数不足，至少需要 3 条，当前: {n_samples}")
    upper_bound = min(float(n_samples - 1), max(1.0, float(n_samples - 1) / 3.0))
    resolved = max(1.0, min(float(requested), upper_bound))
    if abs(resolved - float(requested)) > 1e-9:
        print(f"[INFO] perplexity 自动调整: {requested} -> {resolved:.3f}")
    return resolved


def _fit_tsne(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    pca_components: int,
    perplexity: float,
    random_state: int,
) -> np.ndarray:
    X_df = df.reindex(columns=feature_cols, fill_value=0.0)
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X_df.to_numpy(dtype=np.float32, copy=False)
    X = StandardScaler().fit_transform(X)

    if pca_components > 0 and X.shape[1] > pca_components:
        pca_dim = min(int(pca_components), X.shape[0], X.shape[1])
        if pca_dim >= 2:
            print(f"[INFO] PCA 预降维: {X.shape[1]} -> {pca_dim}")
            X = PCA(n_components=pca_dim, random_state=random_state).fit_transform(X)

    resolved_perplexity = _resolve_perplexity(perplexity, len(df))
    print(
        f"[INFO] 开始 t-SNE: samples={len(df)}, features={len(feature_cols)}, "
        f"perplexity={resolved_perplexity:.3f}, seed={random_state}"
    )
    return TSNE(
        n_components=2, perplexity=resolved_perplexity,
        init="pca", learning_rate="auto", random_state=random_state,
    ).fit_transform(X)


def _plot_embedding(df: pd.DataFrame, *, color_by: str, output_png: Path) -> None:
    plot_df = df.copy()
    plot_df[color_by] = plot_df[color_by].map(_to_text_tsne)
    plot_df.loc[plot_df[color_by] == "", color_by] = "<empty>"

    counts = plot_df[color_by].value_counts(dropna=False).to_dict()
    labels = sorted(counts.keys(), key=lambda key: (-int(counts[key]), str(key)))
    cmap_name = "tab20" if len(labels) <= 20 else "gist_ncar"
    cmap = plt.colormaps.get_cmap(cmap_name).resampled(max(len(labels), 1))

    fig, ax = plt.subplots(figsize=(14, 10))
    for idx, label in enumerate(labels):
        sub_df = plot_df[plot_df[color_by] == label]
        ax.scatter(sub_df["tsne_x"], sub_df["tsne_y"], s=16, alpha=0.75, color=cmap(idx),
                   label=f"{label} (n={len(sub_df)})")

    ax.set_title(f"Feature t-SNE by {color_by}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main_tsne(argv: list[str] | None = None) -> None:
    args = _tsne_parse_args(argv)
    font_name = _configure_matplotlib_for_chinese()
    if font_name:
        print(f"[INFO] matplotlib 中文字体: {font_name}")
    else:
        print("[WARN] 未检测到可用中文字体，图中中文可能显示为方框。")

    bootstrap = build_training_bootstrap(args.config, default_model_type="xgb")
    valid_labels = _resolve_valid_labels(bootstrap, args.all_reasons)
    if not valid_labels:
        raise RuntimeError("没有可用于 t-SNE 的标签。")

    df = load_training_dataframe(bootstrap.dataset_files, valid_labels=valid_labels)
    feature_cols = _resolve_feature_columns_tsne(df)
    color_by = args.color_by
    if color_by not in df.columns:
        raise KeyError(f"数据中缺少 color_by 列: {color_by}")

    sampled_df = _sample_balanced(
        df,
        label_col="reason_id" if "reason_id" in df.columns else color_by,
        max_points=int(args.max_points),
        random_state=int(args.random_state),
    )
    embedding = _fit_tsne(
        sampled_df,
        feature_cols=feature_cols,
        pca_components=int(args.pca_components),
        perplexity=float(args.perplexity),
        random_state=int(args.random_state),
    )

    output_dir = _resolve_output_dir_tsne(args, bootstrap.results_path)
    output_prefix = _to_text_tsne(args.output_prefix) or "feature_tsne"
    output_png = output_dir / f"{output_prefix}_{color_by}.png"
    output_csv = output_dir / f"{output_prefix}_{color_by}.csv"

    sampled_df = sampled_df.copy()
    sampled_df["tsne_x"] = embedding[:, 0]
    sampled_df["tsne_y"] = embedding[:, 1]

    export_cols = [
        "line", "reference", "sn", "sample_id", "group_name", "channel_name",
        "result_id", "result_name", "reason_id", "reason_name", color_by, "tsne_x", "tsne_y",
    ]
    export_cols = [col for col in export_cols if col in sampled_df.columns]
    sampled_df[export_cols].to_csv(output_csv, index=False, encoding="utf-8-sig")
    _plot_embedding(sampled_df, color_by=color_by, output_png=output_png)

    print(f"[INFO] t-SNE 图已保存: {output_png}")
    print(f"[INFO] t-SNE 坐标已保存: {output_csv}")


# ===========================================================================
# 特征选择（原 analysis/select_features.py）
# ===========================================================================

def _sf_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="筛选 AI-2.0 特征，输出特征排名、分组重要性和推荐清单。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=f"训练配置 YAML（默认: {DEFAULT_CONFIG_PATH}）")
    parser.add_argument("--top-k", type=int, default=80, help="推荐保留特征数，默认 80")
    parser.add_argument("--min-features", type=int, default=40, help="相关性去冗余后的最少保留特征数，默认 40")
    parser.add_argument("--candidate-multiplier", type=float, default=3.0, help="自动降维参与相关性去冗余的候选池倍数，默认 3.0")
    parser.add_argument("--corr-threshold", type=float, default=0.995, help="冗余绝对相关系数阈值，默认 0.995")
    parser.add_argument("--sample-size", type=int, default=30000, help="用于筛选的最大样本数；<=0 表示全量，默认 30000")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--n-estimators", type=int, default=300, help="ExtraTrees 树数量")
    parser.add_argument("--with-mutual-info", action="store_true", help="额外计算 mutual information（较慢）")
    parser.add_argument("--output-dir", default=None, help="输出目录；默认 <results_path>/feature_selection")
    return parser.parse_args(argv)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是 dict: {path}")
    return data


def _feature_group(feature: str) -> str:
    if feature.startswith("mel_band_"):
        return "mel_band"
    if feature.startswith("mel_shock_"):
        return "mel_shock"
    if feature.startswith("dwt_"):
        return "dwt"
    if feature in {"mean", "time_std", "duration"} or feature.startswith("time_"):
        return "time"
    return feature.split("_", 1)[0] if "_" in feature else "other"


def _select_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return resolve_feature_columns_by_schema(df)


def _stratified_sample(df: pd.DataFrame, *, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy()
    parts: list[pd.DataFrame] = []
    total = len(df)
    for _, sub in df.groupby("label", sort=True):
        target = max(1, int(math.ceil(max_rows * len(sub) / total)))
        parts.append(sub.sample(n=min(len(sub), target), random_state=random_state))
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state)
    return sampled.reset_index(drop=True)


def _rank_desc(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    series = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return series.rank(method="average", ascending=False, pct=True).to_numpy(dtype=float)


def _safe_f_classif(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(np.unique(y)) < 2:
        return np.zeros(X.shape[1], dtype=float)
    scores, _ = f_classif(X, y)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def _tree_importance(X: np.ndarray, y: np.ndarray, *, random_state: int, n_estimators: int) -> np.ndarray:
    if len(np.unique(y)) < 2:
        return np.zeros(X.shape[1], dtype=float)
    model = ExtraTreesClassifier(
        n_estimators=int(n_estimators), max_features="sqrt",
        class_weight="balanced", random_state=int(random_state), n_jobs=-1,
    )
    model.fit(X, y)
    return np.nan_to_num(model.feature_importances_, nan=0.0)


def _feature_quality_table(X_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    numeric = X_df.loc[:, feature_cols]
    return pd.DataFrame({
        "feature": feature_cols,
        "nan_rate": numeric.isna().mean().to_numpy(dtype=float),
        "nunique": numeric.nunique(dropna=True).to_numpy(dtype=int),
        "variance": numeric.var(axis=0).to_numpy(dtype=float),
        "std": numeric.std(axis=0).to_numpy(dtype=float),
    })


def _correlated_pairs(X_df: pd.DataFrame, feature_order: list[str], *, corr_threshold: float) -> pd.DataFrame:
    if len(feature_order) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"])
    X = X_df.loc[:, feature_order].copy().fillna(X_df.median(numeric_only=True)).fillna(0.0)
    corr = X.corr().abs()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    upper = corr.where(mask)
    rows: list[dict[str, Any]] = []
    for feature_b in upper.columns:
        s = upper[feature_b].dropna()
        for feature_a, value in s[s >= float(corr_threshold)].items():
            rows.append({"feature_a": str(feature_a), "feature_b": str(feature_b), "abs_corr": float(value)})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)


def _auto_select_features(
    ranking_df: pd.DataFrame,
    X_df: pd.DataFrame,
    *,
    top_k: int,
    min_features: int,
    candidate_multiplier: float,
    corr_threshold: float,
) -> tuple[list[str], pd.DataFrame]:
    all_ranked = ranking_df["feature"].astype(str).tolist()
    if not all_ranked:
        return [], pd.DataFrame()

    target = max(1, min(int(top_k), len(all_ranked)))
    min_features = max(1, min(int(min_features), len(all_ranked)))
    target = max(target, min_features)
    candidate_count = min(max(target, int(math.ceil(target * max(float(candidate_multiplier), 1.0)))), len(all_ranked))
    candidate_features = all_ranked[:candidate_count]

    X = X_df.loc[:, candidate_features].copy().fillna(X_df.median(numeric_only=True)).fillna(0.0)
    corr = X.corr().abs()

    selected: list[str] = []
    dropped_rows: list[dict[str, Any]] = []
    rank_pos = {feature: idx + 1 for idx, feature in enumerate(all_ranked)}

    for feature in candidate_features:
        redundant_with = ""
        redundant_corr = 0.0
        for kept in selected:
            value = float(corr.loc[feature, kept])
            if value >= float(corr_threshold):
                redundant_with, redundant_corr = kept, value
                break
        if redundant_with and len(selected) >= min_features:
            dropped_rows.append({
                "dropped_feature": feature, "kept_feature": redundant_with,
                "abs_corr": redundant_corr, "dropped_rank": rank_pos.get(feature),
                "kept_rank": rank_pos.get(redundant_with), "reason": f"abs_corr>={corr_threshold:g}",
            })
            continue
        selected.append(feature)
        if len(selected) >= target:
            break

    if len(selected) < target:
        selected_set = set(selected)
        for feature in all_ranked:
            if feature not in selected_set:
                selected.append(feature)
                selected_set.add(feature)
                if len(selected) >= target:
                    break

    return selected, pd.DataFrame(dropped_rows)


def _write_selected_files(selected: list[str], out_dir: Path, *, name: str) -> tuple[Path, Path]:
    if not selected:
        raise RuntimeError("自动降维没有选出任何特征。")
    top_k = len(selected)
    txt_path = out_dir / f"{name}_{top_k}.txt"
    txt_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    yaml_path = out_dir / f"{name}_{top_k}.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {"train": {"feature_selection": {"selected_features_file": str(txt_path)}}},
            allow_unicode=True, sort_keys=False,
        ),
        encoding="utf-8",
    )
    return txt_path, yaml_path


def main_select_features(argv: list[str] | None = None) -> None:
    args = _sf_parse_args(argv)
    config_path = Path(args.config).expanduser()
    cfg = _read_yaml(config_path)
    train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    default_model_type = str(train_cfg.get("model_type") or "xgb")

    bootstrap = build_training_bootstrap(config_path, default_model_type=default_model_type)
    labeler, label_sample_dict, mlabel_mtype_dict, label_to_mlabel_map = build_label_runtime(
        train_cfg=bootstrap.train_cfg, available_reasons=bootstrap.available_reasons,
    )
    del labeler, mlabel_mtype_dict

    df = load_sampled_training_dataframe(
        bootstrap.dataset_files,
        label_sample_dict=label_sample_dict,
        label_to_mlabel_map=label_to_mlabel_map,
        random_state=int(args.random_state),
        allow_oversample=False,
    )
    df = _stratified_sample(df, max_rows=int(args.sample_size), random_state=int(args.random_state))

    feature_cols = _select_numeric_feature_columns(df)
    X_raw_df = df.reindex(columns=feature_cols, fill_value=0.0).apply(pd.to_numeric, errors="coerce")
    quality_all_df = _feature_quality_table(X_raw_df, feature_cols)
    X_df = X_raw_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    variances = X_df.var(axis=0).to_numpy(dtype=float)
    keep_mask = variances > 1e-12
    dropped_constant = int((~keep_mask).sum())
    feature_cols = [c for c, keep in zip(feature_cols, keep_mask) if keep]
    X = X_df.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_multi = df["label"].to_numpy(dtype=int)

    ok_label = _to_int(bootstrap.train_cfg.get("ok_mlabel"))
    if ok_label is None:
        predictor_cfg = cfg.get("predictor") if isinstance(cfg.get("predictor"), dict) else {}
        ok_label = _to_int(predictor_cfg.get("ok_mlabel"))
    if ok_label is None:
        ok_label = 0
    y_ok_nok = (y_multi != int(ok_label)).astype(int)

    print(f"[INFO] 特征筛选数据: rows={X.shape[0]}, features={X.shape[1]}, "
          f"dropped_constant={dropped_constant}, ok_label={ok_label}")

    f_multi = _safe_f_classif(X, y_multi)
    f_ok_nok = _safe_f_classif(X, y_ok_nok)
    tree_multi = _tree_importance(X, y_multi, random_state=int(args.random_state), n_estimators=int(args.n_estimators))
    tree_ok_nok = _tree_importance(X, y_ok_nok, random_state=int(args.random_state) + 17, n_estimators=int(args.n_estimators))

    mi_multi = mi_ok_nok = np.zeros(X.shape[1], dtype=float)
    if bool(args.with_mutual_info):
        print("[INFO] 计算 mutual information...")
        mi_multi = np.nan_to_num(mutual_info_classif(X, y_multi, random_state=int(args.random_state)), nan=0.0)
        if len(np.unique(y_ok_nok)) >= 2:
            mi_ok_nok = np.nan_to_num(mutual_info_classif(X, y_ok_nok, random_state=int(args.random_state) + 17), nan=0.0)

    if bool(args.with_mutual_info):
        combined_rank = (0.15 * _rank_desc(f_multi) + 0.20 * _rank_desc(f_ok_nok)
                         + 0.15 * _rank_desc(mi_multi) + 0.15 * _rank_desc(mi_ok_nok)
                         + 0.15 * _rank_desc(tree_multi) + 0.20 * _rank_desc(tree_ok_nok))
    else:
        combined_rank = (0.20 * _rank_desc(f_multi) + 0.25 * _rank_desc(f_ok_nok)
                         + 0.20 * _rank_desc(tree_multi) + 0.35 * _rank_desc(tree_ok_nok))

    ranking_df = pd.DataFrame({
        "feature": feature_cols,
        "group": [_feature_group(c) for c in feature_cols],
        "combined_rank_score": combined_rank,
        "f_classif_multiclass": f_multi,
        "f_classif_ok_nok": f_ok_nok,
        "tree_importance_multiclass": tree_multi,
        "tree_importance_ok_nok": tree_ok_nok,
        "mutual_info_multiclass": mi_multi,
        "mutual_info_ok_nok": mi_ok_nok,
    }).sort_values(by=["combined_rank_score", "tree_importance_ok_nok", "f_classif_ok_nok"],
                   ascending=[True, False, False])

    out_dir = Path(args.output_dir).expanduser() if args.output_dir else Path(bootstrap.results_path) / "feature_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = out_dir / "feature_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False, encoding="utf-8-sig")

    group_df = (ranking_df.assign(top_20=ranking_df["combined_rank_score"].rank(method="first") <= 20)
                .groupby("group", as_index=False).agg(
                    feature_count=("feature", "count"), top20_count=("top_20", "sum"),
                    mean_tree_ok_nok=("tree_importance_ok_nok", "mean"),
                    sum_tree_ok_nok=("tree_importance_ok_nok", "sum"),
                    mean_tree_multiclass=("tree_importance_multiclass", "mean"),
                    sum_tree_multiclass=("tree_importance_multiclass", "sum"),
                    best_combined_rank_score=("combined_rank_score", "min"),
                ).sort_values(by=["top20_count", "sum_tree_ok_nok", "sum_tree_multiclass"], ascending=False))
    group_path = out_dir / "feature_group_summary.csv"
    group_df.to_csv(group_path, index=False, encoding="utf-8-sig")

    quality_path = out_dir / "feature_quality.csv"
    quality_all_df.merge(ranking_df.loc[:, ["feature", "combined_rank_score"]], on="feature", how="left") \
        .sort_values(["combined_rank_score", "feature"], na_position="last") \
        .to_csv(quality_path, index=False, encoding="utf-8-sig")

    corr_threshold = min(max(float(args.corr_threshold), 0.0), 1.0)
    redundant_df = _correlated_pairs(X_df, ranking_df["feature"].astype(str).tolist(), corr_threshold=corr_threshold)
    redundant_path = out_dir / "feature_redundant_pairs.csv"
    redundant_df.to_csv(redundant_path, index=False, encoding="utf-8-sig")

    selected_txt, selected_yaml = _write_selected_files(
        ranking_df.head(max(1, min(int(args.top_k), len(ranking_df))))["feature"].astype(str).tolist(),
        out_dir, name="selected_features_top",
    )
    auto_selected, auto_dropped_df = _auto_select_features(
        ranking_df, X_df, top_k=int(args.top_k), min_features=int(args.min_features),
        candidate_multiplier=float(args.candidate_multiplier), corr_threshold=corr_threshold,
    )
    auto_txt, auto_yaml = _write_selected_files(auto_selected, out_dir, name="selected_features_auto")
    auto_dropped_path = out_dir / "feature_auto_dropped_redundant.csv"
    auto_dropped_df.to_csv(auto_dropped_path, index=False, encoding="utf-8-sig")

    for label, path in [
        ("特征排名", ranking_path), ("特征分组摘要", group_path), ("特征质量诊断", quality_path),
        ("高相关冗余对", redundant_path), ("Top-K 特征清单", selected_txt),
        ("自动降维特征清单", auto_txt), ("自动降维丢弃记录", auto_dropped_path),
    ]:
        print(f"[INFO] {label}已保存: {path}")

    print("\nTop 30 features:")
    show_cols = ["feature", "group", "combined_rank_score", "tree_importance_ok_nok", "f_classif_ok_nok", "tree_importance_multiclass"]
    with pd.option_context("display.max_rows", 30, "display.max_colwidth", 80):
        print(ranking_df.loc[:, show_cols].head(30).to_string(index=False))


# ===========================================================================
# CLI 调度
# ===========================================================================

def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("tsne", "features"):
        print("用法: python -m ml.plot {tsne|features} [选项...]")
        print("  tsne      — 绘制 t-SNE 特征分布图")
        print("  features  — 特征重要性筛选与排名")
        sys.exit(1)

    mode = sys.argv[1]
    remaining = sys.argv[2:]
    if mode == "tsne":
        main_tsne(remaining)
    else:
        main_select_features(remaining)


if __name__ == "__main__":
    main()
