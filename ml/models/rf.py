import argparse
import os
import sys
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Any, Dict, List

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_validate, GridSearchCV
)
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
    accuracy_score, precision_score,
    recall_score, f1_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from ml.runtime import (
    build_threshold_scorer,
    build_uniform_threshold_dict,
    predict_with_threshold_rule,
)
from ml.runtime import export_runtime_bundle
try:
    from training_diagnostics import compute_quality_gate_metrics, generate_training_diagnostics
except ModuleNotFoundError:  # pragma: no cover
    from ml.training.diagnostics import compute_quality_gate_metrics, generate_training_diagnostics
try:
    from features import resolve_feature_version
except ModuleNotFoundError:  # pragma: no cover
    from ml.features import resolve_feature_version
try:
    from dataset_split import (
        PreparedCrossValidationData,
        TrainValTestSplit,
        resolve_split_strategy_name,
    )
except ModuleNotFoundError:  # pragma: no cover
    from ml.dataset.split import (
        PreparedCrossValidationData,
        TrainValTestSplit,
        resolve_split_strategy_name,
    )
try:
    from model_support import (
        DEFAULT_CONFIG_PATH,
        NON_FEATURE_COLUMNS,
        SAMPLE_VIEW_EXPORT_COLUMNS,
        FeatureSchemaError,
        _looks_like_numeric_feature,
        _to_int,
        _to_text,
        build_label_runtime as _shared_build_label_runtime,
        build_training_bootstrap as _build_training_bootstrap,
        resolve_configured_feature_columns,
        resolve_feature_columns_by_schema,
    )
except ModuleNotFoundError:  # pragma: no cover
    from ml.training.config import (
        DEFAULT_CONFIG_PATH,
        NON_FEATURE_COLUMNS,
        SAMPLE_VIEW_EXPORT_COLUMNS,
        FeatureSchemaError,
        _looks_like_numeric_feature,
        _to_int,
        _to_text,
        build_label_runtime as _shared_build_label_runtime,
        build_training_bootstrap as _build_training_bootstrap,
        resolve_configured_feature_columns,
        resolve_feature_columns_by_schema,
    )

try:
    from ml.models.base import BaseModel
except ModuleNotFoundError:  # pragma: no cover
    from base_model import BaseModel


MISCLASSIFIED_EXPORT_FILENAME = "validation_test_misclassified_sample_view.csv"
CWD_SAMPLE_VIEW_FILENAME = "sample_view.csv"
CWD_SN_FILENAME = "sn.csv"
MISCLASSIFIED_TOP_K_SN = 0
REVIEW_SAMPLE_VIEW_COLUMNS = [
    "line",
    "sn",
    "sample_id",
    "true_mlabel",
    "true_mtype",
    "pred_mlabel",
    "pred_mtype",
]
REVIEW_SAMPLE_VIEW_EXTRA_COLUMNS = [
    "split",
    "pred_confidence",
    "true_label_proba",
    "score_gap",
    "max_label_proba",
    "model_seed",
    "split_seed",
]


def _parse_args():
    parser = argparse.ArgumentParser(description="使用配置文件训练 RF 模型")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"YAML 配置路径（默认: {DEFAULT_CONFIG_PATH}）",
    )
    return parser.parse_args()


class RFModel(BaseModel):
    DEFAULT_MODEL_TYPE = "rf"

    def __init__(self, config_path: str):
        bootstrap = _build_training_bootstrap(
            config_path,
            default_model_type=getattr(self, "DEFAULT_MODEL_TYPE", "rf"),
        )
        self.cfg = bootstrap.cfg
        self.train_cfg = bootstrap.train_cfg
        self.config_file = bootstrap.config_file
        self.line_name = bootstrap.line_name
        self.random_state = int(_to_int(self.train_cfg.get("random_state")) or 42)
        self.split_random_state = int(_to_int(self.train_cfg.get("split_seed")) or self.random_state)
        self.base_random_state = self.random_state

        self.model_id = bootstrap.model_id
        self.results_path = bootstrap.results_path
        self.active_results_path = self.results_path
        self.feature_version = resolve_feature_version(self.cfg)

        self.dataset_files = list(bootstrap.dataset_files)
        self.feature_columns: List[str] = []
        self.available_reasons = list(bootstrap.available_reasons)
        self.split_strategy = resolve_split_strategy_name(self.train_cfg.get("split_strategy"))

        (
            self.labeler,
            self.label_sample_dict,
            self.mlabel_mtype_dict,
            self.label_to_mlabel_map,
        ) = _shared_build_label_runtime(
            train_cfg=self.train_cfg,
            available_reasons=self.available_reasons,
        )
        self.ok_mlabel = 0
        self.mlabel_threshold_dict = build_uniform_threshold_dict(
            sorted(int(k) for k in self.mlabel_mtype_dict.keys())
        )
        print(f"[INFO] 数据集划分策略: {self.split_strategy}")

        self._write_predict_config(self.results_path, seed=self.base_random_state)
        self.model = self._build_rf_classifier()

    def _write_predict_config(self, output_dir: str, *, seed: int | None = None) -> None:
        data = {
            "line_name": self.line_name,
            "model_id": self.model_id,
            "results_path": output_dir,
            "feature_version": self.feature_version,
            "predictor": {
                "ok_mlabel": int(self.ok_mlabel),
                "label_mapping": [
                    {
                        "mlabel": int(k),
                        "mtype": v,
                        "threshold": float(
                            self.mlabel_threshold_dict.get(int(k), 0.7)
                        ),
                    }
                    for k, v in sorted(self.mlabel_mtype_dict.items())
                ]
            },
        }
        if seed is not None:
            data["train_seed"] = int(seed)

        cfg_out = os.path.join(output_dir, "predict_config.yaml")
        with open(cfg_out, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    def _activate_run_dir(self, seed: int, *, multi_seed: bool) -> str:
        if multi_seed:
            run_dir = str(Path(self.results_path) / "seed_runs" / f"seed_{seed}")
        else:
            run_dir = self.results_path
        os.makedirs(run_dir, exist_ok=True)
        self.active_results_path = run_dir
        self._write_predict_config(run_dir, seed=seed)
        return run_dir

    def _resolve_training_seeds(self) -> List[int]:
        seed_list = self.train_cfg.get("seed_list")
        if isinstance(seed_list, list):
            parsed: List[int] = []
            seen: set[int] = set()
            for item in seed_list:
                seed = _to_int(item)
                if seed is None:
                    continue
                seed = int(seed)
                if seed in seen:
                    continue
                seen.add(seed)
                parsed.append(seed)
            if parsed:
                print(f"[INFO] 使用配置 seed_list（共 {len(parsed)} 个）: {parsed}")
                return parsed

        seed_runs = _to_int(self.train_cfg.get("seed_runs"))
        if seed_runs is None or seed_runs <= 0:
            seed_runs = 1

        if seed_runs == 1:
            print(f"[INFO] 使用单个训练种子: {self.base_random_state}")
            return [int(self.base_random_state)]

        rng = np.random.default_rng(self.base_random_state)
        seeds: List[int] = [int(self.base_random_state)]
        seen: set[int] = {int(self.base_random_state)}
        while len(seeds) < seed_runs:
            seed = int(rng.integers(1, 2_147_483_647))
            if seed in seen:
                continue
            seen.add(seed)
            seeds.append(seed)
        print(f"[INFO] 随机生成训练种子（共 {len(seeds)} 个）: {seeds}")
        return seeds

    def _build_rf_classifier(self, *, extra_params: Dict[str, Any] | None = None) -> RandomForestClassifier:
        defaults = {
            "n_estimators": 100,
            "criterion": "log_loss",
            "max_depth": 8,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 0.8,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        rf_params = self.train_cfg.get("rf_params", {})
        if not isinstance(rf_params, dict):
            print("[WARN] train.rf_params 不是字典，已忽略。")
            rf_params = {}
        defaults.update(rf_params)
        configured_weights = self.train_cfg.get("class_weights")
        if isinstance(configured_weights, dict) and "class_weight" not in defaults:
            class_weight = {
                int(mlabel): float(configured_weights.get(str(mtype), 1.0))
                for mlabel, mtype in self.mlabel_mtype_dict.items()
            }
            if any(abs(weight - 1.0) > 1e-9 for weight in class_weight.values()):
                defaults["class_weight"] = class_weight
                print(f"[INFO] 启用 RF class_weight: {class_weight}")
        if extra_params:
            defaults.update(extra_params)
        return RandomForestClassifier(**defaults)

    def _predict_with_deployment_rule(
        self,
        y_proba: np.ndarray,
        model_classes: np.ndarray,
    ) -> np.ndarray:
        return predict_with_threshold_rule(
            y_proba,
            model_classes,
            self.mlabel_threshold_dict,
            ok_mlabel=self.ok_mlabel,
        )

    def _align_proba_columns(
        self,
        y_proba: np.ndarray,
        model_classes: np.ndarray,
        class_labels: np.ndarray,
    ) -> np.ndarray:
        aligned = np.zeros((y_proba.shape[0], len(class_labels)), dtype=float)
        class_to_idx = {int(c): i for i, c in enumerate(class_labels)}
        for src_idx, cls in enumerate(model_classes):
            dst_idx = class_to_idx.get(int(cls))
            if dst_idx is not None:
                aligned[:, dst_idx] = y_proba[:, src_idx]
        return aligned

    @staticmethod
    def _reorder_review_export_columns(df: pd.DataFrame) -> pd.DataFrame:
        preferred = [*REVIEW_SAMPLE_VIEW_COLUMNS, *REVIEW_SAMPLE_VIEW_EXTRA_COLUMNS]
        ordered = [col for col in preferred if col in df.columns]
        ordered.extend([col for col in df.columns if col not in ordered])
        return df.loc[:, ordered].copy()

    @staticmethod
    def _review_sn_filename(sample_view_filename: str) -> str:
        name = Path(sample_view_filename).name
        if name.endswith("_sample_view.csv"):
            return f"{name[:-len('_sample_view.csv')]}_sn.csv"
        if name.lower().endswith(".csv"):
            return f"{Path(name).stem}_sn.csv"
        return f"{name}_sn.csv"

    @staticmethod
    def _build_review_sn_df(df: pd.DataFrame) -> pd.DataFrame:
        if "sn" not in df.columns:
            return pd.DataFrame(columns=["line", "sn"])
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        line_series = df["line"].map(_to_text) if "line" in df.columns else None
        sn_series = df["sn"].map(_to_text)
        for idx, sn in sn_series.items():
            if not sn or sn in seen:
                continue
            seen.add(sn)
            rows.append({"line": line_series.loc[idx] if line_series is not None else "", "sn": sn})
        return pd.DataFrame(rows, columns=["line", "sn"])

    @classmethod
    def _write_review_exports(
        cls,
        *,
        export_df: pd.DataFrame,
        sample_view_path: str | Path,
    ) -> tuple[pd.DataFrame, str]:
        out_path = Path(sample_view_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_df = cls._reorder_review_export_columns(export_df).reset_index(drop=True)
        export_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        sn_path = out_path.with_name(cls._review_sn_filename(out_path.name))
        cls._build_review_sn_df(export_df).to_csv(sn_path, index=False, encoding="utf-8-sig")
        return export_df, str(sn_path)

    def _build_misclassified_export_df(
        self,
        *,
        split_name: str,
        split_df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        model_classes: np.ndarray,
        model_seed: int | None = None,
        split_seed: int | None = None,
    ) -> pd.DataFrame:
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        model_classes = np.asarray(model_classes, dtype=int)
        if len(split_df) != len(y_true) or len(y_true) != len(y_pred):
            raise ValueError(
                f"{split_name} 数据长度不一致: df={len(split_df)}, y_true={len(y_true)}, y_pred={len(y_pred)}"
            )

        wrong_mask = y_true != y_pred
        metadata_cols = [col for col in SAMPLE_VIEW_EXPORT_COLUMNS if col in split_df.columns]
        extra_cols = []
        for col in ("label", "sample_view_file", "sample_view_row_index"):
            if col in split_df.columns and col not in metadata_cols and col not in extra_cols:
                extra_cols.append(col)
        export_df = split_df.loc[wrong_mask, metadata_cols + extra_cols].copy()
        if export_df.empty:
            return export_df

        class_labels = np.array(sorted(int(k) for k in self.mlabel_mtype_dict.keys()), dtype=int)
        aligned_proba = self._align_proba_columns(y_proba, model_classes, class_labels)
        class_to_idx = {int(cls): idx for idx, cls in enumerate(class_labels)}
        row_indices = np.flatnonzero(wrong_mask)
        pred_confidence = []
        true_label_proba = []
        max_label_proba = []
        score_gap = []
        for row_idx, pred_label, true_label in zip(row_indices, y_pred[wrong_mask], y_true[wrong_mask]):
            pred_idx = class_to_idx.get(int(pred_label))
            true_idx = class_to_idx.get(int(true_label))
            row_proba = aligned_proba[row_idx]
            pred_prob = float(row_proba[pred_idx]) if pred_idx is not None else np.nan
            true_prob = float(row_proba[true_idx]) if true_idx is not None else np.nan
            pred_confidence.append(pred_prob)
            true_label_proba.append(true_prob)
            max_label_proba.append(float(np.max(row_proba)) if row_proba.size > 0 else np.nan)
            score_gap.append(np.nan if np.isnan(pred_prob) or np.isnan(true_prob) else float(pred_prob - true_prob))

        export_df["split"] = str(split_name)
        export_df["true_mlabel"] = y_true[wrong_mask].astype(int)
        export_df["true_mtype"] = [self.mlabel_mtype_dict.get(int(v), str(int(v))) for v in y_true[wrong_mask]]
        export_df["pred_mlabel"] = y_pred[wrong_mask].astype(int)
        export_df["pred_mtype"] = [self.mlabel_mtype_dict.get(int(v), str(int(v))) for v in y_pred[wrong_mask]]
        export_df["pred_confidence"] = pred_confidence
        export_df["true_label_proba"] = true_label_proba
        export_df["score_gap"] = score_gap
        export_df["max_label_proba"] = max_label_proba
        if model_seed is not None:
            export_df["model_seed"] = int(model_seed)
        if split_seed is not None:
            export_df["split_seed"] = int(split_seed)
        return export_df.sort_values(
            by=["score_gap", "pred_confidence", "true_label_proba"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)

    def _save_validation_test_misclassified_sample_view(
        self,
        *,
        val_df: pd.DataFrame,
        y_val: np.ndarray,
        y_pred_val: np.ndarray,
        y_proba_val: np.ndarray,
        test_df: pd.DataFrame,
        y_test: np.ndarray,
        y_pred_test: np.ndarray,
        y_proba_test: np.ndarray,
        model_classes: np.ndarray,
        model_seed: int | None = None,
        split_seed: int | None = None,
    ) -> str:
        frames: List[pd.DataFrame] = []
        for split_name, split_df, y_true, y_pred, y_proba in (
            ("validation", val_df, y_val, y_pred_val, y_proba_val),
            ("test", test_df, y_test, y_pred_test, y_proba_test),
        ):
            frame = self._build_misclassified_export_df(
                split_name=split_name,
                split_df=split_df,
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                model_classes=model_classes,
                model_seed=model_seed,
                split_seed=split_seed,
            )
            if not frame.empty:
                frames.append(frame)
        export_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=[*REVIEW_SAMPLE_VIEW_COLUMNS, *REVIEW_SAMPLE_VIEW_EXTRA_COLUMNS]
        )
        if not export_df.empty:
            export_df = export_df.sort_values(
                by=["score_gap", "pred_confidence", "true_label_proba"],
                ascending=[False, False, True],
                na_position="last",
            ).reset_index(drop=True)
        out_path = Path(self._output_root()) / "eval" / MISCLASSIFIED_EXPORT_FILENAME
        export_df, sn_path = self._write_review_exports(export_df=export_df, sample_view_path=out_path)
        print(
            "[INFO] validation/test 错分样本 sample_view 已保存："
            f"{out_path} | rows={len(export_df)} | unique_sn={len(self._build_review_sn_df(export_df))} | sn_csv={sn_path}"
        )
        return str(out_path)

    def _copy_sample_view_to_cwd(self, *, source_path: str | None) -> str | None:
        src_text = _to_text(source_path)
        if not src_text:
            return None
        src = Path(src_text).expanduser()
        if not src.exists():
            print(f"[WARN] sample_view 源文件不存在，跳过复制到当前目录：{src}")
            return None
        dst = Path.cwd() / CWD_SAMPLE_VIEW_FILENAME
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        src_sn = src.with_name(self._review_sn_filename(src.name))
        dst_sn = Path.cwd() / CWD_SN_FILENAME
        if src_sn.exists() and src_sn.resolve() != dst_sn.resolve():
            shutil.copy2(src_sn, dst_sn)
        print(f"[INFO] 最后一次训练错分 sample_view 已复制到当前工作目录：{dst}")
        return str(dst)

    def _compute_split_roc(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_classes: np.ndarray,
        class_labels: np.ndarray,
    ) -> Dict[str, Any] | None:
        if y_true.size == 0 or y_proba.size == 0:
            return None

        aligned = self._align_proba_columns(y_proba, model_classes, class_labels)
        unique_labels = np.unique(y_true)
        if unique_labels.size < 2:
            return None

        class_curves: List[Dict[str, Any]] = []
        for idx, cls in enumerate(class_labels):
            y_bin_cls = (y_true == int(cls)).astype(int)
            if np.unique(y_bin_cls).size < 2:
                continue
            fpr_cls, tpr_cls, _ = roc_curve(y_bin_cls, aligned[:, idx])
            class_curves.append(
                {
                    "class_id": int(cls),
                    "class_name": self.mlabel_mtype_dict.get(int(cls), str(int(cls))),
                    "fpr": fpr_cls,
                    "tpr": tpr_cls,
                    "auc": float(auc(fpr_cls, tpr_cls)),
                }
            )

        if not class_curves:
            return None

        if len(class_labels) == 2:
            pos_label = int(class_labels[1])
            y_bin = (y_true == pos_label).astype(int)
            scores = aligned[:, 1]
            if np.unique(y_bin).size < 2:
                first_curve = class_curves[0]
                fpr, tpr = first_curve["fpr"], first_curve["tpr"]
                roc_auc = first_curve["auc"]
            else:
                fpr, tpr, _ = roc_curve(y_bin, scores)
                roc_auc = float(auc(fpr, tpr))
        else:
            y_bin = label_binarize(y_true, classes=class_labels)
            if y_bin.ndim != 2 or y_bin.shape[1] != len(class_labels):
                first_curve = class_curves[0]
                fpr, tpr = first_curve["fpr"], first_curve["tpr"]
                roc_auc = first_curve["auc"]
            elif np.unique(y_bin).size < 2:
                first_curve = class_curves[0]
                fpr, tpr = first_curve["fpr"], first_curve["tpr"]
                roc_auc = first_curve["auc"]
            else:
                fpr, tpr, _ = roc_curve(y_bin.ravel(), aligned.ravel())
                roc_auc = float(auc(fpr, tpr))

        return {
            "fpr": fpr,
            "tpr": tpr,
            "auc": float(roc_auc),
            "curves": class_curves,
        }

    def _aggregate_confusion_matrix(self, cm_list: List[np.ndarray]) -> np.ndarray:
        if not cm_list:
            return np.zeros((0, 0), dtype=int)
        cm_stack = np.stack(cm_list, axis=0)
        return np.sum(cm_stack, axis=0).astype(int)

    def _create_seed_overview_figure(
        self,
        *,
        split_cms: Dict[str, List[np.ndarray]],
        split_rocs: Dict[str, List[Dict[str, Any]]],
        figure_title: str,
        plot_all_roc_curves: bool = False,
    ) -> plt.Figure:
        class_labels = np.array(sorted(int(k) for k in self.mlabel_mtype_dict.keys()), dtype=int)
        display_labels = [self.mlabel_mtype_dict[int(v)] for v in class_labels]
        split_order = ["train", "validation", "test"]
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))

        for col, split in enumerate(split_order):
            cm_ax = axes[0, col]
            roc_ax = axes[1, col]

            cm_list = split_cms.get(split, [])
            if cm_list:
                cm_sum = self._aggregate_confusion_matrix(cm_list)
                vmax = int(cm_sum.max()) if cm_sum.size > 0 else 1
                im = cm_ax.imshow(
                    cm_sum,
                    interpolation="nearest",
                    cmap=plt.cm.Blues,
                    vmin=0,
                    vmax=max(vmax, 1),
                )
                cm_ax.figure.colorbar(im, ax=cm_ax, fraction=0.046, pad=0.04)
                if len(cm_list) == 1:
                    cm_ax.set_title(f"{split}.png (count)")
                else:
                    cm_ax.set_title(f"{split}.png (count sum, seeds={len(cm_list)})")
                cm_ax.set_xticks(np.arange(len(display_labels)))
                cm_ax.set_yticks(np.arange(len(display_labels)))
                cm_ax.set_xticklabels(display_labels, rotation=45, ha="right")
                cm_ax.set_yticklabels(display_labels)
                cm_ax.set_xlabel("Predicted")
                cm_ax.set_ylabel("True")

                threshold = float(cm_sum.max()) * 0.5 if cm_sum.size > 0 else 0.0
                for i in range(cm_sum.shape[0]):
                    for j in range(cm_sum.shape[1]):
                        cm_ax.text(
                            j,
                            i,
                            f"{int(cm_sum[i, j])}",
                            ha="center",
                            va="center",
                            color="white" if cm_sum[i, j] > threshold else "black",
                            fontsize=8,
                        )
            else:
                cm_ax.axis("off")
                cm_ax.set_title(f"{split}.png (no data)")

            roc_list = split_rocs.get(split, [])
            if roc_list:
                if (not plot_all_roc_curves) or len(roc_list) == 1:
                    item = roc_list[0]
                    curves = item.get("curves", [])
                    if curves:
                        cmap = plt.cm.get_cmap("tab10", len(curves))
                        for c_idx, curve in enumerate(curves):
                            label_name = curve.get("class_name", str(curve.get("class_id", c_idx)))
                            roc_ax.plot(
                                curve["fpr"],
                                curve["tpr"],
                                color=cmap(c_idx),
                                linewidth=1.8,
                                alpha=0.95,
                                label=f"{label_name} AUC={curve['auc']:.3f}",
                            )
                    else:
                        roc_ax.plot(
                            item["fpr"],
                            item["tpr"],
                            color="tab:red",
                            linewidth=2.5,
                            label=f"AUC={item['auc']:.3f}",
                        )
                else:
                    cmap = plt.cm.get_cmap("tab10", len(roc_list))
                    for idx, item in enumerate(roc_list):
                        if "fpr" not in item or "tpr" not in item:
                            continue
                        roc_ax.plot(
                            item["fpr"],
                            item["tpr"],
                            color=cmap(idx),
                            linewidth=1.8,
                            alpha=0.9,
                            label=f"seed#{idx + 1} AUC={item['auc']:.3f}",
                        )

                roc_ax.plot([0, 1], [0, 1], "k--", linewidth=1)
                roc_ax.set_xlim(0.0, 1.0)
                roc_ax.set_ylim(0.0, 1.05)
                if (not plot_all_roc_curves) or len(roc_list) == 1:
                    first_curves = roc_list[0].get("curves", []) if roc_list else []
                    if len(first_curves) > 1:
                        roc_ax.set_title(f"{split}_ROC.png (per-class)")
                    else:
                        roc_ax.set_title(f"{split}_ROC.png")
                else:
                    roc_ax.set_title(f"{split}_ROC.png (all curves)")
                roc_ax.set_xlabel("False Positive Rate")
                roc_ax.set_ylabel("True Positive Rate")
                roc_ax.grid(True, alpha=0.3)
                legend_items = len(roc_list[0].get("curves", [])) if ((not plot_all_roc_curves) and roc_list) else len(roc_list)
                if legend_items > 4:
                    roc_ax.legend(loc="lower right", fontsize=8, ncol=2)
                else:
                    roc_ax.legend(loc="lower right", fontsize=9)
            else:
                roc_ax.axis("off")
                roc_ax.set_title(f"{split}_ROC.png (no data)")

        fig.suptitle(figure_title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    def _plot_seed_overview(
        self,
        *,
        split_cms: Dict[str, List[np.ndarray]],
        split_rocs: Dict[str, List[Dict[str, Any]]],
        output_dir: str,
        filename_prefix: str = "summary_confusion_roc",
        figure_title: str = "Multi-seed Summary: train / validation / test",
        plot_all_roc_curves: bool = False,
    ) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        fig = self._create_seed_overview_figure(
            split_cms=split_cms,
            split_rocs=split_rocs,
            figure_title=figure_title,
            plot_all_roc_curves=plot_all_roc_curves,
        )
        output_paths = self._save_figure_multi_formats(fig, os.path.join(output_dir, filename_prefix), png_dpi=360)
        plt.close(fig)
        return output_paths

    def _export_best_seed_png_results(self, *, best_run_dir: str, best_seed: int) -> List[str]:
        src_eval_dir = Path(best_run_dir) / "eval"
        if not src_eval_dir.exists():
            print(f"[WARN] 最佳种子 eval 目录不存在，跳过 PNG 导出：{src_eval_dir}")
            return []

        dst_root = Path(self.results_path) / "best_model_eval" / f"seed_{best_seed}"
        if dst_root.exists():
            shutil.rmtree(dst_root)
        dst_root.mkdir(parents=True, exist_ok=True)

        copied_paths: List[str] = []
        for src in sorted(src_eval_dir.rglob("*.png")):
            rel = src.relative_to(src_eval_dir)
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied_paths.append(str(dst))

        if not copied_paths:
            print(f"[WARN] 最佳种子未找到 PNG 文件：{src_eval_dir}")
            return []

        print(f"[INFO] 最佳模型 PNG 已单独保存：{dst_root}（共 {len(copied_paths)} 个）")
        return copied_paths

    def _export_final_report_pdf(
        self,
        *,
        seed_panels: List[Dict[str, Any]],
        summary_split_cms: Dict[str, List[np.ndarray]],
        summary_split_rocs: Dict[str, List[Dict[str, Any]]],
        output_pdf_path: str,
    ) -> int:
        os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
        page_count = 0
        with PdfPages(output_pdf_path) as pdf:
            for panel in seed_panels:
                seed = int(panel["seed"])
                fig = self._create_seed_overview_figure(
                    split_cms=panel["split_cms"],
                    split_rocs=panel["split_rocs"],
                    figure_title=f"Seed {seed} Summary (2x3)",
                    plot_all_roc_curves=False,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                page_count += 1

            fig_summary = self._create_seed_overview_figure(
                split_cms=summary_split_cms,
                split_rocs=summary_split_rocs,
                figure_title="Multi-seed Summary (2x3)",
                plot_all_roc_curves=True,
            )
            pdf.savefig(fig_summary, bbox_inches="tight")
            plt.close(fig_summary)
            page_count += 1

        return page_count

    def _plot_feature_importance(self, df_for_names: pd.DataFrame, top_k=30, title="feature_importance"):
        feature_names = list(self.feature_columns) if self.feature_columns else self._resolve_feature_columns(df_for_names)
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            print("⚠️ 当前 RF 模型不支持 feature_importances_，跳过特征重要性输出。")
            return

        imp = (
            pd.Series(importances, index=feature_names, name="importance")
            .sort_values(ascending=False)
            .to_frame()
        )

        out_dir = os.path.join(self._output_root(), "eval/importance")
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, f"{title}.csv")
        imp.to_csv(csv_path, index=True, encoding="utf-8-sig")
        print(f"✔️ 已保存特征重要性CSV：{csv_path}")

        top = imp.head(top_k)
        plt.figure(figsize=(8, max(4, len(top) * 0.35)))
        plt.barh(top.index[::-1], top["importance"].values[::-1])
        plt.xlabel("Importance")
        plt.title(title)
        plt.tight_layout()

        base_path = os.path.join(out_dir, title)
        paths = self._save_figure_multi_formats(plt.gcf(), base_path, png_dpi=360)
        plt.close()
        for path in paths:
            print(f"✔️ 已保存特征重要性图：{path}")

    def train_and_evaluate(self, split_data: TrainValTestSplit):
        seeds = self._resolve_training_seeds()
        multi_seed = len(seeds) > 1
        run_records: List[Dict[str, Any]] = []
        seed_panels: List[Dict[str, Any]] = []
        class_labels = np.array(sorted(int(k) for k in self.mlabel_mtype_dict.keys()), dtype=int)
        split_cms: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}
        split_rocs: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
        latest_misclassified_sample_view_path: str | None = None

        for idx, seed in enumerate(seeds, start=1):
            print(f"\n===== Seed Run {idx}/{len(seeds)} | seed={seed} =====")
            self.random_state = int(seed)
            self.model = self._build_rf_classifier(extra_params={"random_state": self.random_state})
            run_dir = self._activate_run_dir(seed=seed, multi_seed=multi_seed)
            train = split_data.train_df.copy()
            val = split_data.val_df.copy()
            test = split_data.test_df.copy()

            X_train = self._build_feature_matrix(train, fit=True)
            y_train = train["label"].values

            X_val = self._build_feature_matrix(val, fit=False)
            y_val = val["label"].values

            X_test = self._build_feature_matrix(test, fit=False)
            y_test = test["label"].values

            print("数据分割：", X_train.shape, X_val.shape, X_test.shape)
            self.model.fit(X_train, y_train)
            self._save_model()
            self._print_model_info()

            y_proba_train = self.model.predict_proba(X_train)
            y_proba_val = self.model.predict_proba(X_val)
            y_proba_test = self.model.predict_proba(X_test)
            model_classes = np.array(self.model.classes_, dtype=int)
            y_pred_train = self._predict_with_deployment_rule(y_proba_train, model_classes)
            y_pred_val = self._predict_with_deployment_rule(y_proba_val, model_classes)
            y_pred_test = self._predict_with_deployment_rule(y_proba_test, model_classes)
            latest_misclassified_sample_view_path = self._save_validation_test_misclassified_sample_view(
                val_df=val,
                y_val=y_val,
                y_pred_val=y_pred_val,
                y_proba_val=y_proba_val,
                test_df=test,
                y_test=y_test,
                y_pred_test=y_pred_test,
                y_proba_test=y_proba_test,
                model_classes=model_classes,
                model_seed=seed,
                split_seed=self.split_random_state,
            )

            train_roc = self._compute_split_roc(y_train, y_proba_train, model_classes, class_labels)
            val_roc = self._compute_split_roc(y_val, y_proba_val, model_classes, class_labels)
            test_roc = self._compute_split_roc(y_test, y_proba_test, model_classes, class_labels)

            split_cms["train"].append(confusion_matrix(y_train, y_pred_train, labels=class_labels).astype(int))
            split_cms["validation"].append(confusion_matrix(y_val, y_pred_val, labels=class_labels).astype(int))
            split_cms["test"].append(confusion_matrix(y_test, y_pred_test, labels=class_labels).astype(int))
            if train_roc is not None:
                split_rocs["train"].append(train_roc)
            if val_roc is not None:
                split_rocs["validation"].append(val_roc)
            if test_roc is not None:
                split_rocs["test"].append(test_roc)

            panel_data = {
                "seed": seed,
                "split_cms": {
                    "train": [split_cms["train"][-1].copy()],
                    "validation": [split_cms["validation"][-1].copy()],
                    "test": [split_cms["test"][-1].copy()],
                },
                "split_rocs": {
                    "train": [train_roc] if train_roc is not None else [],
                    "validation": [val_roc] if val_roc is not None else [],
                    "test": [test_roc] if test_roc is not None else [],
                },
            }
            seed_panels.append(panel_data)

            self._evaluate_predictions(y_train, y_pred_train, "train")
            self._plot_multiclass_roc(y_train, y_proba_train, "train")
            self._evaluate_predictions(y_val, y_pred_val, "validation")
            self._plot_multiclass_roc(y_val, y_proba_val, "validation")
            self._evaluate_predictions(y_test, y_pred_test, "test")
            self._plot_multiclass_roc(y_test, y_proba_test, "test")
            self._plot_feature_importance(train, top_k=30, title="train_feature_importance")

            try:
                print("\n[INFO] 生成训练诊断报告...")
                diag_output_dir = os.path.join(run_dir, "eval", "diagnostics")
                generate_training_diagnostics(
                    X_train=X_train,
                    y_train=y_train,
                    y_pred_train=y_pred_train,
                    y_proba_train=y_proba_train,
                    X_val=X_val,
                    y_val=y_val,
                    y_pred_val=y_pred_val,
                    y_proba_val=y_proba_val,
                    X_test=X_test,
                    y_test=y_test,
                    y_pred_test=y_pred_test,
                    y_proba_test=y_proba_test,
                    model_classes=model_classes,
                    mlabel_mtype_dict=self.mlabel_mtype_dict,
                    output_dir=diag_output_dir,
                    seed=seed,
                    ok_label=self.ok_mlabel,
                )
            except Exception as e:
                print(f"[WARN] 训练诊断报告生成失败: {e}")

            seed_plot_paths = self._plot_seed_overview(
                split_cms=panel_data["split_cms"],
                split_rocs=panel_data["split_rocs"],
                output_dir=os.path.join(run_dir, "eval"),
                filename_prefix="seed_confusion_roc",
                figure_title=f"Seed {seed} Summary: train / validation / test",
                plot_all_roc_curves=False,
            )
            for p in seed_plot_paths:
                print(f"[INFO] Seed 汇总图已保存：{p}")

            train_quality_metrics = compute_quality_gate_metrics(
                y_true=y_train,
                y_pred=y_pred_train,
                ok_label=self.ok_mlabel,
            )
            val_quality_metrics = compute_quality_gate_metrics(
                y_true=y_val,
                y_pred=y_pred_val,
                ok_label=self.ok_mlabel,
            )
            test_quality_metrics = compute_quality_gate_metrics(
                y_true=y_test,
                y_pred=y_pred_test,
                ok_label=self.ok_mlabel,
            )

            run_records.append(
                {
                    "seed": seed,
                    "run_dir": run_dir,
                    "train_acc": float(accuracy_score(y_train, y_pred_train)),
                    "val_acc": float(accuracy_score(y_val, y_pred_val)),
                    "test_acc": float(accuracy_score(y_test, y_pred_test)),
                    "val_f1_macro": float(f1_score(y_val, y_pred_val, average="macro", zero_division=0)),
                    "test_f1_macro": float(f1_score(y_test, y_pred_test, average="macro", zero_division=0)),
                    "train_nok_recall": float(train_quality_metrics["nok_recall"]),
                    "val_nok_recall": float(val_quality_metrics["nok_recall"]),
                    "test_nok_recall": float(test_quality_metrics["nok_recall"]),
                    "train_false_ok_rate": float(train_quality_metrics["false_ok_rate"]),
                    "val_false_ok_rate": float(val_quality_metrics["false_ok_rate"]),
                    "test_false_ok_rate": float(test_quality_metrics["false_ok_rate"]),
                    "train_false_reject_rate": float(train_quality_metrics["false_reject_rate"]),
                    "val_false_reject_rate": float(val_quality_metrics["false_reject_rate"]),
                    "test_false_reject_rate": float(test_quality_metrics["false_reject_rate"]),
                    "train_false_ok_count": int(train_quality_metrics["false_ok_count"]),
                    "val_false_ok_count": int(val_quality_metrics["false_ok_count"]),
                    "test_false_ok_count": int(test_quality_metrics["false_ok_count"]),
                    "train_auc": float(train_roc["auc"]) if train_roc is not None else np.nan,
                    "val_auc": float(val_roc["auc"]) if val_roc is not None else np.nan,
                    "test_auc": float(test_roc["auc"]) if test_roc is not None else np.nan,
                }
            )

        if not run_records:
            raise RuntimeError("未产生任何训练结果。")

        summary_df = pd.DataFrame(run_records).sort_values(
            by=["val_false_ok_rate", "val_nok_recall", "val_f1_macro", "seed"],
            ascending=[True, False, False, True],
            na_position="last",
        )
        summary_dir = Path(self.results_path) / "seed_runs"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] 多随机种子训练汇总已保存：{summary_path}")
        self._copy_sample_view_to_cwd(source_path=latest_misclassified_sample_view_path)

        summary_stats = (
            summary_df.select_dtypes(include=[np.number])
            .agg(["mean", "std"])
            .transpose()
            .reset_index()
            .rename(columns={"index": "metric"})
        )
        stats_path = summary_dir / "summary_stats.csv"
        summary_stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] 均值/标准差统计已保存：{stats_path}")

        plot_paths = self._plot_seed_overview(
            split_cms=split_cms,
            split_rocs=split_rocs,
            output_dir=str(summary_dir),
            plot_all_roc_curves=True,
        )
        for p in plot_paths:
            print(f"[INFO] 汇总图已保存：{p}")

        final_pdf_path = str(summary_dir / "final_seed_report.pdf")
        page_count = self._export_final_report_pdf(
            seed_panels=seed_panels,
            summary_split_cms=split_cms,
            summary_split_rocs=split_rocs,
            output_pdf_path=final_pdf_path,
        )
        print(f"[INFO] 最终PDF已保存：{final_pdf_path}（共 {page_count} 页）")

        mean_row = summary_stats.set_index("metric")
        tracked_metrics = [
            "train_acc",
            "val_acc",
            "test_acc",
            "val_f1_macro",
            "test_f1_macro",
            "val_nok_recall",
            "test_nok_recall",
            "val_false_ok_rate",
            "test_false_ok_rate",
            "val_false_reject_rate",
            "test_false_reject_rate",
            "train_auc",
            "val_auc",
            "test_auc",
        ]
        print("[INFO] 多随机种子平均结果：")
        for metric in tracked_metrics:
            if metric not in mean_row.index:
                continue
            m = float(mean_row.loc[metric, "mean"])
            s = float(mean_row.loc[metric, "std"])
            print(f"  {metric:>12}: {m:.4f} ± {s:.4f}")

        best = summary_df.iloc[0].to_dict()
        best_seed = int(best["seed"])
        best_run_dir = str(best["run_dir"])
        print(
            f"[INFO] 最佳种子: {best_seed} | "
            f"val_false_ok_rate={best['val_false_ok_rate']:.4f} | "
            f"val_nok_recall={best['val_nok_recall']:.4f} | "
            f"val_f1_macro={best['val_f1_macro']:.4f}"
        )

        best_model_path = os.path.join(best_run_dir, "model.pkl")
        root_model_path = os.path.join(self.results_path, "model.pkl")
        if os.path.exists(best_model_path):
            same_model_file = False
            try:
                same_model_file = os.path.samefile(best_model_path, root_model_path)
            except FileNotFoundError:
                same_model_file = False
            if not same_model_file:
                shutil.copy2(best_model_path, root_model_path)
            self.active_results_path = self.results_path
            self._write_predict_config(self.results_path, seed=best_seed)
            self._export_best_seed_png_results(best_run_dir=best_run_dir, best_seed=best_seed)
            launcher = export_runtime_bundle(
                self.results_path,
                feature_version=self.feature_version,
            )
            if same_model_file:
                print(f"[INFO] 最佳模型已在根目录，无需复制：{root_model_path}")
            else:
                print(f"[INFO] 已将最佳模型复制到根目录：{root_model_path}")
            print(f"[INFO] 根目录推理运行包已更新：{launcher}")

    def train_and_cross_validate(self, cv_data: PreparedCrossValidationData):
        self.random_state = self.base_random_state
        self.active_results_path = self.results_path
        self.model = self._build_rf_classifier(extra_params={"random_state": self.random_state})
        self._write_predict_config(self.results_path, seed=self.random_state)

        df = cv_data.df.copy()
        X = self._build_feature_matrix(df, fit=True)
        y = df["label"].values
        requested_cv_splits = int(cv_data.requested_splits)
        cv = cv_data.cv
        cv_groups = cv_data.groups
        resolved_cv_strategy = str(cv_data.resolved_strategy)
        scoring = {
            "accuracy": build_threshold_scorer(
                accuracy_score,
                threshold_by_label=self.mlabel_threshold_dict,
                ok_mlabel=self.ok_mlabel,
            ),
            "precision": build_threshold_scorer(
                precision_score,
                threshold_by_label=self.mlabel_threshold_dict,
                ok_mlabel=self.ok_mlabel,
                average="macro",
                zero_division=0,
            ),
            "recall": build_threshold_scorer(
                recall_score,
                threshold_by_label=self.mlabel_threshold_dict,
                ok_mlabel=self.ok_mlabel,
                average="macro",
                zero_division=0,
            ),
            "f1": build_threshold_scorer(
                f1_score,
                threshold_by_label=self.mlabel_threshold_dict,
                ok_mlabel=self.ok_mlabel,
                average="macro",
                zero_division=0,
            ),
        }
        results = cross_validate(
            self.model, X, y, cv=cv, groups=cv_groups, scoring=scoring,
            return_train_score=False, n_jobs=-1
        )
        print(f"\n—— {requested_cv_splits}-折交叉验证 ({resolved_cv_strategy}) ——")
        for m in scoring:
            s = results[f"test_{m}"]
            print(f"{m.title():>10s}: {s.mean():.3f} ± {s.std():.3f}")
        print("\n—— 重训全量并保存 ——")
        self.model.fit(X, y)
        self._save_model()

    def train_and_evaluate_grid(
        self,
        split_data: TrainValTestSplit,
        grid_cv_data: PreparedCrossValidationData,
    ):
        self.random_state = self.base_random_state
        self.active_results_path = self.results_path
        self.model = self._build_rf_classifier(extra_params={"random_state": self.random_state})
        self._write_predict_config(self.results_path, seed=self.random_state)

        train = split_data.train_df.copy()
        val = split_data.val_df.copy()
        test = split_data.test_df.copy()

        X_train = self._build_feature_matrix(train, fit=True)
        y_train = train["label"].values

        X_val = self._build_feature_matrix(val, fit=False)
        y_val = val["label"].values

        X_test = self._build_feature_matrix(test, fit=False)
        y_test = test["label"].values

        print("数据分割：", X_train.shape, X_val.shape, X_test.shape)

        param_grid = self.train_cfg.get("rf_grid", {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 8, 12],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": [0.6, 0.8, 1.0],
        })
        if not isinstance(param_grid, dict):
            print("[WARN] train.rf_grid 不是字典，已回退默认网格。")
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 8, 12],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": [0.6, 0.8, 1.0],
            }

        requested_cv_splits = int(grid_cv_data.requested_splits)
        grid_cv = grid_cv_data.cv
        grid_groups = grid_cv_data.groups
        resolved_cv_strategy = str(grid_cv_data.resolved_strategy)
        print(f"[INFO] GridSearchCV 划分策略: {resolved_cv_strategy}")

        grid = GridSearchCV(
            self._build_rf_classifier(),
            param_grid=param_grid,
            scoring=build_threshold_scorer(
                accuracy_score,
                threshold_by_label=self.mlabel_threshold_dict,
                ok_mlabel=self.ok_mlabel,
            ),
            cv=grid_cv,
            verbose=1,
            n_jobs=-1
        )
        grid.fit(X_train, y_train, groups=grid_groups)
        print(f"最佳参数: {grid.best_params_}, CV={grid.best_score_:.4f}")
        self.model = grid.best_estimator_
        self._save_model()
        self._print_model_info()
        y_proba_train = self.model.predict_proba(X_train)
        y_proba_val = self.model.predict_proba(X_val)
        y_proba_test = self.model.predict_proba(X_test)
        model_classes = np.array(self.model.classes_, dtype=int)
        y_pred_train = self._predict_with_deployment_rule(y_proba_train, model_classes)
        y_pred_val = self._predict_with_deployment_rule(y_proba_val, model_classes)
        y_pred_test = self._predict_with_deployment_rule(y_proba_test, model_classes)
        misclassified_sample_view_path = self._save_validation_test_misclassified_sample_view(
            val_df=val,
            y_val=y_val,
            y_pred_val=y_pred_val,
            y_proba_val=y_proba_val,
            test_df=test,
            y_test=y_test,
            y_pred_test=y_pred_test,
            y_proba_test=y_proba_test,
            model_classes=model_classes,
            model_seed=self.random_state,
            split_seed=self.split_random_state,
        )
        self._copy_sample_view_to_cwd(source_path=misclassified_sample_view_path)
        self._evaluate_predictions(
            y_train,
            y_pred_train,
            "train",
        )
        self._plot_multiclass_roc(y_train, y_proba_train, "train")
        self._evaluate_predictions(
            y_val,
            y_pred_val,
            "validation",
        )
        self._plot_multiclass_roc(y_val, y_proba_val, "validation")
        self._evaluate_predictions(
            y_test,
            y_pred_test,
            "test",
        )
        self._plot_multiclass_roc(y_test, y_proba_test, "test")
        self._plot_feature_importance(train, top_k=30, title="train_feature_importance")


if __name__ == "__main__":
    args = _parse_args()
    model = RFModel(args.config)
    train_mode = _to_text(model.train_cfg.get("train_mode")).lower()
    if not train_mode:
        raise KeyError(f"配置缺少 train.train_mode: {args.config}")
    try:
        from model_registry import run_training
    except ModuleNotFoundError:  # pragma: no cover
        from ml.train import run_training
    run_training(model, train_mode)
