"""BaseModel：所有 ML 分类模型的公共基类。

包含：
  ModelInterface  — Protocol（训练接口契约）
  BaseModel       — 公共基类，内联评估/绘图/导出/阈值调优全部逻辑

子类只需实现以下钩子方法：
    _build_classifier(*, extra_params=None) → sklearn 兼容分类器
    _do_model_fit(X_train, y_train, X_val, y_val, *, sample_weight, feature_weights)
    _do_full_data_fit(X, y, *, sample_weight)          # 交叉验证重训全量
    _compute_feature_weights() → np.ndarray | None     # 默认返回 None
    _plot_training_loss(*, title) → List[str]          # 默认空实现
    _importance_df(importance_type, feature_names)     # 默认空 DataFrame
    _plot_shap_summary(X, df_for_names, title)         # 默认空实现
"""

from __future__ import annotations

import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import yaml

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import label_binarize

from ml.runtime import export_runtime_bundle


def _configure_plot_fonts() -> None:
    """Prefer an installed CJK font so Windows evaluation plots keep labels."""
    candidates = (
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
        "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans",
    )
    try:
        available = {font.name for font in font_manager.fontManager.ttflist}
    except Exception:
        available = set()
    selected = next((name for name in candidates if name in available), None)
    if selected:
        plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False


_configure_plot_fonts()
from ml.runtime import (
    build_threshold_scorer,
    build_uniform_threshold_dict,
    predict_with_threshold_rule,
)
from ml.training.split import (
    PreparedCrossValidationData,
    TrainValTestSplit,
    resolve_split_strategy_name,
)
from ml.training.config import (
    DEFAULT_CONFIG_PATH,
    NON_FEATURE_COLUMNS,
    SAMPLE_VIEW_EXPORT_COLUMNS,
    FeatureSchemaError,
    _looks_like_numeric_feature,
    _to_float,
    _to_int,
    _to_text,
    build_label_runtime as _shared_build_label_runtime,
    build_training_bootstrap as _build_training_bootstrap,
    resolve_configured_feature_columns,
    resolve_feature_columns_by_schema,
)
from ml.training.diagnostics import compute_quality_gate_metrics, generate_training_diagnostics
from ml.training.audit_labels import run as run_label_audit
from ml.features import resolve_feature_version


# ---------------------------------------------------------------------------
# 导出文件名常量
# ---------------------------------------------------------------------------

SPECIAL_REVIEW_REASON_NAMES = ("马达", "震颤")
SPECIAL_REVIEW_EXPORT_FILENAME = "validation_test_motor_zhanchan_mismatch_sample_view.csv"
MISCLASSIFIED_EXPORT_FILENAME = "validation_test_misclassified_sample_view.csv"
CWD_SAMPLE_VIEW_FILENAME = "sample_view.csv"
CWD_SN_FILENAME = "sn.csv"
REVIEW_TOP_K_SN = 500
MISCLASSIFIED_TOP_K_SN = 0

REVIEW_SAMPLE_VIEW_COLUMNS = [
    "line", "sn", "sample_id",
    "true_mlabel", "true_mtype",
    "pred_mlabel", "pred_mtype",
]

REVIEW_SAMPLE_VIEW_EXTRA_COLUMNS = [
    "split", "pred_confidence", "true_label_proba",
    "score_gap", "max_label_proba", "model_seed", "split_seed",
]


# ---------------------------------------------------------------------------
# 训练接口契约
# ---------------------------------------------------------------------------

@runtime_checkable
class ModelInterface(Protocol):
    """Interface consumed by the training orchestrator."""

    train_cfg: dict
    results_path: str

    def train_and_evaluate(self, split_data: Any) -> Any: ...

    def train_and_cross_validate(self, cv_data: Any) -> Any: ...

    def train_and_evaluate_grid(self, split_data: Any, grid_cv_data: Any) -> Any: ...


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class BaseModel:
    """所有 ML 分类模型的公共基类。"""

    DEFAULT_MODEL_TYPE: str = "base"

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #

    def __init__(self, config_path: str):
        bootstrap = _build_training_bootstrap(
            config_path,
            default_model_type=getattr(self, "DEFAULT_MODEL_TYPE", "base"),
        )
        self.cfg = bootstrap.cfg
        self.train_cfg = bootstrap.train_cfg
        self.config_file = bootstrap.config_file
        self.line_name = bootstrap.line_name

        shared_random_state = int(_to_int(self.train_cfg.get("random_state")) or 42)
        self.default_random_state = shared_random_state
        self.split_random_state = int(_to_int(self.train_cfg.get("split_seed")) or shared_random_state)
        self.random_state = int(_to_int(self.train_cfg.get("model_seed")) or shared_random_state)
        self.base_model_random_state = self.random_state
        self.base_random_state = self.base_model_random_state

        self.model_id = bootstrap.model_id
        self.results_path = bootstrap.results_path
        self.active_results_path = self.results_path
        self.feature_version = resolve_feature_version(self.cfg)

        self.feature_columns: List[str] = []
        self.dataset_files = list(bootstrap.dataset_files)
        self.available_reasons = list(bootstrap.available_reasons)

        self.reason_name_by_label: Dict[int, str] = {}
        for item in self.available_reasons:
            reason_id = _to_int(item.get("reason_id"))
            if reason_id is None:
                continue
            self.reason_name_by_label[int(reason_id)] = (
                _to_text(item.get("reason_name")) or str(int(reason_id))
            )

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
        self.default_mlabel_threshold_dict = build_uniform_threshold_dict(
            sorted(int(k) for k in self.mlabel_mtype_dict.keys())
        )
        self.mlabel_threshold_dict = dict(self.default_mlabel_threshold_dict)

        print(
            "[INFO] 训练随机种子配置: "
            f"split_seed={self.split_random_state}, "
            f"base_model_seed={self.base_model_random_state}"
        )
        print(f"[INFO] 数据集划分策略: {self.split_strategy}")

        self._write_predict_config(
            self.results_path,
            model_seed=self.base_model_random_state,
            split_seed=self.split_random_state,
            threshold_source="default",
        )

        self.n_classes = len(self.mlabel_mtype_dict)
        if self.n_classes < 2:
            raise RuntimeError(f"训练类别数不足（当前={self.n_classes}），至少需要 2 个类别。")

        self.objective = "binary:logistic" if self.n_classes == 2 else "multi:softprob"
        self.eval_metric = "logloss" if self.n_classes == 2 else "mlogloss"
        self.model = self._build_classifier()

    # ------------------------------------------------------------------ #
    # 子类必须实现的钩子方法
    # ------------------------------------------------------------------ #

    def _build_classifier(self, *, extra_params: Dict[str, Any] | None = None):
        raise NotImplementedError(f"{type(self).__name__} 必须实现 _build_classifier()")

    def _do_model_fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        feature_weights: np.ndarray | None,
    ) -> None:
        raise NotImplementedError(f"{type(self).__name__} 必须实现 _do_model_fit()")

    def _do_full_data_fit(self, X: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray | None) -> None:
        self.model.fit(X, y, sample_weight=sample_weight)

    def _compute_feature_weights(self) -> np.ndarray | None:
        return None

    def _plot_training_loss(self, *, title: str = "training_loss") -> List[str]:
        return []

    def _importance_df(self, importance_type: str, feature_names: list) -> pd.DataFrame:
        return pd.DataFrame(columns=["importance"])

    def _plot_shap_summary(self, X: np.ndarray, df_for_names: pd.DataFrame, title: str = "feature_importance_shap") -> None:
        pass

    def _get_param_grid(self) -> Dict[str, Any]:
        default_grid: Dict[str, Any] = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
        }
        return self.train_cfg.get("model_grid", default_grid)

    # ------------------------------------------------------------------ #
    # 路径 / 序列化
    # ------------------------------------------------------------------ #

    def _output_root(self) -> str:
        return getattr(self, "active_results_path", self.results_path)

    def _save_model(self) -> None:
        path = os.path.join(self._output_root(), "model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"✅ 模型已保存：{path}")
        launcher = export_runtime_bundle(
            Path(self.results_path).parent,
            self.model_id,
            source_model_dir=self._output_root(),
            feature_version=self.feature_version,
        )
        print(f"✅ 推理运行包已导出：{launcher}")

    def _print_model_info(self) -> None:
        print("\n===== 模型信息 =====")
        print(self.model)
        print("====================")

    # ------------------------------------------------------------------ #
    # 特征矩阵
    # ------------------------------------------------------------------ #

    def _resolve_feature_columns(self, df: pd.DataFrame) -> List[str]:
        feature_cols = resolve_feature_columns_by_schema(df)
        return resolve_configured_feature_columns(
            feature_cols,
            train_cfg=self.train_cfg,
            config_file=self.config_file,
        )

    def _build_feature_matrix(self, df: pd.DataFrame, *, fit: bool) -> np.ndarray:
        if fit or not self.feature_columns:
            self.feature_columns = self._resolve_feature_columns(df)
            print(f"[INFO] 使用特征列数: {len(self.feature_columns)}")
        X_df = df.reindex(columns=self.feature_columns, fill_value=0.0)
        X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return X_df.to_numpy(dtype=np.float32, copy=False)

    # ------------------------------------------------------------------ #
    # 样本权重
    # ------------------------------------------------------------------ #

    _DEFAULT_CLASS_WEIGHTS: Dict[str, float] = {
        "震颤_马达": 1.0,
        "秒表": 1.0,
    }

    def _compute_sample_weight(self, y_train: np.ndarray) -> np.ndarray | None:
        cfg_weights = self.train_cfg.get("class_weights")
        if isinstance(cfg_weights, dict) and cfg_weights:
            name2weight = {str(k): float(v) for k, v in cfg_weights.items()}
        else:
            name2weight = dict(self._DEFAULT_CLASS_WEIGHTS)

        if not name2weight:
            return None

        mlabel2weight: Dict[int, float] = {}
        for mlabel, mtype in self.mlabel_mtype_dict.items():
            mlabel2weight[int(mlabel)] = float(name2weight.get(str(mtype), 1.0))

        if all(abs(w - 1.0) < 1e-9 for w in mlabel2weight.values()):
            return None

        sample_weight = np.fromiter(
            (mlabel2weight.get(int(y), 1.0) for y in y_train),
            dtype=np.float64, count=len(y_train),
        )
        applied = {self.mlabel_mtype_dict[m]: w for m, w in mlabel2weight.items() if abs(w - 1.0) > 1e-9}
        unique, counts = np.unique(y_train, return_counts=True)
        per_class_n = {self.mlabel_mtype_dict.get(int(u), str(int(u))): int(c) for u, c in zip(unique, counts)}
        print(f"[INFO] 启用 class_weight: {applied}")
        print(f"       训练集各类样本数: {per_class_n}")
        return sample_weight

    # ------------------------------------------------------------------ #
    # 训练种子与运行目录
    # ------------------------------------------------------------------ #

    def _resolve_training_seeds(self) -> List[int]:
        seed_list = self.train_cfg.get("model_seed_list") or self.train_cfg.get("seed_list")
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
                print(f"[INFO] 使用配置 model_seed_list（共 {len(parsed)} 个）: {parsed}")
                return parsed

        seed_runs = int(_to_int(self.train_cfg.get("seed_runs")) or 1)
        if seed_runs <= 0:
            seed_runs = 1

        if seed_runs == 1:
            print(f"[INFO] 使用单个 model_seed: {self.base_model_random_state}")
            return [int(self.base_model_random_state)]

        rng = np.random.default_rng(self.base_model_random_state)
        seeds: List[int] = [int(self.base_model_random_state)]
        seen_set: set[int] = {int(self.base_model_random_state)}
        while len(seeds) < seed_runs:
            s = int(rng.integers(1, 2_147_483_647))
            if s in seen_set:
                continue
            seen_set.add(s)
            seeds.append(s)

        print(f"[INFO] 训练 model_seed（共 {len(seeds)} 个）: {seeds}")
        return seeds

    def _activate_run_dir(self, model_seed: int, *, multi_seed: bool) -> str:
        run_dir = (
            str(Path(self.results_path) / "seed_runs" / f"seed_{model_seed}")
            if multi_seed
            else self.results_path
        )
        os.makedirs(run_dir, exist_ok=True)
        self.active_results_path = run_dir
        self._write_predict_config(
            run_dir, model_seed=model_seed,
            split_seed=self.split_random_state, threshold_source="default",
        )
        return run_dir

    # ================================================================== #
    # 评估与绘图
    # ================================================================== #

    def _save_figure_multi_formats(self, fig: plt.Figure, base_path: str, *, png_dpi: int = 300) -> List[str]:
        output_paths: List[str] = []
        for ext in ("png", "pdf", "svg"):
            path = f"{base_path}.{ext}"
            save_kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
            if ext == "png":
                save_kwargs["dpi"] = png_dpi
            fig.savefig(path, **save_kwargs)
            output_paths.append(path)
        return output_paths

    def _evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
        labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(
            cm,
            display_labels=[self.mlabel_mtype_dict[int(v)] for v in labels],
        )
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(title)

        dir_path = os.path.join(self._output_root(), "eval")
        os.makedirs(dir_path, exist_ok=True)
        base_path = os.path.join(dir_path, title)
        saved_paths = self._save_figure_multi_formats(plt.gcf(), base_path, png_dpi=360)
        plt.close()

        print(f"—— {title} ——")
        print(classification_report(y_true, y_pred, zero_division=0))
        for path in saved_paths:
            print(f"✔️ 已保存 {title}：{path}")

    def _plot_multiclass_roc(self, y_true: np.ndarray, y_scores: np.ndarray, title: str) -> None:
        classes = np.unique(y_true)
        n = len(classes)
        plt.figure(figsize=(8, 6))

        if n == 2:
            scores = y_scores[:, 1] if y_scores.ndim > 1 else y_scores
            fpr, tpr, _ = roc_curve(y_true, scores, pos_label=classes[1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{self.mlabel_mtype_dict[classes[1]]} (AUC={roc_auc:.2f})")
        else:
            y_bin = label_binarize(y_true, classes=classes)
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{self.mlabel_mtype_dict[cls]} (AUC={roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{title} ROC")
        plt.legend(loc="lower right")
        plt.grid(True)

        dir_path = os.path.join(self._output_root(), "eval")
        os.makedirs(dir_path, exist_ok=True)
        base_path = os.path.join(dir_path, f"{title}_ROC")
        saved_paths = self._save_figure_multi_formats(plt.gcf(), base_path, png_dpi=360)
        plt.close()
        for path in saved_paths:
            print(f"✔️ 已保存 {title} ROC：{path}")

    def _align_proba_columns(
        self, y_proba: np.ndarray, model_classes: np.ndarray, class_labels: np.ndarray,
    ) -> np.ndarray:
        aligned = np.zeros((y_proba.shape[0], len(class_labels)), dtype=float)
        class_to_idx = {int(c): i for i, c in enumerate(class_labels)}
        for src_idx, cls in enumerate(model_classes):
            dst_idx = class_to_idx.get(int(cls))
            if dst_idx is not None:
                aligned[:, dst_idx] = y_proba[:, src_idx]
        return aligned

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
        if np.unique(y_true).size < 2:
            return None

        class_curves: List[Dict[str, Any]] = []
        for idx, cls in enumerate(class_labels):
            y_bin_cls = (y_true == int(cls)).astype(int)
            if np.unique(y_bin_cls).size < 2:
                continue
            fpr_cls, tpr_cls, _ = roc_curve(y_bin_cls, aligned[:, idx])
            class_curves.append({
                "class_id": int(cls),
                "class_name": self.mlabel_mtype_dict.get(int(cls), str(int(cls))),
                "fpr": fpr_cls, "tpr": tpr_cls,
                "auc": float(auc(fpr_cls, tpr_cls)),
            })

        if not class_curves:
            return None

        if len(class_labels) == 2:
            pos_label = int(class_labels[1])
            y_bin = (y_true == pos_label).astype(int)
            scores = aligned[:, 1]
            if np.unique(y_bin).size < 2:
                fpr, tpr, roc_auc = class_curves[0]["fpr"], class_curves[0]["tpr"], class_curves[0]["auc"]
            else:
                fpr, tpr, _ = roc_curve(y_bin, scores)
                roc_auc = float(auc(fpr, tpr))
        else:
            y_bin = label_binarize(y_true, classes=class_labels)
            if y_bin.ndim != 2 or y_bin.shape[1] != len(class_labels) or np.unique(y_bin).size < 2:
                fpr, tpr, roc_auc = class_curves[0]["fpr"], class_curves[0]["tpr"], class_curves[0]["auc"]
            else:
                fpr, tpr, _ = roc_curve(y_bin.ravel(), aligned.ravel())
                roc_auc = float(auc(fpr, tpr))

        return {"fpr": fpr, "tpr": tpr, "auc": float(roc_auc), "curves": class_curves}

    def _plot_feature_importance(
        self,
        df_for_names: pd.DataFrame,
        importance_type: str = "gain",
        top_k: int = 30,
        title: str = "feature_importance",
    ) -> None:
        feature_names = list(self.feature_columns) if self.feature_columns else self._resolve_feature_columns(df_for_names)
        imp = self._importance_df(importance_type, feature_names)
        if imp.empty:
            return

        out_dir = os.path.join(self._output_root(), "eval/importance")
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, f"{title}_{importance_type}.csv")
        imp.to_csv(csv_path, index=True, encoding="utf-8-sig")
        print(f"✔️ 已保存特征重要性CSV：{csv_path}")

        top = imp.head(top_k)
        plt.figure(figsize=(8, max(4, len(top) * 0.35)))
        plt.barh(top.index[::-1], top["importance"].values[::-1])
        plt.xlabel(importance_type.title())
        plt.title(f"{title} ({importance_type})")
        plt.tight_layout()

        base_path = os.path.join(out_dir, f"{title}_{importance_type}")
        saved_paths = self._save_figure_multi_formats(plt.gcf(), base_path)
        plt.close()
        for path in saved_paths:
            print(f"✔️ 已保存特征重要性图：{path}")

    # ------------------------------------------------------------------ #
    # 阈值调优
    # ------------------------------------------------------------------ #

    def _predict_with_deployment_rule(
        self,
        y_proba: np.ndarray,
        model_classes: np.ndarray,
        threshold_by_label: Dict[int, float] | None = None,
    ) -> np.ndarray:
        threshold_dict = self.mlabel_threshold_dict if threshold_by_label is None else threshold_by_label
        return predict_with_threshold_rule(y_proba, model_classes, threshold_dict, ok_mlabel=self.ok_mlabel)

    def _resolve_threshold_tuning_cfg(self) -> Dict[str, Any]:
        raw_cfg = self.train_cfg.get("threshold_tuning") or {}
        if not isinstance(raw_cfg, dict):
            print("[WARN] train.threshold_tuning 不是字典，已回退默认配置。")
            raw_cfg = {}

        metric = str(raw_cfg.get("metric") or "f1_macro").lower()
        if metric not in {"f1_macro", "accuracy", "precision_macro", "recall_macro"}:
            print(f"[WARN] 不支持的 threshold_tuning.metric={metric}，已回退为 f1_macro。")
            metric = "f1_macro"

        def _f(k, default):
            try:
                v = raw_cfg.get(k)
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        def _i(k, default):
            try:
                v = raw_cfg.get(k)
                return int(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        thr_min = max(0.0, min(float(_f("min", 0.3)), 1.0))
        thr_max = max(thr_min, min(float(_f("max", 0.95)), 1.0))
        thr_step = max(float(_f("step", 0.05)), 1e-3)
        rounds = max(int(_i("rounds", 2)), 1)

        return {
            "enable": bool(raw_cfg.get("enable", True)),
            "metric": metric,
            "min": thr_min, "max": thr_max, "step": thr_step, "rounds": rounds,
            "include_ok": bool(raw_cfg.get("include_ok", True)),
        }

    def _threshold_metric_score(self, y_true: np.ndarray, y_pred: np.ndarray, *, metric_name: str) -> float:
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        if y_true.size == 0 or y_pred.size == 0:
            return float("-inf")
        if metric_name == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if metric_name == "precision_macro":
            return float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        if metric_name == "recall_macro":
            return float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    @staticmethod
    def _build_threshold_candidates(tuning_cfg: Dict[str, Any], *, current_threshold: float) -> List[float]:
        values = list(np.arange(
            float(tuning_cfg["min"]),
            float(tuning_cfg["max"]) + float(tuning_cfg["step"]) * 0.5,
            float(tuning_cfg["step"]), dtype=float,
        ))
        values.append(float(current_threshold))
        return [float(v) for v in sorted({round(float(v), 6) for v in values if 0.0 <= float(v) <= 1.0})]

    def _tune_thresholds_on_validation(
        self,
        *,
        y_val: np.ndarray,
        y_proba_val: np.ndarray,
        model_classes: np.ndarray,
    ) -> Dict[str, Any]:
        tuning_cfg = self._resolve_threshold_tuning_cfg()
        current_thresholds = {int(k): float(v) for k, v in self.mlabel_threshold_dict.items()}
        metric_name = str(tuning_cfg["metric"])
        _disabled = {"enabled": False, "source": "default", "metric": metric_name,
                     "thresholds": current_thresholds, "before_score": np.nan, "after_score": np.nan}

        if not bool(tuning_cfg["enable"]):
            return _disabled

        y_val = np.asarray(y_val, dtype=int)
        y_proba_val = np.asarray(y_proba_val, dtype=float)
        model_classes = np.asarray(model_classes, dtype=int)

        if y_val.size == 0 or y_proba_val.size == 0 or model_classes.size == 0:
            print("[WARN] 验证集为空，跳过阈值调优。")
            return _disabled

        labels_to_tune = [int(x) for x in model_classes.tolist()]
        if not bool(tuning_cfg["include_ok"]):
            labels_to_tune = [lb for lb in labels_to_tune if lb != int(self.ok_mlabel)]
        if not labels_to_tune:
            return _disabled

        before_pred = self._predict_with_deployment_rule(y_proba_val, model_classes, threshold_by_label=current_thresholds)
        before_score = self._threshold_metric_score(y_val, before_pred, metric_name=metric_name)

        best_thresholds = dict(current_thresholds)
        best_score = float(before_score)
        log_rows: List[Dict[str, Any]] = []
        tolerance = 1e-12

        print(
            f"[INFO] 开始验证集阈值调优: metric={metric_name}, "
            f"labels={labels_to_tune}, baseline={before_score:.4f}"
        )

        for round_idx in range(1, int(tuning_cfg["rounds"]) + 1):
            round_changed = False
            for label in labels_to_tune:
                baseline_thr = float(best_thresholds.get(label, 0.7))
                candidates = self._build_threshold_candidates(tuning_cfg, current_threshold=baseline_thr)
                label_best_thr = baseline_thr
                label_best_score = best_score

                for thr in candidates:
                    candidate_map = dict(best_thresholds)
                    candidate_map[int(label)] = float(thr)
                    cand_pred = self._predict_with_deployment_rule(y_proba_val, model_classes, threshold_by_label=candidate_map)
                    score = self._threshold_metric_score(y_val, cand_pred, metric_name=metric_name)
                    if (
                        score > label_best_score + tolerance
                        or (
                            abs(score - label_best_score) <= tolerance
                            and abs(float(thr) - baseline_thr) < abs(label_best_thr - baseline_thr)
                        )
                    ):
                        label_best_score = float(score)
                        label_best_thr = float(thr)

                changed = abs(label_best_thr - baseline_thr) > tolerance and label_best_score >= best_score - tolerance
                best_thresholds[int(label)] = float(label_best_thr)
                if label_best_score > best_score + tolerance:
                    best_score = float(label_best_score)
                if changed:
                    round_changed = True

                log_rows.append({
                    "round": round_idx, "mlabel": int(label),
                    "mtype": self.mlabel_mtype_dict.get(int(label), str(int(label))),
                    "threshold_before": float(baseline_thr), "threshold_after": float(label_best_thr),
                    "metric_score": float(label_best_score), "changed": bool(changed),
                })

            if not round_changed:
                print(f"[INFO] 阈值调优在 round {round_idx} 提前收敛。")
                break

        after_pred = self._predict_with_deployment_rule(y_proba_val, model_classes, threshold_by_label=best_thresholds)
        after_score = self._threshold_metric_score(y_val, after_pred, metric_name=metric_name)

        threshold_changed = any(
            abs(float(best_thresholds.get(int(lb), 0.7)) - float(current_thresholds.get(int(lb), 0.7))) > tolerance
            for lb in sorted(set(best_thresholds) | set(current_thresholds))
        )
        source = "validation_tuned" if threshold_changed else "default"

        out_dir = os.path.join(self._output_root(), "eval")
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame([
            {"mlabel": int(lb), "mtype": self.mlabel_mtype_dict.get(int(lb), str(int(lb))),
             "threshold_before": float(current_thresholds.get(int(lb), 0.7)),
             "threshold_after": float(best_thresholds.get(int(lb), 0.7))}
            for lb in sorted(best_thresholds)
        ]).to_csv(os.path.join(out_dir, "validation_thresholds.csv"), index=False, encoding="utf-8-sig")

        if log_rows:
            log_path = os.path.join(out_dir, "validation_threshold_tuning_steps.csv")
            pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")
            print(f"[INFO] 阈值调优过程已保存：{log_path}")

        print(f"[INFO] 验证集阈值调优: {metric_name} {before_score:.4f} -> {after_score:.4f}")
        if not threshold_changed:
            print("[INFO] 未找到更优阈值，继续使用默认阈值。")

        return {
            "enabled": True, "source": source, "metric": metric_name,
            "thresholds": best_thresholds,
            "before_score": float(before_score), "after_score": float(after_score),
        }

    # ------------------------------------------------------------------ #
    # 汇总可视化
    # ------------------------------------------------------------------ #

    def _aggregate_confusion_matrix(self, cm_list: List[np.ndarray]) -> np.ndarray:
        if not cm_list:
            return np.zeros((0, 0), dtype=int)
        return np.sum(np.stack(cm_list, axis=0), axis=0).astype(int)

    def _cleanup_root_eval_artifacts(self) -> None:
        root = Path(self.results_path)
        stale_paths = [root / "summary.csv", root / "summary_stats.csv", root / "best_model_eval"]
        for stale_path in stale_paths:
            if stale_path.is_dir():
                shutil.rmtree(stale_path)
            elif stale_path.exists():
                stale_path.unlink()
        for stale_path in list(root.glob("final_seed_report*")) + list(root.glob("summary_confusion_roc.*")):
            if stale_path.is_dir():
                shutil.rmtree(stale_path)
            elif stale_path.exists():
                stale_path.unlink()

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
                im = cm_ax.imshow(cm_sum, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=max(vmax, 1))
                cm_ax.figure.colorbar(im, ax=cm_ax, fraction=0.046, pad=0.04)
                suffix = f"(sum, seeds={len(cm_list)})" if len(cm_list) > 1 else "(count)"
                cm_ax.set_title(f"{split} {suffix}")
                cm_ax.set_xticks(np.arange(len(display_labels)))
                cm_ax.set_yticks(np.arange(len(display_labels)))
                cm_ax.set_xticklabels(display_labels, rotation=45, ha="right")
                cm_ax.set_yticklabels(display_labels)
                cm_ax.set_xlabel("Predicted")
                cm_ax.set_ylabel("True")
                threshold = float(cm_sum.max()) * 0.5 if cm_sum.size > 0 else 0.0
                for i in range(cm_sum.shape[0]):
                    for j in range(cm_sum.shape[1]):
                        cm_ax.text(j, i, f"{int(cm_sum[i, j])}", ha="center", va="center",
                                   color="white" if cm_sum[i, j] > threshold else "black", fontsize=8)
            else:
                cm_ax.axis("off")
                cm_ax.set_title(f"{split} (no data)")

            roc_list = split_rocs.get(split, [])
            if roc_list:
                if (not plot_all_roc_curves) or len(roc_list) == 1:
                    item = roc_list[0]
                    curves = item.get("curves", [])
                    if curves:
                        cmap = plt.colormaps.get_cmap("tab10").resampled(len(curves))
                        for c_idx, curve in enumerate(curves):
                            label_name = curve.get("class_name", str(curve.get("class_id", c_idx)))
                            roc_ax.plot(curve["fpr"], curve["tpr"], color=cmap(c_idx), linewidth=1.8,
                                        alpha=0.95, label=f"{label_name} AUC={curve['auc']:.3f}")
                    else:
                        roc_ax.plot(item["fpr"], item["tpr"], color="tab:red", linewidth=2.5,
                                    label=f"AUC={item['auc']:.3f}")
                else:
                    cmap = plt.colormaps.get_cmap("tab10").resampled(len(roc_list))
                    for idx, item in enumerate(roc_list):
                        if "fpr" not in item or "tpr" not in item:
                            continue
                        roc_ax.plot(item["fpr"], item["tpr"], color=cmap(idx), linewidth=1.8, alpha=0.9,
                                    label=f"seed#{idx + 1} AUC={item['auc']:.3f}")

                roc_ax.plot([0, 1], [0, 1], "k--", linewidth=1)
                roc_ax.set_xlim(0.0, 1.0)
                roc_ax.set_ylim(0.0, 1.05)
                first_curves = roc_list[0].get("curves", []) if roc_list else []
                roc_ax.set_title(f"{split}_ROC")
                roc_ax.set_xlabel("False Positive Rate")
                roc_ax.set_ylabel("True Positive Rate")
                roc_ax.grid(True, alpha=0.3)
                legend_items = len(first_curves) if (not plot_all_roc_curves) and roc_list else len(roc_list)
                roc_ax.legend(loc="lower right", fontsize=8 if legend_items > 4 else 9,
                              ncol=2 if legend_items > 4 else 1)
            else:
                roc_ax.axis("off")
                roc_ax.set_title(f"{split}_ROC (no data)")

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
            split_cms=split_cms, split_rocs=split_rocs,
            figure_title=figure_title, plot_all_roc_curves=plot_all_roc_curves,
        )
        output_paths = self._save_figure_multi_formats(fig, os.path.join(output_dir, filename_prefix), png_dpi=360)
        plt.close(fig)
        return output_paths

    def _export_final_report_pngs(
        self,
        *,
        seed_panels: List[Dict[str, Any]],
        summary_split_cms: Dict[str, List[np.ndarray]],
        summary_split_rocs: Dict[str, List[Dict[str, Any]]],
        output_dir: str,
    ) -> List[str]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for stale_path in list(out_dir.glob("final_seed_report*.png")) + [out_dir / "final_seed_report.pdf"]:
            if stale_path.exists():
                stale_path.unlink()

        output_paths: List[str] = []
        for panel in seed_panels:
            seed = int(panel["seed"])
            fig = self._create_seed_overview_figure(
                split_cms=panel["split_cms"], split_rocs=panel["split_rocs"],
                figure_title=f"Seed {seed} Summary (2x3)", plot_all_roc_curves=False,
            )
            page_path = out_dir / f"final_seed_report_seed_{seed}.png"
            fig.savefig(page_path, dpi=360, bbox_inches="tight")
            plt.close(fig)
            output_paths.append(str(page_path))

        fig_summary = self._create_seed_overview_figure(
            split_cms=summary_split_cms, split_rocs=summary_split_rocs,
            figure_title="Multi-seed Summary (2x3)", plot_all_roc_curves=True,
        )
        summary_path = out_dir / "final_seed_report_summary.png"
        fig_summary.savefig(summary_path, dpi=360, bbox_inches="tight")
        plt.close(fig_summary)
        output_paths.append(str(summary_path))
        return output_paths

    def _export_best_seed_png_results(self, *, best_run_dir: str, best_seed: int) -> List[str]:
        src_eval_dir = Path(best_run_dir) / "eval"
        if not src_eval_dir.exists():
            print(f"[WARN] 最佳种子 eval 目录不存在：{src_eval_dir}")
            return []

        dst_root = Path(self.results_path) / "eval" / "best_model_eval" / f"seed_{best_seed}"
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
        else:
            print(f"[INFO] 最佳模型 PNG 已保存：{dst_root}（共 {len(copied_paths)} 个）")
        return copied_paths

    # ================================================================== #
    # 导出（predict_config / 样本 / 误分类）
    # ================================================================== #

    def _write_predict_config(
        self,
        output_dir: str,
        *,
        model_seed: int | None = None,
        split_seed: int | None = None,
        threshold_source: str = "default",
    ) -> None:
        data: Dict[str, Any] = {
            "line_name": self.line_name,
            "model_id": self.model_id,
            "results_path": output_dir,
            "feature_version": self.feature_version,
            "predictor": {
                "ok_mlabel": int(self.ok_mlabel),
                "threshold_source": str(threshold_source or "default"),
                "label_mapping": [
                    {
                        "mlabel": int(k),
                        "mtype": v,
                        "threshold": float(self.mlabel_threshold_dict.get(int(k), 0.7)),
                    }
                    for k, v in sorted(self.mlabel_mtype_dict.items())
                ],
            },
        }
        if model_seed is not None:
            data["train_seed"] = int(model_seed)
            data["model_seed"] = int(model_seed)
        if split_seed is not None:
            data["split_seed"] = int(split_seed)
        if self.feature_columns:
            data["predictor"]["feature_columns"] = [str(name) for name in self.feature_columns]

        cfg_out = os.path.join(output_dir, "predict_config.yaml")
        with open(cfg_out, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)

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
        rows: List[Dict[str, str]] = []
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
    def _limit_review_export_df(cls, df: pd.DataFrame, *, top_k_sn: int = REVIEW_TOP_K_SN) -> pd.DataFrame:
        export_df = df.copy()
        if export_df.empty or int(top_k_sn) <= 0:
            return export_df.reset_index(drop=True)
        if "sn" not in export_df.columns:
            return export_df.head(int(top_k_sn)).reset_index(drop=True)

        ordered_sn_df = cls._build_review_sn_df(export_df)
        top_sns = ordered_sn_df["sn"].head(int(top_k_sn)).tolist()
        if not top_sns:
            return export_df.head(int(top_k_sn)).reset_index(drop=True)

        order_map = {sn: idx for idx, sn in enumerate(top_sns)}
        filtered_df = export_df.loc[export_df["sn"].map(_to_text).isin(top_sns)].copy()
        filtered_df["_sn_rank"] = filtered_df["sn"].map(lambda v: order_map.get(_to_text(v), len(order_map)))
        sort_by = ["_sn_rank"]
        ascending = [True]
        for col, asc in (("score_gap", False), ("pred_confidence", False), ("true_label_proba", True), ("sample_id", True)):
            if col in filtered_df.columns:
                sort_by.append(col)
                ascending.append(asc)
        return filtered_df.sort_values(by=sort_by, ascending=ascending, na_position="last").drop(columns="_sn_rank").reset_index(drop=True)

    @classmethod
    def _write_review_exports(cls, *, export_df: pd.DataFrame, sample_view_path: str | Path, top_k_sn: int = REVIEW_TOP_K_SN) -> tuple[pd.DataFrame, str]:
        out_path = Path(sample_view_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        limited_df = cls._limit_review_export_df(export_df, top_k_sn=top_k_sn)
        limited_df = cls._reorder_review_export_columns(limited_df)
        limited_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        sn_path = out_path.with_name(cls._review_sn_filename(out_path.name))
        cls._build_review_sn_df(limited_df).to_csv(sn_path, index=False, encoding="utf-8-sig")
        return limited_df, str(sn_path)

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
            raise ValueError(f"{split_name} 数据长度不一致: df={len(split_df)}, y_true={len(y_true)}, y_pred={len(y_pred)}")

        wrong_mask = y_true != y_pred
        metadata_cols = [col for col in SAMPLE_VIEW_EXPORT_COLUMNS if col in split_df.columns]
        extra_cols = [col for col in ("label", "sample_view_file", "sample_view_row_index")
                      if col in split_df.columns and col not in metadata_cols]

        export_df = split_df.loc[wrong_mask, metadata_cols + extra_cols].copy()
        if export_df.empty:
            return export_df

        class_labels = np.array(sorted(int(k) for k in self.mlabel_mtype_dict.keys()), dtype=int)
        aligned_proba = self._align_proba_columns(y_proba, model_classes, class_labels)
        class_to_idx = {int(cls): idx for idx, cls in enumerate(class_labels)}
        row_indices = np.flatnonzero(wrong_mask)

        pred_confidence, true_label_proba, max_label_proba, score_gap = [], [], [], []
        for row_idx, pred_label, true_label in zip(row_indices, y_pred[wrong_mask], y_true[wrong_mask]):
            pred_idx = class_to_idx.get(int(pred_label))
            true_idx = class_to_idx.get(int(true_label))
            row_proba = aligned_proba[row_idx]
            pred_prob = float(row_proba[pred_idx]) if pred_idx is not None else np.nan
            true_prob = float(row_proba[true_idx]) if true_idx is not None else np.nan
            pred_confidence.append(pred_prob)
            true_label_proba.append(true_prob)
            max_label_proba.append(float(np.max(row_proba)) if row_proba.size > 0 else np.nan)
            score_gap.append(np.nan if (np.isnan(pred_prob) or np.isnan(true_prob)) else float(pred_prob - true_prob))

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
            ascending=[False, False, True], na_position="last",
        ).reset_index(drop=True)

    def _save_test_misclassified_sample_view(
        self,
        *,
        test_df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        model_classes: np.ndarray,
        model_seed: int | None = None,
        split_seed: int | None = None,
        filename: str = "test_misclassified_sample_view.csv",
        top_k_sn: int = REVIEW_TOP_K_SN,
    ) -> str:
        out_dir = os.path.join(self._output_root(), "eval")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        export_df = self._build_misclassified_export_df(
            split_name="test", split_df=test_df,
            y_true=y_true, y_pred=y_pred, y_proba=y_proba,
            model_classes=model_classes, model_seed=model_seed, split_seed=split_seed,
        )
        export_df, sn_path = self._write_review_exports(export_df=export_df, sample_view_path=out_path, top_k_sn=top_k_sn)
        print(
            f"[INFO] test 错分样本已保存：{out_path} | "
            f"rows={len(export_df)} | unique_sn={len(self._build_review_sn_df(export_df))} | sn_csv={sn_path}"
        )
        return out_path

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
        filename: str = MISCLASSIFIED_EXPORT_FILENAME,
        top_k_sn: int = MISCLASSIFIED_TOP_K_SN,
    ) -> str:
        frames: List[pd.DataFrame] = []
        for split_name, split_df, y_true, y_pred, y_proba in (
            ("validation", val_df, y_val, y_pred_val, y_proba_val),
            ("test", test_df, y_test, y_pred_test, y_proba_test),
        ):
            frame = self._build_misclassified_export_df(
                split_name=split_name, split_df=split_df,
                y_true=y_true, y_pred=y_pred, y_proba=y_proba,
                model_classes=model_classes, model_seed=model_seed, split_seed=split_seed,
            )
            if not frame.empty:
                frames.append(frame)

        export_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=[*REVIEW_SAMPLE_VIEW_COLUMNS, *REVIEW_SAMPLE_VIEW_EXTRA_COLUMNS]
        )
        if not export_df.empty:
            export_df = export_df.sort_values(
                by=["score_gap", "pred_confidence", "true_label_proba"],
                ascending=[False, False, True], na_position="last",
            ).reset_index(drop=True)

        out_dir = os.path.join(self._output_root(), "eval")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        export_df, sn_path = self._write_review_exports(export_df=export_df, sample_view_path=out_path, top_k_sn=top_k_sn)
        print(
            "[INFO] validation/test 错分样本已保存："
            f"{out_path} | rows={len(export_df)} | "
            f"unique_sn={len(self._build_review_sn_df(export_df))} | sn_csv={sn_path}"
        )
        return out_path

    def _build_target_reason_mismatch_export_df(
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
        target_reason_names: tuple[str, ...] = SPECIAL_REVIEW_REASON_NAMES,
    ) -> pd.DataFrame:
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        model_classes = np.asarray(model_classes, dtype=int)

        if len(split_df) != len(y_true) or len(y_true) != len(y_pred):
            raise ValueError(f"{split_name} 数据长度不一致")

        normalized_targets = {_to_text(name) for name in target_reason_names if _to_text(name)}
        if not normalized_targets:
            return pd.DataFrame()

        if "reason_name" in split_df.columns:
            true_reason_names = split_df["reason_name"].map(_to_text).tolist()
        else:
            true_reason_names = [self.reason_name_by_label.get(int(label), str(int(label))) for label in y_true]

        wrong_mask = y_true != y_pred
        target_mask = np.array([name in normalized_targets for name in true_reason_names], dtype=bool)
        selected_mask = wrong_mask & target_mask

        if not np.any(selected_mask):
            return pd.DataFrame()

        metadata_cols = [col for col in SAMPLE_VIEW_EXPORT_COLUMNS if col in split_df.columns]
        extra_cols = [col for col in ("label", "sample_view_file", "sample_view_row_index")
                      if col in split_df.columns and col not in metadata_cols]

        export_df = split_df.loc[selected_mask, metadata_cols + extra_cols].copy()
        class_labels = np.array(sorted(int(v) for v in model_classes.tolist()), dtype=int)
        aligned_proba = self._align_proba_columns(y_proba, model_classes, class_labels)
        class_to_idx = {int(cls): idx for idx, cls in enumerate(class_labels)}
        row_indices = np.flatnonzero(selected_mask)

        pred_confidence, true_label_proba, max_label_proba, pred_mtypes, score_gap = [], [], [], [], []
        for row_idx, pred_label, true_label in zip(row_indices, y_pred[selected_mask], y_true[selected_mask]):
            pred_idx = class_to_idx.get(int(pred_label))
            true_idx = class_to_idx.get(int(true_label))
            row_proba = aligned_proba[row_idx]
            pred_prob = float(row_proba[pred_idx]) if pred_idx is not None else np.nan
            true_prob = float(row_proba[true_idx]) if true_idx is not None else np.nan
            pred_confidence.append(pred_prob)
            true_label_proba.append(true_prob)
            max_label_proba.append(float(np.max(row_proba)) if row_proba.size > 0 else np.nan)
            pred_mtypes.append(self.mlabel_mtype_dict.get(int(pred_label), str(int(pred_label))))
            score_gap.append(np.nan if (np.isnan(pred_prob) or np.isnan(true_prob)) else float(pred_prob - true_prob))

        export_df["split"] = str(split_name)
        export_df["true_mlabel"] = y_true[selected_mask].astype(int)
        export_df["true_mtype"] = [self.mlabel_mtype_dict.get(int(v), str(int(v))) for v in y_true[selected_mask]]
        export_df["pred_mlabel"] = y_pred[selected_mask].astype(int)
        export_df["pred_mtype"] = pred_mtypes
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
            ascending=[False, False, True], na_position="last",
        ).reset_index(drop=True)

    def _save_target_reason_mismatch_sample_view(
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
        filename: str = SPECIAL_REVIEW_EXPORT_FILENAME,
        target_reason_names: tuple[str, ...] = SPECIAL_REVIEW_REASON_NAMES,
        top_k_sn: int = REVIEW_TOP_K_SN,
    ) -> str | None:
        frames: List[pd.DataFrame] = []
        for split_name, split_df, y_true, y_pred, y_proba in (
            ("validation", val_df, y_val, y_pred_val, y_proba_val),
            ("test", test_df, y_test, y_pred_test, y_proba_test),
        ):
            frame = self._build_target_reason_mismatch_export_df(
                split_name=split_name, split_df=split_df,
                y_true=y_true, y_pred=y_pred, y_proba=y_proba,
                model_classes=model_classes, model_seed=model_seed, split_seed=split_seed,
                target_reason_names=target_reason_names,
            )
            if not frame.empty:
                frames.append(frame)

        if not frames:
            print("[INFO] 未找到指定标签预测不一致样本：" + "、".join(target_reason_names))
            return None

        export_df = pd.concat(frames, ignore_index=True).sort_values(
            by=["score_gap", "pred_confidence", "true_label_proba"],
            ascending=[False, False, True], na_position="last",
        ).reset_index(drop=True)

        out_dir = os.path.join(self._output_root(), "eval")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        export_df, sn_path = self._write_review_exports(export_df=export_df, sample_view_path=out_path, top_k_sn=top_k_sn)
        print(
            "[INFO] 指定标签预测不一致 sample_view 已保存："
            f"{out_path} | rows={len(export_df)} | "
            f"reasons={'、'.join(target_reason_names)} | sn_csv={sn_path}"
        )
        return out_path

    def _export_best_seed_misclassified_sample_view(self, *, best_run_dir: str, best_seed: int) -> str | None:
        src = Path(best_run_dir) / "eval" / MISCLASSIFIED_EXPORT_FILENAME
        if not src.exists():
            print(f"[WARN] 最佳种子错分样本不存在，跳过复制：{src}")
            return None

        dst_root = Path(self.results_path) / "eval" / "best_model_eval" / f"seed_{best_seed}"
        dst_root.mkdir(parents=True, exist_ok=True)
        dst = dst_root / src.name
        shutil.copy2(src, dst)

        src_sn = src.with_name(self._review_sn_filename(src.name))
        if src_sn.exists():
            shutil.copy2(src_sn, dst_root / src_sn.name)

        print(f"[INFO] 最佳模型错分样本已保存：{dst}")
        return str(dst)

    def _copy_misclassified_sample_view_to_root_eval(
        self,
        *,
        source_path: str | None,
        filename: str = MISCLASSIFIED_EXPORT_FILENAME,
    ) -> str | None:
        src_text = _to_text(source_path)
        if not src_text:
            return None
        src = Path(src_text).expanduser()
        if not src.exists():
            print(f"[WARN] 错分样本源文件不存在，跳过复制：{src}")
            return None

        dst_root = Path(self.results_path) / "eval"
        dst_root.mkdir(parents=True, exist_ok=True)
        dst = dst_root / filename

        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

        src_sn = src.with_name(self._review_sn_filename(src.name))
        if src_sn.exists():
            dst_sn = dst_root / self._review_sn_filename(filename)
            if src_sn.resolve() != dst_sn.resolve():
                shutil.copy2(src_sn, dst_sn)

        print(f"[INFO] 错分样本已写入根 eval：{dst}")
        return str(dst)

    def _copy_sample_view_to_cwd(
        self,
        *,
        source_path: str | None,
        filename: str = CWD_SAMPLE_VIEW_FILENAME,
        sn_filename: str = CWD_SN_FILENAME,
    ) -> str | None:
        src_text = _to_text(source_path)
        if not src_text:
            return None
        src = Path(src_text).expanduser()
        if not src.exists():
            print(f"[WARN] sample_view 源文件不存在：{src}")
            return None

        dst = Path.cwd() / filename
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        print(f"[INFO] sample_view 已复制到当前目录：{dst}")

        src_sn = src.with_name(self._review_sn_filename(src.name))
        dst_sn = Path.cwd() / sn_filename
        if src_sn.exists():
            shutil.copy2(src_sn, dst_sn)
        else:
            try:
                self._build_review_sn_df(pd.read_csv(src, encoding="utf-8-sig")).to_csv(dst_sn, index=False, encoding="utf-8-sig")
            except Exception as exc:
                print(f"[WARN] 无法生成 sn.csv：{exc}")
                return str(dst)

        print(f"[INFO] sn.csv 已复制到当前目录：{dst_sn}")
        return str(dst)

    def _generate_label_audit_outputs(self) -> None:
        if bool(getattr(self, "_label_audit_generated", False)):
            print("[INFO] 标签复核清单已生成，跳过重复执行。")
            return

        results_dir = Path(self.results_path).expanduser()
        if not results_dir.exists():
            print(f"[WARN] results_path 不存在：{results_dir}")
            return

        try:
            summary = run_label_audit(results_dir=results_dir, output_dir=results_dir / "confirm_label")
        except FileNotFoundError as exc:
            print(f"[WARN] 未生成标签复核清单：{exc}")
            return
        except Exception as exc:
            print(f"[WARN] 标签复核清单生成失败：{exc}")
            return

        print(
            "[INFO] 标签复核清单已生成："
            f" candidates={summary.get('candidates', 0)},"
            f" suspicious_sns={summary.get('suspicious_sns', 0)},"
            f" false_ok={summary.get('false_ok', 0)}"
        )
        self._label_audit_generated = True

    # ================================================================== #
    # 训练入口
    # ================================================================== #

    def train_and_evaluate(self, split_data: TrainValTestSplit):
        self._cleanup_root_eval_artifacts()

        seeds = self._resolve_training_seeds()
        multi_seed = len(seeds) > 1
        shap_each_seed = bool(self.train_cfg.get("shap_each_seed", False))

        run_records: List[Dict[str, Any]] = []
        seed_panels: List[Dict[str, Any]] = []
        threshold_dict_by_seed: Dict[int, Dict[int, float]] = {}
        threshold_source_by_seed: Dict[int, str] = {}

        class_labels = np.array(sorted(int(k) for k in self.mlabel_mtype_dict.keys()), dtype=int)
        split_cms: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}
        split_rocs: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
        latest_misclassified_sample_view_path: str | None = None

        for idx, seed in enumerate(seeds, start=1):
            print(f"\n===== Seed Run {idx}/{len(seeds)} | seed={seed} =====")

            self.random_state = int(seed)
            self.mlabel_threshold_dict = dict(self.default_mlabel_threshold_dict)
            self.model = self._build_classifier(extra_params={"random_state": self.random_state})
            run_dir = self._activate_run_dir(model_seed=seed, multi_seed=multi_seed)

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

            sample_weight = self._compute_sample_weight(np.asarray(y_train))
            feature_weights = self._compute_feature_weights()
            self._do_model_fit(X_train, y_train, X_val, y_val,
                               sample_weight=sample_weight, feature_weights=feature_weights)
            self._plot_training_loss(title="training_loss")
            self._print_model_info()

            y_proba_train = self.model.predict_proba(X_train)
            y_proba_val = self.model.predict_proba(X_val)
            y_proba_test = self.model.predict_proba(X_test)
            model_classes = np.array(self.model.classes_, dtype=int)

            tuning_result = self._tune_thresholds_on_validation(
                y_val=y_val, y_proba_val=y_proba_val, model_classes=model_classes,
            )
            self.mlabel_threshold_dict = {int(k): float(v) for k, v in tuning_result["thresholds"].items()}
            threshold_dict_by_seed[int(seed)] = dict(self.mlabel_threshold_dict)
            threshold_source_by_seed[int(seed)] = str(tuning_result.get("source", "default"))

            self._write_predict_config(
                run_dir, model_seed=seed, split_seed=self.split_random_state,
                threshold_source=str(tuning_result.get("source", "default")),
            )
            self._save_model()

            y_pred_train = self._predict_with_deployment_rule(y_proba_train, model_classes)
            y_pred_val = self._predict_with_deployment_rule(y_proba_val, model_classes)
            y_pred_test = self._predict_with_deployment_rule(y_proba_test, model_classes)

            self._save_test_misclassified_sample_view(
                test_df=test, y_true=y_test, y_pred=y_pred_test,
                y_proba=y_proba_test, model_classes=model_classes,
                model_seed=seed, split_seed=self.split_random_state,
            )
            misclassified_sample_view_path = self._save_validation_test_misclassified_sample_view(
                val_df=val, y_val=y_val, y_pred_val=y_pred_val, y_proba_val=y_proba_val,
                test_df=test, y_test=y_test, y_pred_test=y_pred_test, y_proba_test=y_proba_test,
                model_classes=model_classes, model_seed=seed, split_seed=self.split_random_state,
            )
            latest_misclassified_sample_view_path = misclassified_sample_view_path
            special_reason_sample_view_path = self._save_target_reason_mismatch_sample_view(
                val_df=val, y_val=y_val, y_pred_val=y_pred_val, y_proba_val=y_proba_val,
                test_df=test, y_test=y_test, y_pred_test=y_pred_test, y_proba_test=y_proba_test,
                model_classes=model_classes, model_seed=seed, split_seed=self.split_random_state,
            )

            train_roc = self._compute_split_roc(y_train, y_proba_train, model_classes, class_labels)
            val_roc = self._compute_split_roc(y_val, y_proba_val, model_classes, class_labels)
            test_roc = self._compute_split_roc(y_test, y_proba_test, model_classes, class_labels)

            split_cms["train"].append(confusion_matrix(y_train, y_pred_train, labels=class_labels).astype(int))
            split_cms["validation"].append(confusion_matrix(y_val, y_pred_val, labels=class_labels).astype(int))
            split_cms["test"].append(confusion_matrix(y_test, y_pred_test, labels=class_labels).astype(int))

            for split_key, roc_item in (("train", train_roc), ("validation", val_roc), ("test", test_roc)):
                if roc_item is not None:
                    split_rocs[split_key].append(roc_item)

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

            self._plot_feature_importance(train, importance_type="gain", top_k=30, title="train_feature_importance")

            if (not multi_seed) or shap_each_seed:
                self._plot_shap_summary(X_train, train, title="train_feature_importance_shap")
            elif idx == 1:
                print("[INFO] 多随机种子模式默认跳过 SHAP。可设置 train.shap_each_seed=true 开启。")

            seed_plot_paths = self._plot_seed_overview(
                split_cms=panel_data["split_cms"], split_rocs=panel_data["split_rocs"],
                output_dir=os.path.join(run_dir, "eval"),
                filename_prefix="seed_confusion_roc",
                figure_title=f"Seed {seed} Summary: train / validation / test",
                plot_all_roc_curves=False,
            )
            for p in seed_plot_paths:
                print(f"[INFO] Seed 汇总图已保存：{p}")

            try:
                diag_output_dir = os.path.join(run_dir, "eval", "diagnostics")
                generate_training_diagnostics(
                    X_train=X_train, y_train=y_train,
                    y_pred_train=y_pred_train, y_proba_train=y_proba_train,
                    X_val=X_val, y_val=y_val,
                    y_pred_val=y_pred_val, y_proba_val=y_proba_val,
                    X_test=X_test, y_test=y_test,
                    y_pred_test=y_pred_test, y_proba_test=y_proba_test,
                    model_classes=model_classes,
                    mlabel_mtype_dict=self.mlabel_mtype_dict,
                    output_dir=diag_output_dir,
                    seed=seed, ok_label=self.ok_mlabel,
                )
            except Exception as e:
                print(f"[WARN] 训练诊断报告生成失败: {e}")

            train_qm = compute_quality_gate_metrics(y_true=y_train, y_pred=y_pred_train, ok_label=self.ok_mlabel)
            val_qm = compute_quality_gate_metrics(y_true=y_val, y_pred=y_pred_val, ok_label=self.ok_mlabel)
            test_qm = compute_quality_gate_metrics(y_true=y_test, y_pred=y_pred_test, ok_label=self.ok_mlabel)

            run_records.append({
                "seed": seed,
                "split_seed": int(self.split_random_state),
                "run_dir": run_dir,
                "train_acc": float(accuracy_score(y_train, y_pred_train)),
                "val_acc": float(accuracy_score(y_val, y_pred_val)),
                "test_acc": float(accuracy_score(y_test, y_pred_test)),
                "val_f1_macro": float(f1_score(y_val, y_pred_val, average="macro", zero_division=0)),
                "test_f1_macro": float(f1_score(y_test, y_pred_test, average="macro", zero_division=0)),
                "train_nok_recall": float(train_qm["nok_recall"]),
                "val_nok_recall": float(val_qm["nok_recall"]),
                "test_nok_recall": float(test_qm["nok_recall"]),
                "train_false_ok_rate": float(train_qm["false_ok_rate"]),
                "val_false_ok_rate": float(val_qm["false_ok_rate"]),
                "test_false_ok_rate": float(test_qm["false_ok_rate"]),
                "train_false_reject_rate": float(train_qm["false_reject_rate"]),
                "val_false_reject_rate": float(val_qm["false_reject_rate"]),
                "test_false_reject_rate": float(test_qm["false_reject_rate"]),
                "train_false_ok_count": int(train_qm["false_ok_count"]),
                "val_false_ok_count": int(val_qm["false_ok_count"]),
                "test_false_ok_count": int(test_qm["false_ok_count"]),
                "train_auc": float(train_roc["auc"]) if train_roc is not None else np.nan,
                "val_auc": float(val_roc["auc"]) if val_roc is not None else np.nan,
                "test_auc": float(test_roc["auc"]) if test_roc is not None else np.nan,
                "val_threshold_metric_before": float(tuning_result.get("before_score", np.nan)),
                "val_threshold_metric_after": float(tuning_result.get("after_score", np.nan)),
                "test_misclassified_sample_view": misclassified_sample_view_path,
                "validation_test_misclassified_sample_view": misclassified_sample_view_path,
                "validation_test_special_reason_sample_view": special_reason_sample_view_path,
            })

        if not run_records:
            raise RuntimeError("未产生任何训练结果。")

        summary_df = pd.DataFrame(run_records).sort_values(
            by=["val_false_ok_rate", "val_nok_recall", "val_f1_macro", "seed"],
            ascending=[True, False, False, True], na_position="last",
        )

        summary_dir = Path(self.results_path) / "eval"
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary_path = summary_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] 训练汇总已保存：{summary_path}")

        root_misclassified_sample_view_path = self._copy_misclassified_sample_view_to_root_eval(
            source_path=latest_misclassified_sample_view_path,
        )
        self._copy_sample_view_to_cwd(source_path=root_misclassified_sample_view_path)

        summary_stats = (
            summary_df.select_dtypes(include=[np.number])
            .agg(["mean", "std"])
            .transpose()
            .reset_index()
            .rename(columns={"index": "metric"})
        )
        stats_path = summary_dir / "summary_stats.csv"
        summary_stats.to_csv(stats_path, index=False, encoding="utf-8-sig")

        plot_paths = self._plot_seed_overview(
            split_cms=split_cms, split_rocs=split_rocs,
            output_dir=str(summary_dir), plot_all_roc_curves=True,
        )
        for p in plot_paths:
            print(f"[INFO] 汇总图已保存：{p}")

        final_report_paths = self._export_final_report_pngs(
            seed_panels=seed_panels,
            summary_split_cms=split_cms,
            summary_split_rocs=split_rocs,
            output_dir=str(summary_dir),
        )
        print(f"[INFO] 最终报告 PNG：共 {len(final_report_paths)} 张")

        mean_row = summary_stats.set_index("metric")
        tracked_metrics = [
            "train_acc", "val_acc", "test_acc",
            "val_f1_macro", "test_f1_macro",
            "val_nok_recall", "test_nok_recall",
            "val_false_ok_rate", "test_false_ok_rate",
            "train_auc", "val_auc", "test_auc",
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
            best_thresholds = threshold_dict_by_seed.get(best_seed, dict(self.default_mlabel_threshold_dict))
            self.mlabel_threshold_dict = {int(k): float(v) for k, v in best_thresholds.items()}

            try:
                same_model_file = os.path.samefile(best_model_path, root_model_path)
            except FileNotFoundError:
                same_model_file = False
            if not same_model_file:
                shutil.copy2(best_model_path, root_model_path)

            self.active_results_path = self.results_path
            self._write_predict_config(
                self.results_path, model_seed=best_seed,
                split_seed=self.split_random_state,
                threshold_source=threshold_source_by_seed.get(best_seed, "default"),
            )
            self._export_best_seed_png_results(best_run_dir=best_run_dir, best_seed=best_seed)
            self._export_best_seed_misclassified_sample_view(best_run_dir=best_run_dir, best_seed=best_seed)
            self._generate_label_audit_outputs()

            launcher = export_runtime_bundle(
                Path(self.results_path).parent,
                self.model_id,
                source_model_dir=self.results_path,
                feature_version=self.feature_version,
            )
            print(f"[INFO] 根目录推理运行包已更新：{launcher}")

        seed_runs_dir = Path(self.results_path) / "seed_runs"
        if seed_runs_dir.exists():
            shutil.rmtree(seed_runs_dir)
            print(f"[INFO] 已清理临时多随机种子目录：{seed_runs_dir}")

    def train_and_cross_validate(self, cv_data: PreparedCrossValidationData):
        self._cleanup_root_eval_artifacts()
        self.random_state = self.base_model_random_state
        self.mlabel_threshold_dict = dict(self.default_mlabel_threshold_dict)
        self.active_results_path = self.results_path
        self.model = self._build_classifier(extra_params={"random_state": self.random_state})
        self._write_predict_config(
            self.results_path, model_seed=self.random_state,
            split_seed=self.split_random_state, threshold_source="default",
        )

        df = cv_data.df.copy()
        X = self._build_feature_matrix(df, fit=True)
        y = df["label"].values
        cv = cv_data.cv
        groups = cv_data.groups
        resolved_cv_strategy = str(cv_data.resolved_strategy)

        from sklearn.metrics import precision_score, recall_score as _recall
        scoring = {
            "accuracy": build_threshold_scorer(accuracy_score, threshold_by_label=self.mlabel_threshold_dict, ok_mlabel=self.ok_mlabel),
            "f1": build_threshold_scorer(f1_score, threshold_by_label=self.mlabel_threshold_dict, ok_mlabel=self.ok_mlabel, average="macro", zero_division=0),
            "precision": build_threshold_scorer(precision_score, threshold_by_label=self.mlabel_threshold_dict, ok_mlabel=self.ok_mlabel, average="macro", zero_division=0),
            "recall": build_threshold_scorer(_recall, threshold_by_label=self.mlabel_threshold_dict, ok_mlabel=self.ok_mlabel, average="macro", zero_division=0),
        }

        results = cross_validate(self.model, X, y, cv=cv, groups=groups, scoring=scoring,
                                 return_train_score=False, n_jobs=-1)

        print(f"\n—— {int(cv_data.requested_splits)}-折交叉验证 ({resolved_cv_strategy}) ——")
        for m in scoring:
            s = results[f"test_{m}"]
            print(f"{m.title():>10s}: {s.mean():.3f} ± {s.std():.3f}")

        print("\n—— 重训全量并保存 ——")
        self._do_full_data_fit(X, y, sample_weight=self._compute_sample_weight(np.asarray(y)))
        self._save_model()

    def train_and_evaluate_grid(
        self,
        split_data: TrainValTestSplit,
        grid_cv_data: PreparedCrossValidationData,
    ):
        self._cleanup_root_eval_artifacts()
        self.random_state = self.base_model_random_state
        self.mlabel_threshold_dict = dict(self.default_mlabel_threshold_dict)
        self.active_results_path = self.results_path
        self.model = self._build_classifier(extra_params={"random_state": self.random_state})
        self._write_predict_config(
            self.results_path, model_seed=self.random_state,
            split_seed=self.split_random_state, threshold_source="default",
        )

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

        grid = GridSearchCV(
            self._build_classifier(),
            param_grid=self._get_param_grid(),
            scoring=build_threshold_scorer(accuracy_score, threshold_by_label=self.mlabel_threshold_dict, ok_mlabel=self.ok_mlabel),
            cv=grid_cv_data.cv, verbose=1, n_jobs=-1,
        )

        sample_weight = self._compute_sample_weight(np.asarray(y_train))
        grid.fit(X_train, y_train, groups=grid_cv_data.groups, sample_weight=sample_weight)
        print(f"最佳参数: {grid.best_params_}, CV={grid.best_score_:.4f}")

        self.model = grid.best_estimator_
        self._print_model_info()

        y_proba_train = self.model.predict_proba(X_train)
        y_proba_val = self.model.predict_proba(X_val)
        y_proba_test = self.model.predict_proba(X_test)
        model_classes = np.array(self.model.classes_, dtype=int)

        tuning_result = self._tune_thresholds_on_validation(
            y_val=y_val, y_proba_val=y_proba_val, model_classes=model_classes,
        )
        self.mlabel_threshold_dict = {int(k): float(v) for k, v in tuning_result["thresholds"].items()}

        y_pred_train = self._predict_with_deployment_rule(y_proba_train, model_classes)
        y_pred_val = self._predict_with_deployment_rule(y_proba_val, model_classes)
        y_pred_test = self._predict_with_deployment_rule(y_proba_test, model_classes)

        self._write_predict_config(
            self.results_path, model_seed=self.random_state,
            split_seed=self.split_random_state,
            threshold_source=str(tuning_result.get("source", "default")),
        )
        self._save_model()

        self._save_test_misclassified_sample_view(
            test_df=test, y_true=y_test, y_pred=y_pred_test,
            y_proba=y_proba_test, model_classes=model_classes,
            model_seed=self.random_state, split_seed=self.split_random_state,
        )
        misclassified_sample_view_path = self._save_validation_test_misclassified_sample_view(
            val_df=val, y_val=y_val, y_pred_val=y_pred_val, y_proba_val=y_proba_val,
            test_df=test, y_test=y_test, y_pred_test=y_pred_test, y_proba_test=y_proba_test,
            model_classes=model_classes,
            model_seed=self.random_state, split_seed=self.split_random_state,
        )
        self._copy_sample_view_to_cwd(source_path=misclassified_sample_view_path)

        self._save_target_reason_mismatch_sample_view(
            val_df=val, y_val=y_val, y_pred_val=y_pred_val, y_proba_val=y_proba_val,
            test_df=test, y_test=y_test, y_pred_test=y_pred_test, y_proba_test=y_proba_test,
            model_classes=model_classes,
            model_seed=self.random_state, split_seed=self.split_random_state,
        )

        self._evaluate_predictions(y_train, y_pred_train, "train")
        self._plot_multiclass_roc(y_train, y_proba_train, "train")
        self._evaluate_predictions(y_val, y_pred_val, "validation")
        self._plot_multiclass_roc(y_val, y_proba_val, "validation")
        self._evaluate_predictions(y_test, y_pred_test, "test")
        self._plot_multiclass_roc(y_test, y_proba_test, "test")
