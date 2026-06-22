"""XGBoost 分类模型，继承自 BaseModel。

仅包含 XGBoost 特有逻辑：
  - _build_classifier      → 构建 XGBClassifier
  - _compute_feature_weights → 返回 feature_weights 数组
  - _do_model_fit           → XGB fit（含 eval_set、feature_weights）
  - _do_full_data_fit       → 全量重训（含 feature_weights）
  - _plot_training_loss     → 从 model.evals_result() 读取曲线
  - _importance_df          → 通过 booster.get_score() 获取
  - _plot_shap_summary      → XGBoost SHAP 图
  - _save_figure_multi_formats → PNG-only 覆盖（删除旧 PDF/SVG）
  - _get_param_grid         → 使用 xgb_grid 配置键
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Any, Dict, List

from xgboost import XGBClassifier
from sklearn.metrics import f1_score

try:
    from model_support import DEFAULT_CONFIG_PATH, _to_int, _to_float, _to_text
except ModuleNotFoundError:
    from ml.training.config import DEFAULT_CONFIG_PATH, _to_int, _to_float, _to_text

try:
    from ml.models.base import BaseModel
except ModuleNotFoundError:
    from base_model import BaseModel


class XGBModel(BaseModel):
    """XGBoost 分类模型。"""

    DEFAULT_MODEL_TYPE = "xgb"

    # ------------------------------------------------------------------ #
    # 早停辅助（XGBoost 专用）                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _one_minus_f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """XGBoost 自定义早停判据：1 - f1_macro，越小越好。"""
        if y_pred.ndim == 2:
            y_hat = np.argmax(y_pred, axis=1)
        else:
            y_hat = (np.asarray(y_pred) > 0.5).astype(int)
        return 1.0 - float(f1_score(y_true, y_hat, average="macro", zero_division=0))

    def _early_stopping_on_f1_macro(self) -> bool:
        metric = self.train_cfg.get("early_stopping_metric")
        return isinstance(metric, str) and metric.strip().lower() == "f1_macro"

    # ------------------------------------------------------------------ #
    # 模型构建                                                             #
    # ------------------------------------------------------------------ #

    def _build_classifier(self, *, extra_params: Dict[str, Any] | None = None) -> XGBClassifier:
        """构造 XGBClassifier。参数来源：extra_params > xgb_params（YAML）> 默认值。"""
        xgb_params = self.train_cfg.get("xgb_params", {})
        if not isinstance(xgb_params, dict):
            print("[WARN] train.xgb_params 不是字典，已忽略。")
            xgb_params = {}

        extra_params = extra_params or {}

        eval_metric = extra_params.get("eval_metric", xgb_params.get("eval_metric", self.eval_metric))
        n_estimators = extra_params.get("n_estimators", xgb_params.get("n_estimators", 400))
        max_depth = extra_params.get("max_depth", xgb_params.get("max_depth", 6))
        learning_rate = extra_params.get("learning_rate", xgb_params.get("learning_rate", 0.05))
        min_child_weight = extra_params.get("min_child_weight", xgb_params.get("min_child_weight", 5))
        gamma = extra_params.get("gamma", xgb_params.get("gamma", 1.0))
        reg_lambda = extra_params.get("reg_lambda", xgb_params.get("reg_lambda", 5.0))
        reg_alpha = extra_params.get("reg_alpha", xgb_params.get("reg_alpha", 0.5))
        subsample = extra_params.get("subsample", xgb_params.get("subsample", 0.8))
        colsample_bytree = extra_params.get("colsample_bytree", xgb_params.get("colsample_bytree", 0.7))
        tree_method = extra_params.get("tree_method", xgb_params.get("tree_method", "hist"))
        random_state = extra_params.get("random_state", xgb_params.get("random_state", self.random_state))
        n_jobs = extra_params.get("n_jobs", xgb_params.get("n_jobs", -1))

        early_stopping_rounds = _to_int(self.train_cfg.get("early_stopping_rounds"))
        if early_stopping_rounds is not None and early_stopping_rounds <= 0:
            early_stopping_rounds = None

        if self._early_stopping_on_f1_macro():
            early_stopping_rounds = early_stopping_rounds or 30
            eval_metric = self._one_minus_f1_macro
            print(
                "[INFO] 早停判据 = val_f1_macro "
                f"(1-f1_macro 越小越好), early_stopping_rounds={early_stopping_rounds}"
            )
        elif early_stopping_rounds:
            print(f"[INFO] 启用早停: early_stopping_rounds={early_stopping_rounds}")

        common_kwargs: Dict[str, Any] = dict(
            objective=self.objective,
            eval_metric=eval_metric,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method=tree_method,
            random_state=random_state,
            n_jobs=n_jobs,
            early_stopping_rounds=early_stopping_rounds,
        )

        if self.n_classes > 2:
            common_kwargs["num_class"] = self.n_classes

        return XGBClassifier(**common_kwargs)

    # ------------------------------------------------------------------ #
    # Feature weights                                                      #
    # ------------------------------------------------------------------ #

    def _compute_feature_weights(self) -> np.ndarray | None:
        """根据 train.feature_weights 生成 XGBoost feature_weights 参数。"""
        fw_cfg = self.train_cfg.get("feature_weights")
        if not isinstance(fw_cfg, dict) or not fw_cfg or not self.feature_columns:
            return None

        col_set = set(self.feature_columns)
        feature_weights = np.array(
            [float(fw_cfg.get(col, 1.0)) for col in self.feature_columns],
            dtype=np.float64,
        )

        if not np.any(feature_weights != 1.0):
            return None

        applied = {k: float(v) for k, v in fw_cfg.items() if k in col_set}
        missing = [k for k in fw_cfg if k not in col_set]

        print(f"[INFO] 启用 feature_weights: {applied}")
        if missing:
            print(f"[WARN] feature_weights 中未匹配到特征列: {missing}")

        return feature_weights

    # ------------------------------------------------------------------ #
    # Fit 钩子                                                             #
    # ------------------------------------------------------------------ #

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
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            feature_weights=feature_weights,
            verbose=False,
        )

    def _do_full_data_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
    ) -> None:
        """交叉验证后在全量数据重训：XGBoost 需额外传入 feature_weights。"""
        self.model.fit(
            X, y,
            sample_weight=sample_weight,
            feature_weights=self._compute_feature_weights(),
            verbose=False,
        )

    # ------------------------------------------------------------------ #
    # 训练损失曲线                                                         #
    # ------------------------------------------------------------------ #

    def _plot_training_loss(self, *, title: str = "training_loss") -> List[str]:
        dataset_name_map = {
            "validation_0": "train",
            "validation_1": "val",
            "validation_2": "test",
        }

        try:
            evals_result = self.model.evals_result()
        except Exception:
            return []

        if not evals_result:
            return []

        series_items: List[tuple[str, List[float]]] = []
        for dataset, metrics in evals_result.items():
            if not isinstance(metrics, dict):
                continue
            dataset_label = dataset_name_map.get(str(dataset), str(dataset))
            for metric, values in metrics.items():
                if not isinstance(values, list):
                    continue
                curve = [float(x) if isinstance(x, (int, float)) else np.nan for x in values]
                series_items.append((f"{dataset_label}:{metric}", curve))

        if not series_items:
            return []

        out_dir = os.path.join(self._output_root(), "eval")
        os.makedirs(out_dir, exist_ok=True)

        max_len = max((len(v) for _, v in series_items), default=0)
        if max_len == 0:
            return []

        rounds = np.arange(1, max_len + 1)
        fig, ax = plt.subplots(figsize=(9, 4.8))

        for name, values in series_items:
            y = np.asarray(values, dtype=float)
            if len(y) < max_len:
                y = np.concatenate([y, np.full(max_len - len(y), np.nan)])
            ax.plot(rounds, y, linewidth=1.8, label=name)

        ax.set_xlabel("Round")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        base_path = os.path.join(out_dir, title)
        saved_paths = self._save_figure_multi_formats(fig, base_path, png_dpi=300)
        plt.close()

        csv_data: Dict[str, Any] = {"round": rounds}
        for name, values in series_items:
            col = np.asarray(values, dtype=float)
            if len(col) < max_len:
                col = np.concatenate([col, np.full(max_len - len(col), np.nan)])
            csv_data[name] = col

        loss_df = pd.DataFrame(csv_data)
        csv_path = f"{base_path}.csv"
        loss_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        saved_paths.append(csv_path)

        for path in saved_paths:
            print(f"✔️ 已保存训练损失：{path}")

        return saved_paths

    # ------------------------------------------------------------------ #
    # 特征重要性                                                           #
    # ------------------------------------------------------------------ #

    def _importance_df(self, importance_type: str, feature_names: list) -> pd.DataFrame:
        booster = self.model.get_booster()
        raw = booster.get_score(importance_type=importance_type)
        fmap = {f"f{i}": name for i, name in enumerate(feature_names)}
        renamed = {fmap.get(k, k): v for k, v in raw.items()}
        if not renamed:
            renamed = raw
        return pd.Series(renamed, name="importance").sort_values(ascending=False).to_frame()

    # ------------------------------------------------------------------ #
    # SHAP                                                                 #
    # ------------------------------------------------------------------ #

    def _plot_shap_summary(
        self,
        X: np.ndarray,
        df_for_names: pd.DataFrame,
        title: str = "feature_importance_shap",
    ) -> None:
        try:
            import shap
        except Exception as e:
            print(f"ℹ️ 未安装 shap，跳过SHAP绘图：{e}")
            return

        feature_names = list(self.feature_columns) if self.feature_columns else self._resolve_feature_columns(df_for_names)

        shap_max_samples = _to_int(self.train_cfg.get("shap_max_samples"))
        if shap_max_samples is None or shap_max_samples <= 0:
            shap_max_samples = min(2000, X.shape[0])

        if X.shape[0] > shap_max_samples:
            rng = np.random.default_rng(self.random_state)
            selected_idx = rng.choice(X.shape[0], size=shap_max_samples, replace=False)
            X_shap = X[selected_idx]
            print(f"[INFO] SHAP 样本抽样: {X.shape[0]} -> {X_shap.shape[0]}")
        else:
            X_shap = X

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_shap)

        out_dir = os.path.join(self._output_root(), "eval/importance")
        os.makedirs(out_dir, exist_ok=True)

        plt.figure()
        shap.summary_plot(shap_values, features=X_shap, feature_names=feature_names,
                          plot_type="bar", show=False)
        base_bar = os.path.join(out_dir, f"{title}_bar")
        plt.tight_layout()
        for path in self._save_figure_multi_formats(plt.gcf(), base_bar):
            print(f"✔️ 已保存SHAP条形图：{path}")
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, features=X_shap, feature_names=feature_names, show=False)
        base_bees = os.path.join(out_dir, f"{title}_beeswarm")
        plt.tight_layout()
        for path in self._save_figure_multi_formats(plt.gcf(), base_bees):
            print(f"✔️ 已保存SHAP蜂群图：{path}")
        plt.close()

    # ------------------------------------------------------------------ #
    # PNG-only 图表保存（覆盖基类的 PNG+PDF+SVG）                         #
    # ------------------------------------------------------------------ #

    def _save_figure_multi_formats(
        self,
        fig: plt.Figure,
        base_path: str,
        *,
        png_dpi: int = 300,
    ) -> List[str]:
        for ext in ("pdf", "svg"):
            stale_path = Path(f"{base_path}.{ext}")
            if stale_path.exists():
                stale_path.unlink()

        png_path = f"{base_path}.png"
        fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
        return [png_path]

    # ------------------------------------------------------------------ #
    # 网格搜索参数空间                                                     #
    # ------------------------------------------------------------------ #

    def _get_param_grid(self) -> Dict[str, Any]:
        default_grid: Dict[str, Any] = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
        }
        return self.train_cfg.get("xgb_grid", self.train_cfg.get("model_grid", default_grid))


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    try:
        from model_support import DEFAULT_CONFIG_PATH
    except ModuleNotFoundError:
        from ml.training.config import DEFAULT_CONFIG_PATH

    parser = argparse.ArgumentParser(description="使用配置文件训练 XGB 模型")
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH),
        help=f"YAML 配置路径（默认: {DEFAULT_CONFIG_PATH}）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        from model_registry import run_training
    except ModuleNotFoundError:
        from ml.train import run_training

    model = XGBModel(args.config)
    try:
        from model_support import _to_text
    except ModuleNotFoundError:
        from ml.training.config import _to_text

    train_mode = _to_text(model.train_cfg.get("train_mode")).lower()
    if not train_mode:
        raise KeyError(f"配置缺少 train.train_mode: {args.config}")

    run_training(model, train_mode)
