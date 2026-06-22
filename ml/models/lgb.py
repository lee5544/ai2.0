"""LightGBM 分类模型，继承自 BaseModel（独立，不依赖 XGBModel）。

与 XGBoost 版本的差异：
  - 使用 LGBMClassifier，objective 用 LGB 命名（binary / multiclass）
  - fit 不传 feature_weights（LGB 不支持）；早停通过 callbacks 注入
  - evals_result_ 是属性而非方法
  - 特征重要性通过 model.booster_.feature_importance() 获取
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Any, Dict, List

import lightgbm as lgb
from lightgbm import LGBMClassifier

try:
    from model_support import DEFAULT_CONFIG_PATH, _to_int, _to_float, _to_text
except ModuleNotFoundError:
    from ml.training.config import DEFAULT_CONFIG_PATH, _to_int, _to_float, _to_text

try:
    from ml.models.base import BaseModel
except ModuleNotFoundError:
    from base_model import BaseModel


class LGBModel(BaseModel):
    """LightGBM 分类模型。

    全部训练、评估、导出逻辑继承自 BaseModel，只覆盖以下模型特有方法：
      - _build_classifier      → 构建 LGBMClassifier
      - _do_model_fit          → LGB fit（callbacks 早停，无 feature_weights）
      - _plot_training_loss    → 从 model.evals_result_（属性）读取曲线
      - _importance_df         → 通过 booster_.feature_importance 获取
      - _plot_shap_summary     → LGB SHAP 图（TreeExplainer 同样支持）
      - _get_param_grid        → 使用 lgb_grid 配置键
    """

    DEFAULT_MODEL_TYPE = "lgb"

    def __init__(self, config_path: str):
        # 调用 BaseModel.__init__；其中 _build_classifier() 经 MRO 调用本类覆盖版本
        super().__init__(config_path)
        # BaseModel.__init__ 用 XGB objective 占位；修正为 LGB 命名并重建模型
        self.objective = "binary" if self.n_classes == 2 else "multiclass"
        self.eval_metric = "binary_logloss" if self.n_classes == 2 else "multi_logloss"
        self.model = self._build_classifier()

    # ------------------------------------------------------------------ #
    # 模型构建                                                             #
    # ------------------------------------------------------------------ #

    def _build_classifier(self, *, extra_params: Dict[str, Any] | None = None) -> LGBMClassifier:
        """构造 LGBMClassifier。参数来源：extra_params > lgb_params（YAML）> 默认值。"""
        lgb_params: Dict[str, Any] = self.train_cfg.get("lgb_params", {}) or {}
        if not isinstance(lgb_params, dict):
            print("[WARN] train.lgb_params 不是字典，已忽略。")
            lgb_params = {}

        extra_params = extra_params or {}

        def _get(key: str, default: Any) -> Any:
            return extra_params.get(key, lgb_params.get(key, default))

        n_estimators     = int(_to_int(_get("n_estimators", 400)) or 400)
        num_leaves       = int(_to_int(_get("num_leaves", 63)) or 63)
        max_depth        = int(_to_int(_get("max_depth", -1)) or -1)
        learning_rate    = float(_to_float(_get("learning_rate", 0.05)) or 0.05)
        min_child_samples = int(_to_int(_get("min_child_samples", 20)) or 20)
        min_split_gain   = float(_to_float(_get("min_split_gain", 0.0)) or 0.0)
        reg_lambda       = float(_to_float(_get("reg_lambda", 5.0)) or 5.0)
        reg_alpha        = float(_to_float(_get("reg_alpha", 0.5)) or 0.5)
        subsample        = float(_to_float(_get("subsample", 0.8)) or 0.8)
        subsample_freq   = int(_to_int(_get("subsample_freq", 1)) or 1)
        feature_fraction = float(_to_float(_get("feature_fraction", 0.7)) or 0.7)
        random_state     = int(_to_int(_get("random_state", self.random_state)) or self.random_state)
        n_jobs           = int(_to_int(_get("n_jobs", -1)) or -1)
        verbose          = int(_to_int(_get("verbose", -1)) or -1)

        objective = "binary" if self.n_classes == 2 else "multiclass"
        metric    = "binary_logloss" if self.n_classes == 2 else "multi_logloss"

        kwargs: Dict[str, Any] = dict(
            objective=objective,
            metric=metric,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            min_split_gain=min_split_gain,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            subsample=subsample,
            subsample_freq=subsample_freq,
            feature_fraction=feature_fraction,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        if self.n_classes > 2:
            kwargs["num_class"] = self.n_classes

        early_stopping_rounds = _to_int(self.train_cfg.get("early_stopping_rounds"))
        if early_stopping_rounds is not None and early_stopping_rounds > 0:
            self._lgb_early_stopping_rounds = int(early_stopping_rounds)
            print(f"[INFO] LGB 早停: early_stopping_rounds={self._lgb_early_stopping_rounds}")
        else:
            self._lgb_early_stopping_rounds = 0

        return LGBMClassifier(**kwargs)

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
        """LightGBM fit：eval_set 仅传验证集，通过 callbacks 控制早停与日志。"""
        if feature_weights is not None:
            print("[INFO] LGB 不支持 feature_weights，已忽略。")

        callbacks: list = [lgb.log_evaluation(period=-1)]
        early_stop = getattr(self, "_lgb_early_stopping_rounds", 0)
        if early_stop > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stop, verbose=False))

        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )

    # ------------------------------------------------------------------ #
    # 训练损失曲线（evals_result_ 是属性）                                #
    # ------------------------------------------------------------------ #

    def _plot_training_loss(self, *, title: str = "training_loss") -> List[str]:
        dataset_name_map = {
            "valid_0": "val",
            "training": "train",
        }

        try:
            evals_result: Dict[str, Any] = self.model.evals_result_
        except AttributeError:
            return []

        if not evals_result:
            return []

        series_items: List[tuple[str, List[float]]] = []
        for dataset, metrics in evals_result.items():
            if not isinstance(metrics, dict):
                continue
            dataset_label = dataset_name_map.get(str(dataset), str(dataset))
            for metric, values in metrics.items():
                curve = [float(x) if isinstance(x, (int, float)) else float("nan") for x in values]
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
                y = np.concatenate([y, np.full(max_len - len(y), float("nan"))])
            ax.plot(rounds, y, linewidth=1.8, label=name)

        ax.set_xlabel("Round")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss (LGB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        base_path = os.path.join(out_dir, title)
        saved_paths = self._save_figure_multi_formats(fig, base_path, png_dpi=300)
        plt.close()

        csv_data: Dict[str, Any] = {"round": rounds}
        for name, values in series_items:
            col = np.asarray(values, dtype=float)
            if len(col) < max_len:
                col = np.concatenate([col, np.full(max_len - len(col), float("nan"))])
            csv_data[name] = col

        loss_df = pd.DataFrame(csv_data)
        csv_path = f"{base_path}.csv"
        loss_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        saved_paths.append(csv_path)

        for path in saved_paths:
            print(f"✔️ 已保存训练损失：{path}")
        return saved_paths

    # ------------------------------------------------------------------ #
    # 特征重要性（LGB API 不同于 XGB）                                    #
    # ------------------------------------------------------------------ #

    def _importance_df(self, importance_type: str, feature_names: list) -> pd.DataFrame:
        booster = self.model.booster_
        imp_values = booster.feature_importance(importance_type=importance_type)
        return (
            pd.Series(imp_values, index=feature_names, name="importance")
            .sort_values(ascending=False)
            .to_frame()
        )

    # ------------------------------------------------------------------ #
    # SHAP（TreeExplainer 同样支持 LGB）                                  #
    # ------------------------------------------------------------------ #

    def _plot_shap_summary(
        self,
        X: np.ndarray,
        df_for_names: pd.DataFrame,
        title: str = "feature_importance_shap",
    ) -> None:
        try:
            import shap
        except ImportError as exc:
            print(f"ℹ️ 未安装 shap，跳过SHAP绘图：{exc}")
            return

        feature_names = list(self.feature_columns) if self.feature_columns else self._resolve_feature_columns(df_for_names)

        shap_max_samples = _to_int(self.train_cfg.get("shap_max_samples"))
        if shap_max_samples is None or shap_max_samples <= 0:
            shap_max_samples = min(2000, X.shape[0])

        if X.shape[0] > shap_max_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(X.shape[0], size=shap_max_samples, replace=False)
            X_shap = X[idx]
            print(f"[INFO] SHAP 样本抽样: {X.shape[0]} -> {X_shap.shape[0]}")
        else:
            X_shap = X

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_shap)

        out_dir = os.path.join(self._output_root(), "eval/importance")
        os.makedirs(out_dir, exist_ok=True)

        for plot_type, suffix in (("bar", "_bar"), (None, "_beeswarm")):
            plt.figure()
            shap.summary_plot(
                shap_values, features=X_shap, feature_names=feature_names,
                plot_type=plot_type, show=False,
            )
            base = os.path.join(out_dir, f"{title}{suffix}")
            plt.tight_layout()
            for path in self._save_figure_multi_formats(plt.gcf(), base):
                print(f"✔️ 已保存SHAP图：{path}")
            plt.close()

    # ------------------------------------------------------------------ #
    # 网格搜索参数空间                                                     #
    # ------------------------------------------------------------------ #

    def _get_param_grid(self) -> Dict[str, Any]:
        default_grid: Dict[str, Any] = {
            "n_estimators": [50, 100, 200],
            "num_leaves": [31, 63, 127],
            "learning_rate": [0.01, 0.05, 0.1],
            "feature_fraction": [0.7, 0.8, 1.0],
        }
        return self.train_cfg.get("lgb_grid", self.train_cfg.get("model_grid", default_grid))


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="使用配置文件训练 LGB 模型")
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

    model = LGBModel(args.config)
    run_training(model, train_mode="normal")
