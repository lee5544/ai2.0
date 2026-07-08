"""LightGBM 分类模型，继承自 BaseModel（独立，不依赖 XGBModel）。

与 XGBoost 版本的差异：
  - 使用 LGBMClassifier，objective 使用 LightGBM 命名：binary / multiclass
  - fit 不传 feature_weights；LightGBM sklearn API 不支持该参数
  - early stopping 通过 callbacks 注入
  - evals_result_ 是属性而非方法
  - 特征重要性通过 model.booster_.feature_importance() 获取

主要改进：
  - 修复参数读取时 0 被 `or default` 覆盖的问题
  - eval_set 同时传入 train / val，便于绘制训练集和验证集损失曲线
  - log_evaluation(period=0) 明确关闭训练日志
  - 兼容 feature_fraction 与 colsample_bytree
  - SHAP 兼容二分类、多分类以及不同 shap 版本返回结构
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    训练、评估、导出逻辑继承自 BaseModel，仅覆盖模型相关钩子：

      - _build_classifier      → 构建 LGBMClassifier
      - _do_model_fit          → LightGBM fit，callbacks 早停，无 feature_weights
      - _plot_training_loss    → 从 model.evals_result_ 读取曲线
      - _importance_df         → 通过 booster_.feature_importance 获取重要性
      - _plot_shap_summary     → LightGBM SHAP 图
      - _get_param_grid        → 使用 lgb_grid 配置键

    YAML 参数优先级：
      extra_params > train.lgb_params > 默认值
    """

    DEFAULT_MODEL_TYPE = "lgb"

    def __init__(self, config_path: str):
        # BaseModel.__init__ 中会经 MRO 调用本类 _build_classifier()
        super().__init__(config_path)

        # BaseModel 可能默认使用 XGB 风格 objective；这里修正为 LightGBM 命名
        self.objective = "binary" if self.n_classes == 2 else "multiclass"
        self.eval_metric = "binary_logloss" if self.n_classes == 2 else "multi_logloss"

        # 使用修正后的 objective / metric 重新构建模型
        self.model = self._build_classifier()

    # ------------------------------------------------------------------ #
    # 参数读取工具                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _none_aware_int(value: Any, default: int) -> int:
        """读取 int 参数。

        注意：不能使用 `_to_int(value) or default`，否则用户显式配置 0 会被覆盖。
        """
        parsed = _to_int(value)
        return default if parsed is None else int(parsed)

    @staticmethod
    def _none_aware_float(value: Any, default: float) -> float:
        """读取 float 参数。

        注意：不能使用 `_to_float(value) or default`，否则用户显式配置 0.0 会被覆盖。
        """
        parsed = _to_float(value)
        return default if parsed is None else float(parsed)

    @staticmethod
    def _clip_float(value: float, low: float, high: float, name: str) -> float:
        """将比例型参数裁剪到指定区间，避免非法配置导致训练报错。"""
        if value < low or value > high:
            clipped = min(max(value, low), high)
            print(f"[WARN] {name}={value} 超出范围 [{low}, {high}]，已裁剪为 {clipped}。")
            return clipped
        return value

    # ------------------------------------------------------------------ #
    # 模型构建                                                            #
    # ------------------------------------------------------------------ #

    def _build_classifier(self, *, extra_params: Dict[str, Any] | None = None) -> LGBMClassifier:
        """构造 LGBMClassifier。

        参数来源：
          extra_params > train.lgb_params > 默认值
        """
        lgb_params: Dict[str, Any] = self.train_cfg.get("lgb_params", {}) or {}
        if not isinstance(lgb_params, dict):
            print("[WARN] train.lgb_params 不是字典，已忽略。")
            lgb_params = {}

        extra_params = extra_params or {}

        def _get(key: str, default: Any) -> Any:
            return extra_params.get(key, lgb_params.get(key, default))

        def _get_any(keys: list[str], default: Any) -> Any:
            """按顺序读取多个别名参数，适合兼容 feature_fraction / colsample_bytree。"""
            for key in keys:
                if key in extra_params:
                    return extra_params[key]
                if key in lgb_params:
                    return lgb_params[key]
            return default

        n_estimators = self._none_aware_int(_get("n_estimators", 400), 400)
        num_leaves = self._none_aware_int(_get("num_leaves", 20), 20)
        max_depth = self._none_aware_int(_get("max_depth", -1), -1)
        learning_rate = self._none_aware_float(_get("learning_rate", 0.05), 0.05)

        min_child_samples = self._none_aware_int(_get("min_child_samples", 20), 20)
        min_split_gain = self._none_aware_float(_get("min_split_gain", 0.0), 0.0)

        reg_lambda = self._none_aware_float(_get("reg_lambda", 5.0), 5.0)
        reg_alpha = self._none_aware_float(_get("reg_alpha", 0.5), 0.5)

        subsample = self._none_aware_float(_get("subsample", 0.8), 0.8)
        subsample = self._clip_float(subsample, 0.0, 1.0, "subsample")
        subsample_freq = self._none_aware_int(_get("subsample_freq", 1), 1)

        # 兼容 LightGBM 原生参数名 feature_fraction 与 sklearn 风格 colsample_bytree。
        # 最终传入 colsample_bytree，避免同时传两个等价参数造成混淆。
        colsample_bytree = self._none_aware_float(
            _get_any(["colsample_bytree", "feature_fraction"], 0.7),
            0.7,
        )
        colsample_bytree = self._clip_float(colsample_bytree, 0.0, 1.0, "colsample_bytree")

        random_state = self._none_aware_int(_get("random_state", self.random_state), self.random_state)
        n_jobs = self._none_aware_int(_get("n_jobs", -1), -1)
        verbose = self._none_aware_int(_get("verbose", -1), -1)

        # 可选类别不平衡处理。
        # 二分类可配置 is_unbalance 或 scale_pos_weight；多分类通常不使用这两个参数。
        is_unbalance = _get("is_unbalance", None)
        scale_pos_weight = _get("scale_pos_weight", None)

        objective = "binary" if self.n_classes == 2 else "multiclass"
        metric = "binary_logloss" if self.n_classes == 2 else "multi_logloss"

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
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        if self.n_classes > 2:
            kwargs["num_class"] = self.n_classes
        else:
            # 二分类下才注入类别不平衡参数。
            if is_unbalance is not None:
                kwargs["is_unbalance"] = bool(is_unbalance)
            if scale_pos_weight is not None:
                kwargs["scale_pos_weight"] = self._none_aware_float(scale_pos_weight, 1.0)

        early_stopping_rounds = _to_int(self.train_cfg.get("early_stopping_rounds"))
        if early_stopping_rounds is not None and early_stopping_rounds > 0:
            self._lgb_early_stopping_rounds = int(early_stopping_rounds)
            print(f"[INFO] LGB 早停: early_stopping_rounds={self._lgb_early_stopping_rounds}")
        else:
            self._lgb_early_stopping_rounds = 0

        return LGBMClassifier(**kwargs)

    # ------------------------------------------------------------------ #
    # Fit 钩子                                                            #
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
        """LightGBM fit。

        说明：
          - LightGBM sklearn API 不接收 XGBoost 风格 feature_weights
          - eval_set 同时传入 train / val，便于绘制训练损失和验证损失
          - early stopping 通过 callbacks 控制
        """
        if feature_weights is not None:
            print("[INFO] LGB 不支持 feature_weights，已忽略。")

        callbacks: list = [lgb.log_evaluation(period=0)]

        early_stop = getattr(self, "_lgb_early_stopping_rounds", 0)
        if early_stop > 0:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=early_stop,
                    verbose=False,
                )
            )

        self.model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=["train", "val"],
            callbacks=callbacks,
        )

    # ------------------------------------------------------------------ #
    # 训练损失曲线                                                        #
    # ------------------------------------------------------------------ #

    def _plot_training_loss(self, *, title: str = "training_loss") -> List[str]:
        """绘制 LightGBM 训练曲线。

        读取 self.model.evals_result_，保存多格式图片与 CSV。
        """
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

            dataset_label = str(dataset)

            for metric, values in metrics.items():
                curve: List[float] = []
                for x in values:
                    try:
                        curve.append(float(x))
                    except (TypeError, ValueError):
                        curve.append(float("nan"))

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

        best_iter = getattr(self.model, "best_iteration_", None)
        if best_iter is not None and isinstance(best_iter, int) and best_iter > 0:
            ax.axvline(best_iter, linestyle="--", linewidth=1.2, alpha=0.8, label=f"best_iter={best_iter}")

        ax.set_xlabel("Round")
        ax.set_ylabel("Loss")
        ax.set_title("Training / Validation Loss (LGB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        base_path = os.path.join(out_dir, title)
        saved_paths = self._save_figure_multi_formats(fig, base_path, png_dpi=300)
        plt.close(fig)

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
    # 特征重要性                                                          #
    # ------------------------------------------------------------------ #

    def _importance_df(self, importance_type: str, feature_names: list) -> pd.DataFrame:
        """返回特征重要性 DataFrame。

        LightGBM 常用 importance_type：
          - "split": 特征被用于分裂的次数
          - "gain":  特征带来的总增益
        """
        if not hasattr(self.model, "booster_"):
            raise RuntimeError("模型尚未训练，无法读取 booster_ 特征重要性。")

        booster = self.model.booster_

        try:
            imp_values = booster.feature_importance(importance_type=importance_type)
        except Exception as exc:
            print(f"[WARN] importance_type={importance_type!r} 不可用，回退到 'gain'。原始错误：{exc}")
            imp_values = booster.feature_importance(importance_type="gain")

        if len(imp_values) != len(feature_names):
            print(
                f"[WARN] 特征重要性长度({len(imp_values)})与特征名长度({len(feature_names)})不一致，"
                "将按最短长度截断。"
            )
            n = min(len(imp_values), len(feature_names))
            imp_values = imp_values[:n]
            feature_names = feature_names[:n]

        return (
            pd.Series(imp_values, index=feature_names, name="importance")
            .sort_values(ascending=False)
            .to_frame()
        )

    # ------------------------------------------------------------------ #
    # SHAP                                                                #
    # ------------------------------------------------------------------ #

    def _select_shap_values_for_plot(self, shap_values: Any) -> Any:
        """兼容不同 shap / LightGBM 分类返回结构。

        可能返回：
          - ndarray: 直接用于绘图
          - list[ndarray]: 二分类取正类，多分类保留 list 或按 shap 支持情况绘制
        """
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                # 二分类：通常取正类解释更直观
                return shap_values[1]
            return shap_values

        return shap_values

    def _plot_shap_summary(
        self,
        X: np.ndarray,
        df_for_names: pd.DataFrame,
        title: str = "feature_importance_shap",
    ) -> None:
        """绘制 SHAP summary 图。

        输出：
          - bar 图
          - beeswarm 图
        """
        try:
            import shap
        except ImportError as exc:
            print(f"ℹ️ 未安装 shap，跳过 SHAP 绘图：{exc}")
            return

        if X is None or X.shape[0] == 0:
            print("[WARN] X 为空，跳过 SHAP 绘图。")
            return

        feature_names = (
            list(self.feature_columns)
            if self.feature_columns
            else self._resolve_feature_columns(df_for_names)
        )

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
        shap_values_to_plot = self._select_shap_values_for_plot(shap_values)

        out_dir = os.path.join(self._output_root(), "eval", "importance")
        os.makedirs(out_dir, exist_ok=True)

        for plot_type, suffix in (("bar", "_bar"), (None, "_beeswarm")):
            try:
                plt.figure()
                shap.summary_plot(
                    shap_values_to_plot,
                    features=X_shap,
                    feature_names=feature_names,
                    plot_type=plot_type,
                    show=False,
                )
                base = os.path.join(out_dir, f"{title}{suffix}")
                plt.tight_layout()
                for path in self._save_figure_multi_formats(plt.gcf(), base):
                    print(f"✔️ 已保存 SHAP 图：{path}")
                plt.close()
            except Exception as exc:
                plt.close()
                print(f"[WARN] SHAP {suffix} 图绘制失败，已跳过：{exc}")

    # ------------------------------------------------------------------ #
    # 网格搜索参数空间                                                    #
    # ------------------------------------------------------------------ #

    def _get_param_grid(self) -> Dict[str, Any]:
        """返回 LightGBM 参数搜索空间。"""
        default_grid: Dict[str, Any] = {
            "n_estimators": [300, 600, 1000],
            "learning_rate": [0.01, 0.03, 0.05],
            "num_leaves": [31, 63, 127],
            "min_child_samples": [20, 50, 100],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "subsample": [0.7, 0.8, 1.0],
            "reg_alpha": [0.0, 0.5, 1.0],
            "reg_lambda": [1.0, 5.0, 10.0],
        }

        grid = self.train_cfg.get("lgb_grid", self.train_cfg.get("model_grid", default_grid))
        if not isinstance(grid, dict):
            print("[WARN] lgb_grid/model_grid 不是字典，使用默认 LGB 参数网格。")
            return default_grid

        return grid


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="使用配置文件训练 LightGBM 模型")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
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
