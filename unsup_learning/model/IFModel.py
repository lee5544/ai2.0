import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple, List

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
    roc_curve, auc, roc_auc_score, make_scorer
)

# 如果你的项目结构需要：
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ultis.LabelMapBuilder import LabelMapBuilder

class IFModel:
    """
    孤立森林（Isolation Forest）异常检测模型
    - 二分类：normal(0) vs anomaly(1)
    - 默认仅用 normal 样本训练（fit_on='normal_only'）
    """

    def __init__(self, config_path: str):
        # 1) 读取配置
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.line_name = cfg["line_name"]

        self.model_id = (
            self.line_name + "_"
            + cfg["model"]["model_name"] + "_"
            + cfg["train"]["model_type"]
        )
        self.results_path = cfg["results_path"] + self.model_id + "/"

        self.dataset_files = [
            str(p) for p in Path(self.results_path).rglob("*.csv")
            if p.is_file()
            and not p.name.startswith(".")
            and "features" in p.name.lower()
        ]

        self.feature_start_col = 6

        self.labeler = LabelMapBuilder(cfg["train"]["label_config"])

        # 训练策略：仅正常 / 全部
        self.fit_on = cfg.get("train", {}).get("fit_on", "normal_only")

        # 预测阈值（默认 0，基于 sklearn 的 decision_function）
        # decision_function > threshold => normal；否则 anomaly
        self.score_threshold = cfg.get("predictor", {}).get("score_threshold", 0.0)

        # 写出用于推理的配置（带二分类映射与阈值）
        predictor_cfg = {
            'line_name': self.line_name,
            'model_id': self.model_id,
            'results_path': self.results_path,
            'predictor': {
                'threshold': float(self.score_threshold),
                'fit_on': self.fit_on
            }
        }
        cfg_out = os.path.join(self.results_path, "config.yaml")
        os.makedirs(self.results_path, exist_ok=True)
        with open(cfg_out, 'w', encoding='utf-8') as f:
            yaml.dump(predictor_cfg, f, allow_unicode=True, sort_keys=False)

        # 2) 构造 IF 模型（可从 cfg 中覆盖）
        mcfg = cfg.get("model", {})
        self.model = IsolationForest(
            n_estimators=mcfg.get("n_estimators", 100),
            max_samples=mcfg.get("max_samples", "auto"),
            contamination=mcfg.get("contamination", "auto"),  # 若你知道异常率，可设定 0.01/0.05 等
            max_features=mcfg.get("max_features", 0.8),
            bootstrap=mcfg.get("bootstrap", False),
            random_state=mcfg.get("random_state", 42),
            n_jobs=mcfg.get("n_jobs", -1),
            verbose=mcfg.get("verbose", 0),
            warm_start=mcfg.get("warm_start", False),
        )

        print(self.model)  # 打印模型参数

    # -----------------------
    # 数据 & 特征名
    # -----------------------
    def _load_and_sample_data(self) -> pd.DataFrame:
        dfs = []
        for path in self.dataset_files:
            try:
                dfs.append(pd.read_csv(path))
            except Exception as e:
                print(f"❌ 读取失败 {path}: {e}")
        if not dfs:
            raise RuntimeError("无可用特征文件！")
        df = pd.concat(dfs, ignore_index=True)

        # 仅保留配置中出现过的标签
        valid_labels = list(self.labeler.label_sample_dict.keys())
        df = df[df["label"].isin(valid_labels)]

        # 与 XGB 一致：按各类样本数配置做抽样（为了公平评估）
        out = []
        for label, n in self.labeler.label_sample_dict.items():
            sub = df[df["label"] == label]
            if n < 0:
                sampled = sub.copy()
            elif len(sub) >= n:
                sampled = sub.sample(n=n, random_state=42)
            else:  # 上采样：先取全体，再随机复制补足
                need_extra = n - len(sub)
                sampled = sub.copy() # 先取全部
                extra = sub.sample(n=need_extra, replace=True, random_state=42) # 再从 sub 中随机采样 need_extra 个，有放回
                sampled = pd.concat([sampled, extra], ignore_index=True)
            out.append(sampled)
            print(f"类别 {label}: {len(sub)} → 采样 {n}")
        
        return pd.concat(out).reset_index(drop=True)

    def _get_feature_names(self, df: pd.DataFrame) -> List[str]:
        return list(df.columns[self.feature_start_col:])

    # -----------------------
    # 标签二值化：normal(0) vs anomaly(1)
    # -----------------------
    def _infer_normal_label(self, df: pd.DataFrame) -> int:
        """
        从 LabelMapBuilder 的 mlabel->mtype 中优先找 'normal/ok/正常/良品/healthy'；
        若找不到，使用 df 中样本最多的 label 作为 normal。
        """
        # mlabel -> mtype (字符串)
        mm = getattr(self.labeler, "mlabel_mtype_dict", {})
        # 优先规则（大小写不敏感）
        normals = {'normal', 'ok', 'healthy', '正常', '良品'}
        for mlabel, mtype in mm.items():
            if str(mtype).strip().casefold() in normals:
                return int(mlabel)

        # 回退：使用样本最多的标签
        counts = df["label"].value_counts()
        if len(counts) == 0:
            raise RuntimeError("数据集中没有 label 列或为空")
        return int(counts.idxmax())

    def _to_binary_labels(self, df: pd.DataFrame, normal_mlabel: int) -> np.ndarray:
        """
        将 df['label'] -> 二值标签：normal(0), anomaly(1)
        先把 label -> mlabel（项目已有映射），再与 normal_mlabel 比较
        """
        mlabels = df["label"].map(self.labeler.label_to_mlabel).astype(int).values
        y_bin = (mlabels != int(normal_mlabel)).astype(int)
        return y_bin

    # -----------------------
    # 评估 & 可视化
    # -----------------------
    def _evaluate_binary(self, y_true: np.ndarray, y_pred: np.ndarray, title: str):
        import platform
        if platform.system() == "Darwin":
            plt.rcParams['font.family'] = 'Arial Unicode MS'

        # 标签顺序：normal=0, anomaly=1
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["normal", "anomaly"])
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(title)

        dir_path = os.path.join(self.results_path, "eval")
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"{title}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"—— {title} ——")
        print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"], zero_division=0))

    def _plot_roc_binary(self, y_true: np.ndarray, anomaly_score: np.ndarray, title: str):
        """
        ROC 以“异常分数”画：分数越大越异常
        """
        fpr, tpr, _ = roc_curve(y_true, anomaly_score, pos_label=1)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{title} ROC")
        plt.legend(loc="lower right")
        plt.grid(True)

        dir_path = os.path.join(self.results_path, "eval")
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"{title}_ROC.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✔️ 已保存 {title} ROC：{path}")

    # -----------------------
    # 训练与评估
    # -----------------------
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """与 XGB 保持一致比例：test 10%，val 为剩余的 12.5%（即总 12.5%）"""
        try:
            tmp, test = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)
            train, val = train_test_split(tmp, test_size=0.125, stratify=tmp["label"], random_state=42)
        except ValueError as e:
            raise RuntimeError(f"数据过少无法分割: {e}")
        return train, val, test

    def train_and_evaluate(self):
        # normal_df, abnormal_df = self._load_and_sample_data()
        df = self._load_and_sample_data()

        normal_mlabel = self._infer_normal_label(df)

        train, val, test = self._split_data(df)
        X_train = train.iloc[:, self.feature_start_col:].values
        X_val   = val.iloc[:,   self.feature_start_col:].values
        X_test  = test.iloc[:,  self.feature_start_col:].values

        y_train = self._to_binary_labels(train, normal_mlabel)
        y_val   = self._to_binary_labels(val,   normal_mlabel)
        y_test  = self._to_binary_labels(test,  normal_mlabel)

        print("数据分割：", X_train.shape, X_val.shape, X_test.shape)

        # 训练集选择：仅 normal 或全部
        if self.fit_on == "normal_only":
            X_fit = X_train[y_train == 0]

            # 把 train 中的异常样本(y_train == 1)分流到 val/test
            X_ab_train = X_train[y_train == 1]
            y_ab_train = y_train[y_train == 1]

            if len(X_ab_train) > 0:
                # 一半给 val，一半给 test
                split_idx = len(X_ab_train) // 2
                X_val = np.vstack([X_val, X_ab_train[:split_idx]])
                y_val = np.concatenate([y_val, y_ab_train[:split_idx]])
                X_test = np.vstack([X_test, X_ab_train[split_idx:]])
                y_test = np.concatenate([y_test, y_ab_train[split_idx:]])

            print(f"额外加入 {len(X_ab_train)} 条异常样本到 val/test")
            print(f"训练仅用 normal 样本：{X_fit.shape[0]} / {X_train.shape[0]}")
        else:
            X_fit = X_train
        
        # 拟合模型
        self.model.fit(X_fit)

        # 保存模型
        self._save_model()
        self._print_model_info()

        # —— 评估（train/val/test）——
        for split_name, X, y in [
            ("train", X_train, y_train),
            ("validation", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            # decision_function：越大越正常；我们用 -score 作为“异常分数”
            score = self.model.decision_function(X)
            anomaly_score = -score

            # 使用阈值 -> 预测标签（0 normal / 1 anomaly）
            y_pred = (score <= self.score_threshold).astype(int)

            self._evaluate_binary(y, y_pred, split_name)
            self._plot_roc_binary(y, anomaly_score, split_name)

    # -----------------------
    # 交叉验证（基于 ROC AUC）
    # -----------------------
    def train_and_cross_validate(self, n_splits=5):
        df = self._load_and_sample_data()
        normal_mlabel = self._infer_normal_label(df)

        X = df.iloc[:, self.feature_start_col:].values
        y = self._to_binary_labels(df, normal_mlabel)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 自定义打分：使用 decision_function 的负号作为异常分数
        def _roc_auc_from_estimator(est: IsolationForest, Xv, yv):
            s = est.decision_function(Xv)  # 越大越正常
            return roc_auc_score(yv, -s)   # 越大越异常 -> AUC

        scoring = {"roc_auc": make_scorer(_roc_auc_from_estimator, greater_is_better=True)}

        # 按 fold 训练：若只用 normal 训练，则每折 fit 数据需过滤
        results = []
        for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]

            model = IsolationForest(**self.model.get_params())
            if self.fit_on == "normal_only":
                X_fit = X_tr[y_tr == 0]
            else:
                X_fit = X_tr
            model.fit(X_fit)

            s_te = model.decision_function(X_te)
            auc_te = roc_auc_score(y_te, -s_te)
            results.append(auc_te)
            print(f"Fold {fold}: ROC AUC = {auc_te:.4f}")

        print(f"\n—— {n_splits}-折 ROC AUC ——")
        arr = np.array(results, dtype=float)
        print(f"ROC AUC: {arr.mean():.3f} ± {arr.std():.3f}")

        # 全量（按配置）重训并保存
        if self.fit_on == "normal_only":
            self.model.fit(X[y == 0])
        else:
            self.model.fit(X)
        self._save_model()

    # -----------------------
    # 网格搜索（基于 ROC AUC）
    # -----------------------
    def train_and_evaluate_grid(self):
        df = self._load_and_sample_data()
        normal_mlabel = self._infer_normal_label(df)

        train, val, test = self._split_data(df)
        X_train = train.iloc[:, self.feature_start_col:].values
        X_test  = test.iloc[:,  self.feature_start_col:].values

        y_train = self._to_binary_labels(train, normal_mlabel)
        y_test  = self._to_binary_labels(test,  normal_mlabel)

        print("数据分割：", X_train.shape, X_test.shape)

        # 拟合集合（若只用 normal 训练，则在 cv 的 fit 阶段过滤）
        # 我们自定义一个 Estimator 包装，使 GridSearchCV 在 fit(X,y) 时能按照策略过滤
        class IFWrapper(IsolationForest):
            def __init__(self, fit_on="normal_only", **kwargs):
                super().__init__(**kwargs)
                self.fit_on = fit_on

            def fit(self, X, y=None):
                if self.fit_on == "normal_only" and y is not None:
                    X = X[(y == 0)]
                return super().fit(X, y=None)

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_samples":  ["auto", 256, 512],
            "contamination": ["auto", 0.01, 0.05, 0.1],
            "max_features": [0.8, 1.0],
        }

        # 基于 decision_function 的 ROC AUC
        def roc_auc_est(est: IFWrapper, Xv, yv):
            s = est.decision_function(Xv)
            return roc_auc_score(yv, -s)

        grid = GridSearchCV(
            IFWrapper(fit_on=self.fit_on, **self.model.get_params()),
            param_grid=param_grid,
            scoring=make_scorer(roc_auc_est, greater_is_better=True),
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)
        print(f"最佳参数: {grid.best_params_}, CV ROC AUC={grid.best_score_:.4f}")

        # 用最优模型评估 test
        self.model = grid.best_estimator_
        self._save_model()
        self._print_model_info()

        s_test = self.model.decision_function(X_test)
        y_pred = (s_test <= self.score_threshold).astype(int)
        self._evaluate_binary(y_test, y_pred, "Test")
        self._plot_roc_binary(y_test, -s_test, "Test")

    # -----------------------
    # 工具
    # -----------------------
    def _save_model(self):
        path = os.path.join(self.results_path, "model.pkl")
        os.makedirs(self.results_path, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"✅ 模型已保存：{path}")

    def _print_model_info(self):
        print("\n===== 模型信息 =====")
        print(self.model)
        print("====================")


if __name__ == "__main__":
    # 示例：与你的 XGB 脚本使用方式一致，只需在 YAML 里把
    # model.model_type 改成 iforest（或任意），其余配置保持
    # - train.fit_on: "normal_only" 或 "all"
    # - predictor.score_threshold: 默认 0.0
    config = "epump4_if.yaml"   # 也可新建 epump4_iforest.yaml
    model  = IFModel(config)
    model.train_and_evaluate()
    # model.train_and_cross_validate(n_splits=5)
    # model.train_and_evaluate_grid()
