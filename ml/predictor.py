"""ML 推理预测器。

包含：
  ChannelPredictor  — 加载 model.pkl + predict_config.yaml，推理单信号/TDMS
  CLI main          — 命令行 TDMS 推理（原 inference/cli.py）

同时兼容两种运行模式：
  - 包模式：from ml.runtime import predict_with_threshold_rule
  - 独立运行包（runtime/）: from prediction_logic import predict_with_threshold_rule
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

try:
    from ml.features import DEFAULT_FEATURE_VERSION, get_feature_extractor, resolve_feature_version
    from ml.runtime import predict_with_threshold_rule
except ModuleNotFoundError:  # pragma: no cover — standalone runtime bundle
    from features import DEFAULT_FEATURE_VERSION, get_feature_extractor, resolve_feature_version  # type: ignore[import]
    from prediction_logic import predict_with_threshold_rule  # type: ignore[import]


class ChannelPredictor:
    def __init__(self, model_id: str | None = None, model_dir: str | os.PathLike | None = None):
        """
        model_id: results 下的模型目录名
        model_dir: 模型目录绝对/相对路径
        """
        if model_dir is not None:
            self.results_path = Path(model_dir).expanduser().resolve()
        else:
            if not model_id:
                raise ValueError("model_id 与 model_dir 至少提供一个")
            self.results_path = (Path("./results") / model_id).expanduser().resolve()

        if not self.results_path.exists():
            raise FileNotFoundError(f"模型目录不存在: {self.results_path}")

        self.model = None
        self.mlabel_mtype_dict: dict[int, str] = {}
        self.mlabel_threshold_dict: dict[int, float] = {}
        self.ok_mlabel = 0
        self.feature_columns: list[str] = []
        self.feature_version = DEFAULT_FEATURE_VERSION
        self.extract_features = get_feature_extractor(self.feature_version)

        self.time_thr = 50
        self.mean_thr = 10

        predict_cfg_path = self.results_path / "predict_config.yaml"
        if not predict_cfg_path.exists():
            raise FileNotFoundError(f"预测配置文件不存在: {predict_cfg_path}")

        with predict_cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.feature_version = resolve_feature_version(cfg)
        self.extract_features = get_feature_extractor(self.feature_version)

        self.model = self._load_model(self.results_path / "model.pkl")

        label_mapping = cfg["predictor"]["label_mapping"]
        self.mlabel_mtype_dict = {int(item["mlabel"]): item["mtype"] for item in label_mapping}
        print("mlabel_mtype_dict: ", self.mlabel_mtype_dict)

        self.mlabel_threshold_dict = {
            int(item["mlabel"]): float(item.get("threshold", 0.7)) for item in label_mapping
        }
        print("mlabel_threshold_dict: ", self.mlabel_threshold_dict)
        self.ok_mlabel = int(cfg.get("predictor", {}).get("ok_mlabel", 0))
        print("ok_mlabel: ", self.ok_mlabel)

        feature_columns = cfg.get("predictor", {}).get("feature_columns", [])
        self.feature_columns = [str(name) for name in feature_columns if str(name)] if isinstance(feature_columns, list) else []
        if self.feature_columns:
            print(f"feature_columns: 使用 {len(self.feature_columns)} 个筛选特征")

    def _load_model(self, model_file: str | os.PathLike):
        model_file = Path(model_file)
        with model_file.open("rb") as f:
            model = pickle.load(f)

        print("\n===== 模型信息 =====")
        print(f"模型类型: {type(model).__name__}")
        try:
            if hasattr(model, "n_features_in_"):
                print(f"输入特征数: {model.n_features_in_}")
            if hasattr(model, "classes_"):
                print(f"分类类别: {model.classes_}")
            if hasattr(model, "get_params"):
                print("\n模型参数:")
                for param, value in model.get_params().items():
                    print(f"  {param}: {value}")
        except Exception as e:
            print(f"无法获取详细参数: {e}")
        print("====================\n")
        return model

    def _build_feature_vector(self, features: dict) -> np.ndarray:
        if not self.feature_columns:
            return np.fromiter(features.values(), dtype=float).reshape(1, -1)
        missing = [name for name in self.feature_columns if name not in features]
        if missing:
            preview = ", ".join(missing[:10])
            suffix = " ..." if len(missing) > 10 else ""
            raise ValueError(f"预测特征缺失 {len(missing)} 个: {preview}{suffix}")
        return np.array(
            [float(features[name]) for name in self.feature_columns], dtype=float
        ).reshape(1, -1)

    def _sensor_error_detail(self) -> dict:
        return {
            "result": "NOK",
            "mtype": "传感器错误",
            "mlabel": None,
            "score": 1.0,
            "raw_mtype": "传感器错误",
            "raw_mlabel": None,
            "raw_score": 1.0,
            "scores": {"传感器错误": 1.0},
        }

    def predict_detail(self, channel_raw: np.ndarray, sr: int, up_or_down: str) -> dict:
        """返回预测结果、最终类别分数及全部类别概率。"""
        if channel_raw is None or len(channel_raw) == 0:
            return self._sensor_error_detail()
        if len(channel_raw) > sr * self.time_thr:
            return self._sensor_error_detail()
        if np.mean(channel_raw) > self.mean_thr:
            return self._sensor_error_detail()

        features = self.extract_features(channel_raw, sr)
        X_features = self._build_feature_vector(features)

        proba = self.model.predict_proba(X_features)[0]
        classes = np.array(self.model.classes_, dtype=int)
        mlabel = int(
            predict_with_threshold_rule(
                proba.reshape(1, -1),
                classes,
                self.mlabel_threshold_dict,
                ok_mlabel=self.ok_mlabel,
            )[0]
        )
        mtype = self.mlabel_mtype_dict[mlabel]
        result = "OK" if mlabel == self.ok_mlabel else "NOK"
        scores = {
            self.mlabel_mtype_dict[int(label)]: float(score)
            for label, score in zip(classes, proba)
        }
        selected_index = int(np.where(classes == mlabel)[0][0])
        raw_index = int(np.argmax(proba))
        raw_mlabel = int(classes[raw_index])

        return {
            "result": result,
            "mtype": mtype,
            "mlabel": mlabel,
            "score": float(proba[selected_index]),
            "raw_mtype": self.mlabel_mtype_dict[raw_mlabel],
            "raw_mlabel": raw_mlabel,
            "raw_score": float(proba[raw_index]),
            "scores": scores,
        }

    def predict(self, channel_raw: np.ndarray, sr: int, up_or_down: str) -> tuple[str, str]:
        detail = self.predict_detail(channel_raw, sr, up_or_down)
        return detail["result"], detail["mtype"]

    def predict_signal(self, signal: np.ndarray, sampling_rate: int) -> dict:
        return self.predict_detail(signal, sampling_rate, "signal")

    def predict_tdms(self, tdms_path: str | os.PathLike, *, line: str | None = None) -> dict:
        try:
            from data_manager.tdms_read import read_tdms
        except ModuleNotFoundError:  # pragma: no cover — standalone runtime bundle
            from tdms_read import read_tdms  # type: ignore[import]

        tdms = read_tdms(tdms_path, line=line)
        sampling_rate = tdms.get("sampling_rate")
        if not sampling_rate:
            raise ValueError(f"TDMS sampling rate is missing: {tdms_path}")
        return {
            "up": self.predict_detail(tdms.get("up_data"), sampling_rate, "up"),
            "down": self.predict_detail(tdms.get("down_data"), sampling_rate, "down"),
            "sampling_rate": int(sampling_rate),
        }

    def set_threshold(self, threshold: float) -> None:
        self.setThreshold(threshold)

    def set_thresholds(self, threshold_by_label: dict[int, float]) -> None:
        self.setThresholdDict(threshold_by_label)

    def setThreshold(self, threshold: float) -> None:
        self.mlabel_threshold_dict = {k: threshold for k in self.mlabel_threshold_dict}
        print("统一设置阈值。调整后的阈值列表: ", self.mlabel_threshold_dict)

    def setThresholdDict(self, threshold_dict: dict[int, float]) -> None:
        current_keys = set(self.mlabel_threshold_dict.keys())
        incoming_keys = set(threshold_dict.keys())
        if current_keys != incoming_keys:
            raise ValueError(
                f"mlabel keys 不一致：缺失 {current_keys - incoming_keys}，多余 {incoming_keys - current_keys}"
            )
        self.mlabel_threshold_dict = threshold_dict
        print("单独设置阈值。调整后的阈值列表: ", self.mlabel_threshold_dict)


# ===========================================================================
# CLI（原 inference/cli.py）
# ===========================================================================

def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone TDMS predictor (from results bundle)")
    parser.add_argument("--tdms", required=True, help="TDMS file path")
    parser.add_argument("--line", default=None, help="line name (optional, e.g. epump4)")
    parser.add_argument("--threshold", type=float, default=None, help="global threshold override")
    parser.add_argument(
        "--threshold-dict",
        default=None,
        help="JSON dict override, e.g. '{\"0\": 0.8, \"1\": 0.7}'",
    )
    return parser.parse_args()


def _extract_features_for_probability(predictor: ChannelPredictor, channel_raw: np.ndarray, sampling_rate: int) -> dict:
    extract_features = getattr(predictor, "extract_features", None)
    if extract_features is not None:
        return extract_features(channel_raw, sampling_rate)
    from extract_features_v2 import extract_features_v2  # type: ignore[import]
    return extract_features_v2(channel_raw, sampling_rate)


def _format_label_key(predictor: ChannelPredictor, label) -> str:
    try:
        normalized_label = int(label)
    except (TypeError, ValueError):
        normalized_label = label
    mtype = getattr(predictor, "mlabel_mtype_dict", {}).get(normalized_label)
    return f"{normalized_label}:{mtype}" if mtype is not None else str(normalized_label)


def _class_probabilities(predictor: ChannelPredictor, channel_raw: np.ndarray, sampling_rate: int) -> dict:
    features = _extract_features_for_probability(predictor, channel_raw, sampling_rate)
    build_feature_vector = getattr(predictor, "_build_feature_vector", None)
    X_features = build_feature_vector(features) if callable(build_feature_vector) else np.fromiter(features.values(), dtype=float).reshape(1, -1)

    proba = predictor.model.predict_proba(X_features)[0]
    classes = getattr(predictor.model, "classes_", range(len(proba)))
    proba_by_label: dict = {}
    for label, probability in zip(classes, proba):
        try:
            normalized_label = int(label)
        except (TypeError, ValueError):
            normalized_label = label
        proba_by_label[normalized_label] = round(float(probability), 6)

    probabilities: dict = {}
    for label in getattr(predictor, "mlabel_mtype_dict", {}):
        probabilities[_format_label_key(predictor, label)] = proba_by_label.get(label, 0.0)
    for label, probability in proba_by_label.items():
        key = _format_label_key(predictor, label)
        if key not in probabilities:
            probabilities[key] = probability
    return probabilities


def main() -> None:
    args = _parse_cli_args()

    base_dir = Path(__file__).resolve().parent
    runtime_dir = base_dir / "runtime"
    if not runtime_dir.exists():
        raise FileNotFoundError(f"runtime folder missing: {runtime_dir}")

    sys.path.insert(0, str(runtime_dir))

    from ChannelPredictor import ChannelPredictor as _CP  # type: ignore[import]  # noqa: E402
    from tdms_read import read_tdms  # type: ignore[import]  # noqa: E402

    predictor = _CP(model_dir=base_dir)

    if args.threshold is not None:
        predictor.setThreshold(args.threshold)
    if args.threshold_dict:
        threshold_dict = {int(k): float(v) for k, v in json.loads(args.threshold_dict).items()}
        predictor.setThresholdDict(threshold_dict)

    tdms_data = read_tdms(args.tdms, line=args.line)
    sampling_rate = tdms_data.get("sampling_rate")
    if sampling_rate is None:
        raise ValueError(f"TDMS sampling rate is missing: {args.tdms}")

    up_result, up_name = predictor.predict(tdms_data["up_data"], sampling_rate, "up")
    down_result, down_name = predictor.predict(tdms_data["down_data"], sampling_rate, "down")
    up_prob = _class_probabilities(predictor, tdms_data["up_data"], sampling_rate)
    down_prob = _class_probabilities(predictor, tdms_data["down_data"], sampling_rate)

    print(f"up:   {up_result}, {up_name}, probabilities={json.dumps(up_prob, ensure_ascii=False)}")
    print(f"down: {down_result}, {down_name}, probabilities={json.dumps(down_prob, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
