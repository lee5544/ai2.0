#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.signal
from scipy.io import wavfile


RUNTIME_MODULES = [
    "ChannelPredictor",
    "prediction_logic",
    "features",
    "features.extract_features_v2",
    "features.extract_features_v5",
    "features.extract_features_v7",
    "features.extract_features_v10",
]

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict wav files with a packaged results model")
    parser.add_argument("--folder", required=True, help="Folder containing wav files")
    parser.add_argument("--model-dir", required=True, help="Packaged model directory under results")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--target-sr", type=int, default=20000, help="Sampling rate used for inference")
    parser.add_argument("--pattern", default="*.wav", help="File glob pattern")
    parser.add_argument("--workers", type=int, default=1, help="Number of prediction threads")
    parser.add_argument("--quiet", action="store_true", help="Only print the final summary")
    return parser.parse_args()


def _clear_runtime_modules() -> None:
    for name in RUNTIME_MODULES:
        sys.modules.pop(name, None)


@contextlib.contextmanager
def _runtime_path(runtime_dir: Path):
    _clear_runtime_modules()
    sys.path.insert(0, str(runtime_dir))
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(runtime_dir))
        _clear_runtime_modules()


def _load_predictor(model_dir: Path):
    runtime_dir = model_dir / "runtime"
    if not runtime_dir.exists():
        raise FileNotFoundError(f"runtime folder missing: {runtime_dir}")

    with _runtime_path(runtime_dir):
        predictor_module = importlib.import_module("ChannelPredictor")
        prediction_logic_module = importlib.import_module("prediction_logic")
        with contextlib.redirect_stdout(io.StringIO()):
            predictor = predictor_module.ChannelPredictor(model_dir=model_dir)
        predictor._predict_with_threshold_rule = prediction_logic_module.predict_with_threshold_rule
        return predictor


def _read_wav_mono(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    original_sr, data = wavfile.read(path)
    x = np.asarray(data)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        scale = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float32) / scale
    else:
        x = x.astype(np.float32, copy=False)

    if int(original_sr) != int(target_sr):
        gcd = int(np.gcd(int(original_sr), int(target_sr)))
        x = scipy.signal.resample_poly(x, int(target_sr) // gcd, int(original_sr) // gcd)
        x = x.astype(np.float32, copy=False)
    return x, int(original_sr)


def _direction_from_name(path: Path) -> str:
    lower_name = path.name.lower()
    if re.search(r"(^|[-_])up($|[-_.])", lower_name):
        return "up"
    if re.search(r"(^|[-_])down($|[-_.])", lower_name):
        return "down"
    return ""


def _feature_matrix(predictor: Any, features: dict[str, float]) -> np.ndarray:
    build_feature_vector = getattr(predictor, "_build_feature_vector", None)
    if callable(build_feature_vector):
        return build_feature_vector(features)
    return np.fromiter(features.values(), dtype=float).reshape(1, -1)


def _predict_one(predictor: Any, x: np.ndarray, sr: int) -> dict[str, Any]:
    classes = np.asarray(predictor.model.classes_, dtype=int)

    if len(x) > sr * predictor.time_thr or float(np.mean(x)) > predictor.mean_thr:
        sensor_labels = [
            int(label)
            for label, mtype in predictor.mlabel_mtype_dict.items()
            if str(mtype) == "传感器错误"
        ]
        mlabel = sensor_labels[0] if sensor_labels else int(classes[-1])
        mtype = predictor.mlabel_mtype_dict.get(mlabel, "传感器错误")
        return {
            "pred_result": "NOK",
            "pred_mlabel": mlabel,
            "pred_mtype": mtype,
            "pred_probability": np.nan,
            "raw_top_mlabel": np.nan,
            "raw_top_mtype": "",
            "raw_top_probability": np.nan,
            **{f"proba_{int(label)}": np.nan for label in classes},
        }

    features = predictor.extract_features(x, sr)
    X = _feature_matrix(predictor, features)
    proba = np.asarray(predictor.model.predict_proba(X)[0], dtype=float)

    pred_mlabel = int(
        predictor._predict_with_threshold_rule(
            proba.reshape(1, -1),
            classes,
            predictor.mlabel_threshold_dict,
            ok_mlabel=predictor.ok_mlabel,
        )[0]
    )
    raw_top_idx = int(np.argmax(proba))
    raw_top_mlabel = int(classes[raw_top_idx])
    class_to_idx = {int(label): idx for idx, label in enumerate(classes)}
    pred_idx = class_to_idx[pred_mlabel]
    pred_mtype = predictor.mlabel_mtype_dict[pred_mlabel]
    raw_top_mtype = predictor.mlabel_mtype_dict[raw_top_mlabel]

    return {
        "pred_result": "OK" if pred_mlabel == predictor.ok_mlabel else "NOK",
        "pred_mlabel": pred_mlabel,
        "pred_mtype": pred_mtype,
        "pred_probability": float(proba[pred_idx]),
        "raw_top_mlabel": raw_top_mlabel,
        "raw_top_mtype": raw_top_mtype,
        "raw_top_probability": float(proba[raw_top_idx]),
        **{f"proba_{int(label)}": float(proba[idx]) for idx, label in enumerate(classes)},
    }


def _predict_wav_record(
    wav_path: Path,
    *,
    predictor: Any,
    model_id: str,
    target_sr: int,
    trim_edge_samples: int,
    quiet: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "model_id": model_id,
        "filename": wav_path.name,
        "wav_path": str(wav_path),
        "direction": _direction_from_name(wav_path),
        "sampling_rate_used": target_sr,
        "trim_edge_samples": trim_edge_samples,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": "",
    }
    try:
        x, original_sr = _read_wav_mono(wav_path, target_sr)
        effective_samples = len(x) - 2 * trim_edge_samples if len(x) > 2 * trim_edge_samples else len(x)
        record.update(
            {
                "original_sampling_rate": original_sr,
                "input_samples": int(len(x)),
                "effective_samples_after_trim": int(effective_samples),
                "duration_sec": float(len(x) / target_sr),
            }
        )
        record.update(_predict_one(predictor, x, target_sr))
        if not quiet:
            print(
                f"{model_id} | {wav_path.name} | "
                f"{record['pred_result']} {record['pred_mlabel']} {record['pred_mtype']} "
                f"p={record['pred_probability']:.6f}"
            )
    except Exception as exc:  # noqa: BLE001
        record.update(
            {
                "original_sampling_rate": np.nan,
                "input_samples": np.nan,
                "effective_samples_after_trim": np.nan,
                "duration_sec": np.nan,
                "pred_result": "ERROR",
                "pred_mlabel": np.nan,
                "pred_mtype": "",
                "pred_probability": np.nan,
                "raw_top_mlabel": np.nan,
                "raw_top_mtype": "",
                "raw_top_probability": np.nan,
                "error": str(exc),
            }
        )
        if not quiet:
            print(f"{model_id} | {wav_path.name} | ERROR {exc}")
    return record


def main() -> None:
    args = _parse_args()
    folder = Path(args.folder).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    target_sr = int(args.target_sr)
    workers = max(1, int(args.workers))

    model_id = model_dir.name
    trim_edge_samples = 8000

    wav_paths = [
        path
        for path in sorted(folder.rglob(args.pattern))
        if path.is_file() and not any(part.startswith(".") for part in path.parts)
    ]

    predictor = _load_predictor(model_dir)
    predict_record = partial(
        _predict_wav_record,
        predictor=predictor,
        model_id=model_id,
        target_sr=target_sr,
        trim_edge_samples=trim_edge_samples,
        quiet=args.quiet,
    )
    if workers == 1:
        records = [
            predict_record(wav_path)
            for wav_path in wav_paths
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = []
            for index, record in enumerate(executor.map(predict_record, wav_paths), start=1):
                records.append(record)
                if index % 1000 == 0:
                    print(f"processed {index}/{len(wav_paths)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"saved {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
