from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Tuple


import numpy as np
import pandas as pd
from data_manager.label_database import load_label_dataframe, load_sample_dataframe
import yaml

from data_manager.config import load_data_manager_config
from data_manager.label_internal_registry import label_source_rank
from data_manager.tdms_read import read_tdms

from .preprocess import trim_edges


REQUIRED_SAMPLE_VIEW_COLS = {"line", "sn", "sample_id"}
DEFAULT_CONFIG_PATH = Path("cfg/epump4.yaml")
DEFAULT_OUTPUT_FILE_PREFIX = "dl_pcen_spec"
SCHEMA_FILENAME = "dl_feature_schema.json"
DEFAULT_OUTPUT_FORMAT = "pickle"
SUPPORTED_OUTPUT_FORMATS = {"pickle": ".pkl", "pkl": ".pkl", "csv": ".csv"}
COMPACT_FEATURE_COLUMN = "__feature_tensor__"
_PCEN_TIME_CONSTANT_SEC = 0.4
_PCEN_S = 0.025
_PCEN_ALPHA = 0.98
_PCEN_DELTA = 2.0
_PCEN_R = 0.5
_PCEN_EPS = 1e-6
_PCEN_MAX_RELATIVE_ENERGY = 1e12
DM_CFG = load_data_manager_config()
SAMPLE_VIEW_OUTPUT_COLS = ["line", "sn", "sample_id"]
FEATURE_OUTPUT_METADATA_COLUMNS = [
    "line",
    "sn",
    "sample_id",
    "tdms_path",
    "group_name",
    "channel_name",
    "sampling_rate",
    "seq_length",
    "num_features",
    "label",
    "label_source",
    "label_timestamp",
    "result_key",
    "result_id",
    "result_name",
    "reason_key",
    "reason_id",
    "reason_name",
    "label_key",
    "label_id",
    "label_name",
    "type_key",
    "type_id",
    "type_name",
    "label_version",
    "note",
]


class _ConsoleProgress:
    def __init__(self, *, total: int | None, desc: str, unit: str = "item") -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.count = 0
        self._last_print = 0.0
        self._closed = False

    def __enter__(self):
        print(f"{self.desc}: 开始")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def update(self, n: int = 1) -> None:
        self.count += int(n)
        now = time.time()
        should_print = (now - self._last_print) >= 0.8
        if self.total is not None and self.count >= self.total:
            should_print = True
        if not should_print:
            return
        self._last_print = now
        if self.total is None or self.total <= 0:
            print(f"{self.desc}: {self.count} {self.unit}", flush=True)
            return
        ratio = min(1.0, max(0.0, float(self.count) / float(self.total)))
        print(
            f"{self.desc}: {self.count}/{self.total} {self.unit} ({ratio * 100.0:.1f}%)",
            flush=True,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.total is None or self.total <= 0:
            print(f"{self.desc}: 完成 {self.count} {self.unit}", flush=True)
        else:
            print(f"{self.desc}: 完成 {self.count}/{self.total} {self.unit}", flush=True)


def _create_progress(*, total: int | None, desc: str, unit: str = "item"):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)
    except Exception:
        return _ConsoleProgress(total=total, desc=desc, unit=unit)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _read_yaml(config_path: str | Path | None) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_dl_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = cfg.get("dl")
    return raw if isinstance(raw, dict) else {}


def _resolve_results_root(cfg: Dict[str, Any]) -> Path:
    line_name = _to_text(cfg.get("line_name")) or "line"
    model_name = _to_text((cfg.get("model") or {}).get("model_name")) or "model"
    results_path = Path(str(cfg.get("results_path") or "./results")).expanduser()
    return results_path / f"{line_name}_{model_name}"


def _resolve_output_folder(cfg: Dict[str, Any], cli_output_folder: str | None) -> Path:
    if cli_output_folder:
        return Path(cli_output_folder).expanduser()
    dl_cfg = _resolve_dl_cfg(cfg)
    extract_cfg = dl_cfg.get("extract") if isinstance(dl_cfg.get("extract"), dict) else {}
    output_folder = (
        extract_cfg.get("output_feature_folder")
        or dl_cfg.get("output_feature_folder")
    )
    if output_folder:
        return Path(str(output_folder)).expanduser()
    return _resolve_results_root(cfg) / "dl_dataset_csv"


def _resolve_sample_view_paths(
    *,
    cfg: Dict[str, Any],
    args_sample_view: List[str],
    output_feature_folder: Path,
) -> List[Path]:
    if args_sample_view:
        return [Path(x).expanduser() for x in args_sample_view]

    dl_cfg = _resolve_dl_cfg(cfg)
    extract_cfg = dl_cfg.get("extract") if isinstance(dl_cfg.get("extract"), dict) else {}
    configured = extract_cfg.get("sample_view") or dl_cfg.get("sample_view")
    if isinstance(configured, str) and _to_text(configured):
        return [Path(configured).expanduser()]
    if isinstance(configured, list):
        out = [Path(str(x)).expanduser() for x in configured if _to_text(x)]
        if out:
            return out

    candidates: List[Path] = []
    candidates.append(output_feature_folder / "sample_view.csv")

    train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    line_name = _to_text(cfg.get("line_name"))
    model_name = _to_text((cfg.get("model") or {}).get("model_name"))
    train_model_type = _to_text(train_cfg.get("model_type"))
    if line_name and model_name and train_model_type:
        ml_dataset_dir = Path(str(cfg.get("results_path") or "./results")).expanduser() / (
            f"{line_name}_{model_name}_{train_model_type}"
        ) / "dataset_csv"
        candidates.append(ml_dataset_dir / "sample_view.csv")

    seen: set[str] = set()
    resolved: List[Path] = []
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            resolved.append(path)
    if resolved:
        return resolved

    print("[WARN] 未找到 sample_view.csv，可通过 --sample-view 指定。")
    for path in candidates:
        print(f"  candidate: {path}")
    return []


def _resolve_batch_size(cfg: Dict[str, Any], cli_batch_size: int | None) -> int:
    if cli_batch_size is not None:
        value = cli_batch_size
    else:
        dl_cfg = _resolve_dl_cfg(cfg)
        extract_cfg = dl_cfg.get("extract") if isinstance(dl_cfg.get("extract"), dict) else {}
        value = extract_cfg.get("batch_size", 2000)
    batch_size = int(value)
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须 > 0，当前: {batch_size}")
    return batch_size


def _normalize_output_format(value: Any) -> str:
    text = _to_text(value).lower()
    fmt = text or DEFAULT_OUTPUT_FORMAT
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        supported = ", ".join(sorted(set(SUPPORTED_OUTPUT_FORMATS.keys())))
        raise ValueError(f"不支持的 output_format: {value}，支持: {supported}")
    return "pickle" if fmt == "pkl" else fmt


def _batch_file_suffix(output_format: str) -> str:
    return SUPPORTED_OUTPUT_FORMATS[_normalize_output_format(output_format)]


def _iter_batch_files(*, output_folder: Path, output_file_prefix: str) -> Iterable[Path]:
    seen: set[str] = set()
    for suffix in sorted(set(SUPPORTED_OUTPUT_FORMATS.values())):
        for path in sorted(output_folder.glob(f"{output_file_prefix}_batch_*{suffix}")):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            yield path


def _resolve_pcen_params(
    cfg: Dict[str, Any],
    *,
    cli_n_fft: int | None,
    cli_hop_length: int | None,
    cli_n_mels: int | None,
    cli_max_frames: int | None,
    cli_fmin: float | None,
    cli_fmax: float | None,
) -> tuple[int, int, int, int, float, float | None]:
    dl_cfg = _resolve_dl_cfg(cfg)
    pcen_cfg = dl_cfg.get("pcen") if isinstance(dl_cfg.get("pcen"), dict) else {}
    mel_cfg = dl_cfg.get("mel") if isinstance(dl_cfg.get("mel"), dict) else {}
    n_fft = int(cli_n_fft or pcen_cfg.get("n_fft") or mel_cfg.get("n_fft") or 256)
    hop_length = int(cli_hop_length or pcen_cfg.get("hop_length") or mel_cfg.get("hop_length") or max(1, n_fft // 2))
    n_mels = int(cli_n_mels or pcen_cfg.get("n_mels") or mel_cfg.get("n_mels") or 13)
    max_frames = int(cli_max_frames or pcen_cfg.get("max_frames") or mel_cfg.get("max_frames") or 1024)
    fmin = float(cli_fmin if cli_fmin is not None else pcen_cfg.get("fmin", mel_cfg.get("fmin", 0.0)) or 0.0)
    fmax_raw = cli_fmax if cli_fmax is not None else pcen_cfg.get("fmax", mel_cfg.get("fmax"))
    fmax = None if fmax_raw in (None, "", "None", "null") else float(fmax_raw)
    if n_fft <= 0:
        raise ValueError(f"n_fft 必须 > 0，当前: {n_fft}")
    if hop_length <= 0:
        raise ValueError(f"hop_length 必须 > 0，当前: {hop_length}")
    if n_mels <= 0:
        raise ValueError(f"n_mels 必须 > 0，当前: {n_mels}")
    if max_frames <= 0:
        raise ValueError(f"max_frames 必须 > 0，当前: {max_frames}")
    if fmin < 0:
        raise ValueError(f"fmin 必须 >= 0，当前: {fmin}")
    if fmax is not None and fmax <= fmin:
        raise ValueError(f"fmax 必须 > fmin，当前: fmin={fmin}, fmax={fmax}")
    return n_fft, hop_length, n_mels, max_frames, fmin, fmax


def _collect_sample_view_columns(sample_view_files: List[Path]) -> List[str]:
    for path in sample_view_files:
        if not path.exists():
            raise FileNotFoundError(f"sample_view 不存在: {path}")
        header_df = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        missing = REQUIRED_SAMPLE_VIEW_COLS - set(header_df.columns)
        if missing:
            raise ValueError(f"{path} 缺少列: {sorted(missing)}")
    return list(SAMPLE_VIEW_OUTPUT_COLS)


def _flush_batch(
    *,
    batch: List[Dict[str, Any]],
    batch_index: int,
    output_folder: Path,
    output_file_prefix: str,
    output_format: str,
    fixed_columns: List[str],
    feature_columns: List[str],
) -> int:
    if not batch:
        return batch_index
    df = pd.DataFrame(batch)
    fmt = _normalize_output_format(output_format)
    if fmt == "pickle" and COMPACT_FEATURE_COLUMN in df.columns:
        all_columns = fixed_columns + [COMPACT_FEATURE_COLUMN]
    else:
        all_columns = fixed_columns + feature_columns
    for col in all_columns:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[all_columns]
    out_path = output_folder / f"{output_file_prefix}_batch_{batch_index}{_batch_file_suffix(fmt)}"
    if fmt == "pickle":
        df.to_pickle(out_path, protocol=pickle.HIGHEST_PROTOCOL)
    elif fmt == "csv":
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
    else:  # pragma: no cover
        raise AssertionError(f"未处理的输出格式: {fmt}")
    print(f"已保存批次 {batch_index}: {len(df)} 条 -> {out_path}")
    return batch_index + 1


def _tuple_get(row: Tuple[Any, ...], col_idx: Dict[str, int], col: str) -> Any:
    idx = col_idx.get(col)
    if idx is None:
        return None
    return row[idx]


def _build_sample_meta_lookup(label_records_db_path: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    sample_index_df = load_sample_dataframe(label_records_db_path)
    lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    col_idx = {c: i for i, c in enumerate(sample_index_df.columns)}
    for row in sample_index_df.itertuples(index=False, name=None):
        line = _to_text(_tuple_get(row, col_idx, "line"))
        sn = _to_text(_tuple_get(row, col_idx, "sn"))
        sample_id = _to_text(_tuple_get(row, col_idx, "sample_id"))
        if not line or not sn or not sample_id:
            continue
        lookup[(line, sn, sample_id)] = {
            "group_name": _to_text(_tuple_get(row, col_idx, "group_name")),
            "sampling_rate": _to_int(_tuple_get(row, col_idx, "sampling_rate")),
        }
    return lookup


def _build_manifest_lookup(manifest_path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    manifest_df = pd.read_csv(manifest_path, encoding="utf-8-sig")
    if manifest_df.empty:
        return {}
    if "created_time" in manifest_df.columns:
        manifest_df["_created_ts"] = pd.to_datetime(manifest_df["created_time"], errors="coerce")
    else:
        manifest_df["_created_ts"] = pd.NaT
    manifest_df = manifest_df.sort_values("_created_ts", ascending=True, na_position="last")
    latest_df = manifest_df.drop_duplicates(subset=["line", "sn"], keep="last")

    lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
    col_idx = {c: i for i, c in enumerate(latest_df.columns)}
    for row in latest_df.itertuples(index=False, name=None):
        line = _to_text(_tuple_get(row, col_idx, "line"))
        sn = _to_text(_tuple_get(row, col_idx, "sn"))
        if not line or not sn:
            continue
        storage_root = _to_text(_tuple_get(row, col_idx, "tdms_storage_root"))
        if not storage_root:
            storage_root = _to_text(_tuple_get(row, col_idx, "storage_root"))
        lookup[(line, sn)] = {
            "tdms_storage_root": storage_root,
            "relative_path": _to_text(_tuple_get(row, col_idx, "relative_path")),
        }
    return lookup


def _build_label_lookup(label_records_db_path: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    label_df = load_label_dataframe(label_records_db_path)
    if label_df.empty:
        return {}

    if "timestamp" in label_df.columns:
        label_df["_ts"] = pd.to_datetime(label_df["timestamp"], errors="coerce")
    else:
        label_df["_ts"] = pd.NaT
    source_priority = {"expert": 0, "operator": 1, "model": 2}
    if "source" in label_df.columns:
        label_df["_source_rank"] = label_df["source"].astype(str).map(
            lambda value: label_source_rank(value, source_priority)
        ).fillna(99)
    else:
        label_df["_source_rank"] = 99
    label_df = label_df.sort_values(
        by=["line", "sn", "sample_id", "_ts", "_source_rank"],
        ascending=[True, True, True, False, True],
        na_position="last",
    )
    latest_df = label_df.drop_duplicates(subset=["line", "sn", "sample_id"], keep="first")

    keep_cols = [
        "source",
        "timestamp",
        "result_key",
        "result_id",
        "result_name",
        "reason_key",
        "reason_id",
        "reason_name",
        "label_key",
        "label_id",
        "label_name",
        "type_key",
        "type_id",
        "type_name",
        "label_version",
        "note",
    ]
    lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    col_idx = {c: i for i, c in enumerate(latest_df.columns)}
    for row in latest_df.itertuples(index=False, name=None):
        line = _to_text(_tuple_get(row, col_idx, "line"))
        sn = _to_text(_tuple_get(row, col_idx, "sn"))
        sample_id = _to_text(_tuple_get(row, col_idx, "sample_id"))
        if not line or not sn or not sample_id:
            continue
        lookup[(line, sn, sample_id)] = {k: _tuple_get(row, col_idx, k) for k in keep_cols}
    return lookup


def _pick_signal_by_sample(
    *,
    tdms_ret: Dict[str, Any],
    sample_group_name: str,
    sample_id: str,
) -> tuple[np.ndarray, str]:
    if sample_group_name == tdms_ret["up_group"]:
        signal = tdms_ret.get("up_data")
        if signal is None:
            raise KeyError(f"TDMS 缺少 group={tdms_ret['up_group']} 数据")
        return signal, tdms_ret["up_group"]
    if sample_group_name == tdms_ret["down_group"]:
        signal = tdms_ret.get("down_data")
        if signal is None:
            raise KeyError(f"TDMS 缺少 group={tdms_ret['down_group']} 数据")
        return signal, tdms_ret["down_group"]

    sample_id_lower = str(sample_id).lower()
    if sample_id_lower.endswith("_up"):
        signal = tdms_ret.get("up_data")
        if signal is None:
            raise KeyError(f"TDMS 缺少 group={tdms_ret['up_group']} 数据")
        return signal, tdms_ret["up_group"]
    if sample_id_lower.endswith("_down"):
        signal = tdms_ret.get("down_data")
        if signal is None:
            raise KeyError(f"TDMS 缺少 group={tdms_ret['down_group']} 数据")
        return signal, tdms_ret["down_group"]

    raise KeyError(f"无法根据 sample_id/group_name 映射信号: sample_id={sample_id}, group_name={sample_group_name}")


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (hz / 700.0))


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _cacheable_fmax(value: float | None) -> float:
    return -1.0 if value is None else float(value)


@lru_cache(maxsize=64)
def _cached_hann_window(n_fft: int) -> np.ndarray:
    return np.hanning(int(n_fft)).astype(np.float32)


@lru_cache(maxsize=64)
def _cached_feature_keys(n_mels: int, max_frames: int) -> tuple[str, ...]:
    return tuple(
        f"feat__pcen_{mel_idx:03d}__{frame_idx:04d}"
        for mel_idx in range(int(n_mels))
        for frame_idx in range(int(max_frames))
    )


def _build_mel_filterbank(
    *,
    sampling_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float | None,
) -> np.ndarray:
    sr = float(sampling_rate)
    max_freq = sr / 2.0
    high_freq = max_freq if fmax is None else min(float(fmax), max_freq)
    if high_freq <= float(fmin):
        high_freq = max_freq

    mel_edges = np.linspace(_hz_to_mel(np.array([float(fmin)]))[0], _hz_to_mel(np.array([high_freq]))[0], int(n_mels) + 2)
    hz_edges = _mel_to_hz(mel_edges)
    fft_freqs = np.linspace(0.0, max_freq, int(n_fft // 2) + 1)
    filters = np.zeros((int(n_mels), fft_freqs.shape[0]), dtype=np.float32)

    for mel_idx in range(int(n_mels)):
        left = float(hz_edges[mel_idx])
        center = float(hz_edges[mel_idx + 1])
        right = float(hz_edges[mel_idx + 2])
        if center <= left:
            center = left + 1e-6
        if right <= center:
            right = center + 1e-6

        left_mask = (fft_freqs >= left) & (fft_freqs <= center)
        right_mask = (fft_freqs >= center) & (fft_freqs <= right)
        filters[mel_idx, left_mask] = (fft_freqs[left_mask] - left) / (center - left)
        filters[mel_idx, right_mask] = (right - fft_freqs[right_mask]) / (right - center)

    norm = filters.sum(axis=1, keepdims=True)
    norm[norm <= 0] = 1.0
    filters /= norm
    return filters


@lru_cache(maxsize=64)
def _cached_mel_filterbank(
    sampling_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax_cache_value: float,
) -> np.ndarray:
    fmax = None if float(fmax_cache_value) < 0 else float(fmax_cache_value)
    return _build_mel_filterbank(
        sampling_rate=int(sampling_rate),
        n_fft=int(n_fft),
        n_mels=int(n_mels),
        fmin=float(fmin),
        fmax=fmax,
    )


def _pcen_smoothing_coefficient(frame_sr: float, time_constant_sec: float) -> float:
    t_frames = max(float(frame_sr) * float(time_constant_sec), _PCEN_EPS)
    return float(2.0 / (np.sqrt(1.0 + 4.0 * t_frames * t_frames) + 1.0))


def pcen_transform(
    energy: np.ndarray,
    *,
    s: float = _PCEN_S,
    alpha: float = _PCEN_ALPHA,
    delta: float = _PCEN_DELTA,
    r: float = _PCEN_R,
    eps: float = _PCEN_EPS,
) -> np.ndarray:
    """Per-channel energy normalization for a mel-energy matrix [n_mels, n_frames]."""
    E = np.asarray(energy, dtype=np.float64)
    if E.ndim != 2:
        raise ValueError("energy must be 2-D: [n_mels, n_frames]")
    if E.shape[1] == 0:
        return np.maximum(np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    E = np.maximum(np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    # positive = E[E > 0.0]
    # if positive.size:
    #     energy_ref = max(float(np.median(positive)), np.finfo(np.float64).tiny)
    #     E = np.clip(E / energy_ref, 0.0, _PCEN_MAX_RELATIVE_ENERGY)

    s = float(np.clip(s, _PCEN_EPS, 1.0))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    delta = max(float(delta), 0.0)
    r = max(float(r), _PCEN_EPS)
    eps = max(float(eps), _PCEN_EPS)

    M = np.empty_like(E, dtype=np.float64)
    init_frames = min(E.shape[1], max(3, int(round(1.0 / s))))
    M[:, 0] = np.median(E[:, :init_frames], axis=1)
    for t in range(1, E.shape[1]):
        M[:, t] = (1.0 - s) * M[:, t - 1] + s * E[:, t]

    smooth = np.power(eps + M, alpha)
    return np.power(E / (smooth + _PCEN_EPS) + delta, r) - np.power(delta, r)


def extract_pcen_spectrogram_features(
    signal: np.ndarray,
    *,
    sampling_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    max_frames: int,
    fmin: float,
    fmax: float | None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    data = np.asarray(signal, dtype=np.float32).reshape(-1)
    if data.size == 0:
        raise ValueError("空信号，无法提取 PCEN 频谱特征")

    if data.size < int(n_fft):
        data = np.pad(data, (0, int(n_fft) - int(data.size)), mode="constant")

    last_start = max(0, int(data.size) - int(n_fft))
    starts = list(range(0, last_start + 1, int(hop_length)))
    if not starts:
        starts = [0]
    elif starts[-1] != last_start:
        starts.append(last_start)
    actual_frame_count = min(len(starts), int(max_frames))
    starts = np.asarray(starts[: int(max_frames)], dtype=np.int64)

    window = _cached_hann_window(int(n_fft))
    # 固定输出到 max_frames；当实际帧数不足时，后续时间步保持 0，实现补零。
    stft_power = np.zeros((int(n_fft // 2) + 1, int(max_frames)), dtype=np.float32)
    if actual_frame_count > 0:
        frames_view = np.lib.stride_tricks.sliding_window_view(data, int(n_fft))
        frames = frames_view[starts]
        spectra = np.fft.rfft(frames * window[None, :], n=int(n_fft), axis=1)
        power = (np.abs(spectra) ** 2) / float(max(int(n_fft), 1))
        stft_power[:, :actual_frame_count] = power.T.astype(np.float32, copy=False)

    mel_filters = _cached_mel_filterbank(
        int(sampling_rate),
        int(n_fft),
        int(n_mels),
        float(fmin),
        _cacheable_fmax(fmax),
    )
    mel_spec = np.matmul(mel_filters, stft_power)
    frame_sr = float(sampling_rate) / float(hop_length)
    pcen_s = _pcen_smoothing_coefficient(frame_sr, _PCEN_TIME_CONSTANT_SEC)
    pcen_spec = pcen_transform(mel_spec, s=pcen_s).astype(np.float32, copy=False)

    meta = {
        "frame_count": int(actual_frame_count),
        "feature_steps": int(max_frames),
        "feature_channels": int(n_mels),
        "pcen_s": float(pcen_s),
    }
    return pcen_spec, meta


def _process_one_group(
    *,
    line: str,
    sn: str,
    rows: List[Dict[str, Any]],
    sample_view_columns: List[str],
    data_root: Path,
    sample_meta_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    manifest_lookup: Dict[Tuple[str, str], Dict[str, str]],
    label_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    tdms_cache: Dict[Tuple[str, str, str], Dict[str, Any]],
    tdms_cache_lock: Lock,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    max_frames: int,
    fmin: float,
    fmax: float | None,
    output_format: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    fmt = _normalize_output_format(output_format)
    shared_feature_keys = _cached_feature_keys(int(n_mels), int(max_frames)) if fmt == "csv" else ()
    manifest_meta = manifest_lookup.get((line, sn))
    if not manifest_meta:
        for row_map in rows:
            results.append(
                {
                    "status": "fail",
                    "message": "manifest 中未找到对应 line/sn",
                    "line": line,
                    "sn": sn,
                    "sample_id": _to_text(row_map.get("sample_id")),
                }
            )
        return results

    prepared_rows: List[Dict[str, Any]] = []
    required_groups: set[str] = set()
    has_unknown_group = False
    for row_map in rows:
        sample_id = _to_text(row_map.get("sample_id"))
        key = (line, sn, sample_id)
        label_ret = label_lookup.get(key)
        if not label_ret:
            results.append({"status": "skip", "line": line, "sn": sn, "sample_id": sample_id})
            continue
        sample_meta = sample_meta_lookup.get(key)
        if not sample_meta:
            results.append(
                {
                    "status": "fail",
                    "message": "sample_index 中未找到对应样本",
                    "line": line,
                    "sn": sn,
                    "sample_id": sample_id,
                }
            )
            continue
        sample_group_name = _to_text(sample_meta.get("group_name"))
        if sample_group_name:
            required_groups.add(sample_group_name)
        else:
            has_unknown_group = True
        prepared_rows.append(
            {
                "row_map": row_map,
                "sample_id": sample_id,
                "sample_meta": sample_meta,
                "label_ret": label_ret,
                "sample_group_name": sample_group_name,
            }
        )

    if not prepared_rows:
        return results

    tdms_path = data_root / str(manifest_meta["tdms_storage_root"]) / str(manifest_meta["relative_path"])
    required = None if has_unknown_group else required_groups
    cache_key = (
        str(tdms_path),
        line,
        ",".join(sorted(required)) if required is not None else "*",
    )
    with tdms_cache_lock:
        cached = tdms_cache.get(cache_key)
    if cached is None:
        tdms_ret = read_tdms(tdms_path, line=line, required_groups=required)
        with tdms_cache_lock:
            tdms_cache[cache_key] = tdms_ret
    else:
        tdms_ret = cached

    for item in prepared_rows:
        row_map = item["row_map"]
        sample_id = item["sample_id"]
        label_ret = item["label_ret"]
        sample_meta = item["sample_meta"]
        try:
            signal, mapped_group_name = _pick_signal_by_sample(
                tdms_ret=tdms_ret,
                sample_group_name=item["sample_group_name"],
                sample_id=sample_id,
            )
            signal = trim_edges(signal)  # 统一预处理：裁剪头尾 8000 点
            sampling_rate = _to_int(sample_meta.get("sampling_rate")) or _to_int(tdms_ret.get("sampling_rate"))
            if not sampling_rate or sampling_rate <= 0:
                raise ValueError(f"非法 sampling_rate: {sampling_rate}")
            feature_tensor, feature_meta = extract_pcen_spectrogram_features(
                signal,
                sampling_rate=int(sampling_rate),
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                max_frames=max_frames,
                fmin=fmin,
                fmax=fmax,
            )
            record: Dict[str, Any] = {}
            for col in sample_view_columns:
                record[col] = row_map.get(col)
            record.update(
                {
                    "tdms_path": str(tdms_path),
                    "group_name": mapped_group_name,
                    "channel_name": tdms_ret.get("acc_channel"),
                    "sampling_rate": sampling_rate,
                    "seq_length": int(len(signal)),
                    "num_features": int(feature_tensor.size),
                    "frame_count": int(feature_meta["frame_count"]),
                    "n_fft": int(n_fft),
                    "hop_length": int(hop_length),
                    "n_mels": int(n_mels),
                    "fmin": float(fmin),
                    "fmax": None if fmax is None else float(fmax),
                    "pcen_time_constant_sec": float(_PCEN_TIME_CONSTANT_SEC),
                    "pcen_s": float(feature_meta["pcen_s"]),
                    "pcen_alpha": float(_PCEN_ALPHA),
                    "pcen_delta": float(_PCEN_DELTA),
                    "pcen_r": float(_PCEN_R),
                    "pcen_eps": float(_PCEN_EPS),
                    "feature_steps": int(feature_meta["feature_steps"]),
                    "feature_channels": int(feature_meta["feature_channels"]),
                }
            )

            result_key = label_ret.get("result_key", label_ret.get("label_key"))
            result_id = _to_int(label_ret.get("result_id", label_ret.get("label_id")))
            result_name = label_ret.get("result_name", label_ret.get("label_name"))
            reason_key = label_ret.get("reason_key", label_ret.get("type_key"))
            reason_id = _to_int(label_ret.get("reason_id", label_ret.get("type_id")))
            reason_name = label_ret.get("reason_name", label_ret.get("type_name"))

            record["label_source"] = label_ret.get("source")
            record["label_timestamp"] = label_ret.get("timestamp")
            record["result_key"] = result_key
            record["result_id"] = result_id
            record["result_name"] = result_name
            record["reason_key"] = reason_key
            record["reason_id"] = reason_id
            record["reason_name"] = reason_name
            record["label_key"] = result_key
            record["label_id"] = result_id
            record["label_name"] = result_name
            record["type_key"] = reason_key
            record["type_id"] = reason_id
            record["type_name"] = reason_name
            record["label_version"] = label_ret.get("label_version")
            record["note"] = label_ret.get("note")
            record["label"] = result_id
            if fmt == "pickle":
                record[COMPACT_FEATURE_COLUMN] = np.asarray(feature_tensor, dtype=np.float32, copy=False)
                results.append({"status": "ok", "record": record, "feature_keys": ()})
            else:
                feature_values = dict(zip(shared_feature_keys, feature_tensor.reshape(-1).tolist()))
                record.update(feature_values)
                results.append({"status": "ok", "record": record, "feature_keys": shared_feature_keys})
        except Exception as exc:
            results.append(
                {
                    "status": "fail",
                    "message": f"{type(exc).__name__}: {exc}",
                    "line": line,
                    "sn": sn,
                    "sample_id": sample_id,
                }
            )
    return results


def _write_feature_schema(
    *,
    output_folder: Path,
    output_file_prefix: str,
    output_format: str,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    max_frames: int,
    fmin: float,
    fmax: float | None,
) -> None:
    channel_names = [f"pcen_{idx:03d}" for idx in range(int(n_mels))]
    schema = {
        "feature_version": "pcen_spectrogram_v1",
        "output_file_prefix": output_file_prefix,
        "output_format": _normalize_output_format(output_format),
        "compact_feature_column": COMPACT_FEATURE_COLUMN,
        "channel_names": channel_names,
        "sequence_length": int(max_frames),
        "feature_columns": {
            channel_name: [f"feat__{channel_name}__{idx:04d}" for idx in range(int(max_frames))]
            for channel_name in channel_names
        },
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "n_mels": int(n_mels),
        "max_frames": int(max_frames),
        "fmin": float(fmin),
        "fmax": None if fmax is None else float(fmax),
        "pcen_time_constant_sec": float(_PCEN_TIME_CONSTANT_SEC),
        "pcen_alpha": float(_PCEN_ALPHA),
        "pcen_delta": float(_PCEN_DELTA),
        "pcen_r": float(_PCEN_R),
        "pcen_eps": float(_PCEN_EPS),
    }
    with (output_folder / SCHEMA_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取 PCEN 频谱特征并保存到 dl_dataset_csv")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=f"YAML 配置路径（默认: {DEFAULT_CONFIG_PATH}）")
    parser.add_argument("--feature-type", default=None, help="DL 特征类型：mel/pcen（仅 dl.features 入口使用）")
    parser.add_argument("--sample-view", action="append", default=[], help="sample_view.csv 路径，可重复传入")
    parser.add_argument("--data-root", default=None, help="data_root 路径")
    parser.add_argument("--manifest-path", default=None, help="tdms_manifest.csv 路径")
    parser.add_argument("--label-records-db-path", default=None, help="label_records.db 路径")
    parser.add_argument("--output-feature-folder", default=None, help="特征输出目录")
    parser.add_argument("--output-file-prefix", default=None, help=f"特征批次文件前缀（默认: {DEFAULT_OUTPUT_FILE_PREFIX}）")
    parser.add_argument("--output-format", default=None, help="批次输出格式：pickle/csv（默认: pickle）")
    parser.add_argument("--n-fft", type=int, default=None, help="STFT n_fft")
    parser.add_argument("--hop-length", type=int, default=None, help="STFT hop_length")
    parser.add_argument("--n-mels", type=int, default=None, help="PCEN/mel bin 数")
    parser.add_argument("--max-frames", type=int, default=None, help="最大时间帧数，超出截断，不足补零")
    parser.add_argument("--fmin", type=float, default=None, help="最低频率")
    parser.add_argument("--fmax", type=float, default=None, help="最高频率")
    parser.add_argument("--batch-size", type=int, default=None, help="每个 CSV 分片的样本数")
    parser.add_argument("--num-workers", type=int, default=max(1, min(8, (os.cpu_count() or 1))), help="并行 worker 数")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    """核心提取逻辑，可供外部（如 dl/main_dataset.py）直接调用。"""
    wall_t0 = time.perf_counter()
    cfg = _read_yaml(args.config)
    dl_cfg = _resolve_dl_cfg(cfg)
    extract_cfg = dl_cfg.get("extract") if isinstance(dl_cfg.get("extract"), dict) else {}
    output_feature_folder = _resolve_output_folder(cfg, args.output_feature_folder)
    output_feature_folder.mkdir(parents=True, exist_ok=True)
    output_file_prefix = _to_text(args.output_file_prefix) or _to_text(extract_cfg.get("output_file_prefix")) or DEFAULT_OUTPUT_FILE_PREFIX
    output_format = _normalize_output_format(args.output_format or extract_cfg.get("output_format") or dl_cfg.get("output_format") or DEFAULT_OUTPUT_FORMAT)
    batch_size = _resolve_batch_size(cfg, args.batch_size)
    n_fft, hop_length, n_mels, max_frames, fmin, fmax = _resolve_pcen_params(
        cfg,
        cli_n_fft=args.n_fft,
        cli_hop_length=args.hop_length,
        cli_n_mels=args.n_mels,
        cli_max_frames=args.max_frames,
        cli_fmin=args.fmin,
        cli_fmax=args.fmax,
    )

    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
    data_root_raw = args.data_root or dataset_cfg.get("data_root") or cfg.get("data_root") or DM_CFG.get("data_root")
    if not data_root_raw:
        raise ValueError("未提供 data_root：请用 --data-root 或在 YAML/cfg/core/data_manager.yaml 中配置")
    data_root = Path(str(data_root_raw)).expanduser()
    manifest_path = Path(
        args.manifest_path or dataset_cfg.get("manifest_path") or cfg.get("manifest_path") or (data_root / "metadata" / "tdms_manifest.csv")
    ).expanduser()
    label_records_db_path = Path(
        args.label_records_db_path or dataset_cfg.get("label_records_db_path") or cfg.get("label_records_db_path") or (data_root / "metadata" / "label_records.db")
    ).expanduser()

    sample_view_paths = _resolve_sample_view_paths(
        cfg=cfg,
        args_sample_view=args.sample_view,
        output_feature_folder=output_feature_folder,
    )
    if not sample_view_paths:
        return
    sample_view_columns = _collect_sample_view_columns(sample_view_paths)

    for path in (manifest_path, label_records_db_path):
        if not path.exists():
            raise FileNotFoundError(f"缺少 metadata 文件: {path}")

    sample_view_resolved = {p.resolve() for p in sample_view_paths if p.exists()}
    for old_file in _iter_batch_files(output_folder=output_feature_folder, output_file_prefix=output_file_prefix):
        if old_file.resolve() in sample_view_resolved:
            continue
        old_file.unlink()
        print(f"删除旧批次文件: {old_file}")

    fixed_columns = list(FEATURE_OUTPUT_METADATA_COLUMNS)

    print(f"输出目录: {output_feature_folder}")
    print(f"sample_view 文件数: {len(sample_view_paths)}")
    print(
        f"PCEN 配置: n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels}, "
        f"max_frames={max_frames}, fmin={fmin}, fmax={fmax}, "
        f"time_constant_sec={_PCEN_TIME_CONSTANT_SEC}"
    )
    print(f"输出前缀: {output_file_prefix}")
    print(f"输出格式: {output_format}")
    print(f"批次大小: {batch_size}")
    print(f"并行 worker 数: {max(1, int(args.num_workers))}")

    print("预加载 metadata 索引...")
    sample_meta_lookup = _build_sample_meta_lookup(label_records_db_path)
    manifest_lookup = _build_manifest_lookup(manifest_path)
    label_lookup = _build_label_lookup(label_records_db_path)
    print(
        f"索引完成: sample_meta={len(sample_meta_lookup)}, "
        f"manifest={len(manifest_lookup)}, label={len(label_lookup)}"
    )

    grouped_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    invalid_rows = 0
    total_rows = 0
    for sample_view_path in sample_view_paths:
        df = pd.read_csv(sample_view_path, encoding="utf-8-sig")
        cols = list(df.columns)
        total_rows += len(df)
        for row_idx, row in enumerate(df.itertuples(index=False, name=None)):
            row_map = dict(zip(cols, row))
            row_map["__sample_view_file"] = str(sample_view_path)
            row_map["__sample_view_row_index"] = int(row_idx)
            line = _to_text(row_map.get("line"))
            sn = _to_text(row_map.get("sn"))
            sample_id = _to_text(row_map.get("sample_id"))
            if not line or not sn or not sample_id:
                invalid_rows += 1
                continue
            grouped_rows.setdefault((line, sn), []).append(row_map)
    print(f"sample_view 分组完成: groups={len(grouped_rows)}, total_rows={total_rows}, invalid_rows={invalid_rows}")

    ok_rows = 0
    failed_rows = invalid_rows
    skipped_rows = 0
    batch: List[Dict[str, Any]] = []
    feature_columns: List[str] = []
    batch_index = 1
    tdms_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    tdms_cache_lock = Lock()

    def _consume_result(ret: Dict[str, Any]) -> None:
        nonlocal ok_rows, failed_rows, skipped_rows, batch_index
        status = ret.get("status")
        if status == "ok":
            record = ret["record"]
            for col in ret.get("feature_keys", []):
                if col not in feature_columns:
                    feature_columns.append(col)
            batch.append(record)
            ok_rows += 1
            if len(batch) >= batch_size:
                batch_index = _flush_batch(
                    batch=batch,
                    batch_index=batch_index,
                    output_folder=output_feature_folder,
                    output_file_prefix=output_file_prefix,
                    output_format=output_format,
                    fixed_columns=fixed_columns,
                    feature_columns=feature_columns,
                )
                batch.clear()
        elif status == "skip":
            skipped_rows += 1
        else:
            failed_rows += 1
            print(
                f"处理失败: line={ret.get('line')} sn={ret.get('sn')} sample_id={ret.get('sample_id')} | "
                f"{ret.get('message')}"
            )

    pending = set()
    pending_sizes: Dict[Any, int] = {}
    workers = max(1, int(args.num_workers))
    max_pending = max(8, workers * 2)

    def _consume_done(done_futures) -> int:
        processed = 0
        nonlocal failed_rows
        for future in done_futures:
            group_size = pending_sizes.pop(future, 1)
            try:
                ret_list = future.result()
            except Exception as exc:
                failed_rows += group_size
                print(f"worker 异常: {type(exc).__name__}: {exc}")
                processed += group_size
                continue
            for ret in ret_list:
                _consume_result(ret)
                processed += 1
        return processed

    with ThreadPoolExecutor(max_workers=workers) as executor, _create_progress(
        total=total_rows,
        desc="DL 特征提取进度",
        unit="row",
    ) as progress:
        if invalid_rows > 0:
            progress.update(invalid_rows)
        for (line, sn), rows in grouped_rows.items():
            future = executor.submit(
                _process_one_group,
                line=line,
                sn=sn,
                rows=rows,
                sample_view_columns=sample_view_columns,
                data_root=data_root,
                sample_meta_lookup=sample_meta_lookup,
                    manifest_lookup=manifest_lookup,
                    label_lookup=label_lookup,
                    tdms_cache=tdms_cache,
                    tdms_cache_lock=tdms_cache_lock,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    max_frames=max_frames,
                    fmin=fmin,
                    fmax=fmax,
                    output_format=output_format,
                )
            pending.add(future)
            pending_sizes[future] = len(rows)
            if len(pending) >= max_pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                progress.update(_consume_done(done))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            progress.update(_consume_done(done))

    if batch:
        batch_index = _flush_batch(
            batch=batch,
            batch_index=batch_index,
            output_folder=output_feature_folder,
            output_file_prefix=output_file_prefix,
            output_format=output_format,
            fixed_columns=fixed_columns,
            feature_columns=feature_columns,
        )
        batch.clear()

    _write_feature_schema(
        output_folder=output_feature_folder,
        output_file_prefix=output_file_prefix,
        output_format=output_format,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        max_frames=max_frames,
        fmin=fmin,
        fmax=fmax,
    )

    total_sec = float(time.perf_counter() - wall_t0)
    print("")
    print("DL 特征提取完成")
    print(f"成功行数: {ok_rows}")
    print(f"跳过行数: {skipped_rows}")
    print(f"失败行数: {failed_rows}")
    print(f"总耗时: {total_sec:.3f}s")
    print(f"特征 schema: {output_feature_folder / SCHEMA_FILENAME}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
