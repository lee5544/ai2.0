"""DL 训练数据加载层（对应 ml/training/loader.py）。

职责：读取 mel 特征批次 → DataFrame、解析特征布局（通道/序列长度）、
规范化标签列、扫描可用标签，以及配置/路径/批次格式等底层工具。

数据集划分见同包 split.py；训练过程见 dl/train.py。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from dl.models import normalize_model_arch
from data_manager.label_rules import LABEL_RULES

DEFAULT_OUTPUT_FILE_PREFIX = "dl_mel_spec"


SCHEMA_FILENAME = "dl_feature_schema.json"


FEATURE_PATTERN = re.compile(r"^feat__(?P<channel>.+)__(?P<step>\d+)$")


DEFAULT_BATCH_FORMAT = "pickle"


SUPPORTED_BATCH_FORMATS = {"pickle": ".pkl", "pkl": ".pkl", "csv": ".csv"}


COMPACT_FEATURE_COLUMN = "__feature_tensor__"


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


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _normalize_batch_format(value: Any) -> str:
    text = _to_text(value).lower()
    fmt = text or DEFAULT_BATCH_FORMAT
    if fmt not in SUPPORTED_BATCH_FORMATS:
        supported = ", ".join(sorted(set(SUPPORTED_BATCH_FORMATS.keys())))
        raise ValueError(f"不支持的批次格式: {value}，支持: {supported}")
    return "pickle" if fmt == "pkl" else fmt


def _batch_suffix_for_format(batch_format: str) -> str:
    return SUPPORTED_BATCH_FORMATS[_normalize_batch_format(batch_format)]


def _first_config_value(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
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


def _model_tag_from_arch(model_arch: str) -> str:
    model_arch = normalize_model_arch(model_arch)
    return {
        "cnn1d": "cnn",
        "cnn1d_attention": "cnn1d_attention",
        "cnn2d": "cnn2d",
        "cnn2d_attention": "cnn2d_attention",
        "resnet": "resnet",
        "lstm": "lstm",
        "tcn": "tcn",
        "multiscale_cnn1d": "multiscale_cnn1d",
    }[model_arch]


def _resolve_model_arch(cfg: Dict[str, Any], cli_arch: str | None) -> str:
    dl_cfg = _resolve_dl_cfg(cfg)
    dl_model_cfg = dl_cfg.get("model") if isinstance(dl_cfg.get("model"), dict) else {}
    raw_arch = _first_config_value(
        cli_arch,
        dl_model_cfg.get("arch"),
        dl_model_cfg.get("name"),
        dl_cfg.get("model_type"),
        dl_cfg.get("arch"),
        "cnn",
    )
    return normalize_model_arch(_to_text(raw_arch) or "cnn")


def _resolve_results_root(cfg: Dict[str, Any], cli_output_root: str | None, model_arch: str) -> Path:
    if cli_output_root:
        return Path(cli_output_root).expanduser()
    del model_arch
    line_name = _to_text(cfg.get("line_name")) or "line"
    model_name = _to_text((cfg.get("model") or {}).get("model_name")) or "model"
    results_path = Path(str(cfg.get("results_path") or "./results")).expanduser()
    return results_path / f"{line_name}_{model_name}"


def _resolve_dataset_dir(cfg: Dict[str, Any], cli_dataset_dir: str | None, model_arch: str) -> Path:
    if cli_dataset_dir:
        return Path(cli_dataset_dir).expanduser()
    dl_cfg = _resolve_dl_cfg(cfg)
    extract_cfg = dl_cfg.get("extract") if isinstance(dl_cfg.get("extract"), dict) else {}
    configured = extract_cfg.get("output_feature_folder") or dl_cfg.get("output_feature_folder")
    if configured:
        return Path(str(configured)).expanduser()

    line_name = _to_text(cfg.get("line_name")) or "line"
    model_name = _to_text((cfg.get("model") or {}).get("model_name")) or "model"
    results_path = Path(str(cfg.get("results_path") or "./results")).expanduser()
    del model_arch
    return results_path / f"{line_name}_{model_name}" / "dl_dataset_csv"


def _load_feature_schema(dataset_dir: Path) -> Dict[str, Any]:
    schema_path = dataset_dir / SCHEMA_FILENAME
    if not schema_path.exists():
        return {}
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_dataset_files(
    *,
    dataset_dir: Path,
    output_file_prefix: str,
    batch_format: str | None,
) -> List[Path]:
    candidate_formats: List[str] = []
    if _to_text(batch_format):
        candidate_formats.append(_normalize_batch_format(batch_format))
    for fmt in ("pickle", "csv"):
        if fmt not in candidate_formats:
            candidate_formats.append(fmt)

    for fmt in candidate_formats:
        suffix = _batch_suffix_for_format(fmt)
        files = sorted(dataset_dir.glob(f"{output_file_prefix}_batch_*{suffix}"))
        if files:
            return files

    dataset_files: List[Path] = []
    for suffix in sorted(set(SUPPORTED_BATCH_FORMATS.values())):
        dataset_files.extend(sorted(dataset_dir.glob(f"*{suffix}")))
    dataset_files = [path for path in dataset_files if path.name != "sample_view.csv"]
    return dataset_files


def _read_feature_batch(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    raise RuntimeError(f"不支持的特征批次文件: {path}")


def _resolve_compact_feature_column(schema: Dict[str, Any]) -> str:
    return _to_text(schema.get("compact_feature_column")) or COMPACT_FEATURE_COLUMN


def _resolve_feature_layout(df: pd.DataFrame, schema: Dict[str, Any]) -> tuple[List[str], Dict[str, List[str]], int]:
    # 紧凑列（pickle：mel/pcen/raw 等）：通道与序列长度从 schema / 实际数组推断，
    # 支持 feature_columns 为空（如 raw 不定长）。CSV feat__ 列走下方旧逻辑。
    compact = _resolve_compact_feature_column(schema)
    if compact in df.columns:
        channel_names = [str(name) for name in (schema.get("channel_names") or [])]
        if not channel_names:
            first = np.asarray(df[compact].iloc[0])
            n_ch = int(first.shape[0]) if first.ndim >= 2 else 1
            channel_names = [f"ch_{i}" for i in range(n_ch)]
        feat_cols = schema.get("feature_columns") if isinstance(schema.get("feature_columns"), dict) else {}
        columns_by_channel = {ch: [str(c) for c in feat_cols.get(ch, [])] for ch in channel_names}
        seq_len = int(schema.get("sequence_length") or 0)
        if seq_len <= 0:
            try:
                seq_len = int(max(int(np.asarray(a).shape[-1]) for a in df[compact].tolist() if a is not None))
            except Exception:
                seq_len = 0
        return channel_names, columns_by_channel, int(seq_len)

    feature_columns = schema.get("feature_columns")
    if isinstance(feature_columns, dict):
        channel_names = [str(name) for name in schema.get("channel_names", feature_columns.keys()) if str(name) in feature_columns]
        columns_by_channel: Dict[str, List[str]] = {}
        for channel in channel_names:
            cols = [str(col) for col in feature_columns.get(channel, []) if str(col)]
            if cols:
                columns_by_channel[channel] = cols
        if columns_by_channel:
            seq_len = max(len(cols) for cols in columns_by_channel.values())
            return channel_names, columns_by_channel, int(seq_len)

    tmp: Dict[str, Dict[int, str]] = {}
    for col in df.columns:
        match = FEATURE_PATTERN.match(col)
        if not match:
            continue
        channel = match.group("channel")
        step = int(match.group("step"))
        tmp.setdefault(channel, {})[step] = col
    if not tmp:
        raise RuntimeError("未在 CSV 中找到特征列，期望格式如 feat__mel_000__0000。")

    channel_names = list(tmp.keys())
    seq_len = max(max(steps.keys()) for steps in tmp.values()) + 1
    columns_by_channel: Dict[str, List[str]] = {}
    for channel in channel_names:
        cols = []
        for step in range(seq_len):
            cols.append(tmp[channel].get(step, f"__missing__{channel}__{step}"))
        columns_by_channel[channel] = cols
    return channel_names, columns_by_channel, seq_len


def _load_feature_dataframe(model: Any) -> pd.DataFrame:
    dataset_files = list(getattr(model, "dataset_files"))
    if not dataset_files:
        raise RuntimeError(
            "未找到 dl 特征批次文件，请先执行: "
            f"python dl/main_dataset.py --config {getattr(model, 'config_path', '<config>')}"
        )
    dfs = []
    for path in dataset_files:
        print(f"读取特征批次: {path}")
        dfs.append(_read_feature_batch(path))
    return pd.concat(dfs, ignore_index=True)


def _scan_available_reasons(df: pd.DataFrame) -> List[Dict[str, Any]]:
    id_col = next((c for c in ("reason_id", "type_id", "label") if c in df.columns), None)
    name_col = next((c for c in ("reason_name", "type_name") if c in df.columns), None)
    if id_col is None:
        raise RuntimeError("特征数据中缺少 reason_id/type_id/label 列，无法构建标签映射。")
    work_df = df[[c for c in (id_col, name_col) if c is not None]].copy()
    work_df["reason_id"] = pd.to_numeric(work_df[id_col], errors="coerce")
    work_df = work_df[work_df["reason_id"].notna()].copy()
    work_df["reason_id"] = work_df["reason_id"].astype(int)
    if name_col:
        work_df["reason_name"] = work_df[name_col].map(_to_text)
    else:
        work_df["reason_name"] = ""
    counts = work_df["reason_id"].value_counts().sort_index()
    rows: List[Dict[str, Any]] = []
    for rid, count in counts.items():
        sub = work_df[work_df["reason_id"] == int(rid)]
        name = ""
        for value in sub["reason_name"].tolist():
            if value:
                name = value
                break
        if not name:
            for item in LABEL_RULES.get("reasons", {}).values():
                if _to_int(item.get("id")) == int(rid):
                    name = _to_text(item.get("name"))
                    break
        rows.append({"reason_id": int(rid), "reason_name": name or str(int(rid)), "reason_count": int(count)})
    return rows


def _normalize_train_label(df: pd.DataFrame) -> pd.DataFrame:
    src_col = next((c for c in ("reason_id", "type_id", "label") if c in df.columns), None)
    if src_col is None:
        raise RuntimeError("特征文件中缺少标签列：reason_id/type_id/label")
    out = df.copy()
    out["label"] = pd.to_numeric(out[src_col], errors="coerce")
    out = out[out["label"].notna()].copy()
    out["label"] = out["label"].astype(int)
    print(f"[INFO] 训练标签列: {src_col} -> label")
    return out
