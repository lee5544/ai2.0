from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import yaml

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError("运行 dl/main_train.py 前请先安装 PyTorch：pip install torch") from exc


from dl.models import build_dl_model, normalize_model_arch
from data_manager.label_rules import LABEL_RULES, Label


DEFAULT_CONFIG_PATH = Path("cfg/epump4.yaml")
DEFAULT_OUTPUT_FILE_PREFIX = "dl_mel_spec"
SCHEMA_FILENAME = "dl_feature_schema.json"
FEATURE_PATTERN = re.compile(r"^feat__(?P<channel>.+)__(?P<step>\d+)$")
DEFAULT_BATCH_FORMAT = "pickle"
SUPPORTED_BATCH_FORMATS = {"pickle": ".pkl", "pkl": ".pkl", "csv": ".csv"}
COMPACT_FEATURE_COLUMN = "__feature_tensor__"


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

    def set_postfix(self, **kwargs) -> None:
        if not kwargs:
            return
        items = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"{self.desc}: {items}", flush=True)


def _create_progress(*, total: int | None, desc: str, unit: str = "item"):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)
    except Exception:
        return _ConsoleProgress(total=total, desc=desc, unit=unit)


class WindowFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


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
        "cnn2d": "cnn2d",
        "resnet": "resnet",
        "lstm": "lstm",
        "tcn": "tcn",
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
    line_name = _to_text(cfg.get("line_name")) or "line"
    model_name = _to_text((cfg.get("model") or {}).get("model_name")) or "model"
    model_type = _model_tag_from_arch(model_arch)
    results_path = Path(str(cfg.get("results_path") or "./results")).expanduser()
    return results_path / f"{line_name}_{model_name}_{model_type}"


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
    tags = [
        _model_tag_from_arch(model_arch),
        _to_text(dl_cfg.get("model_type")) or "cnn",
        "cnn",
        "cnn1d",
    ]
    candidates: List[Path] = []
    seen: set[str] = set()
    for tag in tags:
        key = _to_text(tag)
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append(results_path / f"{line_name}_{model_name}_{key}" / "dl_dataset_csv")
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _configure_matplotlib_for_chinese() -> str:
    plt.rcParams["axes.unicode_minus"] = False
    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "STHeiti",
        "Heiti SC",
    ]
    try:
        available = {font.name for font in font_manager.fontManager.ttflist}
    except Exception:
        available = set()
    selected = [name for name in candidates if name in available]
    if selected:
        current = plt.rcParams.get("font.sans-serif", [])
        plt.rcParams["font.family"] = ["sans-serif"]
        plt.rcParams["font.sans-serif"] = selected + [x for x in current if x not in selected]
        return selected[0]
    return ""


def _resolve_device(requested: str | None) -> torch.device:
    request = _to_text(requested).lower() or "auto"
    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if request == "auto":
        if cuda_ok:
            return torch.device("cuda")
        if mps_ok:
            return torch.device("mps")
        return torch.device("cpu")
    if request == "cuda":
        if cuda_ok:
            return torch.device("cuda")
        print("[WARN] CUDA 不可用，回退到 CPU。")
        return torch.device("cpu")
    if request == "mps":
        if mps_ok:
            return torch.device("mps")
        print("[WARN] MPS 不可用，回退到 CPU。")
        return torch.device("cpu")
    return torch.device("cpu")


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _default_auto_group_key(reason_id: int, reason_name: str) -> str:
    if reason_id in (-1, 0) or _to_text(reason_name) in {"干扰", "正常"}:
        return "ok_with_noise"
    return f"reason_{reason_id}"


def _build_runtime_from_entries(entries: List[Dict[str, Any]]):
    raw_to_model_mlabel: Dict[int, int] = {}
    for item in entries:
        raw_mlabel = int(item["mlabel_raw"])
        if raw_mlabel not in raw_to_model_mlabel:
            raw_to_model_mlabel[raw_mlabel] = len(raw_to_model_mlabel)

    label_sample_dict: Dict[int, int] = {}
    label_to_mlabel_map: Dict[int, int] = {}
    mlabel_mtype_dict: Dict[int, str] = {}
    mlabel_to_reason_ids: Dict[int, List[int]] = {}

    tmp_names: Dict[int, List[str]] = {}
    for item in entries:
        reason_id = int(item["reason_id"])
        reason_name = _to_text(item.get("reason_name")) or str(reason_id)
        sample = int(item.get("sample", -1))
        mlabel = raw_to_model_mlabel[int(item["mlabel_raw"])]
        label_sample_dict[reason_id] = sample
        label_to_mlabel_map[reason_id] = mlabel
        mlabel_to_reason_ids.setdefault(mlabel, [])
        if reason_id not in mlabel_to_reason_ids[mlabel]:
            mlabel_to_reason_ids[mlabel].append(reason_id)
        tmp_names.setdefault(mlabel, [])
        if reason_name not in tmp_names[mlabel]:
            tmp_names[mlabel].append(reason_name)

    for mlabel, names in tmp_names.items():
        mlabel_mtype_dict[int(mlabel)] = "_".join(names)
    return label_sample_dict, mlabel_mtype_dict, label_to_mlabel_map, mlabel_to_reason_ids


def build_label_runtime(*, train_cfg: Dict[str, Any] | None, available_reasons: List[Dict[str, Any]]):
    train_cfg = train_cfg or {}
    reason_resolver = Label(LABEL_RULES).build_reason_resolver(available_reasons)
    entries: List[Dict[str, Any]] = []

    mapping_cfg = train_cfg.get("label_mapping")
    if not isinstance(mapping_cfg, dict):
        mapping_cfg = {}
    mapping_enable = bool(mapping_cfg.get("enable"))
    auto_mlabel = bool(mapping_cfg.get("auto_mlabel", True))
    mapping_groups = mapping_cfg.get("groups", [])

    normalized_groups: List[Dict[str, Any]] = []
    if isinstance(mapping_groups, list):
        for item in mapping_groups:
            if isinstance(item, dict):
                normalized_groups.append(item)
            else:
                normalized_groups.append({"reasons": item})
    elif isinstance(mapping_groups, dict):
        for value in mapping_groups.values():
            if isinstance(value, dict):
                normalized_groups.append(value)
            else:
                normalized_groups.append({"reasons": value})

    if mapping_enable and normalized_groups:
        for group_idx, group in enumerate(normalized_groups):
            if not bool(group.get("enable", True)):
                continue
            raw_mlabel = group_idx if auto_mlabel else (_to_int(group.get("mlabel")) or group_idx)
            group_sample = _to_int(group.get("sample"))
            if group_sample is None:
                group_sample = -1
            reason_tokens = group.get("reasons", [])
            if isinstance(reason_tokens, (tuple, set)):
                reason_tokens = list(reason_tokens)
            elif not isinstance(reason_tokens, list):
                reason_tokens = [reason_tokens]

            for token in reason_tokens:
                if isinstance(token, dict):
                    rid, rname = reason_resolver.resolve(
                        reason_id_value=token.get("reason_id", token.get("label")),
                        reason_name_value=token.get("reason", token.get("type", token.get("name"))),
                    )
                    sample = _to_int(token.get("sample"))
                    if sample is None:
                        sample = group_sample
                else:
                    rid, rname = reason_resolver.resolve(
                        reason_id_value=token,
                        reason_name_value=token,
                    )
                    sample = group_sample
                if rid is None:
                    raise ValueError(f"无法解析 reason（label_mapping.groups）: {token}")
                entries.append({"reason_id": rid, "reason_name": rname, "mlabel_raw": raw_mlabel, "sample": int(sample)})
    else:
        if not available_reasons:
            raise RuntimeError("未扫描到可用标签，无法构建 DL 训练映射。")
        group_to_raw_mlabel: Dict[str, int] = {}
        for item in available_reasons:
            rid = _to_int(item.get("reason_id"))
            if rid is None:
                continue
            rname = _to_text(item.get("reason_name")) or str(rid)
            group_key = _default_auto_group_key(rid, rname)
            if group_key not in group_to_raw_mlabel:
                group_to_raw_mlabel[group_key] = len(group_to_raw_mlabel)
            entries.append({"reason_id": rid, "reason_name": rname, "mlabel_raw": group_to_raw_mlabel[group_key], "sample": -1})

    dedup: Dict[int, Dict[str, Any]] = {}
    for item in entries:
        dedup[int(item["reason_id"])] = item
    final_entries = list(dedup.values())

    available_reason_ids = {_to_int(item.get("reason_id")) for item in available_reasons}
    available_reason_ids.discard(None)
    filtered_entries = [item for item in final_entries if int(item["reason_id"]) in available_reason_ids]
    if not filtered_entries:
        raise RuntimeError("可用标签映射为空，请检查 train.label_mapping 配置。")

    label_sample_dict, mlabel_mtype_dict, label_to_mlabel_map, mlabel_to_reason_ids = _build_runtime_from_entries(filtered_entries)
    print("[INFO] DL 训练标签映射（mlabel -> merged reasons）:")
    for mlabel in sorted(mlabel_to_reason_ids):
        reason_ids = sorted(int(x) for x in mlabel_to_reason_ids[mlabel])
        counts = {int(item["reason_id"]): int(item["reason_count"]) for item in available_reasons if _to_int(item.get("reason_id")) is not None}
        total_count = sum(counts.get(rid, 0) for rid in reason_ids)
        sample_values = sorted({int(label_sample_dict.get(rid, -1)) for rid in reason_ids})
        sample_text = str(sample_values[0]) if len(sample_values) == 1 else "/".join(str(v) for v in sample_values)
        print(
            f"  mlabel={mlabel:>2} | count={total_count:>5} | sample={sample_text:>6} | "
            f"reason_ids={reason_ids} | mtype={mlabel_mtype_dict[mlabel]}"
        )
    return label_sample_dict, mlabel_mtype_dict, label_to_mlabel_map, mlabel_to_reason_ids


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


def _resolve_split_groups(df: pd.DataFrame, *, split_name: str) -> pd.Series | None:
    if "sn" not in df.columns:
        print(f"[WARN] {split_name} 缺少 sn 列，回退为按行切分。")
        return None
    groups = df["sn"].map(_to_text)
    empty_mask = groups.eq("")
    if empty_mask.any():
        groups = groups.copy()
        missing_indexes = list(groups[empty_mask].index)
        for idx in missing_indexes:
            groups.at[idx] = f"__missing_sn__{idx}"
        print(f"[WARN] {split_name} 检测到 {len(missing_indexes)} 条 sn 为空，已按独立组处理。")
    if int(groups.nunique()) < 2:
        print(f"[WARN] {split_name} 仅有 {int(groups.nunique())} 个 sn 组，回退为按行切分。")
        return None
    return groups.astype(str)


def _random_row_split(df: pd.DataFrame, *, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 2:
        raise RuntimeError("样本数不足 2，无法切分。")
    rng = np.random.default_rng(int(random_state))
    indexes = np.arange(len(df))
    rng.shuffle(indexes)
    n_test = int(round(len(df) * float(test_size)))
    n_test = max(1, min(len(df) - 1, n_test))
    test_idx = indexes[:n_test]
    train_idx = indexes[n_test:]
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def _stratified_row_split(df: pd.DataFrame, *, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = df.groupby("label").indices
    if len(grouped) < 2:
        raise RuntimeError("类别数不足 2，无法做分层切分。")
    rng = np.random.default_rng(int(random_state))
    train_idx: List[int] = []
    test_idx: List[int] = []
    for _, idx_array in grouped.items():
        indexes = np.array(list(idx_array), dtype=int)
        if len(indexes) < 2:
            raise RuntimeError("存在样本数不足 2 的类别，无法做分层切分。")
        rng.shuffle(indexes)
        n_test = int(round(len(indexes) * float(test_size)))
        n_test = max(1, min(len(indexes) - 1, n_test))
        test_idx.extend(indexes[:n_test].tolist())
        train_idx.extend(indexes[n_test:].tolist())
    if not train_idx or not test_idx:
        raise RuntimeError("分层切分失败：出现空集合。")
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def _safe_row_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df["label"].nunique() > 1:
        try:
            return _stratified_row_split(df, test_size=test_size, random_state=random_state)
        except Exception as exc:
            print(f"[WARN] {split_name} 分层切分失败，改为非分层切分: {exc}")
    return _random_row_split(df, test_size=test_size, random_state=random_state)


def _group_split_score(
    *,
    full_labels: pd.Series,
    train_labels: pd.Series,
    test_labels: pd.Series,
    target_test_rows: float,
) -> float:
    full_counts = full_labels.value_counts()
    train_counts = train_labels.value_counts()
    test_counts = test_labels.value_counts()
    total_rows = max(int(len(full_labels)), 1)
    score = abs(len(test_labels) - target_test_rows) / total_rows
    for label in sorted(full_counts.index.tolist()):
        full_ratio = float(full_counts.get(label, 0)) / total_rows
        train_ratio = float(train_counts.get(label, 0)) / max(int(len(train_labels)), 1)
        test_ratio = float(test_counts.get(label, 0)) / max(int(len(test_labels)), 1)
        expected_test = float(full_counts.get(label, 0)) * (float(target_test_rows) / total_rows)
        score += abs(train_ratio - full_ratio)
        score += abs(test_ratio - full_ratio)
        score += abs(float(test_counts.get(label, 0)) - expected_test) / total_rows
        if full_counts.get(label, 0) > 0 and train_counts.get(label, 0) == 0:
            score += 5.0
    return float(score)


def _safe_group_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    split_name: str,
    group_split_trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _resolve_split_groups(df, split_name=split_name)
    if groups is None:
        return _safe_row_train_test_split(df, test_size=test_size, random_state=random_state, split_name=split_name)

    unique_groups = groups.unique().tolist()
    if len(unique_groups) < 2:
        return _safe_row_train_test_split(df, test_size=test_size, random_state=random_state, split_name=split_name)

    target_test_rows = float(len(df)) * float(test_size)
    best_score: float | None = None
    best_mask: pd.Series | None = None

    for trial in range(max(8, int(group_split_trials))):
        rng = np.random.default_rng(int(random_state) + int(trial))
        choice = rng.random(len(unique_groups)) < float(test_size)
        if int(choice.sum()) <= 0:
            choice[int(rng.integers(0, len(unique_groups)))] = True
        if int(choice.sum()) >= len(unique_groups):
            choice[int(rng.integers(0, len(unique_groups)))] = False
        selected = {unique_groups[idx] for idx, picked in enumerate(choice.tolist()) if picked}
        test_mask = groups.isin(selected)
        if int(test_mask.sum()) <= 0 or int((~test_mask).sum()) <= 0:
            continue
        score = _group_split_score(
            full_labels=df["label"],
            train_labels=df.loc[~test_mask, "label"],
            test_labels=df.loc[test_mask, "label"],
            target_test_rows=target_test_rows,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_mask = test_mask.copy()

    if best_mask is None:
        print(f"[WARN] {split_name} 按 sn 分组切分失败，回退为按行切分。")
        return _safe_row_train_test_split(df, test_size=test_size, random_state=random_state, split_name=split_name)

    train_df = df.loc[~best_mask].copy()
    test_df = df.loc[best_mask].copy()
    print(
        f"[INFO] {split_name} 按 sn 分组切分: "
        f"train_groups={int(groups.loc[~best_mask].nunique())}, "
        f"test_groups={int(groups.loc[best_mask].nunique())}, "
        f"train_rows={len(train_df)}, test_rows={len(test_df)}"
    )
    return train_df, test_df


def _overall_train_fraction(*, test_size: float, val_size: float) -> float:
    keep_after_test = max(0.0, 1.0 - float(test_size))
    keep_for_train = max(0.0, 1.0 - float(val_size))
    fraction = keep_after_test * keep_for_train
    return fraction if fraction > 0 else 1.0


def _prepare_core_and_extra_before_split(
    df: pd.DataFrame,
    *,
    label_sample_dict: Dict[int, int],
    test_size: float,
    val_size: float,
    random_state: int,
    group_sample_trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _resolve_split_groups(df, split_name="预分池")
    if groups is None:
        print("[WARN] 预分池缺少稳定分组键，跳过剩余样本外置逻辑。")
        return df.copy(), df.iloc[0:0].copy()

    train_fraction = _overall_train_fraction(test_size=test_size, val_size=val_size)
    label_counts = df["label"].value_counts().to_dict()
    target_pool_by_label: Dict[int, int] = {}
    for label, target_train_n in label_sample_dict.items():
        cur_n = int(label_counts.get(label, 0))
        if cur_n <= 0:
            continue
        if int(target_train_n) < 0:
            target_pool_by_label[int(label)] = cur_n
            continue
        target_pool_n = int(np.ceil(float(target_train_n) / float(train_fraction)))
        target_pool_n = max(int(target_train_n), target_pool_n)
        target_pool_by_label[int(label)] = min(cur_n, target_pool_n)
    if not target_pool_by_label:
        return df.copy(), df.iloc[0:0].copy()

    group_label_counts = pd.crosstab(groups, df["label"]).astype(int)
    target_series = pd.Series(target_pool_by_label, dtype=int)
    tracked_labels = [int(label) for label in target_series.index.tolist() if int(label) in group_label_counts.columns]
    if not tracked_labels:
        return df.copy(), df.iloc[0:0].copy()
    target_series = target_series.reindex(tracked_labels).fillna(0).astype(int)

    full_take_labels = [label for label in tracked_labels if int(target_series.get(label, 0)) >= int(label_counts.get(label, 0))]
    mandatory_groups: set[str]
    if full_take_labels:
        mandatory_mask = group_label_counts[full_take_labels].sum(axis=1) > 0
        mandatory_groups = set(group_label_counts.index[mandatory_mask].tolist())
    else:
        mandatory_groups = set()

    base_counts = (
        group_label_counts.loc[list(mandatory_groups), tracked_labels].sum(axis=0)
        if mandatory_groups
        else pd.Series(0, index=tracked_labels, dtype=int)
    )
    candidate_groups = [g for g in group_label_counts.index.tolist() if g not in mandatory_groups]
    group_rows_map = {
        group: group_label_counts.loc[group, tracked_labels].reindex(tracked_labels, fill_value=0).astype(int)
        for group in candidate_groups
    }

    best_selected = set(mandatory_groups)
    best_counts = base_counts.copy()
    best_score: tuple[int, int] | None = None
    trial_count = max(8, min(int(group_sample_trials), 96))

    for trial in range(trial_count):
        rng = np.random.default_rng(int(random_state) + int(trial))
        order = list(candidate_groups)
        rng.shuffle(order)
        selected = set(mandatory_groups)
        selected_order: List[str] = []
        current_counts = base_counts.copy()

        for group in order:
            deficit = (target_series - current_counts).clip(lower=0)
            if int(deficit.sum()) <= 0:
                break
            contribution = group_rows_map[group]
            if int(contribution[deficit > 0].sum()) <= 0:
                continue
            selected.add(group)
            selected_order.append(group)
            current_counts = current_counts.add(contribution, fill_value=0).astype(int)

        for group in reversed(selected_order):
            contribution = group_rows_map[group]
            candidate_counts = current_counts.subtract(contribution, fill_value=0).astype(int)
            if bool((candidate_counts >= target_series).all()):
                selected.remove(group)
                current_counts = candidate_counts

        deficit_total = int((target_series - current_counts).clip(lower=0).sum())
        overshoot_total = int((current_counts - target_series).clip(lower=0).sum())
        score = (deficit_total, overshoot_total)
        if best_score is None or score < best_score:
            best_score = score
            best_selected = set(selected)
            best_counts = current_counts.copy()

    core_mask = groups.isin(best_selected)
    core_df = df.loc[core_mask].copy()
    extra_df = df.loc[~core_mask].copy()
    print("[INFO] 预分池完成：")
    for label in tracked_labels:
        cur_n = int(label_counts.get(label, 0))
        target_pool_n = int(target_series.get(label, 0))
        core_n = int(best_counts.get(label, 0))
        extra_n = max(cur_n - core_n, 0)
        if target_pool_n >= cur_n:
            print(f"  类别 {label}: 全取 {cur_n} 条")
        else:
            print(
                f"  类别 {label}: 训练目标={label_sample_dict.get(label, -1)} "
                f"-> 主样本池目标={target_pool_n}，实际={core_n}，剩余={extra_n}"
            )
    return core_df, extra_df


def _nonempty_key_set(df: pd.DataFrame, column: str) -> set[str] | None:
    if column not in df.columns:
        return None
    values = {_to_text(v) for v in df[column].tolist()}
    values.discard("")
    return values


def _print_split_overlap_report(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("[自检] train/val/test 交集检查:")
    split_pairs = [("train∩val", train_df, val_df), ("train∩test", train_df, test_df), ("val∩test", val_df, test_df)]
    for column in ("sample_id", "sn", "tdms_path"):
        if _nonempty_key_set(train_df, column) is None or _nonempty_key_set(val_df, column) is None or _nonempty_key_set(test_df, column) is None:
            print(f"  {column}: 缺少列，跳过")
            continue
        parts = []
        for pair_name, left_df, right_df in split_pairs:
            left_keys = _nonempty_key_set(left_df, column) or set()
            right_keys = _nonempty_key_set(right_df, column) or set()
            parts.append(f"{pair_name}={len(left_keys & right_keys)}")
        print(f"  {column}: " + ", ".join(parts))


def _load_and_split_dataset(
    *,
    df: pd.DataFrame,
    train_cfg: Dict[str, Any],
    label_sample_dict: Dict[int, int],
    label_to_mlabel_map: Dict[int, int],
    random_state: int,
    test_size: float,
    val_size: float,
    group_split_trials: int,
    group_sample_trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _normalize_train_label(df)
    valid_labels = list(label_sample_dict.keys())
    df = df[df["label"].isin(valid_labels)].copy()
    if df.empty:
        raise RuntimeError("过滤后数据为空，请检查 label_mapping 与特征文件中的标签是否一致。")

    core_df, extra_df = _prepare_core_and_extra_before_split(
        df,
        label_sample_dict=label_sample_dict,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        group_sample_trials=group_sample_trials,
    )
    if core_df.empty:
        raise RuntimeError("主样本池为空，无法继续切分。")

    tmp_df, test_df = _safe_group_train_test_split(
        core_df,
        test_size=test_size,
        random_state=random_state,
        split_name="train/test",
        group_split_trials=group_split_trials,
    )
    train_df, val_df = _safe_group_train_test_split(
        tmp_df,
        test_size=val_size,
        random_state=random_state,
        split_name="train/val",
        group_split_trials=group_split_trials,
    )
    if not extra_df.empty:
        test_df = pd.concat([test_df, extra_df], ignore_index=True)
        print(f"[INFO] 预分池剩余样本已并入 test: +{len(extra_df)} 条")

    train_parts = []
    train_counts = train_df["label"].value_counts().to_dict()
    for label, target_n in label_sample_dict.items():
        if label not in train_counts:
            print(f"[WARN] 训练集中不存在 label={label}，跳过扩充。")
            continue
        sub = train_df[train_df["label"] == label]
        cur_n = len(sub)
        if int(target_n) < 0:
            sampled = sub.copy()
            print(f"类别 {label}: 训练集已有 {cur_n} 条，配置要求全取 -> 保留 {len(sampled)} 条")
        elif cur_n > int(target_n):
            sampled = sub.sample(n=int(target_n), random_state=random_state)
            print(f"类别 {label}: 训练集已有 {cur_n} 条 -> 精调到 {target_n}")
        else:
            need_extra = int(target_n) - cur_n
            if need_extra > 0:
                extra = sub.sample(n=need_extra, replace=True, random_state=random_state)
                sampled = pd.concat([sub, extra], ignore_index=True)
                print(f"类别 {label}: 训练集已有 {cur_n} 条 -> 上采样到 {target_n}")
            else:
                sampled = sub.copy()
        train_parts.append(sampled)
    if train_parts:
        train_df = pd.concat(train_parts, ignore_index=True)

    _print_split_overlap_report(train_df, val_df, test_df)

    def _map_labels(df_in: pd.DataFrame) -> pd.DataFrame:
        out = df_in.copy()
        out["label_raw"] = out["label"].astype(int)
        out["label"] = out["label"].map(label_to_mlabel_map).fillna(-1).astype(int)
        return out[out["label"] >= 0].copy()

    train_df = _map_labels(train_df)
    val_df = _map_labels(val_df)
    test_df = _map_labels(test_df)
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("切分后出现空数据集，请检查 sample 数量和标签分布。")
    return train_df, val_df, test_df


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


def _build_feature_tensor(
    df: pd.DataFrame,
    *,
    channel_names: List[str],
    columns_by_channel: Dict[str, List[str]],
    compact_feature_column: str,
) -> np.ndarray:
    if compact_feature_column in df.columns:
        tensors = [np.asarray(item, dtype=np.float32) for item in df[compact_feature_column].tolist()]
        if not tensors:
            return np.empty((0, len(channel_names), 0), dtype=np.float32)
        expected_shape = (len(channel_names), len(columns_by_channel[channel_names[0]]) if channel_names else 0)
        for idx, tensor in enumerate(tensors):
            if tensor.shape != expected_shape:
                raise RuntimeError(
                    f"紧凑特征张量形状不一致: row={idx}, got={tensor.shape}, expected={expected_shape}"
                )
        return np.stack(tensors, axis=0).astype(np.float32, copy=False)

    channels = []
    for channel in channel_names:
        cols = columns_by_channel[channel]
        channel_df = df.reindex(columns=cols, fill_value=0.0).copy()
        for col in cols:
            if col.startswith("__missing__"):
                channel_df[col] = 0.0
        channel_arr = channel_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        channels.append(channel_arr[:, None, :])
    return np.concatenate(channels, axis=1)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    for true_label, pred_label in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= int(true_label) < num_classes and 0 <= int(pred_label) < num_classes:
            cm[int(true_label), int(pred_label)] += 1
    return cm


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, Any]:
    cm = _confusion_matrix(y_true, y_pred, num_classes=num_classes)
    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0
    precision_list: List[float] = []
    recall_list: List[float] = []
    f1_list: List[float] = []
    support_list: List[int] = []
    per_class: List[Dict[str, Any]] = []
    for idx in range(int(num_classes)):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        support = int(cm[idx, :].sum())
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        class_accuracy = recall
        f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        support_list.append(support)
        per_class.append(
            {
                "class_id": int(idx),
                "accuracy": class_accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision_list)) if precision_list else 0.0,
        "macro_recall": float(np.mean(recall_list)) if recall_list else 0.0,
        "macro_f1": float(np.mean(f1_list)) if f1_list else 0.0,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "support": support_list,
    }


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
    num_classes: int,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
) -> Dict[str, Any]:
    train_mode = optimizer is not None
    use_amp = scaler is not None and device.type == "cuda"
    model.train(train_mode)
    total_loss = 0.0
    batch_count = 0
    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []

    with _create_progress(total=len(loader), desc=desc, unit="batch") as progress:
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            if train_mode:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss.detach().item())
            batch_count += 1

            probs = torch.softmax(logits.detach(), dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true_parts.append(batch_y.detach().cpu().numpy())
            y_pred_parts.append(preds.detach().cpu().numpy())

            if hasattr(progress, "set_postfix"):
                progress.set_postfix(loss=f"{float(loss.detach().item()):.4f}")
            progress.update(1)

    y_true = np.concatenate(y_true_parts) if y_true_parts else np.empty(0, dtype=np.int64)
    y_pred = np.concatenate(y_pred_parts) if y_pred_parts else np.empty(0, dtype=np.int64)
    metrics = _compute_metrics(y_true, y_pred, num_classes=max(1, int(num_classes)))
    metrics["loss"] = float(total_loss / max(1, batch_count))
    return metrics


def _draw_history_axes(axes: np.ndarray, history_df: pd.DataFrame) -> None:
    plot_specs = [
        ("Loss", "train_loss", "val_loss", None),
        ("Accuracy", "train_accuracy", "val_accuracy", (0.0, 1.05)),
        ("Macro F1", "train_macro_f1", "val_macro_f1", (0.0, 1.05)),
    ]
    axes_arr = np.atleast_1d(axes).reshape(-1)
    epochs = history_df["epoch"] if "epoch" in history_df.columns else pd.Series(dtype=np.int64)

    for ax, (title, train_col, val_col, ylim) in zip(axes_arr, plot_specs):
        ax.clear()
        if train_col in history_df.columns:
            ax.plot(epochs, history_df[train_col], label="train", linewidth=2.0)
        if val_col in history_df.columns:
            ax.plot(epochs, history_df[val_col], label="val", linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend()


def _save_history_plot(history_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)
    _draw_history_axes(np.asarray(axes), history_df)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


class _LiveHistoryPlotter:
    def __init__(self, *, out_path: Path) -> None:
        self.out_path = Path(out_path)
        self.fig: plt.Figure | None = None
        self.axes: np.ndarray | None = None
        backend = str(plt.get_backend()).lower()
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        self.show_window = has_display and "agg" not in backend

    def _ensure_figure(self) -> tuple[plt.Figure, np.ndarray]:
        if self.fig is None or self.axes is None:
            if self.show_window:
                plt.ion()
            self.fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)
            self.axes = np.atleast_1d(axes).reshape(-1)
            if self.show_window:
                self.fig.show()
        return self.fig, self.axes

    def update(
        self,
        history_df: pd.DataFrame,
        *,
        best_epoch: int | None = None,
        best_score: float | None = None,
    ) -> None:
        if history_df.empty:
            return
        fig, axes = self._ensure_figure()
        _draw_history_axes(axes, history_df)
        title = f"Training Progress ({len(history_df)} epochs)"
        if best_epoch is not None and best_epoch > 0 and best_score is not None:
            title = f"{title} | best_epoch={best_epoch}, best_val_f1={best_score:.4f}"
        fig.suptitle(title)
        fig.savefig(self.out_path, dpi=160)
        if self.show_window:
            fig.canvas.draw_idle()
            plt.show(block=False)
            plt.pause(0.001)

    def close(self) -> None:
        if self.fig is None:
            return
        if self.show_window:
            plt.ioff()
        plt.close(self.fig)
        self.fig = None
        self.axes = None


def _append_per_class_metric_history(
    *,
    history_row: Dict[str, Any],
    prefix: str,
    metrics: Dict[str, Any],
    num_classes: int,
    metric_key: str,
    column_suffix: str,
) -> None:
    per_class = metrics.get("per_class") if isinstance(metrics.get("per_class"), list) else []
    metric_by_class = {
        int(item.get("class_id")): float(item.get(metric_key, 0.0))
        for item in per_class
        if isinstance(item, dict)
    }
    for class_id in range(int(num_classes)):
        history_row[f"{prefix}_{column_suffix}_{class_id}"] = float(metric_by_class.get(class_id, 0.0))


def _save_per_class_metric_plot(
    *,
    history_df: pd.DataFrame,
    class_name_by_id: Dict[int, str],
    out_path: Path,
    column_suffix: str,
    ylabel: str,
) -> None:
    if history_df.empty or not class_name_by_id:
        return

    class_ids = sorted(int(k) for k in class_name_by_id.keys())
    n_classes = len(class_ids)
    ncols = 2 if n_classes > 1 else 1
    nrows = int(np.ceil(n_classes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for plot_idx, class_id in enumerate(class_ids):
        ax = axes_arr[plot_idx]
        train_col = f"train_{column_suffix}_{class_id}"
        val_col = f"val_{column_suffix}_{class_id}"
        if train_col in history_df.columns:
            ax.plot(history_df["epoch"], history_df[train_col], label="train")
        if val_col in history_df.columns:
            ax.plot(history_df["epoch"], history_df[val_col], label="val")
        ax.set_title(f"class {class_id}: {class_name_by_id[class_id]}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend()

    for idx in range(n_classes, len(axes_arr)):
        axes_arr[idx].axis("off")

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
