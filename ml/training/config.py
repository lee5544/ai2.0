from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from data_manager.label_rules import LABEL_RULES, Label
DEFAULT_CONFIG_PATH = Path("cfg/epump4.yaml")

NON_FEATURE_COLUMNS = {
    "sample_view_file",
    "sample_view_row_index",
    "view_name",
    "line",
    "sn",
    "sample_id",
    "reference",
    "time",
    "tdms_storage_root",
    "relative_path",
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
    "label_version",
    "note",
}

SAMPLE_VIEW_EXPORT_COLUMNS = [
    "sample_view_file",
    "sample_view_row_index",
    "view_name",
    "line",
    "sn",
    "sample_id",
    "reference",
    "time",
    "tdms_storage_root",
    "relative_path",
    "tdms_path",
    "group_name",
    "channel_name",
    "sampling_rate",
    "seq_length",
    "num_features",
    "label_source",
    "label_timestamp",
    "result_key",
    "result_id",
    "result_name",
    "reason_key",
    "reason_id",
    "reason_name",
    "label_version",
    "note",
]


@dataclass(frozen=True)
class TrainingBootstrap:
    config_file: Path
    cfg: dict[str, Any]
    train_cfg: dict[str, Any]
    line_name: str
    model_type: str
    model_id: str
    results_path: str
    dataset_files: List[str]
    available_reasons: List[Dict[str, Any]]


def resolve_model_id(cfg: dict[str, Any]) -> str:
    line_name = _to_text(cfg.get("line_name"))
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    model_name = _to_text(model_cfg.get("model_name"))
    if not line_name or not model_name:
        raise ValueError("配置缺少 line_name / model.model_name。")
    return f"{line_name}_{model_name}"


def resolve_results_dir(cfg: dict[str, Any]) -> Path:
    model_id = resolve_model_id(cfg)
    results_root = Path(_to_text(cfg.get("results_path")) or "results").expanduser()
    return results_root / model_id


def resolve_dataset_output_dir(cfg: dict[str, Any]) -> Path:
    return resolve_results_dir(cfg) / "dataset_csv"


def _to_int(value) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    s = str(value).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def _looks_like_numeric_feature(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series, errors="coerce")
    return bool(converted.notna().any())


class FeatureSchemaError(ValueError):
    """训练特征 CSV 的列结构与 num_features 不符时抛出。"""


def resolve_feature_columns_by_schema(df: pd.DataFrame) -> List[str]:
    """
    按约定的 CSV 列结构解析训练特征列：

    1. 读取 ``num_features`` 列值（要求所有行一致）；
    2. 从 ``mean`` 列开始，严格读取 ``num_features`` 列作为特征列；
    3. 若切片长度与 ``num_features`` 不一致，抛 ``FeatureSchemaError`` 立即中断训练。

    这是训练、特征筛选、可视化等所有使用 features_batch_*.csv 的入口应调用的统一口径，
    以保证训练/推理特征维度始终一致。
    """
    columns = list(df.columns)
    if "num_features" not in columns:
        raise FeatureSchemaError(
            "训练特征 CSV 缺少 'num_features' 列，无法按约定解析特征维度。"
            " 请检查 dataset_csv/features_batch_*.csv 是否为最新特征提取器产出。"
        )
    if "mean" not in columns:
        raise FeatureSchemaError(
            "训练特征 CSV 缺少 'mean' 列，无法确定特征起始位置。"
            " 请检查特征提取器版本（期望从 'mean' 列开始为数值特征块）。"
        )

    # 读取并校验 num_features 全行一致
    nf_series = pd.to_numeric(df["num_features"], errors="coerce")
    nf_valid = nf_series.dropna()
    if nf_valid.empty:
        raise FeatureSchemaError(
            "训练特征 CSV 的 'num_features' 列为空或无法解析为整数。"
        )
    unique_nf = sorted({int(v) for v in nf_valid.unique()})
    if len(unique_nf) > 1:
        raise FeatureSchemaError(
            f"训练特征 CSV 中 'num_features' 取值不一致: {unique_nf}。"
            " 请确认 dataset_csv 下所有 features_batch_*.csv 来自同一特征提取版本。"
        )
    nf_expected = int(unique_nf[0])

    mean_pos = columns.index("mean")
    feature_cols = columns[mean_pos:mean_pos + nf_expected]

    if len(feature_cols) != nf_expected:
        raise FeatureSchemaError(
            f"训练特征维度不匹配，训练终止：\n"
            f"  - 从 'mean' 列按 num_features 切片得到 {len(feature_cols)} 列\n"
            f"  - 但 CSV 中 num_features = {nf_expected}\n"
            f"请检查 dataset_csv/features_batch_*.csv 的特征起始位置和列顺序。"
        )

    return list(feature_cols)


def _read_selected_features_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"selected_features_file 不存在: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if df.empty:
            return []
        col = "feature" if "feature" in df.columns else df.columns[0]
        return [_to_text(v) for v in df[col].tolist() if _to_text(v)]
    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, list):
            return [_to_text(v) for v in data if _to_text(v)]
        if isinstance(data, dict):
            values = data.get("features") or data.get("selected_features") or []
            if isinstance(values, list):
                return [_to_text(v) for v in values if _to_text(v)]
        return []
    return [_to_text(line) for line in path.read_text(encoding="utf-8").splitlines() if _to_text(line)]


def resolve_configured_feature_columns(
    feature_cols: List[str],
    *,
    train_cfg: Dict[str, Any] | None,
    config_file: str | Path | None = None,
) -> List[str]:
    train_cfg = train_cfg or {}
    feature_selection_cfg = train_cfg.get("feature_selection")
    if not isinstance(feature_selection_cfg, dict):
        return feature_cols

    selected_features: List[str] = []
    inline_features = feature_selection_cfg.get("selected_features")
    if isinstance(inline_features, list):
        selected_features.extend(_to_text(v) for v in inline_features if _to_text(v))

    selected_file = _to_text(feature_selection_cfg.get("selected_features_file"))
    if selected_file:
        path = Path(selected_file).expanduser()
        if not path.is_absolute() and config_file is not None:
            config_parent = Path(config_file).expanduser().parent
            candidate = config_parent / path
            path = candidate if candidate.exists() else Path(selected_file).expanduser()
        selected_features.extend(_read_selected_features_file(path))

    if not selected_features:
        return feature_cols

    selected_order = []
    seen: set[str] = set()
    for name in selected_features:
        if name in seen:
            continue
        seen.add(name)
        selected_order.append(name)

    available = set(feature_cols)
    filtered = [name for name in selected_order if name in available]
    missing = [name for name in selected_order if name not in available]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        print(f"[WARN] selected_features 中有 {len(missing)} 个特征不存在，已忽略: {preview}{suffix}")
    if not filtered:
        raise RuntimeError("feature_selection 没有匹配到任何可用特征。")

    top_k = _to_int(feature_selection_cfg.get("top_k"))
    if top_k is not None and top_k > 0:
        filtered = filtered[: int(top_k)]
    print(f"[INFO] 启用特征筛选: 使用 {len(filtered)} / {len(feature_cols)} 个特征")
    return filtered


def _scan_available_reasons(dataset_files: List[str]) -> List[Dict[str, Any]]:
    """
    从特征 CSV 中扫描可用 reason_id/reason_name。
    """
    rule_runtime = Label(LABEL_RULES)
    rule_name_by_id = {int(r["id"]): r["name"] for r in rule_runtime.reasons.values()}

    reason_ids: set[int] = set()
    reason_name_by_id: Dict[int, str] = {}
    reason_count_by_id: Dict[int, int] = {}

    for path in dataset_files:
        try:
            header_df = pd.read_csv(path, nrows=0)
        except Exception as e:
            print(f"[WARN] 扫描标签时读取失败（跳过）: {path} | {e}")
            continue

        cols = set(header_df.columns)
        if "reason_id" not in cols:
            continue

        use_cols = ["reason_id"]
        if "reason_name" in cols:
            use_cols.append("reason_name")
        try:
            df = pd.read_csv(path, usecols=use_cols)
        except Exception as e:
            print(f"[WARN] 扫描标签时读取失败（跳过）: {path} | {e}")
            continue

        ids = pd.to_numeric(df["reason_id"], errors="coerce")
        names = df["reason_name"].map(_to_text) if "reason_name" in df.columns else pd.Series([""] * len(df))

        for rid_raw, rname in zip(ids, names):
            if pd.isna(rid_raw):
                continue
            rid = int(rid_raw)
            reason_ids.add(rid)
            reason_count_by_id[rid] = reason_count_by_id.get(rid, 0) + 1
            if rname and rid not in reason_name_by_id:
                reason_name_by_id[rid] = rname

    available = []
    for rid in sorted(reason_ids):
        available.append(
            {
                "reason_id": rid,
                "reason_name": reason_name_by_id.get(rid, rule_name_by_id.get(rid, str(rid))),
                "reason_count": reason_count_by_id.get(rid, 0),
            }
        )
    return available


def _default_auto_group_key(reason_id: int, reason_name: str) -> str:
    """
    自动映射的默认分组规则：
    - 干扰(-1) 与 正常(0) 合并到同一组
    - 其他 reason 各自独立
    """
    reason_name = _to_text(reason_name)
    if reason_id in (-1, 0) or reason_name in {"干扰", "正常", "noisy_normal", "clean_normal"}:
        return "ok_with_noise"
    return f"reason_{reason_id}"


def _build_runtime_from_entries(
    entries: List[Dict[str, Any]],
) -> Tuple[Label, Dict[int, int], Dict[int, str], Dict[int, int]]:
    raw_to_model_mlabel: Dict[int, int] = {}
    for item in entries:
        raw_mlabel = int(item["mlabel_raw"])
        if raw_mlabel not in raw_to_model_mlabel:
            raw_to_model_mlabel[raw_mlabel] = len(raw_to_model_mlabel)

    label_sample_dict: Dict[int, int] = {}
    label_to_mlabel_map: Dict[int, int] = {}
    mlabel_reason_names: Dict[int, List[str]] = {}

    for item in entries:
        reason_id = int(item["reason_id"])
        reason_name = _to_text(item["reason_name"]) or str(reason_id)
        sample = int(item.get("sample", -1))
        model_mlabel = raw_to_model_mlabel[int(item["mlabel_raw"])]

        label_sample_dict[reason_id] = sample
        label_to_mlabel_map[reason_id] = model_mlabel

        mlabel_reason_names.setdefault(model_mlabel, [])
        if reason_name not in mlabel_reason_names[model_mlabel]:
            mlabel_reason_names[model_mlabel].append(reason_name)

    mlabel_mtype_dict = {
        mlabel: "_".join(names) for mlabel, names in sorted(mlabel_reason_names.items())
    }

    reason_name_by_id = {
        int(item["reason_id"]): _to_text(item["reason_name"]) or str(int(item["reason_id"]))
        for item in entries
    }

    results = {}
    reasons = {}
    for model_mlabel in sorted(mlabel_reason_names.keys()):
        result_key = f"mresult_{model_mlabel}"
        results[result_key] = {
            "id": model_mlabel,
            "name": str(model_mlabel),
            "alias": [],
        }
    for reason_id, model_mlabel in sorted(label_to_mlabel_map.items()):
        reason_key = f"reason_{reason_id}"
        reasons[reason_key] = {
            "id": reason_id,
            "name": reason_name_by_id.get(reason_id, str(reason_id)),
            "alias": [],
            "parent": f"mresult_{model_mlabel}",
        }
    labeler = Label({"results": results, "reasons": reasons})

    return labeler, label_sample_dict, mlabel_mtype_dict, label_to_mlabel_map


def build_label_runtime(
    *,
    train_cfg: Dict[str, Any] | None,
    available_reasons: List[Dict[str, Any]],
):
    train_cfg = train_cfg or {}
    reason_resolver = Label(LABEL_RULES).build_reason_resolver(available_reasons)
    reason_name_by_id = reason_resolver.name_by_id

    entries: List[Dict[str, Any]] = []

    mapping_cfg = train_cfg.get("label_mapping", {})
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
        for _, value in mapping_groups.items():
            if isinstance(value, dict):
                normalized_groups.append(value)
            else:
                normalized_groups.append({"reasons": value})

    if mapping_enable and normalized_groups:
        for group_idx, group in enumerate(normalized_groups):
            if not bool(group.get("enable", True)):
                continue

            if auto_mlabel:
                raw_mlabel = group_idx
            else:
                raw_mlabel = _to_int(group.get("mlabel"))
                if raw_mlabel is None:
                    raw_mlabel = group_idx
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
                        reason_id_value=token.get("reason_id"),
                        reason_name_value=token.get("reason"),
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

                entries.append(
                    {
                        "reason_id": rid,
                        "reason_name": rname or reason_name_by_id.get(rid, str(rid)),
                        "mlabel_raw": raw_mlabel,
                        "sample": sample,
                    }
                )

    else:
        if not available_reasons:
            raise RuntimeError("未扫描到可用 reason 标签，无法自动分配 mlabel。")
        group_to_raw_mlabel: Dict[str, int] = {}
        for item in available_reasons:
            rid = _to_int(item.get("reason_id"))
            if rid is None:
                continue
            rname = _to_text(item.get("reason_name")) or reason_name_by_id.get(rid, str(rid))
            group_key = _default_auto_group_key(rid, rname)
            if group_key not in group_to_raw_mlabel:
                group_to_raw_mlabel[group_key] = len(group_to_raw_mlabel)
            entries.append(
                {
                    "reason_id": rid,
                    "reason_name": rname,
                    "mlabel_raw": group_to_raw_mlabel[group_key],
                    "sample": -1,
                }
            )

    dedup: Dict[int, Dict[str, Any]] = {}
    for item in entries:
        dedup[int(item["reason_id"])] = item
    final_entries = list(dedup.values())

    available_reason_ids = {
        rid for rid in (_to_int(item.get("reason_id")) for item in available_reasons) if rid is not None
    }
    if available_reason_ids:
        dropped = [e for e in final_entries if int(e["reason_id"]) not in available_reason_ids]
        if dropped:
            dropped_txt = ", ".join(
                f"{int(e['reason_id'])}:{_to_text(e.get('reason_name')) or int(e['reason_id'])}"
                for e in dropped
            )
            print(f"[WARN] 以下 reason 未出现在特征数据中，训练时忽略: {dropped_txt}")
        final_entries = [e for e in final_entries if int(e["reason_id"]) in available_reason_ids]

    if not final_entries:
        raise RuntimeError("可用标签映射为空，请检查 train.label_mapping。")

    labeler, label_sample_dict, mlabel_mtype_dict, label_to_mlabel_map = _build_runtime_from_entries(
        final_entries
    )

    reason_count_by_id: Dict[int, int] = {}
    for item in available_reasons:
        rid = _to_int(item.get("reason_id"))
        if rid is None:
            continue
        reason_count_by_id[rid] = int(_to_int(item.get("reason_count")) or 0)

    reason_name_by_id = {
        int(e["reason_id"]): _to_text(e.get("reason_name")) or str(int(e["reason_id"]))
        for e in final_entries
    }
    mlabel_to_reason_ids: Dict[int, List[int]] = {}
    for rid, mlabel in label_to_mlabel_map.items():
        mlabel_to_reason_ids.setdefault(int(mlabel), []).append(int(rid))

    print("[INFO] 训练标签映射（mlabel -> merged reasons）:")
    for mlabel in sorted(mlabel_to_reason_ids):
        reason_ids = sorted(mlabel_to_reason_ids[mlabel])
        merged_reason = "_".join(reason_name_by_id.get(rid, str(rid)) for rid in reason_ids)
        sample_values = sorted({label_sample_dict.get(rid, -1) for rid in reason_ids})
        sample_text = str(sample_values[0]) if len(sample_values) == 1 else "/".join(str(v) for v in sample_values)
        total_count = sum(reason_count_by_id.get(rid, 0) for rid in reason_ids)
        reason_ids_text = "[" + ", ".join(str(x) for x in reason_ids) + "]"
        print(
            f"  mlabel={mlabel:>2} | count={total_count:>5} | sample={sample_text:>6} | "
            f"reason_ids={reason_ids_text:<20} | mtype={merged_reason}"
        )

    return labeler, label_sample_dict, mlabel_mtype_dict, label_to_mlabel_map


def build_training_bootstrap(
    config_path: str | Path,
    *,
    default_model_type: str,
) -> TrainingBootstrap:
    config_file = Path(config_path).expanduser()
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    with config_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式错误，顶层必须是 dict: {config_file}")

    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
    if "train" in cfg and not isinstance(cfg.get("train"), dict):
        print("[WARN] train 配置不是字典，已回退为默认训练配置。")

    line_name = _to_text(cfg.get("line_name"))
    if not line_name:
        raise KeyError(f"配置缺少 line_name: {config_file}")

    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    model_name = _to_text(model_cfg.get("model_name"))
    if not model_name:
        raise KeyError(f"配置缺少 model.model_name: {config_file}")

    model_type = _to_text(train_cfg.get("model_type")) or _to_text(default_model_type) or "model"
    train_cfg = dict(train_cfg)
    train_cfg["model_type"] = model_type
    cfg = dict(cfg)
    cfg["train"] = train_cfg
    model_id = resolve_model_id(cfg)

    results_dir = resolve_results_dir(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = str(results_dir)

    from ml.dataset.build import (
        discover_augmented_feature_csvs,
        discover_feature_csvs,
        validate_sample_view_features_alignment,
    )

    base_dataset_files = discover_feature_csvs(results_path)
    if not base_dataset_files:
        expected_dir = results_dir / "dataset_csv"
        raise RuntimeError(
            f"训练前校验失败：未找到特征 CSV（*features*.csv），预期目录: {expected_dir}。"
            "请先执行: python -m ml.dataset.build --config <config.yaml>"
        )

    alignment_report = validate_sample_view_features_alignment(results_path, base_dataset_files)
    print(
        "[INFO] 训练集校验通过: "
        f"sample_view={alignment_report['sample_view_rows']} 行, "
        f"features={alignment_report['feature_rows']} 行, "
        f"unique_keys={alignment_report['unique_keys']}, "
        f"feature_files={alignment_report['feature_files']}, "
        f"key_columns={alignment_report['key_columns']}"
    )

    augmented_dataset_files = discover_augmented_feature_csvs(results_path)
    dataset_files = [*base_dataset_files, *augmented_dataset_files]
    if augmented_dataset_files:
        print(f"[INFO] 加载增强特征: {len(augmented_dataset_files)} 个 CSV")
    available_reasons = _scan_available_reasons(dataset_files)
    if available_reasons:
        print("[INFO] 扫描到可用 reason：")
        for item in available_reasons:
            print(
                f"  reason_id={int(item['reason_id']):>4} | count={int(item.get('reason_count', 0)):>5} | "
                f"reason_name={item['reason_name']}"
            )
    else:
        print("[WARN] 未从特征 CSV 扫描到 reason_id。")

    return TrainingBootstrap(
        config_file=config_file,
        cfg=cfg,
        train_cfg=train_cfg,
        line_name=line_name,
        model_type=model_type,
        model_id=model_id,
        results_path=results_path,
        dataset_files=dataset_files,
        available_reasons=available_reasons,
    )
