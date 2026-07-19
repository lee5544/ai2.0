"""Augment TDMS signals in memory and write only the final feature CSV."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from data_manager.label_database import load_label_dataframe
from data_manager.tdms_read import iter_tdms_files, read_tdms
from ml.features import get_feature_extractor, resolve_feature_version
from ml.dataset.label_filter import LABEL_FILTER_STATUSES, _build_label_row_maps, _pick_training_label

from .methods import METHODS


LABEL_COLUMNS = (
    "source",
    "timestamp",
    "result_key",
    "result_id",
    "result_name",
    "reason_key",
    "reason_id",
    "reason_name",
    "label_version",
    "note",
)


def _text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _database_path(config: dict[str, Any]) -> Path:
    database = config.get("database") if isinstance(config.get("database"), dict) else {}
    dataset = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    raw = (
        database.get("label_records_db_path")
        or config.get("label_records_db_path")
        or dataset.get("label_records_db_path")
        or ""
    )
    if not raw:
        raise ValueError("配置缺少 database.label_records_db_path")
    return Path(str(raw)).expanduser()


def _collect_inputs(input_folders: Iterable[tuple[str | Path, int]]) -> list[tuple[Path, int]]:
    files: dict[Path, int] = {}
    for raw_folder, raw_count in input_folders:
        folder = Path(raw_folder).expanduser().resolve()
        if not folder.is_dir():
            raise NotADirectoryError(f"输入文件夹不存在: {folder}")
        count = int(raw_count)
        if count < 1:
            raise ValueError(f"每个原始 sample 的增强数量必须大于 0: {folder}")
        for path in iter_tdms_files(folder):
            resolved = path.resolve()
            files[resolved] = max(files.get(resolved, 0), count)
    if not files:
        raise FileNotFoundError("输入文件夹中未找到 TDMS / TDMS.ZST 文件")
    return sorted(files.items(), key=lambda item: str(item[0]))


def _label_lookup(config: dict[str, Any], line: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    labels = load_label_dataframe(_database_path(config), statuses=LABEL_FILTER_STATUSES)
    if labels.empty:
        return {}
    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}
    scope = _text(data_cfg.get("label_scope") or "valid").lower()
    by_triplet, by_pair = _build_label_row_maps(labels)
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, rows in by_triplet.items():
        if line and key[0] != line:
            continue
        if scope == "confirmed":
            rows = [row for row in rows if row.get("_source_category") == "expert"]
        selected, _ = _pick_training_label(rows)
        if selected:
            out[key] = selected
    return out


def _apply_methods(data: np.ndarray, methods: list[str], rng: np.random.Generator) -> np.ndarray:
    output = np.asarray(data).copy()
    for method in methods:
        output = METHODS[method](output, rng)
    return output


def run_direct_feature_augmentation(
    *,
    config_path: str | Path,
    input_folders: Iterable[tuple[str | Path, int]],
    output_dir: str | Path,
    methods: Iterable[str],
    line: str = "",
    seed: int | None = None,
) -> Path:
    config_file = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    selected_methods = list(dict.fromkeys(_text(method) for method in methods if _text(method)))
    unknown = [method for method in selected_methods if method not in METHODS]
    if unknown:
        raise ValueError(f"不支持的数据增强方法: {', '.join(unknown)}")
    if not selected_methods:
        raise ValueError("至少选择一种数据增强方法")

    resolved_line = _text(line) or _text(config.get("line_name"))
    inputs = _collect_inputs(input_folders)
    labels = _label_lookup(config, resolved_line)
    feature_version = resolve_feature_version(config)
    extract_features = get_feature_extractor(feature_version)
    base_rng = np.random.default_rng(seed)
    total = sum(count * 2 for _, count in inputs)
    processed = 0
    records: list[dict[str, Any]] = []

    for source, count in inputs:
        tdms = read_tdms(source, line=resolved_line or None)
        source_line = _text(tdms.get("line"))
        source_sn = _text(tdms.get("sn"))
        source_token = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:6]
        channel_specs = (
            ("up", tdms.get("up_data"), tdms.get("up_group"), tdms.get("up_sampling_rate")),
            ("down", tdms.get("down_data"), tdms.get("down_group"), tdms.get("down_sampling_rate")),
        )
        for side, signal, group_name, sampling_rate in channel_specs:
            source_sample_id = f"{source_sn}_{side}"
            label = labels.get((source_line, source_sn, source_sample_id))
            for index in range(1, count + 1):
                processed += 1
                if signal is not None and label is not None:
                    rng = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))
                    augmented = _apply_methods(np.asarray(signal), selected_methods, rng)
                    extracted = extract_features(augmented, int(sampling_rate), return_timing=True)
                    features = extracted[0] if isinstance(extracted, tuple) else extracted
                    token = f"aug{index:03d}{source_token}"
                    augmented_sn = f"{source_sn}_{token}"
                    record: dict[str, Any] = {
                        "line": source_line,
                        "sn": augmented_sn,
                        "sample_id": f"{augmented_sn}_{side}",
                        "source_sn": source_sn,
                        "source_sample_id": source_sample_id,
                        "reference": _text(tdms.get("reference")),
                        "time": _text(tdms.get("time")),
                        "tdms_path": str(source),
                        "group_name": _text(group_name),
                        "channel_name": _text(tdms.get("acc_channel")),
                        "sampling_rate": int(sampling_rate),
                        "seq_length": int(len(augmented)),
                        "num_features": int(len(features)),
                        "augmentation_index": index,
                        "augmentation_methods": ",".join(selected_methods),
                        "feature_version": feature_version,
                    }
                    for column in LABEL_COLUMNS:
                        output_column = "label_source" if column == "source" else "label_timestamp" if column == "timestamp" else column
                        record[output_column] = label.get(column)
                    record["label"] = label.get("reason_id")
                    record.update(features)
                    records.append(record)
                print(
                    f"[AUGMENT_PROGRESS] processed={processed} total={total} file={source.name}",
                    flush=True,
                )

    if not records:
        raise RuntimeError("输入 TDMS 未匹配到已确认标签，未生成增强特征")
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / "features_batch_augmented.csv"
    pd.DataFrame.from_records(records).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {output_path} | rows={len(records)} | feature_version={feature_version}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="内存增强 TDMS 并直接提取最终特征 CSV")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-folder", action="append", required=True)
    parser.add_argument("--folder-count", action="append", type=int, default=[])
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--line", default="")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folder_count and len(args.folder_count) != len(args.input_folder):
        raise ValueError("--folder-count 数量必须与 --input-folder 一致")
    counts = args.folder_count or [args.count] * len(args.input_folder)
    run_direct_feature_augmentation(
        config_path=args.config,
        input_folders=zip(args.input_folder, counts),
        output_dir=args.output_dir,
        methods=args.methods.split(","),
        line=args.line,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
