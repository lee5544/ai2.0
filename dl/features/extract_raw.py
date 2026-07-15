"""DL 原始信号特征提取器（raw）。

- 统一预处理：按采样率裁剪信号头尾各 0.5 秒（与 mel/pcen 共用 ``preprocess.trim_edges``）。
- raw 特有：每条信号 z-score 内部标准化。
- 保留 trim 后的**原始长度（不定长）**，不固定到定长——定长裁剪留到训练阶段做。
- 输出 ``[1, L]`` 单通道序列，仅支持 pickle 紧凑列（不定长无法落 CSV 固定列）。

数据管线（读取 sample_view / tdms / 写批次 / 索引）直接复用 ``extract_mel`` 的实现，
本模块只提供 raw 的核心特征函数、分组处理、schema 与 run。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from data_manager.tdms_read import read_tdms

from .preprocess import standardize, trim_edges
from .extract_mel import (
    COMPACT_FEATURE_COLUMN,
    DEFAULT_CONFIG_PATH,
    DM_CFG,
    FEATURE_OUTPUT_METADATA_COLUMNS,
    SCHEMA_FILENAME,
    _build_label_lookup,
    _build_manifest_lookup,
    _build_sample_meta_lookup,
    _collect_sample_view_columns,
    _create_progress,
    _flush_batch,
    _iter_batch_files,
    _pick_signal_by_sample,
    _read_yaml,
    _resolve_batch_size,
    _resolve_dl_cfg,
    _resolve_output_folder,
    _resolve_sample_view_paths,
    _to_int,
    _to_text,
)

DEFAULT_OUTPUT_FILE_PREFIX = "dl_raw"
DEFAULT_OUTPUT_FORMAT = "pickle"  # raw 不定长，仅支持 pickle
DEFAULT_DOWNSAMPLE_STEP = 5  # raw 降采样：每隔 5 个点取一个（20kHz -> 4kHz）；仅本文件下采样


def extract_raw_features(signal: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """原始信号特征：降采样（每隔 step 取一个）+ z-score 标准化（输入已在上游 trim 头尾）。

    返回 (``[1, L]`` float32, meta)。L = 降采样后长度（不定长，训练时再裁剪）。
    """
    sig = np.asarray(signal, dtype=np.float32).reshape(-1)
    if sig.size == 0:
        raise ValueError("空信号，无法提取 raw 特征")
    if DEFAULT_DOWNSAMPLE_STEP > 1:
        sig = sig[::DEFAULT_DOWNSAMPLE_STEP]  # 抽取降采样：每隔 step 取一个，缩短序列
    sig = standardize(sig)
    feature = sig.reshape(1, -1).astype(np.float32, copy=False)
    return feature, {"length": int(sig.size)}


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
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    manifest_meta = manifest_lookup.get((line, sn))
    if not manifest_meta:
        for row_map in rows:
            results.append({
                "status": "fail", "message": "manifest 中未找到对应 line/sn",
                "line": line, "sn": sn, "sample_id": _to_text(row_map.get("sample_id")),
            })
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
            results.append({
                "status": "fail", "message": "sample_index 中未找到对应样本",
                "line": line, "sn": sn, "sample_id": sample_id,
            })
            continue
        sample_group_name = _to_text(sample_meta.get("group_name"))
        if sample_group_name:
            required_groups.add(sample_group_name)
        else:
            has_unknown_group = True
        prepared_rows.append({
            "row_map": row_map, "sample_id": sample_id, "sample_meta": sample_meta,
            "label_ret": label_ret, "sample_group_name": sample_group_name,
        })

    if not prepared_rows:
        return results

    tdms_path = data_root / str(manifest_meta["tdms_storage_root"]) / str(manifest_meta["relative_path"])
    required = None if has_unknown_group else required_groups
    cache_key = (str(tdms_path), line, ",".join(sorted(required)) if required is not None else "*")
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
            sampling_rate = _to_int(sample_meta.get("sampling_rate")) or _to_int(tdms_ret.get("sampling_rate"))
            signal = trim_edges(signal, sampling_rate)  # 统一预处理：裁剪头尾 0.5 秒
            feature_tensor, feature_meta = extract_raw_features(signal)

            record: Dict[str, Any] = {}
            for col in sample_view_columns:
                record[col] = row_map.get(col)
            record.update({
                "tdms_path": str(tdms_path),
                "group_name": mapped_group_name,
                "channel_name": tdms_ret.get("acc_channel"),
                "sampling_rate": sampling_rate,
                "seq_length": int(len(signal)),
                "num_features": int(feature_tensor.size),
                "feature_steps": int(feature_meta["length"]),
                "feature_channels": 1,
            })

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
            record[COMPACT_FEATURE_COLUMN] = np.asarray(feature_tensor, dtype=np.float32, copy=False)
            results.append({"status": "ok", "record": record, "feature_keys": ()})
        except Exception as exc:
            results.append({
                "status": "fail", "message": f"{type(exc).__name__}: {exc}",
                "line": line, "sn": sn, "sample_id": sample_id,
            })
    return results


def _write_feature_schema(*, output_folder: Path, output_file_prefix: str) -> None:
    schema = {
        "feature_version": "raw_signal_v1",
        "output_file_prefix": output_file_prefix,
        "output_format": DEFAULT_OUTPUT_FORMAT,
        "compact_feature_column": COMPACT_FEATURE_COLUMN,
        "channel_names": ["raw"],
        "sequence_length": -1,  # 不定长：训练阶段再裁剪到固定长度
        "feature_columns": {"raw": []},
        "feature_channels": 1,
        "trim_seconds": 0.5,
        "downsample_step": DEFAULT_DOWNSAMPLE_STEP,
        "standardize": "zscore",
    }
    with (output_folder / SCHEMA_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)


def run(args: argparse.Namespace) -> None:
    """raw 特征提取主逻辑，可供 dl/main_dataset.py 直接调用。"""
    wall_t0 = time.perf_counter()
    cfg = _read_yaml(args.config)
    dl_cfg = _resolve_dl_cfg(cfg)
    extract_cfg = dl_cfg.get("extract") if isinstance(dl_cfg.get("extract"), dict) else {}
    output_feature_folder = _resolve_output_folder(cfg, args.output_feature_folder)
    output_feature_folder.mkdir(parents=True, exist_ok=True)
    output_file_prefix = _to_text(args.output_file_prefix) or _to_text(extract_cfg.get("output_file_prefix")) or DEFAULT_OUTPUT_FILE_PREFIX
    output_format = DEFAULT_OUTPUT_FORMAT  # raw 不定长，固定 pickle
    batch_size = _resolve_batch_size(cfg, args.batch_size)

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
        cfg=cfg, args_sample_view=args.sample_view, output_feature_folder=output_feature_folder,
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
    print(f"raw 配置: 按采样率裁剪头尾 0.5 秒 + 每隔 {DEFAULT_DOWNSAMPLE_STEP} 点降采样 + z-score 标准化，保留不定长（训练时裁剪）")
    print(f"输出前缀: {output_file_prefix}")
    print(f"输出格式: {output_format}")
    print(f"批次大小: {batch_size}")
    print(f"并行 worker 数: {max(1, int(args.num_workers))}")

    print("预加载 metadata 索引...")
    sample_meta_lookup = _build_sample_meta_lookup(label_records_db_path)
    manifest_lookup = _build_manifest_lookup(manifest_path)
    label_lookup = _build_label_lookup(label_records_db_path)
    print(f"索引完成: sample_meta={len(sample_meta_lookup)}, manifest={len(manifest_lookup)}, label={len(label_lookup)}")

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
            batch.append(ret["record"])
            ok_rows += 1
            if len(batch) >= batch_size:
                batch_index = _flush_batch(
                    batch=batch, batch_index=batch_index, output_folder=output_feature_folder,
                    output_file_prefix=output_file_prefix, output_format=output_format,
                    fixed_columns=fixed_columns, feature_columns=feature_columns,
                )
                batch.clear()
        elif status == "skip":
            skipped_rows += 1
        else:
            failed_rows += 1
            print(f"处理失败: line={ret.get('line')} sn={ret.get('sn')} sample_id={ret.get('sample_id')} | {ret.get('message')}")

    pending: set = set()
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
        total=total_rows, desc="DL raw 特征提取进度", unit="row",
    ) as progress:
        if invalid_rows > 0:
            progress.update(invalid_rows)
        for (line, sn), rows in grouped_rows.items():
            future = executor.submit(
                _process_one_group,
                line=line, sn=sn, rows=rows, sample_view_columns=sample_view_columns,
                data_root=data_root, sample_meta_lookup=sample_meta_lookup,
                manifest_lookup=manifest_lookup, label_lookup=label_lookup,
                tdms_cache=tdms_cache, tdms_cache_lock=tdms_cache_lock,
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
            batch=batch, batch_index=batch_index, output_folder=output_feature_folder,
            output_file_prefix=output_file_prefix, output_format=output_format,
            fixed_columns=fixed_columns, feature_columns=feature_columns,
        )
        batch.clear()

    _write_feature_schema(output_folder=output_feature_folder, output_file_prefix=output_file_prefix)

    total_sec = float(time.perf_counter() - wall_t0)
    print("")
    print("DL raw 特征提取完成")
    print(f"成功行数: {ok_rows}")
    print(f"跳过行数: {skipped_rows}")
    print(f"失败行数: {failed_rows}")
    print(f"总耗时: {total_sec:.3f}s")
    print(f"特征 schema: {output_feature_folder / SCHEMA_FILENAME}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取 DL 原始信号特征（raw）并保存到 dl_dataset_csv")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=f"YAML 配置路径（默认: {DEFAULT_CONFIG_PATH}）")
    parser.add_argument("--feature-type", default=None, help="DL 特征类型（仅 dl.features 入口使用）")
    parser.add_argument("--sample-view", action="append", default=[], help="sample_view.csv 路径，可重复传入")
    parser.add_argument("--data-root", default=None, help="data_root 路径")
    parser.add_argument("--manifest-path", default=None, help="tdms_manifest.csv 路径")
    parser.add_argument("--label-records-db-path", default=None, help="label_records.db 路径")
    parser.add_argument("--output-feature-folder", default=None, help="特征输出目录")
    parser.add_argument("--output-file-prefix", default=None, help=f"特征批次文件前缀（默认: {DEFAULT_OUTPUT_FILE_PREFIX}）")
    parser.add_argument("--output-format", default=None, help="（raw 固定 pickle，忽略）")
    parser.add_argument("--batch-size", type=int, default=None, help="每个分片的样本数")
    parser.add_argument("--num-workers", type=int, default=max(1, min(8, (os.cpu_count() or 1))), help="并行 worker 数")
    # 兼容 dl/main_dataset 透传的 mel 参数（raw 不使用，占位以免报未识别）
    for extra in ("--n-fft", "--hop-length", "--n-mels", "--max-frames", "--fmin", "--fmax"):
        parser.add_argument(extra, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
