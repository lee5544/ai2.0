"""DL 数据集构建（生成 sample_view + mel 特征提取 CLI）。

样本筛选见 sample_filter.py，标签筛选见 label_filter.py（sample_filter 复用 label_filter 的
标准列与输出目录，单一来源）。本模块把二者串起来生成 sample_view，并提取 mel 窗口特征 (X, y)。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from data_manager.label_database import load_label_dataframe, resolve_database_path

from dl.dataset.label_filter import LABEL_FILTER_STATUSES, filter_sample_view_dataframe, resolve_output_dir
from dl.dataset.sample_filter import filter_samples

# 所有特征类型（mel/pcen/raw）统一用固定前缀，训练侧据此读取，互不混淆
DL_FEATURE_PREFIX = "dl_feature"

def _read_yaml_cfg(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser()
    if not config_path.exists() and not config_path.is_absolute():
        cfg_candidate = Path("cfg") / config_path
        if cfg_candidate.exists():
            config_path = cfg_candidate
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件不是字典结构: {config_path}")
    return cfg


def _database_path(cfg: dict) -> Path:
    database = cfg.get("database") if isinstance(cfg.get("database"), dict) else {}
    dataset = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    raw = (database.get("label_records_db_path") or cfg.get("label_records_db_path")
           or dataset.get("label_records_db_path") or "")
    return resolve_database_path(str(raw))


def generate(cfg: dict, output_folder: str | Path | None = None) -> Path:
    """生成 DL 训练 sample_view.csv（样本筛选 + 标签筛选）。"""
    candidate_path = filter_samples(cfg, output_folder)
    candidates = pd.read_csv(candidate_path, encoding="utf-8-sig")
    labels = load_label_dataframe(_database_path(cfg), statuses=LABEL_FILTER_STATUSES)
    filtered, _stats = filter_sample_view_dataframe(candidates, labels, cfg)
    filtered.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    print(f"[DL TRAINING DATA] {candidate_path} | candidates={len(candidates)} | kept={len(filtered)}")
    return candidate_path


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    import math
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _resolve_dl_output_folder(cfg: dict[str, Any]) -> Path:
    return resolve_output_dir(cfg)


def _parse_dataset_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DL 数据集生成（sample_view + 特征提取）")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    parser.add_argument("--step", choices=["all", "sample-view", "extract"], default="all",
                        help="all=全流程（默认），sample-view=仅生成 sample_view，extract=仅提取特征")
    parser.add_argument("--output-folder", default=None, help="覆盖 DL 输出目录")
    parser.add_argument("--feature-type", default=None,
                        help="特征提取方法：mel/pcen/raw（默认读 cfg 的 dl.feature_type）")
    parser.add_argument("--num-workers", type=int, default=None, help="并行 worker 数")
    parser.add_argument("--batch-size", type=int, default=None, help="特征分片大小")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[WARN] 忽略未识别参数: {unknown}")
    return args


def _run_sample_view(cfg: dict[str, Any], output_folder: Path) -> Path:
    print(f"\n{'='*60}\n[Step 1] 生成 sample_view.csv\n  输出目录: {output_folder}\n{'='*60}")
    output_folder.mkdir(parents=True, exist_ok=True)
    sv_path = generate(cfg, output_folder)
    print(f"[Step 1] 完成: {sv_path}\n")
    return sv_path


def _run_extract(
    config_path: Path,
    output_folder: Path,
    *,
    num_workers: int | None,
    batch_size: int | None,
    feature_type: str | None = None,
) -> None:
    from dl.features import run as ewf_run, resolve_extractor_name
    selected = feature_type or resolve_extractor_name(_read_yaml_cfg(config_path))
    print(f"\n{'='*60}\n[Step 2] 提取特征（feature_type={selected}）\n  输出目录: {output_folder}\n{'='*60}")
    import argparse as _ap
    fake_args = _ap.Namespace(
        config=str(config_path),
        feature_type=feature_type,  # 显式传入则覆盖 cfg 的 dl.feature_type
        output_feature_folder=str(output_folder),
        sample_view=[str(output_folder / "sample_view.csv")],
        num_workers=num_workers if num_workers is not None else max(1, min(8, (os.cpu_count() or 1))),
        batch_size=batch_size,
        output_file_prefix=DL_FEATURE_PREFIX, output_format=None, data_root=None,
        manifest_path=None, label_records_db_path=None,
        n_fft=None, hop_length=None, n_mels=None, max_frames=None, fmin=None, fmax=None,
    )
    ewf_run(fake_args)
    print(f"[Step 2] 完成\n")


def main(argv: list[str] | None = None) -> None:
    import shutil

    args = _parse_dataset_args(argv)
    config_path = Path(args.config).expanduser()
    cfg = _read_yaml_cfg(config_path)
    output_folder = Path(args.output_folder).expanduser() if args.output_folder else _resolve_dl_output_folder(cfg)

    print(f"[INFO] DL 数据集生成\n  config: {config_path}\n  step: {args.step}\n  output: {output_folder}")

    # 重新生成数据集时，先删除整个 dl_dataset_csv 目录再重建，避免新旧/不同特征文件混在一起
    if args.step in ("all", "sample-view") and output_folder.exists():
        shutil.rmtree(output_folder, ignore_errors=True)
        print(f"[INFO] 已删除旧数据集目录: {output_folder}")

    if args.step in ("all", "sample-view"):
        _run_sample_view(cfg, output_folder)

    if args.step in ("all", "extract"):
        sv_path = output_folder / "sample_view.csv"
        if not sv_path.exists():
            raise FileNotFoundError(f"sample_view.csv 不存在: {sv_path}")
        _run_extract(config_path, output_folder, num_workers=args.num_workers, batch_size=args.batch_size,
                     feature_type=args.feature_type)

    print(f"[INFO] 全部完成。输出目录: {output_folder}")


if __name__ == "__main__":
    main()
