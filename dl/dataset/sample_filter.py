"""DL 训练样本筛选（对齐 ml/dataset/sample_filter.py）。

数据源：label_records.db/samples（active）+ tdms_manifest.csv。
按 line / folders / reference / key_sns 取并集筛 manifest，再与 active samples
按 line+sn inner join。只筛样本，不决定训练标签 y（y 见 label_filter.py）。

sample_view 标准列、标准化与输出目录解析统一来自 label_filter（单一来源）。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from data_manager.label_database import load_sample_dataframe, resolve_database_path

from dl.dataset.label_filter import (
    STANDARD_SAMPLE_VIEW_COLUMNS,
    resolve_output_dir,
    standardize_sample_view,
)

OUTPUT_COLUMNS = STANDARD_SAMPLE_VIEW_COLUMNS


def _list(raw: object) -> list[str]:
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    return [str(x).strip() for x in (raw or []) if str(x).strip()]


def _key_items(raw: object, *, default_line: str) -> list[dict[str, str]]:
    values = raw if isinstance(raw, list) else ([raw] if raw else [])
    out: list[dict[str, str]] = []
    for item in values:
        if isinstance(item, dict):
            path = str(item.get("path") or item.get("folder") or item.get("dir") or "").strip()
            line = str(item.get("line") or default_line).strip()
        else:
            path = str(item or "").strip()
            line = ""
        if path:
            out.append({"path": path, "line": line})
    return out


def filter_samples(cfg: dict, output_folder: str | Path | None = None) -> Path:
    """从 manifest + sample registry 中筛选 DL 候选样本。"""
    database = cfg.get("database") if isinstance(cfg.get("database"), dict) else {}
    dataset = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    data = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}

    db_path = resolve_database_path(
        str(database.get("label_records_db_path") or cfg.get("label_records_db_path")
            or dataset.get("label_records_db_path") or "")
    )
    tdms_root = Path(str(database.get("tdms_root") or "")).expanduser()
    manifest_path = Path(str(
        database.get("manifest_path") or dataset.get("manifest_path")
        or db_path.parent / "tdms_manifest.csv"
    )).expanduser()
    for path, title in ((db_path, "label_records.db"), (manifest_path, "tdms_manifest.csv")):
        if not path.is_file():
            raise FileNotFoundError(f"{title} 不存在: {path}")

    samples = load_sample_dataframe(db_path, active_only=True)
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig").fillna("")
    if samples.empty or manifest.empty:
        merged = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        main_mask = pd.Series(False, index=manifest.index)
        selected_folders = _list(data.get("folders") or data.get("folder"))
        references: set[str] = set()
        line_references: set[tuple[str, str]] = set()
        for token in _list(data.get("reference")):
            if "/" in token:
                line, ref = token.split("/", 1)
                line_references.add((line.strip(), ref.strip()))
            else:
                references.add(token)
        lines = set(_list(data.get("line")))
        if not lines and not selected_folders and not references and not line_references:
            lines = set(_list(cfg.get("line_name")))
        if lines:
            main_mask |= manifest["line"].astype(str).isin(lines)
        if references or line_references:
            main_mask |= manifest.apply(
                lambda row: str(row.get("reference") or "").strip() in references
                or (str(row.get("line") or "").strip(), str(row.get("reference") or "").strip()) in line_references,
                axis=1,
            )

        def prefixes_for(values: list[str]) -> list[str]:
            out: list[str] = []
            for raw in values:
                p = Path(raw).expanduser()
                try:
                    out.append(p.resolve().relative_to(tdms_root.resolve()).as_posix().rstrip("/") + "/")
                except Exception:
                    out.append(str(raw).replace("\\", "/").strip("/") + "/")
            return out

        folder_prefixes = prefixes_for(selected_folders)
        key_items = _key_items(data.get("key_sns"), default_line=str(data.get("key_line") or cfg.get("line_name") or "").strip())
        rel = manifest["relative_path"].astype(str)
        if folder_prefixes:
            main_mask |= rel.map(lambda x: any(x.startswith(p) for p in folder_prefixes))
        if not lines and not references and not line_references and not folder_prefixes:
            main_mask |= True
        for item in key_items:
            prefix = prefixes_for([item["path"]])[0]
            key_mask = rel.str.startswith(prefix)
            item_line = item["line"]
            if not item_line:
                parts = prefix.strip("/").split("/")
                item_line = parts[1] if parts and parts[0] == "prototype" and len(parts) > 1 else parts[0] if parts else ""
            if item_line:
                key_mask &= manifest["line"].astype(str).eq(item_line)
            main_mask |= key_mask
        manifest = manifest[main_mask]

        manifest_cols = ["line", "sn", "reference", "time", "tdms_storage_root", "relative_path"]
        for col in manifest_cols:
            if col not in manifest.columns:
                manifest[col] = ""
        manifest_value_cols = [col for col in manifest_cols if col not in {"line", "sn"}]
        sample_base = samples.drop(columns=[col for col in manifest_value_cols if col in samples.columns])
        merged = sample_base.merge(manifest[manifest_cols].drop_duplicates(), on=["line", "sn"], how="inner")
        merged.insert(0, "view_name", "train")
        for col in OUTPUT_COLUMNS:
            if col not in merged.columns:
                merged[col] = ""
        merged = merged[OUTPUT_COLUMNS].drop_duplicates()
        key_cols = ["line", "sn", "sample_id", "group_name", "channel_name", "tdms_storage_root", "relative_path"]
        key_mask = pd.Series(True, index=merged.index)
        for col in key_cols:
            key_mask &= ~merged[col].isna()
            key_mask &= ~merged[col].astype(str).str.strip().str.lower().isin({"", "nan", "none", "null"})
        dropped = int((~key_mask).sum())
        if dropped:
            print(f"[WARN] 丢弃主键字段不完整的 sample_view 行: {dropped}")
        merged = merged[key_mask].copy()

    merged = standardize_sample_view(merged)
    out = (Path(output_folder).expanduser() if output_folder is not None else resolve_output_dir(cfg)) / "sample_view.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {out} | rows={len(merged)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 DL 候选 sample_view（仅样本筛选）")
    parser.add_argument("--config", required=True, help="项目 YAML 配置")
    parser.add_argument("--output-folder", default=None, help="覆盖 DL 输出目录")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).expanduser().read_text(encoding="utf-8")) or {}
    out = Path(args.output_folder).expanduser() if args.output_folder else None
    path = filter_samples(cfg, out)
    print(f"[SAMPLE FILTER] {path}")
