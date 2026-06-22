#!/usr/bin/env python3
"""ML 训练样本筛选。

数据源：
- ``tdms_manifest.csv``：提供 line、reference 和 TDMS 相对路径。
- ``label_records.db/samples``：提供已注册且处于 active 状态的 sample。

筛选规则（来自 YAML ``data``）：
1. ``line``：保留指定产线；可以是字符串或列表。
2. ``folders`` / ``folder``：按 TDMS ``relative_path`` 的文件夹前缀匹配。
3. ``reference``：``REF-A`` 匹配所有产线的 reference；
   ``epump2/REF-A`` 只匹配指定 ``line/reference``。
4. ``line``、``folders``、``reference`` 之间是 OR（并集）关系。
5. 三者均未配置时，优先使用顶层 ``line_name``；``line_name`` 也为空则保留全部 manifest。
6. ``key_sns`` 指定的文件夹作为附加样本范围，与上述结果取并集；
   字典写法可以额外指定 ``line``。
7. 筛选后的 manifest 按 ``line + sn`` 与 active samples 做 inner join，
   只有同时存在于两个数据源中的样本才输出。

本文件只筛选 TDMS/sample，不决定训练标签 y；
y 由同目录的 ``label_filter.py`` 生成。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from data_manager.label_database import load_sample_dataframe, resolve_database_path
from ml.dataset.label_filter import STANDARD_SAMPLE_VIEW_COLUMNS, standardize_sample_view
from ml.training.config import resolve_dataset_output_dir


OUTPUT_COLUMNS = STANDARD_SAMPLE_VIEW_COLUMNS


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 label_records.db + tdms_manifest.csv 生成 sample_view")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


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
    """Select ML candidate samples from manifest and the sample registry."""
    database = cfg.get("database") if isinstance(cfg.get("database"), dict) else {}
    dataset = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    data = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}

    db_path = resolve_database_path(
        str(
            database.get("label_records_db_path")
            or cfg.get("label_records_db_path")
            or dataset.get("label_records_db_path")
            or ""
        )
    )
    tdms_root = Path(str(database.get("tdms_root") or "")).expanduser()
    manifest_path = Path(
        str(
            database.get("manifest_path")
            or dataset.get("manifest_path")
            or db_path.parent / "tdms_manifest.csv"
        )
    ).expanduser()
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
        references = set()
        line_references = set()
        for token in _list(data.get("reference")):
            if "/" in token:
                line, reference = token.split("/", 1)
                line_references.add((line.strip(), reference.strip()))
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
                or (
                    str(row.get("line") or "").strip(),
                    str(row.get("reference") or "").strip(),
                )
                in line_references,
                axis=1,
            )

        def prefixes_for(values: list[str]) -> list[str]:
            out: list[str] = []
            for raw in values:
                p = Path(raw).expanduser()
                try:
                    out.append(
                        p.resolve().relative_to(tdms_root.resolve()).as_posix().rstrip("/") + "/"
                    )
                except Exception:
                    out.append(str(raw).replace("\\", "/").strip("/") + "/")
            return out

        folder_prefixes = prefixes_for(selected_folders)
        key_items = _key_items(
            data.get("key_sns"),
            default_line=str(data.get("key_line") or cfg.get("line_name") or "").strip(),
        )
        rel = manifest["relative_path"].astype(str)
        if folder_prefixes:
            main_mask |= rel.map(lambda x: any(x.startswith(prefix) for prefix in folder_prefixes))
        if not lines and not references and not line_references and not folder_prefixes:
            main_mask |= True
        for item in key_items:
            prefix = prefixes_for([item["path"]])[0]
            key_mask = rel.str.startswith(prefix)
            item_line = item["line"]
            if not item_line:
                parts = prefix.strip("/").split("/")
                item_line = (
                    parts[1]
                    if parts and parts[0] == "prototype" and len(parts) > 1
                    else parts[0] if parts else ""
                )
            if item_line:
                key_mask &= manifest["line"].astype(str).eq(item_line)
            main_mask |= key_mask
        manifest = manifest[main_mask]

        manifest_cols = ["line", "sn", "reference", "time", "tdms_storage_root", "relative_path"]
        for col in manifest_cols:
            if col not in manifest.columns:
                manifest[col] = ""
        merged = samples.merge(manifest[manifest_cols].drop_duplicates(), on=["line", "sn"], how="inner")
        merged.insert(0, "view_name", "train")
        for col in OUTPUT_COLUMNS:
            if col not in merged.columns:
                merged[col] = ""
        merged = merged[OUTPUT_COLUMNS].drop_duplicates()

    merged = standardize_sample_view(merged)
    out = (
        Path(output_folder).expanduser()
        if output_folder is not None
        else resolve_dataset_output_dir(cfg)
    ) / "sample_view.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {out} | rows={len(merged)} | database={db_path} | manifest={manifest_path}")
    return out


def main() -> None:
    args = _args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    filter_samples(cfg)


if __name__ == "__main__":
    main()
