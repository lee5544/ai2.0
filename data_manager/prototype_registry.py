"""原型 TDMS 注册：只归集原始文件，不生成 XLSX/WAV/PNG 副本。"""
from __future__ import annotations

import shutil
from pathlib import Path


def register_prototype(
    source_tdms: str | Path,
    *,
    data_root: str | Path,
    line: str,
    overwrite: bool = False,
) -> Path:
    source = Path(source_tdms).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"TDMS 不存在: {source}")
    line_name = str(line or "").strip() or "unknown_line"
    destination = Path(data_root).expanduser().resolve() / "prototype" / line_name / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination
    shutil.copy2(source, destination)
    return destination
