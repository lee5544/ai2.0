"""原型 TDMS 注册：只移动原始文件，不生成 XLSX/WAV/PNG 副本。"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def _safe_path_part(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "_").replace("/", "_")
    return text or "未分类"


def register_prototype(
    source_tdms: str | Path,
    *,
    data_root: str | Path,
    line: str,
    reason: str = "",
    overwrite: bool = False,
) -> Path:
    source = Path(source_tdms).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"TDMS 不存在: {source}")
    line_name = str(line or "").strip() or "unknown_line"
    reason_name = _safe_path_part(reason)
    destination = (
        Path(data_root).expanduser().resolve()
        / "factory_raw"
        / line_name
        / "prototype"
        / reason_name
        / source.name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source == destination:
        return destination
    if destination.exists() and not overwrite:
        if source.stat().st_size == destination.stat().st_size:
            source.unlink()
            return destination
        raise FileExistsError(f"Prototype 目标已存在且大小不同: {destination}")
    if destination.exists() and overwrite:
        destination.unlink()
    shutil.move(str(source), str(destination))
    return destination
