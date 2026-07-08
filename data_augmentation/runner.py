"""Batch raw-signal augmentation. No labels, sample views, or features are handled here."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from data_manager.tdms_read import iter_tdms_files, read_tdms, tdms_logical_stem

from .methods import METHODS
from .tdms_writer import write_augmented_tdms

AUGMENTATION_METHODS = {
    "add_noise": "加噪",
    "white_noise": "随机白噪声",
    "reverberation": "混响",
    "time_frequency_mask": "时频遮挡",
    "speed_perturb": "速度扰动",
    "random_crop": "随机裁剪",
    "random_add_segment": "随机增加片段",
    "random_amplitude": "随机振幅缩放",
    "random_time_stretch": "随机时序伸缩",
}

MANIFEST_COLUMNS = (
    "source_tdms_path",
    "tdms_path",
    "sn",
    "line",
    "augmentation_index",
    "methods",
)

ProgressCallback = Callable[[int, int, str], None]
InputFolder = tuple[str | Path, int]


def _collect_inputs(input_folders: Iterable[InputFolder]) -> list[tuple[Path, int]]:
    """Scan folders and de-duplicate overlapping paths, keeping the largest count."""
    files: dict[Path, int] = {}
    for raw_folder, raw_count in input_folders:
        folder = Path(raw_folder).expanduser().resolve()
        if not folder.is_dir():
            raise NotADirectoryError(f"输入文件夹不存在: {folder}")
        count = int(raw_count)
        if count < 1:
            raise ValueError(f"每个 TDMS 的增强数量必须大于 0: {folder}")
        for path in iter_tdms_files(folder):
            resolved = path.resolve()
            files[resolved] = max(files.get(resolved, 0), count)
    if not files:
        raise FileNotFoundError("输入文件夹中未找到 TDMS / TDMS.ZST 文件")
    return sorted(files.items(), key=lambda item: str(item[0]))


def _apply_methods(data: np.ndarray, methods: list[str], rng: np.random.Generator) -> np.ndarray:
    output = np.asarray(data).copy()
    for method in methods:
        output = METHODS[method](output, rng)
    return output


def _augmented_name(source: Path, sn: str, index: int) -> tuple[str, str]:
    token = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:6]
    new_sn = f"{sn}aug{index:03d}{token}" if sn else f"aug{index:03d}{token}"
    stem = tdms_logical_stem(source)
    new_stem = stem.replace(sn, new_sn, 1) if sn and sn in stem else f"{stem}_{new_sn}"
    return new_sn, f"{new_stem}.tdms"


def _write_manifest(rows: list[dict[str, str | int]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run_augmentation(
    *,
    input_folders: Iterable[InputFolder],
    output_dir: str | Path,
    methods: Iterable[str],
    line: str = "",
    seed: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Read raw TDMS signals, augment both vibration channels, and write new TDMS files."""
    selected_methods = list(dict.fromkeys(str(method).strip() for method in methods if str(method).strip()))
    unknown = [method for method in selected_methods if method not in METHODS]
    if unknown:
        raise ValueError(f"不支持的数据增强方法: {', '.join(unknown)}")
    if not selected_methods:
        raise ValueError("至少选择一种数据增强方法")

    inputs = _collect_inputs(input_folders)
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    total = sum(count for _, count in inputs)
    completed = 0
    rows: list[dict[str, str | int]] = []
    base_rng = np.random.default_rng(seed)

    for source, count in inputs:
        tdms = read_tdms(source, line=line or None)
        for index in range(1, count + 1):
            rng = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))
            replacements = {
                (tdms["up_group"], tdms["acc_channel"]): _apply_methods(tdms["up_data"], selected_methods, rng),
                (tdms["down_group"], tdms["acc_channel"]): _apply_methods(tdms["down_data"], selected_methods, rng),
            }
            new_sn, filename = _augmented_name(source, str(tdms.get("sn") or ""), index)
            output_path = write_augmented_tdms(source, target / filename, replacements)
            rows.append({
                "source_tdms_path": str(source),
                "tdms_path": str(output_path),
                "sn": new_sn,
                "line": str(tdms.get("line") or ""),
                "augmentation_index": index,
                "methods": ",".join(selected_methods),
            })
            completed += 1
            detail = f"已增强 {completed} / {total}"
            print(f"[AUGMENT_PROGRESS] processed={completed} total={total} file={output_path.name}", flush=True)
            if progress:
                progress(completed, total, detail)

    manifest_path = target / "augmentation_manifest.csv"
    _write_manifest(rows, manifest_path)
    return {
        "input_tdms_count": len(inputs),
        "generated_tdms_count": len(rows),
        "output_dir": str(target),
        "manifest_path": str(manifest_path),
        "methods": selected_methods,
    }
