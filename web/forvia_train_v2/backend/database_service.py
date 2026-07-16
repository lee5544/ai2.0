"""Web backend orchestration for creating, appending, and refreshing databases."""

from __future__ import annotations

import sqlite3
import shutil
from pathlib import Path
from typing import Any, Callable

from data_manager.label_database import LabelDatabase, resolve_database_path
from data_manager.label_internal_registry import import_label_csvs
from data_manager.label_forvia_to_internal import convert_forvia_csvs_to_internal
from data_manager.sample_generate import rebuild_metadata
from data_manager.tdms_read import (
    compressed_tdms_path,
    compress_tdms_file,
    is_uncompressed_tdms_path,
    iter_tdms_files,
)


FILE_MODES = {"manual", "copy", "move"}
ACTIONS = {"create", "update", "append", "refresh"}
ProgressCallback = Callable[..., None]


def _progress(
    callback: ProgressCallback | None,
    value: float,
    detail: str,
    *,
    processed: int = 0,
    total: int = 0,
    phase: str = "",
) -> None:
    if callback is not None:
        value = max(0, min(100, float(value)))
        try:
            callback(value, detail, processed, total, phase)
        except TypeError:
            callback(value, detail)


def _path(value: object) -> Path:
    return Path(str(value or "")).expanduser().resolve()


def _require_dir(value: object, title: str) -> Path:
    path = _path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"{title}不存在: {path}")
    return path


def _require_file(value: object, title: str) -> Path:
    path = _path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{title}不存在: {path}")
    return path


def _storage_root_for_path(data_root: Path, tdms_path: Path) -> tuple[str, Path]:
    try:
        relative = tdms_path.resolve().relative_to(data_root.resolve())
    except ValueError as exc:
        raise ValueError(f"TDMS 文件夹必须位于 data_root 内: {tdms_path}") from exc
    if not relative.parts:
        raise ValueError("TDMS 文件夹不能直接等于 data_root")
    storage_root = relative.parts[0]
    return storage_root, data_root / storage_root


def _transfer_tdms_files(
    source_folder: Path,
    target_root: Path,
    *,
    file_mode: str,
    line: str,
    progress: ProgressCallback | None = None,
) -> int:
    if target_root == source_folder or source_folder in target_root.parents:
        raise ValueError("目标 TDMS 根目录不能位于输入文件夹内部")
    files = sorted(iter_tdms_files(source_folder))
    if not files:
        raise ValueError(f"输入文件夹中没有 .tdms 或 .tdms.zst: {source_folder}")
    target_base = target_root / line if line else target_root
    transfers = [
        (
            source,
            compressed_tdms_path(target_base / source.relative_to(source_folder))
            if is_uncompressed_tdms_path(source)
            else target_base / source.relative_to(source_folder),
        )
        for source in files
    ]
    conflicts = [target for _, target in transfers if target.exists()]
    if conflicts:
        raise FileExistsError(f"目标位置已有同名 TDMS 文件: {conflicts[0]}")
    total = len(transfers)
    for index, (source, target) in enumerate(transfers, start=1):
        target.parent.mkdir(parents=True, exist_ok=True)
        if is_uncompressed_tdms_path(source):
            compress_tdms_file(source, target_path=target, remove_source=file_mode == "move")
        elif file_mode == "copy":
            shutil.copy2(source, target)
        else:
            shutil.move(str(source), str(target))
        _progress(
            progress,
            index / total * 100,
            f"已处理文件 {index} / {total}",
            processed=index,
            total=total,
            phase="transfer",
        )
    return len(transfers)


def _compress_uncompressed_tdms_files(
    scan_root: Path,
    *,
    line: str = "",
    progress: ProgressCallback | None = None,
) -> int:
    """Compress existing .tdms files under the registered TDMS root in place."""
    root = scan_root / line if line and (scan_root / line).is_dir() else scan_root
    files = [
        path
        for path in sorted(iter_tdms_files(root))
        if is_uncompressed_tdms_path(path)
    ]
    total = len(files)
    if not total:
        _progress(progress, 42, "未发现需要压缩的 .tdms 文件", phase="compress")
        return 0

    for index, source in enumerate(files, start=1):
        target = compressed_tdms_path(source)
        compress_tdms_file(
            source,
            target_path=target,
            overwrite=True,
            remove_source=True,
        )
        _progress(
            progress,
            8 + index / max(1, total) * 30,
            f"正在压缩 TDMS：已处理 {index} / {total}",
            processed=index,
            total=total,
            phase="compress",
        )
    return total


def _rebuild_database_metadata(
    *,
    data_root: Path,
    storage_root: str,
    line: str,
    workers: int,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    metadata = data_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    LabelDatabase(metadata / "label_records.db")
    return rebuild_metadata(
        data_root=data_root,
        storage_root=storage_root,
        line_override=line,
        dry_run=False,
        workers=workers,
        progress_callback=lambda phase, processed, total: _progress(
            progress,
            (45 + processed / max(1, total) * 15)
            if phase == "manifest"
            else (60 + processed / max(1, total) * 27),
            (
                f"正在生成 tdms_manifest.csv：已处理 {processed} / {total}"
                if phase == "manifest"
                else f"正在生成样本索引：已处理 {processed} / {total}"
            ),
            processed=processed,
            total=total,
            phase=phase,
        ),
    )


def _copy_label_database(source: Path, target: Path) -> None:
    """Copy a live SQLite database safely, including uncheckpointed WAL data."""
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"目标 label_records.db 已存在: {target}")
    with sqlite3.connect(source) as source_db, sqlite3.connect(target) as target_db:
        source_db.backup(target_db)


def execute_database_action(
    action: str,
    *,
    update_kind: str = "",
    label_source_type: str = "none",
    source_folder: str = "",
    label_csvs: list[str] | None = None,
    source_label_db: str = "",
    output_data_root: str = "",
    label_records_db_path: str = "",
    tdms_root: str = "",
    storage_root: str = "factory_raw",
    line: str = "",
    file_mode: str = "manual",
    workers: int = 0,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Create, append, or refresh a sample database and TDMS manifest."""
    action = str(action or "").strip().lower()
    label_source_type = str(label_source_type or "none").strip().lower()
    file_mode = str(file_mode or "").strip().lower()
    line = str(line or "").strip()
    storage_root = str(storage_root or "factory_raw").strip().strip("/")
    if action not in ACTIONS:
        raise ValueError(f"不支持的数据库操作: {action}")
    if file_mode not in FILE_MODES:
        raise ValueError(f"不支持的文件处理方式: {file_mode}")

    _progress(progress, 3, "正在校验路径")
    transferred = 0
    csv_paths = [str(path).strip() for path in (label_csvs or []) if str(path).strip()]
    if action == "create":
        source = _require_dir(source_folder, "TDMS 输入文件夹")
        if file_mode == "manual":
            data_root = _path(output_data_root) if output_data_root else source.parent
            storage_root, scan_root = _storage_root_for_path(data_root, source)
        else:
            data_root = _path(output_data_root)
            if not str(output_data_root or "").strip():
                raise ValueError("复制或移动文件时必须填写输出 data_root")
            scan_root = data_root / storage_root
            db_path = data_root / "metadata" / "label_records.db"
            manifest_path = data_root / "metadata" / "tdms_manifest.csv"
            if db_path.exists() or manifest_path.exists():
                raise FileExistsError(f"新建数据库目标已存在 metadata: {data_root / 'metadata'}")
            transferred = _transfer_tdms_files(
                source,
                scan_root,
                file_mode=file_mode,
                line=line,
                progress=lambda value, detail: _progress(
                    progress,
                    8 + value * 0.27,
                    f"{'复制' if file_mode == 'copy' else '移动'} TDMS：{detail}",
                ),
            )
        db_path = data_root / "metadata" / "label_records.db"
        manifest_path = data_root / "metadata" / "tdms_manifest.csv"
        source_db = _require_file(resolve_database_path(source_label_db), "标签来源 DB") if source_label_db else None
        same_database = bool(source_db and source_db.resolve() == db_path.resolve())
        if (db_path.exists() and not same_database) or (manifest_path.exists() and not same_database):
            raise FileExistsError(f"新建数据库目标已存在 metadata: {data_root / 'metadata'}")
        if source_db is not None:
            _progress(progress, 35, "正在复制标签 DB")
            _copy_label_database(source_db, db_path)
    else:
        db_path = _require_file(
            resolve_database_path(label_records_db_path),
            "label_records.db",
        )
        data_root = db_path.parent.parent
        if db_path.name != "label_records.db":
            raise ValueError("数据库文件名必须为 label_records.db")
        configured_tdms_root = _require_dir(tdms_root, "TDMS 根目录")
        storage_root, scan_root = _storage_root_for_path(data_root, configured_tdms_root)
        if action in {"update", "append"}:
            if str(source_folder or "").strip():
                source = _require_dir(source_folder, "TDMS 输入文件夹")
                if file_mode == "manual":
                    try:
                        source.relative_to(scan_root)
                    except ValueError as exc:
                        raise ValueError(
                            f"手动复制模式下，输入文件夹必须位于当前 TDMS 根目录内: {scan_root}"
                        ) from exc
                else:
                    transferred = _transfer_tdms_files(
                        source,
                        scan_root,
                        file_mode=file_mode,
                        line=line,
                        progress=lambda value, detail: _progress(
                            progress,
                            8 + value * 0.27,
                            f"{'复制' if file_mode == 'copy' else '移动'} TDMS：{detail}",
                        ),
                    )

    _progress(progress, 38, "正在扫描 TDMS 文件")
    if not list(iter_tdms_files(scan_root)):
        raise ValueError(f"TDMS 根目录中没有 .tdms 或 .tdms.zst: {scan_root}")
    compressed_existing = 0
    if action in {"update", "refresh"}:
        compressed_existing = _compress_uncompressed_tdms_files(
            scan_root,
            line=line,
            progress=progress,
        )
        _progress(
            progress,
            42,
            f"TDMS 压缩完成：新增 .tdms.zst {compressed_existing} 个",
            processed=compressed_existing,
            total=compressed_existing,
            phase="compress",
        )
    _progress(
        progress,
        45,
        (f"正在重新注册产线 {line} 的 TDMS 路径与 sample_id" if line else "正在重新注册全部 TDMS 路径与 sample_id")
        if action in {"update", "refresh"}
        else "正在重建 tdms_manifest.csv 与样本索引",
    )
    summary = _rebuild_database_metadata(
        data_root=data_root,
        storage_root=storage_root,
        line=line,
        workers=max(0, int(workers or 0)),
        progress=progress,
    )
    db_path = Path(summary["label_records_db_path"]).resolve()
    registered_sample_ids = (
        len(LabelDatabase(db_path, readonly=True).list_samples(line=line, active_only=True))
        if line
        else int(summary.get("sample_rows") or 0)
    )
    result: dict[str, Any] = {
        "action": action,
        "file_mode": file_mode,
        "transferred_files": transferred,
        "compressed_files": compressed_existing,
        "data_root": str(data_root),
        "tdms_root": str(scan_root),
        "label_records_db_path": str(db_path),
        "manifest_path": str(Path(summary["manifest_path"]).resolve()),
        "tdms_total": int(summary.get("tdms_total") or 0),
        "manifest_rows": int(summary.get("manifest_rows") or 0),
        "sample_rows": int(summary.get("sample_rows") or 0),
        "registered_tdms_paths": int(
            (summary.get("tdms_total") if line else summary.get("manifest_rows")) or 0
        ),
        "registered_sample_ids": registered_sample_ids,
        "errors": list(summary.get("errors") or []),
    }
    if csv_paths and label_source_type not in {"internal", "forvia"}:
        raise ValueError("标签 CSV 只支持来源类型 internal 或 forvia")
    if csv_paths:
        _progress(progress, 88, f"正在导入标签 CSV（0 / {len(csv_paths)}）")
        required_csvs = [_require_file(path, "标签 CSV") for path in csv_paths]
        conversion_report = None
        import_csvs = required_csvs
        if label_source_type == "forvia":
            _progress(progress, 86, "正在转换 Forvia 表为内部标签表", phase="labels")
            conversion_report = convert_forvia_csvs_to_internal(
                required_csvs,
                db_path.parent / "internal_label_imports",
                label_records_db=db_path,
                line=line,
            )
            import_csvs = [Path(path) for path in conversion_report["output_csvs"]]
        result["labels"] = import_label_csvs(
            db_path,
            import_csvs,
            line=line,
            progress=lambda index, total, path: _progress(
                progress,
                88 + index / max(1, total) * 11,
                f"正在导入标签 CSV（{index} / {total}）：{path.name}",
                phase="labels",
            ),
        )
        if conversion_report is not None:
            result["labels"]["forvia_conversion"] = conversion_report
    _progress(progress, 100, "数据库操作完成")
    return result
