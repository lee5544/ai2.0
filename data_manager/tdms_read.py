"""TDMS 读取入口：路径、扫描、压缩、解压、底层打开和信号读取。"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping
from uuid import uuid4

import numpy as np
from nptdms import TdmsFile

try:
    import zstandard as zstd
except Exception:  # pragma: no cover
    zstd = None  # type: ignore[assignment]

try:
    from .line_rules import LINE_RULES as DEFAULT_LINE_RULES, normalize_line_time_value
except ImportError:  # pragma: no cover
    try:
        from data_manager.line_rules import LINE_RULES as DEFAULT_LINE_RULES, normalize_line_time_value
    except ModuleNotFoundError:  # pragma: no cover
        from line_rules import LINE_RULES as DEFAULT_LINE_RULES, normalize_line_time_value


TDMS_SUFFIX = ".tdms"
TDMS_ZST_SUFFIX = ".tdms.zst"
DEFAULT_TDMS_ZST_LEVEL = 22
_COPY_CHUNK_SIZE = 4 * 1024 * 1024
_IN_MEMORY_THRESHOLD_BYTES = 64 * 1024 * 1024

TdmsReadMode = Literal["open", "read", "read_metadata"]


def _path_name(path: str | Path) -> str:
    return Path(path).name


def is_compressed_tdms_path(path: str | Path) -> bool:
    return _path_name(path).lower().endswith(TDMS_ZST_SUFFIX)


def is_uncompressed_tdms_path(path: str | Path) -> bool:
    name = _path_name(path).lower()
    return name.endswith(TDMS_SUFFIX) and not name.endswith(TDMS_ZST_SUFFIX)


def is_tdms_path(path: str | Path) -> bool:
    return is_uncompressed_tdms_path(path) or is_compressed_tdms_path(path)


def tdms_file_suffix(path: str | Path) -> str:
    if is_compressed_tdms_path(path):
        return TDMS_ZST_SUFFIX
    if is_uncompressed_tdms_path(path):
        return TDMS_SUFFIX
    return Path(path).suffix


def tdms_logical_stem(path: str | Path) -> str:
    name = _path_name(path)
    lower_name = name.lower()
    if lower_name.endswith(TDMS_ZST_SUFFIX):
        return name[: -len(TDMS_ZST_SUFFIX)]
    if lower_name.endswith(TDMS_SUFFIX):
        return name[: -len(TDMS_SUFFIX)]
    return Path(name).stem


def compressed_tdms_path(path: str | Path) -> Path:
    path = Path(path)
    if is_compressed_tdms_path(path):
        return path
    if is_uncompressed_tdms_path(path):
        return path.with_name(path.name + ".zst")
    return path.with_name(path.name + TDMS_ZST_SUFFIX)


def iter_uncompressed_tdms_files(root: str | Path) -> Iterator[Path]:
    root = Path(root)
    for path in root.rglob("*"):
        if path.is_file() and not any(part.startswith(".") for part in path.parts):
            if is_uncompressed_tdms_path(path):
                yield path


def iter_tdms_files(root: str | Path) -> Iterator[Path]:
    root = Path(root)
    for path in root.rglob("*"):
        if path.is_file() and not any(part.startswith(".") for part in path.parts):
            if is_tdms_path(path):
                yield path


def _zstd_threads(threads: int | None) -> int:
    if threads is None or threads <= 0:
        return max(1, int(os.cpu_count() or 1))
    return int(threads)


def _require_zstandard() -> None:
    if zstd is None:
        raise RuntimeError("zstandard is required to handle .tdms.zst files")


def _decompress_with_system_zstd(compressed_path: Path, output_path: Path) -> None:
    zstd_bin = shutil.which("zstd")
    if not zstd_bin:
        raise RuntimeError(
            "zstandard is required to handle .tdms.zst files, and `zstd` command was not found"
        )
    try:
        subprocess.run(
            [zstd_bin, "-d", "-q", "-f", "-o", str(output_path), str(compressed_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = f": {(exc.stderr or '').strip()}" if (exc.stderr or "").strip() else ""
        raise RuntimeError(f"zstd decompression failed for {compressed_path}{detail}") from exc


@contextmanager
def materialize_seekable_tdms(
    tdms_path: str | Path,
    *,
    in_memory_threshold_bytes: int = _IN_MEMORY_THRESHOLD_BYTES,
) -> Iterator[str | Path | io.BytesIO]:
    tdms_path = Path(tdms_path)
    if not tdms_path.is_file():
        raise FileNotFoundError(f"TDMS file not found: {tdms_path}")
    if not is_tdms_path(tdms_path):
        raise ValueError(f"Unsupported TDMS path suffix: {tdms_path}")

    if not is_compressed_tdms_path(tdms_path):
        yield tdms_path
        return

    if zstd is None:
        with tempfile.TemporaryDirectory(prefix="tdms_zst_") as temp_dir:
            temp_path = Path(temp_dir) / f"{tdms_logical_stem(tdms_path)}{TDMS_SUFFIX}"
            _decompress_with_system_zstd(tdms_path, temp_path)
            yield temp_path
        return

    with tdms_path.open("rb") as compressed_fp:
        dctx = zstd.ZstdDecompressor()
        if tdms_path.stat().st_size <= int(in_memory_threshold_bytes):
            with dctx.stream_reader(compressed_fp) as reader:
                buffer = io.BytesIO(reader.read())
            try:
                yield buffer
            finally:
                buffer.close()
            return

        with tempfile.TemporaryDirectory(prefix="tdms_zst_") as temp_dir:
            temp_path = Path(temp_dir) / f"{tdms_logical_stem(tdms_path)}{TDMS_SUFFIX}"
            with temp_path.open("wb") as out_fp, dctx.stream_reader(compressed_fp) as reader:
                shutil.copyfileobj(reader, out_fp, length=_COPY_CHUNK_SIZE)
            yield temp_path


@contextmanager
def open_tdms(
    tdms_path: str | Path,
    *,
    mode: TdmsReadMode = "open",
    raw_timestamps: bool = False,
    memmap_dir: str | Path | None = None,
) -> Iterator[TdmsFile]:
    with materialize_seekable_tdms(tdms_path) as seekable_tdms:
        if mode == "open":
            with TdmsFile.open(
                seekable_tdms,
                raw_timestamps=raw_timestamps,
                memmap_dir=memmap_dir,
            ) as tdms_file:
                yield tdms_file
            return
        if mode == "read":
            tdms_file = TdmsFile.read(
                seekable_tdms,
                raw_timestamps=raw_timestamps,
                memmap_dir=memmap_dir,
            )
        elif mode == "read_metadata":
            tdms_file = TdmsFile.read_metadata(seekable_tdms, raw_timestamps=raw_timestamps)
        else:
            raise ValueError(f"Unsupported TDMS read mode: {mode}")
        try:
            yield tdms_file
        finally:
            close = getattr(tdms_file, "close", None)
            if callable(close):
                close()


def _stream_sha1(file_obj) -> tuple[int, str]:
    total_bytes = 0
    hasher = hashlib.sha1()
    while chunk := file_obj.read(_COPY_CHUNK_SIZE):
        total_bytes += len(chunk)
        hasher.update(chunk)
    return total_bytes, hasher.hexdigest()


def _verify_compressed_tdms(
    source_path: Path,
    compressed_path: Path,
    *,
    mode: Literal["none", "size", "sha1"],
) -> None:
    verify_mode = str(mode).strip().lower()
    if verify_mode == "none":
        return
    _require_zstandard()
    with source_path.open("rb") as src_fp:
        source_size, source_sha1 = _stream_sha1(src_fp)
    with compressed_path.open("rb") as compressed_fp, zstd.ZstdDecompressor().stream_reader(compressed_fp) as reader:
        if verify_mode == "size":
            restored_size = sum(len(chunk) for chunk in iter(lambda: reader.read(_COPY_CHUNK_SIZE), b""))
            if restored_size != source_size:
                raise IOError(f"TDMS zst verify failed (size mismatch): {source_path} -> {compressed_path}")
            return
        if verify_mode == "sha1":
            _, restored_sha1 = _stream_sha1(reader)
            if restored_sha1 != source_sha1:
                raise IOError(f"TDMS zst verify failed (sha1 mismatch): {source_path} -> {compressed_path}")
            return
    raise ValueError(f"Unsupported verify mode: {mode}")


def compress_tdms_file(
    source_path: str | Path,
    *,
    target_path: str | Path | None = None,
    level: int = DEFAULT_TDMS_ZST_LEVEL,
    threads: int | None = None,
    overwrite: bool = False,
    remove_source: bool = False,
    verify: Literal["none", "size", "sha1"] = "sha1",
    write_checksum: bool = True,
) -> Path:
    source_path = Path(source_path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"TDMS source file not found: {source_path}")
    if not is_uncompressed_tdms_path(source_path):
        raise ValueError(f"Source must be an uncompressed .tdms file: {source_path}")
    target = compressed_tdms_path(source_path if target_path is None else Path(target_path).expanduser())
    if source_path.resolve() == target.resolve():
        raise ValueError(f"Target path must differ from source path: {source_path}")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Compressed TDMS target already exists: {target}")

    _require_zstandard()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_name(f"{target.name}.tmp.{uuid4().hex}")
    cctx = zstd.ZstdCompressor(
        level=int(level), threads=_zstd_threads(threads), write_checksum=bool(write_checksum)
    )
    try:
        with source_path.open("rb") as src_fp, temp_target.open("wb") as dst_fp, cctx.stream_writer(dst_fp) as writer:
            shutil.copyfileobj(src_fp, writer, length=_COPY_CHUNK_SIZE)
        _verify_compressed_tdms(source_path, temp_target, mode=verify)
        if overwrite and target.exists():
            target.unlink()
        temp_target.replace(target)
        if remove_source:
            source_path.unlink()
        return target
    except Exception:
        temp_target.unlink(missing_ok=True)
        raise


def _parse_filename(filename: str, filename_rule: Mapping[str, Any] | None) -> dict[str, str]:
    if not filename_rule:
        return {}

    parts = tdms_logical_stem(filename).split(filename_rule.get("split", "_"))

    def _pick(index_spec: Any) -> str:
        if isinstance(index_spec, int):
            return parts[index_spec] if 0 <= index_spec < len(parts) else "UNKNOWN"
        if isinstance(index_spec, (list, tuple)):
            values = [parts[i] for i in index_spec if isinstance(i, int) and 0 <= i < len(parts)]
            return "_".join(values) if values else "UNKNOWN"
        return "UNKNOWN"

    return {
        "sn": _pick(filename_rule.get("sn_index")),
        "reference": _pick(filename_rule.get("reference_index")),
        "time": _pick(filename_rule.get("time_index")),
    }


def _resolve_line_rule(
    tdms_path: Path,
    line_rules: Mapping[str, Mapping[str, Any]],
    line: str | None,
) -> tuple[str, Mapping[str, Any]]:
    if line is not None:
        if line not in line_rules:
            raise ValueError(f"Line '{line}' is not defined in LINE_RULES")
        return line, line_rules[line]

    valid_lines = [
        key
        for key, rule in line_rules.items()
        if isinstance(rule, Mapping) and rule.get("filename") and rule.get("channels")
    ]
    if len(valid_lines) == 1:
        only = valid_lines[0]
        return only, line_rules[only]

    path_tokens = {part.lower() for part in tdms_path.parts}
    matched = [key for key in valid_lines if key.lower() in path_tokens]
    if len(matched) == 1:
        name = matched[0]
        return name, line_rules[name]

    stem = tdms_logical_stem(tdms_path)
    if stem.startswith("E-Tilt_") and "etilt1" in line_rules:
        return "etilt1", line_rules["etilt1"]
    if "_L4-C162-" in stem and "epump4" in line_rules:
        return "epump4", line_rules["epump4"]

    raise ValueError("Cannot infer line from tdms_path. Please pass line='...' explicitly.")


def _resolve_channel_rule(channels_rule: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    conditional_rules = channels_rule.get("conditional")
    if not conditional_rules:
        return channels_rule

    for rule in conditional_rules:
        when = rule.get("when", {})
        if when.get("reference") == reference:
            return rule
    for rule in conditional_rules:
        when = rule.get("when", {})
        if when.get("reference") == "*":
            return rule

    raise ValueError(f"No matching conditional rule for reference '{reference}'")


def _read_channel(
    tdms_file: TdmsFile,
    group_name: str,
    channel_name: str,
) -> tuple[np.ndarray, int | None]:
    if group_name not in tdms_file:
        raise KeyError(f"Group '{group_name}' not found in TDMS")

    group = tdms_file[group_name]
    if channel_name not in group:
        raise KeyError(f"Channel '{channel_name}' not found in TDMS group '{group_name}'")

    channel = group[channel_name]
    data = channel[:]
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)

    wf_increment = channel.properties.get("wf_increment")
    sampling_rate = int(round(1.0 / wf_increment)) if wf_increment else None
    return data, sampling_rate


def read_tdms(
    tdms_path: str | Path,
    line_rules: Mapping[str, Mapping[str, Any]] = DEFAULT_LINE_RULES,
    *,
    line: str | None = None,
    required_groups: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    输入 tdms 文件路径和 LINE_RULES，提取配置中的 up/down 加速度信号。

    required_groups:
        仅当提供时，尽量只读取其中命中的 group；
        若未命中 up/down 任一 group，则回退读取 up/down 两路。
    """
    tdms_path = Path(tdms_path)
    if not tdms_path.exists():
        raise FileNotFoundError(f"TDMS file not found: {tdms_path}")

    line_name, line_rule = _resolve_line_rule(tdms_path, line_rules, line)
    filename_meta = _parse_filename(tdms_path.name, line_rule.get("filename"))
    filename_meta["time"] = normalize_line_time_value(
        filename_meta.get("time"),
        line=line_name,
        filename_rule=line_rule.get("filename"),
    )
    channels = _resolve_channel_rule(line_rule["channels"], filename_meta.get("reference", ""))

    up_group = channels["up_group"]
    down_group = channels["down_group"]
    acc_channel = channels["acc_channel"]

    read_up = True
    read_down = True
    if required_groups is not None:
        group_set = {str(g).strip() for g in required_groups if str(g).strip()}
        if group_set:
            known_groups = {up_group, down_group}
            if not group_set.issubset(known_groups):
                # 入参与规则不匹配时回退双路读取，保证行为稳定。
                read_up = True
                read_down = True
            else:
                read_up = up_group in group_set
                read_down = down_group in group_set
                if not read_up and not read_down:
                    # 入参与规则不匹配时回退双路读取，保证行为稳定。
                    read_up = True
                    read_down = True

    up_data: np.ndarray | None = None
    down_data: np.ndarray | None = None
    up_sampling_rate: int | None = None
    down_sampling_rate: int | None = None
    with open_tdms(tdms_path, mode="open") as tdms_file:
        if read_up:
            up_data, up_sampling_rate = _read_channel(tdms_file, up_group, acc_channel)
        if read_down:
            down_data, down_sampling_rate = _read_channel(tdms_file, down_group, acc_channel)

    sampling_rate = up_sampling_rate if up_sampling_rate is not None else down_sampling_rate

    return {
        "line": line_name,
        "sn": filename_meta.get("sn"),
        "reference": filename_meta.get("reference"),
        "time": filename_meta.get("time"),
        "tdms_path": str(tdms_path),
        "acc_channel": acc_channel,
        "up_group": up_group,
        "down_group": down_group,
        "sampling_rate": sampling_rate,
        "up_sampling_rate": up_sampling_rate,
        "down_sampling_rate": down_sampling_rate,
        "up_data": up_data,
        "down_data": down_data,
    }
