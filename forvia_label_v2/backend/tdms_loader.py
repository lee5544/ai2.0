"""TDMS 加载 + 信号处理（process_data 自带，不依赖 plot_utils）。

对外提供 load_sample(session, index) -> SampleSignals，带按文件路径的 LRU 缓存
（同一 tdms 的 up/down 共用一次读取）。
"""
from __future__ import annotations

import hashlib
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as _config  # noqa: F401  确保 sys.path 注入先于下面的复用导入
from data_manager.tdms_read import read_tdms          # 复用现网读取


def process_data(raw_data, sr, cut_len: int | None = None):
    """裁掉首尾各 0.5 秒对应的点数，再做 RMS 归一化。"""
    sig = raw_data
    cut_len = round(float(sr) * 0.5) if cut_len is None else max(0, int(cut_len))
    sig = sig[cut_len:-cut_len] if len(sig) > 2 * cut_len else sig[:]
    rms = np.sqrt(np.mean(np.asarray(sig, dtype=np.float64) ** 2)) if len(sig) else 0.0
    if np.isfinite(rms) and rms > 0:
        sig = np.asarray(sig, dtype=np.float32) * (0.1 / rms)
    return np.asarray(sig, dtype=np.float32)

# 解压结果磁盘缓存：首次解压 .tdms.zst 后把信号存到本地（SSD），
# 二次访问/新会话/内存 LRU 淘汰后都直接读本地，免再解压外接卷上的 .zst。
# Docker 里指到持久卷（FORVIA_TDMS_CACHE_DIR=/app/_data/tdms_cache），重启后仍命中缓存。
import os as _os
_DISK_CACHE_DIR = Path(_os.environ.get("FORVIA_TDMS_CACHE_DIR")
                       or (Path(tempfile.gettempdir()) / "forvia_tdms_cache"))


def _disk_key(path: Path) -> str:
    try:
        mt = path.stat().st_mtime_ns
    except OSError:
        mt = 0
    return hashlib.md5(f"{path}|{mt}".encode("utf-8")).hexdigest()


def _disk_load(path: Path) -> dict | None:
    f = _DISK_CACHE_DIR / (_disk_key(path) + ".npz")
    if not f.exists():
        return None
    try:
        with np.load(f, allow_pickle=False) as z:   # 用 with 确保关闭，避免句柄/缓冲泄漏
            files = z.files
            return {
                # .copy() 取出后即与 zip 解耦，可安全关闭
                "up_data": z["up"].copy() if "up" in files else None,
                "down_data": z["down"].copy() if "down" in files else None,
                "sampling_rate": int(z["sr"]) if "sr" in files else 20000,
                "line": str(z["line"]) if "line" in files else "",
            }
    except Exception:
        return None


def _disk_store(path: Path, payload: dict) -> None:
    try:
        _DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        f = _DISK_CACHE_DIR / (_disk_key(path) + ".npz")
        if f.exists():
            return
        kw = {"sr": np.asarray(int(payload.get("sampling_rate") or 20000)),
              "line": np.asarray(str(payload.get("line") or ""))}
        up, down = payload.get("up_data"), payload.get("down_data")
        if up is not None:
            kw["up"] = np.asarray(up, dtype=np.float32)
        if down is not None:
            kw["down"] = np.asarray(down, dtype=np.float32)
        tmp = f.with_name(f.name + f".{id(payload)}.tmp.npz")  # 结尾 .npz，避免 savez 再追加后缀
        np.savez(tmp, **kw)          # 不压缩：本地写/读都快
        tmp.replace(f)
        _prune_disk_cache()
    except Exception:
        pass


_DISK_CACHE_MAX_FILES = 200          # 解压缓存最多保留多少个文件，超出删最旧（防磁盘无限增长）


def _prune_disk_cache() -> None:
    try:
        files = [p for p in _DISK_CACHE_DIR.glob("*.npz") if not p.name.endswith(".tmp.npz")]
        if len(files) <= _DISK_CACHE_MAX_FILES:
            return
        files.sort(key=lambda p: p.stat().st_mtime)        # 旧的在前
        for p in files[:len(files) - _DISK_CACHE_MAX_FILES]:
            try:
                p.unlink()
            except OSError:
                pass
    except Exception:
        pass


@dataclass
class SampleSignals:
    index: int
    sn: str
    sample_id: str
    direction: str                # up / down
    line: str
    sampling_rate: int
    raw: np.ndarray | None        # 原始信号
    proc: np.ndarray | None       # 裁剪+归一化信号
    timings: dict | None = None   # {decompress_ms, process_ms, from_disk_cache, from_mem_cache}


# 按 tdms 文件路径缓存 read_tdms 结果（含 up/down 原始信号）。内存占用主要在这里。
# 解压结果有本地磁盘缓存兜底（秒读），所以内存缓存保持很小即可：上限个数 + 总字节。
_TDMS_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_TDMS_CACHE_MAX = 3
_TDMS_CACHE_MAX_BYTES = 350_000_000        # ~350MB，常驻内存维持在很小水平


def _payload_bytes(p: dict) -> int:
    b = 0
    for k in ("up_data", "down_data"):
        a = p.get(k)
        if a is not None:
            try:
                b += int(np.asarray(a).nbytes)
            except Exception:
                pass
    return b


def _tdms_cache_total_bytes() -> int:
    return sum(_payload_bytes(v) for v in _TDMS_CACHE.values())
# 每文件一把锁：up/down 并行请求时只解压一次，第二个等待后直接命中缓存
_PATH_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


_PATH_LOCKS_MAX = 256


def _path_lock(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lk = _PATH_LOCKS.get(key)
        if lk is None:
            # 超量时清掉"已不在内存缓存里的"锁，避免一文件一锁无限增长
            if len(_PATH_LOCKS) >= _PATH_LOCKS_MAX:
                for k in [k for k in _PATH_LOCKS if k not in _TDMS_CACHE and k != key]:
                    _PATH_LOCKS.pop(k, None)
            lk = _PATH_LOCKS[key] = threading.Lock()
        return lk


def _read_tdms_cached(path: Path, line_hint: str | None) -> dict:
    key = str(path)
    c = _TDMS_CACHE.get(key)
    if c is not None:
        _TDMS_CACHE.move_to_end(key)
        return c
    with _path_lock(key):                      # 同一文件串行：避免并发重复解压
        c = _TDMS_CACHE.get(key)               # 拿到锁后二次检查（可能已被别的请求填好）
        if c is not None:
            _TDMS_CACHE.move_to_end(key)
            return c
        payload = _disk_load(path)             # ① 先查本地磁盘缓存（免解压外接卷 .zst）
        if payload is None:
            payload = read_tdms(path, line=(line_hint or None))   # ② 真解压
            _disk_store(path, payload)         # 落本地，供下次/新会话秒读
        _TDMS_CACHE[key] = payload
        _TDMS_CACHE.move_to_end(key)
        # 个数上限 + 总字节上限双重淘汰（至少保留当前这条）
        while len(_TDMS_CACHE) > 1 and (
                len(_TDMS_CACHE) > _TDMS_CACHE_MAX
                or _tdms_cache_total_bytes() > _TDMS_CACHE_MAX_BYTES):
            _TDMS_CACHE.popitem(last=False)
        return payload


def _direction_of(sample_id: str) -> str:
    sid = str(sample_id).lower()
    if sid.endswith("_down"):
        return "down"
    if sid.endswith("_up"):
        return "up"
    return "up"


def load_sample(session, index: int) -> SampleSignals:
    row = session.row(index)
    sid = str(row.get("sample_id", ""))
    sn = str(row.get("sn", ""))
    line_hint = str(row.get("line", "")).strip()
    direction = _direction_of(sid)

    path = session.path_map.get(sid)
    if path is None:
        raise FileNotFoundError(f"未解析到 tdms 文件: sample_id={sid}")

    import time as _t
    p = Path(path)
    from_mem = str(p) in _TDMS_CACHE
    from_disk = (not from_mem) and (_DISK_CACHE_DIR / (_disk_key(p) + ".npz")).exists()
    t0 = _t.perf_counter()
    payload = _read_tdms_cached(p, line_hint)
    decompress_ms = round((_t.perf_counter() - t0) * 1000, 1)

    sr = int(payload.get("sampling_rate") or 20000)
    raw = payload.get("up_data") if direction == "up" else payload.get("down_data")
    raw = np.asarray(raw, dtype=np.float32) if raw is not None else None
    t1 = _t.perf_counter()
    proc = process_data(raw, sr) if raw is not None and len(raw) else None
    process_ms = round((_t.perf_counter() - t1) * 1000, 1)

    return SampleSignals(
        index=index, sn=sn, sample_id=sid, direction=direction,
        line=str(payload.get("line") or line_hint),
        sampling_rate=sr, raw=raw, proc=proc,
        timings={"decompress_ms": decompress_ms, "process_ms": process_ms,
                 "from_mem_cache": from_mem, "from_disk_cache": from_disk},
    )


def downsample(sig: np.ndarray | None, max_points: int = 2000) -> list[float]:
    """用于预览/JSON：等间隔抽样到 max_points 以内。"""
    if sig is None or len(sig) == 0:
        return []
    n = len(sig)
    if n <= max_points:
        return [float(x) for x in sig]
    step = n / max_points
    idx = (np.arange(max_points) * step).astype(int)
    return [float(x) for x in sig[idx]]
