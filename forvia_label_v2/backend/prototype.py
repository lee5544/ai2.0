"""Prototype（典型异音件）入库：把标记为典型异音的件复制到 prototype/<line>/，并在 db 标注。

替代旧 PrototypeStore：不再单独存 WAV/PNG，只把原始 tdms 归集到 prototype 目录 + db 标记。
权限：仅在 加载了数据库 且 source=expert 时可操作。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from .config import LABEL_HISTORY_COLUMNS, TYPICAL_TAG


def _safe(part: str) -> str:
    """把 line/source/reason 变成安全的目录名（保留中文，去掉路径分隔等非法字符）。"""
    s = str(part or "").strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    return s or "未知"


def _manifest_csv_path(session) -> Path | None:
    """tdms_manifest.csv 位置：与 label_records.db 同目录（通常 metadata/）。"""
    p = str(getattr(session, "label_records_db_path", "") or "")
    if not p:
        return None
    return Path(p).expanduser().parent / "tdms_manifest.csv"


def update_manifest_paths(manifest_csv, updates: dict[tuple[str, str], str]) -> int:
    """把 (line,sn) -> 绝对 tdms_path 写入 tdms_manifest.csv（命中行更新 tdms_path 并清空相对路径；
    没有则新增行）。返回更新/新增的行数。"""
    import csv as _csv
    if not updates or not manifest_csv:
        return 0
    manifest_csv = Path(manifest_csv)
    norm = {(str(l).strip().lower(), str(s).strip().lower()): (l, s, p) for (l, s), p in updates.items()}
    rows, fieldnames, seen = [], None, set()
    if manifest_csv.exists():
        with open(manifest_csv, encoding="utf-8-sig", newline="") as f:
            r = _csv.DictReader(f)
            fieldnames = list(r.fieldnames or [])
            for row in r:
                key = (str(row.get("line", "")).strip().lower(), str(row.get("sn", "")).strip().lower())
                if key in norm:
                    row["tdms_path"] = norm[key][2]
                    if "relative_path" in row:
                        row["relative_path"] = ""
                    if "storage_root" in row:
                        row["storage_root"] = ""
                    seen.add(key)
                rows.append(row)
    if not fieldnames:
        fieldnames = ["line", "sn", "reference", "storage_root", "relative_path", "tdms_path"]
    for c in ("line", "sn", "tdms_path"):
        if c not in fieldnames:
            fieldnames.append(c)
    n = len(seen)
    for key, (l, s, p) in norm.items():
        if key not in seen:
            rows.append({"line": l, "sn": s, "tdms_path": p}); n += 1
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in fieldnames})
    return n

PROTOTYPE_TAG = "[[prototype]]"


def can_operate(session) -> bool:
    return bool(session.has_db and session.default_source == "expert")


def prototype_root(session) -> Path:
    """prototype 根目录：放在 data_root 下（tdms_root 的上一级）。"""
    tr = Path(session.tdms_root).expanduser()
    base = tr.parent if tr.name else tr      # factory_raw 的上一级 = data_root
    return base / "prototype"


def candidates(session) -> list[dict]:
    """可入 prototype 的候选：标记了典型异音、且 tdms 已解析到的样本（取最新标签）。"""
    latest = session.label_table.latest_label_map()
    out = []
    for sid, lab in latest.items():
        note = str(lab.get("note", "") or "")
        if TYPICAL_TAG not in note:
            continue
        path = session.path_map.get(sid)
        out.append({
            "sample_id": sid, "line": str(lab.get("line", "")),
            "sn": str(lab.get("sn", "")), "reason_name": str(lab.get("reason_name", "")),
            "source": str(lab.get("source", "")),
            "tdms": str(path) if path else "",
            "in_prototype": PROTOTYPE_TAG in note,
            "resolvable": path is not None,
        })
    return out


def import_prototypes(session, sample_ids: list[str], dest_root: str | None = None) -> dict:
    """把选中的典型件 tdms 移动到 <dest_root>/<line>/prototype/<reason中文>/，并在 db 标记 [[prototype]]，
    同时更新 tdms_manifest.csv 的路径指向新位置。
    dest_root 是 line 文件夹的父目录（通常 .../factory_raw）；缺省时用建议根目录。"""
    base = Path(dest_root).expanduser() if dest_root else prototype_root(session)
    latest = session.label_table.latest_label_map()
    copied, marked, errors = [], 0, []
    manifest_updates: dict[tuple[str, str], str] = {}   # (real_line, real_sn) -> 新路径
    for sid in sample_ids:
        lab = latest.get(sid) or {}
        real_line = str(lab.get("line", "") or "unknown_line")
        real_sn = str(lab.get("sn", "") or "")
        line = _safe(real_line)
        reason = _safe(lab.get("reason_name", "") or "未标注")
        src = session.path_map.get(sid)
        if src is None or not Path(src).exists():
            errors.append(f"{sid}: tdms 未解析/不存在")
            continue
        try:
            fname = Path(src).name
            dest_dir = base / line / "prototype" / reason   # <base>/<line>/prototype/<reason>/
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / fname
            if Path(src).resolve() == dest.resolve():
                pass                                  # 已在目标位置
            elif dest.exists():
                pass                                  # 目标已有同名文件，保留原 src，不覆盖
            else:
                shutil.move(str(src), str(dest))      # 移动原始 tdms 到 prototype 目录
            copied.append(str(dest))
            if real_sn:
                manifest_updates[(real_line, real_sn)] = str(dest.resolve())
        except Exception as e:
            errors.append(f"{sid}: 移动失败 {e}")
            continue

    # 更新 db：给这些 sample 的最新标签 note 加 [[prototype]]（单条更新 + 缓存一致，不重写整表）
    for sid in sample_ids:
        try:
            if session.label_table.mark_note_tag(sid, PROTOTYPE_TAG):
                marked += 1
        except Exception as e:
            errors.append(f"{sid}: 标记失败 {e}")

    # 更新 metadata：把这些样本在 tdms_manifest.csv 的 tdms_path 指到新的 prototype 位置
    manifest_updated = 0
    try:
        mc = _manifest_csv_path(session)
        if mc is not None:
            manifest_updated = update_manifest_paths(mc, manifest_updates)
            session.label_table.invalidate_cache()     # manifest 变了，下次刷新重读
    except Exception as e:
        errors.append(f"manifest 更新失败: {e}")
    return {"prototype_root": str(base), "copied": copied, "marked": marked,
            "manifest_updated": manifest_updated, "errors": errors}
