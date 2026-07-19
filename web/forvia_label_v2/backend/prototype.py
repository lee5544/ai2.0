"""Prototype（典型异音件）入库：把标记为典型异音的件复制到 prototype/<line>/，并在 db 标注。

替代旧 PrototypeStore：不再单独存 WAV/PNG，只把原始 tdms 归集到 prototype 目录 + db 标记。
权限：仅在 加载了数据库 且来源属于 expert 类别时可操作。
"""
from __future__ import annotations

import re
import sqlite3
import csv
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from .config import LABEL_HISTORY_COLUMNS, TYPICAL_TAG
from data_manager.label_internal_registry import INTERNAL_LABEL_CSV_COLUMNS, normalize_label_source_category
from data_manager.prototype_registry import register_prototype


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


def _label_db_path(session) -> Path | None:
    p = str(getattr(session, "label_records_db_path", "") or "")
    if not p:
        return None
    db = Path(p).expanduser()
    return db if db.exists() else None


def _data_root(session) -> Path | None:
    db = _label_db_path(session)
    if db is not None and db.parent.name == "metadata":
        return db.parent.parent
    root = str(getattr(session, "tdms_root", "") or "").strip()
    if not root:
        return None
    p = Path(root).expanduser()
    if p.name == "factory_raw":
        return p.parent
    if p.parent.name == "factory_raw":
        return p.parent.parent
    return p.parent


def _path_from_fields(data_root: Path | None, tdms_path: str, storage_root: str, relative_path: str) -> Path | None:
    # 路径唯一来源是 manifest：data_root / storage_root / relative_path。
    # tdms_path 是旧导出字段，不再参与定位，避免绝对路径和根目录重复拼接。
    relative_path = str(relative_path or "").strip()
    if not relative_path:
        return None
    if data_root is None:
        return Path(relative_path).expanduser()
    rel = Path(relative_path.replace("\\", "/"))
    rel_parts = [p for p in rel.parts if p not in ("/", "")]
    if rel.is_absolute() or (rel_parts and rel_parts[0].lower() in {"data_root", "factory_raw"}):
        return None
    storage_root = str(storage_root or "factory_raw").strip() or "factory_raw"
    return data_root / storage_root / rel


def _load_sample_paths(session) -> dict[str, Path]:
    """TDMS 路径只从 tdms_manifest.csv 取；samples 表不再保存路径。"""
    del session
    return {}


def _load_manifest_paths(session) -> dict[tuple[str, str], Path]:
    mc = _manifest_csv_path(session)
    data_root = _data_root(session)
    if mc is None or not mc.exists():
        return {}
    import csv as _csv

    out: dict[tuple[str, str], Path] = {}
    try:
        with open(mc, encoding="utf-8-sig", newline="") as f:
            for row in _csv.DictReader(f):
                line = str(row.get("line", "") or "").strip().lower()
                sn = str(row.get("sn", "") or "").strip().lower()
                if not sn:
                    continue
                p = _path_from_fields(
                    data_root,
                    str(row.get("tdms_path", "") or ""),
                    str(row.get("tdms_storage_root", row.get("storage_root", "")) or ""),
                    str(row.get("relative_path", "") or ""),
                )
                if p is None:
                    continue
                if line:
                    out.setdefault((line, sn), p)
                out.setdefault(("", sn), p)
    except Exception:
        return {}
    return out


def _resolve_candidate_path(
    session,
    sample_id: str,
    line: str,
    sn: str,
    sample_paths: dict[str, Path] | None = None,
    manifest_paths: dict[tuple[str, str], Path] | None = None,
) -> Path | None:
    p = session.path_map.get(sample_id)
    if p is not None:
        return Path(p)
    sample_paths = sample_paths if sample_paths is not None else _load_sample_paths(session)
    p = sample_paths.get(sample_id)
    if p is not None:
        return p
    manifest_paths = manifest_paths if manifest_paths is not None else _load_manifest_paths(session)
    key = (str(line or "").strip().lower(), str(sn or "").strip().lower())
    return manifest_paths.get(key) or manifest_paths.get(("", key[1]))


def _manifest_path_for(
    manifest_paths: dict[tuple[str, str], Path] | None,
    line: str,
    sn: str,
) -> Path | None:
    if not manifest_paths:
        return None
    key = (str(line or "").strip().lower(), str(sn or "").strip().lower())
    if not key[1]:
        return None
    return manifest_paths.get(key) or manifest_paths.get(("", key[1]))


def _resolve_existing_candidate_path(
    session,
    sample_id: str,
    line: str,
    sn: str,
    sample_paths: dict[str, Path] | None = None,
    manifest_paths: dict[tuple[str, str], Path] | None = None,
) -> Path | None:
    path = _resolve_candidate_path(session, sample_id, line, sn, sample_paths, manifest_paths)
    if _exists(path):
        return path
    fallback = _manifest_path_for(manifest_paths, line, sn)
    if fallback is not None and fallback != path and _exists(fallback):
        return fallback
    return path


def _exists(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.exists()
    except Exception:
        return False


def _manifest_fields_for_path(manifest_csv: Path, tdms_path: str) -> dict[str, str]:
    p = Path(tdms_path).expanduser()
    data_root = manifest_csv.parent.parent
    storage_root = "factory_raw"
    try:
        rel = p.resolve().relative_to((data_root / storage_root).resolve()).as_posix()
        return {
            "tdms_storage_root": storage_root,
            "storage_root": storage_root,
            "relative_path": rel,
            "tdms_path": str(p.resolve()),
        }
    except Exception:
        return {"tdms_path": str(p)}


def _load_tagged_label_map(session, tag: str) -> dict[str, dict]:
    """按历史事件查 note 标签；入库脚本追加事件后，不能只看最新事件 note。"""
    db = _label_db_path(session)
    if db is None:
        return {}
    fields = [
        "line", "sn", "sample_id", "timestamp", "source",
        "result_key", "result_id", "result_name",
        "reason_key", "reason_id", "reason_name", "reason_confidence",
        "label_version", "note",
    ]
    out: dict[str, tuple[str, int, dict]] = {}
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        sql = """
            SELECT
                s.line, s.sn, s.sample_id,
                e.timestamp, e.source,
                e.result_key, e.result_id, e.result_name,
                e.reason_key, e.reason_id, e.reason_name, e.reason_confidence,
                e.label_version, e.note, e.id
            FROM label_events e
            JOIN samples s ON s.id = e.sample_pk
            WHERE e.note LIKE ?
            ORDER BY s.sample_id, e.timestamp, e.id
        """
        for row in con.execute(sql, (f"%{tag}%",)):
            sid = str(row["sample_id"] or "").strip()
            if not sid:
                continue
            item = {k: str(row[k] or "") for k in fields}
            cur = out.get(sid)
            ts = str(row["timestamp"] or "")
            event_id = int(row["id"] or 0)
            if cur is None or (ts, event_id) >= (cur[0], cur[1]):
                out[sid] = (ts, event_id, item)
        con.close()
    except Exception:
        return {}
    return {sid: {k: str(row.get(k, "") or "") for k in LABEL_HISTORY_COLUMNS} for sid, (_, _, row) in out.items()}


def _current_sample_ids(session) -> set[str]:
    sample_view = getattr(session, "sample_view", None)
    if sample_view is None:
        return set()
    try:
        values = sample_view["sample_id"]
    except Exception:
        try:
            values = [row.get("sample_id", "") for _, row in sample_view.iterrows()]
        except Exception:
            values = []
    if hasattr(values, "tolist"):
        values = values.tolist()
    return {str(value or "").strip() for value in values if str(value or "").strip()}


def update_manifest_paths(manifest_csv, updates: dict[tuple[str, str], str]) -> int:
    """把 (line,sn) -> TDMS 路径写入 tdms_manifest.csv（优先写 factory_raw 相对路径；
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
                    row.update(_manifest_fields_for_path(manifest_csv, norm[key][2]))
                    seen.add(key)
                rows.append(row)
    if not fieldnames:
        fieldnames = ["line", "sn", "reference", "time", "created_time", "tdms_storage_root", "relative_path", "tdms_path"]
    for c in ("line", "sn", "tdms_storage_root", "relative_path", "tdms_path"):
        if c not in fieldnames:
            fieldnames.append(c)
    n = len(seen)
    for key, (l, s, p) in norm.items():
        if key not in seen:
            rows.append({"line": l, "sn": s, **_manifest_fields_for_path(manifest_csv, p)})
            n += 1
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in fieldnames})
    return n


def _refresh_database_metadata(session, lines: set[str]) -> dict:
    """调用 data_manager 的刷新逻辑，重建 tdms_manifest.csv 与样本索引。"""
    clean_lines = sorted({str(line or "").strip() for line in lines if str(line or "").strip()})
    if not clean_lines:
        return {"ok": True, "skipped": True, "reason": "no_lines"}
    data_root = _data_root(session)
    if data_root is None:
        return {"ok": False, "skipped": True, "reason": "missing_data_root"}
    try:
        from data_manager.sample_generate import rebuild_metadata
    except Exception as exc:
        return {"ok": False, "skipped": True, "reason": f"import_failed: {exc}"}

    summaries = []
    errors = []
    for line in clean_lines:
        try:
            with redirect_stdout(StringIO()):
                summary = rebuild_metadata(
                    data_root=data_root,
                    storage_root="factory_raw",
                    line_override=line,
                    dry_run=False,
                    workers=0,
                )
            summaries.append(
                {
                    "line": line,
                    "manifest_rows": int(summary.get("manifest_rows") or 0),
                    "sample_rows": int(summary.get("sample_rows") or 0),
                    "tdms_total": int(summary.get("tdms_total") or 0),
                    "errors": list(summary.get("errors") or []),
                }
            )
            errors.extend(str(e) for e in (summary.get("errors") or []))
        except Exception as exc:
            errors.append(f"{line}: {type(exc).__name__}: {exc}")
    return {
        "ok": not errors,
        "skipped": False,
        "lines": clean_lines,
        "summaries": summaries,
        "errors": errors,
    }

PROTOTYPE_TAG = TYPICAL_TAG
PROTOTYPE_INTERNAL_LABELS = "prototype_internal_labels.csv"


def can_operate(session) -> bool:
    return bool(session.has_db and normalize_label_source_category(session.default_source) == "expert")


def prototype_root(session) -> Path:
    """导入根目录：line 文件夹的父目录，通常是 data_root/factory_raw。"""
    root = str(getattr(session, "tdms_root", "") or "").strip()
    if root:
        tr = Path(root).expanduser()
        if tr.name == "factory_raw":
            return tr
        if tr.parent.name == "factory_raw":
            return tr.parent
        return tr
    data_root = _data_root(session)
    return (data_root / "factory_raw") if data_root is not None else Path("factory_raw")


def _is_under_line_prototype(path: Path | None, *, line: str, data_root: Path | None) -> bool:
    if path is None or data_root is None:
        return False
    try:
        path.resolve().relative_to((data_root / "factory_raw" / line / "prototype").resolve())
        return True
    except Exception:
        return False


def _internal_label_row(lab: dict, *, view_name: str = "prototype") -> dict[str, str]:
    return {
        "view_name": view_name,
        "line": str(lab.get("line", "") or ""),
        "sn": str(lab.get("sn", "") or ""),
        "sample_id": str(lab.get("sample_id", "") or ""),
        "result_key": str(lab.get("result_key", "") or ""),
        "result_id": str(lab.get("result_id", "") or ""),
        "result_name": str(lab.get("result_name", "") or ""),
        "reason_key": str(lab.get("reason_key", "") or ""),
        "reason_id": str(lab.get("reason_id", "") or ""),
        "reason_name": str(lab.get("reason_name", "") or ""),
        "reason_confidence": str(lab.get("reason_confidence", "") or ""),
        "label_version": str(lab.get("label_version", "") or ""),
        "note": str(lab.get("note", "") or ""),
        "timestamp": str(lab.get("timestamp", "") or ""),
        "source": str(lab.get("source", "") or ""),
    }


def _latest_confirmed_expert_labels_for_prototype(session) -> dict[str, dict]:
    db = _label_db_path(session)
    if db is None:
        return {}
    out: dict[str, dict] = {}
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
                s.line, s.sn, s.sample_id,
                e.timestamp, e.source,
                e.result_key, e.result_id, e.result_name,
                e.reason_key, e.reason_id, e.reason_name, e.reason_confidence,
                e.label_version, e.note, e.id
            FROM label_events e
            JOIN samples s ON s.id = e.sample_pk
            WHERE s.is_active = 1
              AND e.status = 'confirmed'
              AND (
                lower(e.source) = 'expert'
                OR lower(e.source) LIKE 'expert\\_%' ESCAPE '\\'
                OR lower(e.source) LIKE 'expert-%'
                OR lower(e.source) LIKE 'expert:%'
                OR lower(e.source) LIKE 'expert.%'
              )
            ORDER BY s.sample_id, e.timestamp, e.id
            """
        ).fetchall()
        con.close()
    except Exception:
        return {}
    for row in rows:
        sid = str(row["sample_id"] or "")
        out[sid] = {key: str(row[key] or "") for key in (
            "line", "sn", "sample_id", "timestamp", "source",
            "result_key", "result_id", "result_name",
            "reason_key", "reason_id", "reason_name", "reason_confidence",
            "label_version", "note",
        )}
    return out


def write_prototype_internal_tables(session, *, lines: set[str] | None = None) -> dict:
    """Rewrite <factory_raw>/<line>/prototype/prototype_internal_labels.csv."""
    data_root = _data_root(session)
    if data_root is None:
        return {"ok": False, "files": [], "rows": 0, "errors": ["无法确定 data_root"]}
    sample_paths = _load_sample_paths(session)
    manifest_paths = _load_manifest_paths(session)
    rows_by_line: dict[str, dict[str, dict[str, str]]] = {}

    typical = _load_tagged_label_map(session, TYPICAL_TAG)
    for sid, lab in typical.items():
        line = str(lab.get("line", "") or "").strip()
        sn = str(lab.get("sn", "") or "").strip()
        if not line or (lines and line not in lines):
            continue
        path = _resolve_existing_candidate_path(session, sid, line, sn, sample_paths, manifest_paths)
        if not _is_under_line_prototype(path, line=line, data_root=data_root):
            continue
        rows_by_line.setdefault(line, {})[sid] = _internal_label_row(lab)

    for sid, lab in _latest_confirmed_expert_labels_for_prototype(session).items():
        line = str(lab.get("line", "") or "").strip()
        sn = str(lab.get("sn", "") or "").strip()
        if not line or (lines and line not in lines):
            continue
        path = _resolve_existing_candidate_path(session, sid, line, sn, sample_paths, manifest_paths)
        if not _is_under_line_prototype(path, line=line, data_root=data_root):
            continue
        rows_by_line.setdefault(line, {}).setdefault(sid, _internal_label_row(lab))

    line_dirs = []
    factory_raw = data_root / "factory_raw"
    if factory_raw.is_dir():
        for path in sorted(factory_raw.iterdir()):
            proto_dir = path / "prototype"
            if path.name == "unknown_line":
                continue
            if path.is_dir() and proto_dir.is_dir() and (not lines or path.name in lines):
                line_dirs.append((path.name, proto_dir))
    for line in sorted(rows_by_line):
        proto_dir = factory_raw / line / "prototype"
        if proto_dir.is_dir() and (line, proto_dir) not in line_dirs:
            line_dirs.append((line, proto_dir))

    files = []
    errors = []
    total_rows = 0
    for line, proto_dir in line_dirs:
        rows = sorted(
            rows_by_line.get(line, {}).values(),
            key=lambda row: (row.get("sn", ""), row.get("sample_id", ""), row.get("timestamp", "")),
        )
        out_path = proto_dir / PROTOTYPE_INTERNAL_LABELS
        try:
            with out_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=INTERNAL_LABEL_CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            files.append({"line": line, "path": str(out_path), "rows": len(rows)})
            total_rows += len(rows)
        except Exception as exc:
            errors.append(f"{line}: {type(exc).__name__}: {exc}")
    return {"ok": not errors, "files": files, "rows": total_rows, "errors": errors}


def update_prototypes(session) -> dict:
    """Refresh prototype metadata and rewrite each line's internal label table."""
    data_root = _data_root(session)
    if data_root is None:
        return {"ok": False, "errors": ["无法确定 data_root"]}
    lines = {
        path.name
        for path in (data_root / "factory_raw").iterdir()
        if path.is_dir() and (path / "prototype").is_dir()
    } if (data_root / "factory_raw").is_dir() else set()
    metadata_refresh = _refresh_database_metadata(session, lines)
    try:
        session.label_table.invalidate_cache()
    except Exception:
        pass
    internal_tables = write_prototype_internal_tables(session, lines=lines or None)
    errors = []
    if not metadata_refresh.get("ok", False):
        errors.extend(metadata_refresh.get("errors") or [str(metadata_refresh.get("reason") or "数据库刷新失败")])
    if not internal_tables.get("ok", False):
        errors.extend(internal_tables.get("errors") or [])
    return {
        "ok": not errors,
        "lines": sorted(lines),
        "metadata_refresh": metadata_refresh,
        "internal_tables": internal_tables,
        "errors": errors,
    }


def candidates(session) -> list[dict]:
    """可入 prototype 的候选：标记了典型异音、且 tdms 已解析到的样本（取最新标签）。"""
    open_ids = _current_sample_ids(session)
    if not open_ids:
        return []
    latest = {
        sid: lab for sid, lab in session.label_table.latest_label_map().items()
        if sid in open_ids
    }
    typical = {
        sid: lab for sid, lab in _load_tagged_label_map(session, TYPICAL_TAG).items()
        if sid in open_ids
    }
    if not typical:
        typical = {
            sid: lab for sid, lab in latest.items()
            if TYPICAL_TAG in str(lab.get("note", "") or "")
        }
    prototype_ids = set(_load_tagged_label_map(session, PROTOTYPE_TAG)) & open_ids
    sample_paths = _load_sample_paths(session)
    manifest_paths = _load_manifest_paths(session)
    data_root = _data_root(session)
    out = []
    for sid, lab in typical.items():
        latest_note = str((latest.get(sid) or {}).get("note", "") or "")
        line = str(lab.get("line", ""))
        sn = str(lab.get("sn", ""))
        path = _resolve_existing_candidate_path(session, sid, line, sn, sample_paths, manifest_paths)
        prototype_marked = sid in prototype_ids or PROTOTYPE_TAG in latest_note
        in_prototype = (
            prototype_marked
            and _exists(path)
            and _is_under_line_prototype(path, line=line, data_root=data_root)
        )
        out.append({
            "sample_id": sid, "line": line,
            "sn": sn, "reason_name": str(lab.get("reason_name", "")),
            "source": str(lab.get("source", "")),
            "tdms": str(path) if path else "",
            "in_prototype": in_prototype,
            "resolvable": _exists(path),
        })
    return out


def import_prototypes(session, sample_ids: list[str], dest_root: str | None = None) -> dict:
    """把选中的典型件 tdms 移动到 <dest_root>/<line>/prototype/<reason中文>/，并在 db 标记 [[prototype]]，
    同时更新 tdms_manifest.csv 的路径指向新位置。
    dest_root 是 line 文件夹的父目录（通常 .../factory_raw）；缺省时用建议根目录。"""
    base = Path(dest_root).expanduser() if dest_root else prototype_root(session)
    latest = session.label_table.latest_label_map()
    copied, marked, errors = [], 0, []
    successful_ids: set[str] = set()
    manifest_updates: dict[tuple[str, str], str] = {}   # (real_line, real_sn) -> 新路径
    refresh_lines: set[str] = set()
    sample_paths = _load_sample_paths(session)
    manifest_paths = _load_manifest_paths(session)
    for sid in sample_ids:
        lab = latest.get(sid) or {}
        real_line = str(lab.get("line", "") or "unknown_line")
        real_sn = str(lab.get("sn", "") or "")
        line = _safe(real_line)
        reason = _safe(lab.get("reason_name", "") or "未标注")
        src = _resolve_existing_candidate_path(session, sid, real_line, real_sn, sample_paths, manifest_paths)
        if src is None or not Path(src).exists():
            errors.append(f"{sid}: tdms 未解析/不存在")
            continue
        try:
            registry_data_root = base.parent if base.name == "factory_raw" else base
            dest = register_prototype(
                src,
                data_root=registry_data_root,
                line=line,
                reason=reason,
                overwrite=False,
            )
            copied.append(str(dest))
            successful_ids.add(sid)
            sample_paths[sid] = dest
            try:
                session.path_map[sid] = dest
            except Exception:
                pass
            if real_sn:
                manifest_updates[(real_line, real_sn)] = str(dest.resolve())
                manifest_paths[(str(real_line).strip().lower(), str(real_sn).strip().lower())] = dest
                manifest_paths[("", str(real_sn).strip().lower())] = dest
            if real_line:
                refresh_lines.add(real_line)
        except Exception as e:
            errors.append(f"{sid}: 移动失败 {e}")
            continue

    # 更新 db：给这些 sample 的最新标签 note 加 [[prototype]]（单条更新 + 缓存一致，不重写整表）
    for sid in successful_ids:
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
    except Exception as e:
        errors.append(f"manifest 更新失败: {e}")

    metadata_refresh = {"ok": True, "skipped": True, "reason": "no_copied_files"}
    if copied:
        metadata_refresh = _refresh_database_metadata(session, refresh_lines)
        if not metadata_refresh.get("ok", False):
            errors.append("数据库刷新失败: " + "; ".join(metadata_refresh.get("errors") or []))
    try:
        session.label_table.invalidate_cache()     # manifest / samples 变了，下次刷新重读
    except Exception:
        pass
    internal_tables = write_prototype_internal_tables(session, lines=refresh_lines or None)
    if not internal_tables.get("ok", False):
        errors.append("Prototype 内部表更新失败: " + "; ".join(internal_tables.get("errors") or []))
    return {"prototype_root": str(base), "copied": copied, "marked": marked,
            "manifest_updated": manifest_updated, "metadata_refresh": metadata_refresh,
            "internal_tables": internal_tables, "errors": errors}
