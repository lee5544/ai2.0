"""FastAPI 后端：样本标签总览面板 + 会话初始化。

接口：
  POST /api/init                     -> 设置 sample_view / tdms_root / 可选标签表，重建会话
  GET  /api/config                   -> 当前会话的数据源与状态
  GET  /api/overview                 -> 总览行 + 统计 + 线名
  GET  /api/sample/{index}           -> 单样本详情（标注视图占位，后续接 tdms/卡片）
  GET  /                             -> 前端页面

运行：uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .card_api import (CardContext, default_card_ids, discover, list_cards,
                       render_card)
from . import task_api, task_runner
from .config import (DEFAULT_LABEL_RECORDS_DB_PATH, DEFAULT_SAMPLE_VIEW_PATH,
                     DEFAULT_TDMS_ROOT)
from .session import get_session, init_session
from .tdms_loader import downsample, load_sample

app = FastAPI(title="Forvia 标注 v2")

def _find_frontend_dir() -> Path:
    """前端目录：开发时在 backend 上一级；打包后 PyInstaller 把它放在
    <_MEIPASS>/forvia_label_v2/frontend（见 .spec datas）。逐个候选取第一个存在的。"""
    cands = [Path(__file__).resolve().parent.parent / "frontend"]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cands += [Path(meipass) / "forvia_label_v2" / "frontend",
                  Path(meipass) / "frontend"]
    repo = os.environ.get("FORVIA_REPO_ROOT")
    if repo:
        cands.append(Path(repo) / "forvia_label_v2" / "frontend")
    for c in cands:
        if (c / "index.html").exists():
            return c
    return cands[0]


FRONTEND_DIR = _find_frontend_dir()


import warnings as _warnings
# librosa 在 n_mels 相对采样率/分辨率偏高时会发 "Empty filters detected" 提醒，
# 对结果无影响（仅个别 mel 滤波器为空），屏蔽以保持日志清爽。
_warnings.filterwarnings("ignore", message="Empty filters detected in mel frequency basis")


def _warmup_jit():
    """后台预热 librosa/numba JIT：首次梅尔频谱等会现场编译约 5s，
    在启动期用小假信号各跑一次，用户点开样本时即已编译完成（~6ms）。"""
    try:
        import numpy as np
        import librosa
        x = (np.random.randn(20000) * 0.1).astype("float32")
        # 触发 librosa/numba 的 JIT 预编译（mel/mfcc），消除首条样本的冷启动延迟
        for fn in (lambda: librosa.feature.melspectrogram(y=x, sr=25000, n_mels=26,
                                                          n_fft=256, hop_length=128),
                   lambda: librosa.feature.mfcc(y=x, sr=25000, n_mfcc=13,
                                                n_fft=256, hop_length=128)):
            try:
                fn()
            except Exception:
                pass
    except Exception:
        pass


@app.on_event("startup")
def _startup():
    discover()              # 扫描 cards/ 注册卡片
    task_api.discover()     # 扫描 tasks/ 注册任务
    init_session(DEFAULT_SAMPLE_VIEW_PATH, DEFAULT_TDMS_ROOT, DEFAULT_LABEL_RECORDS_DB_PATH)
    import threading
    threading.Thread(target=_warmup_jit, daemon=True).start()   # 后台预热，不阻塞启动


# 来源列表由 data_manager.label_internal_registry 统一提供。
# 人工标注只用非 model 来源（model 为任务/模型来源）。
try:
    from data_manager.label_internal_registry import SOURCE as _RULE_SOURCES
    SOURCE_OPTIONS = [s for s in _RULE_SOURCES if s != "model"] or ["operator", "expert"]
except Exception:
    SOURCE_OPTIONS = ["operator", "expert"]


class InitReq(BaseModel):
    name: str = ""                     # 任务名（用于保存到"我的任务"）
    sample_view_path: str = ""
    tdms_root: str = ""
    label_records_db_path: str = ""   # 空 = 不加载标签（从空表开始）
    source: str = "expert"             # 默认标注来源 operator / expert


@app.post("/api/init")
def api_init(req: InitReq):
    try:
        s = init_session(req.sample_view_path or None,
                         req.tdms_root or None,
                         req.label_records_db_path or None,
                         source=req.source or "expert")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"初始化失败: {e}")
    # 成功后自动保存为"任务"，下次可一键进入（仅当指定了真实路径）
    if req.sample_view_path or req.tdms_root:
        from . import projects_store
        projects_store.add_project(req.model_dump())
    return {"ok": True, **api_config()}


# ============ 已保存的任务/项目（数据源配置持久化）============
from . import projects_store


_PATH_LABELS = {"sample_view_path": "sample_view", "tdms_root": "tdms 目录",
                "label_records_db_path": "数据库"}


def _project_missing_paths(p: dict) -> list[dict]:
    """返回该任务里"填了但当前不存在"的路径（用于置灰任务上的具体提示）。"""
    missing = []
    for k in ("sample_view_path", "tdms_root", "label_records_db_path"):
        v = p.get(k, "")
        if v and not Path(v).expanduser().exists():
            missing.append({"field": _PATH_LABELS[k], "path": str(v)})
    return missing


def _project_valid(p: dict) -> bool:
    """检测任务的数据源路径是否仍有效（指定了的路径都必须存在）。"""
    if _project_missing_paths(p):
        return False
    # 至少要有 sample_view 或 tdms_root
    return bool(p.get("sample_view_path") or p.get("tdms_root"))


@app.get("/api/projects")
def api_projects():
    items = projects_store.list_projects()
    for p in items:
        p["missing_paths"] = _project_missing_paths(p)
        p["valid"] = _project_valid(p)
    return {"projects": items}


class SaveProjectReq(BaseModel):
    name: str = ""
    sample_view_path: str = ""
    tdms_root: str = ""
    label_records_db_path: str = ""
    source: str = "operator"


@app.post("/api/projects")
def api_projects_save(req: SaveProjectReq):
    if not req.sample_view_path and not req.tdms_root:
        raise HTTPException(status_code=400, detail="无可保存的路径")
    return {"ok": True, "project": projects_store.add_project(req.model_dump())}


@app.post("/api/projects/{pid}/open")
def api_projects_open(pid: str):
    """点击已保存任务：用其路径重建会话并标记最近使用。"""
    p = projects_store.get_project(pid)
    if p is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    try:
        init_session(p.get("sample_view_path") or None,
                     p.get("tdms_root") or None,
                     p.get("label_records_db_path") or None,
                     source=p.get("source") or "operator")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"打开任务失败: {e}")
    projects_store.touch(pid)
    return {"ok": True, **api_config()}


@app.delete("/api/projects/{pid}")
def api_projects_delete(pid: str):
    return {"ok": projects_store.delete_project(pid)}


@app.get("/api/roots")
def api_roots():
    """常用起始位置：用户主目录、当前会话已用目录、根。"""
    s = get_session()
    roots = []
    # 容器内挂载的数据目录（Docker 默认把宿主数据挂到 /data）。放最前，方便直接浏览。
    data_mount = os.environ.get("FORVIA_DATA_MOUNT", "/data")
    if Path(data_mount).is_dir():
        roots.append({"label": f"数据目录 {data_mount}", "path": data_mount})
    home = str(Path.home())
    roots.append({"label": "主目录 ~", "path": home})
    if s.tdms_root:
        roots.append({"label": "当前 TDMS 目录", "path": s.tdms_root})
    if s.sample_view_path:
        roots.append({"label": "sample_view 目录", "path": str(Path(s.sample_view_path).parent)})
    roots.append({"label": "根 /", "path": "/"})
    # 去重（按 path）
    seen, out = set(), []
    for r in roots:
        if r["path"] and r["path"] not in seen:
            seen.add(r["path"]); out.append(r)
    return {"roots": out}


@app.get("/api/browse")
def api_browse(path: str = ""):
    """网页文件浏览器：列出目录下的子目录与 csv 文件（浏览后端本机文件系统）。"""
    base = Path(path).expanduser() if path else Path.home()
    try:
        base = base.resolve()
    except Exception:
        base = Path.home()
    if not base.exists() or not base.is_dir():
        raise HTTPException(status_code=404, detail=f"目录不存在: {base}")
    dirs, files = [], []
    truncated = False
    try:
        entries = list(base.iterdir())
    except PermissionError:
        raise HTTPException(status_code=403, detail="无权限读取该目录")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"读取目录失败: {e}")
    # 海量目录截断，避免前端卡顿
    LIMIT = 4000
    if len(entries) > LIMIT:
        truncated = True
        entries = entries[:LIMIT]
    for e in entries:
        try:
            if e.name.startswith("."):
                continue
            if e.is_dir():
                dirs.append(e.name)
            elif e.suffix.lower() in {".csv", ".db"}:
                files.append(e.name)
        except Exception:
            continue  # 单个坏条目（断链等）跳过，不影响整体
    dirs.sort(key=str.lower)
    files.sort(key=str.lower)
    parent = str(base.parent) if base.parent != base else ""
    return {"path": str(base), "parent": parent, "dirs": dirs,
            "files": files, "truncated": truncated}


@app.get("/api/pick")
def api_pick(kind: str = "file", title: str = "", initial: str = ""):
    """在运行后端的本机弹出原生 文件/文件夹 选择对话框，返回所选路径。

    kind: file | dir。用户取消返回 {canceled: true}；本机无 GUI 时返回 501。
    """
    from .native_pick import PickCanceled, pick
    if kind not in ("file", "dir"):
        raise HTTPException(status_code=400, detail="kind 必须为 file 或 dir")
    # 容器/服务器（无 GUI）里没有 Tk，系统对话框用不了 → 直接提示用网页文件浏览器，不抛底层报错
    if os.environ.get("FORVIA_IN_DOCKER"):
        raise HTTPException(status_code=501,
                            detail="容器内无系统对话框，请用上方“文件浏览器”选择（或手动粘贴容器内路径，如 /data/...）。")
    try:
        path = pick(kind, title=title, initial=initial or None)
    except PickCanceled:
        return {"canceled": True}
    except Exception:
        raise HTTPException(status_code=501,
                            detail="无法打开系统对话框，请用上方“文件浏览器”选择（或手动粘贴路径，如 /data/...）。")
    return {"canceled": False, "path": path}


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """接收拖拽上传的文件，存到临时目录，返回可被后端读取的绝对路径。"""
    up_dir = Path(tempfile.gettempdir()) / "forvia_v2_uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    dest = up_dir / (file.filename or "uploaded_file")
    data = await file.read()
    dest.write_bytes(data)
    return {"ok": True, "path": str(dest), "name": file.filename}


@app.get("/api/config")
def api_config():
    s = get_session()
    return {
        "sample_view_path": DEFAULT_SAMPLE_VIEW_PATH,
        "tdms_root": s.tdms_root,
        "label_records_db_path": s.label_records_db_path,
        "is_mock": s.is_mock,
        "has_db": s.has_db,
        "default_source": s.default_source,
        "n_samples": len(s.sample_view),
        "n_resolved": len(s.path_map),
        "n_unresolved": len(s.unresolved),
    }


@app.get("/api/diagnose")
def api_diagnose(limit: int = 10):
    """诊断"缺失"样本：列出每个缺失件尝试过的路径、是否存在、find_tdms 扫描的 line 目录。"""
    from pathlib import Path as _P
    from .tdms_locator import ManifestAdapter, TdmsLocator
    s = get_session()
    db_folder = str(_P(s.label_records_db_path).parent) if s.label_records_db_path else None
    loc = TdmsLocator(s.tdms_root or None, manifest=ManifestAdapter(db_folder))
    out = []
    for sid in s.unresolved[:limit]:
        rows = s.sample_view[s.sample_view["sample_id"].astype(str) == sid]
        if rows.empty:
            continue
        out.append({"sample_id": sid, **loc.explain(rows.iloc[0])})
    return {"tdms_root": s.tdms_root, "n_missing": len(s.unresolved), "samples": out}


@app.get("/api/overview")
def api_overview(label_source: str = "db"):
    return get_session().build_overview(label_source=label_source)


@app.get("/api/refresh")
def api_refresh(label_source: str = "db", force: int = 0):
    """一次请求返回 overview + list_view + 配置。标签读全部走内存缓存（不反复读外接盘 DB）。
    force=1（“↻ 刷新”按钮）时先丢弃缓存、从 DB 重新加载一次。"""
    s = get_session()
    if force:
        s.label_table.invalidate_cache()
    events, latest = s.snapshot()
    expert_ids = s._expert_ids_from_events(events)
    return {
        "overview": s.build_overview(latest=latest, label_source=label_source, expert_ids=expert_ids),
        "list": s.list_view(events=events, label_source=label_source, expert_ids=expert_ids),
        "has_db": s.has_db,
        "default_source": s.default_source,
        "from_sample_view": getattr(s, "from_sample_view", False),
    }


# note 里需要清掉的"导入残留字段"键（result=... 及其映射等）
_NOTE_JUNK_KEYS = {
    "result", "reason_auto", "fuzzy", "raw_result", "raw_reason", "raw_label",
    "factory_source_kind", "factory_source_path", "factory_source_row",
    "dedup_same_sample_source_label_count", "group", "reference",
}
_NOTE_MARKERS = ("[[典型异音]]", "[[prototype]]")


def clean_note(note: str) -> str:
    """清理 note：去掉 result=... 等导入残留键值段，保留人工备注与标记。不影响所在行。"""
    s = str(note or "")
    markers = [m for m in _NOTE_MARKERS if m in s]
    for m in _NOTE_MARKERS:
        s = s.replace(m, "")
    kept = []
    for seg in s.split("|"):
        seg = seg.strip()
        if not seg:
            continue
        key = seg.split("=", 1)[0].strip().lower() if "=" in seg else ""
        if key and key in _NOTE_JUNK_KEYS:
            continue          # 丢弃导入残留字段
        kept.append(seg)
    out = " | ".join(kept)
    if markers:
        out = (" ".join(markers) + " " + out).strip()
    return out.strip()


@app.post("/api/db/clean_notes")
def api_db_clean_notes():
    """检查 db：把 note 里的 result=... 等导入残留字段清掉（只改 note 单元格，不删行）。"""
    from .config import LABEL_HISTORY_COLUMNS
    s = get_session()
    store = s.label_table.store
    rows = store.list_all()
    changed, samples = 0, []
    for r in rows:
        old = str(r.get("note", "") or "")
        new = clean_note(old)
        if new != old:
            r["note"] = new
            changed += 1
            if len(samples) < 5:
                samples.append({"sample_id": r.get("sample_id", ""), "before": old, "after": new})
    if changed:
        store._write_all([{c: str(r.get(c, "") or "") for c in LABEL_HISTORY_COLUMNS} for r in rows])
    return {"ok": True, "changed": changed, "total": len(rows), "samples": samples}


# ===== 非标准 reason（不在 cfg/core 标签库）整理成 sample_view =====
def _standard_reason_set() -> set:
    from data_manager.label_rules import LABEL_RULES as _LR
    s = set()
    for key, info in (_LR.get("reasons") or {}).items():
        vals = [key, (info or {}).get("name")] + list((info or {}).get("alias") or [])
        for v in vals:
            if v:
                s.add(str(v).strip()); s.add(str(v).strip().lower())
    return s


def _nonstandard_rows():
    s = get_session()
    std = _standard_reason_set()
    sv_by_id = {}
    for _, r in s.sample_view.iterrows():
        sv_by_id[str(r.get("sample_id", ""))] = r
    out = []
    for sid, lab in s.label_table.latest_label_map().items():
        rn = str(lab.get("reason_name", "") or "").strip()
        if not rn or rn in std or rn.lower() in std:
            continue                       # 空 或 标准 reason → 跳过
        r0 = sv_by_id.get(sid)
        out.append({
            "view_name": "nonstandard_review",
            "line": str(lab.get("line", "")), "sn": str(lab.get("sn", "")), "sample_id": sid,
            "group_name": str(r0.get("group_name", "")) if r0 is not None else "",
            "channel_name": str(r0.get("channel_name", "")) if r0 is not None else "",
            "reason_name": rn, "reason_key": str(lab.get("reason_key", "")),
            "reason_confidence": str(lab.get("reason_confidence", "")),
            "source": str(lab.get("source", "")), "note": str(lab.get("note", "")),
            "nonstandard": "1",
        })
    return out


@app.get("/api/nonstandard_reasons")
def api_nonstandard_reasons():
    rows = _nonstandard_rows()
    return {"count": len(rows), "rows": rows[:200],
            "reasons": sorted({r["reason_name"] for r in rows})}


class NsExportReq(BaseModel):
    out_path: str = ""


@app.post("/api/nonstandard_reasons/export")
def api_nonstandard_reasons_export(req: NsExportReq):
    import csv as _csv
    from datetime import datetime as _dt
    s = get_session()
    rows = _nonstandard_rows()
    cols = ["view_name", "line", "sn", "sample_id", "group_name", "channel_name",
            "reason_name", "reason_key", "reason_confidence", "source", "note", "nonstandard"]
    base = Path(s.sample_view_path).expanduser().parent if s.sample_view_path else Path(tempfile.gettempdir())
    out = req.out_path
    fname = "nonstandard_reason_sample_view_" + _dt.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    if not out:
        out = str(base / fname)
    else:
        p = Path(out).expanduser()
        if p.is_dir() or out.endswith(("/", "\\")):
            out = str(p / fname)
    outp = Path(out).expanduser(); outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return {"ok": True, "path": str(outp), "rows": len(rows)}


@app.post("/api/db/prune_missing")
def api_db_prune_missing():
    """删除数据库中"无对应 tdms 文件(缺失)"的样本及其标签事件。仅 db 模式。"""
    import sqlite3
    s = get_session()
    if not s.has_db:
        raise HTTPException(status_code=400, detail="未加载数据库，无可清理的登记")
    missing = {sid for sid, st in s.status_map.items() if st == "missing"}
    keys = []
    for _, r in s.sample_view.iterrows():
        sid = str(r.get("sample_id", ""))
        if sid in missing:
            keys.append((str(r.get("line", "")), str(r.get("sn", "")), sid))
    if not keys:
        return {"ok": True, "deleted_samples": 0, "deleted_events": 0}
    db = s.label_records_db_path
    ds = de = 0
    con = sqlite3.connect(db)
    try:
        cur = con.cursor()
        for line, sn, sid in keys:
            pks = [row[0] for row in cur.execute(
                "SELECT id FROM samples WHERE line=? AND sn=? AND sample_id=?", (line, sn, sid))]
            for pk in pks:
                de += cur.execute("DELETE FROM label_events WHERE sample_pk=?", (pk,)).rowcount
                ds += cur.execute("DELETE FROM samples WHERE id=?", (pk,)).rowcount
        con.commit()
    finally:
        con.close()
    # 重建会话以刷新（无 sample_view 时会按精简后的 db 重新生成）
    init_session(s.sample_view_path or None, s.tdms_root or None,
                 s.label_records_db_path or None, source=s.default_source)
    return {"ok": True, "deleted_samples": ds, "deleted_events": de}


@app.get("/api/list_view")
def api_list_view():
    """列表总览：动态列 + 行（无 db=sample_view 全列+空标签列；有 db=16列多记录展开）。"""
    return get_session().list_view()


@app.get("/api/label_records")
def api_label_records(labeled_only: bool = True, changed_only: bool = False):
    """完整标签结果表（16 列）。labeled_only=True 只含做过标注的记录；
    changed_only=True 只含本次任务改动过的样本。"""
    from .session import LABEL_TABLE_COLUMNS
    s = get_session()
    return {"columns": LABEL_TABLE_COLUMNS,
            "rows": s.label_records(labeled_only=labeled_only, changed_only=changed_only),
            "has_db": s.has_db, "changed_count": len(s.changed_ids)}


class LabelExportReq(BaseModel):
    out_path: str = ""
    changed_only: bool = True   # 默认只导出本次任务改动过的标签


@app.post("/api/label_records/export")
def api_label_records_export(req: LabelExportReq):
    """导出"标签结果表"为新 CSV（16 列，含 group/channel）。默认仅导出本次任务改动过的标签。"""
    import csv as _csv
    from datetime import datetime
    from .session import LABEL_TABLE_COLUMNS
    s = get_session()
    rows = s.label_records(labeled_only=True, changed_only=req.changed_only)
    out = req.out_path
    base = Path(s.sample_view_path).expanduser().parent if s.sample_view_path else Path(tempfile.gettempdir())
    if not out:
        out = str(base / ("label_result_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"))
    else:
        p = Path(out).expanduser()
        if p.is_dir() or out.endswith(("/", "\\")):
            out = str(p / ("label_result_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"))
    outp = Path(out).expanduser(); outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=LABEL_TABLE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in LABEL_TABLE_COLUMNS})
    return {"ok": True, "path": str(outp), "rows": len(rows)}


class WritebackReq(BaseModel):
    changed_only: bool = True   # 默认只写回本次任务改动过的样本


@app.post("/api/label_records/writeback")
def api_label_records_writeback(req: WritebackReq = WritebackReq()):
    """把最新标签写回输入的 sample_view.csv。默认仅写回本次任务改动过的样本。"""
    s = get_session()
    try:
        path, n = s.write_back_sample_view(changed_only=req.changed_only)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"写回失败: {e}")
    return {"ok": True, "path": path, "updated": n}


@app.get("/api/sample/{index}")
def api_sample(index: int):
    data = get_session().build_overview()
    rows = data["rows"]
    if index < 0 or index >= len(rows):
        raise HTTPException(status_code=404, detail="sample index out of range")
    return {"sample": rows[index],
            "note": "标注视图占位：后续加载 tdms、渲染播放器/曲线/频谱卡片。"}


@app.get("/api/sample/{index}/signal")
def api_sample_signal(index: int, max_points: int = 2000):
    """加载 tdms 并返回信号元信息 + 下采样预览（用于验证 / 轻量前端绘图）。"""
    s = get_session()
    if index < 0 or index >= len(s.sample_view):
        raise HTTPException(status_code=404, detail="sample index out of range")
    try:
        sig = load_sample(s, index)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载 tdms 失败: {e}")
    return {
        "index": sig.index, "sn": sig.sn, "sample_id": sig.sample_id,
        "direction": sig.direction, "line": sig.line,
        "sampling_rate": sig.sampling_rate,
        "raw_len": int(len(sig.raw)) if sig.raw is not None else 0,
        "proc_len": int(len(sig.proc)) if sig.proc is not None else 0,
        "raw_preview": downsample(sig.raw, max_points),
        "proc_preview": downsample(sig.proc, max_points),
    }


@app.get("/api/cards")
def api_cards():
    """已注册的卡片清单 + 默认卡片。"""
    return {"cards": list_cards(), "default": default_card_ids()}


# 卡片图缓存：键=(index, card_id, params)。同一样本/参数重复访问直接命中，免再读 tdms + 重算
from collections import OrderedDict as _OD
_CARD_CACHE: "_OD[tuple, dict]" = _OD()
_CARD_CACHE_MAX = 12          # 缩小：每张图几百 KB，12 张约十几 MB


def _card_cache_key(index, card_id, params):
    return (index, card_id, json.dumps(params or {}, sort_keys=True, ensure_ascii=False))


def _card_cache_get(key):
    v = _CARD_CACHE.get(key)
    if v is not None:
        _CARD_CACHE.move_to_end(key)
    return v


def _card_cache_put(key, val):
    _CARD_CACHE[key] = val
    _CARD_CACHE.move_to_end(key)
    while len(_CARD_CACHE) > _CARD_CACHE_MAX:
        _CARD_CACHE.popitem(last=False)


@app.get("/api/sample/{index}/cards")
def api_sample_cards(index: int, ids: str = ""):
    """渲染指定卡片（逗号分隔；缺省=默认卡片）-> Plotly figure JSON 列表。带缓存。"""
    s = get_session()
    if index < 0 or index >= len(s.sample_view):
        raise HTTPException(status_code=404, detail="sample index out of range")
    import time as _t
    card_ids = [x for x in ids.split(",") if x.strip()] or default_card_ids()
    # 先查缓存，全部命中则无需加载 tdms
    results = {cid: _card_cache_get(_card_cache_key(index, cid, {})) for cid in card_ids}
    sid = str(s.row(index).get("sample_id", ""))
    timings = {"cache_hit": all(v is not None for v in results.values()),
               "load_ms": 0.0, "cards_ms": {}}
    if any(v is None for v in results.values()):
        t0 = _t.perf_counter()
        try:
            sig = load_sample(s, index)
            ctx = CardContext.from_signals(sig)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"加载 tdms 失败: {e}")
        timings["load_ms"] = round((_t.perf_counter() - t0) * 1000, 1)
        timings["load_detail"] = sig.timings or {}
        sid = ctx.sample_id
        for cid in card_ids:
            if results[cid] is not None:
                continue
            tc = _t.perf_counter()
            try:
                r = render_card(cid, ctx)
                _card_cache_put(_card_cache_key(index, cid, {}), r)
                results[cid] = r
            except KeyError:
                continue
            except Exception as e:
                results[cid] = {"id": cid, "title": cid, "category": "", "figure": None, "error": str(e)}
            timings["cards_ms"][cid] = round((_t.perf_counter() - tc) * 1000, 1)
    out = [results[cid] for cid in card_ids if results.get(cid) is not None]
    return {"index": index, "sample_id": sid, "cards": out, "timings": timings}


class PrefetchReq(BaseModel):
    indices: list[int] = []


# 单一预取通道：同一时刻只跑一个预取，新的请求覆盖旧目标，避免快速翻页时
# 堆叠出大量并发解压线程（每个都持有大数组 → 内存暴涨）。
_PREFETCH_LOCK = threading.Lock()
_PREFETCH_TARGET: list[int] = []
_PREFETCH_RUNNING = False


@app.post("/api/prefetch")
def api_prefetch(req: PrefetchReq):
    """后台预解压 + 预渲染默认卡片（用于"下一条"提前就绪）。立即返回，不阻塞。
    全局仅一个预取 worker，快速翻页只保留最新目标。"""
    global _PREFETCH_TARGET, _PREFETCH_RUNNING
    s = get_session()
    n = len(s.sample_view)
    idxs = [i for i in (req.indices or []) if 0 <= i < n]

    def _warm():
        global _PREFETCH_RUNNING
        while True:
            with _PREFETCH_LOCK:
                todo = list(_PREFETCH_TARGET)
                _PREFETCH_TARGET.clear()
                if not todo:
                    _PREFETCH_RUNNING = False
                    return
            for i in todo:
                try:
                    if all(_card_cache_get(_card_cache_key(i, cid, {})) is not None
                           for cid in default_card_ids()):
                        continue
                    ctx = CardContext.from_signals(load_sample(s, i))
                    for cid in default_card_ids():
                        k = _card_cache_key(i, cid, {})
                        if _card_cache_get(k) is None:
                            try:
                                _card_cache_put(k, render_card(cid, ctx))
                            except Exception:
                                pass
                except Exception:
                    pass

    with _PREFETCH_LOCK:
        _PREFETCH_TARGET = idxs          # 覆盖为最新目标
        start = not _PREFETCH_RUNNING
        if start:
            _PREFETCH_RUNNING = True
    if start:
        threading.Thread(target=_warm, daemon=True).start()
    return {"ok": True, "queued": idxs}


class CardReq(BaseModel):
    card_id: str
    params: dict = {}


@app.post("/api/sample/{index}/card")
def api_sample_card(index: int, req: CardReq):
    """带参数渲染单张卡片（高级频谱的分辨率等参数）。"""
    s = get_session()
    if index < 0 or index >= len(s.sample_view):
        raise HTTPException(status_code=404, detail="sample index out of range")
    key = _card_cache_key(index, req.card_id, req.params)
    cached = _card_cache_get(key)
    if cached is not None:
        return {"index": index, "card": cached}
    try:
        ctx = CardContext.from_signals(load_sample(s, index))
        r = render_card(req.card_id, ctx, req.params)
        _card_cache_put(key, r)
        return {"index": index, "card": r}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"未知卡片: {req.card_id}")
    except Exception as e:
        return {"index": index, "card": {"id": req.card_id, "title": req.card_id,
                "category": "", "figure": None, "error": str(e)}}


@app.get("/api/sample/{index}/audio.wav")
def api_sample_audio(index: int):
    from .audio import generate_wav_stream
    s = get_session()
    if index < 0 or index >= len(s.sample_view):
        raise HTTPException(status_code=404, detail="sample index out of range")
    try:
        sig = load_sample(s, index)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if sig.proc is None:
        raise HTTPException(status_code=404, detail="无信号")
    obj = generate_wav_stream(sig.proc, sample_rate=sig.sampling_rate)
    data = obj.getvalue() if hasattr(obj, "getvalue") else obj
    return Response(content=data, media_type="audio/wav")


@app.get("/api/reasons")
def api_reasons():
    from .label_table import CONFIDENCE_OPTIONS, REASONS
    return {"reasons": REASONS, "confidence_options": CONFIDENCE_OPTIONS,
            "sources": SOURCE_OPTIONS, "default_source": get_session().default_source}


class LabelReq(BaseModel):
    index: int
    reason_name: str = ""
    reason_key: str = ""
    reason_confidence: float | None = None
    note: str = ""
    source: str = ""        # 空 = 用会话默认来源


@app.post("/api/label")
def api_label(req: LabelReq):
    import time as _t
    s = get_session()
    if req.index < 0 or req.index >= len(s.sample_view):
        raise HTTPException(status_code=404, detail="sample index out of range")
    row = s.row(req.index)
    sid = str(row.get("sample_id", ""))
    t0 = _t.perf_counter()
    try:
        s.label_table.add(
            line=str(row.get("line", "")), sn=str(row.get("sn", "")), sample_id=sid,
            reason_name=req.reason_name, reason_key=req.reason_key,
            reason_confidence=req.reason_confidence, note=req.note,
            source=req.source or s.default_source,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"保存标注失败: {e}")
    s.mark_changed(sid)
    hist = s.label_table.history_for(sid)
    return {"ok": True, "sample_id": sid, "history": hist,
            "latest": hist[-1] if hist else None,
            "write_ms": round((_t.perf_counter() - t0) * 1000, 1)}


# ---- 单条 label 记录的 确认 / 删除（仅 db 模式在前端展示按钮）----
def _sample_event_global_index(store, sample_id: str, event_index: int):
    """把"某 sample 的第 event_index 条历史"映射到全量 rows 的下标。"""
    rows = store.list_all()
    idxs = [i for i, r in enumerate(rows) if str(r.get("sample_id", "")) == str(sample_id)]
    if event_index < 0 or event_index >= len(idxs):
        return rows, None
    return rows, idxs[event_index]


class ConfirmReq(BaseModel):
    index: int
    event_index: int | None = None
    event_id: int | None = None
    source: str = ""


@app.post("/api/label/confirm")
def api_label_confirm(req: ConfirmReq):
    """确认某条标注正确：复制为 source=expert 的新记录（单条 INSERT + 更新内存缓存，不重读整表）。"""
    import time as _t
    s = get_session()
    sid = str(s.row(req.index).get("sample_id", "")) if 0 <= req.index < len(s.sample_view) else ""
    t0 = _t.perf_counter()
    try:
        history = s.label_table.confirm_event(sid, event_id=req.event_id, event_index=req.event_index)
    except IndexError:
        raise HTTPException(status_code=404, detail="标注记录不存在")
    s.mark_changed(sid)
    return {"ok": True, "sample_id": sid, "source": "expert", "history": history,
            "write_ms": round((_t.perf_counter() - t0) * 1000, 1)}


class DeleteLabelReq(BaseModel):
    index: int
    event_index: int | None = None
    event_id: int | None = None


@app.post("/api/label/delete")
def api_label_delete(req: DeleteLabelReq):
    """物理删除该条标注：单条 DELETE + 更新内存缓存，不重读整表。"""
    import time as _t
    s = get_session()
    sid = str(s.row(req.index).get("sample_id", "")) if 0 <= req.index < len(s.sample_view) else ""
    t0 = _t.perf_counter()
    try:
        history = s.label_table.delete_event(sid, event_id=req.event_id, event_index=req.event_index)
    except IndexError:
        raise HTTPException(status_code=404, detail="标注记录不存在")
    s.mark_changed(sid)
    return {"ok": True, "sample_id": sid, "history": history,
            "write_ms": round((_t.perf_counter() - t0) * 1000, 1)}


class TypicalReq(BaseModel):
    index: int
    flag: bool = True


@app.post("/api/typical")
def api_typical(req: TypicalReq):
    s = get_session()
    if req.index < 0 or req.index >= len(s.sample_view):
        raise HTTPException(status_code=404, detail="sample index out of range")
    sid = str(s.row(req.index).get("sample_id", ""))
    ok = s.label_table.set_typical(sid, req.flag)
    if not ok:
        raise HTTPException(status_code=400, detail="该样本尚无标签，请先标注再标记典型异音")
    s.mark_changed(sid)
    return {"ok": True, "sample_id": sid, "typical": req.flag}


@app.get("/api/sample/{index}/history")
def api_sample_history(index: int):
    s = get_session()
    if index < 0 or index >= len(s.sample_view):
        raise HTTPException(status_code=404, detail="sample index out of range")
    sid = str(s.row(index).get("sample_id", ""))
    return {"sample_id": sid, "history": s.label_table.history_for(sid)}


@app.get("/api/sample/{index}/sv_label")
def api_sample_sv_label(index: int):
    """输入 sample_view 自带的标签（若有）。供"来自 sample_view"面板展示。"""
    s = get_session()
    if index < 0 or index >= len(s.sample_view):
        raise HTTPException(status_code=404, detail="sample index out of range")
    return {"sample_id": str(s.row(index).get("sample_id", "")),
            "label": s.sample_view_label(index)}


class ConfirmSvReq(BaseModel):
    index: int


@app.post("/api/label/confirm_sv")
def api_label_confirm_sv(req: ConfirmSvReq):
    """把 sample_view 自带的标签确认为正确 → 以 source=expert 写入数据库（仅 expert 模式）。"""
    s = get_session()
    if req.index < 0 or req.index >= len(s.sample_view):
        raise HTTPException(status_code=404, detail="sample index out of range")
    if s.default_source != "expert":
        raise HTTPException(status_code=403, detail="仅 expert 模式可确认写入数据库")
    lab = s.sample_view_label(req.index)
    if not lab:
        raise HTTPException(status_code=400, detail="该样本的 sample_view 没有标签")
    row = s.row(req.index)
    conf = lab.get("reason_confidence")
    try:
        conf = float(conf) if str(conf).strip() else None
    except Exception:
        conf = None
    s.label_table.add(
        line=str(row.get("line", "")), sn=str(row.get("sn", "")),
        sample_id=str(row.get("sample_id", "")),
        reason_name=lab.get("reason_name", ""), reason_key=lab.get("reason_key", ""),
        reason_confidence=conf, note=lab.get("note", ""), source="expert",
    )
    s.mark_changed(str(row.get("sample_id", "")))
    return {"ok": True, "sample_id": str(row.get("sample_id", ""))}


class ExportReq(BaseModel):
    out_path: str = ""
    scope_line: str = ""
    only_typical: bool = False


@app.post("/api/export")
def api_export(req: ExportReq):
    from .label_table import LabelTable
    s = get_session()
    out = req.out_path
    if not out:
        base_dir = Path(s.sample_view_path).expanduser().parent if s.sample_view_path \
            else Path(s.label_records_db_path).expanduser().parent
        out = str(base_dir / LabelTable.default_timestamped_name())
    else:
        p = Path(out).expanduser()
        # 若给的是目录（或以分隔符结尾），自动拼接带时间戳的文件名
        if p.is_dir() or out.endswith(("/", "\\")):
            out = str(p / LabelTable.default_timestamped_name())
    try:
        path, n = s.label_table.export(out, scope_line=req.scope_line,
                                       only_typical=req.only_typical)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"导出失败: {e}")
    return {"ok": True, "path": str(path), "rows": n}


# ================= 任务系统 =================
@app.get("/api/tasks")
def api_tasks():
    return {"tasks": task_api.list_tasks()}


class RunTaskReq(BaseModel):
    task_id: str
    scope: str = "all"        # all 或 线名
    params: dict = {}


@app.post("/api/tasks/run")
def api_tasks_run(req: RunTaskReq):
    s = get_session()
    try:
        run = task_runner.start_run(s, req.task_id, req.scope, req.params)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True, "run": run.summary()}


@app.get("/api/tasks/runs")
def api_tasks_runs():
    return {"runs": task_runner.list_runs()}


@app.get("/api/tasks/run/{run_id}")
def api_tasks_run_detail(run_id: str):
    run = task_runner.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    # 附带建议明细（带 sample_id）
    sugg = [{"sample_id": sid, **s} for sid, s in run.suggestions.items()]
    return {**run.summary(), "suggestions": sugg}


class RunExportReq(BaseModel):
    out_path: str = ""


@app.post("/api/tasks/run/{run_id}/export")
def api_tasks_run_export(run_id: str, req: RunExportReq):
    """把某次任务的结果表导出为 CSV（sample_id + 结果列）。"""
    import csv as _csv
    s = get_session()
    run = task_runner.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    # sample_id -> sample_view 行（补 line/sn）
    sv = s.sample_view
    by_id = {str(r.get("sample_id", "")): r for _, r in sv.iterrows()}
    cols, rows = ["line", "sn", "sample_id"], []
    for sid, data in run.results.items():
        r0 = by_id.get(sid)
        row = {"line": str(r0.get("line", "")) if r0 is not None else "",
               "sn": str(r0.get("sn", "")) if r0 is not None else "", "sample_id": sid}
        for k, v in (data or {}).items():
            if k not in cols:
                cols.append(k)
            row[k] = v
        # 附带该 sample 的建议（若有）
        sug = run.suggestions.get(sid)
        if sug:
            row["suggested_reason"] = sug.get("reason_key", "")
            row["suggested_conf"] = sug.get("confidence", "")
        rows.append(row)
    for extra in ("suggested_reason", "suggested_conf"):
        if any(extra in r for r in rows) and extra not in cols:
            cols.append(extra)
    out = req.out_path
    if not out:
        base = Path(s.sample_view_path).expanduser().parent if s.sample_view_path else Path(tempfile.gettempdir())
        out = str(base / f"task_{run.task_id}_{run_id}.csv")
    else:
        p = Path(out).expanduser()
        if p.is_dir() or out.endswith(("/", "\\")):
            out = str(p / f"task_{run.task_id}_{run_id}.csv")
    outp = Path(out).expanduser()
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return {"ok": True, "path": str(outp), "rows": len(rows)}


@app.get("/api/sample/{index}/tasks")
def api_sample_tasks(index: int):
    """该样本的任务结果 + 建议（用于通道内展示/采纳）。"""
    s = get_session()
    if index < 0 or index >= len(s.sample_view):
        raise HTTPException(status_code=404, detail="sample index out of range")
    sid = str(s.row(index).get("sample_id", ""))
    results = []
    for r in task_runner.RUNS.values():
        if sid in r.results:
            results.append({"task_title": r.task_title, "data": r.results[sid]})
    return {"sample_id": sid, "results": results,
            "suggestions": task_runner.suggestions_for_sample(sid)}


class AcceptReq(BaseModel):
    run_id: str
    sample_id: str


@app.post("/api/tasks/accept")
def api_tasks_accept(req: AcceptReq):
    """采纳某条建议 -> 写入 LabelTable（source=model）。"""
    s = get_session()
    run = task_runner.get_run(req.run_id)
    if run is None or req.sample_id not in run.suggestions:
        raise HTTPException(status_code=404, detail="建议不存在")
    sug = run.suggestions[req.sample_id]
    row = s.sample_view[s.sample_view["sample_id"].astype(str) == req.sample_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="sample_id 不在 sample_view")
    r0 = row.iloc[0]
    try:
        s.label_table.add(line=str(r0.get("line", "")), sn=str(r0.get("sn", "")),
                          sample_id=req.sample_id, reason_key=sug["reason_key"],
                          reason_confidence=sug.get("confidence"),
                          note=sug.get("note", ""), source="model")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"采纳失败: {e}")
    sug["accepted"] = True
    s.mark_changed(req.sample_id)
    return {"ok": True, "sample_id": req.sample_id, "reason_key": sug["reason_key"]}


# ================= 重新注册 data_root 下的 TDMS =================
_REGISTER_STATE = {"running": False, "message": "", "error": "", "started": "", "data_root": ""}


def _run_register(data_root, manifest_path, db_path, line, storage_root):
    from pathlib import Path as _P
    try:
        from data_manager.sample_generate import scan_tdms
        from data_manager.label_database import LabelDatabase
        _P(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        LabelDatabase(_P(db_path))   # 确保 db 存在并初始化 schema
        scan_tdms(_P(data_root), _P(manifest_path), _P(db_path),
                  line=line or None, storage_root=storage_root)
        _REGISTER_STATE.update(running=False, message="重新注册完成", error="")
    except Exception as e:
        _REGISTER_STATE.update(running=False, message="", error=str(e))


class RegisterReq(BaseModel):
    line: str = ""        # 留空=全部线


@app.post("/api/register_tdms")
def api_register_tdms(req: RegisterReq):
    """扫描 data_root 下（tdms_root 的上一级）的 TDMS，重建 manifest + sample 登记。后台执行。"""
    import threading
    from datetime import datetime as _dt
    s = get_session()
    if not s.tdms_root:
        raise HTTPException(status_code=400, detail="未设置 TDMS 目录")
    if _REGISTER_STATE["running"]:
        return {"ok": False, "detail": "正在注册中…"}
    tr = Path(s.tdms_root).expanduser()
    data_root = tr.parent
    storage_root = tr.name or "factory_raw"
    meta = data_root / "metadata"
    manifest_path = meta / "tdms_manifest.csv"
    db_path = Path(s.label_records_db_path) if s.has_db else (meta / "label_records.db")
    _REGISTER_STATE.update(running=True, message="", error="",
                           started=_dt.now().isoformat(timespec="seconds"), data_root=str(data_root))
    threading.Thread(target=_run_register,
                     args=(data_root, manifest_path, db_path, req.line, storage_root),
                     daemon=True).start()
    return {"ok": True, "data_root": str(data_root), "storage_root": storage_root}


@app.get("/api/register_tdms/status")
def api_register_status():
    return _REGISTER_STATE


# ================= Prototype（典型件入库，db+expert）=================
@app.get("/api/prototype/candidates")
def api_prototype_candidates():
    from . import prototype
    s = get_session()
    return {"can_operate": prototype.can_operate(s),
            "prototype_root": str(prototype.prototype_root(s)) if s.tdms_root else "",
            "candidates": prototype.candidates(s) if prototype.can_operate(s) else []}


class PrototypeReq(BaseModel):
    sample_ids: list[str] = []
    dest_root: str = ""


@app.post("/api/prototype/import")
def api_prototype_import(req: PrototypeReq):
    from . import prototype
    s = get_session()
    if not prototype.can_operate(s):
        raise HTTPException(status_code=403, detail="需要加载数据库且 source=expert 才能操作")
    if not req.sample_ids:
        raise HTTPException(status_code=400, detail="未选择样本")
    return {"ok": True, **prototype.import_prototypes(s, req.sample_ids, dest_root=req.dest_root or None)}


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
