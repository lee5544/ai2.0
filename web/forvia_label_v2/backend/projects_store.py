"""已保存的"任务/项目"：持久化数据源配置（路径等），下次一键进入。

存到本机一个 JSON 文件，跨重启保留。与 🛠 批处理任务系统无关。
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

# 任务配置 json 存在应用数据目录（打包后用可写的 FORVIA_DATA_HOME，否则源码下的 _data）
_APP_DATA_DIR = Path(os.environ.get("FORVIA_DATA_HOME")
                     or (Path(__file__).resolve().parent.parent / "_data")).expanduser()
STORE_FILE = Path(os.environ.get(
    "FORVIA_PROJECTS_FILE",
    str(_APP_DATA_DIR / "projects.json"),
)).expanduser()

FIELDS = ("name", "sample_view_path", "tdms_root", "label_records_db_path", "source", "workspace_id")
DATA_FIELDS = ("sample_view_path", "tdms_root", "label_records_db_path", "source")


def _load() -> list[dict]:
    try:
        if STORE_FILE.exists():
            items = json.loads(STORE_FILE.read_text(encoding="utf-8")) or []
            for item in items:
                if "label_records_db_path" not in item:
                    item["label_records_db_path"] = str(
                        item.pop("sample_records_db_path", "") or ""
                    )
            return items
    except Exception:
        pass
    return []


def _save(items: list[dict]) -> None:
    try:
        STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STORE_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    except Exception:
        pass


def list_projects() -> list[dict]:
    # 最近使用在前
    return sorted(_load(), key=lambda p: p.get("last_used_at", 0), reverse=True)


def add_project(data: dict) -> dict:
    items = _load()
    cfg = {k: str(data.get(k, "") or "") for k in FIELDS}
    if not cfg["name"]:
        cfg["name"] = Path(cfg["sample_view_path"]).stem or "未命名任务"
    # 同名+同 sample_view 视为更新
    now = time.time()
    for it in items:
        if it.get("name") == cfg["name"] and it.get("sample_view_path") == cfg["sample_view_path"]:
            if not cfg["workspace_id"]:
                cfg["workspace_id"] = str(it.get("workspace_id", "") or "")
            if (any(str(it.get(k, "") or "") != cfg[k] for k in DATA_FIELDS)
                    and cfg["workspace_id"] == str(it.get("workspace_id", "") or "")):
                cfg["workspace_id"] = uuid.uuid4().hex
            it.update(cfg); it["last_used_at"] = now
            _save(items)
            return it
    if not cfg["workspace_id"]:
        cfg["workspace_id"] = uuid.uuid4().hex
    cfg["id"] = uuid.uuid4().hex[:8]
    cfg["created_at"] = now
    cfg["last_used_at"] = now
    items.append(cfg)
    # 最多保留 20 个（按最近使用），淘汰最旧的
    if len(items) > 20:
        items = sorted(items, key=lambda p: p.get("last_used_at", 0), reverse=True)[:20]
    _save(items)
    return cfg


def get_project(pid: str) -> dict | None:
    return next((p for p in _load() if p.get("id") == pid), None)


def touch(pid: str) -> None:
    items = _load()
    for it in items:
        if it.get("id") == pid:
            it["last_used_at"] = time.time()
    _save(items)


def delete_project(pid: str) -> bool:
    items = _load()
    new = [p for p in items if p.get("id") != pid]
    if len(new) != len(items):
        _save(new)
        return True
    return False
