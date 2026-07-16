"""Background orchestration for external data_manager database operations."""

from __future__ import annotations

import copy
import threading
import uuid
from typing import Any

from forvia_train_v2.backend.database_service import execute_database_action
from ml.training.app_api import normalize_project_payload, save_project_config

from . import store


_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _set(job_id: str, **changes: Any) -> None:
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update(changes)


def get(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return copy.deepcopy(job) if job else None


def _save_database(project: dict, result: dict) -> dict:
    database = {
        "label_records_db_path": result["label_records_db_path"],
        "tdms_root": result["tdms_root"],
        "manifest_path": result["manifest_path"],
        "configured": True,
        "last_operation": result.get("action", ""),
    }
    config = copy.deepcopy(project["config"])
    config["database"] = database
    config["label_records_db_path"] = database["label_records_db_path"]
    payload = normalize_project_payload(
        {
            "name": project["name"],
            "line_name": project["line_name"],
            "model_name": project["model_name"],
            "model_type": project["model_type"],
            "config": config,
        }
    )
    payload = save_project_config(payload, str(project.get("config_path") or ""))
    return store.update_project(project["id"], payload) or project


def _run(job_id: str, project: dict | None, payload: dict) -> None:
    try:
        _set(job_id, status="running", progress=1, progress_detail="数据库任务已启动")

        def update_progress(
            value: float,
            detail: str,
            processed: int = 0,
            total: int = 0,
            phase: str = "",
        ) -> None:
            changes: dict[str, Any] = {
                "progress": value,
                "progress_detail": detail,
            }
            if total > 0:
                changes.update(
                    tdms_processed=processed,
                    tdms_total=total,
                    tdms_remaining=max(0, total - processed),
                    tdms_phase=phase,
                )
            _set(job_id, **changes)

        result = execute_database_action(
            **payload,
            progress=update_progress,
        )
        updated = _save_database(project, result) if project else None
        _set(
            job_id,
            status="succeeded",
            progress=100,
            progress_detail="数据库操作完成",
            result=result,
            project=updated,
        )
    except Exception as exc:
        _set(
            job_id,
            status="failed",
            progress_detail="数据库操作失败",
            error=str(exc),
        )


def start(project: dict, payload: dict) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "project_id": project["id"],
        "action": payload.get("action", ""),
        "status": "queued",
        "progress": 0,
        "progress_detail": "等待执行",
        "tdms_processed": 0,
        "tdms_total": 0,
        "tdms_remaining": 0,
        "tdms_phase": "",
        "error": "",
        "result": None,
        "project": None,
    }
    with _LOCK:
        active = [
            item for item in _JOBS.values()
            if item["project_id"] == project["id"] and item["status"] in {"queued", "running"}
        ]
        if active:
            raise RuntimeError("当前项目已有数据库操作正在执行")
        _JOBS[job_id] = job
    threading.Thread(
        target=_run,
        args=(job_id, copy.deepcopy(project), copy.deepcopy(payload)),
        daemon=True,
    ).start()
    return get(job_id) or job


def start_global(payload: dict) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "project_id": "__global_database__",
        "action": payload.get("action", ""),
        "status": "queued",
        "progress": 0,
        "progress_detail": "等待执行",
        "tdms_processed": 0,
        "tdms_total": 0,
        "tdms_remaining": 0,
        "tdms_phase": "",
        "error": "",
        "result": None,
        "project": None,
    }
    with _LOCK:
        active = [
            item for item in _JOBS.values()
            if item["project_id"] == "__global_database__" and item["status"] in {"queued", "running"}
        ]
        if active:
            raise RuntimeError("已有全局数据库操作正在执行")
        _JOBS[job_id] = job
    threading.Thread(
        target=_run,
        args=(job_id, None, copy.deepcopy(payload)),
        daemon=True,
    ).start()
    return get(job_id) or job
