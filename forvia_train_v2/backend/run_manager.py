from __future__ import annotations

import os
import copy
import json
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from ml.training.app_api import model_id, normalize_database_input, validate_config
from .config import PROJECT_ROOT, RESULTS_DIR, RUN_DIR
from .store import (
    create_run,
    get_project,
    get_run,
    list_active_runs,
    now,
    update_run,
)
from ml.dataset.build import (
    discover_feature_csvs,
    validate_sample_view_features_alignment,
)


PROCESSES: dict[str, subprocess.Popen] = {}
LOCK = threading.Lock()

DATA_STAGES: list[tuple[str, str, list[str]]] = [
    (
        "training_data",
        "按 ML 规则筛选样本和标签",
        ["-m", "ml.dataset.generate", "--config", "{config}"],
    ),
    (
        "extract_features",
        "提取特征",
        [
            "ml/dataset/build.py",
            "--config",
            "{config}",
            "--label-records-db-path",
            "{database}",
            "--manifest-path",
            "{manifest}",
            "--data-root",
            "{data_root}",
        ],
    ),
]
TRAIN_STAGES: list[tuple[str, str, list[str]]] = [
    (
        "train_model",
        "训练模型",
        ["-m", "ml.train", "--config", "{config}", "--step", "train"],
    ),
]
STAGES: dict[str, list[tuple[str, str, list[str]]]] = {
    "data": DATA_STAGES,
    "train": TRAIN_STAGES,
    "full": [*DATA_STAGES, *TRAIN_STAGES],
}
CUSTOM_COMMAND_KINDS = {"predict", "augmentation"}


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), *([existing] if existing else [])]
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _python_command(args: list[str]) -> list[str]:
    if getattr(sys, "frozen", False) or os.environ.get("FORVIA_CONSOLE_FROZEN") == "1":
        return [sys.executable, "--forvia-python", *args]
    return [sys.executable, *args]


def _run_paths(project_id: str, run_id: str) -> tuple[Path, Path, Path]:
    run_dir = RUN_DIR / project_id / run_id
    artifact_root = RESULTS_DIR
    log_path = run_dir / "run.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    return run_dir, artifact_root, log_path


def _project_config_path(project: dict) -> Path:
    path = Path(str(project.get("config_path") or "")).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"项目 cfg YAML 不存在: {path}")
    return path.resolve()


def _assert_no_active_model_run(project_id: str, config: dict) -> None:
    target_model_id = model_id(config)
    for active in list_active_runs():
        active_project = get_project(str(active.get("project_id") or ""))
        if not active_project or model_id(active_project["config"]) != target_model_id:
            continue
        raise RuntimeError(
            f"模型 {target_model_id} 已有运行中的任务："
            f"{active.get('kind')} / {active.get('current_stage') or active.get('status')}。"
            "请等待完成或取消后再启动新任务。"
        )


def _validate_train_dataset(config: dict) -> dict:
    model_dir = RESULTS_DIR / model_id(config)
    return validate_sample_view_features_alignment(
        model_dir,
        discover_feature_csvs(model_dir),
    )


def _training_total(config: dict) -> int:
    train_cfg = config.get("train") if isinstance(config.get("train"), dict) else {}
    for key in ("model_seed_list", "seed_list"):
        values = train_cfg.get(key)
        if isinstance(values, list) and values:
            return len(dict.fromkeys(str(value) for value in values))
    try:
        return max(1, int(train_cfg.get("seed_runs") or 1))
    except Exception:
        return 1


def _training_progress_from_line(line: str) -> tuple[int, int] | None:
    match = re.search(r"Seed Run\s+(\d+)\s*/\s*(\d+)", line, flags=re.IGNORECASE)
    if not match:
        return None
    current, total = int(match.group(1)), int(match.group(2))
    return max(0, current - 1), max(1, total)


def _feature_progress_from_line(line: str) -> tuple[int, int] | None:
    match = re.search(
        r"\[FEATURE_PROGRESS\]\s+processed=(\d+)\s+total=(\d+)",
        line,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    processed, total = int(match.group(1)), int(match.group(2))
    return max(0, min(processed, total)), max(1, total)


def _augmentation_progress_from_line(line: str) -> tuple[int, int] | None:
    match = re.search(
        r"\[AUGMENT_PROGRESS\]\s+processed=(\d+)\s+total=(\d+)",
        line,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    processed, total = int(match.group(1)), int(match.group(2))
    return max(0, min(processed, total)), max(1, total)


def _pid_alive(pid: object) -> bool:
    try:
        value = int(pid or 0)
        if value <= 0:
            return False
        os.kill(value, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def reconcile_active_runs() -> None:
    for run in list_active_runs():
        if _pid_alive(run.get("supervisor_pid")) or _pid_alive(run.get("pid")):
            continue
        update_run(
            run["id"],
            status="interrupted",
            error="任务执行进程已退出，无法继续训练。",
            pid=None,
            supervisor_pid=None,
            finished_at=now(),
        )


def _launch_worker(run_id: str) -> int:
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(
        _python_command(["-m", "forvia_train_v2.backend.run_worker", "--run-id", run_id]),
        cwd=str(PROJECT_ROOT),
        env=_subprocess_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
        creationflags=creationflags,
        close_fds=True,
    )
    return proc.pid


def start(project_id: str, kind: str, *, command: list[str] | None = None) -> dict:
    project = get_project(project_id)
    if not project:
        raise KeyError("项目不存在")
    if kind not in STAGES and kind not in CUSTOM_COMMAND_KINDS:
        raise ValueError(f"未知运行类型: {kind}")
    if kind not in CUSTOM_COMMAND_KINDS:
        validation = validate_config(copy.deepcopy(project["config"]))
        if not validation["ok"]:
            raise ValueError("配置校验失败: " + "; ".join(validation["errors"]))
    with LOCK:
        _assert_no_active_model_run(project_id, project["config"])
        if kind == "train":
            _validate_train_dataset(project["config"])
        progress_detail = f"已训练 0 / {_training_total(project['config'])}" if kind == "train" else ""
        run = create_run({"project_id": project_id, "kind": kind, "progress_detail": progress_detail})
    run_dir, artifact_root, log_path = _run_paths(project_id, run["id"])
    config_path = _project_config_path(project)
    update_run(
        run["id"],
        config_path=str(config_path),
        artifact_root=str(artifact_root),
        log_path=str(log_path),
    )
    (run_dir / "job.json").write_text(
        json.dumps({"command": command}, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        supervisor_pid = _launch_worker(run["id"])
    except Exception as exc:
        update_run(
            run["id"],
            status="failed",
            error=f"无法启动独立任务执行器: {exc}",
            finished_at=now(),
        )
        raise
    launched = get_run(run["id"])
    if launched and launched["status"] in ("queued", "running"):
        update_run(run["id"], supervisor_pid=supervisor_pid)
    return get_run(run["id"]) or run


def _execute(run_id: str, config: dict, command: list[str] | None) -> None:
    run = get_run(run_id)
    if not run:
        return
    stages = (
        [(run["kind"], "模型推理" if run["kind"] == "predict" else "原始信号增强", command or [])]
        if run["kind"] in CUSTOM_COMMAND_KINDS
        else STAGES[run["kind"]]
    )
    log_path = Path(run["log_path"])
    config_path = run["config_path"]
    try:
        update_run(run_id, status="running", started_at=now(), progress=0)
        with log_path.open("a", encoding="utf-8") as log:
            for idx, (stage_id, title, template) in enumerate(stages):
                current = get_run(run_id)
                if current and current["status"] == "canceled":
                    return
                normalized = normalize_database_input(copy.deepcopy(config))
                database = str((normalized.get("database") or {}).get("label_records_db_path") or "")
                manifest = str((normalized.get("database") or {}).get("manifest_path") or "")
                data_root = str(normalized.get("data_root") or "")
                args = [
                    part.format(config=config_path, database=database, manifest=manifest, data_root=data_root)
                    for part in template
                ]
                stage_progress = idx / len(stages) * 100
                detail = f"已训练 0 / {_training_total(config)}" if run["kind"] == "train" else ""
                if stage_id == "extract_features":
                    detail = "准备提取特征"
                update_run(run_id, current_stage=stage_id, progress=stage_progress, progress_detail=detail)
                cmd = _python_command(args)
                log.write(f"\n===== {title} =====\n$ {' '.join(cmd)}\n")
                log.flush()
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    env=_subprocess_env(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=(os.name != "nt"),
                    creationflags=creationflags,
                )
                with LOCK:
                    PROCESSES[run_id] = proc
                update_run(run_id, pid=proc.pid)
                for output_line in proc.stdout or []:
                    log.write(output_line)
                    log.flush()
                    if run["kind"] == "train":
                        progress_pair = _training_progress_from_line(output_line)
                        if progress_pair:
                            completed, total = progress_pair
                            update_run(
                                run_id,
                                progress=completed / total * 100,
                                progress_detail=f"已训练 {completed} / {total}",
                            )
                    if stage_id == "extract_features":
                        feature_progress = _feature_progress_from_line(output_line)
                        if feature_progress:
                            processed, total = feature_progress
                            stage_ratio = processed / total
                            update_run(
                                run_id,
                                progress=(idx + stage_ratio) / len(stages) * 100,
                                progress_detail=f"已提取 {processed} / {total}",
                            )
                    if stage_id == "augmentation":
                        augmentation_progress = _augmentation_progress_from_line(output_line)
                        if augmentation_progress:
                            processed, total = augmentation_progress
                            update_run(
                                run_id,
                                progress=processed / total * 100,
                                progress_detail=f"已增强并提取 {processed} / {total}",
                            )
                rc = proc.wait()
                with LOCK:
                    PROCESSES.pop(run_id, None)
                current = get_run(run_id)
                if current and current["status"] == "canceled":
                    return
                if rc != 0:
                    raise RuntimeError(f"{title} 执行失败，返回码 {rc}")
        current = get_run(run_id) or {}
        progress_detail = (
            f"已训练 {_training_total(config)} / {_training_total(config)}"
            if run["kind"] == "train"
            else str(current.get("progress_detail") or "")
        )
        update_run(
            run_id,
            status="succeeded",
            current_stage="done",
            progress=100,
            progress_detail=progress_detail,
            return_code=0,
            pid=None,
            supervisor_pid=None,
            finished_at=now(),
        )
    except Exception as exc:
        update_run(
            run_id,
            status="failed",
            return_code=1,
            error=str(exc),
            pid=None,
            supervisor_pid=None,
            finished_at=now(),
        )


def cancel(run_id: str) -> dict | None:
    run = get_run(run_id)
    if not run or run["status"] not in ("queued", "running"):
        return run
    update_run(run_id, status="canceled", finished_at=now())
    with LOCK:
        proc = PROCESSES.get(run_id)
    if proc and proc.poll() is None:
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
    elif _pid_alive(run.get("pid")):
        try:
            if os.name == "nt":
                os.kill(int(run["pid"]), signal.SIGTERM)
            else:
                os.killpg(int(run["pid"]), signal.SIGTERM)
        except OSError:
            pass
    elif _pid_alive(run.get("supervisor_pid")):
        try:
            if os.name == "nt":
                os.kill(int(run["supervisor_pid"]), signal.SIGTERM)
            else:
                os.killpg(int(run["supervisor_pid"]), signal.SIGTERM)
        except OSError:
            pass
    return get_run(run_id)


def tail_log(run_id: str, max_chars: int = 30000) -> str:
    run = get_run(run_id)
    if not run or not run["log_path"]:
        return ""
    path = Path(run["log_path"])
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]
