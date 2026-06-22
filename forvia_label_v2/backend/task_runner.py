"""任务执行：后台线程跑任务、更新进度、结果存入内存 store。"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime

from .task_api import get_task


class TaskRun:
    def __init__(self, run_id, task_id, task_title, scope, total):
        self.run_id = run_id
        self.task_id = task_id
        self.task_title = task_title
        self.scope = scope
        self.total = total
        self.done = 0
        self.errors = 0
        self.status = "pending"          # pending/running/done/error
        self.error = ""
        self.started = datetime.now().isoformat(timespec="seconds")
        self.finished = ""
        self.results: dict[str, dict] = {}      # sample_id -> result data
        self.suggestions: dict[str, dict] = {}  # sample_id -> {reason_key,confidence,note,accepted}

    def summary(self) -> dict:
        return {"run_id": self.run_id, "task_id": self.task_id, "task_title": self.task_title,
                "scope": self.scope, "status": self.status, "total": self.total,
                "done": self.done, "errors": self.errors, "error": self.error,
                "started": self.started, "finished": self.finished,
                "n_results": len(self.results), "n_suggestions": len(self.suggestions)}


# run_id -> TaskRun
RUNS: "dict[str, TaskRun]" = {}
_LOCK = threading.Lock()


def scope_indices(session, scope: str) -> list[int]:
    """scope: 'all' 或 线名。"""
    sv = session.sample_view
    if not scope or scope == "all":
        return list(range(len(sv)))
    return [i for i, r in sv.reset_index(drop=True).iterrows()
            if str(r.get("line", "")) == scope]


def _execute(run: TaskRun, task, session, indices, params):
    run.status = "running"
    try:
        for idx in indices:
            sid = str(session.row(idx).get("sample_id", ""))
            try:
                out = task.process(session, idx, params) or {}
                if isinstance(out.get("result"), dict):
                    run.results[sid] = out["result"]
                sug = out.get("suggestion")
                if isinstance(sug, dict) and sug.get("reason_key"):
                    sug = dict(sug); sug.setdefault("accepted", False)
                    run.suggestions[sid] = sug
            except Exception:
                run.errors += 1
            run.done += 1
        run.status = "done"
    except Exception as e:
        run.status = "error"; run.error = str(e)
    finally:
        run.finished = datetime.now().isoformat(timespec="seconds")


def start_run(session, task_id: str, scope: str, params: dict) -> TaskRun:
    task = get_task(task_id)
    if task is None:
        raise KeyError(f"未知任务: {task_id}")
    p = task.merge_defaults(params)
    indices = scope_indices(session, scope)
    run = TaskRun(uuid.uuid4().hex[:8], task_id, task.title, scope, len(indices))
    with _LOCK:
        RUNS[run.run_id] = run
        # 防止 run 无限增长：只保留最近 50 个
        if len(RUNS) > 50:
            for rid in sorted(RUNS, key=lambda k: RUNS[k].started)[:len(RUNS) - 50]:
                RUNS.pop(rid, None)
    threading.Thread(target=_execute, args=(run, task, session, indices, p),
                     daemon=True).start()
    return run


def list_runs() -> list[dict]:
    return [r.summary() for r in sorted(RUNS.values(), key=lambda x: x.started, reverse=True)]


def get_run(run_id: str) -> TaskRun | None:
    return RUNS.get(run_id)


def suggestions_for_sample(sample_id: str) -> list[dict]:
    """跨所有 run 收集该 sample 的建议（用于通道内展示/采纳）。"""
    out = []
    for r in RUNS.values():
        s = r.suggestions.get(sample_id)
        if s:
            out.append({"run_id": r.run_id, "task_title": r.task_title, **s})
    return out
