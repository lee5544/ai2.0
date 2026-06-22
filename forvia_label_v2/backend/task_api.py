"""任务插件接口 + 注册表（与卡片对称：tasks/ 下丢一个 py 即扩展）。

Task 对一批样本逐条处理，产出：
  - result:     该样本的计算结果（如异常分数），存入 TaskResultStore.results
  - suggestion: 模型/启发式建议标签 {reason_key, confidence, note}，存入 suggestions，
                等人工"采纳"才写入 LabelTable（多源 → 显式采纳，不污染人工标签）。
"""
from __future__ import annotations

import importlib
import pkgutil


class Task:
    id: str = ""
    title: str = ""
    # 参数声明，前端据此生成控件（同卡片 params 格式）
    params: list[dict] = []

    def process(self, session, index: int, p: dict) -> dict:
        """返回 {"result": {...}} 和/或 {"suggestion": {reason_key, confidence, note}}。"""
        raise NotImplementedError

    def merge_defaults(self, params: dict | None) -> dict:
        out = {spec["key"]: spec.get("default") for spec in self.params}
        if params:
            for k, v in params.items():
                if k in out and v is not None and v != "":
                    out[k] = v
        return out


REGISTRY: "dict[str, Task]" = {}


def register(cls):
    inst = cls()
    if not inst.id:
        raise ValueError(f"Task {cls.__name__} 缺少 id")
    REGISTRY[inst.id] = inst
    return cls


def discover() -> None:
    from . import tasks as tasks_pkg
    for m in pkgutil.iter_modules(tasks_pkg.__path__):
        importlib.import_module(f"{tasks_pkg.__name__}.{m.name}")


def list_tasks() -> list[dict]:
    return [{"id": t.id, "title": t.title, "params": t.params} for t in REGISTRY.values()]


def get_task(task_id: str) -> Task | None:
    return REGISTRY.get(task_id)
