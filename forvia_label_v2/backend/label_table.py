"""标签层：薄封装 SQLite LabelStore。

- add(): 人工标注 -> Label.build_result() -> LabelStore.add_result()（即时落盘）。
- set_typical(): 在最新标注行 note 写/去 [[典型异音]] 标记（方案 A，零 schema 改动）。
- export(): 按范围/典型筛选导出同格式 CSV。
"""
from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config as _config  # noqa: F401 确保 sys.path 注入
from .config import LABEL_HISTORY_COLUMNS, TYPICAL_TAG

from data_manager.label_rules import LABEL_RULES, Label
from data_manager.label_internal_registry import LabelStore

# 共享 Label 运行时（reason/result 解析规则）
RUNTIME = Label(LABEL_RULES)
LABEL_VERSION = str(LABEL_RULES.get("meta", {}).get("version", "unknown"))
DEFAULT_SOURCE = "expert"

REASONS = [{"key": r["key"], "name": r["name"]} for r in RUNTIME.reasons.values()]
CONFIDENCE_OPTIONS = {"强烈 0.9": 0.9, "中等 0.6": 0.6, "微弱 0.3": 0.3}


def build_decision_result(reason_name: str = "", reason_key: str = "",
                          source: str = DEFAULT_SOURCE) -> dict:
    reason_obj = (RUNTIME.get_reason_by_key(reason_key) if reason_key
                  else RUNTIME.get_reason_by_name(reason_name))
    result_obj = RUNTIME.get_result_by_reason(reason_obj["key"])
    return RUNTIME.build_result(result_key=result_obj["key"],
                                reason_key=reason_obj["key"], source=source)


class LabelTable:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.store = LabelStore(self.path)
        self._cache: list[dict] | None = None      # 全部 confirmed 事件（raw，含 id）的内存缓存
        self._lock = threading.RLock()             # 保护缓存 + 写库（确认/删除/保存为并发即发即忘）
        # 镜像（外部数据库）双写：主存储=本地工作空间，每次标注后把该样本同步到外部 db（后台线程，不阻塞）。
        self.mirror_path: str | None = None
        self._mirror_db = None
        self._sync_q = None
        self._sync_thread = None

    # ===== 双写镜像：把本地工作空间的标注同步到外部数据库 =====
    def set_mirror(self, path: str | Path) -> None:
        """设置外部镜像数据库；标注写入本地后会异步把该样本整组事件同步过去。"""
        import queue
        self.mirror_path = str(Path(path).expanduser())
        if self._sync_thread is None:
            self._sync_q = queue.Queue()
            self._sync_thread = threading.Thread(target=self._mirror_worker, daemon=True)
            self._sync_thread.start()

    def _enqueue_sync(self, sample_id) -> None:
        if self._sync_q is not None:
            try:
                self._sync_q.put_nowait(str(sample_id))
            except Exception:
                pass

    def _mirror(self):
        """惰性在后台线程里打开镜像数据库连接（SQLite 连接需在使用它的线程创建）。"""
        if self.mirror_path is None:
            return None
        if self._mirror_db is None:
            try:
                from data_manager.label_database import LabelDatabase
                self._mirror_db = LabelDatabase(Path(self.mirror_path).expanduser())
            except Exception:
                self._mirror_db = None
        return self._mirror_db

    def _mirror_worker(self) -> None:
        while True:
            sid = self._sync_q.get()
            try:
                self._sync_sample_to_mirror(sid)
            except Exception:
                pass
            finally:
                self._sync_q.task_done()

    def _sync_sample_to_mirror(self, sample_id: str) -> None:
        """用本地工作空间里该样本的当前事件，覆盖外部数据库里同一样本的事件（删后插）。"""
        mdb = self._mirror()
        if mdb is None:
            return
        sid = str(sample_id)
        with self._lock:                               # 仅快照本地事件，随后在锁外做外部慢写
            srs = self._srs
            local = [dict(e) for e in (srs.list_label_events(sample_id=sid, statuses={"confirmed"}) if srs else [])]
        try:
            old = mdb.list_label_events(sample_id=sid, statuses={"confirmed"})
            ids = [int(e["id"]) for e in old if e.get("id") is not None]
            if ids:
                mdb.delete_label_events(ids)
        except Exception:
            pass
        if not local:
            return
        rows = [{k: e.get(k) for k in e.keys() if k != "id"} for e in local]
        try:
            mdb.import_label_events(rows)
        except Exception:                              # 样本未登记 → 先登记再重试
            try:
                f0 = local[0]
                mdb.upsert_samples([{"line": f0.get("line", ""), "sn": f0.get("sn", ""),
                                     "sample_id": sid}])
                mdb.import_label_events(rows)
            except Exception:
                pass

    # ===== 内存缓存：读全部从内存，写后只补该样本，避免反复读外接盘上的 DB =====
    @property
    def _srs(self):
        return getattr(self.store, "database", None)

    def _events_raw(self) -> list[dict]:
        if self._cache is None:
            srs = self._srs
            if srs is not None:
                try:
                    self._cache = [dict(e) for e in srs.list_label_events(statuses={"confirmed"})]
                except Exception:
                    self._cache = []
            else:
                self._cache = [dict(r) for r in self.store.list_all()]
        return self._cache

    def invalidate_cache(self) -> None:
        """强制下次从 DB 重新加载（用于"↻ 刷新"按钮，或外部改动后）。"""
        with self._lock:
            self._cache = None

    def _refresh_sample(self, sample_id: str) -> None:
        """写库后只重读该样本的事件并替换缓存里对应条目（一次命中索引的小查询）。"""
        sid = str(sample_id)
        cache = self._events_raw()
        cache[:] = [e for e in cache if str(e.get("sample_id", "")) != sid]
        srs = self._srs
        if srs is not None:
            try:
                cache.extend(dict(e) for e in srs.list_label_events(sample_id=sid, statuses={"confirmed"}))
            except Exception:
                pass

    def _sample_events_raw(self, sample_id: str) -> list[dict]:
        sid = str(sample_id)
        evs = [e for e in self._events_raw() if str(e.get("sample_id", "")) == sid]
        evs.sort(key=lambda e: (str(e.get("timestamp", "")), int(e.get("id", 0) or 0)))
        return evs

    def events_normalized(self) -> list[dict]:
        """全部事件（归一化）——供 snapshot / overview / list_view 使用，全部走内存。"""
        with self._lock:
            return [self.store._normalize_row(e) for e in self._events_raw()]

    def _import_event(self, row: dict) -> None:
        """写入一条标注；若该样本尚未在 samples 表登记（如来自扫描 TDMS 的新件），
        先登记样本再写，避免 'Sample not found'。"""
        srs = self._srs
        try:
            srs.import_label_events([row])
        except Exception:
            srs.upsert_samples([{
                "line": row.get("line", ""), "sn": row.get("sn", ""),
                "sample_id": row.get("sample_id", ""),
                "group_name": row.get("group_name", ""),
                "channel_name": row.get("channel_name", ""),
                "sampling_rate": row.get("sampling_rate"),
            }])
            srs.import_label_events([row])

    def _pick_event(self, sample_id, event_id=None, event_index=None) -> dict | None:
        evs = self._sample_events_raw(sample_id)
        if event_id is not None:
            return next((e for e in evs if int(e.get("id", -1) or -1) == int(event_id)), None)
        if event_index is not None and 0 <= event_index < len(evs):
            return evs[event_index]
        return None

    # ---- 确认正确：复制该条为 source=expert 的新记录（单条 INSERT + 更新缓存）----
    def confirm_event(self, sample_id: str, event_id=None, event_index=None) -> list[dict]:
      with self._lock:
        ev = self._pick_event(sample_id, event_id, event_index)
        if ev is None:
            raise IndexError("event not found")
        new_row = {c: str(ev.get(c, "") or "") for c in LABEL_HISTORY_COLUMNS}
        new_row["source"] = "expert"
        new_row["timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        if self._srs is not None:
            self._import_event(new_row)
            self._refresh_sample(sample_id)
            self._enqueue_sync(sample_id)
        return self.history_for(sample_id)

    # ---- 删除：物理删一条（单条 DELETE + 更新缓存）----
    def delete_event(self, sample_id: str, event_id=None, event_index=None) -> list[dict]:
      with self._lock:
        ev = self._pick_event(sample_id, event_id, event_index)
        if ev is None:
            raise IndexError("event not found")
        eid = int(ev["id"])
        srs = self._srs
        if srs is not None:
            srs.delete_label_events([eid])
            cache = self._events_raw()
            cache[:] = [e for e in cache if int(e.get("id", -1) or -1) != eid]
            self._enqueue_sync(sample_id)
        return self.history_for(sample_id)

    # ---- 写入一条标注（快速路径：仅按 sample_id 查询 + 单条 insert/update，不读/重写整表）----
    def add(self, *, line: str, sn: str, sample_id: str,
            reason_name: str = "", reason_key: str = "",
            reason_confidence: float | None = None,
            note: str = "", source: str = DEFAULT_SOURCE) -> dict:
      with self._lock:
        decision = build_decision_result(reason_name=reason_name,
                                         reason_key=reason_key, source=source)
        srs = getattr(self.store, "database", None)
        if srs is None:                       # 兜底：旧路径
            self.store.add_result(line=line, sn=sn, sample_id=sample_id,
                                  decision_result=decision, label_version=LABEL_VERSION,
                                  reason_confidence=reason_confidence, note=note)
            return decision

        result = decision.get("result") or {}
        reason = decision.get("reason") or {}
        row = self.store._normalize_row({
            "line": line, "sn": sn, "sample_id": sample_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source": decision.get("source"),
            "result_key": result.get("key"), "result_id": result.get("id"),
            "result_name": result.get("name"),
            "reason_key": reason.get("key"), "reason_id": reason.get("id"),
            "reason_name": reason.get("name"), "reason_confidence": reason_confidence,
            "label_version": LABEL_VERSION, "note": note,
        })
        # 同样本「同日 + 同来源」去重：命中则更新该条，否则新增（从内存缓存判断，不读 DB）
        new_key = self.store._same_day_source_key(row)
        match_id = None
        if new_key is not None:
            for ev in self._sample_events_raw(sample_id):
                if self.store._same_day_source_key(self.store._normalize_row(ev)) == new_key:
                    match_id = int(ev["id"])     # 取最后一条匹配
        if match_id is not None:
            srs.update_label_event(match_id, line=line, sn=sn, sample_id=sample_id, label=row)
        else:
            self._import_event(row)              # 样本未登记会自动先登记
        self._refresh_sample(sample_id)          # 写后只补该样本，缓存即时一致
        self._enqueue_sync(sample_id)            # 异步同步该样本到外部数据库
        return decision

    # ---- 最新标签映射（从内存缓存单遍扫描，不读 DB）----
    def latest_label_map(self) -> dict[str, dict]:
      with self._lock:
        latest: dict[str, tuple] = {}
        for e in self._events_raw():
            sid = str(e.get("sample_id", ""))
            ts = str(e.get("timestamp", "") or "")
            cur = latest.get(sid)
            if cur is None or ts >= cur[0]:
                latest[sid] = (ts, e)
        return {sid: {k: str(v.get(k, "") or "") for k in LABEL_HISTORY_COLUMNS}
                for sid, (_, v) in latest.items()}

    def history_for(self, sample_id: str) -> list[dict]:
      with self._lock:
        # 从内存缓存按 sample_id 过滤（不读 DB）；附带稳定的 _id，便于前端按 id 确认/删除
        out = []
        for e in self._sample_events_raw(sample_id):
            row = self.store._normalize_row(e)
            row["_id"] = int(e.get("id", 0) or 0)
            out.append(row)
        return out

    # ---- 给某样本最新标注的 note 加一个标记（如 [[prototype]]），单条更新 + 缓存一致 ----
    def mark_note_tag(self, sample_id: str, tag: str) -> bool:
      with self._lock:
        srs = self._srs
        if srs is None:
            return False
        evs = self._sample_events_raw(sample_id)
        if not evs:
            return False
        ev = evs[-1]
        note = str(ev.get("note", "") or "")
        if tag in note:
            return False                       # 已标记，无需重复
        row = self.store._normalize_row({**ev, "note": (note + " " + tag).strip()})
        srs.update_label_event(int(ev["id"]), line=str(ev.get("line", "")),
                               sn=str(ev.get("sn", "")), sample_id=str(sample_id), label=row)
        self._refresh_sample(sample_id)
        self._enqueue_sync(sample_id)
        return True

    # ---- 标记典型异音（方案 A：note 标记，作用于最新行）----
    def set_typical(self, sample_id: str, flag: bool) -> bool:
      with self._lock:
        srs = self._srs
        if srs is None:
            return self._set_typical_legacy(sample_id, flag)
        events = self._sample_events_raw(sample_id)   # 从缓存取（已按时间排序）
        if not events:
            return False  # 无标签，无法标记
        ev = events[-1]                                # 最新一条
        note = str(ev.get("note", "") or "")
        has = TYPICAL_TAG in note
        if flag and not has:
            note = (TYPICAL_TAG + " " + note).strip()
        elif not flag and has:
            note = note.replace(TYPICAL_TAG, "").strip()
        else:
            return True  # 已是目标状态
        row = self.store._normalize_row({**ev, "note": note})
        srs.update_label_event(int(ev["id"]), line=str(ev.get("line", "")),
                               sn=str(ev.get("sn", "")), sample_id=str(sample_id), label=row)
        self._refresh_sample(sample_id)
        self._enqueue_sync(sample_id)
        return True

    def _set_typical_legacy(self, sample_id: str, flag: bool) -> bool:
        rows = self.store.list_all()
        idxs = [i for i, r in enumerate(rows)
                if str(r.get("sample_id", "")) == str(sample_id)]
        if not idxs:
            return False
        def _ts(i):
            return pd.to_datetime(rows[i].get("timestamp"), errors="coerce")
        latest = max(idxs, key=_ts)
        note = str(rows[latest].get("note", "") or "")
        has = TYPICAL_TAG in note
        if flag and not has:
            note = (TYPICAL_TAG + " " + note).strip()
        elif not flag and has:
            note = note.replace(TYPICAL_TAG, "").strip()
        else:
            return True
        rows[latest]["note"] = note
        self.store._write_all([{k: str(r.get(k, "") or "") for k in LABEL_HISTORY_COLUMNS}
                               for r in rows])
        return True

    # ---- 导出 ----
    def export(self, out_path: str | Path, *, scope_line: str = "",
               only_typical: bool = False) -> tuple[Path, int]:
        rows = self.store.list_all()
        if scope_line:
            rows = [r for r in rows if str(r.get("line", "")) == scope_line]
        if only_typical:
            rows = [r for r in rows if TYPICAL_TAG in str(r.get("note", ""))]
        out = Path(out_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=LABEL_HISTORY_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in LABEL_HISTORY_COLUMNS})
        return out, len(rows)

    @staticmethod
    def default_timestamped_name(base: str = "label_history_export") -> str:
        return f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
