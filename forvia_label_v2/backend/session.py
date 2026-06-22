"""会话/初始化层：LabelSession 是唯一状态中心。

init_session() 一次性：载入 sample_view、解析所有 tdms 路径、载入(可选)标签表。
之后 overview / 取样本 / 标注写入都只依赖 SESSION。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path

import pandas as pd

import tempfile

from .config import LABEL_HISTORY_COLUMNS, TYPICAL_TAG
from .label_table import LabelTable
from .mock import mock_label_history, mock_sample_view
from .tdms_locator import ManifestAdapter, TdmsLocator
from data_manager.label_database import LabelDatabase, resolve_database_path

# 一个 tdms 的分析原子（目前只有 up/down；将来扩展片段/拼接在此扩充）
SAMPLE_ATOMS = ("up", "down")

# 完整标签结果表的列（含 group_name/channel_name）
LABEL_TABLE_COLUMNS = [
    "line", "sn", "sample_id", "group_name", "channel_name", "timestamp", "source",
    "result_key", "result_id", "result_name", "reason_key", "reason_id", "reason_name",
    "reason_confidence", "label_version", "note",
]
_LABEL_ONLY_COLS = ["timestamp", "source", "result_key", "result_id", "result_name",
                    "reason_key", "reason_id", "reason_name", "reason_confidence",
                    "label_version", "note"]


# ---------- CSV 读取 ----------
def _read_csv(path: str | Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        return None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(p, encoding=enc, dtype=str).fillna("")
        except Exception:
            continue
    return None


def build_sample_view_from_tdms(tdms_root: str, *, limit: int = 50000) -> pd.DataFrame:
    """无 sample_view 时：扫描 TDMS 目录所有件，按文件名规则解析 line/sn 生成 sample_view。"""
    from data_manager.tdms_read import _parse_filename, _resolve_line_rule
    from data_manager.line_rules import LINE_RULES
    from data_manager.tdms_read import iter_tdms_files
    root = Path(tdms_root).expanduser()
    rows, n = [], 0
    for f in iter_tdms_files(root):
        n += 1
        if n > limit:
            break
        try:
            line_name, line_rule = _resolve_line_rule(f, LINE_RULES, None)
            meta = _parse_filename(f.name, line_rule.get("filename"))
            sn = str(meta.get("sn") or "")
        except Exception:
            continue
        if not sn:
            continue
        rel = str(f.relative_to(root))
        for d in SAMPLE_ATOMS:
            rows.append({"line": line_name, "sn": sn, "sample_id": f"{sn}_{d}",
                         "relative_path": rel})
    cols = ["line", "sn", "sample_id", "relative_path"]
    return pd.DataFrame(rows, columns=cols)


def build_sample_view_from_db(db_file: Path) -> pd.DataFrame:
    """有数据库时：用 db 已登记的样本生成 sample_view。"""
    rows = LabelDatabase(db_file).list_samples(active_only=True)
    cols = ["line", "sn", "sample_id", "group_name", "channel_name", "sampling_rate"]
    out = [{c: str(r.get(c, "") or "") for c in cols} for r in rows]
    return pd.DataFrame(out, columns=cols)


def _direction_of(sample_id: str) -> str:
    s = str(sample_id).lower()
    if s.endswith("_down"):
        return "down"
    if s.endswith("_up"):
        return "up"
    return "up"


def _explode_list_sample_ids(sv: pd.DataFrame) -> pd.DataFrame:
    """有的 sample_view 把一个 tdms 的 up/down 合并成一行，sample_id 写成列表
    （如 ["AB6A0001_up","AB6A0001_down"]）。这里把这种行拆成每个原子一行，
    避免列表整体被当成一个样本（导致列表里出现 up+down 同一行的脏行）。"""
    if "sample_id" not in sv.columns:
        return sv

    def _parse(v):
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        s = str(v or "").strip()
        if s.startswith("[") and s.endswith("]"):
            for loader in (json.loads, __import__("ast").literal_eval):
                try:
                    arr = loader(s)
                    if isinstance(arr, (list, tuple)):
                        return [str(x) for x in arr]
                except Exception:
                    continue
        return None

    rows, changed = [], False
    for _, r in sv.iterrows():
        lst = _parse(r.get("sample_id"))
        if lst:
            changed = True
            for sid in lst:
                nr = r.to_dict(); nr["sample_id"] = sid; rows.append(nr)
        else:
            rows.append(r.to_dict())
    return pd.DataFrame(rows).reset_index(drop=True) if changed else sv


def complete_samples(sv: pd.DataFrame) -> pd.DataFrame:
    """补齐每个 tdms(按 sn) 的所有分析原子(up/down)，并标识哪些是原始输入。

    - 原始 sample_view 的行：is_input=True，原有列全部保留。
    - 自动补齐的行：is_input=False，复制同 sn 的 line 等共享信息，sample_id=sn_<dir>。
    """
    sv = sv.copy()
    if "sample_id" not in sv.columns or "sn" not in sv.columns:
        sv["is_input"] = True
        return sv.reset_index(drop=True)
    sv["is_input"] = True
    extra_rows = []
    # 直接清空的“通道相关”列（补齐行不应继承错误的通道元数据）
    channel_cols = [c for c in ("group_name", "channel_name", "sample_id") if c in sv.columns]
    for sn, grp in sv.groupby(sv["sn"].astype(str)):
        present = {_direction_of(s) for s in grp["sample_id"].astype(str)}
        missing = [d for d in SAMPLE_ATOMS if d not in present]
        if not missing:
            continue
        ref = grp.iloc[0].to_dict()
        for d in missing:
            new = dict(ref)
            for c in channel_cols:
                new[c] = ""
            new["sample_id"] = f"{sn}_{d}"
            new["sn"] = str(sn)
            new["is_input"] = False
            extra_rows.append(new)
    if extra_rows:
        sv = pd.concat([sv, pd.DataFrame(extra_rows)], ignore_index=True).fillna("")
    # 按 sn + 方向(up 在前) 排序，使 up/down 相邻
    sv["_dord"] = sv["sample_id"].map(lambda s: 0 if _direction_of(s) == "up" else 1)
    sv = sv.sort_values(["sn", "_dord"], kind="stable").drop(columns=["_dord"]).reset_index(drop=True)
    return sv


# ---------- 路径解析 ----------
class PathResolver:
    """tdms_root + sample_view -> {sample_id: 实际 tdms 文件路径}。

    只做"快速直接匹配"，绝不递归扫描整个 tdms_root（外接卷/海量目录会卡死）：
      ① tdms_path 绝对路径；
      ② tdms_root / relative_path（含 .tdms <-> .tdms.zst 互换、去掉多余前缀等变体）。
    解析不到的样本留待加载时按需处理，不在初始化做全盘 rglob。
    """

    TDMS_EXTS = (".tdms.zst", ".tdms")

    def __init__(self, tdms_root: str | Path | None, sample_view_dir: str | Path | None = None):
        self.root = Path(tdms_root).expanduser() if tdms_root else None
        self.sv_dir = Path(sample_view_dir).expanduser() if sample_view_dir else None
        self._name_index: dict[str, Path] | None = None

    def _ext_variants(self, rel: str) -> list[str]:
        """生成 .tdms / .tdms.zst 互换的相对路径候选。"""
        out = [rel]
        low = rel.lower()
        if low.endswith(".tdms.zst"):
            out.append(rel[: -len(".zst")])              # 去掉 .zst
        elif low.endswith(".tdms"):
            out.append(rel + ".zst")                     # 加 .zst
        return out

    def _rel_variants(self, rel: str) -> list[str]:
        """去掉相对路径里多余的 root 名前缀（如 relative_path 含 'factory_raw/'）。"""
        variants = [rel]
        if self.root is not None:
            prefix = self.root.name + "/"
            if rel.startswith(prefix):
                variants.append(rel[len(prefix):])
        # 仅文件名兜底（同名直接放在 root 下）
        base = Path(rel).name
        if base and base != rel:
            variants.append(base)
        return variants

    def _suffix_after_root_name(self, path_str: str) -> str | None:
        """取 path_str 中 tdms_root 目录名（如 factory_raw）之后的相对部分。

        处理 tdms_path 绝对路径挂载前缀不同、relative_path 带 factory_raw/ 前缀等情况。
        """
        if not self.root:
            return None
        name = self.root.name
        if not name:
            return None
        s = path_str.replace("\\", "/")
        marker = "/" + name + "/"
        if marker in s:
            return s.split(marker, 1)[1]
        if s.startswith(name + "/"):
            return s[len(name) + 1:]
        return None

    def candidates(self, row: pd.Series, sn: str) -> list[Path]:
        """按优先级返回所有候选路径（不判断存在性），供解析与诊断复用。"""
        out: list[Path] = []
        tp = str(row.get("tdms_path", "")).strip()
        rp = str(row.get("relative_path", "")).strip()
        # ① tdms_path 绝对路径（含扩展名互换）
        if tp:
            for cand in self._ext_variants(tp):
                out.append(Path(cand).expanduser())
        bases = [b for b in (self.root, self.sv_dir) if b is not None]
        for base in bases:
            # ② base / relative_path 变体
            if rp:
                for rel in self._rel_variants(rp):
                    for cand in self._ext_variants(rel):
                        out.append(base / cand)
            # ③ 按 root 目录名(factory_raw)切分后拼接：兼容挂载前缀不同 / 前缀层级不一致
            for src in (tp, rp):
                suf = self._suffix_after_root_name(src) if src else None
                if suf:
                    for cand in self._ext_variants(suf):
                        out.append(base / cand)
            # ④ 直接放在 base 下的同名文件
            if rp:
                bn = Path(rp).name
                for cand in self._ext_variants(bn):
                    out.append(base / cand)
        # 去重保序
        seen, uniq = set(), []
        for p in out:
            k = str(p)
            if k not in seen:
                seen.add(k); uniq.append(p)
        return uniq

    def _name_idx(self) -> dict[str, Path]:
        """按文件名建索引——只扫描 sample_view 所在目录（数据就在这里，且体量小），
        绝不扫描庞大的 tdms_root，避免外接卷卡死。"""
        if self._name_index is not None:
            return self._name_index
        idx: dict[str, Path] = {}
        scan_dir = self.sv_dir
        if scan_dir and scan_dir.exists():
            count = 0
            try:
                for f in scan_dir.rglob("*"):
                    if not f.is_file():
                        continue
                    nm = f.name
                    if nm.endswith(self.TDMS_EXTS):
                        idx.setdefault(nm, f)
                        for ext in self.TDMS_EXTS:
                            if nm.endswith(ext):
                                idx.setdefault(nm[: -len(ext)], f)  # 去扩展名键
                                break
                        count += 1
                        if count > 200000:   # 安全上限
                            break
            except Exception:
                pass
        self._name_index = idx
        return idx

    def _index_lookup(self, row: pd.Series) -> Path | None:
        rp = str(row.get("relative_path", "")).strip()
        tp = str(row.get("tdms_path", "")).strip()
        base = Path(rp or tp).name
        if not base:
            return None
        idx = self._name_idx()
        if base in idx:
            return idx[base]
        for ext in self.TDMS_EXTS:           # 文件名扩展名互换后再查
            if base.endswith(ext):
                stem = base[: -len(ext)]
                if stem in idx:
                    return idx[stem]
                for ext2 in self.TDMS_EXTS:
                    if (stem + ext2) in idx:
                        return idx[stem + ext2]
        return None

    def resolve_one(self, row: pd.Series, sn: str) -> Path | None:
        for p in self.candidates(row, sn):
            try:
                if p.exists():
                    return p
            except Exception:
                continue
        return self._index_lookup(row)   # 兜底：sample_view 目录内按文件名匹配

    def build(self, sample_view: pd.DataFrame) -> tuple[dict[str, Path], list[str]]:
        """返回 (path_map[sample_id]=Path, 未解析 sample_id 列表)。"""
        path_map: dict[str, Path] = {}
        unresolved: list[str] = []
        # 同一 sn 的多个 sample_id（up/down）共用一个 tdms 文件
        sn_cache: dict[str, Path | None] = {}
        for _, row in sample_view.iterrows():
            sid = str(row.get("sample_id", "")).strip()
            sn = str(row.get("sn", "")).strip()
            if not sid:
                continue
            if sn in sn_cache:
                resolved = sn_cache[sn]
            else:
                resolved = self.resolve_one(row, sn)
                sn_cache[sn] = resolved
            if resolved is not None:
                path_map[sid] = resolved
            else:
                unresolved.append(sid)
        return path_map, unresolved


# ---------- 会话 ----------
STATUS_LABEL = {"registered": "已注册", "unregistered": "未注册", "missing": "缺失"}


class LabelSession:
    def __init__(self, sample_view, label_table, path_map, status_map,
                 tdms_root, label_records_db_path, is_mock, sample_view_path="",
                 default_source="expert", has_db=False):
        self.sample_view: pd.DataFrame = sample_view.reset_index(drop=True)
        self.label_table: LabelTable = label_table
        self.path_map: dict[str, Path] = path_map
        self.status_map: dict[str, str] = status_map
        self.unresolved: list[str] = [sid for sid, st in status_map.items() if st == "missing"]
        self.tdms_root = str(tdms_root or "")
        self.sample_view_path = str(sample_view_path or "")
        self.label_records_db_path = str(label_records_db_path or "")
        self.is_mock = is_mock
        self.default_source = default_source or "expert"
        self.has_db = bool(has_db)
        # 本任务里被人工改动过的 sample_id（保存/确认/删除/标记典型）→ 列表里标绿、可筛选。
        # 跨重启持久：落在任务工作空间 forvia_label_v2/_data/changed/<task_key>.json。
        self.changed_state_path: Path | None = None
        self.changed_ids: set[str] = set()
        self.reference_map: dict[tuple[str, str], str] = {}   # (line,sn)->reference（tdms_manifest.csv）
        self.reference_by_sn: dict[str, str] = {}             # sn->reference（line 缺失时兜底）
        self._changed_lock = threading.Lock()   # 保护 changed_ids 并发写盘

    def _reference_of(self, sv) -> str:
        """该样本的 reference：优先用 tdms_manifest.csv 的映射（(line,sn)→sn 兜底），否则用 sample_view 自带列。"""
        line, sn = str(sv.get("line", "")), str(sv.get("sn", ""))
        ref = self.reference_map.get((line.strip().lower(), sn.strip().lower())) \
            or self.reference_by_sn.get(sn.strip().lower())
        return ref or str(sv.get("reference", "") or "")

    def mark_changed(self, sample_id: str) -> None:
        if not sample_id:
            return
        with self._changed_lock:
            if str(sample_id) in self.changed_ids:
                return
            self.changed_ids.add(str(sample_id))
            self._persist_changed()

    def _persist_changed(self) -> None:
        p = self.changed_state_path
        if not p:
            return
        try:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            Path(p).write_text(json.dumps(sorted(self.changed_ids), ensure_ascii=False),
                               encoding="utf-8")
        except Exception:
            pass

    # 每个 sample_id 的最新标签（纯 python，单遍扫描，避免 pandas 开销）
    @staticmethod
    def _latest_from_events(events: list[dict]) -> dict[str, dict]:
        latest: dict[str, tuple] = {}
        for r in events:
            sid = str(r.get("sample_id", ""))
            ts = str(r.get("timestamp", "") or "")
            cur = latest.get(sid)
            if cur is None or ts >= cur[0]:
                latest[sid] = (ts, r)
        return {sid: {k: str(v.get(k, "") or "") for k in LABEL_HISTORY_COLUMNS}
                for sid, (_, v) in latest.items()}

    def latest_label_map(self, events: list[dict] | None = None) -> dict[str, dict]:
        if events is None:
            events = self.label_table.events_normalized()
        return self._latest_from_events(events)

    @staticmethod
    def _expert_ids_from_events(events: list[dict]) -> set[str]:
        """有任意一条 source=expert 标注的 sample_id 集合。"""
        return {str(e.get("sample_id", "")) for e in events
                if str(e.get("source", "")).strip().lower() == "expert"}

    def row(self, index: int) -> pd.Series:
        return self.sample_view.iloc[index]

    def sample_view_label(self, index: int) -> dict | None:
        """输入了 sample_view 时，原样返回该 sn/sample_id 行的全部列（不挑列名、不论是否带标签）。
        未输入 sample_view（db/扫描 TDMS 模式）则返回 None，不展示该面板。"""
        if not getattr(self, "from_sample_view", False):
            return None
        if index < 0 or index >= len(self.sample_view):
            return None
        sv = self.sample_view.iloc[index]
        out = {}
        for c in self.sample_view.columns:
            if c == "is_input":
                continue
            out[str(c)] = str(sv.get(c, "") or "")
        return out

    def _sv_lookup(self):
        sv_by_id, idx_by_id = {}, {}
        for i, (_, r) in enumerate(self.sample_view.iterrows()):
            sid = str(r.get("sample_id", ""))
            sv_by_id[sid] = {"group_name": str(r.get("group_name", "")),
                             "channel_name": str(r.get("channel_name", "")),
                             "line": str(r.get("line", "")), "sn": str(r.get("sn", ""))}
            idx_by_id.setdefault(sid, i)
        return sv_by_id, idx_by_id

    def label_records(self, labeled_only: bool = True, changed_only: bool = False) -> list[dict]:
        """构造完整标签结果表（16 列）。labeled_only=True 只含有标注记录的行；
        changed_only=True 只含本次任务里被人工改动过的样本（确认/删除/保存/标记典型）。"""
        sv_by_id, idx_by_id = self._sv_lookup()
        changed = self.changed_ids if changed_only else None
        out = []
        if labeled_only:
            for rec in self.label_table.events_normalized():
                sid = str(rec.get("sample_id", ""))
                if changed is not None and sid not in changed:
                    continue
                base = sv_by_id.get(sid, {})
                row = {c: str(rec.get(c, "") or "") for c in LABEL_TABLE_COLUMNS}
                row["group_name"] = base.get("group_name", "") or row.get("group_name", "")
                row["channel_name"] = base.get("channel_name", "") or row.get("channel_name", "")
                row["line"] = row["line"] or base.get("line", "")
                row["sn"] = row["sn"] or base.get("sn", "")
                row["index"] = idx_by_id.get(sid, -1)
                out.append(row)
        else:
            latest = self.latest_label_map()
            for i, (_, r) in enumerate(self.sample_view.iterrows()):
                sid = str(r.get("sample_id", ""))
                if changed is not None and sid not in changed:
                    continue
                lab = latest.get(sid, {})
                row = {"line": str(r.get("line", "")), "sn": str(r.get("sn", "")),
                       "sample_id": sid, "group_name": str(r.get("group_name", "")),
                       "channel_name": str(r.get("channel_name", ""))}
                for c in _LABEL_ONLY_COLS:
                    row[c] = str(lab.get(c, "") or "")
                row["index"] = i
                out.append(row)
        return out

    def _list_base_row(self, i: int, sv) -> dict:
        sid = str(sv.get("sample_id", ""))
        status = self.status_map.get(sid, "missing")
        tdms = "缺失" if status == "missing" else (STATUS_LABEL.get(status, "未注册") if self.has_db else "—")
        is_input = str(sv.get("is_input", True)) not in ("False", "false", "0", "")
        return {"index": i, "is_input": is_input, "tdms": tdms,
                "line": str(sv.get("line", "")), "sn": str(sv.get("sn", "")), "sample_id": sid,
                "reference": self._reference_of(sv),
                "group_name": str(sv.get("group_name", "")),
                "channel_name": str(sv.get("channel_name", ""))}

    def list_view(self, events: list[dict] | None = None,
                  label_source: str = "db",
                  expert_ids: set[str] | None = None) -> dict:
        """列表总览数据 + 动态列。label_source 决定标签列取自哪：
        - 'db'：取自数据库/内部库（有库则 16 列、多事件展开成多行）。
        - 'sample_view'：取自 sample_view 行自带的列（不读库，不展开）。
        has_expert：该样本是否存在任意 source=expert 的标注（用于列表标绿）。
        """
        if expert_ids is None:
            _ev = events if events is not None else self.label_table.events_normalized()
            expert_ids = self._expert_ids_from_events(_ev)
        if self.has_db and label_source == "db":
            columns = ["index", "is_input", "changed"] + LABEL_TABLE_COLUMNS + ["tdms"]
            rows = []
            if events is None:
                events = self.label_table.events_normalized()
            events_by_sid: dict[str, list] = {}
            for rec in events:
                events_by_sid.setdefault(str(rec.get("sample_id", "")), []).append(rec)
            for i, (_, sv) in enumerate(self.sample_view.iterrows()):
                base = self._list_base_row(i, sv)
                changed = base["sample_id"] in self.changed_ids
                has_expert = base["sample_id"] in expert_ids
                evs = events_by_sid.get(base["sample_id"], [])
                if not evs:
                    rows.append({**base, "changed": changed, "has_expert": has_expert,
                                 **{c: "" for c in _LABEL_ONLY_COLS},
                                 "labeled": False, "typical": False})
                else:
                    for e in sorted(evs, key=lambda x: str(x.get("timestamp", ""))):
                        note = str(e.get("note", "") or "")
                        rows.append({**base, "changed": changed, "has_expert": has_expert,
                                     **{c: str(e.get(c, "") or "") for c in _LABEL_ONLY_COLS},
                                     "labeled": bool(e.get("reason_name")),
                                     "typical": TYPICAL_TAG in note})
            return {"columns": columns, "rows": rows, **_distinct_filters(rows)}

        # sample_view 原样 + 补齐标签列。label_source=='db' 时再合并库里的最新标注；
        # =='sample_view' 时只用 sample_view 自带的列（不读库）。
        sv_cols = [c for c in self.sample_view.columns if c != "is_input"]
        extra = [c for c in _LABEL_ONLY_COLS if c not in sv_cols]
        columns = ["index", "is_input", "changed"] + sv_cols + extra + ["tdms"]
        latest = self.latest_label_map(events) if label_source == "db" else {}
        rows = []
        for i, (_, sv) in enumerate(self.sample_view.iterrows()):
            base = self._list_base_row(i, sv)
            row = {"index": i, "is_input": base["is_input"], "tdms": base["tdms"],
                   "changed": base["sample_id"] in self.changed_ids,
                   "has_expert": base["sample_id"] in expert_ids,
                   "reference": base["reference"]}
            for c in sv_cols:
                row[c] = str(sv.get(c, "") or "")
            for c in extra:
                row[c] = ""
            lab = latest.get(base["sample_id"])
            if lab:   # 用库里的最新标注覆盖标签列（包括 sample_view 里原有的同名列）
                for c in _LABEL_ONLY_COLS:
                    row[c] = str(lab.get(c, "") or "")
            row["labeled"] = bool(str(row.get("reason_name", "")))   # 供筛选
            row["typical"] = TYPICAL_TAG in str(row.get("note", ""))
            rows.append(row)
        return {"columns": columns, "rows": rows, **_distinct_filters(rows)}

    def write_back_sample_view(self, changed_only: bool = False) -> tuple[str, int]:
        """把最新标签写回输入的 sample_view.csv（按 sample_id 更新标签相关列，保留原有列）。
        changed_only=True 时只写回本次任务改动过的样本。
        若当前任务只给了 TDMS 目录、没有 sample_view 文件，则在该目录下新建 sample_view.csv。"""
        if self.sample_view_path:
            path = Path(self.sample_view_path).expanduser()
        elif self.tdms_root:
            path = Path(self.tdms_root).expanduser() / "sample_view.csv"
            self.sample_view_path = str(path)     # 记住，后续写回/导出复用
        else:
            raise RuntimeError("当前任务既无 sample_view 文件路径也无 TDMS 目录，无法写回")
        df = self.sample_view.copy()
        latest = self.latest_label_map()
        changed = self.changed_ids if changed_only else None
        # 仅原始输入行写回（补全的行不写回 sample_view）
        for c in _LABEL_ONLY_COLS:
            if c not in df.columns:
                df[c] = ""
        n = 0
        for i in range(len(df)):
            sid = str(df.iloc[i].get("sample_id", ""))
            if changed is not None and sid not in changed:
                continue
            lab = latest.get(sid)
            if not lab:
                continue
            for c in _LABEL_ONLY_COLS:
                df.iat[i, df.columns.get_loc(c)] = str(lab.get(c, "") or "")
            n += 1
        # 不写回内部辅助列
        drop = [c for c in ("is_input",) if c in df.columns]
        df.drop(columns=drop, errors="ignore").to_csv(path, index=False, encoding="utf-8-sig")
        return str(path), n

    def snapshot(self):
        """一次性读取全部 label_events 并计算 latest，供刷新里 overview+list 复用。"""
        events = self.label_table.events_normalized()
        return events, self._latest_from_events(events)

    def _sv_label_fields(self, sv) -> dict:
        """从 sample_view 行本身读取标签字段（用于"标签来源=sample_view"）。"""
        keys = ["timestamp", "source", "result_key", "result_id", "result_name",
                "reason_key", "reason_id", "reason_name", "reason_confidence",
                "label_version", "note", "group_name", "channel_name"]
        return {k: str(sv.get(k, "") or "") for k in keys}

    def build_overview(self, latest: dict | None = None,
                       label_source: str = "db",
                       expert_ids: set[str] | None = None) -> dict:
        if latest is None or expert_ids is None:
            events = self.label_table.events_normalized()
            if latest is None:
                latest = self._latest_from_events(events)
            if expert_ids is None:
                expert_ids = self._expert_ids_from_events(events)
        rows = []
        for idx, sv in self.sample_view.iterrows():
            sid = str(sv.get("sample_id", ""))
            if label_source == "sample_view":
                lab = self._sv_label_fields(sv)
                labeled = bool(lab.get("reason_name") or lab.get("reason_key"))
            else:
                lab = latest.get(sid, {})
                labeled = bool(lab)
            note = str(lab.get("note", ""))
            status = self.status_map.get(sid, "missing")
            if status == "missing":
                tdms_label = "缺失"
            elif self.has_db:
                tdms_label = STATUS_LABEL.get(status, "未注册")   # 已注册 / 未注册
            else:
                tdms_label = "—"      # 无数据库 → 不区分注册状态
            is_input = str(sv.get("is_input", True))
            # 仅放导航/筛选/徽标会用到的精简字段（完整 16 列由 list_view 提供给列表网格），
            # 大幅减小 overview 负载，加快大数据集（上万行）的刷新。
            rows.append({
                "index": int(idx),
                "sn": str(sv.get("sn", "")),
                "sample_id": sid,
                "line": str(sv.get("line", "")),
                "is_input": is_input not in ("False", "false", "0", ""),
                "labeled": labeled,
                "changed": sid in self.changed_ids,
                "has_expert": sid in expert_ids,
                "reference": self._reference_of(sv),
                "reason_name": str(lab.get("reason_name", "")),
                "source": str(lab.get("source", "")),
                "typical": TYPICAL_TAG in note,
                "tdms": tdms_label,
            })
        total = len(rows)
        labeled = sum(1 for r in rows if r["labeled"])
        typical = sum(1 for r in rows if r["typical"])
        st_counts = {"registered": 0, "unregistered": 0, "missing": 0}
        for v in self.status_map.values():
            st_counts[v] = st_counts.get(v, 0) + 1
        return {
            "rows": rows,
            "lines": sorted({r["line"] for r in rows if r["line"]}),
            "stats": {"total": total, "labeled": labeled,
                      "unlabeled": total - labeled, "typical": typical,
                      "unresolved": st_counts["missing"],
                      "registered": st_counts["registered"] if self.has_db else 0,
                      "unregistered": st_counts["unregistered"] if self.has_db else 0},
            "is_mock": self.is_mock,
            "has_db": self.has_db,
            "tdms_root": self.tdms_root,
        }


def _distinct_filters(rows: list[dict]) -> dict:
    """从列表行里取 reason / reference 的去重取值，供前端多选筛选下拉（后端算更可靠）。"""
    reasons, refs = set(), set()
    for r in rows:
        v = str(r.get("reason_name", "") or "")
        if v:
            reasons.add(v)
        rf = str(r.get("reference", "") or "")
        if rf:
            refs.add(rf)
    return {"reason_values": sorted(reasons), "reference_values": sorted(refs)}


# ---------- 全局会话 ----------
SESSION: LabelSession | None = None


def _resolve_db(db_input: str | None) -> tuple[Path | None, str | None]:
    """把"数据库文件夹/文件"输入解析成 (现成 db 文件 或 None, 数据库文件夹 或 None)。

    支持传入文件夹：自动在 文件夹 与 文件夹/metadata 下找 label_records.db。
    找不到现成 db 返回 (None, 文件夹)——manifest 仍会在该文件夹里找 tdms_manifest.csv。
    """
    if not db_input:
        return None, None
    p = Path(db_input).expanduser()
    if p.is_file():
        resolved = resolve_database_path(p)
        return resolved, str(resolved.parent)
    if p.is_dir():
        for cand in (p / "label_records.db", p / "metadata" / "label_records.db"):
            resolved = resolve_database_path(cand)
            if resolved.is_file():
                return resolved, str(p)
        return None, str(p)        # 文件夹存在但无 db
    # 路径不存在：若像 .db 文件名则当作待创建文件，否则当文件夹
    if p.suffix == ".db":
        resolved = resolve_database_path(p)
        return resolved, str(resolved.parent)
    return None, str(p)


def _app_data_dir() -> Path:
    """应用数据目录（标注库 / 任务配置 / 缓存）。
    打包成 .app/.exe 后程序目录只读，故优先用环境变量 FORVIA_DATA_HOME 指向可写的用户目录
    （由启动器设置），否则用源码目录下的 forvia_label_v2/_data。"""
    import os as _os
    home = _os.environ.get("FORVIA_DATA_HOME")
    if home:
        d = Path(home).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        return d
    return Path(__file__).resolve().parent.parent / "_data"


def _workspace_dir() -> Path:
    """持久工作空间根目录（标签库放这里，关机/重启后仍在；不用会被清理的临时目录）。"""
    d = _app_data_dir() / "workspace"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _task_workspace_dir(tdms_root: str | None) -> Path:
    """按 TDMS 目录确定的任务工作空间子目录（确定性 → 重启后定位到同一份标签库）。"""
    key = hashlib.md5(str(tdms_root or "default").encode("utf-8")).hexdigest()[:12]
    d = _workspace_dir() / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _changed_state_path(tdms_root, sample_view_path, db_path) -> Path:
    """本任务"被人工改动过"集合的持久文件（按三条数据源路径确定，跨重启稳定）。"""
    key = hashlib.md5("|".join([str(tdms_root or ""), str(sample_view_path or ""),
                                str(db_path or "")]).encode("utf-8")).hexdigest()[:12]
    return _app_data_dir() / "changed" / f"{key}.json"


def _load_changed_ids(path: Path) -> set[str]:
    try:
        if Path(path).exists():
            return set(str(x) for x in json.loads(Path(path).read_text(encoding="utf-8")) or [])
    except Exception:
        pass
    return set()


def _writable_db_path(sample_view_path, is_mock, tdms_root=None) -> Path:
    """无现成数据库时的可写标签库位置（SQLite label_records.db），保证持久化。"""
    if is_mock:
        return Path(tempfile.gettempdir()) / "forvia_v2_mock_label_records.db"
    if sample_view_path:
        return Path(sample_view_path).expanduser().parent / "label_records.db"
    # 无 sample_view：落到按 TDMS 目录确定的持久工作空间子目录（不污染数据目录）
    return _task_workspace_dir(tdms_root) / "label_records.db"


def init_session(sample_view_path: str | None = None,
                 tdms_root: str | None = None,
                 label_records_db_path: str | None = None,
                 source: str = "expert") -> LabelSession:
    global SESSION
    # 数据库文件夹/文件 → 现成 db 文件 + 数据库文件夹（用于 manifest 查找 / 无 sample_view 时生成）
    db_file, db_folder = _resolve_db(label_records_db_path)

    # sample_view 生成规则（由"第二步是否勾选 sample_view"决定）：
    #  · 勾选（给了 sample_view 路径）→ 按 sample_view 生成；
    #  · 不勾选 → 根据 TDMS 目录(文件夹)扫描生成 sample_view（数据库只用于标签/manifest）；
    #  · 扫描不到再退回按数据库生成；都没有则演示数据。
    sv = None
    is_mock = False
    csv_sv = _read_csv(sample_view_path)
    from_sample_view = csv_sv is not None and not csv_sv.empty   # 第二步是否勾选了 sample_view
    if from_sample_view:                                      # 勾选 sample_view → 按它生成
        sv = csv_sv
    elif tdms_root:                                           # 不勾选 → 根据文件夹(扫描 TDMS)创立
        try:
            sv = build_sample_view_from_tdms(tdms_root)
        except Exception:
            sv = None
    if (sv is None or sv.empty) and db_file is not None:      # 兜底：扫描不到再按数据库生成
        try:
            sv = build_sample_view_from_db(db_file)
        except Exception:
            sv = None
    if sv is None or sv.empty:
        sv = mock_sample_view()                               # 仍为空 → 演示数据
        is_mock = True

    # 先把 sample_id 是列表（up/down 合并成一行）的行拆开，每个原子一行。
    if not is_mock:
        sv = _explode_list_sample_ids(sv)

    # 规整 sn / sample_id（兼容只带 sn 或只带 sample_id 的输入表）：
    #  · 有 sample_id 缺 sn → 从 sample_id 去掉结尾 _up/_down 推导 sn；
    #  · 有 sn 缺 sample_id → sample_id=sn_up（complete_samples 再补 _down）。
    # 否则按 sn 的 tdms 定位 / up-down 补齐都会失败。
    if not is_mock and ("sn" in sv.columns or "sample_id" in sv.columns):
        import re as _re
        if "sn" not in sv.columns:
            sv["sn"] = ""
        if "sample_id" not in sv.columns:
            sv["sample_id"] = ""
        need_sn = (sv["sn"].astype(str).str.strip() == "") & (sv["sample_id"].astype(str).str.strip() != "")
        if need_sn.any():
            sv.loc[need_sn, "sn"] = sv.loc[need_sn, "sample_id"].astype(str).map(
                lambda s: _re.sub(r"[_-](up|down)$", "", s, flags=_re.I))
        need_sid = (sv["sample_id"].astype(str).str.strip() == "") & (sv["sn"].astype(str).str.strip() != "")
        if need_sid.any():
            sv.loc[need_sid, "sample_id"] = sv.loc[need_sid, "sn"].astype(str) + "_up"

    # 列名别名：把 sample_view 里常见的简写列映射到标准标签列（用于筛选/展示/写入）。
    #   reason → reason_name；confidence/score → reason_confidence；result → result_name。
    if not is_mock:
        _ALIASES = {"reason_name": ["reason"],
                    "reason_confidence": ["confidence", "score"],
                    "result_name": ["result"]}
        for target, srcs in _ALIASES.items():
            if target not in sv.columns:
                sv[target] = ""
            empty = sv[target].astype(str).str.strip() == ""
            for src in srcs:
                if src in sv.columns and empty.any():
                    sv.loc[empty, target] = sv.loc[empty, src].astype(str)
                    empty = sv[target].astype(str).str.strip() == ""

    # 补齐每个 tdms 的 up/down 原子，并标记原始输入样本
    sv = complete_samples(sv)

    # 标注主存储 = 本地工作空间 forvia_label_v2/_data/workspace/<key>/label_records.db
    # （快、可离线，导出/写回都从它出）。若加载了外部数据库：首次打开把它整库拷过来作为种子（含
    # samples + 历史标签），并设为镜像——之后每次标注在本地写完，再异步“立即写入”外部数据库（双写）。
    # 不在打开的数据文件夹里落任何文件（只有导出/写回时才写 csv 到打开的文件夹）。
    if is_mock:
        store_path = Path(tempfile.gettempdir()) / "forvia_v2_mock_label_records.db"
        mirror_db_path = None
    else:
        store_path = _task_workspace_dir(tdms_root) / "label_records.db"
        mirror_db_path = str(db_file) if (db_file is not None and Path(db_file).exists()) else None
        if mirror_db_path is not None and not store_path.exists():
            try:
                shutil.copy2(mirror_db_path, store_path)   # 种子：把外部库整库拷到工作空间
            except Exception:
                pass
    seed_new = not Path(store_path).exists()

    try:
        # 工作空间新建且无外部种子时，用 sample_view 登记样本
        if seed_new and str(store_path).endswith(".db"):
            LabelDatabase(store_path, auto_export=False).replace_samples(
                sv.fillna("").to_dict("records"), all_samples=True)
        label_table = LabelTable(store_path)
        if is_mock:   # 演示模式：种子标签
            rows = mock_label_history(sv).to_dict("records")
            label_table.store._write_all(
                [{c: str(r.get(c, "") or "") for c in LABEL_HISTORY_COLUMNS} for r in rows])
    except Exception:
        # 工作空间库不可用 → 退回持久可写位置，保证不崩且重启仍在
        store_path = _writable_db_path(sample_view_path, is_mock, tdms_root)
        label_table = LabelTable(store_path)

    if mirror_db_path is not None:        # 双写：标注同步到外部数据库
        try:
            label_table.set_mirror(mirror_db_path)
        except Exception:
            pass

    # tdms 定位 + 注册状态：源数据文件夹=tdms_root；数据库文件夹里找 manifest
    locator = TdmsLocator(tdms_root, manifest=ManifestAdapter(db_folder))
    # 若 sample_view 的 line 为空（db samples 无 line），用 tdms_manifest.csv 按 sn 回填，
    # 否则按 (line,sn) 找不到 tdms → 全部“缺失”。
    try:
        man = locator.manifest
        if man is not None and "sn" in sv.columns:
            man.available()
            if "line" not in sv.columns:
                sv["line"] = ""
            need = sv["line"].astype(str).str.strip() == ""
            if need.any():
                sv.loc[need, "line"] = sv.loc[need, "sn"].astype(str).map(
                    lambda s: man.line_for(s)).fillna("")
    except Exception:
        pass
    path_map, status_map = locator.build(sv)

    # 数据库模式（无 sample_view）：数据库含全部 line（epump2/epump3…），
    # 这里把样本限定为"tdms 实际落在所选 TDMS 目录下"的那些，避免列出目录外的件。
    if db_file is not None and not from_sample_view and tdms_root and not is_mock:
        try:
            import os as _os
            root = _os.path.normpath(str(Path(tdms_root).expanduser())).rstrip("/") + "/"

            def _under_root(sid: str) -> bool:
                p = path_map.get(str(sid))
                if not p:
                    return False
                return (_os.path.normpath(str(p)) + "/").startswith(root)

            mask = sv["sample_id"].astype(str).map(_under_root)
            if mask.any():
                sv = sv[mask].reset_index(drop=True)
                path_map, status_map = locator.build(sv)   # 用过滤后的样本重建定位
        except Exception:
            pass

    # 加载了数据库时：把 sample_view 里尚未登记的样本（如扫描 TDMS 得到的新件）注册进 samples 表。
    # 这样后续写标注不会再 "Sample not found"，也让这些件在库里可见。
    if db_file is not None and not is_mock:
        try:
            srs = getattr(label_table.store, "database", None)
            if srs is not None:
                existing = {(str(r.get("line", "")), str(r.get("sn", "")), str(r.get("sample_id", "")))
                            for r in srs.list_samples(active_only=False)}
                new_rows = []
                for _, r in sv.iterrows():
                    key = (str(r.get("line", "") or ""), str(r.get("sn", "") or ""),
                           str(r.get("sample_id", "") or ""))
                    if all(key) and key not in existing:
                        new_rows.append({"line": key[0], "sn": key[1], "sample_id": key[2],
                                         "group_name": str(r.get("group_name", "") or ""),
                                         "channel_name": str(r.get("channel_name", "") or ""),
                                         "sampling_rate": r.get("sampling_rate") or None})
                if new_rows:
                    srs.upsert_samples(new_rows)
        except Exception:
            pass

    # label_records_db_path 对外语义 = 加载的外部数据库（用于 manifest 查找 / db_folder / 重新加载 /
    # 导出基准目录）；本地工作空间库另存于 label_table.path，不混用。无外部库时退回工作空间路径。
    session_db_path = (str(db_file) if (db_file is not None and Path(db_file).exists())
                       else str(store_path))
    SESSION = LabelSession(sv, label_table, path_map, status_map,
                           tdms_root, session_db_path, is_mock,
                           sample_view_path=sample_view_path or "",
                           default_source=source,
                           has_db=db_file is not None)
    SESSION.from_sample_view = from_sample_view   # 第二步勾选了 sample_view → 展示"来自 sample_view"面板
    # tdms_manifest.csv（数据库文件夹）里的 reference 列 → (line,sn) 映射，供列表 reference 筛选
    try:
        if locator.manifest:
            locator.manifest.available()    # 确保已读取 manifest
            SESSION.reference_map = dict(locator.manifest.reference_map)
            SESSION.reference_by_sn = dict(locator.manifest.reference_by_sn)
        else:
            SESSION.reference_map = {}
            SESSION.reference_by_sn = {}
    except Exception:
        SESSION.reference_map = {}
        SESSION.reference_by_sn = {}
    # 跨重启加载"被人工改动过"集合（落在任务工作空间）
    SESSION.changed_state_path = _changed_state_path(tdms_root, sample_view_path,
                                                     label_records_db_path)
    SESSION.changed_ids = _load_changed_ids(SESSION.changed_state_path)
    return SESSION


def get_session() -> LabelSession:
    global SESSION
    if SESSION is None:
        init_session()  # 兜底：用默认/mock 初始化
    return SESSION
