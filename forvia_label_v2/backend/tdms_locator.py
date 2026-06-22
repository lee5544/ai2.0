"""TDMS 定位 + 注册状态判定。

解析顺序（每条样本 line, sn）：
  ① 数据库(label_records.db / manifest) 按 (line, sn) 查登记路径 → 命中存在 = 已注册
  ② sample_view 自带 tdms_path / 源数据文件夹+relative_path 直接命中（快）
  ③ find_tdms(源数据文件夹, line, sn)：限 line 子目录、文件名含 sn
  都找不到 = 缺失；②③命中但不在登记表 = 未注册
状态：registered / unregistered / missing
"""
from __future__ import annotations

from pathlib import Path

TDMS_EXTS = (".tdms.zst", ".tdms")


def _norm(s: object) -> str:
    return str(s or "").strip().lower()


def _safe_exists(p) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False


class ManifestAdapter:
    """登记适配器。

    - 注册集合 reg_keys：样本是否已登记。来源 label_records.db 的 samples 表
      （按 line/sn/sample_id）+ tdms_manifest.csv（按 line/sn）。db 的 samples 表无路径列。
    - 路径表 path_map：(line, sn) -> tdms 绝对路径，仅来自 tdms_manifest.csv。
    """

    def __init__(self, db_folder: str | Path | None):
        self.folder = Path(db_folder).expanduser() if db_folder else None
        self._loaded = False
        self.reg_keys: set[tuple] = set()
        self.sn_keys: set[str] = set()                        # 仅按 sn 的登记集合（sample_view 无 line 时兜底）
        self.path_map: dict[tuple[str, str], str] = {}
        self.path_by_sn: dict[str, str] = {}                  # sn -> tdms 路径（兜底）
        self.reference_map: dict[tuple[str, str], str] = {}   # (line,sn) -> reference（来自 tdms_manifest.csv）
        self.reference_by_sn: dict[str, str] = {}             # sn -> reference（兜底）
        self.line_by_sn: dict[str, str] = {}                  # sn -> line（用于回填 sample_view 缺失的 line）

    def _candidate_dirs(self) -> list[Path]:
        if not self.folder:
            return []
        return [d for d in (self.folder, self.folder / "metadata") if d.exists()]

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.folder and self.folder.exists():
            self._load_sqlite()
            self._load_csv()

    def _load_sqlite(self) -> None:
        """读 samples 表 → 注册集合（line,sn,sample_id 与 line,sn）。samples 无路径列。"""
        dbs = []
        for d in self._candidate_dirs():
            dbs += list(d.glob("*.db"))
        import sqlite3
        for db in dbs:
            try:
                con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                cur = con.cursor()
                tables = [r[0] for r in cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")]
                for t in tables:
                    cols = [c[1].lower() for c in cur.execute(f"PRAGMA table_info({t})")]
                    if "line" in cols and "sn" in cols:
                        has_sid = "sample_id" in cols
                        sel = "line, sn" + (", sample_id" if has_sid else "")
                        path_col = next((c for c in ("tdms_path", "relative_path",
                                         "storage_root", "path") if c in cols), None)
                        q = f"SELECT {sel}{', ' + path_col if path_col else ''} FROM {t}"
                        for row in cur.execute(q):
                            line, sn = _norm(row[0]), _norm(row[1])
                            self.reg_keys.add((line, sn))
                            if sn:
                                self.sn_keys.add(sn)
                                if line:
                                    self.line_by_sn.setdefault(sn, line)
                            if has_sid:
                                self.reg_keys.add((line, sn, _norm(row[2])))
                            if path_col:
                                p = row[-1]
                                if p:
                                    self.path_map.setdefault((line, sn), str(p))
                                    if sn:
                                        self.path_by_sn.setdefault(sn, str(p))
                con.close()
            except Exception:
                continue

    def _load_csv(self) -> None:
        cands = []
        for d in self._candidate_dirs():
            cands += list(d.glob("*manifest*.csv"))
        import csv
        for path in cands:
            try:
                with open(path, encoding="utf-8-sig", newline="") as f:
                    for row in csv.DictReader(f):
                        line = _norm(row.get("line"))
                        sn = _norm(row.get("sn"))
                        if not sn:           # 至少要有 sn（line 可缺，按 sn 兜底）
                            continue
                        self.sn_keys.add(sn)
                        if line:
                            self.reg_keys.add((line, sn))
                            self.line_by_sn.setdefault(sn, line)
                        sr = str(row.get("storage_root", "") or "").strip()
                        rp = str(row.get("relative_path", "") or "").strip()
                        tp = str(row.get("tdms_path", "") or "").strip()
                        full = tp or (str(Path(sr) / rp) if (sr and rp) else rp)
                        if full:
                            if line:
                                self.path_map.setdefault((line, sn), full)
                            self.path_by_sn.setdefault(sn, full)
                        ref = str(row.get("reference", "") or "").strip()
                        if ref:
                            if line:
                                self.reference_map.setdefault((line, sn), ref)
                            self.reference_by_sn.setdefault(sn, ref)
            except Exception:
                continue

    def available(self) -> bool:
        self._load()
        return bool(self.reg_keys)

    def is_registered(self, line: str, sn: str, sample_id: str = "") -> bool:
        self._load()
        ln, s = _norm(line), _norm(sn)
        if sample_id and (ln, s, _norm(sample_id)) in self.reg_keys:
            return True
        if (ln, s) in self.reg_keys:
            return True
        return s in self.sn_keys        # line 缺失时按 sn 判断

    def get(self, line: str, sn: str) -> str | None:
        self._load()
        s = _norm(sn)
        return self.path_map.get((_norm(line), s)) or self.path_by_sn.get(s)

    def reference_for(self, line: str, sn: str) -> str:
        self._load()
        s = _norm(sn)
        return self.reference_map.get((_norm(line), s)) or self.reference_by_sn.get(s, "")

    def line_for(self, sn: str) -> str:
        self._load()
        return self.line_by_sn.get(_norm(sn), "")


class TdmsLocator:
    def __init__(self, source_root: str | Path | None,
                 manifest: ManifestAdapter | None = None):
        self.root = Path(source_root).expanduser() if source_root else None
        self.manifest = manifest
        self._line_index: dict[str, dict[str, Path]] = {}   # line -> {sn_token_lower: path}

    # ---------- find_tdms：限 line 子目录、文件名含 sn ----------
    def _line_dirs(self, line: str) -> list[Path]:
        if not self.root or not self.root.exists():
            return []
        ln = _norm(line)
        if not ln:
            return [self.root]
        out = []
        try:
            for d in self.root.iterdir():
                if d.is_dir():
                    dn = _norm(d.name)
                    if ln in dn or dn in ln:
                        out.append(d)
        except Exception:
            pass
        return out or [self.root]   # 没匹配到 line 子目录则退回整个 root

    def _index_line(self, line: str) -> dict[str, Path]:
        ln = _norm(line)
        if ln in self._line_index:
            return self._line_index[ln]
        idx: dict[str, Path] = {}
        count = 0
        for d in self._line_dirs(line):
            try:
                for f in d.rglob("*"):
                    if not f.is_file():
                        continue
                    nm = f.name
                    if nm.endswith(TDMS_EXTS):
                        idx[nm.lower()] = f          # 整文件名键
                        count += 1
                        if count > 200000:
                            break
            except Exception:
                continue
        self._line_index[ln] = idx
        return idx

    def find_tdms(self, line: str, sn: str) -> Path | None:
        snl = _norm(sn)
        if not snl:
            return None
        for name, path in self._index_line(line).items():
            if snl in name:
                return path
        return None

    # ---------- 直接路径（sample_view 自带 tdms_path / relative_path）----------
    def _ext_variants(self, s: str) -> list[str]:
        out = [s]
        low = s.lower()
        if low.endswith(".tdms.zst"):
            out.append(s[:-4])
        elif low.endswith(".tdms"):
            out.append(s + ".zst")
        return out

    @staticmethod
    def _overlap_join(root: Path, rel: str) -> Path:
        """把相对路径 rel（如 factory_raw/epump2/0319/x.tdms.zst）拼到 root 上，
        自动消除 root 末尾与 rel 开头的重叠部分，使其对“root=数据根/根=factory_raw/
        根=factory_raw/epump2 子目录”三种情况都能正确拼出绝对路径。"""
        rel_parts = [p for p in Path(rel.replace("\\", "/")).parts if p not in ("/", "")]
        root_parts = root.parts
        maxk = min(len(rel_parts), len(root_parts))
        best_k = 0
        for k in range(maxk, 0, -1):
            if [p.lower() for p in root_parts[-k:]] == [p.lower() for p in rel_parts[:k]]:
                best_k = k
                break
        return root.joinpath(*rel_parts[best_k:]) if rel_parts[best_k:] else root

    def _manifest_abs_candidates(self, reg: str) -> list[Path]:
        """把 manifest 路径（可能是 storage_root+relative 的相对路径）拼成绝对候选。"""
        out: list[Path] = []
        p = Path(reg).expanduser()
        if p.is_absolute():
            for c in self._ext_variants(reg):
                out.append(Path(c).expanduser())
            return out
        if self.root is not None:
            # ① 重叠合并：对 root=数据根 / factory_raw / factory_raw/epump2 子目录 都成立
            for c in self._ext_variants(reg):
                out.append(self._overlap_join(self.root, c))
            # ② 兼容旧逻辑：data_root/factory_raw/relative...
            data_root = self.root.parent
            for c in self._ext_variants(reg):
                out.append(data_root / c)
            # ③ 去掉前导 storage_root 后直接拼到 tdms_root
            name = self.root.name
            s = reg.replace("\\", "/")
            suffix = s[len(name) + 1:] if s.startswith(name + "/") else s
            for c in self._ext_variants(suffix):
                out.append(self.root / c)
        return out

    def _direct(self, row) -> Path | None:
        tp = str(row.get("tdms_path", "") or "").strip()
        for cand in (self._ext_variants(tp) if tp else []):
            p = Path(cand).expanduser()
            try:
                if p.exists():
                    return p
            except Exception:
                pass
        rp = str(row.get("relative_path", "") or "").strip()
        if self.root and rp:
            for cand in self._ext_variants(rp):
                for p in (self._overlap_join(self.root, cand), self.root / cand):
                    try:
                        if p.exists():
                            return p
                    except Exception:
                        pass
        return None

    # ---------- 综合解析：返回 (path|None, status) ----------
    def resolve(self, row) -> tuple[Path | None, str]:
        """解析 tdms 路径。为加载海量样本（上万行）时不卡顿，**不做逐样本的磁盘存在性检查**：
        manifest / sample_view 已给出路径就直接信任（拼成绝对路径返回）。真正文件是否存在，
        留到打开该样本、读取 tdms 时再判断。只有完全没有路径线索时才扫描目录。"""
        line = str(row.get("line", "") or "")
        sn = str(row.get("sn", "") or "")
        sample_id = str(row.get("sample_id", "") or "")
        registered = self.manifest.is_registered(line, sn, sample_id) if self.manifest else False
        status = "registered" if registered else "unregistered"

        # ① manifest 登记路径（信任，不校验存在）
        if self.manifest is not None:
            reg = self.manifest.get(line, sn)
            if reg:
                p = Path(reg).expanduser()
                if not p.is_absolute() and self.root is not None:
                    p = self._overlap_join(self.root, reg)
                return p, "registered"
        # ② sample_view 自带 tdms_path（绝对路径，信任）
        tp = str(row.get("tdms_path", "") or "").strip()
        if tp:
            return Path(tp).expanduser(), status
        # ③ sample_view 自带 relative_path（与 tdms_root 重叠合并）
        rp = str(row.get("relative_path", "") or "").strip()
        if rp and self.root is not None:
            return self._overlap_join(self.root, rp), status
        # ④ 没有任何路径线索 → 扫描目录（按 line 建索引，缓存；命中文件名含 sn）
        p = self.find_tdms(line, sn)
        if p is not None:
            return p, status
        return None, "missing"

    def explain(self, row) -> dict:
        """诊断：列出该样本尝试过的候选路径与是否存在，便于排查"缺失"。"""
        line = str(row.get("line", "") or "")
        sn = str(row.get("sn", "") or "")
        sid = str(row.get("sample_id", "") or "")
        tried = []
        reg = self.manifest.get(line, sn) if self.manifest else None
        if reg:
            tried.append({"src": "manifest原始值", "path": reg, "exists": _safe_exists(reg)})
            for p in self._manifest_abs_candidates(reg):
                tried.append({"src": "manifest拼接", "path": str(p), "exists": _safe_exists(p)})
        tp = str(row.get("tdms_path", "") or "").strip()
        for cand in (self._ext_variants(tp) if tp else []):
            p = Path(cand).expanduser()
            tried.append({"src": "tdms_path", "path": str(p), "exists": _safe_exists(p)})
        rp = str(row.get("relative_path", "") or "").strip()
        if self.root and rp:
            for cand in self._ext_variants(rp):
                p = self.root / cand
                tried.append({"src": "tdms_root/relative_path", "path": str(p), "exists": _safe_exists(p)})
        line_dirs = [str(d) for d in self._line_dirs(line)] if self.root else []
        ft = self.find_tdms(line, sn)
        return {
            "line": line, "sn": sn, "registered": self.manifest.is_registered(line, sn, sid) if self.manifest else False,
            "find_tdms_line_dirs": line_dirs,
            "find_tdms_result": str(ft) if ft else "",
            "tried": tried,
        }

    def build(self, sample_view) -> tuple[dict[str, Path], dict[str, str]]:
        """返回 path_map[sample_id]=Path 与 status_map[sample_id]=状态。"""
        path_map: dict[str, Path] = {}
        status_map: dict[str, str] = {}
        sn_cache: dict[str, tuple[Path | None, str]] = {}
        for _, row in sample_view.iterrows():
            sid = str(row.get("sample_id", "")).strip()
            sn = str(row.get("sn", "")).strip()
            if not sid:
                continue
            if sn in sn_cache:
                p, st = sn_cache[sn]
            else:
                p, st = self.resolve(row)
                sn_cache[sn] = (p, st)
            status_map[sid] = st
            if p is not None:
                path_map[sid] = p
        return path_map, status_map
