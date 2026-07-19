"""TDMS 定位 + 注册状态判定。

解析顺序（每条样本 line, sn）：
  ① 只按 tdms_manifest.csv 查路径/reference/缺失状态
  ② manifest 不存在或没有该 SN，即缺失
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

    - 注册集合 reg_keys：样本是否已登记。来源 tdms_manifest.csv（按 line/sn）。
    - 路径表 path_map：(line, sn) -> manifest 原始路径，仅来自 tdms_manifest.csv。
      绝对路径统一由 TdmsLocator 解析，避免在这里重复拼接根目录。
    """

    def __init__(self, db_folder: str | Path | None):
        self.folder = Path(db_folder).expanduser() if db_folder else None
        self._loaded = False
        self.reg_keys: set[tuple] = set()
        self.sn_keys: set[str] = set()                        # 仅按 sn 的登记集合（sample_view 无 line 时兜底）
        self.path_map: dict[tuple[str, str], str] = {}
        self.path_by_sn: dict[str, str] = {}                  # sn -> tdms 路径（兜底）
        self.path_candidates: dict[tuple[str, str], list[str]] = {}
        self.path_candidates_by_sn: dict[str, list[str]] = {}
        self.storage_map: dict[tuple[str, str], str] = {}
        self.storage_by_sn: dict[str, str] = {}
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
            self._load_csv()

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
                        rp = str(row.get("relative_path", "") or "").strip()
                        storage = str(row.get("tdms_storage_root", "") or "").strip().replace("\\", "/").strip("/")
                        # 唯一规则：data_root / tdms_storage_root / relative_path。
                        # manifest 中的旧 tdms_path 不再参与定位。
                        valid_rel = bool(rp) and not Path(rp).is_absolute() and not rp.lower().startswith("data_root/")
                        if valid_rel and storage:
                            if line:
                                self.path_candidates.setdefault((line, sn), []).append(rp)
                                self.path_map[(line, sn)] = rp
                                self.storage_map[(line, sn)] = storage
                            self.path_candidates_by_sn.setdefault(sn, []).append(rp)
                            self.path_by_sn[sn] = rp
                            self.storage_by_sn[sn] = storage
                        ref = str(row.get("reference", "") or "").strip()
                        if ref:
                            if line:
                                self.reference_map[(line, sn)] = ref
                            self.reference_by_sn[sn] = ref
            except Exception:
                continue

    def available(self) -> bool:
        self._load()
        return bool(self.reg_keys)

    def is_registered(self, line: str, sn: str, sample_id: str = "") -> bool:
        del sample_id
        self._load()
        ln, s = _norm(line), _norm(sn)
        if (ln, s) in self.reg_keys:
            return True
        return s in self.sn_keys        # line 缺失时按 sn 判断

    def get(self, line: str, sn: str) -> str | None:
        self._load()
        s = _norm(sn)
        return self.path_map.get((_norm(line), s)) or self.path_by_sn.get(s)

    def candidates_for(self, line: str, sn: str) -> list[str]:
        self._load()
        s = _norm(sn)
        return self.path_candidates.get((_norm(line), s)) or self.path_candidates_by_sn.get(s, [])

    def storage_for(self, line: str, sn: str) -> str:
        self._load()
        s = _norm(sn)
        return self.storage_map.get((_norm(line), s)) or self.storage_by_sn.get(s, "")

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

    def _manifest_abs_candidates(self, reg: str) -> list[Path]:
        """按唯一规则生成路径：data_root / factory_raw / relative_path。"""
        if not self.root or not reg:
            return []
        rel = str(reg).replace("\\", "/").strip("/")
        if Path(rel).is_absolute() or rel.lower().startswith(("data_root/", "factory_raw/")):
            return []
        data_root = self._data_root()
        storage = "factory_raw"
        return [data_root / storage / rel]

    def _data_root(self) -> Path:
        if self.manifest and self.manifest.folder:
            folder = self.manifest.folder
            if folder.name.lower() == "metadata":
                return folder.parent
        root = self.root
        if root.name.lower() == "factory_raw":
            return root.parent
        if root.parent.name.lower() == "factory_raw":
            return root.parent.parent
        return root

    # ---------- 综合解析：返回 (path|None, status) ----------
    def resolve(self, row) -> tuple[Path | None, str]:
        """解析 tdms 路径。页面路径/缺失状态只以 tdms_manifest.csv 为准。"""
        line = str(row.get("line", "") or "")
        sn = str(row.get("sn", "") or "")
        manifest_available = self.manifest.available() if self.manifest else False

        # manifest 是页面路径/缺失状态的唯一口径，信任登记路径，不逐样本校验存在。
        if self.manifest is None or not manifest_available:
            return None, "missing"
        regs = self.manifest.candidates_for(line, sn)
        storage = self.manifest.storage_for(line, sn) if self.manifest else "factory_raw"
        for reg in reversed(regs):
            rel = str(reg).replace("\\", "/").strip("/")
            if Path(rel).is_absolute() or rel.lower().startswith(("data_root/", "factory_raw/")):
                continue
            cand = self._data_root() / storage / rel
            if _safe_exists(cand):
                return cand, "registered"
        reg = regs[-1] if regs else self.manifest.get(line, sn)
        if reg:
            rel = str(reg).replace("\\", "/").strip("/")
            if not Path(rel).is_absolute() and not rel.lower().startswith(("data_root/", "factory_raw/")):
                return self._data_root() / storage / rel, "registered"
        return None, "missing"

    def explain(self, row) -> dict:
        """诊断：只列出 manifest 路径候选，便于排查"缺失"。"""
        line = str(row.get("line", "") or "")
        sn = str(row.get("sn", "") or "")
        sid = str(row.get("sample_id", "") or "")
        tried = []
        reg = self.manifest.get(line, sn) if self.manifest else None
        if reg:
            tried.append({"src": "manifest原始值", "path": reg, "exists": _safe_exists(reg)})
            for p in self._manifest_abs_candidates(reg):
                tried.append({"src": "manifest拼接", "path": str(p), "exists": _safe_exists(p)})
        return {
            "line": line, "sn": sn, "registered": self.manifest.is_registered(line, sn, sid) if self.manifest else False,
            "find_tdms_line_dirs": [],
            "find_tdms_result": "",
            "tried": tried,
        }

    def build(self, sample_view) -> tuple[dict[str, Path], dict[str, str]]:
        """返回 path_map[sample_id]=Path 与 status_map[sample_id]=状态。"""
        path_map: dict[str, Path] = {}
        status_map: dict[str, str] = {}
        sn_cache: dict[tuple[str, str], tuple[Path | None, str]] = {}
        for _, row in sample_view.iterrows():
            sid = str(row.get("sample_id", "")).strip()
            sn = str(row.get("sn", "")).strip()
            if not sid:
                continue
            cache_key = (_norm(str(row.get("line", "") or "")), sn)
            if cache_key in sn_cache:
                p, st = sn_cache[cache_key]
            else:
                p, st = self.resolve(row)
                sn_cache[cache_key] = (p, st)
            status_map[sid] = st
            if p is not None:
                path_map[sid] = p
        return path_map, status_map
