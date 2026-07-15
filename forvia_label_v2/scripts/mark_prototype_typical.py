#!/usr/bin/env python3
"""把 <line>/prototype/ 下所有 .tdms.zst 标记为「典型异音」，登记进数据库，
并更新 tdms_manifest.csv 的路径指向 prototype 位置。

布局：.../factory_raw/<line>/prototype/<reason>/xxx.tdms.zst

用法（在仓库根 AI-2.0 下，用你机器上的 Python 跑）：
    python forvia_label_v2/scripts/mark_prototype_typical.py

按需改 SCAN_ROOT / DB / MANIFEST / ATOMS。逻辑：
  - 在 SCAN_ROOT 下递归找路径里含 "prototype" 目录的 *.tdms.zst；line 取 "prototype" 上一级目录名，
    sn 从文件名解析（默认按 "_" 取第 2 段）；
  - 每个 sn：复用 db 已有 sample_id；db 里没有就插入 sn_up/sn_down；最新标签 note 加 [[prototype]]（无标签则新建一条）；
  - 把 (line,sn) 在 tdms_manifest.csv 的 tdms_path 指到该 prototype 文件（清掉相对路径）。
幂等：已是典型/已指向同路径的不重复处理。
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path

# ===== 改这里 =====
SCAN_ROOT = "/Volumes/18555440521/fault/data_root/factory_raw"   # 在其下找 <line>/prototype/...
DB = "/Volumes/18555440521/fault/data_root/metadata/label_records.db"
MANIFEST = "/Volumes/18555440521/fault/data_root/metadata/tdms_manifest.csv"
SOURCE = "expert"
ATOMS = ("up", "down")          # 每个 tdms 标两路；只标一路改成 ("up",)
# ==================

TYPICAL = "[[prototype]]"

_REPO = os.environ.get("FORVIA_REPO_ROOT") or str(Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from data_manager.label_database import LabelDatabase  # noqa: E402


def parse_sn(filename: str) -> str:
    stem = filename
    for suf in (".tdms.zst", ".tdms"):
        if stem.lower().endswith(suf):
            stem = stem[: -len(suf)]
            break
    parts = stem.split("_")
    return parts[1] if len(parts) > 1 else stem


def update_manifest_paths(manifest_csv: Path, updates: dict[tuple[str, str], str]) -> int:
    if not updates:
        return 0
    norm = {(l.strip().lower(), s.strip().lower()): (l, s, p) for (l, s), p in updates.items()}
    rows, fieldnames, seen = [], None, set()
    if manifest_csv.exists():
        with open(manifest_csv, encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f); fieldnames = list(r.fieldnames or [])
            for row in r:
                key = (str(row.get("line", "")).strip().lower(), str(row.get("sn", "")).strip().lower())
                if key in norm:
                    row["tdms_path"] = norm[key][2]
                    if "relative_path" in row: row["relative_path"] = ""
                    if "storage_root" in row: row["storage_root"] = ""
                    seen.add(key)
                rows.append(row)
    if not fieldnames:
        fieldnames = ["line", "sn", "reference", "storage_root", "relative_path", "tdms_path"]
    for c in ("line", "sn", "tdms_path"):
        if c not in fieldnames:
            fieldnames.append(c)
    n = len(seen)
    for key, (l, s, p) in norm.items():
        if key not in seen:
            rows.append({"line": l, "sn": s, "tdms_path": p}); n += 1
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in fieldnames})
    return n


def line_of(f: Path) -> str:
    """从 .../<line>/prototype/<reason>/file 取 "prototype" 上一级目录名作为 line。"""
    parts = f.parts
    for i in range(len(parts) - 1, 0, -1):
        if parts[i] == "prototype" and i >= 1:
            return parts[i - 1]
    return "unknown_line"


def main() -> None:
    root = Path(SCAN_ROOT).expanduser()
    if not root.is_dir():
        print(f"✗ 目录不存在: {root}"); sys.exit(1)
    if not Path(DB).expanduser().is_file():
        print(f"✗ 数据库不存在: {DB}"); sys.exit(1)

    db = LabelDatabase(Path(DB).expanduser())
    # 先建 db 已有样本索引：sn(小写) -> [sample_id...]，复用 db 里的 sample_id，避免重复造 sn_up/down
    sn_to_ids: dict[str, list[str]] = {}
    try:
        for r in db.list_samples(active_only=False):
            sn_to_ids.setdefault(str(r.get("sn", "")).strip().lower(), []).append(str(r.get("sample_id", "")))
    except Exception:
        pass

    files = [f for f in root.rglob("*.tdms.zst") if "prototype" in f.parts]
    print(f"找到 {len(files)} 个 prototype 下的 .tdms.zst")
    added, tagged, skipped = 0, 0, 0
    manifest_updates: dict[tuple[str, str], str] = {}
    for f in sorted(files):
        line = line_of(f)
        sn = parse_sn(f.name)
        if not sn:
            continue
        manifest_updates[(line, sn)] = str(f.resolve())
        # 优先用 db 已登记的 sample_id；没有才退回 sn_up/sn_down
        existing = [s for s in sn_to_ids.get(sn.strip().lower(), []) if s]
        sids = existing if existing else [f"{sn}_{d}" for d in ATOMS]
        for sid in sids:
            db.upsert_samples([{"line": line, "sn": sn, "sample_id": sid}])
            evs = db.list_label_events(sample_id=sid, statuses={"confirmed"})
            if evs:
                ev = max(evs, key=lambda e: str(e.get("timestamp", "")))
                note = str(ev.get("note", "") or "")
                if TYPICAL in note:
                    skipped += 1; continue
                label = {k: str(ev.get(k, "") or "") for k in (
                    "timestamp", "source", "result_key", "result_id", "result_name",
                    "reason_key", "reason_id", "reason_name", "reason_confidence",
                    "label_version", "note")}
                label["note"] = (note + " " + TYPICAL).strip()
                db.update_label_event(int(ev["id"]), line=line, sn=sn, sample_id=sid, label=label)
                tagged += 1
            else:
                db.import_label_events([{
                    "line": line, "sn": sn, "sample_id": sid,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "source": SOURCE, "note": TYPICAL,
                }])
                added += 1
    n_manifest = update_manifest_paths(Path(MANIFEST).expanduser(), manifest_updates)
    print(f"完成：新建标记 {added}、追加标记 {tagged}、跳过 {skipped}；manifest 路径更新 {n_manifest} 条。")


if __name__ == "__main__":
    main()
