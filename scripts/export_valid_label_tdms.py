#!/usr/bin/env python3
"""导出「有效标签（专家标签 + 多个员工标注一致）」对应的 SN 表格与 tdms.zst 文件夹。

有效标签定义（沿用 copy_epump2_relabel_tdms.py 的分级）：
  - expert        专家标签
  - operator一致  多个员工(operator)对同一 sample 的标注完全一致
  仅保留以上两类，排除 operator单个 / operator不一致 / other。

reason 分组：
  - 震颤 -> {震颤}
  - 摩擦 -> {摩擦, 摩擦acc}

对每个 (line, reason) 任务：
  - 选出「至少有一个有效标签样本命中目标 reason」的 SN；
  - 生成 SN 表格 (csv)；
  - 从 data_root 复制该 SN 的 tdms.zst 到输出目录（不写入 data_root）。

输出目录默认在 data_root 之外：/Volumes/18555440521/fault/有效标签导出_<日期>
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import os
DATA_ROOT = Path(os.environ.get("FAULT_DATA_ROOT", "/Volumes/18555440521/fault/data_root"))
LABEL_DB = DATA_ROOT / "metadata" / "label_records.db"
TDMS_MANIFEST = DATA_ROOT / "metadata" / "tdms_manifest.csv"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_manager.label_database import load_label_rows  # noqa: E402

# 任务定义： (line, reason_key中文, 命中的 reason_name 集合)
REASON_GROUPS = {
    "震颤": {"震颤"},
    "摩擦": {"摩擦", "摩擦acc"},
}

TASKS = [
    ("epump2", "震颤"),
    ("epump3", "摩擦"),
    ("epump3", "震颤"),
]

GRADE_RANK = {"expert": 3, "operator一致": 2}


def norm(v: object) -> str:
    t = ("" if v is None else str(v)).strip()
    return t if t else "(missing)"


def source_kind(source: object) -> str:
    t = norm(source)
    if t == "expert":
        return "expert"
    if t.startswith("operator"):
        return "operator"
    return "other"  # expert_prototype_registry 等按 other 排除


def label_key(row: dict) -> tuple[str, str]:
    return norm(row.get("result_name")), norm(row.get("reason_name"))


def ts_key(row: dict) -> tuple[bool, str, int]:
    ts = norm(row.get("timestamp"))
    return ts != "(missing)", ts, int(row["_rownum"])


def load_rows(line: str) -> list[dict]:
    rows = []
    allrows = load_label_rows(LABEL_DB, statuses=("confirmed", "unconfirmed"))
    for i, r in enumerate(allrows, start=1):
        if norm(r.get("line")).lower() == line.lower():
            r["_rownum"] = str(i)
            rows.append(r)
    return rows


def resolve_sample_labels(rows) -> list[dict]:
    """每个 sample 选出一条代表标签并给出分级。"""
    by_sample = defaultdict(list)
    for r in rows:
        sid = norm(r.get("sample_id"))
        if sid == "(missing)":
            sid = f"__missing_{r['_rownum']}"
        by_sample[sid].append(r)

    resolved = []
    for group in by_sample.values():
        experts = [r for r in group if source_kind(r.get("source")) == "expert"]
        operators = [r for r in group if source_kind(r.get("source")) == "operator"]
        if experts:
            chosen, grade = max(experts, key=ts_key), "expert"
        elif operators:
            counts = Counter(label_key(r) for r in operators)
            if len(operators) == 1:
                chosen, grade = operators[0], "operator单个"
            elif len(counts) == 1:
                chosen, grade = max(operators, key=ts_key), "operator一致"
            else:
                grade = "operator不一致"
                mx = max(counts.values())
                cand = {k for k, v in counts.items() if v == mx}
                chosen = max([r for r in operators if label_key(r) in cand], key=ts_key)
        else:
            chosen, grade = max(group, key=ts_key), "other"
        out = dict(chosen)
        out["_grade"] = grade
        resolved.append(out)
    return resolved


def read_manifest(line: str):
    sn_to_file, sn_to_ref = {}, {}
    with TDMS_MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if norm(row.get("line")).lower() != line.lower():
                continue
            rel = norm(row.get("relative_path"))
            sr = norm(row.get("tdms_storage_root"))
            if sr == "factory_raw":
                p = DATA_ROOT / "factory_raw" / rel
            elif sr == "(missing)":
                p = DATA_ROOT / rel
            else:
                p = DATA_ROOT / sr / rel
            sn = norm(row.get("sn"))
            sn_to_file[sn] = p
            sn_to_ref[sn] = norm(row.get("reference"))
    return sn_to_file, sn_to_ref


def select_sns(resolved, reason_names: set[str]):
    """返回 {sn: 该SN命中目标reason的最佳有效标签行}。"""
    best: dict[str, dict] = {}
    for r in resolved:
        if r["_grade"] not in GRADE_RANK:
            continue
        if norm(r.get("reason_name")) not in reason_names:
            continue
        sn = norm(r.get("sn"))
        cur = best.get(sn)
        if cur is None or GRADE_RANK[r["_grade"]] > GRADE_RANK[cur["_grade"]] or (
            GRADE_RANK[r["_grade"]] == GRADE_RANK[cur["_grade"]] and ts_key(r) > ts_key(cur)
        ):
            best[sn] = r
    return best


def run_task(line, reason, out_root, execute, overwrite):
    reason_names = REASON_GROUPS[reason]
    rows = load_rows(line)
    resolved = resolve_sample_labels(rows)
    best = select_sns(resolved, reason_names)
    sn_to_file, sn_to_ref = read_manifest(line)

    task_dir = out_root / f"{line}_{reason}"
    tdms_dir = task_dir / "tdms"
    records, missing = [], []
    grade_counter = Counter()
    total_bytes = 0
    for sn in sorted(best):
        r = best[sn]
        grade_counter[r["_grade"]] += 1
        src = sn_to_file.get(sn)
        exists = bool(src and src.exists())
        size = src.stat().st_size if exists else 0
        total_bytes += size
        rec = {
            "sn": sn,
            "line": line,
            "reason": reason,
            "grade": r["_grade"],
            "reason_name": norm(r.get("reason_name")),
            "result_name": norm(r.get("result_name")),
            "reference": norm(r.get("reference")) if norm(r.get("reference")) != "(missing)"
                         else sn_to_ref.get(sn, "(missing)"),
            "sample_id": norm(r.get("sample_id")),
            "source": norm(r.get("source")),
            "timestamp": norm(r.get("timestamp")),
            "tdms_src": str(src) if src else "(not in manifest)",
            "tdms_dest": str(tdms_dir / src.name) if exists else "",
            "size_bytes": size,
        }
        records.append(rec)
        if not exists:
            missing.append({"sn": sn, "reason": "missing_in_manifest" if not src else "file_not_found",
                            "tdms_src": str(src) if src else ""})

    if execute:
        tdms_dir.mkdir(parents=True, exist_ok=True)
        for sn in sorted(best):
            src = sn_to_file.get(sn)
            if src and src.exists():
                dest = tdms_dir / src.name
                if dest.exists() and not overwrite:
                    continue
                shutil.copy2(src, dest)

    task_dir.mkdir(parents=True, exist_ok=True)
    with (task_dir / f"{line}_{reason}_SN.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else
                           ["sn", "line", "reason", "grade", "reason_name", "result_name",
                            "reference", "sample_id", "source", "timestamp",
                            "tdms_src", "tdms_dest", "size_bytes"])
        w.writeheader()
        w.writerows(records)
    if missing:
        with (task_dir / f"{line}_{reason}_missing.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sn", "reason", "tdms_src"])
            w.writeheader()
            w.writerows(missing)

    print(f"[{line} / {reason}] SN={len(best)} "
          f"expert={grade_counter.get('expert',0)} operator一致={grade_counter.get('operator一致',0)} "
          f"缺失tdms={len(missing)} 大小={total_bytes/1024**3:.2f}GiB -> {task_dir}")
    return len(best), len(missing)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="输出根目录（默认 fault/有效标签导出_<日期>）")
    ap.add_argument("--execute", action="store_true", help="实际复制 tdms.zst；不加则仅生成表格与统计")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的目标文件")
    ap.add_argument("--task", action="append", default=None,
                    help="指定任务 line:reason，可多次；默认三项全跑")
    args = ap.parse_args()

    out_root = Path(args.out) if args.out else (
        DATA_ROOT.parent / f"有效标签导出_{datetime.now():%Y%m%d}")
    tasks = TASKS
    if args.task:
        tasks = [tuple(t.split(":", 1)) for t in args.task]

    print(f"输出根目录: {out_root}")
    print(f"模式: {'执行复制' if args.execute else '仅统计(dry-run)'}\n")
    for line, reason in tasks:
        run_task(line, reason, out_root, args.execute, args.overwrite)


if __name__ == "__main__":
    main()
