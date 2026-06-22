#!/usr/bin/env python3
"""
统计 metadata 下核心存储的数据概览与一致性。

默认统计:
- tdms_manifest.csv
- label_records.db
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from data_manager.label_database import load_label_rows


DEFAULT_DATA_ROOT = Path("/Volumes/18555440521/fault/data_root_a")
DEFAULT_TOP_N = 10
UNKNOWN_reference = "<UNKNOWN_reference>"
EMPTY_reference = "<EMPTY_reference>"
EMPTY_LINE = "<EMPTY_LINE>"
EMPTY_SN = "<EMPTY_SN>"


def _norm(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator * 100.0 / denominator


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    encodings = ("utf-8-sig", "utf-8", "gbk")
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with path.open("r", newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    return []
                return [dict(row) for row in reader]
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(f"读取 CSV 失败: {path} | {last_err}")


def _count_by(rows: Iterable[Dict[str, str]], key: str) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        counter[_norm(row.get(key)) or "<EMPTY>"] += 1
    return counter


def _print_counter(title: str, counter: Counter, top_n: int) -> None:
    print(title)
    if not counter:
        print("  (empty)")
        return
    for name, cnt in counter.most_common(top_n):
        print(f"  {name}: {cnt}")


def _count_manifest_line_reference(rows: Iterable[Dict[str, str]]) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        line = _line_value(row)
        reference = _reference_value_from_manifest_row(row)
        counter[f"{line} | {reference}"] += 1
    return counter


def _print_manifest_minimal_stats(rows: List[Dict[str, str]]) -> None:
    line_counter: Counter = Counter()
    line_reference_sn_map: Dict[str, Dict[str, Set[str]]] = {}

    for row in rows:
        line = _line_value(row)
        reference = _reference_value_from_manifest_row(row)
        sn = _sn_value(row)
        line_counter[line] += 1
        if line not in line_reference_sn_map:
            line_reference_sn_map[line] = {}
        if reference not in line_reference_sn_map[line]:
            line_reference_sn_map[line][reference] = set()
        line_reference_sn_map[line][reference].add(sn)

    print("by line:")
    if not line_counter:
        print("  (empty)")
    else:
        for line, cnt in sorted(line_counter.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {line}: {cnt}")

    print("by line x reference:")
    if not line_reference_sn_map:
        print("  (empty)")
    else:
        for line in sorted(line_reference_sn_map.keys(), key=lambda x: (-line_counter[x], x)):
            ref_items = []
            for reference, sn_set in sorted(
                line_reference_sn_map[line].items(),
                key=lambda x: (-len(x[1]), x[0]),
            ):
                sn_count = len({s for s in sn_set if s != EMPTY_SN})
                ref_items.append(f"[{reference}, {sn_count}]")
            print(f"  {line}{{{', '.join(ref_items)}}}")


def _sample_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        _norm(row.get("line")),
        _norm(row.get("sn")),
        _norm(row.get("sample_id")),
    )


def _line_value(row: Dict[str, str]) -> str:
    return _norm(row.get("line")) or EMPTY_LINE


def _sn_value(row: Dict[str, str]) -> str:
    return _norm(row.get("sn")) or EMPTY_SN


def _reference_value_from_manifest_row(row: Dict[str, str]) -> str:
    return _norm(row.get("reference")) or EMPTY_reference


def _build_line_sn_to_reference_map(
    manifest_rows: List[Dict[str, str]],
) -> Tuple[Dict[Tuple[str, str], str], int]:
    """
    用 manifest 建立 (line, sn) -> reference(reference) 映射。
    若同一 (line,sn) 对应多个 reference，取出现频次最高的 reference。
    返回: (映射字典, 冲突键数量)
    """
    bucket: Dict[Tuple[str, str], Counter] = {}
    for row in manifest_rows:
        line = _line_value(row)
        sn = _sn_value(row)
        reference = _reference_value_from_manifest_row(row)
        key = (line, sn)
        if key not in bucket:
            bucket[key] = Counter()
        bucket[key][reference] += 1

    mapping: Dict[Tuple[str, str], str] = {}
    conflict_cnt = 0
    for key, reference_counter in bucket.items():
        if not reference_counter:
            continue
        if len(reference_counter) > 1:
            conflict_cnt += 1
        mapping[key] = reference_counter.most_common(1)[0][0]
    return mapping, conflict_cnt


def _print_line_reference_stats(
    *,
    title: str,
    rows: List[Dict[str, str]],
    top_n: int,
    reference_by_line_sn: Optional[Dict[Tuple[str, str], str]] = None,
) -> None:
    """
    统计并打印：
    - 每条产线
    - 每个型号
    - 产线×型号
    """
    print(title)
    if not rows:
        print("  (empty)")
        return

    line_rows: Counter = Counter()
    line_sns: Dict[str, Set[str]] = {}
    line_references: Dict[str, Set[str]] = {}

    reference_rows: Counter = Counter()
    reference_lines: Dict[str, Set[str]] = {}
    reference_sns: Dict[str, Set[str]] = {}

    line_reference_rows: Counter = Counter()
    line_reference_sns: Dict[Tuple[str, str], Set[str]] = {}

    for row in rows:
        line = _line_value(row)
        sn = _sn_value(row)
        if reference_by_line_sn is None:
            reference = _reference_value_from_manifest_row(row)
        else:
            reference = reference_by_line_sn.get((line, sn), UNKNOWN_reference)

        line_rows[line] += 1
        line_sns.setdefault(line, set()).add(sn)
        line_references.setdefault(line, set()).add(reference)

        reference_rows[reference] += 1
        reference_lines.setdefault(reference, set()).add(line)
        reference_sns.setdefault(reference, set()).add(sn)

        lm_key = (line, reference)
        line_reference_rows[lm_key] += 1
        line_reference_sns.setdefault(lm_key, set()).add(sn)

    print("  by line:")
    for line, cnt in line_rows.most_common(top_n):
        print(
            f"    {line}: rows={cnt}, unique_sn={len(line_sns.get(line, set()))}, "
            f"unique_reference={len(line_references.get(line, set()))}"
        )

    print("  by reference:")
    for reference, cnt in reference_rows.most_common(top_n):
        print(
            f"    {reference}: rows={cnt}, unique_line={len(reference_lines.get(reference, set()))}, "
            f"unique_sn={len(reference_sns.get(reference, set()))}"
        )

    print("  by line x reference:")
    for (line, reference), cnt in line_reference_rows.most_common(top_n):
        print(
            f"    ({line}, {reference}): rows={cnt}, "
            f"unique_sn={len(line_reference_sns.get((line, reference), set()))}"
        )


def _parse_ts(ts: str) -> Optional[datetime]:
    s = _norm(ts)
    if not s:
        return None
    formats = (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M",
    )
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _history_sample_identity(row: Dict[str, str], row_idx: int) -> Tuple[str, str, str]:
    """
    以 sample 为粒度去重：
    - 优先使用 (line, sn, sample_id)
    - 若 sample_id 为空，保留原始行（用行号兜底，避免误合并）
    """
    line = _line_value(row)
    sn = _sn_value(row)
    sample_id = _norm(row.get("sample_id"))
    if sample_id:
        return (line, sn, sample_id)
    return (line, sn, f"<ROW_{row_idx}>")


def _history_sn_identity(row: Dict[str, str], row_idx: int) -> str:
    """
    以 sn 粒度去重：
    - sn 为空时不参与合并（使用行号兜底）
    """
    sn = _sn_value(row)
    if sn != EMPTY_SN:
        return sn
    return f"<ROW_{row_idx}>"


def _dedup_history_rows_by_sample(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    每个 sample 仅保留一条：
    - timestamp 可解析时，保留最新 timestamp
    - timestamp 不可解析时，保留文件中靠后的记录
    """
    latest: Dict[Tuple[str, str, str], Tuple[Optional[datetime], int, Dict[str, str]]] = {}
    for idx, row in enumerate(rows):
        key = _history_sample_identity(row, idx)
        ts = _parse_ts(_norm(row.get("timestamp")))
        old = latest.get(key)
        if old is None:
            latest[key] = (ts, idx, row)
            continue
        old_ts, old_idx, _old_row = old
        replace = False
        if ts is not None and old_ts is not None:
            replace = ts > old_ts or (ts == old_ts and idx > old_idx)
        elif ts is not None and old_ts is None:
            replace = True
        elif ts is None and old_ts is None:
            replace = idx > old_idx
        if replace:
            latest[key] = (ts, idx, row)
    return [item[2] for item in sorted(latest.values(), key=lambda x: x[1])]


def _dedup_history_rows_by_sn(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    每个 sn 仅保留一条：
    - timestamp 可解析时，保留最新 timestamp
    - timestamp 不可解析时，保留文件中靠后的记录
    """
    latest: Dict[str, Tuple[Optional[datetime], int, Dict[str, str]]] = {}
    for idx, row in enumerate(rows):
        key = _history_sn_identity(row, idx)
        ts = _parse_ts(_norm(row.get("timestamp")))
        old = latest.get(key)
        if old is None:
            latest[key] = (ts, idx, row)
            continue
        old_ts, old_idx, _old_row = old
        replace = False
        if ts is not None and old_ts is not None:
            replace = ts > old_ts or (ts == old_ts and idx > old_idx)
        elif ts is not None and old_ts is None:
            replace = True
        elif ts is None and old_ts is None:
            replace = idx > old_idx
        if replace:
            latest[key] = (ts, idx, row)
    return [item[2] for item in sorted(latest.values(), key=lambda x: x[1])]


def _report_manifest(
    rows: List[Dict[str, str]],
    top_n: int,
) -> Tuple[Set[str], Dict[Tuple[str, str], str]]:
    _ = top_n
    print("\n[tdms_manifest.csv]")
    _print_manifest_minimal_stats(rows)
    sn_set = {_norm(r.get("sn")) for r in rows if _norm(r.get("sn"))}
    line_sn_to_reference, _ = _build_line_sn_to_reference_map(rows)
    return sn_set, line_sn_to_reference


def _report_sample_index(
    rows: List[Dict[str, str]],
    top_n: int,
    line_sn_to_reference: Dict[Tuple[str, str], str],
) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    print("\n[label_records.db samples]")
    print(f"rows: {len(rows)}")

    if not rows:
        return set(), set()

    sn_set = {_norm(r.get("sn")) for r in rows if _norm(r.get("sn"))}
    sample_keys = {_sample_key(r) for r in rows if all(_sample_key(r))}
    print(f"unique sn: {len(sn_set)}")
    print(f"unique sample(line,sn,sample_id): {len(sample_keys)}")

    _print_counter("line distribution:", _count_by(rows, "line"), top_n)
    _print_counter("channel distribution:", _count_by(rows, "channel_name"), top_n)
    _print_line_reference_stats(
        title="line/reference stats (reference mapped from manifest by line+sn):",
        rows=rows,
        top_n=top_n,
        reference_by_line_sn=line_sn_to_reference,
    )
    return sn_set, sample_keys


def _report_label_history(
    rows: List[Dict[str, str]],
    top_n: int,
    line_sn_to_reference: Dict[Tuple[str, str], str],
) -> Set[Tuple[str, str, str]]:
    print("\n[label_records.db labels]")
    print(f"rows(raw): {len(rows)}")

    if not rows:
        return set()

    dedup_rows = _dedup_history_rows_by_sample(rows)
    print(f"rows(dedup by sample): {len(dedup_rows)}")

    sample_keys = {_sample_key(r) for r in dedup_rows if all(_sample_key(r))}
    print(f"unique sample(line,sn,sample_id): {len(sample_keys)}")

    _print_counter("source distribution:", _count_by(dedup_rows, "source"), top_n)
    _print_counter("result_key distribution:", _count_by(dedup_rows, "result_key"), top_n)
    _print_counter("reason_key distribution:", _count_by(dedup_rows, "reason_key"), top_n)
    _print_line_reference_stats(
        title="line/reference stats (reference mapped from manifest by line+sn, sample-level):",
        rows=dedup_rows,
        top_n=top_n,
        reference_by_line_sn=line_sn_to_reference,
    )

    ts_list = [_parse_ts(_norm(r.get("timestamp"))) for r in dedup_rows]
    ts_valid = [t for t in ts_list if t is not None]
    if ts_valid:
        print(f"timestamp range: {min(ts_valid).isoformat()} ~ {max(ts_valid).isoformat()}")
    else:
        print("timestamp range: (no valid timestamp parsed)")

    inconsistent_stats = _collect_inconsistent_sn_stats(
        dedup_rows,
        line_sn_to_reference,
        preview_limit=top_n,
    )
    print(f"inconsistent sn labels: {int(inconsistent_stats.get('total', 0) or 0)}")
    for item in list(inconsistent_stats.get("examples") or [])[:top_n]:
        labels = "; ".join(
            f"{str(label.get('label', '')).strip()}:{int(label.get('count', 0) or 0)}"
            for label in list(item.get("labels") or [])
        )
        print(
            "  "
            f"{item.get('line', EMPTY_LINE)} | {item.get('reference', UNKNOWN_reference)} | "
            f"{item.get('sn', EMPTY_SN)} | labels={labels}"
        )

    return sample_keys


def _report_consistency(
    manifest_sn: Set[str],
    history_keys: Set[Tuple[str, str, str]],
    top_n: int,
) -> None:
    print("\n[cross-file consistency]")
    history_sn = {sn for _, sn, _ in history_keys if sn}
    history_sn_missing_manifest = history_sn - manifest_sn
    manifest_sn_without_history = manifest_sn - history_sn

    print(f"label_history sn not in manifest: {len(history_sn_missing_manifest)}")
    if history_sn_missing_manifest:
        preview = sorted(history_sn_missing_manifest)[:top_n]
        print(f"  preview: {preview}")

    print(f"manifest sn without label_history: {len(manifest_sn_without_history)}")
    if manifest_sn_without_history:
        preview = sorted(manifest_sn_without_history)[:top_n]
        print(f"  preview: {preview}")

    print(f"label_history samples: {len(history_keys)}")


def _top_line_rows(
    counter: Counter,
    top_n: int,
    preferred_order: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    if not counter:
        return []
    limit = max(1, int(top_n))
    rows: List[Dict[str, object]] = []
    used: Set[str] = set()
    for line in preferred_order or []:
        if line in counter and line not in used:
            rows.append({"line": line, "count": int(counter[line])})
            used.add(line)
    tail = sorted(
        [(line, cnt) for line, cnt in counter.items() if line not in used],
        key=lambda x: (-x[1], x[0]),
    )
    rows.extend({"line": line, "count": cnt} for line, cnt in tail)
    return rows[:limit]


def _top_line_reference_rows(counter: Counter, top_n: int) -> List[Dict[str, object]]:
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))[: max(1, int(top_n))]
    return [{"line": line, "reference": reference, "count": cnt} for (line, reference), cnt in items]


def _top_simple_rows(counter: Counter, top_n: int, key_name: str) -> List[Dict[str, object]]:
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[: max(1, int(top_n))]
    return [{key_name: key, "count": cnt} for key, cnt in items]


def _extract_label_identity(row: Dict[str, str]) -> Tuple[str, str, str]:
    label_id = (
        _norm(row.get("result_id"))
        or "<EMPTY_ID>"
    )
    label_name = (
        _norm(row.get("result_name"))
        or _norm(row.get("result_key"))
        or "<EMPTY_LABEL>"
    )
    return label_id, label_name, f"{label_id}:{label_name}"


def _extract_reason_identity(row: Dict[str, str]) -> Tuple[str, str, str]:
    reason_id = (
        _norm(row.get("reason_id"))
        or "<EMPTY_REASON_ID>"
    )
    reason_name = (
        _norm(row.get("reason_name"))
        or _norm(row.get("reason_key"))
        or "<EMPTY_REASON>"
    )
    return reason_id, reason_name, f"{reason_id}:{reason_name}"


def _collect_inconsistent_sn_stats(
    rows: List[Dict[str, str]],
    line_sn_to_reference: Dict[Tuple[str, str], str],
    *,
    preferred_line_order: Optional[List[str]] = None,
    preview_limit: int = 20,
) -> Dict[str, object]:
    line_rank = {line: idx for idx, line in enumerate(preferred_line_order or [])}
    bucket: Dict[Tuple[str, str], Dict[str, object]] = {}

    for row in rows:
        line = _line_value(row)
        sn = _sn_value(row)
        reference = line_sn_to_reference.get((line, sn), UNKNOWN_reference)
        sample_id = _norm(row.get("sample_id"))
        _label_id, _label_name, label_token = _extract_label_identity(row)

        entry = bucket.setdefault(
            (line, sn),
            {
                "line": line,
                "sn": sn,
                "reference": reference,
                "label_counter": Counter(),
                "sample_ids": set(),
                "row_count": 0,
            },
        )
        label_counter = entry["label_counter"]
        if isinstance(label_counter, Counter):
            label_counter[label_token] += 1
        sample_ids = entry["sample_ids"]
        if isinstance(sample_ids, set) and sample_id:
            sample_ids.add(sample_id)
        entry["row_count"] = int(entry.get("row_count", 0) or 0) + 1

    by_line: Counter = Counter()
    by_line_reference: Counter = Counter()
    examples: List[Dict[str, object]] = []
    for entry in bucket.values():
        label_counter = entry.get("label_counter")
        if not isinstance(label_counter, Counter):
            continue
        if len(label_counter) <= 1:
            continue

        line = str(entry.get("line") or EMPTY_LINE)
        reference = str(entry.get("reference") or UNKNOWN_reference)
        by_line[line] += 1
        by_line_reference[(line, reference)] += 1
        sample_ids = entry.get("sample_ids")
        sample_id_list = sorted(sample_ids) if isinstance(sample_ids, set) else []
        examples.append(
            {
                "line": line,
                "reference": reference,
                "sn": str(entry.get("sn") or EMPTY_SN),
                "sample_count": len(sample_id_list) or int(entry.get("row_count", 0) or 0),
                "label_count": len(label_counter),
                "labels": [
                    {"label": label_token, "count": cnt}
                    for label_token, cnt in label_counter.most_common()
                ],
                "sample_ids": sample_id_list[:6],
            }
        )

    examples.sort(
        key=lambda row: (
            line_rank.get(str(row.get("line") or ""), 10**9),
            str(row.get("reference") or ""),
            str(row.get("sn") or ""),
        )
    )
    return {
        "total": len(examples),
        "by_line_counter": by_line,
        "by_line_reference_counter": by_line_reference,
        "examples": examples[: max(1, int(preview_limit))],
    }


def _is_abnormal_label_row(row: Dict[str, str]) -> bool:
    result_key = _norm(row.get("result_key")).lower()
    result_name = _norm(row.get("result_name"))
    result_id_raw = _norm(row.get("result_id"))

    if result_key in {"nok", "ng", "abnormal", "fail", "error"}:
        return True
    if result_name and any(token in result_name for token in ("异常", "故障", "NG", "NOK")):
        return True
    if result_id_raw:
        try:
            return int(float(result_id_raw)) == 1
        except Exception:
            return False
    return False


def _build_summary_payload(
    manifest_rows: List[Dict[str, str]],
    history_rows: List[Dict[str, str]],
    top_n: int,
) -> Dict[str, object]:
    line_sn_to_reference, _ = _build_line_sn_to_reference_map(manifest_rows)
    dedup_history_rows = _dedup_history_rows_by_sample(history_rows)

    tdms_by_line: Counter = Counter()
    tdms_by_line_reference: Counter = Counter()
    for row in manifest_rows:
        line = _line_value(row)
        reference = _reference_value_from_manifest_row(row)
        tdms_by_line[line] += 1
        tdms_by_line_reference[(line, reference)] += 1
    tdms_line_order = [line for line, _ in sorted(tdms_by_line.items(), key=lambda x: (-x[1], x[0]))]
    tdms_line_rank = {line: idx for idx, line in enumerate(tdms_line_order)}

    label_by_line: Counter = Counter()
    label_by_line_reference: Counter = Counter()
    label_by_label: Counter = Counter()
    label_by_label_id: Counter = Counter()
    abnormal_reason_counter: Counter = Counter()
    label_line_label_counter: Dict[str, Counter] = {}
    label_line_reference_label_counter: Dict[Tuple[str, str], Counter] = {}
    label_line_abnormal_reason_counter: Dict[str, Counter] = {}
    label_line_reference_abnormal_reason_counter: Dict[Tuple[str, str], Counter] = {}
    inconsistent_stats = _collect_inconsistent_sn_stats(
        dedup_history_rows,
        line_sn_to_reference,
        preferred_line_order=tdms_line_order,
        preview_limit=top_n,
    )
    inconsistent_by_line = inconsistent_stats.get("by_line_counter")
    if not isinstance(inconsistent_by_line, Counter):
        inconsistent_by_line = Counter()
    inconsistent_by_line_reference = inconsistent_stats.get("by_line_reference_counter")
    if not isinstance(inconsistent_by_line_reference, Counter):
        inconsistent_by_line_reference = Counter()

    for row in dedup_history_rows:
        line = _line_value(row)
        sn = _sn_value(row)
        reference = line_sn_to_reference.get((line, sn), UNKNOWN_reference)
        label_by_line[line] += 1
        label_by_line_reference[(line, reference)] += 1

        label_id, _label_name, label_token = _extract_label_identity(row)
        label_by_label[label_token] += 1
        label_by_label_id[label_id] += 1
        label_line_label_counter.setdefault(line, Counter())[label_token] += 1
        label_line_reference_label_counter.setdefault((line, reference), Counter())[label_token] += 1
        if _is_abnormal_label_row(row):
            _reason_id, _reason_name, reason_token = _extract_reason_identity(row)
            abnormal_reason_counter[reason_token] += 1
            label_line_abnormal_reason_counter.setdefault(line, Counter())[reason_token] += 1
            label_line_reference_abnormal_reason_counter.setdefault((line, reference), Counter())[reason_token] += 1

    label_line_rows: List[Dict[str, object]] = []
    for line, total in sorted(
        label_by_line.items(),
        key=lambda x: (tdms_line_rank.get(x[0], 10**9), -x[1], x[0]),
    ):
        line_label_dist = [
            {"label": label_token, "count": cnt}
            for label_token, cnt in label_line_label_counter.get(line, Counter()).most_common()
        ]
        line_abnormal_reason_dist = [
            {"reason": reason_token, "count": cnt}
            for reason_token, cnt in label_line_abnormal_reason_counter.get(line, Counter()).most_common()
        ]
        refs: List[Dict[str, object]] = []
        ref_keys = [
            (ref, cnt)
            for (k_line, ref), cnt in label_by_line_reference.items()
            if k_line == line
        ]
        ref_keys.sort(key=lambda x: (-x[1], x[0]))
        for reference, ref_count in ref_keys:
            ref_label_dist = [
                {"label": label_token, "count": cnt}
                for label_token, cnt in label_line_reference_label_counter.get(
                    (line, reference), Counter()
                ).most_common()
            ]
            ref_abnormal_reason_dist = [
                {"reason": reason_token, "count": cnt}
                for reason_token, cnt in label_line_reference_abnormal_reason_counter.get(
                    (line, reference), Counter()
                ).most_common()
            ]
            refs.append(
                {
                    "reference": reference,
                    "count": ref_count,
                    "labels": ref_label_dist,
                    "abnormal_reasons": ref_abnormal_reason_dist,
                }
            )
        label_line_rows.append(
            {
                "line": line,
                "count": total,
                "reference_count": len(refs),
                "inconsistent_sn_count": int(inconsistent_by_line.get(line, 0) or 0),
                "line_labels": line_label_dist,
                "line_abnormal_reasons": line_abnormal_reason_dist,
                "references": refs,
            }
        )

    for line_row in label_line_rows:
        refs = list(line_row.get("references") or [])
        for ref in refs:
            reference = str(ref.get("reference") or UNKNOWN_reference)
            ref["inconsistent_sn_count"] = int(
                inconsistent_by_line_reference.get((str(line_row.get("line") or EMPTY_LINE), reference), 0) or 0
            )

    return {
        "tdms": {
            "total": len(manifest_rows),
            "by_line": _top_line_rows(tdms_by_line, top_n=top_n),
            "by_line_reference": _top_line_reference_rows(tdms_by_line_reference, top_n=top_n),
        },
        "label": {
            "total": len(dedup_history_rows),
            "raw_total": len(history_rows),
            "dedup_key": "sample",
            "by_line": _top_line_rows(label_by_line, top_n=top_n, preferred_order=tdms_line_order),
            "by_line_reference": _top_line_reference_rows(label_by_line_reference, top_n=top_n),
            "by_label": _top_simple_rows(label_by_label, top_n=top_n, key_name="label"),
            "by_label_id": _top_simple_rows(label_by_label_id, top_n=top_n, key_name="label_id"),
            "by_abnormal_reason": _top_simple_rows(
                abnormal_reason_counter, top_n=top_n, key_name="reason"
            ),
            "inconsistent_sn_total": int(inconsistent_stats.get("total", 0) or 0),
            "inconsistent_sn_by_line": _top_line_rows(
                inconsistent_by_line,
                top_n=top_n,
                preferred_order=tdms_line_order,
            ),
            "inconsistent_sn_by_line_reference": _top_line_reference_rows(
                inconsistent_by_line_reference,
                top_n=top_n,
            ),
            "inconsistent_sn_examples": list(inconsistent_stats.get("examples") or []),
            "line_rows": label_line_rows,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 metadata CSV 概览与一致性")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"data_root 路径（默认: {DEFAULT_DATA_ROOT}）",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help="metadata 目录路径（默认使用 data_root/metadata）",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"分布/预览输出前 N 项（默认: {DEFAULT_TOP_N}）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出结构化 JSON（用于 Web 展示）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_dir = args.metadata_dir or (args.data_root / "metadata")
    metadata_dir = Path(metadata_dir).expanduser()

    manifest_path = metadata_dir / "tdms_manifest.csv"
    label_records_db_path = metadata_dir / "label_records.db"

    manifest_rows = _read_csv_rows(manifest_path)
    history_rows = load_label_rows(label_records_db_path)

    if args.json:
        payload = _build_summary_payload(
            manifest_rows=manifest_rows,
            history_rows=history_rows,
            top_n=max(1, int(args.top_n)),
        )
        print(json.dumps(payload, ensure_ascii=False))
        return

    print(f"metadata_dir: {metadata_dir}")
    print(f"manifest: {manifest_path} | exists={manifest_path.exists()}")
    print(f"label_records_db: {label_records_db_path} | exists={label_records_db_path.exists()}")

    manifest_sn, line_sn_to_reference = _report_manifest(manifest_rows, args.top_n)
    history_keys = _report_label_history(history_rows, args.top_n, line_sn_to_reference)
    _report_consistency(
        manifest_sn=manifest_sn,
        history_keys=history_keys,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
