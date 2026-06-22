#!/usr/bin/env python3
"""Forvia CSV 标签解析与注册。

导入“已标注 CSV”到 label_records.db，并导出历史差异 sample_view.csv。

输入 CSV 至少需要列：
- sn
- up / down（result_key，建议值：ok/nok/boundary）
- up_type / down_type（reason_name，中文；ok/boundary 可空）

功能：
1) up_type/down_type 按 label_rules 做模糊匹配到 reason_key
2) 当 up/down=ok 或 boundary 且 *_type 为空时，自动补默认 reason
3) 将 up/down 两个方向写入 label_records.db
4) 与导入前历史“最新标签”比较，若不一致则写入当前目录 sample_view.csv
"""

from __future__ import annotations

import argparse
import csv
import difflib
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import sys
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_manager.label_rules import LABEL_RULES, Label
from data_manager.label_internal_registry import LabelStore
from data_manager.config import load_data_manager_config
from data_manager.label_internal_registry import build_operator_label_source, label_source_rank


# Forvia 标准表头（standard 模式）
FORVIA_STANDARD_HEADER = ("sn", "up", "up_type", "down", "down_type")

# Forvia 员工多人标注表头（employee_operator 模式）
FORVIA_EMPLOYEE_BASE_HEADER = ("sn", "device_id")
FORVIA_EMPLOYEE_LABEL_HEADER_TEMPLATE = (
    "label_{index}_operator",
    "label_{index}_up",
    "label_{index}_up_type",
    "label_{index}_down",
    "label_{index}_down_type",
)

REQUIRED_COLUMNS = set(FORVIA_STANDARD_HEADER)
SOURCE_PRIORITY = {"expert": 0, "operator": 1, "model": 2}
DEFAULT_SAMPLE_VIEW = "different_label_sample_view.csv"
DEFAULT_OPERATOR_DISAGREEMENT_VIEW = "operator_disagreement_sample_view.csv"
IMPORT_MODE_AUTO = "auto"
IMPORT_MODE_STANDARD = "standard"
IMPORT_MODE_EMPLOYEE_OPERATOR = "employee_operator"
DM_CFG = load_data_manager_config()
LABEL_RECORDS_DB_PATH = Path(DM_CFG["label_records_db_path"]).expanduser()
SEVERITY_REASON_CONFIDENCE = {
    "轻微": 0.6,
    "中等": 0.8,
    "强烈": 0.95,
}


@dataclass
class ImportRecord:
    line: str
    sn: str
    sample_id: str
    direction: str
    result_key: str
    result_name: str
    reason_key: str
    reason_name: str
    timestamp: str
    source: str
    note: str
    reason_confidence: float
    fuzzy_score: float
    matched_token: str


def _required_header_hint() -> str:
    return (
        "输入 CSV 缺少必需列，需要至少包含以下表头：\n"
        "| sn  | up              | up_type   | down | down_type |\n"
        "| :-: | --------------- | :-------: | :--: | --------- |\n"
        "| xxx | ok/nok/boundary | 震颤/秒表 |      |           |"
    )


def _employee_header_hint() -> str:
    return (
        "员工标注宽表至少需要包含以下列：\n"
        "| sn | device_id | label_1_operator | label_1_up | label_1_up_type | label_1_down | label_1_down_type |\n"
        "| :-: | :-: | :-: | :-: | :-: | :-: | :-: |\n"
        "| xxx | C2-PC3 | F4348 | nok | 中等-马达音 | ok | |"
    )


def _norm(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _norm_reason_text(value: object) -> str:
    s = unicodedata.normalize("NFKC", _norm(value)).lower()
    if not s:
        return ""
    s = re.sub(r"[\s_\-—,，。.:：;；、/\\()（）\[\]{}]+", "", s)
    return s


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
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
    raise RuntimeError(f"读取 CSV 失败: {path} | {type(last_err).__name__}: {last_err}")


def _parse_dt_or_min(text: object) -> datetime:
    s = _norm(text)
    if not s:
        return datetime.min
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.min


def _safe_iso_timestamp(raw: object) -> str:
    s = _norm(raw)
    if not s:
        return datetime.now().isoformat(timespec="seconds")

    formats = (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
    )
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).isoformat(timespec="seconds")
        except ValueError:
            continue
    return s


def _resolve_import_mode(header_keys: Iterable[str], requested_mode: object) -> str:
    mode = _norm(requested_mode).lower() or IMPORT_MODE_AUTO
    if mode in {IMPORT_MODE_STANDARD, IMPORT_MODE_EMPLOYEE_OPERATOR}:
        return mode

    normalized_keys = {_norm(key) for key in header_keys if _norm(key)}
    has_operator_group = any(re.match(r"^label_\d+_operator$", key) for key in normalized_keys)
    has_group_result = any(re.match(r"^label_\d+_(up|down)$", key) for key in normalized_keys)
    if has_operator_group and has_group_result:
        return IMPORT_MODE_EMPLOYEE_OPERATOR
    return IMPORT_MODE_STANDARD


def _collect_employee_group_indices(header_keys: Iterable[str]) -> List[int]:
    indices: set[int] = set()
    for key in header_keys:
        matched = re.match(r"^label_(\d+)_(operator|up|up_type|down|down_type|result)$", _norm(key))
        if matched:
            indices.add(int(matched.group(1)))
    if any(_norm(key) in {"up", "up_type", "down", "down_type", "result"} for key in header_keys):
        indices.add(1)
    return sorted(indices)


def _pick_employee_group_field(row: Dict[str, str], group_idx: int, field_name: str) -> str:
    candidates = [f"label_{group_idx}_{field_name}"]
    if group_idx == 1:
        fallback_map = {
            "up": ["up"],
            "up_type": ["up_type"],
            "down": ["down"],
            "down_type": ["down_type"],
            "result": ["result", "label_1_result"],
        }
        candidates.extend(fallback_map.get(field_name, []))
    for key in candidates:
        value = _norm(row.get(key))
        if value:
            return value
    return ""


def _build_employee_label_timestamp(row: Dict[str, str]) -> str:
    direct_ts = _safe_iso_timestamp(row.get("timestamp") or row.get("label_time"))
    if _norm(row.get("timestamp")) or _norm(row.get("label_time")):
        return direct_ts

    date_token = _norm(row.get("date_day") or row.get("date"))
    if not date_token:
        return direct_ts

    parsed = _parse_dt_or_min(date_token)
    if parsed == datetime.min:
        return _safe_iso_timestamp(date_token)

    hour_token = _norm(row.get("hour_block") or row.get("hour"))
    if hour_token:
        try:
            hour = int(float(hour_token))
            if 0 <= hour <= 23:
                parsed = parsed.replace(hour=hour, minute=0, second=0, microsecond=0)
        except ValueError:
            pass
    return parsed.isoformat(timespec="seconds")


def _build_reason_candidates(label_runtime: Label) -> List[Tuple[str, str, str]]:
    """
    返回 [(normalized_token, reason_key, raw_token)].
    raw_token 包含 reason name 与 alias。
    """
    out: List[Tuple[str, str, str]] = []
    seen = set()
    for reason_key, reason in label_runtime.reasons.items():
        tokens = [reason.get("name", "")]
        tokens.extend(reason.get("alias") or [])
        for token in tokens:
            token = _norm(token)
            if not token:
                continue
            norm_token = _norm_reason_text(token)
            if not norm_token:
                continue
            key = (norm_token, reason_key)
            if key in seen:
                continue
            seen.add(key)
            out.append((norm_token, reason_key, token))
    return out


def _resolve_result_key(raw: object, label_runtime: Label) -> str:
    token = _norm(raw).lower()
    if token in label_runtime.results:
        return token

    alias_map = {
        "正常": "ok",
        "ok": "ok",
        "异常": "nok",
        "故障": "nok",
        "nok": "nok",
        "边界": "boundary",
        "不确定": "boundary",
        "boundary": "boundary",
    }
    mapped = alias_map.get(_norm(raw))
    if mapped and mapped in label_runtime.results:
        return mapped
    raise ValueError(f"无法识别 result_key: {_norm(raw)}")


def _resolve_reason_key_fuzzy(
    reason_name: object,
    *,
    label_runtime: Label,
    candidates: List[Tuple[str, str, str]],
    cutoff: float,
) -> Tuple[str, str, float]:
    """
    返回 (reason_key, matched_token, score).
    """
    raw = _norm(reason_name)
    if not raw:
        raise ValueError("reason_name 为空")

    try:
        exact = label_runtime.get_reason_by_name(raw)
        return exact["key"], raw, 1.0
    except KeyError:
        pass

    target = _norm_reason_text(raw)
    if not target:
        raise ValueError(f"reason_name 无有效内容: {raw}")

    for token, reason_key, raw_token in candidates:
        if token == target:
            return reason_key, raw_token, 0.999

    best: Tuple[str, str, float] | None = None
    for token, reason_key, raw_token in candidates:
        ratio = difflib.SequenceMatcher(a=target, b=token).ratio()
        if target in token or token in target:
            ratio = max(ratio, 0.90)
        if best is None or ratio > best[2]:
            best = (reason_key, raw_token, ratio)

    if best is None or best[2] < cutoff:
        suggestions = difflib.get_close_matches(
            target, [c[0] for c in candidates], n=3, cutoff=max(cutoff - 0.2, 0.3)
        )
        raise ValueError(
            f"无法匹配 reason_name='{raw}'，建议={suggestions or '无'}"
        )
    return best


def _pick_default_reason_key_for_result(result_key: str, label_runtime: Label) -> str:
    preferred_map = {
        "ok": ("clean_normal",),
        "boundary": ("boundary",),
    }
    for reason_key in preferred_map.get(result_key, ()):
        reason = label_runtime.reasons.get(reason_key)
        if reason and reason.get("parent") == result_key:
            return reason_key

    candidates = [
        reason["key"]
        for reason in label_runtime.reasons.values()
        if reason.get("parent") == result_key
    ]
    if len(candidates) == 1:
        return candidates[0]
    return ""


def _resolve_reason_confidence(reason_name: object, default_confidence: float) -> float:
    raw = unicodedata.normalize("NFKC", _norm(reason_name))
    if not raw:
        return default_confidence
    for token, confidence in SEVERITY_REASON_CONFIDENCE.items():
        if token in raw:
            return confidence
    return default_confidence


def _build_latest_label_map(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    latest: Dict[str, Dict[str, str]] = {}
    latest_key: Dict[str, Tuple[int, datetime]] = {}

    for row in rows:
        sample_id = _norm(row.get("sample_id"))
        if not sample_id:
            continue

        ts = _safe_iso_timestamp(row.get("timestamp"))
        dt = datetime.min
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            pass

        source_rank = label_source_rank(row.get("source"), SOURCE_PRIORITY)
        key = (source_rank, dt)
        old_key = latest_key.get(sample_id)
        # 时间优先，时间相同再比较 source 优先级（expert > operator > model）
        if old_key is None or (key[1] > old_key[1]) or (key[1] == old_key[1] and key[0] < old_key[0]):
            latest[sample_id] = dict(row)
            latest_key[sample_id] = key
    return latest


def _build_sn_line_map_from_history(rows: Iterable[Dict[str, str]]) -> Dict[str, str]:
    sn_line: Dict[str, str] = {}
    ts_map: Dict[str, datetime] = {}
    for row in rows:
        sn = _norm(row.get("sn"))
        line = _norm(row.get("line"))
        if not sn or not line:
            continue
        ts = _parse_dt_or_min(row.get("timestamp"))
        old_ts = ts_map.get(sn)
        if old_ts is None or ts >= old_ts:
            sn_line[sn] = line
            ts_map[sn] = ts
    return sn_line


def _load_manifest_line_maps(label_history: Path) -> Tuple[Dict[str, str], Dict[str, str], str]:
    """
    从 label_history 同目录的 tdms_manifest.csv 推断：
    - sn -> line
    - reference -> line
    - 若 manifest 只有唯一 line，则返回该 line 作为全局默认
    """
    manifest_path = label_history.parent / "tdms_manifest.csv"
    if not manifest_path.exists():
        return {}, {}, ""

    try:
        rows = _read_csv_rows(manifest_path)
    except Exception:
        return {}, {}, ""

    sn_line: Dict[str, str] = {}
    ref_line: Dict[str, str] = {}
    sn_ts: Dict[str, datetime] = {}
    ref_ts: Dict[str, datetime] = {}
    all_lines = set()

    for row in rows:
        line = _norm(row.get("line"))
        if not line:
            continue
        all_lines.add(line)
        ts = _parse_dt_or_min(row.get("created_time"))

        sn = _norm(row.get("sn"))
        if sn:
            old = sn_ts.get(sn)
            if old is None or ts >= old:
                sn_line[sn] = line
                sn_ts[sn] = ts

        ref = _norm(row.get("reference"))
        if ref:
            old = ref_ts.get(ref)
            if old is None or ts >= old:
                ref_line[ref] = line
                ref_ts[ref] = ts

    one_line = next(iter(all_lines)) if len(all_lines) == 1 else ""
    return sn_line, ref_line, one_line


def _infer_line_from_input_csv_path(input_csv: Path) -> str:
    data_root = _norm(DM_CFG.get("data_root"))
    storage_root = _norm(DM_CFG.get("tdms_storage_root")).strip("/")
    if not data_root:
        return ""
    if not storage_root:
        storage_root = "factory_raw"

    try:
        tdms_root = (Path(data_root).expanduser() / storage_root).resolve()
        rel = input_csv.expanduser().resolve().relative_to(tdms_root)
    except Exception:
        return ""

    if len(rel.parts) < 2:
        return ""
    return _norm(rel.parts[0])


def _build_employee_import_records(
    rows: List[Dict[str, str]],
    *,
    group_indices: List[int],
    line_default: str,
    history_sn_line_map: Dict[str, str],
    manifest_sn_line_map: Dict[str, str],
    manifest_ref_line_map: Dict[str, str],
    fallback_line: str,
    label_runtime: Label,
    label_version: str,
    reason_confidence: float,
    fuzzy_cutoff: float,
) -> Tuple[List[ImportRecord], List[str]]:
    candidates = _build_reason_candidates(label_runtime)
    records: List[ImportRecord] = []
    errors: List[str] = []

    for idx, row in enumerate(rows, start=2):
        sn = _norm(row.get("sn"))
        if not sn:
            errors.append(f"第{idx}行 sn 为空")
            continue

        line = _norm(row.get("line")) or line_default
        if not line:
            line = history_sn_line_map.get(sn, "")
        if not line:
            line = manifest_sn_line_map.get(sn, "")
        if not line:
            reference = _norm(row.get("reference"))
            if reference:
                line = manifest_ref_line_map.get(reference, "")
        if not line:
            line = fallback_line
        if not line:
            errors.append(f"第{idx}行 line 为空（未能从 --line/历史/manifest 推断）")
            continue

        reference = _norm(row.get("reference"))
        generic_note = _norm(row.get("note"))
        timestamp = _build_employee_label_timestamp(row)

        for group_idx in group_indices:
            operator_id = _pick_employee_group_field(row, group_idx, "operator")
            group_result = _pick_employee_group_field(row, group_idx, "result")
            up_value = _pick_employee_group_field(row, group_idx, "up")
            up_reason = _pick_employee_group_field(row, group_idx, "up_type")
            down_value = _pick_employee_group_field(row, group_idx, "down")
            down_reason = _pick_employee_group_field(row, group_idx, "down_type")

            if not any((operator_id, group_result, up_value, up_reason, down_value, down_reason)):
                continue

            source = build_operator_label_source(operator_id)
            for direction, direction_value, direction_reason in (
                ("up", up_value or group_result, up_reason),
                ("down", down_value or group_result, down_reason),
            ):
                if not any((direction_value, direction_reason)):
                    continue

                try:
                    result_key = _resolve_result_key(direction_value, label_runtime)
                except ValueError as exc:
                    errors.append(f"第{idx}行 label_{group_idx}_{direction} 解析失败: {exc}")
                    continue

                record_reason_confidence = _resolve_reason_confidence(
                    direction_reason,
                    float(reason_confidence),
                )
                reason_input_norm = _norm(direction_reason)
                reason_auto_filled = False
                if not reason_input_norm:
                    reason_key = _pick_default_reason_key_for_result(result_key, label_runtime)
                    if not reason_key:
                        errors.append(f"第{idx}行 label_{group_idx}_{direction}_type 解析失败: reason_name 为空")
                        continue
                    reason_obj = label_runtime.get_reason_by_key(reason_key)
                    matched_token = reason_obj["name"]
                    score = 1.0
                    reason_auto_filled = True
                else:
                    try:
                        reason_key, matched_token, score = _resolve_reason_key_fuzzy(
                            direction_reason,
                            label_runtime=label_runtime,
                            candidates=candidates,
                            cutoff=fuzzy_cutoff,
                        )
                    except ValueError as exc:
                        errors.append(f"第{idx}行 label_{group_idx}_{direction}_type 解析失败: {exc}")
                        continue

                reason_obj = label_runtime.get_reason_by_key(reason_key)
                result_obj = label_runtime.get_result_by_key(result_key)

                note_parts = [generic_note, f"employee_group=label_{group_idx}"]
                if reference:
                    note_parts.append(f"reference={reference}")
                if group_result:
                    note_parts.append(f"group_result={group_result}")
                if reason_auto_filled:
                    note_parts.append(f"reason_auto={result_key}->{reason_obj['name']}")
                if score < 0.999:
                    note_parts.append(f"fuzzy={_norm(direction_reason)}->{matched_token}({score:.2f})")
                note = " | ".join(part for part in note_parts if part)

                records.append(
                    ImportRecord(
                        line=line,
                        sn=sn,
                        sample_id=f"{sn}_{direction}",
                        direction=direction,
                        result_key=result_obj["key"],
                        result_name=result_obj["name"],
                        reason_key=reason_obj["key"],
                        reason_name=reason_obj["name"],
                        timestamp=timestamp,
                        source=source,
                        note=note,
                        reason_confidence=record_reason_confidence,
                        fuzzy_score=score,
                        matched_token=matched_token,
                    )
                )

    return records, errors


def _build_import_records(
    rows: List[Dict[str, str]],
    *,
    line_default: str,
    history_sn_line_map: Dict[str, str],
    manifest_sn_line_map: Dict[str, str],
    manifest_ref_line_map: Dict[str, str],
    fallback_line: str,
    source: str,
    label_runtime: Label,
    label_version: str,
    reason_confidence: float,
    fuzzy_cutoff: float,
) -> Tuple[List[ImportRecord], List[str]]:
    candidates = _build_reason_candidates(label_runtime)
    records: List[ImportRecord] = []
    errors: List[str] = []

    for idx, row in enumerate(rows, start=2):
        sn = _norm(row.get("sn"))
        if not sn:
            errors.append(f"第{idx}行 sn 为空")
            continue

        line = _norm(row.get("line")) or line_default
        if not line:
            line = history_sn_line_map.get(sn, "")
        if not line:
            line = manifest_sn_line_map.get(sn, "")
        if not line:
            reference = _norm(row.get("reference"))
            if reference:
                line = manifest_ref_line_map.get(reference, "")
        if not line:
            line = fallback_line
        if not line:
            errors.append(
                f"第{idx}行 line 为空（未能从 --line/历史/manifest 推断）"
            )
            continue

        timestamp = _safe_iso_timestamp(row.get("timestamp") or row.get("label_time"))
        generic_note = _norm(row.get("note"))
        result_hint = _norm(row.get("result"))

        for direction in ("up", "down"):
            try:
                result_key = _resolve_result_key(row.get(direction), label_runtime)
            except ValueError as exc:
                errors.append(f"第{idx}行 {direction} 解析失败: {exc}")
                continue

            reason_input = row.get(f"{direction}_type")
            reason_input_norm = _norm(reason_input)
            record_reason_confidence = _resolve_reason_confidence(
                reason_input,
                float(reason_confidence),
            )
            reason_auto_filled = False
            if not reason_input_norm:
                reason_key = _pick_default_reason_key_for_result(result_key, label_runtime)
                if not reason_key:
                    errors.append(f"第{idx}行 {direction}_type 解析失败: reason_name 为空")
                    continue
                reason_obj = label_runtime.get_reason_by_key(reason_key)
                matched_token = reason_obj["name"]
                score = 1.0
                reason_auto_filled = True
            else:
                try:
                    reason_key, matched_token, score = _resolve_reason_key_fuzzy(
                        reason_input,
                        label_runtime=label_runtime,
                        candidates=candidates,
                        cutoff=fuzzy_cutoff,
                    )
                except ValueError as exc:
                    errors.append(f"第{idx}行 {direction}_type 解析失败: {exc}")
                    continue

            reason_obj = label_runtime.get_reason_by_key(reason_key)
            result_obj = label_runtime.get_result_by_key(result_key)

            direction_note = _norm(row.get(f"{direction}_note"))
            note_parts = [x for x in (direction_note, generic_note) if x]
            if result_hint:
                note_parts.append(f"result={result_hint}")
            if reason_auto_filled:
                note_parts.append(f"reason_auto={result_key}->{reason_obj['name']}")
            if score < 0.999:
                note_parts.append(f"fuzzy={_norm(reason_input)}->{matched_token}({score:.2f})")
            note = " | ".join(note_parts)

            records.append(
                ImportRecord(
                    line=line,
                    sn=sn,
                    sample_id=f"{sn}_{direction}",
                    direction=direction,
                    result_key=result_obj["key"],
                    result_name=result_obj["name"],
                    reason_key=reason_obj["key"],
                    reason_name=reason_obj["name"],
                    timestamp=timestamp,
                    source=source,
                    note=note,
                    reason_confidence=record_reason_confidence,
                    fuzzy_score=score,
                    matched_token=matched_token,
                )
            )
    return records, errors


def _pick_latest_import(records: Iterable[ImportRecord]) -> Dict[str, ImportRecord]:
    out: Dict[str, ImportRecord] = {}
    ts_cache: Dict[str, datetime] = {}
    for r in records:
        try:
            dt = datetime.fromisoformat(r.timestamp)
        except ValueError:
            dt = datetime.min
        old_dt = ts_cache.get(r.sample_id)
        if old_dt is None or dt >= old_dt:
            out[r.sample_id] = r
            ts_cache[r.sample_id] = dt
    return out


def _write_diff_sample_view(
    diff_rows: List[Dict[str, str]],
    output_path: Path,
) -> None:
    header = [
        "view_name",
        "line",
        "sn",
        "sample_id",
        "old_result_key",
        "old_reason_key",
        "old_reason_name",
        "old_reason_confidence",
        "new_result_key",
        "new_reason_key",
        "new_reason_name",
        "new_reason_confidence",
        "old_timestamp",
        "new_timestamp",
        "old_source",
        "new_source",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(diff_rows)


def _split_records_by_operator_agreement(
    records: List[ImportRecord],
) -> Tuple[List[ImportRecord], List[Dict[str, str]]]:
    """
    把同一 sample_id 上多位 operator 的标注分成两类：
    - 全部一致（result_key + reason_key 相同）-> 保留写入 label_history
    - 不一致 -> 排除，并把每位 operator 的投票收集到差异表里供复判
    仅对 source 以 ``operator_`` 开头的多 operator 组生效；
    单 operator / 单 source 记录原样保留。
    """
    grouped: Dict[Tuple[str, str, str], List[ImportRecord]] = {}
    for rec in records:
        key = (rec.line, rec.sn, rec.sample_id)
        grouped.setdefault(key, []).append(rec)

    agreed: List[ImportRecord] = []
    disagreement_rows: List[Dict[str, str]] = []

    for _, group in grouped.items():
        operator_records = [r for r in group if r.source.startswith("operator_")]
        non_operator_records = [r for r in group if not r.source.startswith("operator_")]

        # 不同 operator 数量去重，少于 2 个时无需做"是否一致"的判定
        unique_operators = {r.source for r in operator_records}
        if len(unique_operators) <= 1:
            agreed.extend(group)
            continue

        unique_decisions = {(r.result_key, r.reason_key) for r in operator_records}
        if len(unique_decisions) == 1:
            agreed.extend(group)
            continue

        # 不一致：operator 投票不写入 label_history，单独导出复判表
        for rec in operator_records:
            disagreement_rows.append(
                {
                    "view_name": "operator_disagreement",
                    "line": rec.line,
                    "sn": rec.sn,
                    "sample_id": rec.sample_id,
                    "direction": rec.direction,
                    "operator": rec.source,
                    "result_key": rec.result_key,
                    "result_name": rec.result_name,
                    "reason_key": rec.reason_key,
                    "reason_name": rec.reason_name,
                    "reason_confidence": f"{rec.reason_confidence:.3g}",
                    "timestamp": rec.timestamp,
                    "note": rec.note,
                }
            )
        # 非 operator 记录（如 expert/model）不参与一致性判定，原样保留
        agreed.extend(non_operator_records)

    # 保持稳定的输出顺序（按 sn / direction / source）
    disagreement_rows.sort(key=lambda r: (r.get("sn", ""), r.get("direction", ""), r.get("operator", "")))
    return agreed, disagreement_rows


def _write_operator_disagreement_view(
    rows: List[Dict[str, str]],
    output_path: Path,
) -> None:
    header = [
        "view_name",
        "line",
        "sn",
        "sample_id",
        "direction",
        "operator",
        "result_key",
        "result_name",
        "reason_key",
        "reason_name",
        "reason_confidence",
        "timestamp",
        "note",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导入标注 CSV 到 label_records.db，并导出历史差异 sample_view.csv",
    )
    parser.add_argument("--input-csv", required=True, help="输入标签 CSV 路径")
    parser.add_argument(
        "--label-records-db",
        default=str(LABEL_RECORDS_DB_PATH),
        help=f"label_records.db 路径，默认: {LABEL_RECORDS_DB_PATH}",
    )
    parser.add_argument(
        "--line",
        default="",
        help="默认 line（若输入 CSV 无 line 列时使用）",
    )
    parser.add_argument(
        "--mode",
        default=IMPORT_MODE_AUTO,
        choices=[IMPORT_MODE_AUTO, IMPORT_MODE_STANDARD, IMPORT_MODE_EMPLOYEE_OPERATOR],
        help="导入模式：auto/standard/employee_operator，默认 auto",
    )
    parser.add_argument("--source", default="operator", help="写入 source，默认 operator")
    parser.add_argument(
        "--reason-confidence",
        type=float,
        default=0.9,
        help="未识别轻微/中等/强烈时写入的默认 reason_confidence，默认 0.9",
    )
    parser.add_argument(
        "--fuzzy-cutoff",
        type=float,
        default=0.60,
        help="reason_name 模糊匹配阈值，默认 0.60",
    )
    parser.add_argument(
        "--sample-view-out",
        default=DEFAULT_SAMPLE_VIEW,
        help="差异 sample_view 输出路径，默认当前目录 ./sample_view.csv",
    )
    parser.add_argument(
        "--operator-disagreement-out",
        default=DEFAULT_OPERATOR_DISAGREEMENT_VIEW,
        help=(
            "员工复判模式下，多位 operator 标注不一致时的差异表输出路径，"
            f"默认当前目录 ./{DEFAULT_OPERATOR_DISAGREEMENT_VIEW}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅解析和比较，不写 label_records.db",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_csv = Path(args.input_csv).expanduser()
    label_records_db = Path(args.label_records_db).expanduser()
    sample_view_out = Path(args.sample_view_out).expanduser()
    operator_disagreement_out = Path(args.operator_disagreement_out).expanduser()

    if not input_csv.exists():
        raise FileNotFoundError(f"输入 CSV 不存在: {input_csv}")

    input_rows = _read_csv_rows(input_csv)
    if not input_rows:
        raise ValueError(f"输入 CSV 为空: {input_csv}")

    import_mode = _resolve_import_mode(input_rows[0].keys(), args.mode)

    label_runtime = Label(LABEL_RULES)
    label_version = str(LABEL_RULES.get("meta", {}).get("version", "unknown"))

    store = LabelStore(label_records_db)
    old_rows = store.list_all()
    old_latest_map = _build_latest_label_map(old_rows)
    history_sn_line_map = _build_sn_line_map_from_history(old_rows)
    manifest_sn_line_map, manifest_ref_line_map, manifest_one_line = _load_manifest_line_maps(
        label_records_db
    )

    history_lines = {_norm(r.get("line")) for r in old_rows if _norm(r.get("line"))}
    history_one_line = next(iter(history_lines)) if len(history_lines) == 1 else ""
    path_line = _infer_line_from_input_csv_path(input_csv)
    fallback_line = _norm(args.line) or path_line or history_one_line or manifest_one_line
    if path_line and not _norm(args.line):
        print(f"[INFO] 从输入 CSV 路径推断默认 line={path_line}")
    elif fallback_line and not _norm(args.line):
        print(f"[INFO] 自动推断默认 line={fallback_line}")

    if import_mode == IMPORT_MODE_EMPLOYEE_OPERATOR:
        group_indices = _collect_employee_group_indices(input_rows[0].keys())
        if not group_indices:
            raise ValueError(_employee_header_hint())
        records, parse_errors = _build_employee_import_records(
            input_rows,
            group_indices=group_indices,
            line_default=_norm(args.line),
            history_sn_line_map=history_sn_line_map,
            manifest_sn_line_map=manifest_sn_line_map,
            manifest_ref_line_map=manifest_ref_line_map,
            fallback_line=fallback_line,
            label_runtime=label_runtime,
            label_version=label_version,
            reason_confidence=float(args.reason_confidence),
            fuzzy_cutoff=float(args.fuzzy_cutoff),
        )
    else:
        missing = REQUIRED_COLUMNS - set(input_rows[0].keys())
        if missing:
            raise ValueError(f"{_required_header_hint()}\n缺少列: {sorted(missing)}")
        records, parse_errors = _build_import_records(
            input_rows,
            line_default=_norm(args.line),
            history_sn_line_map=history_sn_line_map,
            manifest_sn_line_map=manifest_sn_line_map,
            manifest_ref_line_map=manifest_ref_line_map,
            fallback_line=fallback_line,
            source=_norm(args.source) or "operator",
            label_runtime=label_runtime,
            label_version=label_version,
            reason_confidence=float(args.reason_confidence),
            fuzzy_cutoff=float(args.fuzzy_cutoff),
        )
    if parse_errors:
        print("[WARN] 存在解析失败行：")
        for err in parse_errors:
            print(f"  - {err}")

    operator_disagreement_rows: List[Dict[str, str]] = []
    if import_mode == IMPORT_MODE_EMPLOYEE_OPERATOR:
        records, operator_disagreement_rows = _split_records_by_operator_agreement(records)
        if operator_disagreement_rows:
            disagreement_sample_ids = sorted(
                {row.get("sample_id", "") for row in operator_disagreement_rows}
            )
            print(
                f"[INFO] 检测到 {len(disagreement_sample_ids)} 个 sample_id 上 operator 标注不一致，"
                f"涉及 {len(operator_disagreement_rows)} 条 operator 投票，未写入 label_history。"
            )

    _write_operator_disagreement_view(operator_disagreement_rows, operator_disagreement_out)

    if not records:
        raise RuntimeError(
            "没有可写入的记录，请检查输入数据。"
            "可通过 --line 指定产线，或确保 label_history/tdms_manifest 可推断 line。"
        )

    if not args.dry_run:
        for rec in records:
            decision_result = label_runtime.build_result(
                result_key=rec.result_key,
                reason_key=rec.reason_key,
                source=rec.source,
            )
            store.add_result(
                line=rec.line,
                sn=rec.sn,
                sample_id=rec.sample_id,
                decision_result=decision_result,
                label_version=label_version,
                reason_confidence=rec.reason_confidence,
                note=rec.note,
                timestamp=rec.timestamp,
            )

    latest_import_map = _pick_latest_import(records)
    diff_rows: List[Dict[str, str]] = []
    for sample_id, new_rec in latest_import_map.items():
        old = old_latest_map.get(sample_id)
        if not old:
            continue
        old_result = _norm(old.get("result_key"))
        old_reason = _norm(old.get("reason_key"))
        if old_result == new_rec.result_key and old_reason == new_rec.reason_key:
            continue
        diff_rows.append(
            {
                "view_name": "registered_labels_diff",
                "line": new_rec.line,
                "sn": new_rec.sn,
                "sample_id": sample_id,
                "old_result_key": old_result,
                "old_reason_key": old_reason,
                "old_reason_name": _norm(old.get("reason_name")),
                "old_reason_confidence": _norm(old.get("reason_confidence")),
                "new_result_key": new_rec.result_key,
                "new_reason_key": new_rec.reason_key,
                "new_reason_name": new_rec.reason_name,
                "new_reason_confidence": f"{new_rec.reason_confidence:.3g}",
                "old_timestamp": _norm(old.get("timestamp")),
                "new_timestamp": new_rec.timestamp,
                "old_source": _norm(old.get("source")),
                "new_source": new_rec.source,
            }
        )

    _write_diff_sample_view(diff_rows, sample_view_out)

    print(f"input_csv: {input_csv}")
    print(f"label_records_db: {label_records_db}")
    print(f"import_mode: {import_mode}")
    print(f"dry_run: {args.dry_run}")
    print(f"输入行数: {len(input_rows)}")
    print(f"成功解析记录数(up/down): {len(records)}")
    print(f"解析失败数: {len(parse_errors)}")
    print(f"历史差异样本数: {len(diff_rows)}")
    print(f"sample_view.csv: {sample_view_out}")
    if import_mode == IMPORT_MODE_EMPLOYEE_OPERATOR:
        print(f"operator 不一致投票数: {len(operator_disagreement_rows)}")
        print(f"operator_disagreement_sample_view.csv: {operator_disagreement_out}")


def register_forvia_labels(
    *, input_csv: str | Path, label_records_db: str | Path, line: str = ""
) -> dict[str, str]:
    """Forvia CSV 注册公共入口。"""
    db_path = Path(label_records_db).expanduser()
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--input-csv", str(Path(input_csv).expanduser()),
        "--label-records-db", str(db_path),
        "--sample-view-out", str(db_path.parent / "latest_label_import_diff.csv"),
        "--operator-disagreement-out", str(db_path.parent / "latest_operator_disagreement.csv"),
    ]
    if line:
        command.extend(["--line", line])
    import subprocess
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return {"log": completed.stdout[-10000:]}


if __name__ == "__main__":
    main()
