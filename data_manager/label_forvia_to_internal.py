"""Convert Forvia wide label CSV files to the unified internal sample_view label CSV."""

from __future__ import annotations

import csv
import difflib
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from data_manager.config import load_data_manager_config
from data_manager.label_internal_registry import (
    INTERNAL_LABEL_CSV_COLUMNS,
    build_operator_label_source,
)
from data_manager.label_rules import LABEL_RULES, Label


FORVIA_STANDARD_HEADER = ("sn", "up", "up_type", "down", "down_type")
REQUIRED_COLUMNS = set(FORVIA_STANDARD_HEADER)
IMPORT_MODE_AUTO = "auto"
IMPORT_MODE_STANDARD = "standard"
IMPORT_MODE_EMPLOYEE_OPERATOR = "employee_operator"
SEVERITY_REASON_CONFIDENCE = {"轻微": 0.6, "中等": 0.8, "强烈": 0.95}


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
        "| sn | up | up_type | down | down_type |"
    )


def _employee_header_hint() -> str:
    return (
        "员工标注宽表至少需要包含 label_N_operator / label_N_up / "
        "label_N_up_type / label_N_down / label_N_down_type"
    )


def _norm(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _norm_reason_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", _norm(value)).lower()
    if not text:
        return ""
    return re.sub(r"[\s_\-—,，。.:：;；、/\\()（）\[\]{}]+", "", text)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            with path.open("r", newline="", encoding=encoding) as stream:
                reader = csv.DictReader(stream)
                return [dict(row) for row in reader] if reader.fieldnames else []
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"读取 CSV 失败: {path} | {type(last_error).__name__}: {last_error}")


def _parse_dt_or_min(value: object) -> datetime:
    text = _norm(value)
    if not text:
        return datetime.min
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return datetime.min


def _safe_iso_timestamp(value: object) -> str:
    text = _norm(value)
    if not text:
        return datetime.now().isoformat(timespec="seconds")
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text, fmt).isoformat(timespec="seconds")
        except ValueError:
            pass
    return text


def _resolve_import_mode(header_keys: Iterable[str], requested_mode: object) -> str:
    mode = _norm(requested_mode).lower() or IMPORT_MODE_AUTO
    if mode in {IMPORT_MODE_STANDARD, IMPORT_MODE_EMPLOYEE_OPERATOR}:
        return mode
    normalized = {_norm(key) for key in header_keys if _norm(key)}
    has_operator_group = any(re.match(r"^label_\d+_operator$", key) for key in normalized)
    has_group_result = any(re.match(r"^label_\d+_(up|down)$", key) for key in normalized)
    return IMPORT_MODE_EMPLOYEE_OPERATOR if has_operator_group and has_group_result else IMPORT_MODE_STANDARD


def _collect_employee_group_indices(header_keys: Iterable[str]) -> list[int]:
    indices: set[int] = set()
    for key in header_keys:
        matched = re.match(r"^label_(\d+)_(operator|up|up_type|down|down_type|result)$", _norm(key))
        if matched:
            indices.add(int(matched.group(1)))
    if any(_norm(key) in {"up", "up_type", "down", "down_type", "result"} for key in header_keys):
        indices.add(1)
    return sorted(indices)


def _pick_employee_group_field(row: dict[str, str], group_idx: int, field_name: str) -> str:
    candidates = [f"label_{group_idx}_{field_name}"]
    if group_idx == 1:
        candidates.extend({
            "up": ["up"],
            "up_type": ["up_type"],
            "down": ["down"],
            "down_type": ["down_type"],
            "result": ["result", "label_1_result"],
        }.get(field_name, []))
    for key in candidates:
        value = _norm(row.get(key))
        if value:
            return value
    return ""


def _build_employee_label_timestamp(row: dict[str, str]) -> str:
    direct = _safe_iso_timestamp(row.get("timestamp") or row.get("label_time"))
    if _norm(row.get("timestamp")) or _norm(row.get("label_time")):
        return direct
    date_token = _norm(row.get("date_day") or row.get("date"))
    if not date_token:
        return direct
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


def _build_reason_candidates(label_runtime: Label) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for reason_key, reason in label_runtime.reasons.items():
        tokens = [reason.get("name", "")]
        tokens.extend(reason.get("alias") or [])
        for token_raw in tokens:
            token = _norm(token_raw)
            norm_token = _norm_reason_text(token)
            key = (norm_token, reason_key)
            if not norm_token or key in seen:
                continue
            seen.add(key)
            out.append((norm_token, reason_key, token))
    return out


def _resolve_result_key(raw: object, label_runtime: Label) -> str:
    token = _norm(raw).lower()
    if token in label_runtime.results:
        return token
    mapped = {
        "正常": "ok",
        "ok": "ok",
        "异常": "nok",
        "故障": "nok",
        "nok": "nok",
        "边界": "boundary",
        "不确定": "boundary",
        "boundary": "boundary",
    }.get(_norm(raw))
    if mapped and mapped in label_runtime.results:
        return mapped
    raise ValueError(f"无法识别 result_key: {_norm(raw)}")


def _resolve_reason_key_fuzzy(
    reason_name: object,
    *,
    label_runtime: Label,
    candidates: list[tuple[str, str, str]],
    cutoff: float,
) -> tuple[str, str, float]:
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
    best: tuple[str, str, float] | None = None
    for token, reason_key, raw_token in candidates:
        ratio = difflib.SequenceMatcher(a=target, b=token).ratio()
        if target in token or token in target:
            ratio = max(ratio, 0.90)
        if best is None or ratio > best[2]:
            best = (reason_key, raw_token, ratio)
    if best is None or best[2] < cutoff:
        suggestions = difflib.get_close_matches(
            target, [candidate[0] for candidate in candidates], n=3, cutoff=max(cutoff - 0.2, 0.3)
        )
        raise ValueError(f"无法匹配 reason_name='{raw}'，建议={suggestions or '无'}")
    return best


def _pick_default_reason_key_for_result(result_key: str, label_runtime: Label) -> str:
    for reason_key in {"ok": ("clean_normal",), "boundary": ("boundary",)}.get(result_key, ()):
        reason = label_runtime.reasons.get(reason_key)
        if reason and reason.get("parent") == result_key:
            return reason_key
    candidates = [
        reason["key"]
        for reason in label_runtime.reasons.values()
        if reason.get("parent") == result_key
    ]
    return candidates[0] if len(candidates) == 1 else ""


def _resolve_reason_confidence(reason_name: object, default_confidence: float) -> float:
    raw = unicodedata.normalize("NFKC", _norm(reason_name))
    for token, confidence in SEVERITY_REASON_CONFIDENCE.items():
        if token in raw:
            return confidence
    return default_confidence


def _build_sn_line_map_from_history(rows: Iterable[dict[str, str]]) -> dict[str, str]:
    sn_line: dict[str, str] = {}
    ts_map: dict[str, datetime] = {}
    for row in rows:
        sn = _norm(row.get("sn"))
        line = _norm(row.get("line"))
        if not sn or not line:
            continue
        timestamp = _parse_dt_or_min(row.get("timestamp"))
        if sn not in ts_map or timestamp >= ts_map[sn]:
            sn_line[sn] = line
            ts_map[sn] = timestamp
    return sn_line


def _load_manifest_line_maps(label_records_db: Path) -> tuple[dict[str, str], dict[str, str], str]:
    manifest_path = label_records_db.parent / "tdms_manifest.csv"
    if not manifest_path.exists():
        return {}, {}, ""
    try:
        rows = _read_csv_rows(manifest_path)
    except Exception:
        return {}, {}, ""
    sn_line: dict[str, str] = {}
    ref_line: dict[str, str] = {}
    sn_ts: dict[str, datetime] = {}
    ref_ts: dict[str, datetime] = {}
    all_lines: set[str] = set()
    for row in rows:
        line = _norm(row.get("line"))
        if not line:
            continue
        all_lines.add(line)
        timestamp = _parse_dt_or_min(row.get("created_time"))
        sn = _norm(row.get("sn"))
        if sn and (sn not in sn_ts or timestamp >= sn_ts[sn]):
            sn_line[sn] = line
            sn_ts[sn] = timestamp
        reference = _norm(row.get("reference"))
        if reference and (reference not in ref_ts or timestamp >= ref_ts[reference]):
            ref_line[reference] = line
            ref_ts[reference] = timestamp
    return sn_line, ref_line, next(iter(all_lines)) if len(all_lines) == 1 else ""


def _infer_line_from_input_csv_path(input_csv: Path) -> str:
    cfg = load_data_manager_config()
    data_root = _norm(cfg.get("data_root"))
    storage_root = _norm(cfg.get("tdms_storage_root")).strip("/") or "factory_raw"
    if not data_root:
        return ""
    try:
        tdms_root = (Path(data_root).expanduser() / storage_root).resolve()
        rel = input_csv.expanduser().resolve().relative_to(tdms_root)
    except Exception:
        return ""
    return _norm(rel.parts[0]) if len(rel.parts) >= 2 else ""


def _build_employee_import_records(
    rows: list[dict[str, str]],
    *,
    group_indices: list[int],
    line_default: str,
    history_sn_line_map: dict[str, str],
    manifest_sn_line_map: dict[str, str],
    manifest_ref_line_map: dict[str, str],
    fallback_line: str,
    label_runtime: Label,
    label_version: str,
    reason_confidence: float,
    fuzzy_cutoff: float,
) -> tuple[list[ImportRecord], list[str]]:
    del label_version
    candidates = _build_reason_candidates(label_runtime)
    records: list[ImportRecord] = []
    errors: list[str] = []
    for idx, row in enumerate(rows, start=2):
        sn = _norm(row.get("sn"))
        if not sn:
            errors.append(f"第{idx}行 sn 为空")
            continue
        line = (
            _norm(row.get("line"))
            or line_default
            or history_sn_line_map.get(sn, "")
            or manifest_sn_line_map.get(sn, "")
            or manifest_ref_line_map.get(_norm(row.get("reference")), "")
            or fallback_line
        )
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
                record_confidence = _resolve_reason_confidence(direction_reason, float(reason_confidence))
                reason_auto = False
                if not _norm(direction_reason):
                    reason_key = _pick_default_reason_key_for_result(result_key, label_runtime)
                    if not reason_key:
                        errors.append(f"第{idx}行 label_{group_idx}_{direction}_type 解析失败: reason_name 为空")
                        continue
                    reason_obj = label_runtime.get_reason_by_key(reason_key)
                    matched_token = reason_obj["name"]
                    score = 1.0
                    reason_auto = True
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
                if reason_auto:
                    note_parts.append(f"reason_auto={result_key}->{reason_obj['name']}")
                if score < 0.999:
                    note_parts.append(f"fuzzy={_norm(direction_reason)}->{matched_token}({score:.2f})")
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
                        note=" | ".join(part for part in note_parts if part),
                        reason_confidence=record_confidence,
                        fuzzy_score=score,
                        matched_token=matched_token,
                    )
                )
    return records, errors


def _build_import_records(
    rows: list[dict[str, str]],
    *,
    line_default: str,
    history_sn_line_map: dict[str, str],
    manifest_sn_line_map: dict[str, str],
    manifest_ref_line_map: dict[str, str],
    fallback_line: str,
    source: str,
    label_runtime: Label,
    label_version: str,
    reason_confidence: float,
    fuzzy_cutoff: float,
) -> tuple[list[ImportRecord], list[str]]:
    del label_version
    candidates = _build_reason_candidates(label_runtime)
    records: list[ImportRecord] = []
    errors: list[str] = []
    for idx, row in enumerate(rows, start=2):
        sn = _norm(row.get("sn"))
        if not sn:
            errors.append(f"第{idx}行 sn 为空")
            continue
        line = (
            _norm(row.get("line"))
            or line_default
            or history_sn_line_map.get(sn, "")
            or manifest_sn_line_map.get(sn, "")
            or manifest_ref_line_map.get(_norm(row.get("reference")), "")
            or fallback_line
        )
        if not line:
            errors.append(f"第{idx}行 line 为空（未能从 --line/历史/manifest 推断）")
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
            record_confidence = _resolve_reason_confidence(reason_input, float(reason_confidence))
            reason_auto = False
            if not _norm(reason_input):
                reason_key = _pick_default_reason_key_for_result(result_key, label_runtime)
                if not reason_key:
                    errors.append(f"第{idx}行 {direction}_type 解析失败: reason_name 为空")
                    continue
                reason_obj = label_runtime.get_reason_by_key(reason_key)
                matched_token = reason_obj["name"]
                score = 1.0
                reason_auto = True
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
            note_parts = [part for part in (_norm(row.get(f"{direction}_note")), generic_note) if part]
            if result_hint:
                note_parts.append(f"result={result_hint}")
            if reason_auto:
                note_parts.append(f"reason_auto={result_key}->{reason_obj['name']}")
            if score < 0.999:
                note_parts.append(f"fuzzy={_norm(reason_input)}->{matched_token}({score:.2f})")
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
                    note=" | ".join(note_parts),
                    reason_confidence=record_confidence,
                    fuzzy_score=score,
                    matched_token=matched_token,
                )
            )
    return records, errors


def _default_output_path(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_internal.csv")


def _history_context(label_records_db: Path | None) -> tuple[dict[str, str], dict[str, str], dict[str, str], str, str]:
    if label_records_db is None or not label_records_db.exists():
        return {}, {}, {}, "", ""
    from data_manager.label_internal_registry import LabelStore

    try:
        old_rows = LabelStore(label_records_db).list_all()
    except Exception:
        old_rows = []
    history_sn_line_map = _build_sn_line_map_from_history(old_rows)
    manifest_sn_line_map, manifest_ref_line_map, manifest_one_line = _load_manifest_line_maps(label_records_db)
    history_lines = {_norm(row.get("line")) for row in old_rows if _norm(row.get("line"))}
    history_one_line = next(iter(history_lines)) if len(history_lines) == 1 else ""
    return history_sn_line_map, manifest_sn_line_map, manifest_ref_line_map, history_one_line, manifest_one_line


def _record_to_internal_row(record, *, view_name: str) -> dict[str, str]:
    return {
        "view_name": view_name,
        "line": record.line,
        "sn": record.sn,
        "sample_id": record.sample_id,
        "result_key": record.result_key,
        "result_id": "",
        "result_name": record.result_name,
        "reason_key": record.reason_key,
        "reason_id": "",
        "reason_name": record.reason_name,
        "reason_confidence": f"{record.reason_confidence:.3g}",
        "label_version": str(LABEL_RULES.get("meta", {}).get("version", "unknown")),
        "note": record.note,
        "timestamp": record.timestamp,
        "source": record.source,
    }


def convert_forvia_csv_to_internal(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    *,
    label_records_db: str | Path | None = None,
    line: str = "",
    mode: str = IMPORT_MODE_AUTO,
    source: str = "operator",
    reason_confidence: float = 0.9,
    fuzzy_cutoff: float = 0.60,
) -> dict[str, Any]:
    """Convert a Forvia standard/employee wide table into the internal label CSV.

    The converter preserves each operator vote as one label event. Training conflict
    handling remains the responsibility of the downstream label_filter rules.
    """
    input_path = Path(input_csv).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Forvia 标签 CSV 不存在: {input_path}")
    output_path = Path(output_csv).expanduser().resolve() if output_csv else _default_output_path(input_path)
    rows = _read_csv_rows(input_path)
    if not rows:
        raise ValueError(f"Forvia 标签 CSV 为空: {input_path}")

    import_mode = _resolve_import_mode(rows[0].keys(), mode)
    label_runtime = Label(LABEL_RULES)
    label_version = str(LABEL_RULES.get("meta", {}).get("version", "unknown"))
    db_path = Path(label_records_db).expanduser().resolve() if label_records_db else None
    (
        history_sn_line_map,
        manifest_sn_line_map,
        manifest_ref_line_map,
        history_one_line,
        manifest_one_line,
    ) = _history_context(db_path)
    path_line = _infer_line_from_input_csv_path(input_path)
    fallback_line = _norm(line) or path_line or history_one_line or manifest_one_line

    if import_mode == IMPORT_MODE_EMPLOYEE_OPERATOR:
        group_indices = _collect_employee_group_indices(rows[0].keys())
        if not group_indices:
            raise ValueError(_employee_header_hint())
        records, parse_errors = _build_employee_import_records(
            rows,
            group_indices=group_indices,
            line_default=_norm(line),
            history_sn_line_map=history_sn_line_map,
            manifest_sn_line_map=manifest_sn_line_map,
            manifest_ref_line_map=manifest_ref_line_map,
            fallback_line=fallback_line,
            label_runtime=label_runtime,
            label_version=label_version,
            reason_confidence=float(reason_confidence),
            fuzzy_cutoff=float(fuzzy_cutoff),
        )
    else:
        missing = REQUIRED_COLUMNS - set(rows[0].keys())
        if missing:
            raise ValueError(f"{_required_header_hint()}\n缺少列: {sorted(missing)}")
        records, parse_errors = _build_import_records(
            rows,
            line_default=_norm(line),
            history_sn_line_map=history_sn_line_map,
            manifest_sn_line_map=manifest_sn_line_map,
            manifest_ref_line_map=manifest_ref_line_map,
            fallback_line=fallback_line,
            source=_norm(source) or "operator",
            label_runtime=label_runtime,
            label_version=label_version,
            reason_confidence=float(reason_confidence),
            fuzzy_cutoff=float(fuzzy_cutoff),
        )

    internal_rows = [
        _record_to_internal_row(record, view_name=f"forvia_{import_mode}")
        for record in records
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=INTERNAL_LABEL_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(internal_rows)
    return {
        "format": import_mode,
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "input_rows": len(rows),
        "converted_rows": len(internal_rows),
        "parse_errors": parse_errors,
        "label_version": label_version,
        "fallback_line": fallback_line,
    }


def convert_forvia_csvs_to_internal(
    csv_paths: list[str | Path],
    output_dir: str | Path,
    *,
    label_records_db: str | Path | None = None,
    line: str = "",
) -> dict[str, Any]:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    outputs = []
    for path in csv_paths:
        input_path = Path(path).expanduser().resolve()
        output_path = output_root / f"{input_path.stem}_internal.csv"
        report = convert_forvia_csv_to_internal(
            input_path,
            output_path,
            label_records_db=label_records_db,
            line=line,
        )
        reports.append(report)
        outputs.append(output_path)
    return {
        "files": reports,
        "output_csvs": [str(path) for path in outputs],
        "converted_rows": sum(int(item.get("converted_rows") or 0) for item in reports),
        "parse_errors": [error for item in reports for error in (item.get("parse_errors") or [])],
    }
