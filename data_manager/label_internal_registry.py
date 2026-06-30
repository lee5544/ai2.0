"""自有 Web/内部标签注册。"""

from __future__ import annotations

import re
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from data_manager.label_database import LabelDatabase, require_database

SOURCE = ["model", "operator", "expert"]
DEFAULT_LABEL_SOURCE_PRIORITY = {"expert": 0, "operator": 1, "model": 2}
DEFAULT_LABEL_SOURCE_ALIASES = {
    "expert": "expert", "人工": "expert", "专家": "expert",
    "operator": "operator", "员工": "operator", "作业员": "operator",
    "model": "model", "ai": "model", "auto": "model", "模型": "model",
}
SOURCE_CATEGORY_PREFIXES = ("expert", "operator", "model")
INTERNAL_LABEL_CSV_COLUMNS = [
    "view_name",
    "line",
    "sn",
    "sample_id",
    "result_key",
    "result_id",
    "result_name",
    "reason_key",
    "reason_id",
    "reason_name",
    "reason_confidence",
    "label_version",
    "note",
    "timestamp",
    "source",
]
LABEL_EVENT_CSV_COLUMNS = [
    "line",
    "sn",
    "sample_id",
    "timestamp",
    "source",
    "result_key",
    "result_id",
    "result_name",
    "reason_key",
    "reason_id",
    "reason_name",
    "reason_confidence",
    "label_version",
    "note",
]


def _label_to_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def norm_label_source_text(value: object) -> str:
    return _label_to_text(value)


def sanitize_label_source_part(value: object) -> str:
    text = norm_label_source_text(value)
    if not text:
        return ""
    text = re.sub(r"\s+", "", text).replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("._-")


def normalize_label_source_category(
    raw_source: object,
    aliases: Mapping[str, str] | None = None,
) -> str:
    raw = norm_label_source_text(raw_source).lower()
    if not raw:
        return ""
    alias_map = dict(DEFAULT_LABEL_SOURCE_ALIASES)
    if aliases:
        alias_map.update({str(key).strip().lower(): str(value).strip().lower() for key, value in aliases.items()})
    mapped = alias_map.get(raw, raw)
    for prefix in SOURCE_CATEGORY_PREFIXES:
        if mapped == prefix or any(mapped.startswith(f"{prefix}{separator}") for separator in ("_", "-", ":", ".")):
            return prefix
    return mapped


def label_source_rank(
    raw_source: object,
    priority: Mapping[str, int] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> int:
    category = normalize_label_source_category(raw_source, aliases=aliases)
    return int((priority or DEFAULT_LABEL_SOURCE_PRIORITY).get(category, 99))


def build_prefixed_label_source(prefix: str, *parts: object) -> str:
    prefix_token = sanitize_label_source_part(prefix) or sanitize_label_source_part(
        normalize_label_source_category(prefix)
    )
    tokens = [prefix_token or "source"]
    tokens.extend(token for part in parts if (token := sanitize_label_source_part(part)))
    return "_".join(tokens)


def build_operator_label_source(operator_id: object) -> str:
    return build_prefixed_label_source("operator", operator_id)


def _read_csv(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as stream:
                return [dict(row) for row in csv.DictReader(stream)]
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"无法识别 CSV 编码: {path}")


def _validate_internal_label_header(header: Iterable[str]) -> None:
    names = {str(name or "").strip() for name in header}
    missing = [column for column in INTERNAL_LABEL_CSV_COLUMNS if column not in names]
    if missing:
        raise ValueError(
            "内部标签 CSV 缺少必需列："
            + ", ".join(missing)
            + "；当前仅支持统一内部标签事件表。"
        )


def _has_label_value(row: Mapping[str, object]) -> bool:
    return any(
        _label_to_text(row.get(column))
        for column in ("result_key", "result_name", "reason_key", "reason_name")
    )


def _normalize_internal_label_row(row: Mapping[str, object], *, fallback_line: str = "") -> dict[str, str]:
    normalized = {column: _label_to_text(row.get(column)) for column in LABEL_EVENT_CSV_COLUMNS}
    if not normalized["line"]:
        normalized["line"] = _label_to_text(fallback_line)
    return normalized


def import_label_csv(
    db_path: str | Path,
    csv_path: str | Path,
    *,
    line: str = "",
) -> dict[str, Any]:
    """Import the unified internal label event table into label_records.db."""
    database_path = Path(db_path).expanduser().resolve()
    input_path = Path(csv_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"标签 CSV 不存在: {input_path}")

    rows = _read_csv(input_path)
    if not rows:
        return {
            "format": "internal_label_event",
            "imported_labels": 0,
            "skipped_rows": 0,
            "label_csv": str(input_path),
            "errors": [],
        }
    _validate_internal_label_header(rows[0].keys())

    database = LabelDatabase(database_path)
    imported = 0
    skipped = 0
    errors: list[str] = []
    for row_index, row in enumerate(rows, start=2):
        normalized = _normalize_internal_label_row(row, fallback_line=line)
        if not normalized["line"] or not normalized["sn"] or not normalized["sample_id"]:
            skipped += 1
            errors.append(f"第 {row_index} 行缺少 line/sn/sample_id")
            continue
        if not normalized["source"]:
            skipped += 1
            errors.append(f"第 {row_index} 行缺少 source")
            continue
        if not _has_label_value(normalized):
            skipped += 1
            continue
        try:
            database.append_label(
                line=normalized["line"],
                sn=normalized["sn"],
                sample_id=normalized["sample_id"],
                label=normalized,
            )
            imported += 1
        except Exception as exc:
            skipped += 1
            errors.append(f"第 {row_index} 行导入失败: {type(exc).__name__}: {exc}")

    return {
        "format": "internal_label_event",
        "imported_labels": imported,
        "skipped_rows": skipped,
        "label_csv": str(input_path),
        "errors": errors,
    }


def import_label_csvs(
    db_path: str | Path,
    csv_paths: Iterable[str | Path],
    *,
    line: str = "",
    progress: Callable[[int, int, Path], None] | None = None,
) -> dict[str, Any]:
    """Import multiple unified internal label CSV files in order."""
    paths = [Path(path).expanduser().resolve() for path in csv_paths]
    reports: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        if progress is not None:
            progress(index, len(paths), path)
        reports.append(import_label_csv(db_path, path, line=line))
    return {
        "files": reports,
        "csv_count": len(reports),
        "imported_labels": sum(int(item.get("imported_labels") or 0) for item in reports),
        "skipped_rows": sum(int(item.get("skipped_rows") or 0) for item in reports),
        "errors": [error for item in reports for error in (item.get("errors") or [])],
    }


class LabelStore:
    """Manage confirmed label events in label_records.db."""

    HEADER = [
        "line", "sn", "sample_id", "timestamp", "source",
        "result_key", "result_id", "result_name", "reason_key", "reason_id",
        "reason_name", "reason_confidence", "label_version", "note",
    ]

    def __init__(self, label_records_db_path: Path):
        self.label_records_db_path = require_database(label_records_db_path)
        self.database = LabelDatabase(self.label_records_db_path)
        self._rows_cache: List[Dict[str, str]] = []
        self._same_day_source_index: Dict[tuple[str, str, str], int] = {}
        self._refresh_cache()

    _norm_text = staticmethod(_label_to_text)

    @classmethod
    def _normalize_row(cls, row: Dict[str, object]) -> Dict[str, str]:
        return {column: cls._norm_text(row.get(column)) for column in cls.HEADER}

    @classmethod
    def _parse_timestamp(cls, value: object) -> datetime | None:
        text = cls._norm_text(value)
        if not text:
            return None
        for candidate in (text, text.replace("Z", "+00:00")):
            try:
                return datetime.fromisoformat(candidate)
            except ValueError:
                pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                pass
        return None

    @classmethod
    def _day_token(cls, value: object) -> str:
        text = cls._norm_text(value)
        if not text:
            return ""
        parsed = cls._parse_timestamp(text)
        if parsed is not None:
            return parsed.date().isoformat()
        matched = re.match(r"^(\d{4}[-/]\d{2}[-/]\d{2})", text)
        return matched.group(1).replace("/", "-") if matched else ""

    @classmethod
    def _same_day_source_key(cls, row: Dict[str, str]) -> tuple[str, str, str] | None:
        sample_id = cls._norm_text(row.get("sample_id"))
        source_text = cls._norm_text(row.get("source"))
        source = source_text.lower()
        day = cls._day_token(row.get("timestamp"))
        if not sample_id or not source or not day:
            return None
        if source != normalize_label_source_category(source_text):
            return None
        return sample_id, day, source

    @classmethod
    def _prefer_right_row(cls, left_row, left_idx: int, right_row, right_idx: int) -> bool:
        left_ts = cls._parse_timestamp(left_row.get("timestamp"))
        right_ts = cls._parse_timestamp(right_row.get("timestamp"))
        if left_ts is not None and right_ts is not None:
            return right_ts > left_ts if right_ts != left_ts else right_idx >= left_idx
        if right_ts is not None:
            return True
        if left_ts is not None:
            return False
        return right_idx >= left_idx

    def _rebuild_same_day_source_index(self) -> None:
        self._same_day_source_index = {}
        for index, row in enumerate(self._rows_cache):
            key = self._same_day_source_key(row)
            if key is not None:
                self._same_day_source_index[key] = index

    def _refresh_cache(self, *, force: bool = False) -> None:
        del force
        self._rows_cache = self._load_all()
        self._rebuild_same_day_source_index()

    def _write_all(self, rows: List[Dict[str, str]]) -> None:
        normalized = [self._normalize_row(row) for row in rows]
        self.database.replace_label_events(normalized)
        self._rows_cache = normalized
        self._rebuild_same_day_source_index()

    def add_result(
        self,
        *,
        line: str,
        sn: str,
        sample_id: str,
        decision_result: Dict,
        label_version: str,
        reason_confidence: Optional[float] = None,
        note: Optional[str] = "",
        timestamp: Optional[str] = None,
    ) -> None:
        result = decision_result.get("result") or {}
        reason = decision_result.get("reason") or {}
        row = self._normalize_row({
            "line": line, "sn": sn, "sample_id": sample_id,
            "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
            "source": decision_result.get("source"),
            "result_key": result.get("key"), "result_id": result.get("id"), "result_name": result.get("name"),
            "reason_key": reason.get("key"), "reason_id": reason.get("id"), "reason_name": reason.get("name"),
            "reason_confidence": reason_confidence, "label_version": label_version, "note": note,
        })
        self._refresh_cache()
        key = self._same_day_source_key(row)
        if key is not None and key in self._same_day_source_index:
            replace_idx = self._same_day_source_index[key]
            existing = self._rows_cache[replace_idx]
            if not self._prefer_right_row(existing, replace_idx, row, len(self._rows_cache)):
                return
            matching = [
                event for event in self.database.list_label_events(
                    sn=sn, sample_id=sample_id, statuses={"confirmed"},
                ) if self._same_day_source_key(self._normalize_row(event)) == key
            ]
            if not matching:
                raise RuntimeError(f"SQLite label event missing for dedup key: {key}")
            self.database.update_label_event(
                int(matching[-1]["id"]), line=line, sn=sn, sample_id=sample_id, label=row,
            )
        else:
            self.database.import_label_events([row])
        self._refresh_cache(force=True)

    def list_all(self) -> List[Dict[str, str]]:
        return self._load_all()

    def filter_by_sample(self, *, sn: str, sample_id: str) -> List[Dict[str, str]]:
        return [row for row in self._load_all() if row["sn"] == sn and row["sample_id"] == sample_id]

    def filter_by_source(self, source: str) -> List[Dict[str, str]]:
        source_text = self._norm_text(source)
        target = normalize_label_source_category(source_text)
        if target:
            return [row for row in self._load_all() if normalize_label_source_category(row.get("source")) == target]
        return [row for row in self._load_all() if self._norm_text(row.get("source")) == source_text]

    def _load_all(self) -> List[Dict[str, str]]:
        return [
            self._normalize_row(row)
            for row in self.database.list_label_events(statuses={"confirmed"})
        ]
