"""自有 Web/内部标签注册。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from data_manager.label_database import LabelDatabase, require_database

SOURCE = ["model", "operator", "expert"]
DEFAULT_LABEL_SOURCE_PRIORITY = {"expert": 0, "operator": 1, "model": 2}
DEFAULT_LABEL_SOURCE_ALIASES = {
    "expert": "expert", "人工": "expert", "专家": "expert",
    "operator": "operator", "员工": "operator", "作业员": "operator",
    "model": "model", "ai": "model", "auto": "model", "模型": "model",
}
SOURCE_CATEGORY_PREFIXES = ("expert", "operator", "model")


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
