from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from data_manager.config import LINE_RULES_PATH
except ModuleNotFoundError:  # Standalone inference bundle
    LINE_RULES_PATH = Path(__file__).with_name("line_rules.yaml")

_TIME_ORDER_ALIASES = {
    "dmy": "dmy",
    "ddmmyyyy": "dmy",
    "day_first": "dmy",
    "ymd": "ymd",
    "yyyymmdd": "ymd",
    "year_first": "ymd",
}
_NORMALIZED_FILENAME_TIME_FMT = "%Y%m%d_%H%M%S"


def get_line_time_order(
    line: object,
    *,
    filename_rule: Mapping[str, Any] | None = None,
) -> str:
    if isinstance(filename_rule, Mapping):
        raw_order = str(filename_rule.get("time_order") or "").strip().lower()
        normalized = _TIME_ORDER_ALIASES.get(raw_order)
        if normalized:
            return normalized
    return "ymd"


def normalize_line_time_value(
    value: object,
    *,
    line: object = "",
    filename_rule: Mapping[str, Any] | None = None,
) -> str:
    raw = str(value or "").strip()
    if not raw or raw.upper().startswith("UNKNOWN"):
        return raw

    digits = re.sub(r"\D", "", raw)
    if len(digits) != 14:
        return raw

    preferred_order = get_line_time_order(line, filename_rule=filename_rule)
    for order in (preferred_order, "ymd" if preferred_order == "dmy" else "dmy"):
        compact_fmt = "%d%m%Y%H%M%S" if order == "dmy" else "%Y%m%d%H%M%S"
        try:
            parsed = datetime.strptime(digits, compact_fmt)
            return parsed.strftime(_NORMALIZED_FILENAME_TIME_FMT)
        except ValueError:
            continue
    return raw


def _load_line_rules() -> dict[str, dict[str, Any]]:
    if LINE_RULES_PATH.exists():
        with LINE_RULES_PATH.open("r", encoding="utf-8") as f:
            rules = yaml.safe_load(f) or {}
        if not isinstance(rules, dict) or not rules:
            raise ValueError(f"line_rules 配置格式错误: {LINE_RULES_PATH}")
        return rules
    raise FileNotFoundError(f"line_rules 配置不存在: {LINE_RULES_PATH}")


LINE_RULES = _load_line_rules()


def reload_line_rules() -> dict[str, dict[str, Any]]:
    """Reload the single YAML source after it is changed by the web console."""
    global LINE_RULES
    LINE_RULES = _load_line_rules()
    return LINE_RULES
