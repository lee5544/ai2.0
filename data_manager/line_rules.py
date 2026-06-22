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

_EMBEDDED_LINE_RULES = {
    "epump2": {
        "filename": {
            "split": "_",
            "sn_index": 1,
            "reference_index": 2,
            "time_index": [3, 4],
            "time_order": "dmy",
        },
        "channels": {
            "up_group": "Vib Up_0",
            "down_group": "Vib Down_0",
            "acc_channel": "ACC",
        },
        "note": "固定规则",
    },
    "epump3": {
        "filename": {
            "split": "_",
            "sn_index": 1,
            "reference_index": 2,
            "time_index": [3, 4],
            "time_order": "dmy",
        },
        "channels": {
            "up_group": "Vib Up_0",
            "down_group": "Vib Down_0",
            "acc_channel": "ACC",
        },
        "note": "固定规则",
    },
    "epump4": {
        "filename": {
            "split": "_",
            "sn_index": 0,
            "reference_index": 1,
            "time_index": [3, 4],
            "time_order": "ymd",
        },
        "channels": {
            "up_group": "Vib Up_0",
            "down_group": "Vib Down_0",
            "acc_channel": "ACC",
        },
        "note": "固定规则",
    },
    "etilt1": {
        "filename": {
            "split": "_",
            "sn_index": 4,
            "reference_index": 1,
            "time_index": [2, 3],
            "time_order": "ymd",
        },
        "channels": {
            "conditional": [
                {
                    "when": {"reference": "4031033"},
                    "up_group": "Measure_9",
                    "down_group": "Measure_10",
                    "acc_channel": "ACC1",
                },
                {
                    "when": {"reference": "*"},
                    "up_group": "Measure_1",
                    "down_group": "Measure_2",
                    "acc_channel": "ACC1",
                },
            ]
        },
        "note": 'reference == "4031033" -> Measure_9 / 10; else -> Measure_1 / 2',
    },
    "etilt2": {
        "filename": None,
        "channels": None,
        "note": "规则尚未定义",
    },
}
_LINE_TIME_ORDER_DEFAULTS = {
    "epump2": "dmy",
    "epump3": "dmy",
}
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
    return _LINE_TIME_ORDER_DEFAULTS.get(str(line or "").strip().lower(), "ymd")


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
    return _EMBEDDED_LINE_RULES


LINE_RULES = _load_line_rules()
