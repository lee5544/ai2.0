from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def as_config_dict(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def first_config_value(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def pick_config_value(model_cfg: Any, train_cfg: Any, *keys: str, default: Any = None) -> Any:
    model_cfg = as_config_dict(model_cfg)
    train_cfg = as_config_dict(train_cfg)
    for key in keys:
        value = first_config_value(model_cfg.get(key), train_cfg.get(key))
        if value is not None:
            return value
    return default


def to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def to_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def to_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(str(value).strip())
    except Exception:
        return float(default)


def normalize_channel_list(values: Iterable[int], *, default: list[int]) -> list[int]:
    channels = [int(x) for x in values if int(x) > 0]
    return channels or list(default)


def to_int_list(value: Any, *, default: list[int]) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return list(default)
    return normalize_channel_list(value, default=default)
