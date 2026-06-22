"""Result/reason definitions loaded from cfg/core/label_rules.yaml."""
from __future__ import annotations

from typing import Optional
import yaml

from data_manager.config import LABEL_RULES_PATH


def _text(value) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _integer(value) -> Optional[int]:
    try:
        return None if not _text(value) else int(float(value))
    except (TypeError, ValueError):
        return None


def _keys(value) -> list[str]:
    text = _text(value)
    normalized = text.lower().replace(" ", "").replace("_", "").replace("-", "")
    return list(dict.fromkeys((text, normalized))) if text else []


def load_label_rules() -> dict:
    with LABEL_RULES_PATH.open("r", encoding="utf-8") as stream:
        rules = yaml.safe_load(stream) or {}
    if not isinstance(rules.get("results"), dict) or not isinstance(rules.get("reasons"), dict):
        raise ValueError(f"label_rules 缺少 results / reasons: {LABEL_RULES_PATH}")
    return rules


LABEL_RULES = load_label_rules()


class ReasonResolver:
    def __init__(self):
        self.name_by_id: dict[int, str] = {}
        self.id_by_key: dict[str, int] = {}

    def register(self, name, reason_id: int, *, overwrite: bool) -> None:
        for key in _keys(name):
            if overwrite:
                self.id_by_key[key] = reason_id
            else:
                self.id_by_key.setdefault(key, reason_id)

    def resolve(self, *, reason_id_value=None, reason_name_value=None) -> tuple[Optional[int], str]:
        reason_id, reason_name = _integer(reason_id_value), _text(reason_name_value)
        if reason_id is None:
            reason_id = next((self.id_by_key[key] for key in _keys(reason_name) if key in self.id_by_key), None)
        if reason_id is None:
            return None, reason_name
        if not reason_name:
            reason_name = self.name_by_id.get(reason_id, str(reason_id))
        return reason_id, reason_name


class Label:
    def __init__(self, rules: dict = LABEL_RULES):
        self.results = {key: {"key": key, **value} for key, value in rules["results"].items()}
        self.reasons = {key: {"key": key, **value} for key, value in rules["reasons"].items()}
        self.reason_to_result = {key: value["parent"] for key, value in self.reasons.items()}
        self.result_name_to_key = {
            name: key for key, value in self.results.items() for name in (value["name"], *value.get("alias", []))
        }
        self.reason_name_to_key = {
            name: key for key, value in self.reasons.items() for name in (value["name"], *value.get("alias", []))
        }
        if any(parent not in self.results for parent in self.reason_to_result.values()):
            raise ValueError("reason parent result 不存在")

    def get_result_by_key(self, key: str) -> dict:
        if key not in self.results: raise KeyError(f"Unknown result_key: {key}")
        return self.results[key]

    def get_reason_by_key(self, key: str) -> dict:
        if key not in self.reasons: raise KeyError(f"Unknown reason_key: {key}")
        return self.reasons[key]

    def get_result_by_reason(self, key: str) -> dict:
        return self.get_result_by_key(self.reason_to_result[key])

    def get_result_by_name(self, name: str) -> dict:
        return self.get_result_by_key(self.result_name_to_key[name])

    def get_reason_by_name(self, name: str) -> dict:
        return self.get_reason_by_key(self.reason_name_to_key[name])

    def build_result(self, *, result_key: str, reason_key: str, source: str, confidence=None, extra=None) -> dict:
        result, reason = self.get_result_by_key(result_key), self.get_reason_by_key(reason_key)
        payload = {"result": {k: result[k] for k in ("key", "id", "name")},
                   "reason": {k: reason[k] for k in ("key", "id", "name")}, "source": source}
        if confidence is not None: payload["confidence"] = confidence
        if extra: payload.update(extra)
        return payload

    def build_reason_resolver(self, available_reasons=None) -> ReasonResolver:
        resolver = ReasonResolver()
        for reason in self.reasons.values():
            reason_id = _integer(reason.get("id"))
            if reason_id is None: continue
            resolver.name_by_id[reason_id] = _text(reason.get("name")) or str(reason_id)
            resolver.register(reason.get("name"), reason_id, overwrite=True)
            for alias in reason.get("alias", []): resolver.register(alias, reason_id, overwrite=False)
        for item in available_reasons or []:
            reason_id, name = _integer(item.get("reason_id")), _text(item.get("reason_name"))
            if reason_id is not None and name:
                resolver.name_by_id[reason_id] = name; resolver.register(name, reason_id, overwrite=True)
        return resolver
