"""Generate and persist samples discovered in registered TDMS files."""

from pathlib import Path
from typing import Any, Dict, List

from .line_rules import LINE_RULES
from .label_database import LabelDatabase, require_database
from .tdms_read import open_tdms


class SampleGenerator:
    """Build and persist analysis samples from TDMS channel rules."""

    HEADER = [
        "line", "sn", "sample_id", "group_name", "channel_name", "sampling_rate",
        "reference", "time", "tdms_storage_root", "relative_path", "tdms_path",
    ]

    def __init__(self, label_records_db_path: Path):
        self.label_records_db_path = require_database(label_records_db_path)
        self.database = LabelDatabase(self.label_records_db_path)

    def build_from_tdms(
        self,
        tdms_path: Path,
        *,
        line: str,
        sn: str,
        reference: str,
    ) -> int:
        if line not in LINE_RULES:
            raise ValueError(f"Line '{line}' is not defined in LINE_RULES")
        channel_rules = LINE_RULES[line].get("channels") or {}
        rules = (
            self._select_conditional_rules(
                channel_rules["conditional"], reference=reference
            )
            if "conditional" in channel_rules
            else [channel_rules]
        )
        existing = {
            (
                row["line"],
                row["sn"],
                row["sample_id"],
                row["group_name"],
                row["channel_name"],
            )
            for row in self.list_all()
        }
        new_rows = []
        with open_tdms(tdms_path, mode="read_metadata") as tdms:
            for rule in rules:
                for direction, group_name in (
                    ("up", rule["up_group"]),
                    ("down", rule["down_group"]),
                ):
                    channel_name = rule["acc_channel"]
                    sample_id = f"{sn}_{direction}"
                    key = (line, sn, sample_id, group_name, channel_name)
                    if key in existing:
                        continue
                    if group_name not in tdms or channel_name not in tdms[group_name]:
                        continue
                    increment = tdms[group_name][channel_name].properties.get(
                        "wf_increment"
                    )
                    new_rows.append(
                        {
                            "line": line,
                            "sn": sn,
                            "sample_id": sample_id,
                            "group_name": group_name,
                            "channel_name": channel_name,
                            "sampling_rate": int(1 / increment) if increment else None,
                        }
                    )
        return self.database.upsert_samples(new_rows)

    @staticmethod
    def _select_conditional_rules(
        rules: List[Dict], *, reference: str
    ) -> List[Dict]:
        for rule in rules:
            if rule.get("when", {}).get("reference") == reference:
                return [rule]
        for rule in rules:
            if rule.get("when", {}).get("reference") == "*":
                return [rule]
        raise ValueError(f"No matching conditional rule for reference '{reference}'")

    @classmethod
    def _normalize_row(cls, row: Dict[str, object]) -> Dict[str, str]:
        return {column: str(row.get(column, "") or "") for column in cls.HEADER}

    def list_all(self) -> List[Dict[str, str]]:
        return [self._normalize_row(row) for row in self.database.list_samples()]

    def filter_by_sn(self, sn: str) -> List[Dict[str, str]]:
        return [row for row in self.list_all() if row["sn"] == sn]

    def replace_scope(self, rows, *, lines=None, origins=None) -> int:
        normalized_rows = []
        for row in rows:
            normalized = self._normalize_row(row)
            if row.get("origin"):
                normalized["origin"] = str(row["origin"]).strip()
            normalized_rows.append(normalized)
        line_scope = {
            str(line).strip() for line in (lines or set()) if str(line).strip()
        }
        return self.database.replace_samples(
            normalized_rows,
            lines=line_scope or None,
            origins=origins,
            all_samples=not line_scope and not origins,
        )


def scan_tdms(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Scan TDMS files, update the manifest, and generate samples."""
    from data_manager.tdms_scan import scan_factory_raw

    return scan_factory_raw(*args, **kwargs)


def check_unregistered_tdms(*args: Any, **kwargs: Any) -> int:
    """Check consistency between TDMS files, manifest rows, and samples."""
    from data_manager.tdms_scan import check_unregistered_tdms as check

    return check(*args, **kwargs)


def rebuild_metadata(**kwargs: Any) -> Dict[str, Any]:
    """Rebuild tdms_manifest.csv and the sample index."""
    from data_manager.data_root_check import _rebuild_metadata

    return _rebuild_metadata(**kwargs)
