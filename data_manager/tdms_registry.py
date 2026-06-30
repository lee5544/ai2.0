"""Scan and register TDMS files in tdms_manifest.csv."""

import csv
from pathlib import Path
from typing import Iterable, List, Dict, Set
from datetime import datetime

from .line_rules import LINE_RULES, normalize_line_time_value
from .tdms_read import is_compressed_tdms_path, tdms_logical_stem


class TdmsRegistry:
    """
    TDMS 文件登记表（文件级事实）

    作用：
        - 登记系统已经“认识”的 TDMS 文件
        - 防止重复登记
        - 提供基础查询能力

    不负责：
        - TDMS 内容解析
        - sample 定义
        - 标签管理
    """

    HEADER = [
        "line",
        "sn",
        "reference",
        "time",
        "created_time",
        "tdms_storage_root",
        "relative_path",
    ]

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self._init_manifest()

    def _init_manifest(self):
        """初始化 manifest CSV"""
        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADER)

    # ------------------------------------------------------------------
    # 文件名解析（纯执行器）
    # ------------------------------------------------------------------

    @staticmethod
    def parse_filename(
        filename: str,
        rule: dict,
        delimiter: str = "_",
    ) -> dict:
        """
        根据规则解析文件名（不含后缀）

        rule 示例：
            {
                "sn_index": 1,
                "reference_index": 2,
                "time_index": [3, 4],
            }
        """
        stem = tdms_logical_stem(filename)
        generated_parts = stem.split("__", 3)
        if len(generated_parts) == 4 and generated_parts[0] in LINE_RULES:
            original = generated_parts[3]
            original_parsed = TdmsRegistry.parse_filename(
                original,
                rule=rule,
                delimiter=delimiter,
            )
            return {
                "sn": generated_parts[1],
                "reference": generated_parts[2],
                "time": original_parsed.get("time", ""),
            }

        parts = stem.split(delimiter)

        def _get(idx):
            if isinstance(idx, int):
                return parts[idx] if idx < len(parts) else "UNKNOWN"
            elif isinstance(idx, (list, tuple)):
                vals = [parts[i] for i in idx if i < len(parts)]
                return "_".join(vals) if vals else "UNKNOWN"
            else:
                raise ValueError(f"Invalid index spec: {idx}")

        return {
            "sn": _get(rule.get("sn_index")),
            "reference": _get(rule.get("reference_index")),
            "time": _get(rule.get("time_index")),
        }

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def register_file(
        self,
        storage_root: str,
        relative_path: str,
        *,
        line: str,
        require_compressed: bool = True,
    ) -> bool:
        """
        登记一个 TDMS 文件

        filename 解析规则来自 line_rules.py
        """

        relative_path = relative_path.replace("\\", "/")
        if require_compressed and not is_compressed_tdms_path(relative_path):
            raise ValueError(
                f"Only compressed TDMS files can be registered: {relative_path}"
            )

        if line not in LINE_RULES:
            raise ValueError(f"Line '{line}' is not defined in LINE_RULES")

        filename_rule = LINE_RULES[line].get("filename")
        if not filename_rule:
            raise ValueError(f"Filename rule is not defined for line '{line}'")

        delimiter = filename_rule.get("split", "_")
        filename = Path(relative_path).name

        parsed = self.parse_filename(
            filename=filename,
            rule=filename_rule,
            delimiter=delimiter,
        )

        if self._exists(storage_root, relative_path):
            return False
        created_time = datetime.now().isoformat(timespec="seconds")

        row = {
            "line": line,
            "sn": parsed.get("sn", "UNKNOWN_SN"),
            "reference": parsed.get("reference", filename),
            "time": normalize_line_time_value(
                parsed.get("time", "UNKNOWN_TIME"),
                line=line,
                filename_rule=filename_rule,
            ),
            "created_time": created_time,
            "tdms_storage_root": storage_root,
            "relative_path": relative_path,
        }

        self._append_row(row)
        return True

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def list_all(self) -> List[Dict[str, str]]:
        return self._load_all()

    def filter_by_sn(self, sn: str) -> List[Dict[str, str]]:
        return [r for r in self._load_all() if r["sn"] == sn]

    def filter_by_line(self, line: str) -> List[Dict[str, str]]:
        return [r for r in self._load_all() if r["line"] == line]

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _exists(self, storage_root: str, relative_path: str) -> bool:
        for row in self._load_all():
            if row["tdms_storage_root"] == storage_root and row["relative_path"] == relative_path:
                return True
        return False

    def _exists_sn(self, storage_root: str, sn: str) -> bool:
        sn = str(sn or "").strip()
        if not sn:
            return False
        for row in self._load_all():
            if row["tdms_storage_root"] == storage_root and str(row.get("sn", "") or "").strip() == sn:
                return True
        return False

    def _append_row(self, row: Dict[str, str]):
        with open(self.manifest_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADER)
            writer.writerow(row)

    def _write_all(self, rows: Iterable[Dict[str, str]]) -> None:
        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADER)
            writer.writeheader()
            writer.writerows(rows)

    def _normalize_row(self, row: Dict[str, str]) -> Dict[str, str]:
        out = {col: str(row.get(col, "") or "") for col in self.HEADER}
        out["time"] = normalize_line_time_value(out["time"], line=out["line"])
        out["relative_path"] = out["relative_path"].replace("\\", "/")
        return out

    def replace_scope(
        self,
        rows: Iterable[Dict[str, str]],
        *,
        storage_root: str | None = None,
        lines: Set[str] | None = None,
    ) -> int:
        storage_root_norm = str(storage_root or "").strip()
        line_scope = {str(line).strip() for line in (lines or set()) if str(line).strip()}

        preserved_rows: List[Dict[str, str]] = []
        for row in self._load_all():
            row_storage_root = str(row.get("tdms_storage_root") or "").strip()
            row_line = str(row.get("line", "") or "").strip()

            in_scope = True
            if storage_root_norm and row_storage_root != storage_root_norm:
                in_scope = False
            if line_scope and row_line not in line_scope:
                in_scope = False

            if not in_scope:
                preserved_rows.append(self._normalize_row(row))

        normalized_rows = [self._normalize_row(row) for row in rows]
        self._write_all([*preserved_rows, *normalized_rows])
        return len(normalized_rows)

    def _load_all(self) -> List[Dict[str, str]]:
        with open(self.manifest_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return list(reader)
